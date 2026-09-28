"""The Corpus page's API for document details (docs/internal/PLAN_0.5.0.md 1.8).

Registered by :func:`desktop_backend.server.create_app`, kept apart so the
server's route list does not grow by a page's worth.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from core.io.cleaning import count_stage_directions
from core.io.metadata_table import MatchReport, match_metadata, read_table
from core.io.reader import read_text
from desktop_backend.fields import (
    FieldRow,
    FieldSettings,
    cleaning,
    delete_field,
    details,
    field_names,
    filename_template,
    import_fields,
    name_problem,
    resolved_axis,
    save_settings,
    set_fields,
    settings,
)
from desktop_backend.runner import Runner
from desktop_backend.store import Workspace

__all__ = ["register_field_routes"]


class SetFieldsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[FieldRow] = Field(max_length=80_000)
    expected_revision: int | None = None


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: FieldSettings
    expected_revision: int | None = None


class ImportBody(BaseModel):
    """A spreadsheet of details, previewed (``dry_run``) and then applied with the choices made in the preview."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    #: The file's bytes, base64-encoded (at most 10 MB before encoding).
    content: str = Field(max_length=14_000_000)
    name_column: str | None = None
    #: Column -> detail name; "" leaves the column out. Unlisted columns get the proposed name.
    targets: dict[str, str] | None = None
    dry_run: bool = True
    expected_revision: int | None = None


#: How many unmatched names the preview lists (the counts are always whole).
_LISTED = 50


def _report(report: MatchReport, diagnostics: Sequence[Any]) -> dict[str, Any]:
    ways: dict[str, int] = {}
    for how in report.matched_by.values():
        ways[how] = ways.get(how, 0) + 1
    return {
        "name_column": report.name_column,
        "matched_on": report.matched_on,
        "columns": [
            {
                "column": plan.column,
                "target": plan.target,
                "kind": plan.kind,
                "filled": plan.filled,
                "samples": list(plan.samples),
                "problem": name_problem(plan.target) if plan.target else "",
            }
            for plan in report.columns
        ],
        "matched": len(report.values),
        "matched_by": ways,
        "unmatched_rows": list(report.unmatched_rows[:_LISTED]),
        "unmatched_row_count": len(report.unmatched_rows),
        "unmatched_documents": list(report.unmatched_documents[:_LISTED]),
        "unmatched_document_count": len(report.unmatched_documents),
        "duplicate_rows": list(report.duplicate_rows[:_LISTED]),
        "preview": [{"document": name, "values": values} for name, values in report.preview],
        "diagnostics": [{"severity": d.severity.name, "code": d.code, "message": d.message} for d in diagnostics],
    }


def _template(workspace: Workspace, project_id: str) -> dict[str, Any] | None:
    template = filename_template(workspace, project_id)
    if template is None:
        return None
    return {
        "parts": list(template.parts),
        "separator": template.separator,
        "fits": template.fits,
        "total": template.total,
        "misses": list(template.misses[:50]),
        "describe": template.describe(),
    }


def _cleaning_preview(workspace: Workspace, project_id: str) -> dict[str, Any]:
    """What the project's text cleaning would take out, counted and never changed.

    The Corpus page's offer: "Found 2,314 bracketed stage directions like
    (Applause.), (Laughter.); leave them out of analyses?" Reading every
    document is what makes it exact; the page asks once per project.
    """
    rule = cleaning(workspace, project_id)
    texts = []
    for document in workspace.documents(project_id):
        text = read_text(workspace.project_dir(project_id) / "corpus" / str(document["stored_name"]))
        if text.value is not None:
            texts.append(text.unwrap())
    found, samples = count_stage_directions(texts, rule.extra_terms)
    return {"found": found, "samples": samples, "documents": len(texts), "cleaning": rule.model_dump()}


def register_field_routes(app: FastAPI, workspace: Workspace, runner: Runner, secured: Sequence[Any]) -> None:
    dependencies = list(secured)

    @app.get("/api/projects/{project_id}/fields", dependencies=dependencies)
    def get_fields(project_id: str) -> dict[str, Any]:
        """Every document's details with their sources, the detail names, the file-name pattern, the settings."""
        documents = workspace.documents(project_id)
        current = settings(workspace, project_id)
        found = details(workspace, project_id, documents)
        return {
            "names": field_names(workspace, project_id),
            "documents": found,
            "template": _template(workspace, project_id),
            "settings": current["settings"],
            "revision": current["revision"],
            "axis": resolved_axis(workspace, project_id, documents, found),
        }

    @app.post("/api/projects/{project_id}/fields", dependencies=dependencies)
    def post_fields(project_id: str, body: SetFieldsBody) -> dict[str, Any]:
        with runner.lock:
            return set_fields(workspace, project_id, body.rows, body.expected_revision)

    @app.post("/api/projects/{project_id}/fields/import", dependencies=dependencies)
    def post_import(project_id: str, body: ImportBody) -> dict[str, Any]:
        """Preview a spreadsheet of details (``dry_run``), or store it with the preview's choices."""
        try:
            data = base64.b64decode(body.content, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("The file did not arrive intact; choose it again.") from exc
        table = read_table(data, body.filename)
        if table.value is None:
            raise ValueError(" ".join(d.message for d in table.diagnostics))
        documents = workspace.documents(project_id)
        names = [str(doc["name"]) for doc in documents]
        found = details(workspace, project_id, documents)
        # A sheet may be keyed by a detail the documents already have
        # (Speaker -> Party) rather than by file name.
        current = {
            str(doc["name"]): {name: item["value"] for name, item in found.get(str(doc["id"]), {}).items()}
            for doc in documents
        }
        matched = match_metadata(
            table.value, names, details=current, name_column=body.name_column, targets=body.targets
        )
        if matched.value is None:
            raise ValueError(" ".join(d.message for d in matched.diagnostics))
        report = matched.value
        answer = _report(report, matched.diagnostics)
        answer["applied"] = False
        if not body.dry_run:
            targets = [plan.target for plan in report.columns if plan.target]
            if not targets:
                raise ValueError("Every column is left out, so there is nothing to import.")
            if not report.values:
                raise ValueError("No row matched a document, so there is nothing to import.")
            with runner.lock:
                stored = import_fields(workspace, project_id, report.values, targets, body.expected_revision)
            answer["applied"] = True
            answer["revision"] = stored["revision"]
        return answer

    @app.delete("/api/projects/{project_id}/fields/{name}", dependencies=dependencies)
    def remove_field(project_id: str, name: str) -> dict[str, bool]:
        with runner.lock:
            delete_field(workspace, project_id, name)
        return {"deleted": True}

    @app.get("/api/projects/{project_id}/settings", dependencies=dependencies)
    def get_settings(project_id: str) -> dict[str, Any]:
        workspace.project(project_id)
        return settings(workspace, project_id)

    @app.get("/api/projects/{project_id}/cleaning_preview", dependencies=dependencies)
    def get_cleaning_preview(project_id: str) -> dict[str, Any]:
        workspace.project(project_id)
        return _cleaning_preview(workspace, project_id)

    @app.post("/api/projects/{project_id}/settings", dependencies=dependencies)
    def post_settings(project_id: str, body: SettingsBody) -> dict[str, Any]:
        with runner.lock:
            return save_settings(workspace, project_id, body.settings, body.expected_revision)
