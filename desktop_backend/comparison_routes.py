"""The Compare page's API: saved comparisons, a preview of their sides, and running them.

Registered by :func:`desktop_backend.server.create_app`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, StrictStr

from desktop_backend.comparisons import (
    CONTRAST_TOOL,
    ComparisonBody,
    ComparisonDefinition,
    comparison,
    comparisons,
    delete_comparison,
    duplicate_comparison,
    resolve_side,
    save_comparison,
    update_comparison,
)
from desktop_backend.runner import Runner
from desktop_backend.store import Workspace

__all__ = ["preview", "register_comparison_routes"]


class RunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser: StrictStr = "spacy"


class PreviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: ComparisonDefinition


def preview(workspace: Workspace, definition: ComparisonDefinition) -> dict[str, Any]:
    """What each side holds, and which details every side has (the alignment choices).

    A side that cannot be read (missing project, empty selection) is reported
    in its own entry, so the page can say which card to fix.
    """
    sides: list[dict[str, Any]] = []
    detail_values: list[dict[str, set[str]]] = []
    spellings: list[dict[str, str]] = []
    for side in definition.sides:
        try:
            _project, documents = resolve_side(workspace, side)
        except ValueError as exc:
            sides.append({"name": side.name, "error": str(exc)})
            detail_values.append({})
            spellings.append({})
            continue
        dates = sorted(doc["document_date"] for doc in documents if doc.get("document_date"))
        values: dict[str, set[str]] = {}
        # Keyed by the folded name: the engine reads a detail whatever the
        # case of its name (field_value), so "Speaker" on one side and
        # "speaker" on the other line up, and the preview must offer them.
        spelling: dict[str, str] = {}
        for doc in documents:
            for name, value in doc["fields"].items():
                key = name.casefold()
                if key not in ("side", "project", "date"):
                    spelling.setdefault(key, name)
                    values.setdefault(key, set()).add(str(value).strip())
        detail_values.append(values)
        spellings.append(spelling)
        sides.append(
            {
                "name": side.name,
                "documents": len(documents),
                "words": sum(int(doc.get("words") or 0) for doc in documents),
                "first_date": dates[0] if dates else None,
                "last_date": dates[-1] if dates else None,
                # A detail with one value is named ("Kind: sotu"), not counted ("1 Kind").
                "details": {
                    spelling[key]: {"count": len(found), "examples": sorted(found, key=str.casefold)[:3]}
                    for key, found in sorted(values.items())
                },
            }
        )
    readable = [values for values, side in zip(detail_values, sides, strict=True) if "error" not in side]
    shared: list[dict[str, Any]] = []
    if len(readable) == len(sides) and readable:
        names = set.intersection(*(set(values) for values in readable))
        for key in sorted(names):
            common = set.intersection(*(values[key] for values in readable))
            # Spelled as the first side spells it.
            shared.append({"name": spellings[0][key], "shared_values": len(common), "examples": sorted(common)[:5]})
    # Periods line up only when every side has dates.
    return {"sides": sides, "alignments": shared, "dated": all(side.get("first_date") for side in sides)}


def register_comparison_routes(
    app: FastAPI,
    workspace: Workspace,
    runner: Runner,
    secured: Sequence[Any],
    public_job: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    dependencies = list(secured)
    base = "/api/projects/{project_id}/comparisons"

    @app.get(base, dependencies=dependencies)
    def list_comparisons(project_id: str) -> list[dict[str, Any]]:
        return comparisons(workspace, project_id)

    @app.post(base, dependencies=dependencies)
    def create_comparison(project_id: str, body: ComparisonBody) -> dict[str, Any]:
        with runner.lock:
            return save_comparison(workspace, project_id, body)

    @app.post(base + "/preview", dependencies=dependencies)
    def preview_comparison(project_id: str, body: PreviewBody) -> dict[str, Any]:
        workspace.project(project_id)
        return preview(workspace, body.definition)

    @app.get(base + "/runs", dependencies=dependencies)
    def comparison_runs(project_id: str) -> list[dict[str, Any]]:
        """This project's comparison runs, newest first, each saying which saved comparison it ran."""
        found = []
        for job in workspace.jobs(project_id):
            if job["tool"] != CONTRAST_TOOL:
                continue
            params = json.loads(job["request"]).get("params", {}).get(CONTRAST_TOOL, {})
            found.append(
                {
                    **public_job(job),
                    "comparison_id": params.get("comparison-id", ""),
                    "comparison_name": params.get("comparison-name", ""),
                }
            )
        return found

    @app.get(base + "/{comparison_id}", dependencies=dependencies)
    def get_comparison(project_id: str, comparison_id: str) -> dict[str, Any]:
        return comparison(workspace, project_id, comparison_id)

    @app.post(base + "/{comparison_id}", dependencies=dependencies)
    def save(project_id: str, comparison_id: str, body: ComparisonBody) -> dict[str, Any]:
        with runner.lock:
            return update_comparison(workspace, project_id, comparison_id, body)

    @app.post(base + "/{comparison_id}/duplicate", dependencies=dependencies)
    def duplicate(project_id: str, comparison_id: str) -> dict[str, Any]:
        with runner.lock:
            return duplicate_comparison(workspace, project_id, comparison_id)

    @app.post(base + "/{comparison_id}/delete", dependencies=dependencies)
    def delete(project_id: str, comparison_id: str) -> dict[str, bool]:
        with runner.lock:
            delete_comparison(workspace, project_id, comparison_id)
        return {"deleted": True}

    @app.post(base + "/{comparison_id}/run", dependencies=dependencies)
    def run(project_id: str, comparison_id: str, body: RunBody) -> dict[str, Any]:
        saved = comparison(workspace, project_id, comparison_id)
        definition = ComparisonDefinition(**saved["definition"])
        return public_job(
            runner.submit_contrast(project_id, definition, body.parser, name=saved["name"], comparison_id=saved["id"])
        )
