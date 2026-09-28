"""The Scripts page's API: notebooks, their kernels, "Run and save", the reference and the AI guide.

Registered by :func:`desktop_backend.server.create_app`, kept in its own module
so the server's route list does not grow by a page's worth for one page.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from desktop_backend.kernels import KernelManager
from desktop_backend.notebook_export import export_notebook, safe_file_name
from desktop_backend.notebooks import NotebookBody, new_notebook, normalize_notebook, notebook_from_template
from desktop_backend.previews import MAX_PREVIEW_BYTES, PreviewTickets
from desktop_backend.runner import Runner
from desktop_backend.store import Workspace

__all__ = ["register_notebook_routes"]


class CreateNotebookBody(BaseModel):
    """A new notebook: empty, from a template, or imported from an ``.ipynb`` file."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1, max_length=120)
    template: StrictStr | None = None
    #: An imported file's JSON. Marked as imported whatever it claims, so the
    #: page shows the "came from outside the app" banner.
    imported: dict[str, Any] | None = None


class ExecBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell: StrictStr = Field(max_length=64)
    code: StrictStr = Field(max_length=1_000_000)
    parser: StrictStr = "spacy"


class LintBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: StrictStr = Field(max_length=1_000_000)


class RunNotebookBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser: StrictStr = "spacy"


_EMPTY = [("code", "import nlpsuite as nlp\n\ncorpus = nlp.corpus()\ncorpus.documents")]


def register_notebook_routes(
    app: FastAPI,
    workspace: Workspace,
    runner: Runner,
    kernels: KernelManager,
    secured: Sequence[Any],
    public_job: Callable[[dict[str, Any]], dict[str, Any]],
    previews: PreviewTickets,
) -> None:
    dependencies = list(secured)

    @app.get("/api/script/reference", dependencies=dependencies)
    def script_reference() -> dict[str, Any]:
        """The library reference for the Scripts page's side panel, and the templates list.

        Templates carry their cells because the Learn page shows them as the
        scripting examples: shown code is the code the suite runs, so an
        example cannot promise a call the library does not have.
        """
        from core.script.reference import library_functions, reference_markdown
        from core.script.templates import TEMPLATES

        return {
            "markdown": reference_markdown(),
            "functions": library_functions(),
            "templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "request": t.request,
                    "cells": [{"kind": kind, "text": text} for kind, text in t.cells],
                }
                for t in TEMPLATES
            ],
        }

    @app.post("/api/script/lint", dependencies=dependencies)
    def script_lint(body: LintBody) -> dict[str, Any]:
        """Lines worth reading in code from outside the app. Parsed, never run."""
        from core.script.lint import lint

        return {"warnings": lint(body.code)}

    @app.get("/api/projects/{project_id}/script-corpus", dependencies=dependencies)
    def script_corpus(project_id: str) -> dict[str, Any] | None:
        """The corpus as the reference panel describes it: columns, counts, samples. Never text."""
        from core.script.api import Corpus
        from core.script.reference import corpus_summary
        from core.script.session import Session
        from desktop_backend.project_source import ProjectSource

        if not workspace.documents(project_id):
            return None
        source = ProjectSource(workspace, project_id)
        return corpus_summary(Corpus(Session(source), None).documents, name=source.name)

    @app.get("/api/projects/{project_id}/script-guide", dependencies=dependencies)
    def script_guide(project_id: str, version: str = "short", corpus: bool = True) -> PlainTextResponse:
        """The guide a researcher pastes into their own chatbot (docs/internal/PLAN_0.5.0.md 4.8).

        With ``corpus``, it ends with the project's document columns and sample
        values -- names, dates, counts, never text.
        """
        from core.script.api import Corpus
        from core.script.reference import corpus_summary, guide
        from core.script.session import Session
        from desktop_backend.project_source import ProjectSource

        summary = None
        if corpus and workspace.documents(project_id):
            source = ProjectSource(workspace, project_id)
            summary = corpus_summary(Corpus(Session(source), None).documents, name=source.name)
        return PlainTextResponse(guide(version=version, corpus=summary), media_type="text/markdown; charset=utf-8")

    @app.get("/api/projects/{project_id}/notebooks", dependencies=dependencies)
    def list_notebooks(project_id: str) -> list[dict[str, Any]]:
        return workspace.notebooks(project_id)

    @app.post("/api/projects/{project_id}/notebooks", dependencies=dependencies)
    def create_notebook(project_id: str, body: CreateNotebookBody) -> dict[str, Any]:
        if body.imported is not None:
            content = normalize_notebook(body.imported)
            content["metadata"]["nlpsuite"]["created_by"] = "import"
        elif body.template:
            content = notebook_from_template(body.template)
        else:
            content = new_notebook(_EMPTY)
        with runner.lock:
            return workspace.save_notebook(project_id, NotebookBody(name=body.name, content=content))

    @app.get("/api/projects/{project_id}/notebooks/{notebook_id}", dependencies=dependencies)
    def get_notebook(project_id: str, notebook_id: str) -> dict[str, Any]:
        return workspace.notebook(project_id, notebook_id)

    @app.post("/api/projects/{project_id}/notebooks/{notebook_id}", dependencies=dependencies)
    def save_notebook(project_id: str, notebook_id: str, body: NotebookBody) -> dict[str, Any]:
        with runner.lock:
            return workspace.update_notebook(project_id, notebook_id, body)

    @app.post("/api/projects/{project_id}/notebooks/{notebook_id}/duplicate", dependencies=dependencies)
    def duplicate_notebook(project_id: str, notebook_id: str) -> dict[str, Any]:
        with runner.lock:
            return workspace.duplicate_notebook(project_id, notebook_id)

    @app.post("/api/projects/{project_id}/notebooks/{notebook_id}/delete", dependencies=dependencies)
    def delete_notebook(project_id: str, notebook_id: str) -> dict[str, bool]:
        kernels.stop(project_id, notebook_id)
        with runner.lock:
            workspace.delete_notebook(project_id, notebook_id)
        return {"deleted": True}

    @app.get("/api/projects/{project_id}/notebooks/{notebook_id}/download", dependencies=dependencies)
    def download_notebook(project_id: str, notebook_id: str) -> PlainTextResponse:
        """The notebook as an ``.ipynb`` file (code only; a saved run holds the outputs)."""
        import json

        notebook = workspace.notebook(project_id, notebook_id)
        name = safe_file_name(notebook["name"])
        return PlainTextResponse(
            json.dumps(notebook["content"], indent=1, ensure_ascii=False),
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f'attachment; filename="{name}.ipynb"'},
        )

    @app.get("/api/projects/{project_id}/notebooks/{notebook_id}/export", dependencies=dependencies)
    def export(project_id: str, notebook_id: str) -> Response:
        """A zip: the notebook, its latest saved run's tables and figures, the guide, and an offline stand-in."""
        name, data = export_notebook(workspace, project_id, notebook_id)
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.post("/api/projects/{project_id}/notebooks/{notebook_id}/run", dependencies=dependencies)
    def run_and_save(project_id: str, notebook_id: str, body: RunNotebookBody) -> dict[str, Any]:
        """Queue "Run and save": the saved revision, top to bottom, published as a run."""
        return public_job(runner.submit_notebook(project_id, notebook_id, body.parser))

    # -- the kernel ------------------------------------------------------------

    @app.post("/api/projects/{project_id}/notebooks/{notebook_id}/kernel/exec", dependencies=dependencies)
    def execute_cell(project_id: str, notebook_id: str, body: ExecBody) -> dict[str, Any]:
        workspace.notebook(project_id, notebook_id)
        return kernels.execute(project_id, notebook_id, body.cell, body.code, body.parser)

    @app.get("/api/projects/{project_id}/notebooks/{notebook_id}/kernel/events", dependencies=dependencies)
    def kernel_events(project_id: str, notebook_id: str, after: int = 0) -> dict[str, Any]:
        return kernels.events(project_id, notebook_id, after)

    @app.post("/api/projects/{project_id}/notebooks/{notebook_id}/kernel/stop", dependencies=dependencies)
    def stop_kernel(project_id: str, notebook_id: str) -> dict[str, bool]:
        kernels.stop(project_id, notebook_id)
        return {"stopped": True}

    @app.get("/api/projects/{project_id}/notebooks/{notebook_id}/kernel/files/{name}", dependencies=dependencies)
    def kernel_file(project_id: str, notebook_id: str, name: str) -> FileResponse:
        path = kernels.file(project_id, notebook_id, name)
        media = {".png": "image/png", ".svg": "image/svg+xml", ".csv": "text/csv", ".html": "text/html"}.get(
            path.suffix, "application/octet-stream"
        )
        return FileResponse(path, media_type=media, filename=name)

    @app.post(
        "/api/projects/{project_id}/notebooks/{notebook_id}/kernel/files/{name}/preview",
        dependencies=dependencies,
    )
    def prepare_kernel_preview(project_id: str, notebook_id: str, name: str) -> dict[str, str]:
        """A ticket to frame one kernel HTML chart without the workspace token.

        Interactive charts are shown in a sandboxed frame, which cannot carry
        an Authorization header. This issues the same scoped, expiring ticket a
        run artifact's preview uses, so the frame fetches by ticket alone.
        """
        path = kernels.file(project_id, notebook_id, name)
        if path.suffix.lower() not in (".html", ".htm") or path.stat().st_size > MAX_PREVIEW_BYTES:
            raise ValueError("Only an interactive HTML chart can be previewed in a frame.")
        return {"path": "/api/previews/" + previews.issue_kernel(project_id, notebook_id, name)}

    @app.get("/api/projects/{project_id}/notebooks/{notebook_id}/tables.zip", dependencies=dependencies)
    def tables_zip(project_id: str, notebook_id: str) -> Response:
        """Every table the notebook's kernel has shown or saved, as one zip.

        The one-click "Save all CSVs": a reader who wants the numbers behind a
        session's tables should not have to press a download button per table.
        """
        workspace.notebook(project_id, notebook_id)
        try:
            data = kernels.tables_archive(project_id, notebook_id)
        except KeyError as exc:
            raise ValueError("This notebook has no running kernel; run a cell first.") from exc
        name = safe_file_name(workspace.notebook(project_id, notebook_id)["name"])
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name} tables.zip"'},
        )
