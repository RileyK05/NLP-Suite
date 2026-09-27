"""The split dialog's API: preview a document's sections, then store them (docs/PLAN_0.5.0.md 2.4).

Registered by :func:`desktop_backend.server.create_app`, beside the fields
routes, so the server's route list does not grow by a page's worth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI

from desktop_backend.runner import Runner
from desktop_backend.sections import SplitBody, public_plan, public_turns, split_document
from desktop_backend.store import Workspace

__all__ = ["register_section_routes"]


def register_section_routes(app: FastAPI, workspace: Workspace, runner: Runner, secured: Sequence[Any]) -> None:
    dependencies = list(secured)

    @app.post("/api/projects/{project_id}/documents/{document_id}/sections/preview", dependencies=dependencies)
    def preview_sections(project_id: str, document_id: str, body: SplitBody) -> dict[str, Any]:
        """Where the chosen rule would cut the document, or its speakers for a transcript. Nothing is written."""
        if body.transcript:
            return public_turns(workspace, project_id, document_id, body)
        return public_plan(workspace, project_id, document_id, body)

    @app.post("/api/projects/{project_id}/documents/{document_id}/sections/apply", dependencies=dependencies)
    def apply_sections(project_id: str, document_id: str, body: SplitBody) -> dict[str, Any]:
        """Store the sections as documents; the book goes to Trash unless kept."""
        with runner.lock:
            created = split_document(workspace, project_id, document_id, body)
        return {"documents": created}
