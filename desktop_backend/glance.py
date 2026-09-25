"""Corpus at a glance, as a desktop job: submit, status, summary, figures.

The recipe and the sentences are :mod:`core.insight.glance`. Here: a glance
is one job (tool ``corpus_glance``) that parses once, runs the recipe as a
batch, and publishes each tool as a child run with its figures. The job is
its own cache: :func:`status` finds the newest glance of this project, and
says ``stale`` when the documents have changed since (the key is the
documents' contents, so a rename is not a change). Uploading a document
never starts one; only the button does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.insight.glance import GLANCE_FIGURES, RECIPE, glance_key, summarize
from core.profiler.labels import tool_label
from core.profiler.plan import Plan, build_plan
from core.result import Result
from desktop_backend.store import Workspace

__all__ = ["GLANCE_TOOL", "figure_file", "glance_plan", "status"]

GLANCE_TOOL = "corpus_glance"


def glance_plan() -> Result[Plan]:
    return build_plan(list(RECIPE), RECIPE)


def current_key(workspace: Workspace, project_id: str) -> str:
    return glance_key([str(document["sha256"]) for document in workspace.documents(project_id)])


def _children(run_dir: Path) -> list[dict[str, Any]]:
    manifest = run_dir / "batch.json"
    if not manifest.is_file():
        return []
    children = json.loads(manifest.read_text(encoding="utf-8")).get("children", [])
    return [child for child in children if isinstance(child, dict)]


def _child_dir(run_dir: Path, child: dict[str, Any]) -> Path | None:
    raw = child.get("run_dir")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else run_dir.parent / path


def _tables(folder: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in folder.glob("*.csv"):
        try:
            tables[path.name] = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    return tables


def status(workspace: Workspace, project_id: str) -> dict[str, Any]:
    """The newest glance of this project and what it found, or that there is none."""
    jobs = [job for job in workspace.jobs(project_id) if job["tool"] == GLANCE_TOOL]
    key = current_key(workspace, project_id)
    if not jobs:
        return {"state": "none", "key": key}
    job = max(jobs, key=lambda item: str(item.get("created") or ""))
    request = json.loads(job.get("request") or "{}") if isinstance(job.get("request"), str) else {}
    stale = request.get("glance_key") not in (None, key)
    # The raw job row; the server passes it through public_job, which hides
    # the frozen request (document paths) from the app.
    answer: dict[str, Any] = {"state": "stale" if stale else job["state"].lower(), "job": job, "key": key}
    if job["state"] not in ("DONE", "PARTIAL") or not job.get("run_dir"):
        return answer
    project_dir = workspace.project_dir(project_id)
    run_dir = project_dir / str(job["run_dir"])
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    figures: list[dict[str, str]] = []
    for child in _children(run_dir):
        tool = str(child.get("tool", ""))
        folder = _child_dir(run_dir, child)
        if not tool or folder is None or not folder.is_dir():
            continue
        tables[tool] = _tables(folder)
        for panel in GLANCE_FIGURES.get(tool, ()):
            image = folder / "figures" / f"{panel}.png"
            if image.is_file():
                figures.append(
                    {
                        "tool": tool,
                        "label": tool_label(tool),
                        "panel": panel,
                        "path": image.relative_to(project_dir).as_posix(),
                    }
                )
                break
    answer["summary"] = summarize(tables)
    answer["figures"] = figures
    if not stale:
        answer["state"] = "ready"
    return answer


def figure_file(workspace: Workspace, project_id: str, relative: str) -> Path:
    """A glance figure's file, refusing anything outside the project's runs."""
    project_dir = workspace.project_dir(project_id).resolve()
    path = (project_dir / relative).resolve()
    runs = project_dir / "runs"
    if runs not in path.parents or path.suffix.lower() != ".png" or not path.is_file():
        raise KeyError("figure not found")
    return path
