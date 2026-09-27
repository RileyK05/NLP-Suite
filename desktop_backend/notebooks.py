"""Notebooks: what one is, the ones the app offers to start from, and running one as a run.

A notebook is stored as nbformat 4 JSON, the ``.ipynb`` shape, from the start:
export is then a copy rather than a conversion, and a notebook written in the
app opens in Jupyter or VS Code. Outputs are not stored with it. They belong
to a run -- "Run and save" (:func:`run_notebook`) -- which is how a figure in
a paper can say which revision of which notebook produced it.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
from typing import TYPE_CHECKING, Any
import uuid

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.script.session import Output
    from desktop_backend.store import Workspace

__all__ = [
    "MAX_NOTEBOOKS_PER_PROJECT",
    "NOTEBOOK_TOOL",
    "NotebookBody",
    "new_notebook",
    "normalize_notebook",
    "notebook_from_template",
    "run_notebook",
]

NOTEBOOK_TOOL = "notebook"
#: The Jupyter notebook format version stored and accepted.
NBFORMAT = 4
MAX_NOTEBOOKS_PER_PROJECT = 200
MAX_CELLS = 500
MAX_SOURCE_CHARACTERS = 1_000_000
MAX_NOTEBOOK_NAME = 120
CREATED_BY = ("app", "import", "pasted")


def _source(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return "".join(value)
    raise ValueError("A cell's source must be text.")


def normalize_notebook(content: dict[str, Any]) -> dict[str, Any]:
    """*content* as a clean nbformat 4 notebook without outputs, or a refusal.

    Accepts what Jupyter writes (source as a list of lines, outputs, execution
    counts, extra metadata) and keeps what a notebook is: the cells, their
    kind, their text, an id each, and the suite's own metadata.
    """
    if not isinstance(content, dict) or content.get("nbformat") != NBFORMAT:
        raise ValueError("This is not a Jupyter notebook (nbformat 4).")
    cells = content.get("cells")
    if not isinstance(cells, list):
        raise ValueError("A notebook needs a list of cells.")
    if len(cells) > MAX_CELLS:
        raise ValueError(f"A notebook may have at most {MAX_CELLS} cells.")
    clean: list[dict[str, Any]] = []
    total = 0
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Every cell must be an object.")
        kind = cell.get("cell_type")
        if kind == "raw":
            kind = "markdown"
        if kind not in ("code", "markdown"):
            raise ValueError(f"Cells are code or markdown, not {kind!r}.")
        source = _source(cell.get("source", ""))
        total += len(source)
        identifier = str(cell.get("id") or uuid.uuid4().hex[:8])
        if identifier in seen:
            identifier = uuid.uuid4().hex[:8]
        seen.add(identifier)
        entry: dict[str, Any] = {"cell_type": kind, "id": identifier, "metadata": {}, "source": source}
        # One piece of cell metadata survives: that its code came from outside
        # the app (pasted from a chatbot), so the page keeps saying "read this
        # before running it" until the reader says they have.
        cell_meta = cell.get("metadata")
        suite_meta = cell_meta.get("nlpsuite") if isinstance(cell_meta, dict) else None
        if isinstance(suite_meta, dict) and suite_meta.get("origin") in ("pasted", "import"):
            entry["metadata"] = {"nlpsuite": {"origin": suite_meta["origin"]}}
        if kind == "code":
            entry["execution_count"] = None
            entry["outputs"] = []
        clean.append(entry)
    if total > MAX_SOURCE_CHARACTERS:
        raise ValueError("A notebook's text may be at most 1,000,000 characters.")
    metadata = content.get("metadata")
    suite = metadata.get("nlpsuite") if isinstance(metadata, dict) else None
    claimed = suite.get("created_by") if isinstance(suite, dict) else None
    created_by = claimed if claimed in CREATED_BY else "import"
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "nlpsuite": {"library": "1", "created_by": created_by},
        },
        "cells": clean,
    }


def new_notebook(cells: list[tuple[str, str]], *, created_by: str = "app") -> dict[str, Any]:
    """A notebook from ``(kind, text)`` pairs."""
    return normalize_notebook(
        {
            "nbformat": 4,
            "metadata": {"nlpsuite": {"created_by": created_by}},
            "cells": [{"cell_type": kind, "source": text} for kind, text in cells],
        }
    )


def notebook_from_template(template_id: str) -> dict[str, Any]:
    """A new notebook holding one of the templates in :mod:`core.script.templates`."""
    from core.script.templates import TEMPLATES

    for template in TEMPLATES:
        if template.id == template_id:
            return new_notebook(list(template.cells))
    raise KeyError(f"No template called {template_id!r}.")


class NotebookBody(BaseModel):
    """A notebook as the page sends it."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1, max_length=MAX_NOTEBOOK_NAME)
    content: dict[str, Any]
    expected_revision: int | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Give the notebook a name.")
        return value

    @field_validator("content")
    @classmethod
    def _content(cls, value: dict[str, Any]) -> dict[str, Any]:
        return normalize_notebook(value)


# ------------------------------------------------------------ run and save --


def _html_preview(frame: Any) -> str:
    return str(frame.head(20).to_html(index=False, border=0))


def _executed_cell_outputs(outputs: list[Output], stdout: str) -> list[dict[str, Any]]:
    """A cell's outputs in nbformat's shapes, so the saved notebook reads in Jupyter too."""
    from core.script.session import FigureOutput, NoteOutput, TableOutput

    shaped: list[dict[str, Any]] = []
    if stdout:
        shaped.append({"output_type": "stream", "name": "stdout", "text": stdout})
    for output in outputs:
        if isinstance(output, TableOutput) and not output.saved:
            shaped.append(
                {
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {"text/html": _html_preview(output.frame), "text/plain": output.frame.head(20).to_string()},
                }
            )
        elif isinstance(output, FigureOutput):
            shaped.append(
                {
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {"image/png": base64.b64encode(output.png).decode("ascii"), "text/plain": output.name},
                }
            )
        elif isinstance(output, NoteOutput):
            shaped.append({"output_type": "stream", "name": "stdout", "text": output.text + "\n"})
    return shaped


def run_notebook(workspace: Workspace, job: dict[str, Any], request: dict[str, Any]) -> int:  # noqa: PLR0915
    """Run every cell in a fresh namespace and publish what it showed, drew and saved.

    Stops at the first cell that fails: later cells usually depend on it, and
    running them would bury the one error that matters under the ones it
    caused. What ran before the failure is still published (a PARTIAL run),
    with the error as its diagnostic.
    """
    import matplotlib

    matplotlib.use("Agg")
    from core.io.writer import OutputWriter
    from core.result import Diagnostic
    from core.script.session import FigureOutput, NoteOutput, Session, TableOutput, open_session
    from desktop_backend.kernel import display_value, keep_open_figures, run_cell
    from desktop_backend.project_source import ProjectSource

    project_id = job["project_id"]
    project_dir = workspace.project_dir(project_id)
    content = request["notebook"]
    workspace.update_job(job["id"], state="RUNNING", stage="Running the notebook's cells")
    collected: list[Output] = []
    session = Session(
        ProjectSource(workspace, project_id, parser=request.get("parser", "spacy"), documents=request["documents"]),
        on_output=collected.append,
    )
    namespace: dict[str, Any] = {"__name__": "__main__"}
    executed = json.loads(json.dumps(content))
    failure: dict[str, Any] | None = None
    count = 0
    with open_session(session):
        for index, cell in enumerate(executed["cells"]):
            if cell["cell_type"] != "code" or not cell["source"].strip():
                continue
            count += 1
            workspace.update_job(job["id"], state="RUNNING", stage=f"Running cell {index + 1}")
            before = len(collected)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
                result = run_cell(cell["source"], namespace, f"<cell {index + 1}>", display_value)
                if result.ok:
                    keep_open_figures()
            cell["execution_count"] = count
            cell["outputs"] = _executed_cell_outputs(collected[before:], printed.getvalue())
            if not result.ok:
                failure = {"cell": index + 1, **(result.error or {})}
                cell["outputs"].append(
                    {
                        "output_type": "error",
                        "ename": failure.get("type", "Error"),
                        "evalue": failure.get("message", ""),
                        "traceback": failure.get("trace", []),
                    }
                )
                break

    code = "\n\n".join(cell["source"] for cell in content["cells"] if cell["cell_type"] == "code")
    documents = request["documents"]
    writer = OutputWriter(
        project_dir / "runs",
        tool=NOTEBOOK_TOOL,
        params={
            "notebook": request.get("name", ""),
            "notebook-id": request.get("notebook_id", ""),
            "revision": request.get("revision", 0),
            "code-sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        },
        inputs=[project_dir / "corpus" / str(item["stored_name"]) for item in documents],
    )
    try:
        charts: dict[str, Any] = {}
        notes: list[str] = []
        for output in collected:
            if isinstance(output, TableOutput):
                folder = "data" if output.saved else "tables"
                path = f"{folder}/{output.name}.csv"
                written = writer.write_table(output.frame, path, description=output.title or output.name)
                if not written.ok:
                    raise ValueError("; ".join(d.message for d in written.diagnostics))
                if output.chart is not None:
                    charts[path] = output.chart
            elif isinstance(output, FigureOutput):
                for suffix, data in (("png", output.png), ("svg", output.svg)):
                    written = writer.write_bytes(data, f"figures/{output.name}.{suffix}", description=output.name)
                    if not written.ok:
                        raise ValueError("; ".join(d.message for d in written.diagnostics))
            elif isinstance(output, NoteOutput):
                notes.append(output.text)
        writer.write_text(json.dumps(executed, indent=1, ensure_ascii=False), "notebook.ipynb", kind="notebook")
        if notes:
            writer.write_text("\n\n".join(notes) + "\n", "notes.md", kind="report")
        import matplotlib as mpl
        import pandas as pd

        writer.write_json(
            {
                "notebook": request.get("name", ""),
                "revision": request.get("revision", 0),
                "library": "1",
                "calls": [
                    {"function": c.function, "arguments": c.arguments, "seconds": c.seconds, "rows": c.rows}
                    for c in session.calls
                ],
                "charts": charts,
                "versions": {"pandas": pd.__version__, "matplotlib": mpl.__version__},
                "failed_cell": failure,
            },
            "provenance.json",
            kind="provenance",
        )
        if failure is not None:
            writer.add_diagnostics(
                Diagnostic.error(
                    "NOTEBOOK_CELL_FAILED",
                    f"Cell {failure['cell']} stopped the notebook: {failure.get('type')}: {failure.get('message')}",
                )
            )
        envelope = writer.finalize()
        if not envelope.ok:
            raise ValueError("; ".join(d.message for d in envelope.diagnostics))
    except Exception:
        writer.abandon()
        raise
    error = (
        [
            {
                "severity": "ERROR",
                "code": "NOTEBOOK_CELL_FAILED",
                "message": f"Cell {failure['cell']}: {failure.get('message')}",
            }
        ]
        if failure
        else []
    )
    workspace.update_job(
        job["id"],
        state="PARTIAL" if failure else "DONE",
        stage="Stopped at a failing cell" if failure else "Results ready",
        run_dir=writer.run_dir.relative_to(project_dir).as_posix(),
        diagnostics=error,
    )
    return 1 if failure else 0
