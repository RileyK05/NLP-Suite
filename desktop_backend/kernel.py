"""The process that runs a notebook's Python cells.

**Why a separate process.** A notebook runs code its author wrote. Inside the
server, a loop that never ends would freeze the app and a crash would take it
down; inside the job worker it would block every analysis queued behind it.
So each open notebook gets a kernel: the same engine executable started with
``--script-kernel``, holding the notebook's variables between cells.
:mod:`desktop_backend.kernels` starts, feeds and stops them.

**What this does and does not protect.** A kernel runs with the same rights
as the app, because Python cannot be sandboxed from inside Python -- blocking
``open`` or ``socket`` is undone by one import. What the design does promise:
a hang or crash costs the notebook's variables, never the app or its data;
the kernel is never given the server's access token, so it cannot drive the
API as the user; its working directory is a scratch folder that is thrown
away, so results reach the project only through the library (``nlp.show``,
``nlp.figure``, ``nlp.save``). docs/SECURITY.md says the same to users.

**The protocol** is one JSON object per line, like the job worker's. The
kernel duplicates its standard output for the protocol at start-up and then
points file descriptor 1 at the log, so a C extension printing straight to
fd 1 cannot write into the middle of a message. Cell output (``print``) is
captured by replacing ``sys.stdout`` per cell and sent as ``stream`` events.

This is the only module that calls ``exec``/``eval`` (tests/test_security.py
allowlist); "Run and save" (:mod:`desktop_backend.notebooks`) runs cells
through :func:`run_cell` here. docs/SECURITY.md explains the limits.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, TextIO

__all__ = ["CellResult", "OutputSink", "display_value", "kernel_loop", "run_cell"]

#: Text a single cell may print before the rest is dropped (with a note).
MAX_CELL_TEXT = 1_000_000
#: Rows of a shown table sent inline; the whole table is a file the page can fetch.
PREVIEW_ROWS = 200
#: Rows the chart beside a table is drawn from.
CHART_ROWS = 5000
_CELL_NAME = "<cell"
#: Printed text is sent at each newline, or once this many characters wait.
_STREAM_BATCH = 4096


@dataclass(frozen=True)
class CellResult:
    """How one cell ended."""

    ok: bool
    seconds: float
    error: dict[str, Any] | None = None


def _cell_frames(tb: Any) -> list[str]:
    """The traceback lines that point into the user's cells, not into the engine."""
    lines: list[str] = []
    for frame in traceback.extract_tb(tb):
        if frame.filename.startswith(_CELL_NAME):
            where = frame.filename.strip("<>")
            lines.append(f"{where}, line {frame.lineno}" + (f": {frame.line}" if frame.line else ""))
    return lines


def _describe_error(exc: BaseException) -> dict[str, Any]:
    from core.script.session import SuiteError

    message = str(exc) or type(exc).__name__
    if isinstance(exc, SystemExit):
        message = "exit() ends nothing in a notebook; the cell stopped here."
    kind = "NLP Suite" if isinstance(exc, SuiteError) else type(exc).__name__
    return {"type": kind, "message": message, "trace": _cell_frames(exc.__traceback__)}


def run_cell(code: str, namespace: dict[str, Any], filename: str, display: Callable[[Any], None]) -> CellResult:
    """Run *code* in *namespace*; show its last expression's value, like Jupyter.

    The cell is parsed once. If its last statement is an expression, that
    expression is evaluated separately and its value handed to *display* --
    so a cell ending in ``rates`` shows the table without ``nlp.show``.
    Errors never escape: they come back as a :class:`CellResult` whose trace
    names only the user's own lines.
    """
    started = time.perf_counter()
    try:
        tree = ast.parse(code, filename=filename, mode="exec")
        last: ast.Expression | None = None
        final = tree.body[-1] if tree.body else None
        if isinstance(final, ast.Expr):
            tree.body.pop()
            last = ast.Expression(final.value)
        exec(compile(tree, filename, "exec"), namespace)  # noqa: S102 -- the notebook's own code, in its kernel
        if last is not None:
            value = eval(compile(last, filename, "eval"), namespace)  # noqa: S307 -- same cell, its last line
            if value is not None:
                display(value)
    except SyntaxError as exc:
        where = f"line {exc.lineno}" if exc.lineno else "this cell"
        return CellResult(
            False,
            round(time.perf_counter() - started, 3),
            {
                "type": "SyntaxError",
                "message": f"{exc.msg} ({where})",
                "trace": [exc.text.rstrip()] if exc.text else [],
            },
        )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        return CellResult(False, round(time.perf_counter() - started, 3), _describe_error(exc))
    return CellResult(True, round(time.perf_counter() - started, 3))


def display_value(value: Any) -> None:
    """What a cell's last value looks like under the cell."""
    import pandas as pd

    from core.script.session import NoteOutput, current
    import nlpsuite as nlp

    if isinstance(value, (pd.DataFrame, pd.Series, nlp.RunResult)):
        nlp.show(value)
    elif hasattr(value, "savefig"):
        nlp.figure(value)
    elif hasattr(value, "figure") and hasattr(value.figure, "savefig"):
        nlp.figure(value.figure)
    else:
        text = repr(value)
        current().emit(NoteOutput(text if len(text) <= MAX_CELL_TEXT else text[:MAX_CELL_TEXT] + " …"))


def keep_open_figures() -> None:
    """Figures a cell drew but never passed to ``nlp.figure``: shown, then closed.

    Jupyter shows ``plt.plot(...)`` without being asked; people expect the
    same, and a figure left open in a long-lived kernel is memory that never
    comes back.
    """
    import matplotlib.pyplot as plt

    import nlpsuite as nlp

    for number in plt.get_fignums():
        nlp.figure(plt.figure(number))


class _Stream(io.TextIOBase):
    """``sys.stdout`` for one cell: what it prints becomes ``stream`` events."""

    def __init__(self, name: str, send: Callable[[dict[str, Any]], None], exec_id: int) -> None:
        self.name_ = name
        self.send = send
        self.exec_id = exec_id
        self.sent = 0
        self.buffer_: list[str] = []
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if self.truncated:
            return len(text)
        if self.sent + len(text) > MAX_CELL_TEXT:
            text = text[: MAX_CELL_TEXT - self.sent] + "\n[output cut off: a cell may print at most 1 MB]\n"
            self.truncated = True
        self.buffer_.append(text)
        self.sent += len(text)
        if "\n" in text or sum(map(len, self.buffer_)) > _STREAM_BATCH:
            self.flush()
        return len(text)

    def flush(self) -> None:
        if self.buffer_:
            self.send({"event": "stream", "id": self.exec_id, "name": self.name_, "text": "".join(self.buffer_)})
            self.buffer_ = []


class OutputSink:
    """Where a cell's outputs go: files in the kernel's session folder, and an event each.

    Tables are sent as their first rows plus the file holding all of them;
    figures as PNG and SVG files. The page fetches files by name, so a large
    table never travels through the pipe.
    """

    def __init__(self, directory: Path, send: Callable[[dict[str, Any]], None]) -> None:
        self.directory = directory
        self.send = send
        self.exec_id = 0
        self.count = 0

    def __call__(self, output: Any) -> None:
        from core.script.session import FigureOutput, HtmlOutput, NoteOutput, TableOutput
        from desktop_backend.live import as_table

        self.count += 1
        stem = f"{self.exec_id:04d}_{self.count:03d}"
        if isinstance(output, HtmlOutput):
            # An interactive chart (nlp.chart): the HTML fragment is a file the
            # page frames, not a preview in an event -- the fragment is large
            # and the page fetches it like any other kernel output file.
            name = f"{stem}_{output.name}.html"
            (self.directory / name).write_text(output.html, encoding="utf-8")
            event: dict[str, Any] = {"kind": "html", "name": output.name, "file": name}
        elif isinstance(output, TableOutput):
            name = f"{stem}_{output.name}.csv"
            output.frame.to_csv(self.directory / name, index=False, encoding="utf-8")
            event = {
                "kind": "table",
                "name": output.name,
                "title": output.title,
                "file": name,
                "saved": output.saved,
                "preview": as_table(output.frame, PREVIEW_ROWS),
                "chart": output.chart,
                "chart_note": output.chart_note,
            }
            if output.chart is not None:
                event["chart_rows"] = as_table(
                    output.chart_frame if output.chart_frame is not None else output.frame, CHART_ROWS
                )
        elif isinstance(output, FigureOutput):
            png = f"{stem}_{output.name}.png"
            (self.directory / png).write_bytes(output.png)
            event = {"kind": "figure", "name": output.name, "png": png, "svg": None}
            # The SVG is optional (see core.script.session.FigureOutput): a
            # packaged runtime can lack matplotlib's SVG writer, and the page
            # shows the PNG with its download button alone.
            if output.svg is not None:
                svg = f"{stem}_{output.name}.svg"
                (self.directory / svg).write_bytes(output.svg)
                event["svg"] = svg
        elif isinstance(output, NoteOutput):
            event = {"kind": "text", "text": output.text}
        else:  # pragma: no cover - the library emits only the four kinds
            return
        self.send({"event": "output", "id": self.exec_id, **event})


def _report_scratch(send: Callable[[dict[str, Any]], None], exec_id: int, written: list[str]) -> None:
    """Say which files a cell wrote into the scratch folder, which is thrown away."""
    if not written:
        return
    send(
        {
            "event": "output",
            "id": exec_id,
            "kind": "text",
            "text": f"Written to a scratch folder and thrown away: {', '.join(written[:5])}. "
            "Keep a table with nlp.save(table, 'name') or a figure with nlp.figure(fig, 'name').",
        }
    )


def kernel_loop(root: Path, project_id: str, session_dir: Path, parser: str) -> int:
    """Serve cells from stdin until told to stop or the parent goes away."""
    protocol: TextIO = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)  # stray writes to fd 1 land in the log, never in the protocol
    sys.stdout = sys.stderr

    def send(message: dict[str, Any]) -> None:
        protocol.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
        protocol.flush()

    import matplotlib

    matplotlib.use("Agg")
    from core.script.session import Session, set_current
    from desktop_backend.project_source import ProjectSource
    from desktop_backend.store import Workspace
    import nlpsuite

    session_dir.mkdir(parents=True, exist_ok=True)
    scratch = session_dir / "scratch"
    scratch.mkdir(exist_ok=True)
    os.chdir(scratch)
    sink = OutputSink(session_dir, send)
    session = Session(ProjectSource(Workspace(root), project_id, parser=parser), on_output=sink)
    set_current(session)
    namespace: dict[str, Any] = {"__name__": "__main__", "nlp": nlpsuite}
    send({"event": "ready", "library": "1"})
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get("op") == "shutdown":
            break
        if message.get("op") != "exec":
            continue
        exec_id = int(message["id"])
        sink.exec_id = exec_id
        session.start_cell()
        before = set(os.listdir(scratch))
        out, err = _Stream("stdout", send, exec_id), _Stream("stderr", send, exec_id)
        sys.stdout, sys.stderr = out, err
        try:
            result = run_cell(
                str(message.get("code", "")), namespace, f"<cell {message.get('cell', exec_id)}>", display_value
            )
            try:
                keep_open_figures()
            except Exception as exc:
                err.write(f"A figure could not be kept: {exc}\n")
        finally:
            out.flush()
            err.flush()
            sys.stdout, sys.stderr = sys.__stderr__, sys.__stderr__
        _report_scratch(send, exec_id, sorted(set(os.listdir(scratch)) - before))
        send({"event": "done", "id": exec_id, "ok": result.ok, "seconds": result.seconds, "error": result.error})
    protocol.close()
    return 0
