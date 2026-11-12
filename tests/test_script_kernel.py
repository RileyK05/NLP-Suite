"""The notebook kernel: cells, their outputs, and stopping a runaway one.

The unit tests call :func:`run_cell` directly. The process tests start a real
kernel through :class:`KernelManager` (the same command the app uses) and
always stop it in ``finally`` -- a leftover kernel holds files open and
outlives the test run.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import pytest

from desktop_backend.kernel import run_cell
from desktop_backend.kernels import KernelManager, kernel_command
from desktop_backend.store import Workspace


class TestRunCell:
    def test_variables_persist_and_the_last_value_is_shown(self) -> None:
        namespace: dict[str, Any] = {}
        shown: list[Any] = []
        assert run_cell("x = 20", namespace, "<cell 1>", shown.append).ok
        assert run_cell("y = x + 1\ny * 2", namespace, "<cell 2>", shown.append).ok
        assert shown == [42]

    def test_a_statement_ending_shows_nothing(self) -> None:
        shown: list[Any] = []
        assert run_cell("z = 1", {}, "<cell 1>", shown.append).ok
        assert shown == []

    def test_an_error_names_the_users_line_not_the_engine(self) -> None:
        result = run_cell("a = 1\nb = a / 0", {}, "<cell 3>", print)
        assert not result.ok and result.error is not None
        assert result.error["type"] == "ZeroDivisionError"
        assert result.error["trace"] == ["cell 3, line 2"]

    def test_a_syntax_error_says_where(self) -> None:
        result = run_cell("for x in", {}, "<cell 1>", print)
        assert result.error is not None and result.error["type"] == "SyntaxError"
        assert "line 1" in result.error["message"]

    def test_exit_ends_the_cell_not_the_kernel(self) -> None:
        result = run_cell("exit()", {}, "<cell 1>", print)
        assert result.error is not None and "ends nothing" in result.error["message"]

    def test_suite_errors_read_as_the_suites(self) -> None:
        result = run_cell("import nlpsuite as nlp\nnlp.corpus()", {}, "<cell 1>", print)
        assert result.error is not None
        assert result.error["type"] == "NLP Suite"
        assert "not connected" in result.error["message"]


def test_the_kernel_is_never_handed_the_access_token(tmp_path: Path) -> None:
    command = kernel_command(tmp_path, "project", tmp_path / "session", "spacy")
    assert not any("token" in part.lower() for part in command)


# -------------------------------------------------------------- process --


def _wait(
    manager: KernelManager, key: tuple[str, str], exec_id: int, after: int, timeout: float = 120
) -> list[dict[str, Any]]:
    """Events for one cell, up to and including its ``done``."""
    events: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    cursor = after
    while time.monotonic() < deadline:
        state = manager.events(*key, cursor)
        cursor = state["next"]
        events.extend(e for e in state["events"] if e.get("id") == exec_id)
        if any(e.get("event") == "done" for e in events):
            return events
        time.sleep(0.1)
    raise AssertionError(f"cell {exec_id} did not finish: {events}")


@pytest.fixture
def kernel(tmp_path: Path) -> Any:
    workspace = Workspace(tmp_path / "ws")
    project = workspace.create("Kernel")
    workspace.import_document(project["id"], "1990-01-01_a.txt", b"The border was closed. Peace came.")
    workspace.import_document(project["id"], "2000-01-01_b.txt", b"Immigrants crossed the border.")
    manager = KernelManager(workspace)
    try:
        yield manager, (project["id"], "notebook-1")
    finally:
        manager.close()


def test_cells_run_print_and_show_tables(kernel: Any) -> None:
    manager, key = kernel
    first = manager.execute(*key, "c1", "import nlpsuite as nlp\nx = 41\nprint('hello')")
    events = _wait(manager, key, first["exec"], first["after"])
    assert [e["text"] for e in events if e.get("event") == "stream"] == ["hello\n"]
    assert events[-1]["ok"] is True

    second = manager.execute(
        *key, "c2", "rates = nlp.term_rates(nlp.corpus(), ['border'], match='exact-lowercase')\nrates"
    )
    events = _wait(manager, key, second["exec"], second["after"])
    [table] = [e for e in events if e.get("kind") == "table"]
    assert table["preview"]["total"] == 2
    assert "terms per 1000" in table["preview"]["columns"]  # a plain list is the group "terms"
    assert manager.file(*key, table["file"]).is_file()

    third = manager.execute(*key, "c3", "x + 1")
    events = _wait(manager, key, third["exec"], third["after"])
    assert [e["text"] for e in events if e.get("kind") == "text"] == ["42"]


def test_a_figure_without_an_svg_still_reaches_the_page(kernel: Any) -> None:
    """A missing SVG writer must cost the SVG, not the figure (kernel event)."""
    from core.script.session import FigureOutput
    from desktop_backend.kernel import OutputSink

    manager, _key = kernel
    sent: list[dict[str, Any]] = []
    directory = manager.root / "figure-test"
    directory.mkdir(parents=True, exist_ok=True)
    out = OutputSink(directory, sent.append)
    out.exec_id = 1
    out(FigureOutput("trend", b"\x89PNG\r\n\x1a\n", None))
    [event] = sent
    assert event["kind"] == "figure" and event["png"] and event["svg"] is None
    assert (directory / event["png"]).read_bytes().startswith(b"\x89PNG")
    assert not list(directory.glob("*.svg"))


def test_an_interactive_chart_is_written_and_announced(kernel: Any) -> None:
    """`nlp.chart` writes an HTML file the page can frame."""
    from core.script.session import HtmlOutput
    from desktop_backend.kernel import OutputSink

    manager, _key = kernel
    sent: list[dict[str, Any]] = []
    directory = manager.root / "chart-test"
    directory.mkdir(parents=True, exist_ok=True)
    out = OutputSink(directory, sent.append)
    out.exec_id = 1
    out(HtmlOutput("flows", "<div id='x'></div>"))
    [event] = sent
    assert event["kind"] == "html" and event["file"].endswith("_flows.html")
    assert (directory / event["file"]).read_text(encoding="utf-8") == "<div id='x'></div>"


def test_tables_archive_gathers_every_shown_table(kernel: Any) -> None:
    """The "Save all CSVs" download: every table so far, renamed, as one zip."""
    import io
    import zipfile

    manager, key = kernel
    first = manager.execute(*key, "c1", "import nlpsuite as nlp\nnlp.term_rates(nlp.corpus(), ['border'])")
    _wait(manager, key, first["exec"], first["after"])
    second = manager.execute(*key, "c2", "nlp.save(nlp.corpus().documents, 'documents')")
    _wait(manager, key, second["exec"], second["after"])

    data = manager.tables_archive(*key)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.testzip() is None
        names = sorted(archive.namelist())
        documents = archive.read("documents.csv").decode("utf-8")
    # Both tables are present; the unnamed one keeps its kernel name, the saved
    # one comes back as the name its author gave it.
    assert "documents.csv" in names and len(names) == 2, names
    assert "Document" in documents


def test_tables_archive_refuses_a_notebook_with_no_kernel(tmp_path: Path) -> None:
    manager = KernelManager(Workspace(tmp_path / "ws"))
    try:
        with pytest.raises(KeyError):
            manager.tables_archive("no-project", "no-notebook")
    finally:
        manager.close()


def test_errors_and_scratch_files_are_reported(kernel: Any) -> None:
    manager, key = kernel
    started = manager.execute(*key, "c1", "open('out.csv', 'w').write('a')\n1 / 0")
    events = _wait(manager, key, started["exec"], started["after"])
    done = events[-1]
    assert done["ok"] is False and done["error"]["type"] == "ZeroDivisionError"
    assert any("thrown away" in e.get("text", "") for e in events)


def test_stop_ends_a_runaway_cell_quickly(kernel: Any) -> None:
    manager, key = kernel
    started = manager.execute(*key, "c1", "while True:\n    pass")
    time.sleep(1.0)
    stopping = time.monotonic()
    manager.stop(*key)
    assert time.monotonic() - stopping < 5
    # The stopped kernel is gone; the next cell starts a fresh one with no variables.
    again = manager.execute(*key, "c2", "'x' in dir()")
    events = _wait(manager, key, again["exec"], again["after"])
    assert [e["text"] for e in events if e.get("kind") == "text"] == ["False"]
    assert started["exec"] == again["exec"]  # numbering restarts with the kernel


def test_a_changed_parser_restarts_the_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A kernel started under spaCy must not keep answering once the page asks for Stanza."""
    from types import SimpleNamespace

    from desktop_backend.kernels import Kernel

    workspace = Workspace(tmp_path / "ws")
    manager = KernelManager(workspace)
    started: list[Kernel] = []
    stopped: list[Kernel] = []

    def start(key: tuple[str, str], parser: str) -> Kernel:
        kernel = Kernel(key, SimpleNamespace(poll=lambda: None), tmp_path, parser=parser)  # type: ignore[arg-type]
        started.append(kernel)
        return kernel

    monkeypatch.setattr(manager, "_start", start)
    monkeypatch.setattr(manager, "_stop", stopped.append)
    try:
        first = manager.kernel("p", "n", "spacy")
        assert manager.kernel("p", "n", "spacy") is first
        first.busy = 3
        with pytest.raises(ValueError, match="still running with spacy"):
            manager.kernel("p", "n", "stanza")
        first.busy = None
        second = manager.kernel("p", "n", "stanza")
        assert second is not first and second.parser == "stanza"
        assert stopped == [first]
    finally:
        monkeypatch.setattr(manager, "_kernels", {})
        manager.close()


def test_save_all_csvs_keeps_every_table_under_its_own_name(tmp_path: Path) -> None:
    """Past output 999 the prefix still comes off, in run order; names differing only by case stay apart."""
    import io
    import zipfile

    from desktop_backend.kernels import Kernel

    workspace = Workspace(tmp_path / "ws")
    manager = KernelManager(workspace)
    directory = tmp_path / "scratch"
    directory.mkdir()
    for name in ("0003_999_early.csv", "0003_1000_late.csv", "0004_001_Data.csv", "0005_001_data.csv"):
        (directory / name).write_text(name, encoding="utf-8")
    manager._kernels[("p", "n")] = Kernel(("p", "n"), None, directory)  # type: ignore[arg-type]
    try:
        names = zipfile.ZipFile(io.BytesIO(manager.tables_archive("p", "n"))).namelist()
    finally:
        manager._kernels.clear()
        manager.close()
    assert names == ["early.csv", "late.csv", "Data.csv", "data_2.csv"]


def test_a_table_appears_only_once_it_is_completely_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save all CSVs globs *.csv while cells run; a table mid-write must not be one of them."""
    import pandas as pd

    from core.script.session import TableOutput
    from desktop_backend.kernel import OutputSink

    seen_during_write: list[list[str]] = []
    real_to_csv = pd.DataFrame.to_csv

    def watching(self: pd.DataFrame, path: Any, *args: Any, **kwargs: Any) -> Any:
        # The last moment of the write: everything is on disk but the sink
        # has not yet said the table is done.
        result = real_to_csv(self, path, *args, **kwargs)
        seen_during_write.append(sorted(p.name for p in tmp_path.glob("*.csv")))
        return result

    monkeypatch.setattr(pd.DataFrame, "to_csv", watching)
    out = OutputSink(tmp_path, lambda event: None)
    out.exec_id = 1
    out(TableOutput("rates", pd.DataFrame({"a": [1]}), saved=False))
    assert seen_during_write == [[]]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0001_001_rates.csv"]
