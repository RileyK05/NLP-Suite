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
