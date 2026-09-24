"""Every state a run can be in must have a word for it in the desktop.

A job's state is an identifier the runner writes into SQLite. The desktop had
a word for exactly one of them, ``DONE`` -> "Completed", and printed the rest
in lower case, so a finished run and a failed one were labelled in two
different registers: "Completed" beside "failed".

Two things can go wrong and neither shows up at runtime: the runner learns a
new state that the desktop has no word for, or the declared tuple stops
matching the states the runner actually writes. Both are checked here.
"""

from __future__ import annotations

from pathlib import Path
import re

from desktop_backend.runner import JOB_STATES

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "desktop_backend" / "runner.py"
STORE = ROOT / "desktop_backend" / "store.py"
STATUS = ROOT / "desktop" / "src" / "RunStatus.tsx"


def _states_written() -> set[str]:
    """States the code actually assigns, recovered from the source.

    Three spellings, because the runner writes a state three ways: a keyword
    argument (``state="FAILED"``), an UPDATE (``SET state='CANCELLED'``) and
    the INSERT that creates the row (``VALUES (?, ?, ?, 'QUEUED', ...)``).
    """
    found: set[str] = set()
    for path in (RUNNER, STORE):
        source = path.read_text(encoding="utf-8")
        found |= set(re.findall(r"""state\s*=\s*["']([A-Z_]+)["']""", source))
        found |= set(re.findall(r"""SET\s+state\s*=\s*'([A-Z_]+)'""", source))
        found |= set(re.findall(r"""VALUES\s*\([^)]*?'([A-Z_]+)'""", source, re.DOTALL))
    return found


def _desktop_labels() -> dict[str, str]:
    source = STATUS.read_text(encoding="utf-8")
    body = source.split("const STATE_LABELS", 1)[1].split("};", 1)[0]
    return dict(re.findall(r"(\w+):\s*\"([^\"]+)\"", body))


class TestTheDeclaredStatesAreTheRealOnes:
    def test_every_state_the_runner_writes_is_declared(self) -> None:
        undeclared = sorted(_states_written() - set(JOB_STATES))
        assert not undeclared, f"the runner writes {undeclared}, which JOB_STATES does not name"

    def test_the_extractor_still_finds_the_states_it_should(self) -> None:
        """Guards the guard: a regex that matches nothing passes vacuously."""
        written = _states_written()
        assert {"QUEUED", "RUNNING", "DONE", "FAILED", "CANCELLED"} <= written, sorted(written)

    def test_the_conditional_completion_states_are_declared(self) -> None:
        """``DONE if ok else PARTIAL if child else FAILED`` assigns three at once."""
        assert {"DONE", "PARTIAL", "FAILED"} <= set(JOB_STATES)

    def test_no_state_is_named_twice(self) -> None:
        assert len(JOB_STATES) == len(set(JOB_STATES))


class TestTheDesktopHasAWordForEachOne:
    def test_every_state_is_labelled(self) -> None:
        labels = _desktop_labels()
        assert labels, f"recovered no STATE_LABELS from {STATUS}; the extractor needs updating"
        missing = sorted(set(JOB_STATES) - set(labels))
        assert not missing, (
            f"{missing} would be shown to the reader as the identifier in lower case; "
            "add them to STATE_LABELS in desktop/src/RunStatus.tsx"
        )

    def test_no_label_names_a_state_that_cannot_happen(self) -> None:
        orphans = sorted(set(_desktop_labels()) - set(JOB_STATES))
        assert not orphans, f"STATE_LABELS names {orphans}, which no job can be in"

    def test_the_labels_are_written_in_one_register(self) -> None:
        for state, label in _desktop_labels().items():
            assert label[0].isupper(), f"{state} -> {label!r}"
            assert label != state, f"{state} is labelled with its own identifier"
            assert not label.isupper(), f"{state} -> {label!r} is shouting"

    def test_the_labels_are_distinct(self) -> None:
        """Two states sharing a word means a reader cannot tell them apart."""
        labels = list(_desktop_labels().values())
        assert len(labels) == len(set(labels)), labels
