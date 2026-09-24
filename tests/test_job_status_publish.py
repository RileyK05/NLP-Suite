"""Atomic status publication tolerates brief Windows reader/scanner locks."""

import json
from pathlib import Path

import pytest

from core import jobs


def _locked() -> PermissionError:
    error = PermissionError("Windows sharing violation")
    error.winerror = 32
    return error


def test_status_retries_transient_windows_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs._write_status(str(tmp_path), jobs.JobStatus(id="job", tool="doctor"))
    original = jobs.os.replace
    attempts = []
    sleeps = []

    def replace(source: str, target: str) -> None:
        attempts.append(source)
        if len(attempts) < 3:
            assert json.loads(Path(target).read_text())["state"] == "RUNNING"
            raise _locked()
        original(source, target)

    monkeypatch.setattr(jobs.os, "replace", replace)
    monkeypatch.setattr(jobs.time, "sleep", sleeps.append)
    jobs._write_status(str(tmp_path), jobs.JobStatus(id="job", tool="doctor", state="DONE"))
    assert len(attempts) == 3
    assert sleeps == [0.01, 0.02]
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "DONE"


@pytest.mark.parametrize("windows", [False, True])
def test_status_publication_failure_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: bool) -> None:
    jobs._write_status(str(tmp_path), jobs.JobStatus(id="job", tool="doctor"))
    attempts = []
    sleeps = []

    def replace(source: str, target: str) -> None:
        attempts.append(source)
        raise _locked() if windows else PermissionError("Permanent denial")

    monkeypatch.setattr(jobs.os, "replace", replace)
    monkeypatch.setattr(jobs.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        jobs._write_status(str(tmp_path), jobs.JobStatus(id="job", tool="doctor", state="DONE"))
    assert len(attempts) == (7 if windows else 1)
    assert len(sleeps) == (6 if windows else 0)
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "RUNNING"
