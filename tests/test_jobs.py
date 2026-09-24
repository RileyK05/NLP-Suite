"""FR-8.4 — background job contract tests (fast tools only)."""

from __future__ import annotations

from pathlib import Path
import time

import pytest

from core.jobs import JobStatus, list_jobs, read_status, submit


def _wait(job_id: str, root: str, timeout: float = 120.0) -> JobStatus:
    """Poll until the job leaves RUNNING.

    The 120s cap (not 30) is load headroom: the supervisor and the tool are
    two fresh interpreters, and under a full-suite run the in-process model
    tests contend for CPU/disk long enough to starve interpreter startup.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = read_status(job_id, jobs_root=root).unwrap()
        if status.state != "RUNNING":
            return status
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} still RUNNING after {timeout}s")


class TestJobs:
    def test_submit_and_finish(self, tmp_path: Path) -> None:
        root = str(tmp_path / "jobs")
        job = submit("doctor", ["--help"], jobs_root=root).unwrap()
        assert job.status.state == "RUNNING"
        final = _wait(job.status.id, root)
        assert final.state == "DONE"
        assert final.exit_code == 0
        assert (Path(job.directory) / "stdout.log").is_file()

    def test_failing_tool_records_failed(self, tmp_path: Path) -> None:
        root = str(tmp_path / "jobs")
        job = submit("no-such-tool-xyz", [], jobs_root=root)
        assert not job.ok
        assert job.diagnostics[0].code == "JOBS_UNKNOWN_TOOL"

    def test_tool_error_becomes_failed_outcome(self, tmp_path: Path) -> None:
        root = str(tmp_path / "jobs")
        # filenames with no argv exits non-zero (argparse) — a FAILED job, not a crash
        job = submit("filenames", [], jobs_root=root).unwrap()
        final = _wait(job.status.id, root)
        assert final.state == "FAILED"
        assert final.exit_code != 0

    def test_unknown_id(self, tmp_path: Path) -> None:
        res = read_status("nope", jobs_root=str(tmp_path / "jobs"))
        assert not res.ok
        assert res.diagnostics[0].code == "JOBS_UNKNOWN_ID"

    def test_list_newest_first(self, tmp_path: Path) -> None:
        root = str(tmp_path / "jobs")
        first = submit("doctor", ["--help"], jobs_root=root).unwrap()
        time.sleep(1.1)  # job ids sort by second-precision timestamp
        second = submit("doctor", ["--help"], jobs_root=root).unwrap()
        listed = list_jobs(jobs_root=root).unwrap()
        ids = [s.id for s in listed]
        assert ids == sorted(ids, reverse=True)
        assert {first.status.id, second.status.id} <= set(ids)

    def test_list_empty_root(self, tmp_path: Path) -> None:
        assert list_jobs(jobs_root=str(tmp_path / "nothing")).unwrap() == []


class TestJobsCli:
    def test_cli_submit_status_list(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.jobs import main

        root = str(tmp_path / "jobs")
        assert main(["--jobs-root", root, "submit", "doctor", "--help"]) == 0
        job_id = capsys.readouterr().out.strip()
        assert main(["--jobs-root", root, "status", job_id]) == 0
        assert job_id in capsys.readouterr().out
        assert main(["--jobs-root", root, "list"]) == 0

    def test_cli_unknown_tool(self, tmp_path: Path) -> None:
        from tools.jobs import main

        assert main(["--jobs-root", str(tmp_path / "jobs"), "submit", "nope"]) == 1
