"""Background jobs (FR-8.4) — run any tool CLI off the critical path.

A job runs ``python -m tools.<name> <argv>`` in a subprocess with stdout /
stderr captured to the job directory; ``status.json`` tracks RUNNING →
DONE / FAILED with the exit code. Subprocesses (not threads-attached-to
``sys.stdout``) keep each job's logs exact: ``contextlib.redirect_stdout``
replaces ``sys.stdout`` process-wide and would swallow concurrent output
from the caller. An independent supervisor records completion even after
the submitting CLI exits. Status updates use atomic file replacement.

Job directories live under a jobs root (default ``out/jobs``), one
directory per job: ``status.json``, ``stdout.log``, ``stderr.log``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import importlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import threading
import time
import uuid

from core.result import Diagnostic, Result

__all__ = ["Job", "JobStatus", "list_jobs", "read_status", "submit"]

DEFAULT_JOBS_ROOT = "out/jobs"
_STATUS_REPLACE_ATTEMPTS = 7


@dataclass(frozen=True, slots=True)
class JobStatus:
    """Serializable state of one job."""

    id: str
    tool: str
    argv: tuple[str, ...] = ()
    state: str = "RUNNING"  # RUNNING | DONE | FAILED
    exit_code: int | None = None
    started: str = ""
    finished: str = ""
    error: str = ""


@dataclass(frozen=True, slots=True)
class Job:
    """A submitted job: its status plus where its logs live."""

    status: JobStatus
    directory: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _job_dir(root: str, job_id: str) -> str:
    path = os.path.join(root, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def _write_status(directory: str, status: JobStatus) -> None:
    temporary = os.path.join(directory, f".status-{uuid.uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(asdict(status), handle, indent=2)
    # A polling reader or scanner can briefly hold a Windows handle that
    # disallows replacement. Retain atomic publication; never truncate the
    # visible status or retry permanent errors indefinitely.
    for attempt in range(_STATUS_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, os.path.join(directory, "status.json"))
            break
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == _STATUS_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.01 * 2**attempt)


def _resolve_entry(tool: str) -> Diagnostic | None:
    """Fail fast on unknown tools without spawning a process."""
    try:
        module = importlib.import_module(f"tools.{tool}")
    except ImportError:
        return Diagnostic.error("JOBS_UNKNOWN_TOOL", f"unknown tool {tool!r}", tool=tool)
    if not callable(getattr(module, "main", None)):
        return Diagnostic.error("JOBS_NO_ENTRY", f"tool {tool!r} has no main()", tool=tool)
    return None


def _run_job(directory: str) -> int:
    """Independent supervisor: survives the submitting CLI and owns final status."""
    path = Path(directory)
    status = read_status(path.name, jobs_root=str(path.parent)).unwrap()
    try:
        exit_code = subprocess.run(  # noqa: S603
            [sys.executable, "-m", f"tools.{status.tool}", *status.argv], check=False
        ).returncode
        error = ""
    except BaseException as exc:  # the watcher must not die silently
        exit_code = 1
        error = f"{type(exc).__name__}: {exc}"
    finished = JobStatus(
        id=status.id,
        tool=status.tool,
        argv=status.argv,
        state="DONE" if exit_code == 0 else "FAILED",
        exit_code=exit_code,
        started=status.started,
        finished=_now(),
        error=error,
    )
    _write_status(directory, finished)
    return exit_code


def submit(tool: str, argv: Sequence[str] | None = None, *, jobs_root: str = DEFAULT_JOBS_ROOT) -> Result[Job]:
    """Start ``python -m tools.<tool>`` in the background; returns immediately."""
    problem = _resolve_entry(tool)
    if problem is not None:
        return Result.failure(problem)
    job_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    status = JobStatus(id=job_id, tool=tool, argv=tuple(argv or ()), started=_now())
    try:
        directory = os.path.abspath(_job_dir(jobs_root, job_id))
        _write_status(directory, status)
    except OSError as exc:
        return Result.failure(Diagnostic.error("JOBS_WRITE_FAILED", f"cannot create job: {exc}"))
    try:
        with (
            open(os.path.join(directory, "stdout.log"), "w", encoding="utf-8") as out,
            open(os.path.join(directory, "stderr.log"), "w", encoding="utf-8") as err,
        ):
            # Running the analyst's chosen tool is the feature (shell=False,
            # argv list, same interpreter); unsanitized input cannot escape
            # the argv array.
            proc = subprocess.Popen(  # noqa: S603
                [sys.executable, "-m", "core.jobs", directory],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                cwd=os.getcwd(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # Windows-only flag
                start_new_session=os.name != "nt",
            )
    except OSError as exc:
        failed = replace(
            status,
            state="FAILED",
            exit_code=1,
            finished=_now(),
            error=f"OSError: {exc}",
        )
        _write_status(directory, failed)
        return Result(
            Job(status=failed, directory=directory),
            (Diagnostic.error("JOBS_START_FAILED", failed.error),),
        )
    # Reap while this process lives; completion recording is in the supervisor,
    # not this daemon thread (which disappears when a submitting CLI exits).
    watcher = threading.Thread(target=proc.wait, name=f"nlp-job-{job_id}", daemon=True)
    watcher.start()
    return Result.success(Job(status=status, directory=directory))


def read_status(job_id: str, *, jobs_root: str = DEFAULT_JOBS_ROOT) -> Result[JobStatus]:
    """Read a job's current status (RUNNING until the worker rewrites it)."""
    if not job_id or job_id in (".", "..") or "/" in job_id or "\\" in job_id or PureWindowsPath(job_id).drive:
        return Result.failure(Diagnostic.error("JOBS_UNKNOWN_ID", f"invalid job id {job_id!r}"))
    root = Path(jobs_root).resolve()
    target = root / job_id / "status.json"
    if not target.resolve().is_relative_to(root):
        return Result.failure(Diagnostic.error("JOBS_UNKNOWN_ID", f"job {job_id!r} escapes jobs root"))
    if not os.path.isfile(target):
        return Result.failure(Diagnostic.error("JOBS_UNKNOWN_ID", f"no job {job_id!r} under {jobs_root}"))
    try:
        payload = _read_status_payload(target)
        if not isinstance(payload, dict) or payload.get("id") != job_id:
            raise ValueError("status must be an object with a matching job id")
        argv = payload.get("argv", [])
        if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("status argv must be a list of strings")
        payload["argv"] = tuple(argv)
        return Result.success(JobStatus(**{k: payload[k] for k in JobStatus.__dataclass_fields__ if k in payload}))
    except (OSError, ValueError, TypeError) as exc:
        return Result.failure(Diagnostic.error("JOBS_STATUS_UNREADABLE", f"cannot read status for {job_id!r}: {exc}"))


def _read_status_payload(target: Path) -> object:
    # Atomic replacement can briefly deny a Windows reader as well as a writer.
    # Retry only OS sharing/access errors, never malformed JSON or missing files.
    for attempt in range(_STATUS_REPLACE_ATTEMPTS):
        try:
            with open(target, encoding="utf-8") as handle:
                return json.load(handle)
        except OSError as exc:
            # Python's CRT-backed open() may expose EACCES without winerror.
            sharing_error = getattr(exc, "winerror", None) in (5, 32, 33) or (
                os.name == "nt" and isinstance(exc, PermissionError)
            )
            if not sharing_error or attempt == _STATUS_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.01 * 2**attempt)
    raise AssertionError("Unreachable status read retry state")


def list_jobs(*, jobs_root: str = DEFAULT_JOBS_ROOT) -> Result[list[JobStatus]]:
    """Newest-first statuses for every job under the root."""
    if not os.path.isdir(jobs_root):
        return Result.success([])
    found: list[JobStatus] = []
    for job_id in sorted(os.listdir(jobs_root), reverse=True):
        status = read_status(job_id, jobs_root=jobs_root)
        if status.value is not None:
            found.append(status.unwrap())
    return Result.success(found)


if __name__ == "__main__":
    raise SystemExit(_run_job(sys.argv[1]))
