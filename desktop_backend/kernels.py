"""Starting, feeding and stopping notebook kernels (see :mod:`desktop_backend.kernel`).

One kernel per open notebook, at most :data:`MAX_KERNELS` at once (the one
used longest ago is stopped to make room), and none left idle past
:data:`IDLE_SECONDS`. The page talks to a kernel by polling: it sends a cell
and gets an ``exec`` number back at once, then asks for events after the last
one it has seen. A cell that runs for ten minutes costs ten minutes of short
requests, not one request the browser gives up on.

"Stop" cannot interrupt Python code from outside on Windows, so it is honest
about what it does: it ends the kernel. The notebook's variables go with it,
and the page says so.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
import uuid
import zipfile

from desktop_backend.runner import _close_pipes
from desktop_backend.store import Workspace

__all__ = ["IDLE_SECONDS", "MAX_KERNELS", "KernelManager"]

MAX_KERNELS = 2
IDLE_SECONDS = 15 * 60
_REAP_POLL_SECONDS = 30.0
#: Events kept per kernel; the page reads them within a second, so this is slack.
_EVENTS_KEPT = 5000


def kernel_command(root: Path, project_id: str, session_dir: Path, parser: str) -> list[str]:
    args = [
        "--data-dir",
        str(root),
        "--script-kernel",
        project_id,
        "--session",
        str(session_dir),
        "--parser",
        parser,
    ]
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "desktop_backend.server", *args]


@dataclass
class Kernel:
    """One running kernel and everything it has said."""

    key: tuple[str, str]
    process: subprocess.Popen[str]
    directory: Path
    events: list[dict[str, Any]] = field(default_factory=list)
    first_seq: int = 0
    next_exec: int = 1
    busy: int | None = None
    ready: bool = False
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def add(self, event: dict[str, Any]) -> None:
        with self.lock:
            if event.get("event") == "ready":
                self.ready = True
            if event.get("event") == "done" and event.get("id") == self.busy:
                self.busy = None
            self.events.append(event)
            overflow = len(self.events) - _EVENTS_KEPT
            if overflow > 0:
                del self.events[:overflow]
                self.first_seq += overflow


class KernelManager:
    """Every notebook kernel this server has started."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.root = workspace.root / "kernels"
        self._kernels: dict[tuple[str, str], Kernel] = {}
        self._lock = threading.RLock()
        self._closing = False
        # Folders from kernels of an earlier server are scratch nobody can use.
        shutil.rmtree(self.root, ignore_errors=True)
        threading.Thread(target=self._reap, daemon=True, name="kernel-reaper").start()

    # -- lifecycle -----------------------------------------------------------

    def _start(self, key: tuple[str, str], parser: str) -> Kernel:
        directory = self.root / uuid.uuid4().hex[:12]
        directory.mkdir(parents=True)
        log = (directory / "kernel.log").open("ab")
        try:
            process = subprocess.Popen(  # noqa: S603 -- fixed engine argv, no shell
                kernel_command(self.workspace.root, key[0], directory, parser),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        finally:
            log.close()
        kernel = Kernel(key, process, directory)
        threading.Thread(target=self._read, args=(kernel,), daemon=True, name="kernel-reader").start()
        return kernel

    def _read(self, kernel: Kernel) -> None:
        assert kernel.process.stdout is not None  # noqa: S101
        with contextlib.suppress(OSError, ValueError):
            for line in kernel.process.stdout:
                with contextlib.suppress(ValueError):
                    kernel.add(json.loads(line))
        # The pipe closed: the kernel ended, on purpose or not. A cell that was
        # running gets an ending the page can show instead of spinning forever.
        if kernel.busy is not None:
            kernel.add(
                {
                    "event": "done",
                    "id": kernel.busy,
                    "ok": False,
                    "seconds": None,
                    "error": {
                        "type": "Stopped",
                        "message": "The notebook's Python stopped, and its variables went with it. Run the cells again.",
                        "trace": [],
                    },
                }
            )
        kernel.add({"event": "ended"})

    def _stop(self, kernel: Kernel) -> None:
        """Kill first, then close the pipes (see Runner._stop_worker for why)."""
        if kernel.process.poll() is None:
            kernel.process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                kernel.process.wait(timeout=10)
        _close_pipes(kernel.process)
        shutil.rmtree(kernel.directory, ignore_errors=True)

    def kernel(self, project_id: str, notebook_id: str, parser: str = "spacy") -> Kernel:
        """The notebook's kernel, started if it has none (or its last one ended)."""
        key = (project_id, notebook_id)
        with self._lock:
            if self._closing:
                raise ValueError("The app is closing.")
            found = self._kernels.get(key)
            if found is not None and found.alive:
                found.last_used = time.monotonic()
                return found
            if found is not None:
                self._stop(found)
                del self._kernels[key]
            while len(self._kernels) >= MAX_KERNELS:
                oldest = min(self._kernels.values(), key=lambda k: k.last_used)
                self._stop(oldest)
                del self._kernels[oldest.key]
            self._kernels[key] = self._start(key, parser)
            return self._kernels[key]

    def stop(self, project_id: str, notebook_id: str) -> None:
        with self._lock:
            found = self._kernels.pop((project_id, notebook_id), None)
        if found is not None:
            self._stop(found)

    def close(self) -> None:
        with self._lock:
            self._closing = True
            kernels = list(self._kernels.values())
            self._kernels.clear()
        for kernel in kernels:
            self._stop(kernel)

    def _reap(self) -> None:
        while not self._closing:
            time.sleep(_REAP_POLL_SECONDS)
            with self._lock:
                idle = [
                    k
                    for k in self._kernels.values()
                    if k.busy is None and time.monotonic() - k.last_used > IDLE_SECONDS
                ]
                for kernel in idle:
                    del self._kernels[kernel.key]
            for kernel in idle:
                self._stop(kernel)

    # -- talking to one ------------------------------------------------------

    def execute(self, project_id: str, notebook_id: str, cell: str, code: str, parser: str = "spacy") -> dict[str, Any]:
        kernel = self.kernel(project_id, notebook_id, parser)
        with kernel.lock:
            if kernel.busy is not None:
                raise ValueError("A cell is still running. Wait for it, or stop the notebook.")
            exec_id = kernel.next_exec
            kernel.next_exec += 1
            kernel.busy = exec_id
            after = kernel.first_seq + len(kernel.events)
        assert kernel.process.stdin is not None  # noqa: S101
        try:
            kernel.process.stdin.write(json.dumps({"op": "exec", "id": exec_id, "cell": cell, "code": code}) + "\n")
            kernel.process.stdin.flush()
        except (OSError, ValueError) as exc:
            kernel.busy = None
            raise ValueError("The notebook's Python has stopped. Run the cell again to start it.") from exc
        kernel.last_used = time.monotonic()
        return {"exec": exec_id, "after": after}

    def events(self, project_id: str, notebook_id: str, after: int) -> dict[str, Any]:
        """Events numbered *after* and later, and the kernel's state."""
        with self._lock:
            kernel = self._kernels.get((project_id, notebook_id))
        if kernel is None:
            return {"events": [], "next": after, "running": False, "busy": None, "ready": False}
        kernel.last_used = time.monotonic()
        with kernel.lock:
            start = max(after - kernel.first_seq, 0)
            fresh = kernel.events[start:]
            following = kernel.first_seq + len(kernel.events)
            return {
                "events": fresh,
                "next": following,
                "running": kernel.alive,
                "busy": kernel.busy,
                "ready": kernel.ready,
            }

    def file(self, project_id: str, notebook_id: str, name: str) -> Path:
        """One output file of the notebook's current kernel, refusing anything outside its folder."""
        with self._lock:
            kernel = self._kernels.get((project_id, notebook_id))
        if kernel is None:
            raise KeyError("This notebook has no running kernel.")
        path = (kernel.directory / name).resolve()
        if path.parent != kernel.directory.resolve() or not path.is_file():
            raise KeyError("No such output.")
        return path

    def tables_archive(self, project_id: str, notebook_id: str) -> bytes:
        """Every table this kernel has written, as one zip (``<name>.csv`` each).

        Files are named ``0001_002_rates.csv`` (exec, output, table), which
        sorts in cells' run order; here they are renamed to the table's own
        name so the zip reads as the writer's tables rather than as scratch.
        Duplicates keep a numeric suffix. Nothing else in the folder (figures,
        the kernel log) is included: this is the "Save all CSVs" download.
        """
        with self._lock:
            kernel = self._kernels.get((project_id, notebook_id))
        if kernel is None:
            raise KeyError("This notebook has no running kernel.")
        directory = kernel.directory
        buffer = io.BytesIO()
        used: set[str] = set()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(directory.glob("*.csv")):
                # Strip the kernel's "0031_002_" prefix; a name the kernel did
                # not give (a file the notebook wrote itself) is kept as is.
                stem = re.sub(r"^\d{4}_\d{3}_", "", path.stem) or path.stem
                target = f"{stem}.csv"
                counter = 2
                while target in used:
                    target = f"{stem}_{counter}.csv"
                    counter += 1
                used.add(target)
                archive.write(path, target)
        return buffer.getvalue()
