"""The Models page's engine side: list models, download, cancel, delete.

A download runs on a background thread in the engine process: models are
shared by every project, so they do not fit the project-scoped job runner,
and a download is network time, not analysis work a worker should hold.
Progress is polled through ``GET /api/models`` like a job's status. The
finished files land in the user's models directory
(``core.models.locate.user_models_dir``), which the server points at
``<app data>/models`` so they survive app updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

from core.models import download as model_download
from core.models.locate import model_dir, status, user_models_dir
from core.models.registry import ModelSpec, get_model, models_of_kind
from core.profiler.registry import TOOL_REGISTRY
from desktop_backend.catalog import INTERNAL_TOOLS

__all__ = ["KIND_LABELS", "ModelDownloads"]

KIND_LABELS = {
    "token_embeddings": "Word vectors",
    "sentence_embeddings": "Sentence and document vectors",
    "classifier": "Sentiment",
}


@dataclass
class _Download:
    thread: threading.Thread | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    state: str = "downloading"  # downloading | failed | cancelled
    done: int = 0
    total: int = 0
    message: str = ""


def _used_by(spec: ModelSpec) -> list[str]:
    """Tools that can run this model (its id is among their model choices)."""
    names = []
    for tool in TOOL_REGISTRY:
        if tool.name in INTERNAL_TOOLS:
            continue
        for param in tool.params:
            if param.name == "model" and spec.id in param.choices:
                names.append(tool.name)
    return names


class ModelDownloads:
    """At most one download per model; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: dict[str, _Download] = {}

    def describe(self, spec: ModelSpec) -> dict[str, Any]:
        with self._lock:
            job = self._running.get(spec.id)
        state = status(spec)
        folder = model_dir(spec)
        described: dict[str, Any] = {
            "id": spec.id,
            "name": spec.display_name,
            "kind": spec.kind,
            "kindLabel": KIND_LABELS[spec.kind],
            "description": spec.description,
            "sizeMb": spec.size_mb,
            "status": state,
            "bundled": spec.bundled,
            # A bundled model reads from inside the app; only a downloaded copy
            # can be removed.
            "removable": folder is not None and folder.parent == user_models_dir(),
            "license": spec.license,
            "source": f"https://huggingface.co/{spec.source}",
            "precision": spec.precision,
            "usedBy": _used_by(spec),
            "download": None,
        }
        if job is not None:
            described["download"] = {
                "state": job.state,
                "done": job.done,
                "total": job.total,
                "message": job.message,
            }
            if job.state == "downloading":
                described["status"] = "downloading"
        return described

    def listing(self) -> dict[str, Any]:
        return {
            "models": [self.describe(spec) for spec in models_of_kind()],
            "folder": str(user_models_dir()),
        }

    def start(self, model_id: str) -> dict[str, Any]:
        spec = get_model(model_id)
        if spec is None:
            raise KeyError(model_id)
        with self._lock:
            running = self._running.get(spec.id)
            busy = running is not None and running.state == "downloading"
            ready = not busy and status(spec) == "ready"
            if ready:
                self._running.pop(spec.id, None)
            if busy or ready:
                job = None
            else:
                job = _Download(total=spec.size_bytes)
                self._running[spec.id] = job
        if job is None:
            return self.describe(spec)
        self._launch(spec, job)
        return self.describe(spec)

    def _launch(self, spec: ModelSpec, job: _Download) -> None:

        def progress(done: int, total: int) -> None:
            job.done = done
            job.total = total

        def run() -> None:
            result = model_download.download(spec, progress=progress, cancel=job.cancel)
            with self._lock:
                if result.ok:
                    # Finished: the model's status now says ready on its own.
                    self._running.pop(spec.id, None)
                    return
                diag = result.diagnostics[0]
                job.state = "cancelled" if diag.code == "MODEL_DOWNLOAD_CANCELLED" else "failed"
                job.message = diag.message

        job.thread = threading.Thread(target=run, name=f"model-download-{spec.id}", daemon=True)
        job.thread.start()

    def cancel(self, model_id: str) -> dict[str, Any]:
        spec = get_model(model_id)
        if spec is None:
            raise KeyError(model_id)
        with self._lock:
            job = self._running.get(spec.id)
        if job is not None:
            job.cancel.set()
        return self.describe(spec)

    def wait(self, model_id: str, timeout: float | None = None) -> None:
        """Block until a running download ends (tests and shutdown)."""
        spec = get_model(model_id)
        with self._lock:
            job = self._running.get(spec.id) if spec is not None else None
        if job is not None and job.thread is not None:
            job.thread.join(timeout)

    def delete(self, model_id: str) -> dict[str, Any]:
        spec = get_model(model_id)
        if spec is None:
            raise KeyError(model_id)
        self.cancel(spec.id)
        self.wait(spec.id, timeout=30)
        result = model_download.delete(spec)
        with self._lock:
            self._running.pop(spec.id, None)
        described = self.describe(spec)
        described["freedMb"] = round((result.value or 0) / 1_000_000)
        if not result.ok:
            described["error"] = result.diagnostics[0].message
        return described

    def close(self) -> None:
        with self._lock:
            jobs = list(self._running.values())
        for job in jobs:
            job.cancel.set()
