"""Download a published model into the user's models directory, and remove one.

A download streams each file to ``<file>.partial`` and **resumes** from
there with an HTTP ``Range`` request, so a dropped connection or a closed
laptop costs only what was not yet fetched. Each file is checked against
its published size and SHA-256 before it is renamed into place, which is
atomic: a model folder never holds a half-written ``model.onnx`` under its
real name. Running a finished download again is a no-op.

Every failure is one plain sentence (no traceback): offline or blocked by
a proxy, not enough disk, a file that arrived damaged (deleted, so the
next attempt starts clean), or cancelled (partial files kept, so the next
attempt resumes).
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from http import HTTPStatus
import http.client
import os
from pathlib import Path
import shutil
import ssl
import threading
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from core.models.locate import user_models_dir
from core.models.registry import ModelFile, ModelSpec, release_url
from core.result import Diagnostic, Result

__all__ = ["delete", "download", "free_space_needed"]

_CHUNK = 1 << 20
_TIMEOUT = 60.0
#: Keep this much more free than the download itself needs.
_HEADROOM = 1.10

Progress = Callable[[int, int], None]
Opener = Callable[..., Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_space_needed(spec: ModelSpec, root: Path | None = None) -> int:
    """Bytes still to fetch (files already in place and verified count as done)."""
    folder = (root or user_models_dir()) / spec.id
    remaining = 0
    for item in spec.files:
        final = folder / item.name
        if final.is_file() and final.stat().st_size == item.size:
            continue
        partial = folder / f"{item.name}.partial"
        already = partial.stat().st_size if partial.is_file() else 0
        remaining += max(0, item.size - already)
    return remaining


def _checked(url: str) -> Diagnostic | None:
    """Only https (and loopback http, for tests) may be fetched."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")):
        return None
    return Diagnostic.error("MODEL_DOWNLOAD_REFUSED", f"refusing to download from {url!r}: not an https URL", url=url)


def _offline(exc: BaseException) -> Diagnostic:
    return Diagnostic.error(
        "MODEL_DOWNLOAD_OFFLINE",
        "Couldn't reach the download server. Check your internet connection (or proxy) and try again.",
        reason=str(exc),
    )


def _fetch(  # noqa: PLR0911, PLR0912, PLR0913 -- resume, restart, cancel and each failure are separate paths
    url: str,
    item: ModelFile,
    folder: Path,
    *,
    opener: Opener,
    cancel: threading.Event | None,
    advance: Callable[[int], None],
) -> Diagnostic | None:
    """One file into place, resuming a partial; a Diagnostic says why not."""
    final = folder / item.name
    partial = folder / f"{item.name}.partial"
    start = partial.stat().st_size if partial.is_file() else 0
    if start > item.size:
        partial.unlink()
        start = 0
    digest = hashlib.sha256()
    if start:
        with partial.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
        advance(start)
    if start < item.size:
        refused = _checked(url)
        if refused is not None:
            return refused
        request = urllib.request.Request(url, headers={"User-Agent": "NLP-Suite model download"})  # noqa: S310
        if start:
            request.add_header("Range", f"bytes={start}-")
        try:
            with opener(request, timeout=_TIMEOUT) as response:
                status = getattr(response, "status", 200)
                if start and status != HTTPStatus.PARTIAL_CONTENT:
                    # The server ignored the Range: start this file over.
                    advance(-start)
                    start = 0
                    digest = hashlib.sha256()
                with partial.open("ab" if start else "wb") as out:
                    while True:
                        if cancel is not None and cancel.is_set():
                            return Diagnostic.warning(
                                "MODEL_DOWNLOAD_CANCELLED",
                                "Download cancelled. Starting it again picks up where it stopped.",
                            )
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        digest.update(chunk)
                        advance(len(chunk))
        except urllib.error.HTTPError as exc:
            if exc.code == HTTPStatus.NOT_FOUND:
                return Diagnostic.error(
                    "MODEL_DOWNLOAD_MISSING",
                    "This model's files are not on the download server yet. Try again after the next update.",
                    url=url,
                )
            return _offline(exc)
        except (urllib.error.URLError, OSError, ssl.SSLError, http.client.HTTPException) as exc:
            return _offline(exc)
    size = partial.stat().st_size
    if size < item.size:
        # The connection ended early without an error (a dropped socket reads
        # as a short body). Keep what arrived: the next attempt resumes it.
        return _offline(ConnectionError(f"{item.name}: {size} of {item.size} bytes arrived"))
    if size != item.size or digest.hexdigest() != item.sha256:
        partial.unlink(missing_ok=True)
        return Diagnostic.error(
            "MODEL_DOWNLOAD_CORRUPT",
            "A model file arrived damaged and was deleted. Try the download again.",
            file=item.name,
        )
    os.replace(partial, final)
    return None


def download(
    spec: ModelSpec,
    *,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    root: Path | None = None,
    opener: Opener | None = None,
) -> Result[Path]:
    """Fetch *spec*'s files into ``<root>/<id>/``; *progress(done, total)* in bytes."""
    if not spec.published:
        return Result.failure(
            Diagnostic.error("MODEL_UNPUBLISHED", f"{spec.display_name} has not been published yet.", model=spec.id)
        )
    folder = (root or user_models_dir()) / spec.id
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Result.failure(
            Diagnostic.error("MODEL_DOWNLOAD_DISK", f"Can't write to the models folder ({exc}).", folder=str(folder))
        )
    needed = free_space_needed(spec, folder.parent)
    free = shutil.disk_usage(folder).free
    if needed and free < needed * _HEADROOM:
        return Result.failure(
            Diagnostic.error(
                "MODEL_DOWNLOAD_DISK",
                f"{spec.display_name} needs {needed * _HEADROOM / 1e9:.1f} GB free; this disk has {free / 1e9:.1f} GB.",
                needed=needed,
                free=free,
            )
        )
    total = spec.size_bytes
    done = 0

    def advance(count: int) -> None:
        nonlocal done
        done += count
        if progress is not None:
            progress(done, total)

    fetch = opener or urllib.request.urlopen
    for item in spec.files:
        final = folder / item.name
        if final.is_file() and final.stat().st_size == item.size and _sha256(final) == item.sha256:
            advance(item.size)
            continue
        final.unlink(missing_ok=True)
        problem = _fetch(release_url(spec, item.name), item, folder, opener=fetch, cancel=cancel, advance=advance)
        if problem is not None:
            return Result.failure(problem)
    return Result.success(folder)


def delete(spec: ModelSpec, *, root: Path | None = None) -> Result[int]:
    """Remove a downloaded model; the bytes freed. Bundled models stay."""
    folder = (root or user_models_dir()) / spec.id
    if not folder.is_dir():
        return Result.success(0)
    freed = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    from core.models import onnx_backend

    # Drop cached sessions first: an open model file cannot be removed on Windows.
    onnx_backend.forget(folder)
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        return Result.failure(
            Diagnostic.error(
                "MODEL_DELETE_FAILED",
                f"Couldn't remove {spec.display_name}: a file is in use. Close any running analysis and try again.",
                reason=str(exc),
            )
        )
    return Result.success(freed)
