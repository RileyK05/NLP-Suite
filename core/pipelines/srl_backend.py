"""SRL backend — semantic roles via an isolated Python 3.8 worker env.

The worker is outside this suite (FR-5.3): ``NLP_SUITE_SRL_PYTHON`` must
point at a Python whose environment provides ``transformer_srl`` (the
legacy pinned stack from ``src/SRL_worker.py``). Every entry point
probes the worker first; a missing or broken worker fails with the setup
pointer instead of an import traceback. Full SRL frame mapping over the
worker protocol is the documented remainder of FR-5.3.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pandas as pd

from core.io.reader import Corpus
from core.result import Diagnostic, Result

__all__ = ["SRLPipeline", "build_srl_pipeline", "probe_worker", "worker_python"]

_WORKER_ENV_VAR = "NLP_SUITE_SRL_PYTHON"
_WORKER_TIMEOUT = 60.0


class SRLPipeline:
    backend = "srl"

    def __init__(self, language: str, tasks: frozenset[str]) -> None:
        self.language = language
        self.tasks = tasks

    def supports(self, language: str, tasks: object = ()) -> bool:
        return language == "en"

    def parse(self, corpus: Corpus) -> Result[pd.DataFrame]:
        _ = corpus
        probed = probe_worker()
        if probed.value is None:
            return Result.failure(*probed.diagnostics)
        return Result.failure(
            Diagnostic.error(
                "SRL_FRAMES_UNMAPPED",
                "SRL worker is reachable, but full SRL frame mapping over the worker protocol is not implemented yet (FR-5.3 remainder)",
                backend="srl",
            )
        )


def build_srl_pipeline(language: str, tasks: frozenset[str]) -> Result[SRLPipeline]:
    if language != "en":
        return Result.failure(
            Diagnostic.error(
                "PIPELINE_UNSUPPORTED", f"SRL supports only English, got {language!r}", backend="srl", language=language
            )
        )
    return Result.success(SRLPipeline(language=language, tasks=tasks))


def worker_python() -> str | None:
    """Configured worker interpreter, or None when unset/unusable."""
    candidate = os.environ.get(_WORKER_ENV_VAR, "").strip()
    if not candidate:
        return None
    resolved = shutil.which(candidate) or candidate
    if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
        return resolved
    return None


def probe_worker(python: str | None = None) -> Result[dict[str, object]]:
    """Prove the worker interpreter imports ``transformer_srl``."""
    worker = python or worker_python()
    if worker is None:
        return Result.failure(
            Diagnostic.error(
                "SRL_WORKER_MISSING",
                f"no SRL worker configured (set {_WORKER_ENV_VAR} to a Python with transformer_srl); "
                "SRL runs in an isolated Python 3.8 env per the legacy SRL_worker header",
                backend="srl",
                fix=f"set {_WORKER_ENV_VAR}=<worker python> (isolated 3.8 env with transformer_srl)",
            )
        )
    try:
        completed = subprocess.run(  # noqa: S603
            [worker, "-c", "import transformer_srl"],
            capture_output=True,
            text=True,
            timeout=_WORKER_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Result.failure(
            Diagnostic.error("SRL_WORKER_BROKEN", f"SRL worker {worker!r} would not start: {exc}", backend="srl")
        )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        hint = tail[-1] if tail else f"exit {completed.returncode}"
        return Result.failure(
            Diagnostic.error(
                "SRL_WORKER_BROKEN",
                f"SRL worker {worker!r} has no transformer_srl ({hint})",
                backend="srl",
            )
        )
    return Result.success({"worker": worker})
