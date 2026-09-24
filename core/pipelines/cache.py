"""Pipeline cache — one model per (backend, language, tasks).

ARCHITECTURE.md R5: the legacy suite instantiated a parser per word or per
sentence — up to 100,000 instantiations per corpus. Here exactly one live
pipeline exists for each distinct key.

The cache owns no global state; create one and pass it where it is needed.
A module-level :data:`default_cache` is provided for the common case.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import Lock

from core.config import NLPConfig
from core.pipelines.protocol import Pipeline, supports
from core.result import Diagnostic, Result

__all__ = ["PipelineCache", "default_cache"]


Builder = Callable[[str, frozenset[str]], Result[Pipeline]]


class PipelineCache:
    """Thread-safe cache keyed by ``(backend, language, tasks)``."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str, frozenset[str]], Pipeline] = {}
        self._builders: dict[str, Builder] = {}
        self._lock = Lock()

    def register(self, backend: str, builder: Builder) -> None:
        """Register a builder for *backend*.

        The builder is called as ``builder(language, tasks)`` and must return
        ``Result[Pipeline]`` without touching global state.
        """
        self._builders[backend] = builder

    def has_builder(self, backend: str) -> bool:
        """Whether a builder is already registered for *backend*.

        Lets a caller add the standard builders without displacing one that
        was deliberately supplied -- a fake in a test, or a preconfigured
        pipeline in an embedding application.
        """
        return backend in self._builders

    def get(
        self,
        backend: str,
        language: str,
        tasks: Iterable[str] = (),
        *,
        config: NLPConfig | None = None,
    ) -> Result[Pipeline]:
        """Return the cached pipeline for the key, building it if necessary."""
        task_set = frozenset(str(t) for t in tasks)
        key = (backend, language, task_set)

        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                return Result.success(cached)

        if config is not None and (config.parser != backend or config.language != language):
            return Result.failure(
                Diagnostic.error(
                    "PIPELINE_CONFIG_MISMATCH",
                    f"requested ({backend}, {language}) does not match config ({config.parser}, {config.language})",
                    backend=backend,
                    language=language,
                )
            )

        if not supports(backend, language, task_set):
            return Result.failure(
                Diagnostic.error(
                    "PIPELINE_UNSUPPORTED",
                    f"backend {backend!r} has no confirmed model for language {language!r}",
                    backend=backend,
                    language=language,
                )
            )

        builder = self._builders.get(backend)
        if builder is None:
            return Result.failure(
                Diagnostic.error(
                    "PIPELINE_NO_BUILDER",
                    f"no builder registered for backend {backend!r}",
                    backend=backend,
                )
            )

        built = builder(language, task_set)
        if built.value is None:
            return built

        pipeline = built.unwrap()
        with self._lock:
            existing = self._store.get(key)
            if existing is not None:
                return Result.success(existing)
            self._store[key] = pipeline
        return Result.success(pipeline, *built.diagnostics)

    def clear(self) -> None:
        """Empty the cache. Primarily for tests."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def contains(self, backend: str, language: str, tasks: Iterable[str] = ()) -> bool:
        key = (backend, language, frozenset(str(t) for t in tasks))
        with self._lock:
            return key in self._store


default_cache = PipelineCache()
