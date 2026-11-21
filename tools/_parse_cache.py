"""Persistent CLI and Streamlit parse cache backed by parquet.

Shared across CLI invocations (via ``tools/_cli.py``) and Streamlit tools
so parsing a corpus once with spaCy or Stanza is reused whenever the exact
corpus documents, parser backend, model version, and tasks match. Opt out
with ``--no-parse-cache`` or ``NLP_NO_PARSE_CACHE=1``.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import uuid

import pandas as pd

from core.conll.schema import SCHEMA_VERSION
from core.hardware import held_corpora_limit
from core.io.reader import Corpus
from core.pipelines.protocol import Pipeline
from core.result import Diagnostic, Result, Severity

__all__ = ["CachedPipeline", "default_parse_cache_dir", "parse_with_cache"]

_SUPPORTED_BACKENDS = frozenset({"spacy", "stanza", "corenlp"})
_SEVERITY_FROM_VALUE = {s.value: s for s in Severity}
_MIN_FINGERPRINT_LEN = 16


def default_parse_cache_dir() -> Path:
    override = os.environ.get("NLP_PARSE_CACHE_DIR", "").strip()
    if override:
        return Path(override)
    system = sys.platform
    if system == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "NLP Suite" / "cache" / "parses"
    if system == "darwin":
        return Path.home() / "Library" / "Caches" / "NLP Suite" / "parses"
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "nlp-suite" / "parses"


def _installed_version(package: str) -> str:
    if not package:
        return "absent"
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def _cache_key(pipeline: Pipeline, corpus: Corpus) -> str:
    model = str(getattr(pipeline, "model_name", "") or "")
    payload = {
        "schema": SCHEMA_VERSION,
        "backend": pipeline.backend,
        "language": pipeline.language,
        "model": model,
        "backend_version": _installed_version(pipeline.backend),
        "model_version": _installed_version(model) if model else "absent",
        "tasks": sorted(str(t) for t in getattr(pipeline, "tasks", ())),
        "corpus_sha256": corpus.sha256,
        "docs": [[doc.doc_id, doc.path.name, doc.sha256, doc.label] for doc in corpus.docs],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def _serialize_diags(diagnostics: tuple[Diagnostic, ...]) -> str:
    return json.dumps([d.to_dict() for d in diagnostics])


def _deserialize_diags(raw: str) -> tuple[Diagnostic, ...]:
    items = json.loads(raw)
    out: list[Diagnostic] = []
    for item in items:
        sev = _SEVERITY_FROM_VALUE.get(str(item.get("severity", "INFO")), Severity.INFO)
        ctx = item.get("context")
        out.append(
            Diagnostic(
                code=str(item.get("code", "")),
                severity=sev,
                message=str(item.get("message", "")),
                context=dict(ctx) if isinstance(ctx, dict) else {},
            )
        )
    return tuple(out)


def _evict_if_over_budget(cache_dir: Path) -> None:
    budget_bytes = (256 if held_corpora_limit() <= 1 else 512) * 1024 * 1024
    try:
        entries = [p for p in cache_dir.glob("*.parquet") if p.is_file()]
    except OSError:
        return
    total = 0
    sized: list[tuple[float, int, Path]] = []
    for entry in entries:
        try:
            st = entry.stat()
        except OSError:
            continue
        total += st.st_size
        sized.append((st.st_mtime, st.st_size, entry))
    if total <= budget_bytes:
        return
    sized.sort(key=lambda item: item[0])
    for _, size, entry in sized:
        if total <= budget_bytes:
            break
        try:
            entry.unlink(missing_ok=True)
            entry.with_suffix(".diag.json").unlink(missing_ok=True)
            total -= size
        except OSError:
            continue


def parse_with_cache(
    pipeline: Pipeline,
    corpus: Corpus,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> Result[pd.DataFrame]:
    """Parse *corpus* with *pipeline*, reading/writing a parquet cache when eligible."""
    cache_enabled = (
        use_cache
        and not os.environ.get("NLP_NO_PARSE_CACHE")
        and pipeline.backend in _SUPPORTED_BACKENDS
        and bool(corpus.docs)
        and len(corpus.sha256) >= _MIN_FINGERPRINT_LEN
    )
    if not cache_enabled:
        return pipeline.parse(corpus)

    root = cache_dir or default_parse_cache_dir()
    key = _cache_key(pipeline, corpus)
    parquet_path = root / f"{key}.parquet"
    diag_path = root / f"{key}.diag.json"

    if parquet_path.is_file():
        with suppress(Exception):
            frame = pd.read_parquet(parquet_path)
            diags: tuple[Diagnostic, ...] = ()
            if diag_path.is_file():
                diags = _deserialize_diags(diag_path.read_text(encoding="utf-8"))
            return Result.success(frame, *diags)

    parsed = pipeline.parse(corpus)
    if parsed.value is not None and not parsed.has_errors and not parsed.value.empty:
        try:
            root.mkdir(parents=True, exist_ok=True)
            suffix = uuid.uuid4().hex[:8]
            tmp_parquet = root / f".{key}.{suffix}.partial"
            parsed.value.to_parquet(tmp_parquet, index=False)
            tmp_parquet.replace(parquet_path)
            if parsed.diagnostics:
                tmp_diag = root / f".{key}.{suffix}.diag.partial"
                tmp_diag.write_text(_serialize_diags(parsed.diagnostics), encoding="utf-8")
                tmp_diag.replace(diag_path)
            else:
                diag_path.unlink(missing_ok=True)
            _evict_if_over_budget(root)
        except OSError:
            pass
    return parsed


class CachedPipeline:
    """Transparent ``Pipeline`` proxy backed by :func:`parse_with_cache`."""

    def __init__(self, inner: Pipeline, *, cache_dir: Path | None = None) -> None:
        self._inner = inner
        self._cache_dir = cache_dir

    @property
    def backend(self) -> str:
        return self._inner.backend

    @property
    def language(self) -> str:
        return self._inner.language

    @property
    def tasks(self) -> frozenset[str]:
        return self._inner.tasks

    @property
    def model_name(self) -> str:
        return str(getattr(self._inner, "model_name", "") or "")

    def supports(self, language: str, tasks: Iterable[str] = ()) -> bool:
        return self._inner.supports(language, tasks)

    def tokenize(self, text: str) -> tuple[str, ...]:
        return self._inner.tokenize(text)

    def parse(self, corpus: Corpus) -> Result[pd.DataFrame]:
        return parse_with_cache(self._inner, corpus, cache_dir=self._cache_dir, use_cache=True)
