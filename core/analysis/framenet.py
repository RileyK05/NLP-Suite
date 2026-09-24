"""FrameNet frame aggregation (FR-4.5).

Ports legacy ``aggregate_FrameNet``: each noun/verb lemma maps to its
first FrameNet frame (alphabetically first of the frames whose lexical
unit is ``lemma.pos``), unknown lemmas become ``Not found`` rows, plus a
frequency table over the found frames.

The per-process lemma index is built once per backend instance (one pass
over all frames — the legacy's per-lemma ``fn.lus`` regex scan is the
documented performance defect this intentionally does not reproduce).
The NLTK backend is lazy (optional ``wordnet`` extra); tests inject a
hand-built fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "NltkFrameNet",
    "FrameNetBackend",
    "aggregate",
    "category_counts",
    "default_backend",
]

_COLUMNS: tuple[str, str] = ("Word", "FrameNet Frame")
_COUNT_COLUMNS: tuple[str, str] = ("FrameNet Frame", "Frequency")
_POS_VALUES: tuple[str, str] = ("NOUN", "VERB")


class FrameNetBackend(Protocol):
    """Minimal FrameNet surface the aggregation needs."""

    def frames_for(self, lemma: str, pos: str) -> list[str]:
        """Sorted frame names for a lemma + ``n``/``v`` POS, possibly empty."""


class NltkFrameNet:
    """Production backend: NLTK FrameNet 1.7 with a cached lemma index."""

    def __init__(self, framenet: Any) -> None:
        self._fn = framenet
        self._index: dict[tuple[str, str], list[str]] | None = None

    def _build_index(self) -> dict[tuple[str, str], list[str]]:
        index: dict[tuple[str, str], set[str]] = {}
        for frame in self._fn.frames():
            name = str(frame.name)
            for lu_name in frame.lexUnit:
                lemma, _, pos = str(lu_name).rpartition(".")
                if pos:
                    index.setdefault((lemma.lower(), pos.lower()), set()).add(name)
        return {key: sorted(names) for key, names in index.items()}

    def frames_for(self, lemma: str, pos: str) -> list[str]:
        if self._index is None:
            self._index = self._build_index()
        return list(self._index.get((lemma.lower(), pos.lower()), []))


def default_backend(data_dir: str | Path | None = None) -> Result[NltkFrameNet]:
    """Build the NLTK FrameNet backend, or fail with the exact install command."""
    try:
        import nltk
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "FRAMENET_BACKEND_MISSING",
                "NLTK is not installed; FrameNet aggregation needs the optional 'wordnet' extra",
                fix="python -m pip install nlp-suite-ng[wordnet]",
            )
        )
    try:
        if data_dir is not None:
            nltk.data.path.insert(0, str(data_dir))
        from nltk.corpus import framenet as fn

        len(fn.frames())  # a real query: proves the corpus loads
    except LookupError:
        target = (
            str(Path(data_dir) / "corpora" / "framenet_v17")
            if data_dir is not None
            else "<NLTK_DATA>/corpora/framenet_v17"
        )
        return Result.failure(
            Diagnostic.error(
                "FRAMENET_DATA_MISSING",
                f"FrameNet 1.7 corpus not found at {target}",
                fix="python -m nltk.downloader -d <NLTK_DATA> framenet_v17",
            )
        )
    return Result.success(NltkFrameNet(fn))


def aggregate(
    words: Sequence[str | float | None],
    *,
    pos: str = "NOUN",
    backend: FrameNetBackend | None = None,
) -> Result[pd.DataFrame]:
    """Map lemmas to their first FrameNet frame, in input order."""
    if pos not in _POS_VALUES:
        return Result.failure(Diagnostic.error("FRAMENET_BAD_POS", f"pos must be NOUN or VERB, got {pos!r}", pos=pos))
    if backend is None:
        resolved = default_backend()
        if resolved.value is None:
            return Result.failure(*resolved.diagnostics)
        backend = resolved.unwrap()
        diags: list[Diagnostic] = list(resolved.diagnostics)
    else:
        diags = []
    short_pos = "v" if pos == "VERB" else "n"
    rows: list[dict[str, object]] = []
    for raw in words:
        if raw is None:
            continue
        if not isinstance(raw, str):
            try:
                if bool(pd.isna(raw)):
                    continue
            except (TypeError, ValueError):
                pass
        cleaned = str(raw).strip().lower()
        if not cleaned:
            continue
        try:
            frames = backend.frames_for(cleaned, short_pos)
        except Exception as exc:
            return Result.failure(
                Diagnostic.error(
                    "FRAMENET_LOOKUP_FAILED", f"FrameNet lookup failed for {cleaned!r}: {exc}", word=cleaned
                )
            )
        rows.append({"Word": cleaned, "FrameNet Frame": frames[0] if frames else "Not found"})
    frame = pd.DataFrame(rows, columns=list(_COLUMNS))
    if not rows:
        diags.append(Diagnostic.warning("FRAMENET_EMPTY_INPUT", "no words to aggregate"))
    elif all(row["FrameNet Frame"] == "Not found" for row in rows):
        diags.append(
            Diagnostic.warning(
                "FRAMENET_ALL_NOT_FOUND", f"FrameNet found none of the {len(rows)} word(s); all rows are 'Not found'"
            )
        )
    return Result.success(frame, *diags)


def category_counts(aggregated: pd.DataFrame) -> Result[pd.DataFrame]:
    """Frequency table over an aggregation frame, highest first (``Not found`` excluded)."""
    if "FrameNet Frame" not in aggregated.columns:
        return Result.failure(
            Diagnostic.error("FRAMENET_BAD_FRAME", "aggregation frame needs a 'FrameNet Frame' column")
        )
    found = aggregated.loc[aggregated["FrameNet Frame"] != "Not found", "FrameNet Frame"].astype(str)
    tallies = found.value_counts()
    frame = pd.DataFrame(
        [{"FrameNet Frame": name, "Frequency": int(count)} for name, count in tallies.items()],
        columns=list(_COUNT_COLUMNS),
    )
    return Result.success(frame)
