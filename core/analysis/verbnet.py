"""VerbNet class aggregation (FR-4.5).

Ports legacy ``aggregate_VerbNet``: each verb lemma maps to its first
VerbNet class (``vn.classids`` order — the first/most-basic class per
lemma), unknown verbs become ``Not found`` rows, plus a frequency table
over the found classes. VerbNet classifies verbs only; the input is a
verb list by contract.

The NLTK backend is lazy (optional ``wordnet`` extra — VerbNet ships
with NLTK, not with this suite); tests inject a hand-built fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "NltkVerbNet",
    "VerbNetBackend",
    "aggregate_verbs",
    "category_counts",
    "default_backend",
]

_COLUMNS: tuple[str, str] = ("Word", "VerbNet Class")
_COUNT_COLUMNS: tuple[str, str] = ("VerbNet Class", "Frequency")


class VerbNetBackend(Protocol):
    """Minimal VerbNet surface the aggregation needs."""

    def class_ids(self, lemma: str) -> list[str]: ...


class NltkVerbNet:
    """Production backend: NLTK VerbNet 3.x (lazy, offline)."""

    def __init__(self, verbnet: Any) -> None:
        self._vn = verbnet

    def class_ids(self, lemma: str) -> list[str]:
        return [str(class_id) for class_id in self._vn.classids(lemma=lemma)]


def default_backend(data_dir: str | Path | None = None) -> Result[NltkVerbNet]:
    """Build the NLTK VerbNet backend, or fail with the exact install command."""
    try:
        import nltk
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "VERBNET_BACKEND_MISSING",
                "NLTK is not installed; VerbNet aggregation needs the optional 'wordnet' extra",
                fix="python -m pip install nlp-suite-ng[wordnet]",
            )
        )
    try:
        if data_dir is not None:
            nltk.data.path.insert(0, str(data_dir))
        from nltk.corpus import verbnet as vn

        vn.classids(lemma="run")  # a real query: proves the corpus loads
    except LookupError:
        target = str(Path(data_dir) / "corpora" / "verbnet") if data_dir is not None else "<NLTK_DATA>/corpora/verbnet"
        return Result.failure(
            Diagnostic.error(
                "VERBNET_DATA_MISSING",
                f"VerbNet corpus not found at {target}",
                fix="python -m nltk.downloader -d <NLTK_DATA> verbnet",
            )
        )
    return Result.success(NltkVerbNet(vn))


def aggregate_verbs(
    words: Sequence[str | float | None],
    *,
    backend: VerbNetBackend | None = None,
) -> Result[pd.DataFrame]:
    """Map verb lemmas to their first VerbNet class, in input order."""
    if backend is None:
        resolved = default_backend()
        if resolved.value is None:
            return Result.failure(*resolved.diagnostics)
        backend = resolved.unwrap()
        diags: list[Diagnostic] = list(resolved.diagnostics)
    else:
        diags = []
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
            class_ids = backend.class_ids(cleaned)
        except Exception as exc:
            return Result.failure(
                Diagnostic.error("VERBNET_LOOKUP_FAILED", f"VerbNet lookup failed for {cleaned!r}: {exc}", word=cleaned)
            )
        rows.append({"Word": cleaned, "VerbNet Class": class_ids[0] if class_ids else "Not found"})
    frame = pd.DataFrame(rows, columns=list(_COLUMNS))
    if not rows:
        diags.append(Diagnostic.warning("VERBNET_EMPTY_INPUT", "no words to aggregate"))
    elif all(row["VerbNet Class"] == "Not found" for row in rows):
        diags.append(
            Diagnostic.warning(
                "VERBNET_ALL_NOT_FOUND", f"VerbNet found none of the {len(rows)} word(s); all rows are 'Not found'"
            )
        )
    return Result.success(frame, *diags)


def category_counts(aggregated: pd.DataFrame) -> Result[pd.DataFrame]:
    """Frequency table over an aggregation frame, highest first (``Not found`` excluded)."""
    if "VerbNet Class" not in aggregated.columns:
        return Result.failure(Diagnostic.error("VERBNET_BAD_FRAME", "aggregation frame needs a 'VerbNet Class' column"))
    found = aggregated.loc[aggregated["VerbNet Class"] != "Not found", "VerbNet Class"].astype(str)
    tallies = found.value_counts()
    frame = pd.DataFrame(
        [{"VerbNet Class": class_id, "Frequency": int(count)} for class_id, count in tallies.items()],
        columns=list(_COUNT_COLUMNS),
    )
    return Result.success(frame)
