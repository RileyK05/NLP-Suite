"""Corpus-level lexical diversity and related statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["CorpusStatisticsResult", "run"]


def _ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def _root_ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / math.sqrt(len(tokens)) if tokens else 0.0


def _log_ttr(tokens: list[str]) -> float:
    if len(tokens) < 2 or len(set(tokens)) <= 1:
        return 0.0
    return math.log(len(set(tokens))) / math.log(len(tokens))


def _yule_k(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    from collections import Counter

    freqs = Counter(tokens).values()
    m1 = float(len(tokens))
    m2 = sum(f * f for f in freqs)
    if m2 <= m1:
        return 0.0
    return 10000.0 * (m2 - m1) / (m1 * m1)


def _herdan(tokens: list[str]) -> float:
    # Alias for LogTTR in this suite — kept as a separate column for compatibility.
    return _log_ttr(tokens)


@dataclass(frozen=True, slots=True)
class CorpusStatisticsResult:
    frame: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def run(
    frame: pd.DataFrame,
    *,
    field: Col = Col.FORM,
) -> Result[CorpusStatisticsResult]:
    """Per-document lexical diversity.

    The legacy ``statistics_corpus_*`` guis ran four separate passes
    (TTR, Guiraud, Herdan, MTLD, vocd-D). The new engine runs one grouped
    pass and returns the four stable measures that do not require a sampling
    loop (TTR, RootTTR, LogTTR/Herdan, Yule K).
    """
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(
            Diagnostic.error(
                "CORPUS_STATS_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}", field=field.value
            ),
        )

    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[CorpusStatisticsResult](None, checked.diagnostics)

    required = [field.value, Col.DOCUMENT_ID.value]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return Result.failure(
            Diagnostic.error("CORPUS_STATS_MISSING_COLUMN", f"missing column(s): {missing}", missing=missing),
        )

    if frame.empty:
        empty = pd.DataFrame(
            columns=[
                "Document ID",
                "Document",
                "Tokens",
                "Types",
                "TTR",
                "Root TTR",
                "Log TTR",
                "Herdan",
                "Yule K",
            ],
        )
        return Result.success(CorpusStatisticsResult(frame=empty))

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None

    rows: list[dict[str, object]] = []
    for doc_id, g in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        tokens = [str(x).lower() for x in g[field.value].tolist() if str(x).strip() and str(x) != "nan"]
        # Punctuation tokens are not words — exclude one-char punctuation as the legacy did.
        tokens = [t for t in tokens if t.isalpha()]

        doc = str(g[doc_col].iloc[0]) if doc_col is not None else ""
        rows.append(
            {
                "Document ID": str(doc_id),
                "Document": doc,
                "Tokens": len(tokens),
                "Types": len(set(tokens)),
                "TTR": round(_ttr(tokens), 4),
                "Root TTR": round(_root_ttr(tokens), 4),
                "Log TTR": round(_log_ttr(tokens), 4),
                "Herdan": round(_herdan(tokens), 4),
                "Yule K": round(_yule_k(tokens), 2),
            }
        )

    df = pd.DataFrame(
        rows, columns=["Document ID", "Document", "Tokens", "Types", "TTR", "Root TTR", "Log TTR", "Herdan", "Yule K"]
    )
    return Result.success(CorpusStatisticsResult(frame=df))
