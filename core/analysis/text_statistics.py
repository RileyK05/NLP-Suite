"""Text-level statistics derived from a CoNLL frame."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["TextStatisticsResult", "run"]

_VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _syllables(word: str) -> int:
    """Naive English syllable estimator — good enough for length bands.

    Counts vowel groups, adjusts for silent trailing e. The legacy used
    ``textstat.syllable_count`` which required an extra dependency; this
    heuristic tracks it within ±1 for the vast majority of words and keeps
    the core dependency-free.
    """
    w = word.lower().strip()
    if not w or not w.isalpha():
        return 0
    groups = len(_VOWEL_RE.findall(w))
    if w.endswith("e") and groups > 1:
        groups -= 1
    return max(groups, 1)


@dataclass(frozen=True, slots=True)
class TextStatisticsResult:
    frame: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def run(frame: pd.DataFrame) -> Result[TextStatisticsResult]:
    """Per-document sentence/word/syllable summary.

    The legacy ``statistics_txt_util.compute_corpus_statistics`` needed the
    raw text and a Stanza pass; this engine derives the same table from the
    CoNLL parse the suite already produced (one token row each, grouped by
    Sentence ID). Documents with no alphabetic tokens still emit a row so
    down-stream joins stay aligned.
    """
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[TextStatisticsResult](None, checked.diagnostics)

    required = [Col.FORM.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return Result.failure(
            Diagnostic.error("TEXT_STATS_MISSING_COLUMN", f"missing column(s): {missing}", missing=missing),
        )

    if frame.empty:
        empty = pd.DataFrame(
            columns=[
                "Document ID",
                "Document",
                "Sentences",
                "Tokens",
                "Types",
                "Avg Sentence Length",
                "Avg Word Length",
                "Syllables",
                "Avg Syllables per Word",
            ],
        )
        return Result.success(TextStatisticsResult(frame=empty))

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None

    rows: list[dict[str, object]] = []
    for doc_id, g in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(g[doc_col].iloc[0]) if doc_col is not None else ""
        sentences = g[Col.SENTENCE_ID.value].unique()
        n_sent = len(sentences)

        tokens = [str(x) for x in g[Col.FORM.value].tolist()]
        # Word membership = alphabetic, as in corpus_statistics.
        word_tokens = [t for t in tokens if t.isalpha()]
        n_tokens = len(word_tokens)
        n_types = len({t.lower() for t in word_tokens})
        avg_sent_len = round(n_tokens / n_sent, 2) if n_sent else 0.0
        avg_word_len = round(sum(len(t) for t in word_tokens) / n_tokens, 2) if n_tokens else 0.0
        n_syll = sum(_syllables(t) for t in word_tokens)
        avg_syll = round(n_syll / n_tokens, 2) if n_tokens else 0.0

        rows.append(
            {
                "Document ID": str(doc_id),
                "Document": doc,
                "Sentences": n_sent,
                "Tokens": n_tokens,
                "Types": n_types,
                "Avg Sentence Length": avg_sent_len,
                "Avg Word Length": avg_word_len,
                "Syllables": n_syll,
                "Avg Syllables per Word": avg_syll,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Document ID",
            "Document",
            "Sentences",
            "Tokens",
            "Types",
            "Avg Sentence Length",
            "Avg Word Length",
            "Syllables",
            "Avg Syllables per Word",
        ],
    )
    return Result.success(TextStatisticsResult(frame=df))
