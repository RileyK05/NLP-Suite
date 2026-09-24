"""Topic models — Gensim LDA over CoNLL lemmas (FR-5.7).

Thin frame adapter over :mod:`core.analysis.lda`: per-document token lists
come from the canonical CoNLL table, the model itself is seeded Gensim LDA.
The old TF-IDF round-robin stand-in is gone; ``run`` returns the real
per-topic top-word frame. Callers that also need the dominant-topic table
(use :func:`core.analysis.lda.fit_lda`) — the CLI writes both artifacts.
"""

from __future__ import annotations

import pandas as pd

from core.analysis.lda import fit_lda, tokens_from_frame
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["run"]


def run(
    frame: pd.DataFrame,
    *,
    n_topics: int = 3,
    top_n: int = 5,
    field: Col = Col.LEMMA,
    seed: int = 100,
    remove_stopwords: bool = True,
    nouns_only: bool = False,
) -> Result[pd.DataFrame]:
    """Per-topic top words from seeded LDA (Topic 0-based, Word, Weight)."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("TOPIC_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if field.value not in frame.columns or Col.DOCUMENT_ID.value not in frame.columns:
        return Result.failure(Diagnostic.error("TOPIC_MISSING_COLUMN", "missing field or Document ID"))
    if frame.empty:
        return Result.success(
            pd.DataFrame(columns=["Topic", "Word", "Weight"]),
            Diagnostic.warning("TOPIC_NO_DOCUMENTS", "empty input frame; no topics to model"),
        )
    doc_tokens = tokens_from_frame(frame, field=field, nouns_only=nouns_only, remove_stopwords=remove_stopwords)
    if not doc_tokens:
        return Result.success(
            pd.DataFrame(columns=["Topic", "Word", "Weight"]),
            Diagnostic.warning("TOPIC_NO_DOCUMENTS", "no usable tokens; no topics to model"),
        )
    fitted = fit_lda(doc_tokens, n_topics=n_topics, top_n=top_n, seed=seed, remove_stopwords=False, coherence=False)
    if fitted.value is None:
        return Result.failure(*fitted.diagnostics)
    return Result.success(fitted.unwrap().topics, *fitted.diagnostics)
