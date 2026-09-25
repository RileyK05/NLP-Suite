"""BERT extractive summarization (FR-5.9).

Ports legacy ``doc_summary_BERT`` (which shelled the unpinned
``bert-extractive-summarizer`` package) as centroid extractive
summarization over the injectable ``EmbeddingBackend`` (the same
protocol as ``core.analysis.contextual``): embed each sentence, rank by
cosine to the document centroid, keep the top-k in original order.

The ranking method differs from the legacy package on purpose — the
replacement bar is a deterministic, dependency-honest summary, not a
bug-for-bug port of an unpinned black box. The default backend is the
real transformer (model downloads on first use); tests inject the hash
double from ``conftest``.
"""

from __future__ import annotations

import math

import pandas as pd

from core.analysis.contextual import EmbeddingBackend, embed_sentences, text_backend, truncation_note
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["summarize"]

_COLUMNS = ["Document ID", "Document", "Summary", "Summary Sentences", "Sentence IDs"]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


def summarize(
    frame: pd.DataFrame,
    *,
    sentences: int = 3,
    backend: EmbeddingBackend | None = None,
    model: str | None = None,
) -> Result[pd.DataFrame]:
    """Top-``sentences`` centroid-ranked sentences per document, in order."""
    if isinstance(sentences, bool) or not isinstance(sentences, int) or sentences < 1:
        return Result.failure(
            Diagnostic.error("BERT_BAD_COUNT", f"sentences must be a positive int, got {sentences!r}")
        )
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    for need in (Col.FORM.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value):
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("BERT_MISSING_COLUMN", f"parse table needs column {need!r}"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_COLUMNS))

    resolved = backend
    diags: list[Diagnostic] = []
    if resolved is None:
        decided = text_backend(model or "bert-base-uncased")
        if decided.value is None:
            return Result.failure(*decided.diagnostics)
        resolved = decided.unwrap()
        diags = list(decided.diagnostics)

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        sent_ids: list[object] = []
        texts: list[str] = []
        for sent_id, sent_group in group.groupby(Col.SENTENCE_ID.value, sort=False):
            text = " ".join(str(v) for v in sent_group[Col.FORM.value].tolist()).strip()
            if text:
                sent_ids.append(sent_id)
                texts.append(text)
        if not texts:
            continue
        try:
            vectors = embed_sentences(resolved, texts)
        except (RuntimeError, ValueError) as exc:
            return Result.failure(Diagnostic.error("BERT_MODEL_MISSING", str(exc)))
        dim = len(vectors[0])
        centroid = [sum(vec[i] for vec in vectors) / len(vectors) for i in range(dim)]
        ranked = sorted(range(len(texts)), key=lambda i: _cosine(vectors[i], centroid), reverse=True)
        picked = sorted(ranked[:sentences])
        rows.append(
            {
                "Document ID": doc_id,
                "Document": doc,
                "Summary": " ".join(texts[i] for i in picked),
                "Summary Sentences": len(picked),
                "Sentence IDs": ",".join(str(sent_ids[i]) for i in picked),
            }
        )
    diags.extend(truncation_note(resolved))
    return Result.success(pd.DataFrame(rows, columns=_COLUMNS), *diags)
