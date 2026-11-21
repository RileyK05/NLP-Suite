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

import numpy as np
import pandas as pd

from core.analysis.contextual import EmbeddingBackend, embed_sentences, text_backend, truncation_note
from core.conll.frames import col_str_list, sentence_indices
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["summarize"]

_COLUMNS = ["Document ID", "Document", "Summary", "Summary Sentences", "Sentence IDs"]


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
    forms = col_str_list(frame, Col.FORM)
    raw_doc_ids = frame[Col.DOCUMENT_ID.value].tolist()
    raw_sent_ids = frame[Col.SENTENCE_ID.value].tolist()
    doc_names = col_str_list(frame, doc_col) if doc_col is not None else [""] * len(frame)

    doc_entries: list[tuple[object, str, list[object], list[str]]] = []
    by_doc: dict[object, int] = {}
    all_texts: list[str] = []
    doc_slices: list[tuple[int, int]] = []

    for sent_rows in sentence_indices(frame).values():
        first_idx = sent_rows[0]
        did = raw_doc_ids[first_idx]
        sid = raw_sent_ids[first_idx]
        text = " ".join(forms[i] for i in sent_rows).strip()
        if not text:
            continue
        pos = by_doc.get(did)
        if pos is None:
            pos = len(doc_entries)
            by_doc[did] = pos
            doc_entries.append((did, doc_names[first_idx] if doc_col is not None else "", [], []))
        doc_entries[pos][2].append(sid)
        doc_entries[pos][3].append(text)

    if not doc_entries:
        diags.extend(truncation_note(resolved))
        return Result.success(pd.DataFrame(columns=_COLUMNS), *diags)

    for _, _, _, texts in doc_entries:
        start = len(all_texts)
        all_texts.extend(texts)
        doc_slices.append((start, len(all_texts)))

    try:
        all_vectors = np.asarray(embed_sentences(resolved, all_texts), dtype=float)
    except (RuntimeError, ValueError) as exc:
        return Result.failure(Diagnostic.error("BERT_MODEL_MISSING", str(exc)))

    norms = np.linalg.norm(all_vectors, axis=1)
    norms[norms == 0] = 1.0

    rows: list[dict[str, object]] = []
    for (did, doc, sent_ids, texts), (start, stop) in zip(doc_entries, doc_slices, strict=True):
        vecs = all_vectors[start:stop]
        vec_norms = norms[start:stop]
        centroid = vecs.mean(axis=0)
        c_norm = float(np.linalg.norm(centroid)) or 1.0
        scores = (vecs @ centroid) / (vec_norms * c_norm)
        ranked = np.argsort(-scores, kind="stable").tolist()
        picked = sorted(ranked[:sentences])
        rows.append(
            {
                "Document ID": did,
                "Document": doc,
                "Summary": " ".join(texts[i] for i in picked),
                "Summary Sentences": len(picked),
                "Sentence IDs": ",".join(str(sent_ids[i]) for i in picked),
            }
        )
    diags.extend(truncation_note(resolved))
    return Result.success(pd.DataFrame(rows, columns=_COLUMNS), *diags)
