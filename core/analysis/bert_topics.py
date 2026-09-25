"""BERT topics (CAP-TOPIC-04) — KMeans over sentence/document embeddings.

Legacy reference: ``topic_modeling_bert_util.py`` — an abandoned notebook in
the legacy suite. Ours is the deliberate completion: embed each document
(one vector per document over its sentence-mean token vectors) through the
injectable :class:`EmbeddingBackend` protocol (the same one as
``contextual``/``bert_extract``), then seeded KMeans groups the documents
and each cluster's top discriminative terms become the topic label. The
term score is a rate contrast, not TF-IDF: a term's rate inside the cluster
times its margin over its rate in the other clusters
(``own_rate * (own_rate - other_rate + 1)``), so a word that is both
frequent in the cluster and rarer outside it outranks one that is merely
frequent everywhere. Ties break alphabetically, so the label is stable.

Like LDA (CAP-TOPIC-02) this is a randomized algorithm: verification is
structural — same seed twice is byte-identical, clusters partition, k
honored — and the *content* of topics depends on the backend model. Tests
inject the deterministic hash backend; real use pulls
``bert-base-uncased`` on first embed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from core.analysis.contextual import EmbeddingBackend, embed_sentences, text_backend, truncation_note
from core.analysis.lda import STOPWORDS
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["bert_topics"]

_TOPIC_COLUMNS = ["Topic", "Word", "Weight"]
_DOCUMENT_COLUMNS = ["Document ID", "Document", "Topic", "Distance"]


def bert_topics(
    frame: pd.DataFrame,
    *,
    n_topics: int = 3,
    top_n: int = 5,
    seed: int = 42,
    backend: EmbeddingBackend | None = None,
    model: str | None = None,
) -> Result[tuple[pd.DataFrame, pd.DataFrame]]:
    """KMeans topics over document embeddings, labeled by discriminative terms.

    Returns ``(topics, documents)``: per-topic top terms with their rate-contrast
    weight (see the module docstring -- not TF-IDF), and per-document cluster
    assignments with distance to the assigned centroid.
    """
    if isinstance(n_topics, bool) or not isinstance(n_topics, int) or n_topics < 2:
        return Result.failure(Diagnostic.error("BERTOPIC_BAD_K", f"n_topics must be >= 2, got {n_topics!r}"))
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        return Result.failure(Diagnostic.error("BERTOPIC_BAD_TOPN", f"top_n must be >= 1, got {top_n!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[tuple[pd.DataFrame, pd.DataFrame]](None, checked.diagnostics)
    for need in (Col.FORM.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value):
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("BERTOPIC_MISSING_COLUMN", f"missing {need!r}"))
    if frame.empty:
        return Result.failure(Diagnostic.error("BERTOPIC_EMPTY", "frame is empty"))

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    names: list[str] = []
    doc_ids: list[str] = []
    doc_tokens: list[list[str]] = []
    for did, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc_ids.append(str(did))
        names.append(str(group[doc_col].iloc[0]) if doc_col is not None else f"doc-{did}")
        toks = [str(v).lower() for v in group[Col.FORM.value].tolist() if str(v).strip() and str(v).isalpha()]
        doc_tokens.append([t for t in toks if t not in STOPWORDS])
    n_docs = len(doc_ids)
    if n_docs < n_topics:
        return Result.failure(
            Diagnostic.error(
                "BERTOPIC_TOO_SMALL",
                f"{n_topics} topics need at least {n_topics} documents, corpus has {n_docs}",
            )
        )

    resolved = backend
    diags: list[Diagnostic] = []
    if resolved is None:
        decided = text_backend(model or "bert-base-uncased")
        if decided.value is None:
            return Result.failure(*decided.diagnostics)
        resolved = decided.unwrap()
        diags = list(decided.diagnostics)

    # One embedding per document: mean over the document's sentences, each
    # sentence embedded as (its tokens, its own text) like bert_extract does.
    doc_vectors: list[np.ndarray] = []
    for doc_id, group in zip(doc_ids, frame.groupby(Col.DOCUMENT_ID.value, sort=False), strict=True):
        _did, g = group
        sentences: list[str] = []
        for _, sent_group in g.groupby(Col.SENTENCE_ID.value, sort=False):
            text = " ".join(str(v) for v in sent_group[Col.FORM.value].tolist()).strip()
            if text:
                sentences.append(text)
        if not sentences:
            doc_vectors.append(np.zeros(resolved.dimension(), dtype=float))
            continue
        try:
            vectors = np.array(embed_sentences(resolved, sentences), dtype=np.float64)
        except (RuntimeError, ValueError) as exc:
            return Result.failure(Diagnostic.error("BERTOPIC_MODEL_MISSING", str(exc)))
        doc_vectors.append(vectors.mean(axis=0))
    matrix = np.array(doc_vectors, dtype=np.float64)

    km = KMeans(n_clusters=n_topics, random_state=seed, n_init=10)
    labels = km.fit_predict(matrix)

    # Discriminative terms: per-cluster token rates vs the other clusters.
    # Deterministic ordering (score desc, then term).
    cluster_docs: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        cluster_docs.setdefault(int(label), []).append(index)
    cluster_counters: dict[int, Counter[str]] = {}
    for label, members in cluster_docs.items():
        counter: Counter[str] = Counter()
        for i in members:
            counter.update(t for t in doc_tokens[i] if t.isalpha())
        cluster_counters[label] = counter
    all_terms = sorted(set().union(*(set(c) for c in cluster_counters.values())) if cluster_counters else set())

    topic_rows: list[dict[str, object]] = []
    for label in range(n_topics):
        own = cluster_counters.get(label, Counter())
        others: Counter[str] = Counter()
        for other_label, counter in cluster_counters.items():
            if other_label != label:
                others.update(counter)
        n_own_tokens = sum(own.values()) or 1
        n_other_tokens = sum(others.values()) or 1
        scored: list[tuple[float, str]] = []
        for term in all_terms:
            own_rate = own[term] / n_own_tokens
            other_rate = others[term] / n_other_tokens
            score = own_rate * (own_rate - other_rate + 1.0)
            if own[term] > 0:
                scored.append((score, term))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        for score, term in scored[:top_n]:
            topic_rows.append({"Topic": label, "Word": term, "Weight": round(float(score), 6)})
    topics = pd.DataFrame(topic_rows, columns=_TOPIC_COLUMNS)

    doc_rows: list[dict[str, object]] = []
    for index, (did, name, label) in enumerate(zip(doc_ids, names, labels.tolist(), strict=True)):
        doc_rows.append(
            {
                "Document ID": did,
                "Document": name,
                "Topic": int(label),
                "Distance": round(float(np.linalg.norm(matrix[index] - km.cluster_centers_[label])), 4),
            }
        )
    documents = pd.DataFrame(doc_rows, columns=_DOCUMENT_COLUMNS)
    diags.extend(truncation_note(resolved))
    return Result.success((topics, documents), *diags)
