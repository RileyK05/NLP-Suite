"""Document and sentence embeddings: which texts mean alike, and a map of them.

A sentence model (Granite by default, Qwen from the Models page) reads the
corpus and gives one vector per text, trained so that texts meaning alike
lie close. From those vectors:

* ``doc_vectors`` -- one row per document (or per sentence), the vector as
  comma-joined numbers like every other vector table in the suite.
* ``doc_pairs`` -- every document pair's similarity, in the shape of the
  TF-IDF ``doc_similarity`` table so the same heatmap and neighbour figures
  read both and the two methods (shared words against shared meaning) can
  be compared side by side. ``Similarity`` is *relative*: the cosine, as a
  percentage, after removing the corpus's average document, so 0 is "as
  alike as a typical pair" (it can go negative). A sentence model reads
  every speech of one genre as about 98% alike, which is true and tells the
  speeches apart not at all; the raw cosine is kept in ``Cosine``.
* ``doc_neighbours`` -- each document's nearest few by meaning.
* ``doc_map`` -- a 2-D map (PCA, then t-SNE when there are enough texts)
  with k-means clusters, k chosen by silhouette over 2..10.
* ``search_results`` -- with a query, the sentences closest to it in
  meaning (semantic search; Qwen reads the query with its instruction).

A document is longer than any sentence model reads at once (512 pieces), so
it is read in passages of whole sentences up to ``_PASSAGE_WORDS`` words,
and its vector is the word-weighted mean of its passages' vectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.analysis.contextual import embed_sentences, text_backend, truncation_note
from core.analysis.detokenize import detokenize
from core.analysis.doc_similarity import CLASS_LABELS, band_index
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["DEFAULT_MODEL", "EmbeddingTables", "embed_corpus", "semantic_search"]

DEFAULT_MODEL = "granite-embedding-english-r2"
#: Words per passage when a document is read in pieces (well inside 512 pieces).
_PASSAGE_WORDS = 200
_TSNE_MIN = 8
_MAX_CLUSTERS = 10

_VECTOR_COLUMNS = ["Document ID", "Document", "Sentence ID", "Words", "Model", "Vector"]
_PAIR_COLUMNS = ["Document ID A", "Document A", "Document ID B", "Document B", "Similarity", "Band", "Cosine"]
_NEIGHBOUR_COLUMNS = ["Document ID", "Document", "Rank", "Neighbor", "Similarity", "Cosine"]
_MAP_COLUMNS = ["Document ID", "Document", "Sentence ID", "X", "Y", "Cluster"]
_SEARCH_COLUMNS = ["Rank", "Document ID", "Document", "Sentence ID", "Sentence", "Cosine"]


@dataclass(frozen=True, slots=True)
class EmbeddingTables:
    vectors: pd.DataFrame
    pairs: pd.DataFrame
    neighbours: pd.DataFrame
    map: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _Sentence:
    doc_id: str
    document: str
    sent_id: object
    text: str


def _sentences(frame: pd.DataFrame) -> list[_Sentence]:
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    out: list[_Sentence] = []
    for (doc_id, sent_id), group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        # As written, not as tokenized: sentence models were trained on text
        # people write ("we've", "$30,000"), and the search shows it too.
        text = detokenize(group[Col.FORM.value].astype(str).tolist())
        if text:
            name = str(group[doc_col].iloc[0]) if doc_col is not None else f"doc-{doc_id}"
            out.append(_Sentence(str(doc_id), name, sent_id, text))
    return out


def _passages(sentences: Sequence[_Sentence]) -> list[str]:
    """Whole sentences joined until a passage reaches ``_PASSAGE_WORDS`` words."""
    passages: list[str] = []
    current: list[str] = []
    words = 0
    for sentence in sentences:
        length = len(sentence.text.split())
        if current and words + length > _PASSAGE_WORDS:
            passages.append(" ".join(current))
            current, words = [], 0
        current.append(sentence.text)
        words += length
    if current:
        passages.append(" ".join(current))
    return passages


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit: np.ndarray = matrix / norms
    return unit


def _clusters(matrix: np.ndarray, seed: int) -> tuple[list[int], list[Diagnostic]]:
    """k-means labels with k chosen by silhouette over 2..10 (0 for everyone if too few)."""
    count = len(matrix)
    # Repeated texts ("Thank you.") have identical vectors: k can be at most
    # the number of distinct points, less one for silhouette to be defined.
    distinct = len(np.unique(matrix.round(6), axis=0))
    if count < 4 or distinct < 3:
        return [0] * count, []
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    top = min(_MAX_CLUSTERS, count - 1, distinct - 1)
    best: tuple[float, int, list[int]] | None = None
    for k in range(2, top + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(matrix)
        score = float(silhouette_score(matrix, labels, metric="cosine"))
        if best is None or score > best[0]:
            best = (score, k, labels.tolist())
    if best is None:
        return [0] * count, []
    score, k, labels = best
    return labels, [
        Diagnostic.info(
            "DOC_EMBED_CLUSTERS",
            f"{k} clusters (silhouette {score:.2f}, the best of k = 2..{top})",
            clusters=k,
            silhouette=round(score, 3),
        )
    ]


def _project(matrix: np.ndarray, seed: int) -> np.ndarray:
    """2-D coordinates: PCA to at most 50 dimensions, then t-SNE when there are enough points."""
    from sklearn.decomposition import PCA

    count = len(matrix)
    if count < 3:
        return np.zeros((count, 2)) if count < 2 else np.array([[0.0, 0.0], [1.0, 0.0]])[:count]
    reduced = PCA(n_components=min(50, count - 1, matrix.shape[1]), random_state=seed).fit_transform(matrix)
    if count < _TSNE_MIN:
        return reduced[:, :2] if reduced.shape[1] >= 2 else np.column_stack([reduced[:, 0], np.zeros(count)])
    from sklearn.manifold import TSNE

    perplexity = float(min(30, max(2, (count - 1) // 3)))
    coords: np.ndarray = TSNE(
        n_components=2, perplexity=perplexity, random_state=seed, init="pca", metric="cosine"
    ).fit_transform(reduced)
    return coords


def embed_corpus(
    frame: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    unit: str = "document",
    top_n: int = 5,
    seed: int = 42,
    backend: Any = None,
) -> Result[EmbeddingTables]:
    """Vectors, pairwise similarity, neighbours and a map for documents or sentences."""
    if unit not in ("document", "sentence"):
        return Result.failure(
            Diagnostic.error("DOC_EMBED_BAD_UNIT", f"unit must be document or sentence, got {unit!r}")
        )
    if isinstance(top_n, bool) or top_n < 1:
        return Result.failure(Diagnostic.error("DOC_EMBED_BAD_K", f"top_n must be >= 1, got {top_n!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[EmbeddingTables](None, checked.diagnostics)
    sentences = _sentences(frame)
    if not sentences:
        return Result.failure(Diagnostic.error("DOC_EMBED_EMPTY", "no sentences to embed"))
    diags: list[Diagnostic] = []
    resolved = backend
    if resolved is None:
        opened = text_backend(model)
        if opened.value is None:
            return Result.failure(*opened.diagnostics)
        resolved = opened.value
    name = getattr(resolved, "model_name", model)

    try:
        if unit == "sentence":
            matrix = _unit(np.asarray(embed_sentences(resolved, [s.text for s in sentences]), dtype=float))
            rows = [(s.doc_id, s.document, s.sent_id, len(s.text.split())) for s in sentences]
        else:
            by_doc: dict[str, list[_Sentence]] = {}
            for sentence in sentences:
                by_doc.setdefault(sentence.doc_id, []).append(sentence)
            passages: list[str] = []
            owners: list[int] = []
            for index, members in enumerate(by_doc.values()):
                for passage in _passages(members):
                    passages.append(passage)
                    owners.append(index)
            passage_vectors = np.asarray(embed_sentences(resolved, passages), dtype=float)
            weights = np.array([len(passage.split()) for passage in passages], dtype=float)
            matrix = np.zeros((len(by_doc), passage_vectors.shape[1]))
            for vector, owner, weight in zip(passage_vectors, owners, weights, strict=True):
                matrix[owner] += weight * vector
            matrix = _unit(matrix)
            rows = [
                (doc_id, members[0].document, None, sum(len(s.text.split()) for s in members))
                for doc_id, members in by_doc.items()
            ]
    except (RuntimeError, ValueError) as exc:
        return Result.failure(Diagnostic.error("DOC_EMBED_MODEL_FAILED", str(exc)))
    diags.extend(truncation_note(resolved))

    vectors = pd.DataFrame(
        [
            {
                "Document ID": doc_id,
                "Document": document,
                "Sentence ID": sent_id,
                "Words": words,
                "Model": name,
                "Vector": ",".join(f"{value:.4f}" for value in vector),
            }
            for (doc_id, document, sent_id, words), vector in zip(rows, matrix, strict=True)
        ],
        columns=_VECTOR_COLUMNS,
    )

    similarity = matrix @ matrix.T
    # Relative similarity: the cosine after removing what every document here
    # shares (the corpus's mean vector). A sentence model reads every speech of
    # one genre as ~98% alike -- true, and useless for telling them apart.
    # Centred, 0 is "as alike as a typical pair" and the spread is the finding.
    relative = _unit(matrix - matrix.mean(axis=0)) if len(matrix) > 2 else matrix
    related = relative @ relative.T
    labels = [f"{document} #{sent_id}" if sent_id is not None else document for _, document, sent_id, _ in rows]
    pair_rows = []
    neighbour_rows = []
    # Pairs and neighbours are between documents; for sentences they would be
    # millions of rows, so a sentence run keeps its vectors and map only.
    if unit == "document":
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                percent = float(related[i, j]) * 100.0
                band = band_index(percent) if percent > 0 else None
                pair_rows.append(
                    {
                        "Document ID A": rows[i][0],
                        "Document A": rows[i][1],
                        "Document ID B": rows[j][0],
                        "Document B": rows[j][1],
                        "Similarity": round(percent, 1),
                        "Band": CLASS_LABELS[band] if band is not None else "below average",
                        "Cosine": round(float(similarity[i, j]), 4),
                    }
                )
            order = [j for j in np.argsort(-related[i], kind="stable").tolist() if j != i][:top_n]
            for rank, j in enumerate(order, 1):
                neighbour_rows.append(
                    {
                        "Document ID": rows[i][0],
                        "Document": rows[i][1],
                        "Rank": rank,
                        "Neighbor": rows[j][1],
                        "Similarity": round(float(related[i, j]) * 100.0, 1),
                        "Cosine": round(float(similarity[i, j]), 4),
                    }
                )
    pairs = pd.DataFrame(pair_rows, columns=_PAIR_COLUMNS)
    if not pairs.empty:
        pairs = pairs.sort_values(["Similarity", "Document ID A", "Document ID B"], ascending=[False, True, True])
        pairs = pairs.reset_index(drop=True)

    coords = _project(matrix, seed)
    clusters, cluster_notes = _clusters(matrix, seed)
    diags.extend(cluster_notes)
    mapped = pd.DataFrame(
        [
            {
                "Document ID": doc_id,
                "Document": label,
                "Sentence ID": sent_id,
                "X": round(float(x), 4),
                "Y": round(float(y), 4),
                "Cluster": f"Cluster {cluster + 1}",
            }
            for (doc_id, _, sent_id, _), label, (x, y), cluster in zip(rows, labels, coords, clusters, strict=True)
        ],
        columns=_MAP_COLUMNS,
    )
    return Result.success(
        EmbeddingTables(
            vectors=vectors,
            pairs=pairs,
            neighbours=pd.DataFrame(neighbour_rows, columns=_NEIGHBOUR_COLUMNS),
            map=mapped,
        ),
        *diags,
    )


def semantic_search(
    frame: pd.DataFrame,
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    top_n: int = 20,
    backend: Any = None,
) -> Result[pd.DataFrame]:
    """The sentences closest in meaning to *query*, best first."""
    if not query or not query.strip():
        return Result.failure(Diagnostic.error("DOC_EMBED_BAD_QUERY", "query must be non-empty"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    sentences = _sentences(frame)
    if not sentences:
        return Result.success(pd.DataFrame(columns=_SEARCH_COLUMNS))
    resolved = backend
    if resolved is None:
        opened = text_backend(model)
        if opened.value is None:
            return Result.failure(*opened.diagnostics)
        resolved = opened.value
    try:
        documents = _unit(np.asarray(embed_sentences(resolved, [s.text for s in sentences]), dtype=float))
        embed_texts = getattr(resolved, "embed_texts", None)
        if callable(embed_texts):
            try:
                asked = embed_texts([query.strip()], query=True)
            except TypeError:  # a token model has no query mode
                asked = embed_texts([query.strip()])
        else:
            asked = resolved.embed([query.strip()], [query.strip()])
        target = _unit(np.asarray(asked, dtype=float))[0]
    except (RuntimeError, ValueError) as exc:
        return Result.failure(Diagnostic.error("DOC_EMBED_MODEL_FAILED", str(exc)))
    scores = documents @ target
    order = np.argsort(-scores, kind="stable")[:top_n]
    rows = [
        {
            "Rank": rank,
            "Document ID": sentences[i].doc_id,
            "Document": sentences[i].document,
            "Sentence ID": sentences[i].sent_id,
            "Sentence": sentences[i].text,
            "Cosine": round(float(scores[i]), 4),
        }
        for rank, i in enumerate(order.tolist(), 1)
    ]
    return Result.success(pd.DataFrame(rows, columns=_SEARCH_COLUMNS), *truncation_note(resolved))
