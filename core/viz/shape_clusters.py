"""Story-shape clustering (CAP-VIZ-08) — KMeans over per-document shape vectors.

Legacy reference: ``src/shape_of_stories_vectorizer_util.py`` — each document
becomes a fixed-length vector (legacy: bucketed sentiment per window), PCA on
the vector set recommends the number of clusters by explained variance, and
KMeans groups documents with similar narrative trajectories. Ours uses the
deterministic shape metrics from :func:`core.viz.shapes.story_shape`
(tokens, noun ratio, verb ratio per sentence) resampled to a fixed length —
no sentiment model, so the clustering is reproducible from the parse alone.

The suggested-k heuristic follows the legacy shape: PCA explained variance
reaching ``variance_threshold`` picks k. KMeans itself is seeded (``n_init``
fixed, ``random_state`` given) for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from core.result import Diagnostic, Result

__all__ = ["cluster_shapes", "suggest_n_clusters"]

_RESAMPLE_LEN = 32  # every document trajectory becomes 32 points
_MIN_DOCS = 3  # KMeans needs at least 2; below 3 the variance heuristic is noise
_VECTORS = ("Tokens", "Noun Ratio", "Verb Ratio")


def _trajectory(sentences: pd.DataFrame) -> np.ndarray | None:
    """Resample one document's per-sentence shape metrics to a fixed length."""
    if sentences.empty:
        return None
    # Sentence order is the narrative x-axis; resample each metric to a
    # common length so documents of different sizes are comparable.
    x = np.linspace(0.0, 1.0, num=_RESAMPLE_LEN)
    xp = np.linspace(0.0, 1.0, num=len(sentences))
    parts = []
    for metric in _VECTORS:
        values = pd.to_numeric(sentences[metric], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        parts.append(np.interp(x, xp, values))
    return np.concatenate(parts)  # type: ignore[no-any-return]


def suggest_n_clusters(vectors: np.ndarray, variance_threshold: float = 0.9) -> Result[int]:
    """Legacy heuristic: first PCA component count reaching the variance floor."""
    n_docs, n_features = vectors.shape
    if n_docs < 3:
        return Result.failure(
            Diagnostic.error(
                "SHAPE_CLUSTER_TOO_SMALL",
                f"clustering needs at least 3 documents, got {n_docs} — the corpus is too "
                "small for data-reduction algorithms (legacy message)",
            )
        )
    n_components = min(n_docs, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(vectors)
    explained = np.sort(pca.explained_variance_ratio_)[::-1]
    cumulative = 0.0
    for i, share in enumerate(explained, start=1):
        cumulative += float(share)
        if cumulative >= variance_threshold:
            return Result.success(max(2, i))
    return Result.success(max(2, n_components))


def cluster_shapes(
    shape: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    seed: int = 42,
) -> Result[tuple[pd.DataFrame, int]]:
    """KMeans over per-document shape trajectories.

    Returns ``(assignments, effective_k)`` where assignments carry one row per
    document (Document ID, Document, Cluster, Distance) and *n_clusters* the
    cluster count actually used (the legacy PCA heuristic when not given).
    """
    checked = _check(shape)
    if checked is not None:
        return Result[tuple[pd.DataFrame, int]](None, (checked,))

    doc_col = "Document" if "Document" in shape.columns else None
    doc_ids: list[str] = []
    doc_names: list[str] = []
    trajectories: list[np.ndarray] = []
    for did, g in shape.groupby("Document ID", sort=False):
        traj = _trajectory(g)
        if traj is not None:
            doc_ids.append(str(did))
            doc_names.append(str(g[doc_col].iloc[0]) if doc_col else f"doc-{did}")
            trajectories.append(traj)
    if len(trajectories) < 3:
        return Result[tuple[pd.DataFrame, int]](
            None,
            (
                Diagnostic.error(
                    "SHAPE_CLUSTER_TOO_SMALL",
                    f"clustering needs at least 3 documents, got {len(trajectories)}",
                ),
            ),
        )

    matrix = np.array(trajectories, dtype=np.float64)
    if n_clusters is None:
        suggested = suggest_n_clusters(matrix)
        if suggested.value is None:
            return Result[tuple[pd.DataFrame, int]](None, suggested.diagnostics)
        n_clusters = suggested.unwrap()
    n_clusters = max(2, min(int(n_clusters), len(trajectories)))

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(matrix)
    rows: list[dict[str, object]] = []
    for index, ((did, doc_name), cluster) in enumerate(
        zip(zip(doc_ids, doc_names, strict=True), labels.tolist(), strict=True)
    ):
        rows.append(
            {
                "Document ID": did,
                "Document": doc_name,
                "Cluster": int(cluster),
                "Distance": round(float(np.linalg.norm(matrix[index] - km.cluster_centers_[cluster])), 4),
            }
        )
    return Result.success((pd.DataFrame(rows, columns=["Document ID", "Document", "Cluster", "Distance"]), n_clusters))


def _check(shape: pd.DataFrame) -> Diagnostic | None:
    if shape.empty:
        return Diagnostic.error("SHAPE_CLUSTER_EMPTY", "shape frame is empty")
    missing = [c for c in ("Document ID", *_VECTORS) if c not in shape.columns]
    if missing:
        return Diagnostic.error("SHAPE_CLUSTER_BAD_COLUMN", f"column(s) not in frame: {missing}", missing=missing)
    return None
