"""Story-shape data reduction — HC, SVD, and NMF over shape trajectories (HW5).

The syllabus names the comparison itself: "Data reduction algorithms:
Hierarchical Clustering (HC), Singular Value Decomposition (SVD),
Non-Negative Matrix Factorization (NMF)" and asks which one produces better
results and what the DIFFERENCE between them is. So these are three separate
tools over ONE shared matrix — every reduction sees the same numbers, and the
difference between the answers is the finding.

The matrix is the deterministic story shape of :mod:`core.viz.shapes`
(Tokens / Noun Ratio / Verb Ratio per sentence), resampled per document to a
fixed length exactly as :mod:`core.viz.shape_clusters` does for KMeans, so
these three reductions sit beside that clustering on identical inputs.

What each one means, in the plainest terms the guides repeat:

* ``hierarchical_cluster`` groups documents whose trajectories look alike and
  shows the grouping as a tree of merge heights.
* ``svd_reduce`` finds orthogonal axes of variation, each with its share of
  the variance — the "shape components" stories can be projected onto.
* ``nmf_reduce`` finds additive non-negative parts. NMF cannot see negative
  numbers, so each feature column is shifted up by its own minimum before the
  fit and the shift is NOT undone: the parts read as non-negative
  contributions and the reconstruction is of the shifted matrix.

Heavy imports (``scipy``, ``sklearn``) are lazy; nothing here writes to disk
(R3) and everything is deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "HcResult",
    "ReductionResult",
    "ShapeMatrix",
    "build_shape_matrix",
    "hierarchical_cluster",
    "matrix_frame",
    "nmf_reduce",
    "svd_reduce",
]

#: The three deterministic metrics of core.viz.shapes.story_shape, in the
#: order core.viz.shape_clusters concatenates them into one trajectory.
METRICS: tuple[str, ...] = ("Tokens", "Noun Ratio", "Verb Ratio")
_SHAPE_COLUMNS = ["Document ID", "Sentence ID", *METRICS]
_DEFAULT_RESAMPLE = 32
_METHODS = ("ward", "average", "complete", "single")


@dataclass(frozen=True, slots=True)
class ShapeMatrix:
    """One row per document: its story trajectory, resampled to fixed length."""

    documents: tuple[str, ...]
    labels: tuple[str, ...]
    values: np.ndarray
    resample: int
    metrics: tuple[str, ...] = METRICS

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"{metric} {index:02d}" for metric in self.metrics for index in range(1, self.resample + 1))


@dataclass(frozen=True, slots=True)
class HcResult:
    """Hierarchical clustering: assignments, the merge table, and its tree."""

    assignments: pd.DataFrame
    merges: pd.DataFrame
    heights: tuple[float, ...]
    method: str
    n_clusters: int
    cophenetic: float | None
    leaf_order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReductionResult:
    """SVD or NMF: document scores, feature loadings, and the variance story."""

    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained: pd.DataFrame
    method: str
    n_components: int
    seed: int
    shift: float = 0.0


def _check(sentences: pd.DataFrame) -> Diagnostic | None:
    missing = [c for c in _SHAPE_COLUMNS if c not in sentences.columns]
    if missing:
        return Diagnostic.error("SHAPE_REDUCTION_MISSING_COLUMN", f"missing {missing[0]!r}")
    return None


def _resample(values: np.ndarray, resample: int) -> np.ndarray:
    """Linear-interp one metric's per-sentence series to ``resample`` points."""
    x = np.linspace(0.0, 1.0, num=resample)
    xp = np.linspace(0.0, 1.0, num=len(values))
    return np.asarray(np.interp(x, xp, values), dtype=np.float64)


def build_shape_matrix(sentences: pd.DataFrame, *, resample: int = _DEFAULT_RESAMPLE) -> Result[ShapeMatrix]:
    """Turn per-sentence shape metrics into one comparable row per document.

    Documents keep first-appearance order — sorting the string ids would put
    "10" before "2", the trap ``core.viz.shapes.story_shape`` documents. An
    empty frame is an empty matrix (typed, zero rows) rather than a failure;
    ONE document is a failure, because a reduction of a single trajectory has
    nothing to compare.
    """
    if resample < 2:
        return Result.failure(
            Diagnostic.error(
                "SHAPE_REDUCTION_BAD_RESAMPLE",
                f"resample must be >=2, got {resample}",
                fix="resample=32 matches the story-shape clustering default",
            )
        )
    checked = _check(sentences) if not sentences.empty else None
    if checked is not None:
        return Result.failure(checked)
    if sentences.empty:
        return Result.success(
            ShapeMatrix(documents=(), labels=(), values=np.zeros((0, len(METRICS) * resample)), resample=resample)
        )
    doc_col = "Document" if "Document" in sentences.columns else None
    documents: list[str] = []
    labels: list[str] = []
    trajectories: list[np.ndarray] = []
    for doc_id, group in sentences.groupby("Document ID", sort=False):
        parts: list[np.ndarray] = []
        for metric in METRICS:
            series = pd.to_numeric(group[metric], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            parts.append(_resample(series, resample))
        documents.append(str(doc_id))
        labels.append(str(group[doc_col].iloc[0]) if doc_col is not None else f"doc-{doc_id}")
        trajectories.append(np.concatenate(parts))
    if len(trajectories) < 2:
        return Result.failure(
            Diagnostic.error(
                "SHAPE_REDUCTION_TOO_SMALL",
                f"data reduction needs at least 2 documents with sentences, got {len(trajectories)}",
                fix="the corpus is too small for data-reduction algorithms (the legacy suite's own message)",
            )
        )
    return Result.success(
        ShapeMatrix(
            documents=tuple(documents),
            labels=tuple(labels),
            values=np.array(trajectories, dtype=np.float64),
            resample=resample,
        )
    )


def matrix_frame(shape: ShapeMatrix) -> pd.DataFrame:
    """The matrix as a table: Document ID, Document, then one column per point."""
    columns = ["Document ID", "Document", *shape.feature_names]
    if not shape.documents:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "Document ID": doc,
            "Document": label,
            **{name: round(float(v), 4) for name, v in zip(shape.feature_names, row, strict=True)},
        }
        for doc, label, row in zip(shape.documents, shape.labels, shape.values, strict=True)
    ]
    return pd.DataFrame(rows, columns=columns)


def _too_small(shape: ShapeMatrix, need: int = 2) -> Result[object] | None:
    if len(shape.documents) >= need:
        return None
    return Result.failure(
        Diagnostic.error(
            "SHAPE_REDUCTION_TOO_SMALL",
            f"data reduction needs at least {need} documents, got {len(shape.documents)}",
            fix="the corpus is too small for data-reduction algorithms (the legacy suite's own message)",
        )
    )


def _score_frame(documents: tuple[str, ...], labels: tuple[str, ...], scores: np.ndarray) -> pd.DataFrame:
    names = [f"Component {i + 1}" for i in range(scores.shape[1])]
    rows = [
        {
            "Document ID": doc,
            "Document": label,
            **{name: round(float(v), 4) for name, v in zip(names, row, strict=True)},
        }
        for doc, label, row in zip(documents, labels, scores, strict=True)
    ]
    return pd.DataFrame(rows, columns=["Document ID", "Document", *names])


def _loading_frame(feature_names: tuple[str, ...], loadings: np.ndarray) -> pd.DataFrame:
    """One row per FEATURE, one column per component (loadings are k x F)."""
    names = [f"Component {i + 1}" for i in range(loadings.shape[0])]
    rows = []
    for index, feature in enumerate(feature_names):
        rows.append(
            {
                "Feature": feature,
                **{name: round(float(loadings[slot, index]), 4) for slot, name in enumerate(names)},
            }
        )
    return pd.DataFrame(rows, columns=["Feature", *names])


def _explained_frame(shares: list[float], share_label: str) -> pd.DataFrame:
    rows = []
    cumulative = 0.0
    for index, share in enumerate(shares, start=1):
        cumulative += share
        rows.append({"Component": index, share_label: round(share, 4), "Cumulative": round(cumulative, 4)})
    return pd.DataFrame(rows, columns=["Component", share_label, "Cumulative"])


def hierarchical_cluster(
    shape: ShapeMatrix,
    *,
    method: str = "ward",
    n_clusters: int = 2,
) -> Result[HcResult]:
    """Group documents by trajectory shape and expose the tree that did it.

    ``Distance`` is each document's distance to its cluster's centroid in the
    same feature space the tree was built on, so a large value flags the
    member that fits its group least.
    """
    guard = _too_small(shape, 2)
    if guard is not None:
        return guard  # type: ignore[return-value]
    if method not in _METHODS:
        return Result.failure(
            Diagnostic.error("SHAPE_HC_BAD_METHOD", f"method must be one of {_METHODS}, got {method!r}")
        )
    n_docs = len(shape.documents)
    if not 2 <= n_clusters < n_docs + 1 or n_clusters > n_docs:
        return Result.failure(
            Diagnostic.error(
                "SHAPE_HC_BAD_K",
                f"n_clusters must be between 2 and {n_docs}, got {n_clusters}",
                fix=f"choose 2..{n_docs}",
            )
        )
    from scipy.cluster.hierarchy import cophenet, fcluster, leaves_list, linkage
    from scipy.spatial.distance import pdist

    tree = linkage(shape.values, method=method)
    flat = fcluster(tree, t=n_clusters, criterion="maxclust")
    order = tuple(int(i) for i in leaves_list(tree))
    try:
        coefficient = float(cophenet(tree, pdist(shape.values))[0])
    except Exception:
        coefficient = None

    rows = []
    for index, (doc, label, cluster) in enumerate(zip(shape.documents, shape.labels, flat.tolist(), strict=True)):
        members = [i for i, c in enumerate(flat.tolist()) if c == cluster]
        centroid = shape.values[members].mean(axis=0)
        distance = float(np.linalg.norm(shape.values[index] - centroid))
        rows.append({"Document ID": doc, "Document": label, "Cluster": int(cluster), "Distance": round(distance, 4)})
    assignments = pd.DataFrame(rows, columns=["Document ID", "Document", "Cluster", "Distance"])

    heights = tuple(round(float(row[2]), 4) for row in tree)
    names: dict[int, str] = {i: label for i, label in enumerate(shape.labels)}
    merges = []
    for index, (left, right, height, _count) in enumerate(tree, start=1):
        left_id, right_id = int(left), int(right)
        merges.append(
            {
                "Merge": index,
                "Left": names.get(left_id, f"cluster {left_id - n_docs + 1}"),
                "Right": names.get(right_id, f"cluster {right_id - n_docs + 1}"),
                "Height": round(float(height), 4),
                "Left id": left_id,
                "Right id": right_id,
            }
        )
    merges_frame = pd.DataFrame(merges, columns=["Merge", "Left", "Right", "Height", "Left id", "Right id"])
    return Result.success(
        HcResult(
            assignments=assignments,
            merges=merges_frame,
            heights=heights,
            method=method,
            n_clusters=n_clusters,
            cophenetic=coefficient,
            leaf_order=order,
        )
    )


def svd_reduce(
    shape: ShapeMatrix,
    *,
    n_components: int = 2,
    seed: int = 42,
) -> Result[ReductionResult]:
    """Orthogonal axes of variation, each with its share of the variance.

    ``TruncatedSVD``'s randomized algorithm is seeded, so the same seed and
    the same matrix give back the same axes.
    """
    guard = _too_small(shape, 2)
    if guard is not None:
        return guard  # type: ignore[return-value]
    n_docs, n_features = shape.values.shape
    highest = max(1, min(n_docs, n_features - 1))
    if not 1 <= n_components <= highest:
        return Result.failure(
            Diagnostic.error(
                "SHAPE_SVD_BAD_K",
                f"n_components must be between 1 and {highest}, got {n_components}",
                fix=f"choose 1..{highest} for this corpus",
            )
        )
    from sklearn.decomposition import TruncatedSVD

    model = TruncatedSVD(n_components=n_components, random_state=seed)
    scores = model.fit_transform(shape.values)
    shares = [float(v) for v in model.explained_variance_ratio_]
    total = sum(shares) or 1.0
    normalized = [share / total for share in shares]
    return Result.success(
        ReductionResult(
            scores=_score_frame(shape.documents, shape.labels, scores),
            loadings=_loading_frame(shape.feature_names, model.components_),
            explained=_explained_frame(normalized, "Explained variance"),
            method="svd",
            n_components=n_components,
            seed=seed,
        )
    )


def nmf_reduce(
    shape: ShapeMatrix,
    *,
    n_components: int = 2,
    seed: int = 42,
    max_iter: int = 200,
) -> Result[ReductionResult]:
    """Additive non-negative parts of the story trajectories.

    Each FEATURE column is shifted up by its own minimum so nothing is
    negative, and the shift is deliberately not undone: the parts are
    interpretable as non-negative contributions, and the reconstruction is of
    the shifted matrix. ``shift`` reports the largest single column shift.
    """
    guard = _too_small(shape, 2)
    if guard is not None:
        return guard  # type: ignore[return-value]
    if max_iter < 10:
        return Result.failure(Diagnostic.error("SHAPE_NMF_BAD_ITER", f"max_iter must be >=10, got {max_iter}"))
    n_docs, n_features = shape.values.shape
    highest = max(1, min(n_docs, n_features))
    if not 1 <= n_components <= highest:
        return Result.failure(
            Diagnostic.error(
                "SHAPE_NMF_BAD_K",
                f"n_components must be between 1 and {highest}, got {n_components}",
                fix=f"choose 1..{highest} for this corpus",
            )
        )
    from sklearn.decomposition import NMF

    column_mins = shape.values.min(axis=0)
    shifted = shape.values - column_mins
    shift = float(column_mins.max()) if len(column_mins) else 0.0
    init = "nndsvda" if n_components <= min(shifted.shape) else "random"
    model = NMF(n_components=n_components, init=init, random_state=seed, max_iter=max_iter)
    scores = model.fit_transform(shifted)
    loadings = model.components_
    masses = [float(scores[:, i].sum() * loadings[i].sum()) for i in range(n_components)]
    total = sum(masses) or 1.0
    share_label = "Share of mass"
    diags = [
        Diagnostic.info(
            "SHAPE_NMF_SHIFT",
            f"NMF saw values shifted up by up to {shift:.4f} (non-negative input)",
            shift=round(shift, 4),
        )
    ]
    # Shares stay in component order (not sorted): row i of the table is
    # component i's share, so the table can be read against the score and
    # loading columns without a translation step.
    return Result.success(
        ReductionResult(
            scores=_score_frame(shape.documents, shape.labels, scores),
            loadings=_loading_frame(shape.feature_names, loadings),
            explained=_explained_frame([mass / total for mass in masses], share_label),
            method="nmf",
            n_components=n_components,
            seed=seed,
            shift=shift,
        ),
        *diags,
    )
