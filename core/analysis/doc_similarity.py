"""Document similarity (FR-2.9) — pairwise TF-IDF cosine over a corpus.

Parity basis: ``plagiarist_util.compute_and_write`` (legacy). Symmetric
cosine, 0-100 scale, the ten Java-matching bands, duplicate thresholding.
Stopwords travel as an explicit list parameter — never a magic file path.

C6-9 corrections: band boundaries follow the documented intervals exactly
((0,10] via floor, no integer rounding); thresholds are validated in every
public entry point (NaN/inf/out-of-range rejected); rows carry document ID
AND relative path (duplicate basenames stay distinct); empty documents are
skipped with warnings; equal similarities order deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.io.reader import Corpus, Document, display_names
from core.result import Diagnostic, Result

__all__ = [
    "CLASS_LABELS",
    "DocPairsResult",
    "DuplicatesResult",
    "band_index",
    "find_duplicates",
    "pairwise_similarity",
]


CLASS_LABELS = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]


def band_index(pct: float) -> int | None:
    """Map a similarity percentage to a class index 0..9 (None when <= 0).

    C6-9: intervals are exactly the documented ones — (0,10], (10,20], ...
    (90,100] — computed by floor on the raw float, NO integer rounding
    (10.0 -> band 0, 10.1 -> band 1). NaN/inf raise ValueError; callers
    validate thresholds before reaching for bands.
    """
    if math.isnan(pct) or math.isinf(pct):
        raise ValueError(f"band index requires a finite percentage, got {pct}")
    if pct <= 0:
        return None
    return max(0, min(9, int((pct - 1e-9) // 10)))


def _validate_threshold(threshold: float) -> Diagnostic | None:
    """0-100 finite numbers only; bool is not a threshold."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return Diagnostic.error(
            "DUPS_BAD_THRESHOLD",
            f"threshold must be a number, got {type(threshold).__name__}",
            threshold=str(threshold),
        )
    if math.isnan(float(threshold)) or math.isinf(float(threshold)):
        return Diagnostic.error("DUPS_BAD_THRESHOLD", "threshold must be finite", threshold=threshold)
    if not 0 <= float(threshold) <= 100:
        return Diagnostic.error("DUPS_BAD_THRESHOLD", f"threshold must be 0-100, got {threshold}", threshold=threshold)
    return None


@dataclass(frozen=True, slots=True)
class DocPairsResult:
    frame: pd.DataFrame  # Document ID/path both sides, Similarity, Band
    # Kept out of the public frame: thresholding must use the unrounded score
    # even though the CSV-facing Similarity column is rounded for readability.
    raw_similarities: tuple[float, ...] = ()

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


@dataclass(frozen=True, slots=True)
class DuplicatesResult:
    frame: pd.DataFrame  # Document ID/path both sides, Similarity (>= threshold)

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


@dataclass(frozen=True, slots=True)
class _Entry:
    doc_id: int
    display: str


def _hash_stopwords(stopwords: list[str] | None) -> str:
    """Deterministic fingerprint of the stopword list (provenance, C6-4)."""
    if stopwords is None:
        return ""
    joined = "\n".join(stopwords)
    return "sw:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _matrix(
    corpus: Corpus, stopwords: list[str] | None
) -> tuple[list[_Entry], np.ndarray, list[Diagnostic]] | Diagnostic:
    entries: list[_Entry] = []
    texts: list[str] = []
    diags: list[Diagnostic] = []
    # Computed over the whole corpus, not just the documents that survive the
    # emptiness filter, so a label never changes because some other file was
    # excluded from this particular run.
    names = display_names(corpus.docs)
    for doc in corpus.docs:
        if doc.text.strip():
            entries.append(_Entry(doc_id=doc.doc_id, display=names[doc.doc_id]))
            texts.append(doc.text)
        else:
            diags.append(
                Diagnostic.warning("SIM_EMPTY_DOC", f"{names[doc.doc_id]} is empty; excluded", doc_id=doc.doc_id)
            )
    if len(texts) < 2:
        return Diagnostic.error("SIM_TOO_FEW_DOCS", "need at least 2 non-empty documents", n=len(texts))
    vectorizer = TfidfVectorizer(stop_words=stopwords, sublinear_tf=True)
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError as exc:
        return Diagnostic.error("SIM_VECTORIZE_FAILED", f"TF-IDF failed: {exc}")
    sim = cosine_similarity(tfidf) * 100.0
    return entries, sim, diags


_PAIR_COLUMNS = [
    "Document ID A",
    "Document A",
    "Document ID B",
    "Document B",
    "Similarity",
    "Band",
]


def pairwise_similarity(corpus: Corpus, stopwords: list[str] | None = None) -> Result[DocPairsResult]:
    """Every unordered document pair with its 0-100 similarity and band."""
    built = _matrix(corpus, stopwords)
    if isinstance(built, Diagnostic):
        return Result.failure(built)
    entries, sim, diags = built
    n_entries = len(entries)
    iu, ju = np.triu_indices(n_entries, k=1)
    sim_arr = np.asarray(sim, dtype=float)
    raw_arr = sim_arr[iu, ju] if n_entries else np.empty(0, dtype=float)
    pct_arr = np.asarray([round(float(v), 1) for v in raw_arr], dtype=float)
    id_a_arr = np.array([entries[int(i)].doc_id for i in iu])
    id_b_arr = np.array([entries[int(j)].doc_id for j in ju])
    if len(iu):
        order = np.lexsort((id_b_arr, id_a_arr, -pct_arr))
        iu_ord = iu[order]
        ju_ord = ju[order]
        raw_sorted = raw_arr[order]
        pct_sorted = pct_arr[order]
        bands = []
        for val in raw_sorted:
            band = band_index(float(val))
            bands.append(CLASS_LABELS[band] if band is not None else "0%")
        out = pd.DataFrame(
            {
                "Document ID A": id_a_arr[order],
                "Document A": [entries[int(i)].display for i in iu_ord],
                "Document ID B": id_b_arr[order],
                "Document B": [entries[int(j)].display for j in ju_ord],
                "Similarity": pct_sorted,
                "Band": bands,
            },
            columns=_PAIR_COLUMNS,
        )
        raw = [float(v) for v in raw_sorted]
    else:
        out = pd.DataFrame(columns=_PAIR_COLUMNS)
        raw = []
    return Result.success(DocPairsResult(frame=out, raw_similarities=tuple(raw)), *diags)


def find_duplicates(
    corpus: Corpus,
    threshold: float = 80.0,
    stopwords: list[str] | None = None,
    *,
    pairs: DocPairsResult | None = None,
) -> Result[DuplicatesResult]:
    """Pairs at or above *threshold* (0-100), ranked by similarity descending."""
    threshold_problem = _validate_threshold(threshold)
    if threshold_problem is not None:
        return Result.failure(threshold_problem)
    diags: tuple[Diagnostic, ...] = ()
    if pairs is None:
        pairs_result = pairwise_similarity(corpus, stopwords)
        if pairs_result.value is None:
            return Result.failure(*pairs_result.diagnostics)
        pair_value = pairs_result.unwrap()
        diags = tuple(pairs_result.diagnostics)
    else:
        pair_value = pairs
    pairs_frame = pair_value.frame
    # ``Similarity`` is intentionally rounded in the public output. Use the
    # hidden raw scores for the decision so 79.96 does not become an 80.0
    # duplicate just because of display rounding.
    raw_scores = pair_value.raw_similarities
    if len(raw_scores) != len(pairs_frame):
        raw_scores = tuple(float(value) for value in pairs_frame["Similarity"].tolist())
    limit = float(threshold)
    mask = [score >= limit for score in raw_scores]
    dupes = pairs_frame.loc[mask, _PAIR_COLUMNS[:-1]].reset_index(drop=True)
    return Result.success(DuplicatesResult(frame=dupes), *diags)


def stopword_fingerprint(stopwords: list[str] | None) -> str:
    """Record the exact stopword list in use (C6-4 provenance)."""
    return _hash_stopwords(stopwords)
