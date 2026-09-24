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
) -> tuple[list[_Entry], list[list[float]], list[Diagnostic]] | Diagnostic:
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
    sim = cosine_similarity(tfidf)
    matrix = [[float(sim[i, j]) * 100.0 for j in range(len(entries))] for i in range(len(entries))]
    return entries, matrix, diags


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
    entries, matrix, diags = built
    rows = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            raw_pct = matrix[i][j]
            # Classification uses the raw cosine percentage. Rounding is a
            # display concern only: 10.04% belongs to the 10-20% interval,
            # not the 0-10% interval merely because it displays as 10.0.
            band = band_index(raw_pct)
            pct = round(raw_pct, 1)
            rows.append(
                {
                    "Document ID A": entries[i].doc_id,
                    "Document A": entries[i].display,
                    "Document ID B": entries[j].doc_id,
                    "Document B": entries[j].display,
                    "Similarity": pct,
                    "Band": CLASS_LABELS[band] if band is not None else "0%",
                    "_raw_similarity": raw_pct,
                }
            )
    out = pd.DataFrame(rows, columns=_PAIR_COLUMNS)
    if rows:
        # Sort the hidden raw scores together with their display rows, then
        # drop the implementation detail before exposing the frame.
        raw = (
            pd.DataFrame(rows)
            .sort_values(["Similarity", "Document ID A", "Document ID B"], ascending=[False, True, True])[
                "_raw_similarity"
            ]
            .astype(float)
            .tolist()
        )
        out = out.sort_values(
            ["Similarity", "Document ID A", "Document ID B"], ascending=[False, True, True]
        ).reset_index(drop=True)
    else:
        raw = []
    return Result.success(DocPairsResult(frame=out, raw_similarities=tuple(raw)), *diags)


def find_duplicates(
    corpus: Corpus, threshold: float = 80.0, stopwords: list[str] | None = None
) -> Result[DuplicatesResult]:
    """Pairs at or above *threshold* (0-100), ranked by similarity descending."""
    threshold_problem = _validate_threshold(threshold)
    if threshold_problem is not None:
        return Result.failure(threshold_problem)
    pairs_result = pairwise_similarity(corpus, stopwords)
    if pairs_result.value is None:
        return Result.failure(*pairs_result.diagnostics)
    pair_value = pairs_result.unwrap()
    pairs = pair_value.to_frame()
    # ``Similarity`` is intentionally rounded in the public output. Use the
    # hidden raw scores for the decision so 79.96 does not become an 80.0
    # duplicate just because of display rounding.
    raw_scores = pair_value.raw_similarities
    if len(raw_scores) != len(pairs):
        raw_scores = tuple(float(value) for value in pairs["Similarity"].tolist())
    mask = [score >= float(threshold) for score in raw_scores]
    dupes = pairs.loc[mask].reset_index(drop=True)
    return Result.success(DuplicatesResult(frame=dupes[_PAIR_COLUMNS[:-1]]), *pairs_result.diagnostics)


def stopword_fingerprint(stopwords: list[str] | None) -> str:
    """Record the exact stopword list in use (C6-4 provenance)."""
    return _hash_stopwords(stopwords)
