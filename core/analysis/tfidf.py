"""TF-IDF term weighting over a parsed corpus (CAP-STATS-12).

Raw frequency tells you a document uses "the" a lot. TF-IDF asks a better
question: which terms are frequent *here* and rare *elsewhere*? Those are the
terms that distinguish one document from the rest of the collection, which is
what you want when labelling documents, building a search index or picking
seed terms for closer reading.

The weight is ``tf * idf`` with the smoothed, standard definitions:

* **tf** -- the term's count in the document, or ``1 + ln(count)`` under
  ``sublinear_tf``. Sublinear scaling reflects that the tenth occurrence of a
  word adds less evidence than the second.
* **idf** -- ``ln((1 + N) / (1 + df)) + 1``, where *N* is the document count
  and *df* the number of documents containing the term. The ``+1`` terms are
  the "smooth idf" convention: they behave as if one extra document contained
  every term, so a term appearing everywhere gets idf 1 rather than 0 and is
  down-weighted instead of deleted.
* **L2 normalisation** (optional, on by default) divides each document's
  vector by its Euclidean norm, so long and short documents are comparable.
  Compare weights *within* a document freely; compare *across* documents only
  with normalisation on.

These are the same conventions as ``sklearn.feature_extraction.text.
TfidfVectorizer(smooth_idf=True)``, and the tests check that agreement
directly. The computation is done here rather than by handing raw text to
sklearn because the suite already has a canonical parse: reusing it means
TF-IDF sees exactly the tokens every other tool sees, including lemmas, which
a separate vectorizer's own tokenizer would silently disagree with.

Legacy note: ``statistics_corpus_tfidf_util.py`` wrote a dense
document-by-term matrix. That is quadratic in corpus size and unreadable past
a few dozen documents, so the output here is long-format -- one row per
(document, term) -- truncated to the top terms per document, which is the
shape the legacy tool's own "top terms" companion file used.
"""

from __future__ import annotations

from collections import Counter
import math

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["tfidf"]

_COLUMNS = [
    "Document ID",
    "Document",
    "Term",
    "Count",
    "Document Frequency",
    "TF",
    "IDF",
    "TF-IDF",
    "Rank",
]


def _document_terms(frame: pd.DataFrame, field_col: str, min_length: int) -> dict[object, Counter[str]]:
    """Term counts per document, in first-seen document order."""
    per_document: dict[object, Counter[str]] = {}
    for document_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        counts: Counter[str] = Counter()
        for value in group[field_col]:
            token = str(value).casefold()
            if token.isalpha() and len(token) >= min_length:
                counts[token] += 1
        per_document[document_id] = counts
    return per_document


def tfidf(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    top_n: int = 20,
    min_df: int = 1,
    max_df_ratio: float = 1.0,
    min_length: int = 1,
    sublinear_tf: bool = False,
    normalize: bool = True,
) -> Result[pd.DataFrame]:
    """Top TF-IDF terms per document, long format.

    Rows are sorted by document, then TF-IDF descending, then term
    alphabetically, so ties never reorder between runs (R6). ``Rank`` is the
    term's position within its own document, starting at 1.
    """
    if top_n < 1:
        return Result.failure(Diagnostic.error("TFIDF_BAD_TOP_N", f"top_n must be >= 1, got {top_n}"))
    if min_df < 1:
        return Result.failure(Diagnostic.error("TFIDF_BAD_MIN_DF", f"min_df must be >= 1, got {min_df}"))
    if not 0.0 < max_df_ratio <= 1.0:
        return Result.failure(
            Diagnostic.error("TFIDF_BAD_MAX_DF", f"max_df_ratio must be in (0, 1], got {max_df_ratio}")
        )
    if min_length < 1:
        return Result.failure(Diagnostic.error("TFIDF_BAD_MIN_LENGTH", f"min_length must be >= 1, got {min_length}"))
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("TFIDF_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_COLUMNS))

    per_document = _document_terms(frame, field.value, min_length)
    total_documents = len(per_document)
    if total_documents == 0:
        return Result.success(pd.DataFrame(columns=_COLUMNS))

    document_frequency: Counter[str] = Counter()
    for counts in per_document.values():
        document_frequency.update(counts.keys())

    max_df = max_df_ratio * total_documents
    vocabulary = {term: df for term, df in document_frequency.items() if df >= min_df and df <= max_df}
    if not vocabulary:
        return Result.success(
            pd.DataFrame(columns=_COLUMNS),
            Diagnostic.warning(
                "TFIDF_EMPTY_VOCABULARY",
                f"no term survived min_df={min_df} and max_df_ratio={max_df_ratio}",
            ),
        )

    names: dict[object, str] = {}
    if Col.DOCUMENT.value in frame.columns:
        names = frame.groupby(Col.DOCUMENT_ID.value)[Col.DOCUMENT.value].first().astype(str).to_dict()

    rows: list[dict[str, object]] = []
    for document_id, counts in per_document.items():
        weighted: list[tuple[str, int, float, float, float]] = []
        for term, count in counts.items():
            if term not in vocabulary:
                continue
            df = vocabulary[term]
            tf = 1.0 + math.log(count) if sublinear_tf else float(count)
            idf = math.log((1.0 + total_documents) / (1.0 + df)) + 1.0
            weighted.append((term, count, tf, idf, tf * idf))
        if not weighted:
            continue
        if normalize:
            norm = math.sqrt(sum(weight**2 for _t, _c, _tf, _idf, weight in weighted))
            if norm > 0:
                weighted = [(t, c, tf, idf, weight / norm) for t, c, tf, idf, weight in weighted]
        # Total order before ranking: score descending, then term ascending.
        weighted.sort(key=lambda item: (-item[4], item[0]))
        for rank, (term, count, tf, idf, weight) in enumerate(weighted[:top_n], start=1):
            rows.append(
                {
                    "Document ID": str(document_id),
                    "Document": names.get(document_id, ""),
                    "Term": term,
                    "Count": int(count),
                    "Document Frequency": int(vocabulary[term]),
                    "TF": round(tf, 6),
                    "IDF": round(idf, 6),
                    "TF-IDF": round(weight, 6),
                    "Rank": rank,
                }
            )
    out = pd.DataFrame(rows, columns=_COLUMNS)
    diagnostics: list[Diagnostic] = []
    if total_documents == 1:
        diagnostics.append(
            Diagnostic.warning(
                "TFIDF_SINGLE_DOCUMENT",
                "one document: every term has the same IDF, so the ranking is just term frequency",
            )
        )
    return Result.success(out, *diagnostics)
