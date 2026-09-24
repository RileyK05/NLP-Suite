"""N-grams and collocation statistics."""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["collocation_network", "collocations", "ngrams"]


def ngrams(
    frame: pd.DataFrame,
    *,
    n: int = 2,
    field: Col = Col.FORM,
) -> Result[pd.DataFrame]:
    """Extract n-grams per sentence."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("NGRAM_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    for need in [field.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value]:
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("NGRAM_MISSING_COLUMN", f"missing {need!r}"))
    if n < 1 or n > 5:
        return Result.failure(Diagnostic.error("NGRAM_BAD_N", f"n must be 1..5, got {n}"))
    if frame.empty:
        return Result.success(
            pd.DataFrame(columns=["N-gram", "Frequency in Document", "Frequency in Corpus", "Document ID", "Document"])
        )

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None

    # Legacy parity (src/NGrams_util.py:143-155): counts are per (document,
    # n-gram) with the corpus total alongside, so document-level comparisons
    # and attribution are valid. A corpus-wide count claiming one document's
    # identity supports neither.
    doc_counts: Counter[tuple[tuple[str, ...], str]] = Counter()
    corpus_counts: Counter[tuple[str, ...]] = Counter()

    for (_did, _sid), sent in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        toks = [str(x).lower() for x in sent[field.value].tolist() if str(x).strip() and str(x).isalpha()]
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            ng = tuple(toks[i : i + n])
            doc_counts[(ng, str(_did))] += 1
            corpus_counts[ng] += 1

    if not corpus_counts:
        return Result.success(
            pd.DataFrame(columns=["N-gram", "Frequency in Document", "Frequency in Corpus", "Document ID", "Document"])
        )

    rows = [
        {
            "N-gram": " ".join(ng),
            "Frequency in Document": int(c),
            "Frequency in Corpus": int(corpus_counts[ng]),
            "Document ID": did,
            "Document": "",
        }
        for (ng, did), c in sorted(doc_counts.items(), key=lambda kv: (-corpus_counts[kv[0][0]], kv[0][0], kv[0][1]))
    ]
    df = pd.DataFrame(
        rows, columns=["N-gram", "Frequency in Document", "Frequency in Corpus", "Document ID", "Document"]
    )
    if doc_col is not None and Col.DOCUMENT_ID.value in frame.columns:
        names = frame.groupby(Col.DOCUMENT_ID.value)[doc_col].first().astype(str).to_dict()
        names = {str(key): value for key, value in names.items()}
        df["Document"] = df["Document ID"].map(names).fillna("")
    return Result.success(df)


def collocations(
    frame: pd.DataFrame,
    *,
    field: Col = Col.FORM,
    min_count: int = 2,
) -> Result[pd.DataFrame]:
    """Bigram collocations via pointwise mutual information."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("COLLOC_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Word 1", "Word 2", "Count", "PMI"]))

    # Build unigram and bigram counts
    uni = Counter[str]()
    bi = Counter[tuple[str, str]]()
    total_bigrams = 0

    for (_did, _sid), sent in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        toks = [str(x).lower() for x in sent[field.value].tolist() if str(x).strip() and str(x).isalpha()]
        for w in toks:
            uni[w] += 1
        for i in range(len(toks) - 1):
            bi[(toks[i], toks[i + 1])] += 1
            total_bigrams += 1

    total_unigrams = sum(uni.values()) or 1
    rows: list[dict[str, object]] = []
    for (w1, w2), c in bi.items():
        if c < min_count:
            continue
        p_w1 = uni[w1] / total_unigrams
        p_w2 = uni[w2] / total_unigrams
        p_bi = c / total_bigrams if total_bigrams else 0
        if p_w1 == 0 or p_w2 == 0 or p_bi == 0:
            continue
        pmi = math.log2(p_bi / (p_w1 * p_w2))
        rows.append({"Word 1": w1, "Word 2": w2, "Count": int(c), "PMI": round(float(pmi), 4)})

    if not rows:
        return Result.success(pd.DataFrame(columns=["Word 1", "Word 2", "Count", "PMI"]))
    df = pd.DataFrame(rows, columns=["Word 1", "Word 2", "Count", "PMI"])
    df = df.sort_values("PMI", ascending=False).reset_index(drop=True)
    return Result.success(df)


def collocation_network(
    frame: pd.DataFrame,
    *,
    field: Col = Col.FORM,
    min_count: int = 2,
    min_pmi: float = 0.0,
    top_edges: int = 200,
) -> Result[pd.DataFrame]:
    """Undirected edge list for the collocation network (Word 1, Word 2, Weight).

    Same bigram counts and PMI as :func:`collocations`; edges below
    *min_pmi* are dropped and only the strongest *top_edges* survive, so the
    GEXF stays readable in Gephi instead of a hairball. Word order within a
    pair is canonicalized (lexically smaller first) so ``dog cat`` and
    ``cat dog`` share one edge.
    """
    if isinstance(top_edges, bool) or not isinstance(top_edges, int) or top_edges < 1:
        return Result.failure(
            Diagnostic.error("NET_BAD_TOP_EDGES", f"top-edges must be a positive int, got {top_edges!r}")
        )
    result = collocations(frame, field=field, min_count=min_count)
    if result.value is None:
        return Result[pd.DataFrame](None, result.diagnostics)
    coll = result.unwrap()
    if coll.empty:
        return Result.success(pd.DataFrame(columns=["Word 1", "Word 2", "Weight", "PMI"]), *result.diagnostics)
    strong = coll[coll["PMI"] >= min_pmi]
    if strong.empty:
        return Result.success(
            pd.DataFrame(columns=["Word 1", "Word 2", "Weight", "PMI"]),
            Diagnostic.warning(
                "NET_NO_EDGES",
                f"no collocation has PMI >= {min_pmi}; lower --min-pmi (e.g. 0) to keep all bigrams at the count threshold",
            ),
        )
    canon: list[dict[str, object]] = []
    for _, row in strong.iterrows():
        a, b = str(row["Word 1"]), str(row["Word 2"])
        if b < a:
            a, b = b, a
        canon.append({"Word 1": a, "Word 2": b, "Count": int(row["Count"]), "PMI": float(row["PMI"])})
    edges = pd.DataFrame(canon, columns=["Word 1", "Word 2", "Count", "PMI"])
    edges = edges.groupby(["Word 1", "Word 2"], as_index=False).agg(Count=("Count", "sum"), PMI=("PMI", "max"))
    edges = edges.sort_values("PMI", ascending=False).head(top_edges)
    edges = edges.sort_values(["PMI", "Count"], ascending=[False, False]).reset_index(drop=True)
    out = pd.DataFrame(
        {
            "Word 1": edges["Word 1"],
            "Word 2": edges["Word 2"],
            "Weight": edges["Count"].astype(float),
            "PMI": edges["PMI"].round(4),
        }
    )
    diags = list(result.diagnostics)
    if len(strong) > top_edges:
        diags.append(
            Diagnostic.warning(
                "NET_EDGES_TRUNCATED",
                f"kept the top {top_edges} of {len(strong)} edges by PMI; raise --top-edges for a denser graph",
            )
        )
    return Result.success(out, *diags)
