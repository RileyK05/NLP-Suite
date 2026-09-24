"""N-gram co-occurrences — word pairs co-occurring within a window."""

from __future__ import annotations

import itertools
from collections import Counter

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["run"]


def run(
    frame: pd.DataFrame,
    *,
    window: int = 5,
    field: Col = Col.LEMMA,
    min_count: int = 1,
) -> Result[pd.DataFrame]:
    """Count unordered word pairs that co-occur within *window* per sentence.

    Two words co-occur if they appear in the same sentence within *window - 1*
    intervening positions (a window of 5 admits a gap of up to 4 tokens
    between the pair, matching the docstring's "within a window of 5").
    Positions are indexed over the ORIGINAL sentence token order — punctuation,
    digits and symbols occupy positions even though they can never be pair
    members — so the window keeps the user's units. Indexing over the filtered
    list would silently widen the window by the local punctuation density
    (review finding S6). Pairs are unordered ("a b" == "b a") and counted at
    most once per sentence.
    """
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("COOCC_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    for need in [field.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value]:
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("COOCC_MISSING_COLUMN", f"missing {need!r}"))
    if window < 2 or window > 20:
        return Result.failure(Diagnostic.error("COOCC_BAD_WINDOW", f"window must be 2..20, got {window}"))
    if min_count < 1:
        return Result.failure(Diagnostic.error("COOCC_BAD_MIN", f"min_count must be >=1, got {min_count}"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Word 1", "Word 2", "Count", "Document ID", "Document"]))

    pair_counts: Counter[tuple[tuple[str, str], str]] = Counter()

    for (_did, _sid), sent in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        # Positions over the ORIGINAL token order; only alphabetic tokens can
        # be pair members, but punctuation/digits still consume positions.
        positions: list[tuple[int, str]] = [
            (idx, str(x).lower())
            for idx, x in enumerate(sent[field.value].tolist())
            if str(x).strip() and str(x).isalpha()
        ]
        uniq = sorted({w for _, w in positions})
        if len(uniq) < 2:
            continue
        # Position map for the window check (original sentence coordinates).
        pos: dict[str, list[int]] = {}
        for idx, w in positions:
            pos.setdefault(w, []).append(idx)
        for w1, w2 in itertools.combinations(uniq, 2):
            # Any occurrence pair within the window (gap < window)? A window
            # of 5 admits up to 4 intervening tokens, hence strict <.
            found = False
            for p1 in pos[w1]:
                for p2 in pos[w2]:
                    if abs(p1 - p2) < window:
                        found = True
                        break
                if found:
                    break
            if found:
                pair_counts[((w1, w2), str(_did))] += 1

    filtered = [(pair, c) for pair, c in pair_counts.items() if c >= min_count]
    if not filtered:
        return Result.success(pd.DataFrame(columns=["Word 1", "Word 2", "Count", "Document ID", "Document"]))

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    names = (
        frame.groupby(Col.DOCUMENT_ID.value)[doc_col].first().astype(str).to_dict()
        if doc_col is not None and Col.DOCUMENT_ID.value in frame.columns
        else {}
    )
    filtered.sort(key=lambda x: (-x[1], x[0]))
    rows = [
        {
            "Word 1": w1,
            "Word 2": w2,
            "Count": int(c),
            "Document ID": did,
            "Document": names.get(did, ""),
        }
        for ((w1, w2), did), c in filtered
    ]
    df = pd.DataFrame(rows, columns=["Word 1", "Word 2", "Count", "Document ID", "Document"])
    return Result.success(df)
