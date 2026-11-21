"""Vectorized column and sentence extraction helpers for canonical CoNLL tables (R9)."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from core.conll.schema import Col

__all__ = [
    "col_int_list",
    "col_str_list",
    "iter_sentence_groups",
    "sentence_indices",
]


def col_str_list(frame: pd.DataFrame, col: Col | str, default: str = "") -> list[str]:
    """Extract a column as Python strings in C-speed without per-row Series allocation."""
    name = col.value if isinstance(col, Col) else col
    if name not in frame.columns or frame.empty:
        return []
    series = frame[name]
    if series.isna().any():
        filled: list[str] = [str(v) for v in series.fillna(default).tolist()]
        return filled
    values: list[str] = [str(v) for v in series.tolist()]
    return values


def col_int_list(frame: pd.DataFrame, col: Col | str, default: int = 0) -> list[int]:
    """Extract an integer-valued column cleanly without per-row casting."""
    name = col.value if isinstance(col, Col) else col
    if name not in frame.columns or frame.empty:
        return []
    numeric = pd.to_numeric(frame[name], errors="coerce")
    if numeric.isna().any():
        numeric = numeric.fillna(default)
    result: list[int] = [int(v) for v in numeric.tolist()]
    return result


def sentence_indices(frame: pd.DataFrame) -> dict[tuple[str, str], list[int]]:
    """Map ``(Document ID, Sentence ID)`` -> ordered row positions in O(n)."""
    if frame.empty:
        return {}
    doc_ids = col_str_list(frame, Col.DOCUMENT_ID, "") if Col.DOCUMENT_ID.value in frame.columns else [""] * len(frame)
    sent_ids = (
        col_str_list(frame, Col.SENTENCE_ID, "") if Col.SENTENCE_ID.value in frame.columns else ["1"] * len(frame)
    )
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, (doc_id, sent_id) in enumerate(zip(doc_ids, sent_ids, strict=True)):
        key = (doc_id, sent_id)
        bucket = groups.get(key)
        if bucket is None:
            groups[key] = [idx]
        else:
            bucket.append(idx)
    return groups


def iter_sentence_groups(frame: pd.DataFrame) -> Iterator[tuple[str, str, list[int]]]:
    """Yield ``(doc_id, sent_id, row_indices)`` preserving first-appearance sentence order."""
    for (doc_id, sent_id), rows in sentence_indices(frame).items():
        yield doc_id, sent_id, rows
