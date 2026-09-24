"""CoNLL table search — filter rows with composable predicates."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

import pandas as pd

from core.conll.schema import validate_columns
from core.result import Diagnostic, Result

__all__ = ["SearchFilter", "SearchOp", "SearchResult", "search"]

SearchOp = Literal["eq", "contains", "starts_with", "ends_with", "regex"]


@dataclass(frozen=True, slots=True)
class SearchFilter:
    """One predicate over a single column."""

    field: str
    op: SearchOp = "eq"
    value: str = ""
    case_sensitive: bool = False
    negate: bool = False


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The filtered view and its counts."""

    frame: pd.DataFrame
    total_rows: int
    matched_rows: int

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def _cell_matches(cell: object, filt: SearchFilter) -> bool:
    text = str(cell) if cell is not None else ""
    needle = filt.value
    if not filt.case_sensitive:
        text = text.lower()
        needle = needle.lower()

    if filt.op == "eq":
        matched = text == needle
    elif filt.op == "contains":
        matched = needle in text
    elif filt.op == "starts_with":
        matched = text.startswith(needle)
    elif filt.op == "ends_with":
        matched = text.endswith(needle)
    elif filt.op == "regex":
        try:
            flags = 0 if filt.case_sensitive else re.IGNORECASE
            matched = re.search(needle, str(cell) if cell is not None else "", flags=flags) is not None
        except re.error:
            matched = False
    else:
        matched = False  # type: ignore[unreachable]

    return not matched if filt.negate else matched


def search(
    frame: pd.DataFrame,
    filters: list[SearchFilter],
    *,
    logic: Literal["AND", "OR"] = "AND",
) -> Result[SearchResult]:
    """Filter a CoNLL table by a list of predicates.

    Each filter inspects exactly one column. ``logic`` controls whether a row
    must satisfy every predicate (AND) or any predicate (OR).
    """
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[SearchResult](None, checked.diagnostics)

    if not filters:
        return Result.failure(
            Diagnostic.error("TABLE_SEARCH_NO_FILTERS", "at least one filter is required"),
        )

    if logic not in ("AND", "OR"):
        return Result.failure(
            Diagnostic.error("TABLE_SEARCH_BAD_LOGIC", f"logic must be AND or OR, got {logic!r}", logic=logic),
        )

    # Validate fields and ops up front so the caller gets a diagnostic, not a KeyError.
    allowed_ops: set[str] = {"eq", "contains", "starts_with", "ends_with", "regex"}
    for filt in filters:
        if filt.field not in frame.columns:
            return Result.failure(
                Diagnostic.error(
                    "TABLE_SEARCH_BAD_FIELD",
                    f"unknown column {filt.field!r}",
                    field=filt.field,
                    columns=list(frame.columns),
                ),
            )
        if filt.op not in allowed_ops:
            return Result.failure(
                Diagnostic.error("TABLE_SEARCH_BAD_OP", f"unknown op {filt.op!r}", op=filt.op),
            )
        if filt.op == "regex":
            try:
                flags = 0 if filt.case_sensitive else re.IGNORECASE
                re.compile(filt.value, flags=flags)
            except re.error as exc:
                return Result.failure(
                    Diagnostic.error(
                        "TABLE_SEARCH_BAD_REGEX", f"invalid regex {filt.value!r}: {exc}", value=filt.value
                    ),
                )

    if frame.empty:
        return Result.success(SearchResult(frame=frame.copy(), total_rows=0, matched_rows=0))

    # Row-wise predicate evaluation — small tables, and keeps string semantics explicit.
    matched_mask: list[bool] = []
    for _, row in frame.iterrows():
        results = [_cell_matches(row[filt.field], filt) for filt in filters]
        matched_mask.append(all(results) if logic == "AND" else any(results))

    matched = frame.loc[matched_mask].copy()
    return Result.success(
        SearchResult(frame=matched, total_rows=len(frame), matched_rows=len(matched)),
    )
