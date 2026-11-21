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


def _texts(series: pd.Series) -> pd.Series:
    """The column as strings, with every missing value a blank.

    ``NaN``/``None``/``NaT`` used to become the literal words ``"nan"`` or
    ``"None"`` through ``str(cell)``, so searching for "nan" matched empty
    cells. A blank cell is blank. Conversion stays in pandas (one C pass)
    rather than a Python list of per-row ``str`` calls.
    """
    texts = series.astype(str)
    missing = series.isna()
    if bool(missing.any()):
        texts = texts.where(~missing, "")
    return texts


def _filter_mask(
    texts: pd.Series,
    folded: pd.Series | None,
    filt: SearchFilter,
    compiled: re.Pattern[str] | None,
) -> pd.Series:
    """One predicate as a boolean mask, vectorized over the column."""
    if filt.op == "regex":
        if compiled is None:  # pragma: no cover - search() validates every regex
            matched = pd.Series(False, index=texts.index)
        else:
            matched = texts.str.contains(compiled, na=False)
    else:
        haystack = folded if folded is not None else texts
        needle = filt.value if filt.case_sensitive else filt.value.lower()
        matched = pd.Series(False, index=texts.index)
        if filt.op == "eq":
            matched = haystack.eq(needle)
        elif filt.op == "contains":
            matched = haystack.str.contains(needle, regex=False, na=False)
        elif filt.op == "starts_with":
            matched = haystack.str.startswith(needle, na=False)
        elif filt.op == "ends_with":
            matched = haystack.str.endswith(needle, na=False)

    return ~matched if filt.negate else matched


def search(
    frame: pd.DataFrame,
    filters: list[SearchFilter],
    *,
    logic: Literal["AND", "OR"] = "AND",
) -> Result[SearchResult]:
    """Filter a CoNLL table by a list of predicates.

    Each filter inspects exactly one column. ``logic`` controls whether a row
    must satisfy every predicate (AND) or any predicate (OR). A missing cell
    (``None``/``NaN``/``NaT``) is blank; it never matches the text ``"None"``
    or ``"nan"``, though ``contains ""`` still matches it.
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
    compiled_patterns: list[re.Pattern[str] | None] = []
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
                compiled_patterns.append(re.compile(filt.value, flags=flags))
            except re.error as exc:
                return Result.failure(
                    Diagnostic.error(
                        "TABLE_SEARCH_BAD_REGEX", f"invalid regex {filt.value!r}: {exc}", value=filt.value
                    ),
                )
        else:
            compiled_patterns.append(None)

    if frame.empty:
        return Result.success(SearchResult(frame=frame.copy(), total_rows=0, matched_rows=0))

    # One C-level conversion per column, shared by every filter that reads it.
    # The old form built two Python lists and a list of booleans per filter,
    # then zipped them row by row: several full passes and one Series allocated
    # per row.
    text_columns: dict[str, pd.Series] = {}
    folded_columns: dict[str, pd.Series] = {}
    per_filter_masks: list[pd.Series] = []
    for filt, compiled in zip(filters, compiled_patterns, strict=True):
        texts = text_columns.get(filt.field)
        if texts is None:
            texts = _texts(frame[filt.field])
            text_columns[filt.field] = texts
        folded: pd.Series | None = None
        if filt.op != "regex" and not filt.case_sensitive:
            folded = folded_columns.get(filt.field)
            if folded is None:
                folded = texts.str.lower()
                folded_columns[filt.field] = folded
        per_filter_masks.append(_filter_mask(texts, folded, filt, compiled))

    matched_mask = per_filter_masks[0]
    for other in per_filter_masks[1:]:
        matched_mask = matched_mask & other if logic == "AND" else matched_mask | other

    matched = frame.loc[matched_mask].copy()
    return Result.success(
        SearchResult(frame=matched, total_rows=len(frame), matched_rows=len(matched)),
    )
