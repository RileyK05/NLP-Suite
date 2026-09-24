"""What kind of thing is each column?

Every recommendation and every readout rests on this, so the roles are
deliberately few and the thresholds are explicit constants rather than
judgement calls buried in branches. A column gets exactly one role.

The roles carry chart consequences, which is the whole reason they exist:

* ``CONSTANT`` -- one distinct value. Never an axis: there is nothing to
  compare.
* ``IDENTIFIER`` -- (near) unique per row, or a name that says so. Never a
  category axis: "every row is its own bar" is a list, not a comparison.
* ``CATEGORICAL`` -- a modest number of repeated labels. The natural x for
  bar, pie and grouping.
* ``COUNT`` -- non-negative integers. Sums meaningfully; the natural y for a
  bar chart.
* ``PROPORTION`` -- floats inside [0, 1]. Averages meaningfully, sums do not.
* ``SCORE`` -- any other numeric. Averages; may be negative or unbounded.
* ``TEXT`` -- long or highly varied free text. Not an axis at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

import pandas as pd

__all__ = ["ColumnProfile", "ColumnRole", "profile_column", "profile_frame"]

# A column is "identifier-like" when nearly every row has its own value.
_UNIQUE_RATIO_FOR_IDENTIFIER = 0.95
# Above this many distinct labels a categorical axis stops being readable.
_MAX_READABLE_CATEGORIES = 50
# Strings averaging longer than this are prose, not labels.
_TEXT_MEAN_LENGTH = 40
# Column names that are identifiers whatever their distribution looks like.
_IDENTIFIER_NAMES = ("id", "rank", "index", "key", "row")
# A CSV carries no dtypes, so a date read back from a published table is the
# string "2024-03-07". Without this every date column was an IDENTIFIER (one
# per row) or a CATEGORICAL, and the timeline recommendation -- which asks for
# a DATE column and nothing else -- could never fire on a file.
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Years a corpus of documents could plausibly span. Wide on purpose: the job
# is to separate a year column from a count column, not to police history.
_EARLIEST_YEAR, _LATEST_YEAR = 1000, 2999


class ColumnRole(Enum):
    """The one role a column plays when charting or describing a table."""

    CONSTANT = "constant"
    DATE = "date"
    IDENTIFIER = "identifier"
    CATEGORICAL = "categorical"
    COUNT = "count"
    PROPORTION = "proportion"
    SCORE = "score"
    TEXT = "text"
    EMPTY = "empty"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """One column's role and the facts a readout or chart rule needs."""

    name: str
    role: ColumnRole
    distinct: int
    non_null: int
    minimum: float | None = None
    maximum: float | None = None

    @property
    def is_numeric(self) -> bool:
        return self.role in (ColumnRole.COUNT, ColumnRole.PROPORTION, ColumnRole.SCORE)

    @property
    def is_axis_candidate(self) -> bool:
        """Usable as a chart axis at all."""
        return self.role not in (ColumnRole.CONSTANT, ColumnRole.EMPTY, ColumnRole.TEXT)


def _looks_like_identifier_name(name: str) -> bool:
    """``Document ID`` and ``Rank`` are identifiers even when they repeat."""
    words = name.casefold().replace("-", " ").replace("_", " ").split()
    return any(word in _IDENTIFIER_NAMES for word in words)


def _numeric_role(values: pd.Series, name: str, rows: int, distinct: int) -> ColumnRole:
    """Which of the four numeric roles a column of numbers plays.

    Order matters and is not arbitrary. Both an identifier and a year are
    columns of integers that a count rule would happily claim, and claiming
    either of them puts a meaningless quantity on the measure axis: a chart
    ranking rows by Record ID, or by year descending, which is a timeline
    drawn backwards.
    """
    low, high = float(values.min()), float(values.max())
    # A near-unique integer column named like a key is an identifier, not a
    # measurement: ranking rows by "Record ID" charts nothing.
    if _looks_like_identifier_name(name) and (
        distinct / max(rows, 1) >= _UNIQUE_RATIO_FOR_IDENTIFIER or _is_integral(values)
    ):
        return ColumnRole.IDENTIFIER
    # Summing years produces a number about nothing. A year is where a row
    # sits, not what it is worth, so it belongs on an axis, in order.
    if _looks_like_year(name) and _is_integral(values) and low >= _EARLIEST_YEAR and high <= _LATEST_YEAR:
        return ColumnRole.DATE
    if _is_integral(values) and low >= 0:
        return ColumnRole.COUNT
    if low >= 0.0 and high <= 1.0:
        return ColumnRole.PROPORTION
    return ColumnRole.SCORE


def _other_role(values: pd.Series, name: str, rows: int, distinct: int) -> ColumnRole:
    """The role of a column that is not numbers: a date, a label, or prose.

    The ISO-string rule is the one that had to be added. A CSV carries no
    dtypes, so a date written by any tool here and read back by pandas is the
    string "2024-03-07" -- which every other rule reads as a label that
    happens to be unique per row. The timeline recommendation asks for a DATE
    column and nothing else, so without this it could never fire on a file.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return ColumnRole.DATE
    text = values.astype(str)
    if bool(text.str.match(_ISO_DAY).all()):
        return ColumnRole.DATE
    mean_length = float(text.str.len().mean())
    if _looks_like_identifier_name(name) or distinct / max(rows, 1) >= _UNIQUE_RATIO_FOR_IDENTIFIER:
        return ColumnRole.IDENTIFIER if mean_length <= _TEXT_MEAN_LENGTH else ColumnRole.TEXT
    if mean_length > _TEXT_MEAN_LENGTH:
        return ColumnRole.TEXT
    return ColumnRole.CATEGORICAL


def _looks_like_year(name: str) -> bool:
    """``Year``, ``Publication Year`` -- a column of years, by its name."""
    words = name.casefold().replace("-", " ").replace("_", " ").split()
    return bool(words) and words[-1] == "year"


def profile_column(series: pd.Series, name: str, rows: int) -> ColumnProfile:
    """Classify one column. Never raises: an unreadable column becomes TEXT."""
    values = series.dropna()
    non_null = len(values)
    if non_null == 0:
        return ColumnProfile(name=name, role=ColumnRole.EMPTY, distinct=0, non_null=0)
    try:
        distinct = int(values.nunique())
    except TypeError:  # unhashable cell contents
        return ColumnProfile(name=name, role=ColumnRole.TEXT, distinct=non_null, non_null=non_null)
    if distinct == 1:
        return ColumnProfile(name=name, role=ColumnRole.CONSTANT, distinct=1, non_null=non_null)

    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return ColumnProfile(
            name=name,
            role=_numeric_role(values, name, rows, distinct),
            distinct=distinct,
            non_null=non_null,
            minimum=float(values.min()),
            maximum=float(values.max()),
        )
    role = _other_role(values, name, rows, distinct)
    return ColumnProfile(name=name, role=role, distinct=distinct, non_null=non_null)


def _is_integral(values: pd.Series) -> bool:
    """Integer dtype, or floats that all happen to be whole numbers."""
    if pd.api.types.is_integer_dtype(values):
        return True
    try:
        return bool((values % 1 == 0).all())
    except TypeError:
        return False


def profile_frame(frame: pd.DataFrame) -> dict[str, ColumnProfile]:
    """Profile every column, in column order."""
    rows = len(frame)
    return {str(name): profile_column(frame[name], str(name), rows) for name in frame.columns}


def readable_categories(profile: ColumnProfile) -> bool:
    """Few enough distinct labels to put on an axis without a top-N cut."""
    return profile.role is ColumnRole.CATEGORICAL and profile.distinct <= _MAX_READABLE_CATEGORIES
