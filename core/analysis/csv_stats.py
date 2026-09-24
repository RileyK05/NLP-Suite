"""CSV statistics — per-column summaries and simple tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["CsvStatsResult", "correlation", "correlation_matrix", "describe", "rank_correlation"]


@dataclass(frozen=True, slots=True)
class CsvStatsResult:
    frame: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def _numeric_summary(series: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "Count": 0,
            "Mean": 0.0,
            "Std": 0.0,
            "Min": 0.0,
            "Q1": 0.0,
            "Median": 0.0,
            "Q3": 0.0,
            "Max": 0.0,
        }
    return {
        "Count": int(s.count()),
        "Mean": float(s.mean()),
        "Std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "Min": float(s.min()),
        "Q1": float(s.quantile(0.25)),
        "Median": float(s.median()),
        "Q3": float(s.quantile(0.75)),
        "Max": float(s.max()),
    }


def describe(
    frame: pd.DataFrame,
    *,
    group_by: str | None = None,
) -> Result[CsvStatsResult]:
    """Summarize numeric columns, optionally per group."""
    if frame.empty:
        empty = pd.DataFrame(columns=["Group", "Column", "Count", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"])
        return Result.success(CsvStatsResult(frame=empty))

    if group_by is not None and group_by not in frame.columns:
        return Result.failure(
            Diagnostic.error("CSV_STATS_BAD_GROUP", f"group_by column {group_by!r} not in frame", column=group_by)
        )

    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    # Also treat columns that can be coerced to numeric with at least one numeric value.
    if not numeric_cols:
        for col in frame.columns:
            if col == group_by:
                continue
            coerced = pd.to_numeric(frame[col], errors="coerce")
            if coerced.notna().any():
                numeric_cols.append(col)

    if not numeric_cols:
        return Result.failure(
            Diagnostic.error("CSV_STATS_NO_NUMERIC", "frame has no numeric columns to summarize"),
        )

    rows: list[dict[str, object]] = []

    if group_by is None:
        for col in numeric_cols:
            stats = _numeric_summary(frame[col])
            rows.append({"Group": "all", "Column": col, **stats})
    else:
        for grp, sub in frame.groupby(group_by, sort=False, dropna=False):
            grp_label = str(grp)
            for col in numeric_cols:
                stats = _numeric_summary(sub[col])
                rows.append({"Group": grp_label, "Column": col, **stats})

    out = pd.DataFrame(rows, columns=["Group", "Column", "Count", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"])
    # Round floats for stable CSV output (6 decimals like envelopes).
    for c in ["Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]:
        out[c] = out[c].astype(float).round(6)
    return Result.success(CsvStatsResult(frame=out))


def correlation(
    frame: pd.DataFrame,
    col_x: str,
    col_y: str,
) -> Result[float]:
    """Pearson correlation between two numeric columns."""
    if col_x not in frame.columns or col_y not in frame.columns:
        return Result.failure(
            Diagnostic.error(
                "CSV_CORR_BAD_COLUMN", f"columns {col_x!r}, {col_y!r} must be in frame", columns=list(frame.columns)
            ),
        )
    sx = pd.to_numeric(frame[col_x], errors="coerce").dropna()
    sy = pd.to_numeric(frame[col_y], errors="coerce").dropna()
    # Align on original index intersection
    idx = sx.index.intersection(sy.index)
    sx = sx.loc[idx]
    sy = sy.loc[idx]
    if len(sx) < 2:
        return Result.failure(
            Diagnostic.error("CSV_CORR_TOO_FEW", "need at least 2 paired observations", n=len(sx)),
        )
    mx = float(sx.mean())
    my = float(sy.mean())
    num = float(((sx - mx) * (sy - my)).sum())
    den_x = math.sqrt(float(((sx - mx) ** 2).sum()))
    den_y = math.sqrt(float(((sy - my) ** 2).sum()))
    den = den_x * den_y
    if den == 0:
        return Result.success(0.0)
    return Result.success(round(num / den, 6))


def rank_correlation(
    frame: pd.DataFrame,
    col_x: str,
    col_y: str,
    method: str = "spearman",
) -> Result[float]:
    """Spearman or Kendall rank correlation (FR-2.5; Pearson lives in ``correlation``).

    Additive to this module: existing outputs are untouched.
    """
    method_lower = method.lower()
    if method_lower not in ("spearman", "kendall"):
        return Result.failure(
            Diagnostic.error(
                "CSV_CORR_BAD_METHOD", f"method must be 'spearman' or 'kendall', got {method!r}", method=method
            ),
        )
    if col_x not in frame.columns or col_y not in frame.columns:
        return Result.failure(
            Diagnostic.error(
                "CSV_CORR_BAD_COLUMN", f"columns {col_x!r}, {col_y!r} must be in frame", columns=list(frame.columns)
            ),
        )
    sub = frame[[col_x, col_y]].copy()
    for col in (col_x, col_y):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 5:
        return Result.failure(
            Diagnostic.error("CSV_CORR_TOO_FEW", "need at least 5 paired observations", n=len(sub)),
        )
    if float(sub[col_x].std()) == 0 or float(sub[col_y].std()) == 0:
        return Result.failure(
            Diagnostic.error("CSV_CORR_CONSTANT", "one column has zero variance; correlation is undefined"),
        )
    # Local import: scipy is a core dep but stays out of this module's import cost.
    from scipy import stats as scipy_stats

    if method_lower == "kendall":
        coef, _ = scipy_stats.kendalltau(sub[col_x], sub[col_y])
    else:
        coef, _ = scipy_stats.spearmanr(sub[col_x], sub[col_y])
    return Result.success(round(float(coef), 6))


def correlation_matrix(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> Result[pd.DataFrame]:
    """Every pairwise correlation between numeric columns, as one matrix.

    The long-format pair output (``correlation`` per chosen pair) cannot draw
    a heatmap: the panel needs the full square. Rows and columns are the
    same ordered category list, the diagonal is 1.0 by construction and is
    still written so the matrix can be checked by eye.

    ``method`` is pearson, or spearman/kendall by rank (FR-2.5's measures
    over more than one pair). Columns that are constant, or that fewer than
    two rows have values for, are refused with the column named rather than
    silently filling the matrix with NaN.
    """
    method_lower = method.lower()
    if method_lower not in ("pearson", "spearman", "kendall"):
        return Result.failure(
            Diagnostic.error(
                "CSV_CORR_BAD_METHOD",
                f"method must be 'pearson', 'spearman' or 'kendall', got {method!r}",
                method=method,
            ),
        )
    numeric = frame.select_dtypes(include="number")
    if columns is not None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            return Result.failure(
                Diagnostic.error(
                    "CSV_CORR_BAD_COLUMN", f"column(s) not in frame: {missing}", columns=list(frame.columns)
                )
            )
        numeric = frame[columns]
    chosen = list(numeric.columns)
    if len(chosen) < 2:
        return Result.failure(
            Diagnostic.error(
                "CSV_CORR_TOO_FEW_COLUMNS",
                "a correlation matrix needs at least 2 numeric columns",
                numeric_columns=chosen,
            )
        )
    working = numeric.apply(pd.to_numeric, errors="coerce")
    degenerate = [column for column in chosen if working[column].dropna().nunique() < 2]
    if degenerate:
        return Result.failure(
            Diagnostic.error(
                "CSV_CORR_CONSTANT",
                f"column(s) {degenerate} have no variation; correlation with them is undefined",
                columns=degenerate,
            )
        )
    if method_lower == "pearson":
        matrix = working.corr(method="pearson")
    else:
        matrix = working.corr(method=method_lower)
    matrix = matrix.round(6)
    matrix.index.name = "Column"
    matrix.columns.name = ""
    return Result.success(matrix.reset_index())
