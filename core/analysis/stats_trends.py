"""Trend statistics (FR-2.5) — Mann-Kendall + Sen's slope, rank-correlation tables.

Parity basis: ``statistics_statistical_tests_util.py`` (legacy) — the
Mann-Kendall and Spearman/Kendall runners. The S loop, tie-corrected
variance, continuity-corrected Z, Sen's slope/intercept, trend labels, and
the Fisher-z CI (Spearman, n > 3) are ported verbatim.

Intentional differences from legacy: messageboxes become ERROR diagnostics;
charts and the automatic-test coupling are out of scope (FR-6.5, FR-2.3+).
The rank coefficient itself lives in ``csv_stats.rank_correlation`` (additive);
this module owns the Mann-Kendall test and the correlation summary table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.analysis.csv_stats import rank_correlation
from core.result import Diagnostic, Result

__all__ = [
    "MannKendallResult",
    "correlation_summary",
    "interpret_coefficient",
    "mann_kendall",
]

_MK_MIN_N = 8

_COEF_BANDS: tuple[tuple[float, str], ...] = (
    (0.1, "negligible"),
    (0.3, "weak"),
    (0.5, "moderate"),
    (0.7, "strong"),
    (float("inf"), "very strong"),
)


@dataclass(frozen=True, slots=True)
class MannKendallResult:
    summary: pd.DataFrame
    trend: pd.DataFrame  # date, value, trend line


def interpret_coefficient(abs_coef: float) -> str:
    for threshold, label in _COEF_BANDS:
        if abs_coef < threshold:
            return label
    return "very strong"  # unreachable; keeps mypy total


def correlation_summary(
    frame: pd.DataFrame, col_x: str, col_y: str, method: str = "spearman", *, alpha: float = 0.05
) -> Result[pd.DataFrame]:
    """Legacy-shaped summary table for a rank correlation with Fisher-z CI."""
    coef_result = rank_correlation(frame, col_x, col_y, method)
    if coef_result.value is None:
        return Result.failure(*coef_result.diagnostics)
    coef = coef_result.unwrap()
    sub = frame[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(sub)
    if method.lower() == "kendall":
        _, p_value = scipy_stats.kendalltau(sub[col_x], sub[col_y])
        label = "Kendall"
    else:
        _, p_value = scipy_stats.spearmanr(sub[col_x], sub[col_y])
        label = "Spearman"
    ci_lower: object = ""
    ci_upper: object = ""
    ci_note: object = ""
    if method.lower() == "spearman" and n > 3:
        # C6-6: Fisher-z on Spearman is an APPROXIMATION (the transform is
        # exact for Pearson). The summary says so instead of implying exactness.
        if abs(coef) >= 1.0:
            # |r| = 1: the Fisher interval degenerates; report unbounded, honestly.
            ci_lower = ""
            ci_upper = ""
            ci_note = "undefined at |r|=1 (Fisher-z approximation, exact for Pearson only)"
        else:
            z = float(np.arctanh(coef))
            se = 1.0 / math.sqrt(n - 3)
            ci_lower = round(float(np.tanh(z - 1.96 * se)), 4)
            ci_upper = round(float(np.tanh(z + 1.96 * se)), 4)
            ci_note = "Fisher-z approximation (exact for Pearson only)"
    summary = pd.DataFrame(
        {
            "Statistic": [
                "Method",
                "Coefficient",
                "p-value",
                "95% CI lower",
                "95% CI upper",
                "CI method",
                "Interpretation",
                f"Significant (alpha={alpha:g})",
                "N",
            ],
            "Value": [
                label,
                round(coef, 4),
                round(float(p_value), 6),
                ci_lower,
                ci_upper,
                ci_note,
                interpret_coefficient(abs(coef)),
                "Yes" if p_value < alpha else "No",
                n,
            ],
        }
    )
    return Result.success(summary)


def _mann_kendall_stats(values: np.ndarray, alpha: float) -> dict[str, float | str]:
    x = np.asarray(values, dtype=float)
    n = len(x)
    stat_s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = x[j] - x[k]
            if diff > 0:
                stat_s += 1
            elif diff < 0:
                stat_s -= 1
    _, tie_counts = np.unique(x, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    for tied in tie_counts[tie_counts > 1]:
        var_s -= (tied * (tied - 1) * (2 * tied + 5)) / 18.0
    if stat_s > 0:
        z = (stat_s - 1) / math.sqrt(var_s) if var_s > 0 else 0.0
    elif stat_s < 0:
        z = (stat_s + 1) / math.sqrt(var_s) if var_s > 0 else 0.0
    else:
        z = 0.0
    p_value = 2.0 * float(scipy_stats.norm.sf(abs(z)))
    tau = stat_s / (n * (n - 1) / 2.0)
    slopes = [(x[j] - x[k]) / (j - k) for k in range(n - 1) for j in range(k + 1, n)]
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(x - slope * np.arange(n)))
    if p_value < alpha:
        trend: str = "increasing" if tau > 0 else "decreasing"
    else:
        trend = "no significant trend"
    return {"tau": tau, "p_value": p_value, "slope": slope, "intercept": intercept, "trend": trend, "z": z, "s": stat_s}


def mann_kendall(
    frame: pd.DataFrame, date_col: str, value_col: str, *, alpha: float = 0.05
) -> Result[MannKendallResult]:
    """Mann-Kendall trend test over a date column (sorted ascending first).

    C6-6: alpha is validated and drives BOTH the significance flag and the
    trend label (never "increasing" while "not significant"). Dates: rows
    that fail to parse are DROPPED with counts + MK_DROPPED_DATES (never
    silently included or excluded); duplicate dates are kept and treated as
    separate observations by the S statistic (documented). Sen's slope is
    per OBSERVATION INDEX — stated in the summary; the trend line uses the
    same x-axis semantics.
    """
    if not 0 < alpha < 1:
        return Result.failure(Diagnostic.error("STATS_BAD_ALPHA", f"alpha must be in (0, 1), got {alpha}", alpha=alpha))
    if date_col not in frame.columns or value_col not in frame.columns:
        missing = [c for c in (date_col, value_col) if c not in frame.columns]
        return Result.failure(
            Diagnostic.error("STATS_BAD_COLUMN", f"column(s) not in frame: {missing}", missing=missing)
        )
    sub = frame[[date_col, value_col]].copy()
    raw_values = sub[value_col]
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    # C6-6: NaN/infinity present BEFORE coercion is invalid input (rejected);
    # NaN appearing only after coercion is unparseable text (missing, dropped
    # with the existing STATS_NO_NUMERIC path when nothing remains).
    if bool(
        raw_values.apply(
            lambda v: not math.isfinite(v) if isinstance(v, (int, float)) and not pd.isna(v) else bool(pd.isna(v))
        ).any()
    ):
        return Result.failure(
            Diagnostic.error("STATS_BAD_FREQUENCY", "value column contains NaN or infinity", column=value_col)
        )
    sub = sub.dropna()
    if sub.empty:
        return Result.failure(Diagnostic.error("STATS_NO_NUMERIC", f"{value_col!r} has no numeric values"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # C6-6: per-element parse so PARTIALLY invalid date columns hit the
        # drop-with-warning path instead of failing wholesale.
        parsed = pd.to_datetime(sub[date_col], errors="coerce")
    if parsed is None or bool(parsed.isna().all()):
        return Result.failure(
            Diagnostic.error("MK_BAD_DATES", f"could not parse dates in {date_col!r}", column=date_col)
        )
    dropped = int(parsed.isna().sum())
    if dropped:
        sub = sub[~parsed.isna()]
        parsed = parsed.dropna()
    sub[date_col] = parsed
    diags: list[Diagnostic] = []
    duplicate_dates = int(sub[date_col].duplicated().sum())
    if duplicate_dates:
        diags.append(
            Diagnostic.warning(
                "MK_DUPLICATE_DATES",
                f"{duplicate_dates} duplicate date(s) kept as separate observations",
                count=duplicate_dates,
            )
        )
    if dropped:
        diags.append(
            Diagnostic.warning(
                "MK_DROPPED_DATES",
                f"{dropped} row(s) dropped: dates unparseable",
                count=dropped,
            )
        )
    sub = sub.sort_values(date_col).reset_index(drop=True)
    if len(sub) < _MK_MIN_N:
        return Result.failure(
            Diagnostic.error("STATS_TOO_FEW_ROWS", "need at least 8 observations for Mann-Kendall", n=len(sub))
        )
    computed = _mann_kendall_stats(sub[value_col].values, alpha)
    tau = float(computed["tau"])
    p_value = float(computed["p_value"])
    slope = float(computed["slope"])
    intercept = float(computed["intercept"])
    summary = pd.DataFrame(
        {
            "Statistic": [
                "Kendall's tau",
                "p-value",
                "Sen's slope",
                "Sen's intercept",
                "Trend direction",
                "S statistic",
                "Z statistic",
                f"Significant (alpha={alpha:g})",
                "N",
                "Slope units",
            ],
            "Value": [
                round(tau, 4),
                round(p_value, 6),
                round(slope, 6),
                round(intercept, 4),
                computed["trend"],
                int(computed["s"]),
                round(float(computed["z"]), 4),
                "Yes" if p_value < alpha else "No",
                len(sub),
                "observation index (not elapsed time)",
            ],
        }
    )
    trend_line = np.round(intercept + slope * np.arange(len(sub)), 4)
    trend = pd.DataFrame(
        {
            date_col: sub[date_col].dt.strftime("%Y-%m-%d"),
            value_col: sub[value_col].values,
            "Trend Line": trend_line,
        }
    )
    return Result.success(MannKendallResult(summary=summary, trend=trend), *diags)
