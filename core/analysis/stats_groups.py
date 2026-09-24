"""Group comparison statistics (FR-2.4) — Mann-Whitney, Kruskal-Wallis, Dunn.

Parity basis: ``statistics_statistical_tests_util.py`` (legacy) — the
Mann-Whitney, Kruskal-Wallis, and Dunn runners. Group order is alphabetical,
medians tables keep legacy shapes, post-hoc Dunn runs only when p < alpha
with 3+ groups.

Intentional differences from legacy: messageboxes become ERROR diagnostics;
the >2-group truncation becomes a WARNING; charts and the automatic-test
coupling are out of scope (FR-6.5, FR-2.5+); Holm-Bonferroni joins Bonferroni
as a post-hoc method (default stays Bonferroni).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.result import Diagnostic, Result

__all__ = [
    "KruskalResult",
    "MannWhitneyResult",
    "dunn",
    "kruskal_wallis",
    "mann_whitney",
]

_MW_MIN_N = 3
_KW_MIN_N = 2

_CLIFF_BANDS: tuple[tuple[float, str], ...] = (
    (0.147, "negligible"),
    (0.33, "small"),
    (0.474, "medium"),
    (float("inf"), "large"),
)


@dataclass(frozen=True, slots=True)
class MannWhitneyResult:
    summary: pd.DataFrame
    medians: pd.DataFrame


@dataclass(frozen=True, slots=True)
class KruskalResult:
    summary: pd.DataFrame
    posthoc: pd.DataFrame  # empty (with columns) when not significant or k < 3
    medians: pd.DataFrame


def _validate_alpha(alpha: float) -> Diagnostic | None:
    if not 0 < alpha < 1:
        return Diagnostic.error("STATS_BAD_ALPHA", f"alpha must be in (0, 1), got {alpha}", alpha=alpha)
    return None


def _prepare(
    frame: pd.DataFrame, value_col: str, group_col: str
) -> tuple[pd.DataFrame, list[object], dict[object, object]] | Diagnostic:
    """Return (filtered frame, display labels, display->string key map).

    C6-5: original group-label values are preserved for display (numeric
    labels stay numeric); lookups use a deterministic string key. Mixed-type
    labels key on str(value), which is documented and collision-checked.
    """
    if value_col not in frame.columns or group_col not in frame.columns:
        missing = [c for c in (value_col, group_col) if c not in frame.columns]
        return Diagnostic.error("STATS_BAD_COLUMN", f"column(s) not in frame: {missing}", missing=missing)
    sub = frame[[value_col, group_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if not np.isfinite(sub[value_col].to_numpy(dtype=float)).all():
        return Diagnostic.error("STATS_NONFINITE_VALUE", f"{value_col!r} contains non-finite values")
    if sub.empty:
        return Diagnostic.error("STATS_NO_NUMERIC", f"{value_col!r} has no numeric values")
    display_labels = sorted(sub[group_col].unique(), key=str)
    # Detect display collisions: distinct raw values with the same str() key
    # (e.g. int 1 and string "1") would silently merge in the str-key map.
    keys: dict[object, object] = {}
    for label in display_labels:
        key = str(label)
        if key in keys and keys[key] != label:
            return Diagnostic.error(
                "STATS_LABEL_COLLISION",
                f"distinct group labels share key {key!r}; cast the column to string first",
                key=key,
                labels=[str(l) for l in display_labels],
            )
        keys[key] = label
    return sub, display_labels, keys


def _cliff_interp(abs_delta: float) -> str:
    for threshold, label in _CLIFF_BANDS:
        if abs_delta < threshold:
            return label
    return "large"  # unreachable; keeps mypy total


def mann_whitney(
    frame: pd.DataFrame, value_col: str, group_col: str, *, alpha: float = 0.05
) -> Result[MannWhitneyResult]:
    """Two-group rank comparison with Cliff's delta (first two groups win)."""
    alpha_problem = _validate_alpha(alpha)
    if alpha_problem is not None:
        return Result.failure(alpha_problem)
    prepared = _prepare(frame, value_col, group_col)
    if isinstance(prepared, Diagnostic):
        return Result.failure(prepared)
    sub, display_labels, _keys = prepared
    if len(display_labels) < 2:
        return Result.failure(Diagnostic.error("STATS_TOO_FEW_GROUPS", "need at least 2 groups", n=len(display_labels)))
    diags: list[Diagnostic] = []
    if len(display_labels) > 2:
        diags.append(
            Diagnostic.warning(
                "MWU_TRUNCATED_GROUPS",
                f"more than 2 groups; using first two ({display_labels[0]}, {display_labels[1]})",
                kept=[str(display_labels[0]), str(display_labels[1])],
            )
        )
        display_labels = display_labels[:2]
        key_a, key_b = str(display_labels[0]), str(display_labels[1])
        sub = sub[sub[group_col].astype(str).isin([key_a, key_b])]
    else:
        key_a, key_b = str(display_labels[0]), str(display_labels[1])
    group_a = sub[sub[group_col].astype(str) == key_a][value_col]
    group_b = sub[sub[group_col].astype(str) == key_b][value_col]
    if len(group_a) < _MW_MIN_N or len(group_b) < _MW_MIN_N:
        return Result.failure(
            Diagnostic.error(
                "STATS_TOO_FEW_ROWS",
                "each group needs at least 3 observations",
                n_a=len(group_a),
                n_b=len(group_b),
            )
        )
    stat_u, p_value = scipy_stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    n1, n2 = len(group_a), len(group_b)
    cliffs_delta = (2.0 * float(stat_u)) / (n1 * n2) - 1.0
    summary = pd.DataFrame(
        {
            "Statistic": [
                "U statistic",
                "p-value",
                "Cliff's delta",
                "Effect size interpretation",
                f"Significant (alpha={alpha:g})",
                "Group A",
                "Group A n",
                "Group A median",
                "Group A IQR (25-75%)",
                "Group B",
                "Group B n",
                "Group B median",
                "Group B IQR (25-75%)",
            ],
            "Value": [
                round(float(stat_u), 4),
                round(float(p_value), 6),
                round(cliffs_delta, 4),
                _cliff_interp(abs(cliffs_delta)),
                "Yes" if p_value < alpha else "No",
                str(display_labels[0]),
                n1,
                round(float(group_a.median()), 4),
                f"{round(float(group_a.quantile(0.25)), 4)} - {round(float(group_a.quantile(0.75)), 4)}",
                str(display_labels[1]),
                n2,
                round(float(group_b.median()), 4),
                f"{round(float(group_b.quantile(0.25)), 4)} - {round(float(group_b.quantile(0.75)), 4)}",
            ],
        }
    )
    medians = pd.DataFrame(
        {
            "Group": [str(display_labels[0]), str(display_labels[1])],
            "Median": [round(float(group_a.median()), 4), round(float(group_b.median()), 4)],
            "Q25": [round(float(group_a.quantile(0.25)), 4), round(float(group_b.quantile(0.25)), 4)],
            "Q75": [round(float(group_a.quantile(0.75)), 4), round(float(group_b.quantile(0.75)), 4)],
        }
    )
    return Result.success(MannWhitneyResult(summary=summary, medians=medians), *diags)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (order-preserving)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min(p_values[idx] * (m - rank), 1.0))
        adjusted[idx] = running_max
    return adjusted


def dunn(
    groups: list[list[float]],
    group_labels: list[str],
    *,
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Dunn post-hoc pairwise comparisons with Bonferroni or Holm correction.

    C6-5: alpha is a parameter (previously hard-coded 0.05); empty groups
    and label-length mismatches are rejected; a zero standard error (all
    values identical across both groups) yields z=0 AND a warning column
    value of NaN p — no evidence either way, never a fabricated 0.
    """
    if method not in ("bonferroni", "holm"):
        raise ValueError(f"method must be 'bonferroni' or 'holm', got {method!r}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(groups) != len(group_labels):
        raise ValueError(f"groups ({len(groups)}) and group_labels ({len(group_labels)}) must have equal lengths")
    if any(len(g) == 0 for g in groups):
        raise ValueError("dunn() received an empty group")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("dunn() groups must contain only finite values")
    all_values = np.concatenate(arrays)
    n_total = len(all_values)
    ranks = scipy_stats.rankdata(all_values)
    group_ranks: list[np.ndarray] = []
    idx = 0
    for g in arrays:
        group_ranks.append(ranks[idx : idx + len(g)])
        idx += len(g)
    pairs = list(combinations(range(len(arrays)), 2))
    tied = np.unique(ranks, return_counts=True)[1]
    tie_correction = float(np.sum(tied**3 - tied) / (12.0 * (n_total - 1)))
    raw_p: list[float] = []
    rows: list[dict[str, object]] = []
    for i, j in pairs:
        n_i, n_j = len(arrays[i]), len(arrays[j])
        diff = float(np.mean(group_ranks[i]) - np.mean(group_ranks[j]))
        se = math.sqrt(((n_total * (n_total + 1) / 12.0) - tie_correction) * (1.0 / n_i + 1.0 / n_j))
        if se > 0:
            z_val = diff / se
            p_val = 2.0 * float(scipy_stats.norm.sf(abs(z_val)))
        else:
            # Identical values in both groups: no evidence, explicit NaN.
            z_val = 0.0
            p_val = float("nan")
        raw_p.append(p_val)
        rows.append(
            {
                "Group A": group_labels[i],
                "Group B": group_labels[j],
                "Mean Rank A": round(float(np.mean(group_ranks[i])), 4),
                "Mean Rank B": round(float(np.mean(group_ranks[j])), 4),
                "Z statistic": round(z_val, 4),
                "p-value": round(p_val, 6) if math.isfinite(p_val) else p_val,
            }
        )
    if method == "bonferroni":
        for row, p_val in zip(rows, raw_p, strict=True):
            adj = min(p_val * len(pairs), 1.0) if math.isfinite(p_val) else float("nan")
            row["p-value (Bonferroni)"] = round(adj, 6) if math.isfinite(adj) else adj
            row[f"Significant (alpha={alpha:g})"] = "Yes" if adj < alpha else "No"
    else:
        finite_p = [p for p in raw_p if math.isfinite(p)]
        adjusted = _holm_adjust(finite_p)
        by_value = dict(zip(finite_p, adjusted, strict=True))
        for row, p_val in zip(rows, raw_p, strict=True):
            adj = by_value.get(p_val, float("nan"))
            row["p-value (Holm)"] = round(adj, 6) if math.isfinite(adj) else adj
            row[f"Significant (alpha={alpha:g})"] = "Yes" if adj < alpha else "No"
    return pd.DataFrame(rows)


def kruskal_wallis(
    frame: pd.DataFrame,
    value_col: str,
    group_col: str,
    *,
    alpha: float = 0.05,
    posthoc_method: str = "bonferroni",
) -> Result[KruskalResult]:
    """Multi-group rank comparison with epsilon-squared and optional post-hoc."""
    alpha_problem = _validate_alpha(alpha)
    if alpha_problem is not None:
        return Result.failure(alpha_problem)
    prepared = _prepare(frame, value_col, group_col)
    if isinstance(prepared, Diagnostic):
        return Result.failure(prepared)
    sub, display_labels, _keys = prepared
    if len(display_labels) < 2:
        return Result.failure(Diagnostic.error("STATS_TOO_FEW_GROUPS", "need at least 2 groups", n=len(display_labels)))
    group_keys = [str(lbl) for lbl in display_labels]
    groups = [sub[sub[group_col].astype(str) == key][value_col].dropna().values for key in group_keys]
    for lbl, g in zip(display_labels, groups, strict=True):
        if len(g) < _KW_MIN_N:
            return Result.failure(
                Diagnostic.error("STATS_TOO_FEW_ROWS", f"group {lbl!r} has <2 observations", group=str(lbl))
            )
    try:
        stat_h, p_value = scipy_stats.kruskal(*groups)
    except ValueError:
        # All values identical: scipy raises instead of returning H=0.
        # Legacy crashed here; the honest answer is no evidence of difference.
        stat_h, p_value = 0.0, 1.0
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    eps_squared = (float(stat_h) - k + 1) / (n_total - k) if n_total > k else 0.0
    eps_squared = max(0.0, min(eps_squared, 1.0))
    summary_rows: list[list[object]] = [
        ["H statistic", round(float(stat_h), 4)],
        ["p-value", round(float(p_value), 6)],
        ["Epsilon-squared (effect size)", round(eps_squared, 4)],
        ["Number of groups", k],
        ["Total N", n_total],
        [f"Significant (alpha={alpha:g})", "Yes" if p_value < alpha else "No"],
    ]
    for lbl, g in zip(display_labels, groups, strict=True):
        summary_rows.append([f"Group: {lbl} (n)", len(g)])
        summary_rows.append([f"Group: {lbl} (median)", round(float(np.median(g)), 4)])
    summary = pd.DataFrame(summary_rows, columns=["Statistic", "Value"])
    if p_value < alpha and k >= 3:
        posthoc = dunn(
            [[float(v) for v in g] for g in groups],
            [str(lbl) for lbl in display_labels],
            method=posthoc_method,
            alpha=alpha,
        )
    else:
        # C6-5: empty posthoc schema matches the selected correction method.
        posthoc = pd.DataFrame(
            columns=[
                "Group A",
                "Group B",
                "Mean Rank A",
                "Mean Rank B",
                "Z statistic",
                "p-value",
                "p-value (Holm)" if posthoc_method == "holm" else "p-value (Bonferroni)",
                "Significant (alpha=0.05)",
            ]
        )
    medians = pd.DataFrame(
        {
            "Group": [str(lbl) for lbl in display_labels],
            "Median": [round(float(np.median(g)), 4) for g in groups],
        }
    )
    return Result.success(KruskalResult(summary=summary, posthoc=posthoc, medians=medians))
