"""Association statistics (FR-2.3) — crosstabs, chi-square, keyness.

Parity basis: ``statistics_statistical_tests_util.py`` (legacy) — the
chi-square/crosstab/log-likelihood runners. Tables keep legacy shapes (CSV
ready, flat index) so future goldens compare directly.

Intentional differences from legacy: messageboxes become ERROR diagnostics
(``STATS_*``); the low-expected-frequency alert becomes a WARNING; charts and
the automatic-test coupling are out of scope (FR-6.5, FR-2.4+); Phi is added
for 2x2 tables alongside Cramer's V. Yates' continuity correction stays on
for 2x2 (the scipy default legacy used) — documented here, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
from scipy import stats as scipy_stats

from core.result import Diagnostic, Result

__all__ = [
    "ChiSquareResult",
    "CrosstabResult",
    "KeynessResult",
    "chi_square",
    "crosstab",
    "log_likelihood",
]


@dataclass(frozen=True, slots=True)
class CrosstabResult:
    counts: pd.DataFrame  # with Total margins, flat index
    row_pct: pd.DataFrame  # row percentages, flat index


@dataclass(frozen=True, slots=True)
class ChiSquareResult:
    summary: pd.DataFrame  # Statistic/Value rows
    residuals: pd.DataFrame  # standardized residuals, flat index
    obs_exp: pd.DataFrame  # observed vs expected per row category


@dataclass(frozen=True, slots=True)
class KeynessResult:
    frame: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def _need_columns(frame: pd.DataFrame, cols: list[str]) -> Diagnostic | None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        return Diagnostic.error("STATS_BAD_COLUMN", f"column(s) not in frame: {missing}", missing=missing)
    return None


def crosstab(frame: pd.DataFrame, col1: str, col2: str) -> Result[CrosstabResult]:
    """Counts with margins plus row percentages for two categorical columns."""
    bad = _need_columns(frame, [col1, col2])
    if bad is not None:
        return Result.failure(bad)
    sub = frame[[col1, col2]].dropna()
    if sub.empty:
        return Result.failure(
            Diagnostic.error("STATS_NO_DATA", "no rows remain after dropping blanks", col1=col1, col2=col2)
        )
    counts = pd.crosstab(sub[col1], sub[col2], margins=True, margins_name="Total").reset_index()
    row_pct = (pd.crosstab(sub[col1], sub[col2], normalize="index") * 100).round(2).reset_index()
    return Result.success(CrosstabResult(counts=counts, row_pct=row_pct))


def chi_square(frame: pd.DataFrame, col1: str, col2: str, *, alpha: float = 0.05) -> Result[ChiSquareResult]:
    """Chi-square test of independence with Cramer's V (and Phi for 2x2)."""
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0 < alpha < 1
    ):
        return Result.failure(Diagnostic.error("STATS_BAD_ALPHA", f"alpha must be in (0, 1), got {alpha}", alpha=alpha))
    bad = _need_columns(frame, [col1, col2])
    if bad is not None:
        return Result.failure(bad)
    sub = frame[[col1, col2]].dropna()
    if len(sub) < 5:
        return Result.failure(Diagnostic.error("STATS_TOO_FEW_ROWS", "need at least 5 rows for chi-square", n=len(sub)))
    ct = pd.crosstab(sub[col1], sub[col2])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return Result.failure(
            Diagnostic.error(
                "STATS_TOO_FEW_CATEGORIES",
                "both columns need at least 2 unique values",
                rows=int(ct.shape[0]),
                cols=int(ct.shape[1]),
            )
        )
    chi2, p_value, dof, expected = scipy_stats.chi2_contingency(ct)
    n = float(ct.values.sum())
    min_dim = min(ct.shape) - 1
    cramers_v = math.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
    min_expected = float(expected.min())

    diags: list[Diagnostic] = []
    if min_expected < 5:
        diags.append(
            Diagnostic.warning(
                "CHI2_LOW_EXPECTED",
                f"minimum expected frequency {min_expected:.2f} < 5; approximation may be unreliable",
                min_expected=round(min_expected, 2),
            )
        )
    stats_rows: list[tuple[str, object]] = [
        ("Chi-Square", round(float(chi2), 4)),
        ("p-value", round(float(p_value), 6)),
        ("Degrees of freedom", int(dof)),
        ("Cramer's V (effect size)", round(cramers_v, 4)),
        ("N (total observations)", int(n)),
        (f"Significant (alpha={alpha:g})", "Yes" if p_value < alpha else "No"),
        ("Minimum expected frequency", round(min_expected, 2)),
        ("Row variable", col1),
        ("Column variable", col2),
    ]
    if ct.shape == (2, 2):
        stats_rows.insert(4, ("Phi (2x2 only)", round(math.sqrt(chi2 / n) if n > 0 else 0.0, 4)))
    summary = pd.DataFrame(stats_rows, columns=["Statistic", "Value"])

    # Pearson residuals: (O - E) / sqrt(E). Naming is precise — these are
    # NOT adjusted standardized residuals (which would divide by the
    # marginal-aware sqrt(E*(1-row%)(1-col%))). C6-5: named honestly.
    residuals = pd.DataFrame(
        (ct.values - expected) / (expected**0.5),
        index=ct.index,
        columns=ct.columns,
    )
    residuals.index.name = col1
    residuals = residuals.reset_index()

    # C6-5: the old obs-vs-exp table compared row SUMS, which are identical
    # by construction. The informative table is per-CELL observed vs expected.
    cells: list[dict[str, object]] = []
    for i, row_label in enumerate(ct.index):
        for j, col_label in enumerate(ct.columns):
            cells.append(
                {
                    col1: str(row_label),
                    col2: str(col_label),
                    "Observed": int(ct.values[i, j]),
                    "Expected": round(float(expected[i, j]), 2),
                    "Pearson Residual": round(float((ct.values[i, j] - expected[i, j]) / (expected[i, j] ** 0.5)), 4),
                }
            )
    obs_exp = pd.DataFrame(cells)
    return Result.success(ChiSquareResult(summary=summary, residuals=residuals, obs_exp=obs_exp), *diags)


LL_SMOOTHING = 0.5
"""Additive smoothing constant for Log Ratio / Pct Diff when a word has
frequency 0 in one corpus (Hardie's +0.5 convention; documented and
testable — the legacy's 1e-10 produced astronomically arbitrary ratios)."""


def _checked_frequencies(values: object, label: str) -> Result[list[float]]:
    """Coerce a column to non-negative finite counts; reject everything else."""
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    raw = series.tolist()
    checked: list[float] = []
    for i, value in enumerate(raw):
        if pd.isna(value):
            return Result.failure(
                Diagnostic.error(
                    "STATS_BAD_FREQUENCY",
                    f"{label} row {i + 1} is not a usable number",
                    column=label,
                    row=i + 1,
                )
            )
        f = float(value)
        if not math.isfinite(f):
            return Result.failure(
                Diagnostic.error("STATS_BAD_FREQUENCY", f"{label} row {i + 1} is not finite", column=label, row=i + 1)
            )
        if f < 0:
            return Result.failure(
                Diagnostic.error(
                    "STATS_NEGATIVE_FREQUENCY", f"{label} row {i + 1} is negative", column=label, row=i + 1
                )
            )
        if f != int(f):
            return Result.failure(
                Diagnostic.error(
                    "STATS_FRACTIONAL_FREQUENCY", f"{label} row {i + 1} is not a whole count", column=label, row=i + 1
                )
            )
        checked.append(f)
    return Result.success(checked)


def log_likelihood(
    frame: pd.DataFrame,
    word_col: str,
    freq_col1: str,
    freq_col2: str | None = None,
    *,
    corpus_col: str | None = None,
    smoothing: float = LL_SMOOTHING,
) -> Result[KeynessResult]:
    """Keyness (G2 log-likelihood) for two frequency columns or a corpus pivot.

    Mode A: ``freq_col1``/``freq_col2`` hold per-word counts. Mode B: ``freq_col1``
    holds counts and ``corpus_col`` identifies the two corpora (first two used).

    Validation (C6-5): frequencies must be whole, finite, non-negative counts;
    duplicate word rows are aggregated (summed); corpus totals must be
    strictly positive and finite; blank words are kept as their literal
    string (they are data). Log Ratio uses additive ``smoothing`` (default
    0.5, Hardie) so zero-frequency cells produce bounded, documented values.
    """
    bad = _need_columns(frame, [word_col])
    if bad is not None:
        return Result.failure(bad)
    if corpus_col is not None and corpus_col not in frame.columns:
        return Result.failure(
            Diagnostic.error(
                "STATS_BAD_COLUMN",
                f"requested corpus column {corpus_col!r} not in frame",
                column=corpus_col,
                columns=list(frame.columns),
            )
        )
    diags: list[Diagnostic] = []
    if corpus_col is not None:
        need = _need_columns(frame, [freq_col1])
        if need is not None:
            return Result.failure(need)
        freq_check = _checked_frequencies(frame[freq_col1], freq_col1)
        if freq_check.value is None:
            return Result.failure(*freq_check.diagnostics)
        numeric_frame = frame[[word_col, corpus_col]].copy()
        numeric_frame[freq_col1] = pd.to_numeric(frame[freq_col1], errors="coerce")
        pivot = numeric_frame.pivot_table(
            index=word_col, columns=corpus_col, values=freq_col1, aggfunc="sum", fill_value=0
        )
        labels = list(pivot.columns)
        if len(labels) < 2:
            return Result.failure(
                Diagnostic.error("STATS_TOO_FEW_GROUPS", f"need 2 corpus groups in {corpus_col!r}", n=len(labels))
            )
        if len(labels) > 2:
            diags.append(
                Diagnostic.warning(
                    "LL_TRUNCATED_CORPORA",
                    f"more than 2 corpora; using first two ({labels[0]}, {labels[1]})",
                    kept=[str(labels[0]), str(labels[1])],
                )
            )
            labels = labels[:2]
        words = [str(w) for w in pivot.index.tolist()]
        freq_a = pivot[labels[0]].astype(float).tolist()
        freq_b = pivot[labels[1]].astype(float).tolist()
        label_a, label_b = str(labels[0]), str(labels[1])
    elif freq_col2 is not None:
        need = _need_columns(frame, [freq_col1, freq_col2])
        if need is not None:
            return Result.failure(need)
        check_a = _checked_frequencies(frame[freq_col1], freq_col1)
        if check_a.value is None:
            return Result.failure(*check_a.diagnostics)
        check_b = _checked_frequencies(frame[freq_col2], freq_col2)
        if check_b.value is None:
            return Result.failure(*check_b.diagnostics)
        words = ["" if pd.isna(v) else str(v) for v in frame[word_col].tolist()]
        freq_a = check_a.unwrap()
        freq_b = check_b.unwrap()
        if len(words) != len(freq_a) or len(words) != len(freq_b):
            return Result.failure(
                Diagnostic.error("STATS_LENGTH_MISMATCH", "word and frequency columns have different lengths")
            )
        # Aggregate duplicate word rows explicitly (sum), announced once.
        if len({w for w in words}) != len(words):
            diags.append(
                Diagnostic.warning(
                    "LL_DUPLICATE_WORDS_AGGREGATED",
                    "duplicate word rows summed before scoring",
                )
            )
            agg: dict[str, list[float]] = {}
            for word, a, b in zip(words, freq_a, freq_b, strict=True):
                acc = agg.setdefault(word, [0.0, 0.0])
                acc[0] += a
                acc[1] += b
            words = list(agg)
            freq_a = [agg[w][0] for w in words]
            freq_b = [agg[w][1] for w in words]
        label_a, label_b = freq_col1, freq_col2
    else:
        return Result.failure(
            Diagnostic.error(
                "STATS_BAD_PARAMS", "provide freq_col2 or corpus_col", freq_col2=freq_col2, corpus_col=corpus_col
            )
        )

    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) or not math.isfinite(float(smoothing)):
        return Result.failure(
            Diagnostic.error("STATS_BAD_SMOOTHING", "smoothing must be finite and positive", smoothing=smoothing)
        )
    smoothing = float(smoothing)
    if smoothing <= 0:
        return Result.failure(
            Diagnostic.error("STATS_BAD_SMOOTHING", "smoothing must be strictly positive", smoothing=smoothing)
        )

    total_a = float(sum(freq_a))
    total_b = float(sum(freq_b))
    for name, total in ((label_a, total_a), (label_b, total_b)):
        if not math.isfinite(total) or total <= 0:
            return Result.failure(
                Diagnostic.error(
                    "STATS_EMPTY_CORPUS",
                    f"corpus total for {name!r} must be strictly positive and finite",
                    corpus=name,
                    total=total,
                )
            )

    rows: list[dict[str, object]] = []
    for word, a_raw, b_raw in zip(words, freq_a, freq_b, strict=True):
        a = float(a_raw)
        b = float(b_raw)
        e1 = total_a * (a + b) / (total_a + total_b)
        e2 = total_b * (a + b) / (total_a + total_b)
        g2 = 0.0
        if a > 0 and e1 > 0:
            g2 += a * math.log(a / e1)
        if b > 0 and e2 > 0:
            g2 += b * math.log(b / e2)
        g2 *= 2
        if not math.isfinite(g2):
            return Result.failure(
                Diagnostic.error("STATS_NUMERIC_FAILURE", f"G2 not finite for word {word!r}", word=word)
            )
        p_value = float(scipy_stats.chi2.sf(g2, 1)) if g2 > 0 else 1.0
        vocab_size = len(words)
        smoothed_a = (a + smoothing) / (total_a + smoothing * vocab_size)
        smoothed_b = (b + smoothing) / (total_b + smoothing * vocab_size)
        log_ratio = math.log2(smoothed_a / smoothed_b)
        pct_denom = smoothed_b
        rows.append(
            {
                "Word": word,
                f"Freq {label_a}": int(a),
                f"Freq {label_b}": int(b),
                "G2 (log-likelihood)": round(g2, 4),
                "p-value": round(p_value, 6),
                "Log Ratio": round(log_ratio, 4),
                "Pct Diff": round(100.0 * (smoothed_a - smoothed_b) / pct_denom, 2),
                "BIC": round(g2 - math.log(total_a + total_b), 4) if g2 > 0 else 0.0,
                "Overrepresented in": label_a if log_ratio > 0 else label_b,
            }
        )
    # Ties in G2 are common (every word with the same two counts scores the
    # same), so the sort must be total, not just correct: an unstable sort on
    # G2 alone let tied words permute between runs, which changes WHICH rows
    # a top-N cut keeps. Word is the tie-break, and the sort is stable.
    out = (
        pd.DataFrame(rows)
        .sort_values(["G2 (log-likelihood)", "Word"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    return Result.success(KeynessResult(frame=out), *diags)
