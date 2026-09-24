"""FR-2.5 trends — Spearman/Kendall options, Mann-Kendall + Sen's slope.

Oracles are definitional (hand-ranked S/tau/slope, Fisher CI containment),
never implementation copies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.analysis import stats_trends as T
from core.analysis.csv_stats import rank_correlation


def _val(summary: pd.DataFrame, name: str) -> object:
    return summary.loc[summary["Statistic"] == name, "Value"].iloc[0]


def _trend_frame(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="ME").strftime("%Y-%m-%d")
    return pd.DataFrame({"Date": list(dates), "Value": values})


class TestMannKendall:
    def test_monotonic_increase(self) -> None:
        result = T.mann_kendall(_trend_frame([float(i) for i in range(1, 11)]), "Date", "Value")
        assert result.ok
        summary = result.unwrap().summary
        assert _val(summary, "S statistic") == 45  # every pair concordant: 10*9/2
        assert float(_val(summary, "Kendall's tau")) == pytest.approx(1.0)
        assert _val(summary, "Trend direction") == "increasing"
        assert _val(summary, "Significant (alpha=0.05)") == "Yes"

    def test_monotonic_decrease(self) -> None:
        result = T.mann_kendall(_trend_frame([float(i) for i in range(10, 0, -1)]), "Date", "Value")
        summary = result.unwrap().summary
        assert _val(summary, "S statistic") == -45
        assert float(_val(summary, "Kendall's tau")) == pytest.approx(-1.0)
        assert _val(summary, "Trend direction") == "decreasing"

    def test_constant_series_has_no_trend(self) -> None:
        summary = T.mann_kendall(_trend_frame([5.0] * 10), "Date", "Value").unwrap().summary
        assert _val(summary, "S statistic") == 0
        assert _val(summary, "Trend direction") == "no significant trend"

    def test_sens_slope_recovers_linear_trend(self) -> None:
        values = [2 * i + 1 for i in range(10)]
        summary = T.mann_kendall(_trend_frame([float(v) for v in values]), "Date", "Value").unwrap().summary
        assert float(_val(summary, "Sen's slope")) == pytest.approx(2.0)
        assert float(_val(summary, "Sen's intercept")) == pytest.approx(1.0)

    def test_unsorted_dates_are_sorted(self) -> None:
        values = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0]
        ordered = T.mann_kendall(_trend_frame(values), "Date", "Value").unwrap().summary
        frame = _trend_frame(values)
        shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
        result = T.mann_kendall(shuffled, "Date", "Value").unwrap().summary
        pd.testing.assert_frame_equal(ordered, result)

    def test_trend_table_shape(self) -> None:
        trend = T.mann_kendall(_trend_frame([float(i) for i in range(1, 11)]), "Date", "Value").unwrap().trend
        assert list(trend.columns) == ["Date", "Value", "Trend Line"]
        assert len(trend) == 10

    def test_guards(self) -> None:
        assert T.mann_kendall(_trend_frame([1.0] * 5), "Date", "Value").value is None
        bad_dates = pd.DataFrame({"Date": ["not-a-date"] * 10, "Value": [float(i) for i in range(10)]})
        result = T.mann_kendall(bad_dates, "Date", "Value")
        assert result.value is None
        assert any(d.code == "MK_BAD_DATES" for d in result.diagnostics)
        assert T.mann_kendall(_trend_frame([1.0] * 10), "Date", "Nope").value is None


class TestRankCorrelation:
    def test_spearman_catches_monotonic_nonlinear(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 4.0, 9.0, 16.0, 25.0]})
        coef = rank_correlation(frame, "x", "y", method="spearman").unwrap()
        assert coef == pytest.approx(1.0)

    def test_kendall_hand_case(self) -> None:
        # 10 pairs, 2 discordant -> (8-2)/10 = 0.6
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 3.0, 2.0, 5.0, 4.0]})
        assert rank_correlation(frame, "x", "y", method="kendall").unwrap() == pytest.approx(0.6)

    def test_bad_method_rejected(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
        result = rank_correlation(frame, "x", "y", method="pearson")
        assert result.value is None
        assert any(d.code == "CSV_CORR_BAD_METHOD" for d in result.diagnostics)

    def test_constant_column_is_an_error(self) -> None:
        frame = pd.DataFrame({"x": [1.0] * 6, "y": [float(i) for i in range(6)]})
        result = rank_correlation(frame, "x", "y", method="spearman")
        assert result.value is None
        assert any(d.code == "CSV_CORR_CONSTANT" for d in result.diagnostics)

    def test_too_few_rows_is_an_error(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 1.0]})
        assert rank_correlation(frame, "x", "y", method="spearman").value is None


class TestCli:
    def test_trend_cli(self, tmp_path: Path) -> None:
        from tools.stats_trends import main

        src = tmp_path / "src"
        src.mkdir()
        csv = src / "series.csv"
        _trend_frame([float(i) for i in range(1, 11)]).to_csv(csv, index=False)
        out = tmp_path / "out"
        assert main(["trend", str(csv), str(out), "--date-col", "Date", "--value-col", "Value"]) == 0
        run_dir = next(out.iterdir())
        assert (run_dir / "mann_kendall_summary.csv").is_file()
        assert (run_dir / "mann_kendall_trend.csv").is_file()

    def test_rankcorr_cli(self, tmp_path: Path) -> None:
        from tools.stats_trends import main

        src = tmp_path / "src"
        src.mkdir()
        csv = src / "pairs.csv"
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 1.0, 4.0, 3.0, 5.0]}).to_csv(csv, index=False)
        out = tmp_path / "out"
        assert main(["rankcorr", str(csv), str(out), "--col-x", "x", "--col-y", "y", "--method", "kendall"]) == 0
        assert (next(out.iterdir()) / "rank_correlation.csv").is_file()


class TestTrendCorrections:
    """C6-6: alpha semantics, date handling, slope units, CI honesty."""

    def test_alpha_drives_trend_label(self) -> None:
        # A mostly-noisy series with a faint upward drift: significant at
        # alpha=0.2, not at 0.01 (verified numerically when written).
        values = [1.0, 1.4, 0.8, 0.3, 0.8, 0.3, 1.4, 2.8, 1.0, 0.9, 2.1, 2.0, 1.8, 0.8, 1.8, 2.6]
        summary_loose = T.mann_kendall(_trend_frame(values), "Date", "Value", alpha=0.2).unwrap().summary
        summary_tight = T.mann_kendall(_trend_frame(values), "Date", "Value", alpha=0.01).unwrap().summary
        loose_label = _val(summary_loose, "Trend direction")
        tight_label = _val(summary_tight, "Trend direction")
        tight_sig = _val(summary_tight, "Significant (alpha=0.01)")
        assert tight_sig == "No" and tight_label == "no significant trend"
        # Never "increasing" while "not significant" under the same alpha.
        loose_sig = _val(summary_loose, "Significant (alpha=0.2)")
        assert (loose_sig == "No") == (loose_label == "no significant trend")

    def test_alpha_validated(self) -> None:
        result = T.mann_kendall(_trend_frame([float(i) for i in range(10)]), "Date", "Value", alpha=1.5)
        assert result.value is None
        assert any(d.code == "STATS_BAD_ALPHA" for d in result.diagnostics)

    def test_nonfinite_values_rejected(self) -> None:
        frame = _trend_frame([1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = T.mann_kendall(frame, "Date", "Value")
        assert result.value is None
        assert any(d.code == "STATS_BAD_FREQUENCY" for d in result.diagnostics)
        frame_inf = _trend_frame([1.0, 2.0, float("inf"), 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert any(d.code == "STATS_BAD_FREQUENCY" for d in T.mann_kendall(frame_inf, "Date", "Value").diagnostics)

    def test_invalid_dates_dropped_with_warning(self) -> None:
        frame = _trend_frame([float(i) for i in range(1, 11)])
        frame.loc[3, "Date"] = "not-a-date"
        result = T.mann_kendall(frame, "Date", "Value")
        assert result.ok
        assert any(d.code == "MK_DROPPED_DATES" for d in result.diagnostics)
        summary = result.unwrap().summary
        assert _val(summary, "N") == 9  # dropped, not silently kept

    def test_duplicate_dates_warned_and_kept(self) -> None:
        frame = _trend_frame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        frame.loc[5, "Date"] = frame.loc[4, "Date"]
        result = T.mann_kendall(frame, "Date", "Value")
        assert result.ok
        assert any(d.code == "MK_DUPLICATE_DATES" for d in result.diagnostics)

    def test_slope_units_declared(self) -> None:
        summary = T.mann_kendall(_trend_frame([float(i) for i in range(10)]), "Date", "Value").unwrap().summary
        assert _val(summary, "Slope units") == "observation index (not elapsed time)"

    def test_spearman_ci_labeled_as_approximation(self) -> None:
        frame = pd.DataFrame({"x": [float(i) for i in range(10)], "y": [float(i * 2) for i in range(10)]})
        summary = T.correlation_summary(frame, "x", "y", "spearman").unwrap()
        note = summary.loc[summary["Statistic"] == "CI method", "Value"].iloc[0]
        assert "approximation" in str(note).lower()

    def test_constant_rank_correlation_defined(self) -> None:
        frame = pd.DataFrame({"x": [1.0] * 6, "y": [float(i) for i in range(6)]})
        result = rank_correlation(frame, "x", "y", method="kendall")
        assert result.value is None  # same constant-column rule as Pearson/spearman
