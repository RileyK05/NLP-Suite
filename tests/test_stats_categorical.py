"""FR-2.3 association tests — crosstabs, chi-square + effect sizes, keyness.

Numeric oracles are definitional (recomputed inline from textbook formulas),
never copied from the implementation. scipy appears only as the p-value
survival function both sides agree must hold, plus guards and layout.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from core.analysis import stats_categorical as S


def _voting_frame() -> pd.DataFrame:
    # Gender xVote: [[30, 10], [20, 40]] — textbook 2x2.
    rows = [("F", "yes")] * 30 + [("F", "no")] * 10 + [("M", "yes")] * 20 + [("M", "no")] * 40
    return pd.DataFrame(rows, columns=["Gender", "Vote"])


class TestCrosstab:
    def test_counts_with_margins(self) -> None:
        result = S.crosstab(_voting_frame(), "Gender", "Vote")
        assert result.ok
        counts = result.unwrap().counts

        def cell(g: str, v: str) -> object:
            return counts.loc[counts["Gender"] == g, v].iloc[0]

        assert cell("F", "yes") == 30
        assert cell("M", "no") == 40
        assert cell("Total", "Total") == 100

    def test_row_percentages(self) -> None:
        row_pct = S.crosstab(_voting_frame(), "Gender", "Vote").unwrap().row_pct

        def cell(g: str, v: str) -> object:
            return row_pct.loc[row_pct["Gender"] == g, v].iloc[0]

        assert cell("F", "yes") == pytest.approx(75.0)
        assert cell("M", "no") == pytest.approx(200 / 3, rel=1e-3)

    def test_missing_column_is_an_error(self) -> None:
        result = S.crosstab(_voting_frame(), "Gender", "Nope")
        assert result.value is None
        assert any(d.code == "STATS_BAD_COLUMN" for d in result.diagnostics)


class TestChiSquare:
    def test_statistic_matches_definition(self) -> None:
        result = S.chi_square(_voting_frame(), "Gender", "Vote")
        assert result.ok
        summary = result.unwrap().summary
        chi2 = float(summary.loc[summary["Statistic"] == "Chi-Square", "Value"].iloc[0])
        # Definitional with Yates' correction (scipy/legacy default on 2x2):
        # 2*(9.5^2/20) + 2*(9.5^2/30) == 15.0417
        assert chi2 == pytest.approx(2 * (9.5**2 / 20) + 2 * (9.5**2 / 30), rel=1e-3)

    def test_cramers_v_and_phi_on_2x2(self) -> None:
        summary = S.chi_square(_voting_frame(), "Gender", "Vote").unwrap().summary

        def val(name: str) -> float:
            return float(summary.loc[summary["Statistic"] == name, "Value"].iloc[0])

        yates_chi2 = 2 * (9.5**2 / 20) + 2 * (9.5**2 / 30)
        assert val("Cramer's V (effect size)") == pytest.approx(math.sqrt(yates_chi2 / 100), rel=1e-3)
        assert val("Phi (2x2 only)") == pytest.approx(math.sqrt(yates_chi2 / 100), rel=1e-3)
        assert val("Degrees of freedom") == 1
        sig = summary.loc[summary["Statistic"] == "Significant (alpha=0.05)", "Value"].iloc[0]
        assert sig == "Yes"

    def test_nonsignificant_table(self) -> None:
        frame = pd.DataFrame(
            [("a", "x")] * 25 + [("a", "y")] * 25 + [("b", "x")] * 25 + [("b", "y")] * 25, columns=["A", "B"]
        )
        summary = S.chi_square(frame, "A", "B").unwrap().summary
        assert float(summary.loc[summary["Statistic"] == "Chi-Square", "Value"].iloc[0]) == pytest.approx(0.0)
        assert summary.loc[summary["Statistic"] == "Significant (alpha=0.05)", "Value"].iloc[0] == "No"

    def test_residuals_match_definition(self) -> None:
        resid = S.chi_square(_voting_frame(), "Gender", "Vote").unwrap().residuals
        # E(F,yes) = 40*50/100 = 20 -> (30-20)/sqrt(20)
        assert float(resid.loc[resid["Gender"] == "F", "yes"].iloc[0]) == pytest.approx(10 / math.sqrt(20))

    def test_low_expected_frequency_warns(self) -> None:
        frame = pd.DataFrame(
            [("a", "x")] * 20 + [("a", "y")] * 2 + [("b", "x")] * 20 + [("b", "y")] * 2,
            columns=["A", "B"],
        )
        result = S.chi_square(frame, "A", "B")
        assert result.ok
        assert any(d.code == "CHI2_LOW_EXPECTED" for d in result.diagnostics)

    def test_guards(self) -> None:
        tiny = pd.DataFrame([("a", "x")] * 4, columns=["A", "B"])
        assert S.chi_square(tiny, "A", "B").value is None
        one_cat = pd.DataFrame([("a", "x")] * 10 + [("a", "y")] * 10, columns=["A", "B"])
        result = S.chi_square(one_cat, "A", "B")
        assert result.value is None
        assert any(d.code == "STATS_TOO_FEW_CATEGORIES" for d in result.diagnostics)


class TestKeyness:
    def _freq_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Word": ["w1", "w2"], "FreqA": [10, 0], "FreqB": [5, 20]},
            columns=["Word", "FreqA", "FreqB"],
        )

    def test_g2_matches_hand_computation(self) -> None:
        table = S.log_likelihood(self._freq_frame(), "Word", "FreqA", "FreqB").unwrap().to_frame()
        g2_w1 = float(table.loc[table["Word"] == "w1", "G2 (log-likelihood)"].iloc[0])
        # totals 10 and 25; e1 = 10*15/35, e2 = 25*15/35
        e1, e2 = 10 * 15 / 35, 25 * 15 / 35
        assert g2_w1 == pytest.approx(2 * (10 * math.log(10 / e1) + 5 * math.log(5 / e2)), rel=1e-3)

    def test_sorted_desc_and_overrep_labels(self) -> None:
        table = S.log_likelihood(self._freq_frame(), "Word", "FreqA", "FreqB").unwrap().to_frame()
        assert table.iloc[0]["Word"] == "w2"  # larger G2 first
        assert table.loc[table["Word"] == "w1", "Overrepresented in"].iloc[0] == "FreqA"
        assert table.loc[table["Word"] == "w2", "Overrepresented in"].iloc[0] == "FreqB"

    def test_pivot_mode_matches_two_column_mode(self) -> None:
        long = pd.DataFrame(
            {
                "Word": ["w1", "w1", "w2", "w2"],
                "Corpus": ["A", "B", "A", "B"],
                "Freq": [10, 5, 0, 20],
            }
        )
        via_pivot = S.log_likelihood(long, "Word", "Freq", corpus_col="Corpus").unwrap().to_frame()
        via_cols = S.log_likelihood(self._freq_frame(), "Word", "FreqA", "FreqB").unwrap().to_frame()
        assert via_pivot["G2 (log-likelihood)"].tolist() == pytest.approx(via_cols["G2 (log-likelihood)"].tolist())

    def test_zero_total_corpus_is_an_error(self) -> None:
        frame = pd.DataFrame({"Word": ["w1"], "FreqA": [0], "FreqB": [5]})
        result = S.log_likelihood(frame, "Word", "FreqA", "FreqB")
        assert result.value is None
        assert any(d.code == "STATS_EMPTY_CORPUS" for d in result.diagnostics)


class TestKeynessOrdering:
    """A top-N cut must keep the same rows on every run.

    G2 ties are the common case, not an edge case: any two words with the same
    pair of counts score identically. Sorting on G2 alone with an unstable sort
    let those rows permute per process, so `--top-n` returned a different set of
    words each time. The ordering is now total (G2 desc, then Word asc).
    """

    @staticmethod
    def _tied(n: int = 16) -> pd.DataFrame:
        names = [f"word{i:02d}" for i in range(n)]
        return pd.DataFrame([{"Word": w, "A": 10, "B": 2} for w in names])

    def test_ties_break_alphabetically(self) -> None:
        out = S.log_likelihood(self._tied(), "Word", "A", "B").unwrap().frame
        tied = out[out["G2 (log-likelihood)"] == out["G2 (log-likelihood)"].iloc[0]]
        assert tied["Word"].tolist() == sorted(tied["Word"].tolist())

    def test_row_order_is_independent_of_input_order(self) -> None:
        frame = self._tied()
        forward = S.log_likelihood(frame, "Word", "A", "B").unwrap().frame
        reversed_in = frame.iloc[::-1].reset_index(drop=True)
        backward = S.log_likelihood(reversed_in, "Word", "A", "B").unwrap().frame
        assert forward["Word"].tolist() == backward["Word"].tolist()

    def test_top_n_cut_is_reproducible(self) -> None:
        frame = self._tied(24)
        first = S.log_likelihood(frame, "Word", "A", "B").unwrap().frame.head(5)["Word"].tolist()
        shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
        second = S.log_likelihood(shuffled, "Word", "A", "B").unwrap().frame.head(5)["Word"].tolist()
        assert first == second


class TestCli:
    def test_chi2_cli_writes_three_tables(self, tmp_path: Path) -> None:
        from tools.stats_categorical import main

        src = tmp_path / "src"
        src.mkdir()
        csv = src / "votes.csv"
        _voting_frame().to_csv(csv, index=False)
        out = tmp_path / "out"
        assert main(["chi2", str(csv), str(out), "--col1", "Gender", "--col2", "Vote"]) == 0
        run_dir = next(out.iterdir())
        for name in ("chi2_summary.csv", "chi2_residuals.csv", "chi2_obs_vs_exp.csv"):
            assert (run_dir / name).is_file(), name

    def test_keyness_cli(self, tmp_path: Path) -> None:
        from tools.stats_categorical import main

        src = tmp_path / "src"
        src.mkdir()
        csv = src / "freq.csv"
        pd.DataFrame({"Word": ["w1", "w2"], "FreqA": [10, 0], "FreqB": [5, 20]}).to_csv(csv, index=False)
        out = tmp_path / "out"
        assert main(["keyness", str(csv), str(out), "--word-col", "Word", "--freq1", "FreqA", "--freq2", "FreqB"]) == 0
        assert (next(out.iterdir()) / "keyness.csv").is_file()

    def test_cli_missing_file_fails(self, tmp_path: Path) -> None:
        from tools.stats_categorical import main

        assert main(["chi2", str(tmp_path / "ghost.csv"), str(tmp_path / "out"), "--col1", "A", "--col2", "B"]) == 2


class TestFrequencyValidation:
    """C6-5: checked frequency conversion; no silent NaN/negative/inf/floats."""

    def test_nan_rejected(self) -> None:
        frame = pd.DataFrame({"Word": ["a"], "FreqA": [float("nan")], "FreqB": [1]})
        result = S.log_likelihood(frame, "Word", "FreqA", "FreqB")
        assert result.value is None
        assert any(d.code == "STATS_BAD_FREQUENCY" for d in result.diagnostics)

    def test_negative_rejected(self) -> None:
        frame = pd.DataFrame({"Word": ["a"], "FreqA": [-2], "FreqB": [1]})
        assert any(
            d.code == "STATS_NEGATIVE_FREQUENCY" for d in S.log_likelihood(frame, "Word", "FreqA", "FreqB").diagnostics
        )

    def test_infinite_rejected(self) -> None:
        frame = pd.DataFrame({"Word": ["a"], "FreqA": [float("inf")], "FreqB": [1]})
        assert any(
            d.code == "STATS_BAD_FREQUENCY" for d in S.log_likelihood(frame, "Word", "FreqA", "FreqB").diagnostics
        )

    def test_fractional_rejected(self) -> None:
        frame = pd.DataFrame({"Word": ["a"], "FreqA": [2.5], "FreqB": [1]})
        assert any(
            d.code == "STATS_FRACTIONAL_FREQUENCY"
            for d in S.log_likelihood(frame, "Word", "FreqA", "FreqB").diagnostics
        )

    def test_non_numeric_rejected(self) -> None:
        frame = pd.DataFrame({"Word": ["a"], "FreqA": ["many"], "FreqB": [1]})
        assert any(
            d.code == "STATS_BAD_FREQUENCY" for d in S.log_likelihood(frame, "Word", "FreqA", "FreqB").diagnostics
        )

    def test_missing_corpus_col_named(self) -> None:
        frame = pd.DataFrame({"Word": ["a"], "Freq": [1]})
        result = S.log_likelihood(frame, "Word", "Freq", corpus_col="Ghost")
        assert result.value is None
        assert any(d.code == "STATS_BAD_COLUMN" for d in result.diagnostics)

    def test_duplicates_aggregated_with_warning(self) -> None:
        frame = pd.DataFrame(
            {"Word": ["w", "w"], "FreqA": [1, 2], "FreqB": [3, 4]},
            columns=["Word", "FreqA", "FreqB"],
        )
        result = S.log_likelihood(frame, "Word", "FreqA", "FreqB")
        assert result.ok
        assert any(d.code == "LL_DUPLICATE_WORDS_AGGREGATED" for d in result.diagnostics)
        table = result.unwrap().to_frame()
        assert len(table) == 1
        assert int(table.iloc[0]["Freq FreqA"]) == 3

    def test_smoothing_bounded_and_configurable(self) -> None:
        # zero-in-one-corpus word: legacy 1e-10 produced astronomic ratios.
        # (Corpus totals stay positive; only the single word can be 0.)
        frame = pd.DataFrame({"Word": ["w", "pad"], "FreqA": [0, 10], "FreqB": [10, 10]})
        default = S.log_likelihood(frame, "Word", "FreqA", "FreqB").unwrap().to_frame()
        assert abs(float(default.iloc[0]["Log Ratio"])) < 25  # bounded, not 1e9
        bigger = S.log_likelihood(frame, "Word", "FreqA", "FreqB", smoothing=5.0).unwrap().to_frame()
        assert abs(float(bigger.iloc[0]["Log Ratio"])) < abs(float(default.iloc[0]["Log Ratio"]))

    def test_blank_word_is_kept_as_data(self) -> None:
        frame = pd.DataFrame({"Word": ["", "w"], "FreqA": [1, 2], "FreqB": [3, 4]})
        result = S.log_likelihood(frame, "Word", "FreqA", "FreqB")
        assert result.ok
        assert "" in set(result.unwrap().to_frame()["Word"].tolist())

    def test_multicategory_and_low_expected(self) -> None:
        # 3x3 table with a low-expected cell: works, warns, per-cell obs/exp
        rows = []
        for a, n in [("x", 30), ("y", 5), ("z", 5)]:
            for b, m in [("p", 8), ("q", 2), ("r", 2)]:
                rows.extend([(a, b)] * (n * m // 15 + 1))
        frame = pd.DataFrame(rows, columns=["A", "B"])
        result = S.chi_square(frame, "A", "B")
        assert result.ok
        cells = result.unwrap().obs_exp
        n_a = frame["A"].nunique()
        n_b = frame["B"].nunique()
        assert len(cells) == n_a * n_b  # per-CELL table
        # per-cell expected differs from per-row totals
        assert (cells["Expected"] != cells["Observed"]).any()
        assert any(d.code == "CHI2_LOW_EXPECTED" for d in result.diagnostics)
