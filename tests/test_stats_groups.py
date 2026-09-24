"""FR-2.4 group tests — Mann-Whitney, Kruskal-Wallis, Dunn, corrections.

Oracles are hand-ranked definitional cases, never implementation copies.
scipy appears in tests only where the definition itself is the library call
the legacy named (noted inline).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.analysis import stats_groups as G


def _val(summary: pd.DataFrame, name: str) -> object:
    return summary.loc[summary["Statistic"] == name, "Value"].iloc[0]


def _two_groups() -> pd.DataFrame:
    # n=5 per group: U=0 gives two-sided p ~= 0.008 (n=3 bottoms out at p=0.1).
    return pd.DataFrame(
        {"Score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Group": ["A"] * 5 + ["B"] * 5},
        columns=["Score", "Group"],
    )


def _three_groups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Score": [1, 2, 3, 10, 11, 12, 20, 21, 22],
            "Group": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        },
        columns=["Score", "Group"],
    )


class TestMannWhitney:
    def test_u_matches_hand_ranking(self) -> None:
        result = G.mann_whitney(_two_groups(), "Score", "Group")
        assert result.ok
        summary = result.unwrap().summary
        # Ranks of A: 1+2+3+4+5=15; U = 15 - 5*6/2 = 0
        assert float(_val(summary, "U statistic")) == 0.0
        assert _val(summary, "Significant (alpha=0.05)") == "Yes"

    def test_cliffs_delta_definition(self) -> None:
        summary = G.mann_whitney(_two_groups(), "Score", "Group").unwrap().summary
        # (2*0)/(5*5) - 1 = -1 -> large
        assert float(_val(summary, "Cliff's delta")) == pytest.approx(-1.0)
        assert _val(summary, "Effect size interpretation") == "large"

    def test_identical_groups_negligible(self) -> None:
        frame = pd.DataFrame({"Score": [1, 2, 3, 1, 2, 3], "Group": ["A"] * 3 + ["B"] * 3})
        summary = G.mann_whitney(frame, "Score", "Group").unwrap().summary
        assert float(_val(summary, "Cliff's delta")) == pytest.approx(0.0)
        assert _val(summary, "Effect size interpretation") == "negligible"

    def test_more_than_two_groups_uses_first_two_with_warning(self) -> None:
        frame = pd.DataFrame({"Score": [1, 2, 3, 4, 5, 6, 7, 8, 9], "Group": ["C", "A", "B"] * 3})
        result = G.mann_whitney(frame, "Score", "Group")
        assert result.ok
        assert any(d.code == "MWU_TRUNCATED_GROUPS" for d in result.diagnostics)

    def test_guards(self) -> None:
        assert G.mann_whitney(_two_groups(), "Score", "Nope").value is None
        one_group = pd.DataFrame({"Score": [1, 2, 3], "Group": ["A"] * 3})
        assert any(d.code == "STATS_TOO_FEW_GROUPS" for d in G.mann_whitney(one_group, "Score", "Group").diagnostics)
        tiny = pd.DataFrame({"Score": [1, 2, 3, 4], "Group": ["A", "A", "B", "B"]})
        assert any(d.code == "STATS_TOO_FEW_ROWS" for d in G.mann_whitney(tiny, "Score", "Group").diagnostics)
        strings = pd.DataFrame({"Score": ["x", "y"], "Group": ["A", "B"]})
        assert any(d.code == "STATS_NO_NUMERIC" for d in G.mann_whitney(strings, "Score", "Group").diagnostics)


class TestKruskalWallis:
    def test_separated_groups_significant(self) -> None:
        result = G.kruskal_wallis(_three_groups(), "Score", "Group")
        assert result.ok
        summary = result.unwrap().summary
        assert _val(summary, "Significant (alpha=0.05)") == "Yes"
        assert _val(summary, "Number of groups") == 3
        assert 0.0 <= float(_val(summary, "Epsilon-squared (effect size)")) <= 1.0

    def test_identical_groups_null(self) -> None:
        frame = pd.DataFrame({"Score": [5] * 9, "Group": ["A"] * 3 + ["B"] * 3 + ["C"] * 3})
        summary = G.kruskal_wallis(frame, "Score", "Group").unwrap().summary
        assert float(_val(summary, "H statistic")) == pytest.approx(0.0)
        assert _val(summary, "Significant (alpha=0.05)") == "No"

    def test_posthoc_runs_only_when_significant_and_k3(self) -> None:
        sig = G.kruskal_wallis(_three_groups(), "Score", "Group").unwrap()
        assert len(sig.posthoc) == 3  # 3 choose 2
        assert set(sig.posthoc.columns) >= {"Group A", "Group B", "p-value (Bonferroni)"}
        null = G.kruskal_wallis(
            pd.DataFrame({"Score": [5] * 9, "Group": ["A"] * 3 + ["B"] * 3 + ["C"] * 3}),
            "Score",
            "Group",
        ).unwrap()
        assert null.posthoc.empty

    def test_group_guard(self) -> None:
        thin = pd.DataFrame({"Score": [1, 2, 3], "Group": ["A", "B", "B"]})
        assert any(d.code == "STATS_TOO_FEW_ROWS" for d in G.kruskal_wallis(thin, "Score", "Group").diagnostics)


class TestDunn:
    def test_bonferroni_definition(self) -> None:
        posthoc = G.kruskal_wallis(_three_groups(), "Score", "Group").unwrap().posthoc
        for _, row in posthoc.iterrows():
            assert float(row["p-value (Bonferroni)"]) == pytest.approx(min(float(row["p-value"]) * 3, 1.0), rel=1e-3)

    def test_holm_chain(self) -> None:
        groups = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [8.0, 9.0, 10.0, 11.0]]
        bonf = G.dunn(groups, ["A", "B", "C"], method="bonferroni")
        holm = G.dunn(groups, ["A", "B", "C"], method="holm")
        assert len(bonf) == len(holm) == 3
        for (_, b), (_, h) in zip(bonf.iterrows(), holm.iterrows(), strict=True):
            assert float(b["p-value"]) <= float(h["p-value (Holm)"]) <= float(b["p-value (Bonferroni)"])

    def test_bad_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="method"):
            G.dunn([[1.0, 2.0], [3.0, 4.0]], ["A", "B"], method="nope")


class TestCli:
    def test_mw_cli(self, tmp_path: Path) -> None:
        from tools.stats_groups import main

        src = tmp_path / "src"
        src.mkdir()
        csv = src / "scores.csv"
        _two_groups().to_csv(csv, index=False)
        out = tmp_path / "out"
        assert main(["mw", str(csv), str(out), "--value-col", "Score", "--group-col", "Group"]) == 0
        run_dir = next(out.iterdir())
        assert (run_dir / "mann_whitney_summary.csv").is_file()
        assert (run_dir / "mann_whitney_medians.csv").is_file()

    def test_kw_cli_with_posthoc(self, tmp_path: Path) -> None:
        from tools.stats_groups import main

        src = tmp_path / "src"
        src.mkdir()
        csv = src / "scores.csv"
        _three_groups().to_csv(csv, index=False)
        out = tmp_path / "out"
        assert main(["kw", str(csv), str(out), "--value-col", "Score", "--group-col", "Group"]) == 0
        assert (next(out.iterdir()) / "kruskal_wallis_posthoc.csv").is_file()


class TestGroupValidation:
    """C6-5: labels, alpha, dunn contract, zero-SE behavior."""

    def test_numeric_group_labels_preserved(self) -> None:
        frame = pd.DataFrame({"Score": [1, 2, 3, 4, 5, 6], "Group": [10, 10, 10, 20, 20, 20]})
        result = G.mann_whitney(frame, "Score", "Group")
        assert result.ok
        summary = result.unwrap().summary
        labels = summary.loc[summary["Statistic"].isin(["Group A", "Group B"]), "Value"].tolist()
        assert "10" in labels and "20" in labels  # original values preserved for display

    def test_label_collision_detected(self) -> None:
        frame = pd.DataFrame({"Score": [1, 2, 3, 4, 5, 6], "Group": ["1", "1", "1", 1, 1, 1]}, dtype=object)
        result = G.mann_whitney(frame, "Score", "Group")
        assert result.value is None
        assert any(d.code == "STATS_LABEL_COLLISION" for d in result.diagnostics)

    def test_alpha_validated(self) -> None:
        frame = _two_groups()
        assert any(d.code == "STATS_BAD_ALPHA" for d in G.mann_whitney(frame, "Score", "Group", alpha=1.5).diagnostics)
        assert any(d.code == "STATS_BAD_ALPHA" for d in G.mann_whitney(frame, "Score", "Group", alpha=0.0).diagnostics)
        assert any(d.code == "STATS_BAD_ALPHA" for d in G.kruskal_wallis(frame, "Score", "Group", alpha=-1).diagnostics)

    def test_dunn_alpha_in_output(self) -> None:
        groups = [[1.0, 2, 3, 4], [2.0, 3, 4, 5], [8.0, 9, 10, 11]]
        out = G.dunn(groups, ["A", "B", "C"], method="holm", alpha=0.01)
        for _, row in out.iterrows():
            # A p-value that survives 0.01 must also survive 0.05; the flag
            # column must agree with the requested alpha, not a constant.
            expected = "Yes" if float(row["p-value (Holm)"]) < 0.01 else "No"
            assert row["Significant (alpha=0.01)"] == expected

    def test_empty_posthoc_schema_matches_method(self) -> None:
        frame = pd.DataFrame({"Score": [5] * 9, "Group": ["A"] * 3 + ["B"] * 3 + ["C"] * 3})
        bonf = G.kruskal_wallis(frame, "Score", "Group", posthoc_method="bonferroni").unwrap().posthoc
        holm = G.kruskal_wallis(frame, "Score", "Group", posthoc_method="holm").unwrap().posthoc
        assert "p-value (Bonferroni)" in bonf.columns
        assert "p-value (Holm)" in holm.columns
        assert "p-value (Holm)" not in bonf.columns

    def test_dunn_contract(self) -> None:
        with pytest.raises(ValueError, match="equal lengths"):
            G.dunn([[1.0, 2.0]], ["A", "B"])
        with pytest.raises(ValueError, match="empty group"):
            G.dunn([[1.0], []], ["A", "B"])
        with pytest.raises(ValueError, match="alpha"):
            G.dunn([[1.0, 2], [3.0, 4]], ["A", "B"], alpha=2.0)

    def test_zero_se_explicit_nan(self) -> None:
        out = G.dunn([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], ["A", "B"])
        assert pd.isna(out.iloc[0]["p-value"])  # no evidence, not a fabricated 0
