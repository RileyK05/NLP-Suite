"""Statistics figures, built from the engines' own output tables.

Each fixture is produced by running the real statistics function and passing
its table through a CSV round trip, which is what a run does (a CSV keeps no
dtypes and no column-index name). What these pin:

* the chi-square heatmap colours residuals on the diverging scale, keeps
  observed and expected in the hover, and warns when expected counts are
  too small for the test;
* a crosstab composition stretches every row to one length;
* medians figures say the raw values are not in the result;
* the trend draws observations as points and the Sen's slope as the line;
* one recipe covers both doors to each engine (``table_*`` and ``stats_*``).
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import pytest

from core.analysis.stats_categorical import chi_square, crosstab
from core.analysis.stats_groups import kruskal_wallis
from core.analysis.stats_trends import mann_kendall
from core.result import Result
from core.viz.panels_stats import STATS_PANELS
from core.viz.panelspec import PanelDefinition, PreparedPanel, Provenance

BY_NAME = {panel.name: panel for panel in STATS_PANELS}


def via_csv(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)))


def build(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    definition: PanelDefinition = BY_NAME[name]
    return definition.build(
        frame, definition.defaults() | (params or {}), Provenance(tool=definition.tool, panel=name, source="t.csv")
    )


def entities() -> pd.DataFrame:
    """PERSON rising, ORG falling across three decades."""
    rows = []
    for decade, (person, org, place) in {"1950s": (10, 40, 30), "1990s": (30, 25, 30), "2020s": (45, 10, 30)}.items():
        rows += [{"NER Tag": "PERSON", "Decade": decade}] * person
        rows += [{"NER Tag": "ORG", "Decade": decade}] * org
        rows += [{"NER Tag": "GPE", "Decade": decade}] * place
    return pd.DataFrame(rows)


class TestChiSquare:
    def panel(self) -> PreparedPanel:
        obs_exp = via_csv(chi_square(entities(), "NER Tag", "Decade", alpha=0.05).unwrap().obs_exp)
        result = build("table_chi2_residuals", obs_exp)
        assert result.ok, [str(d) for d in result.diagnostics]
        return result.unwrap()

    def test_residuals_on_a_diverging_scale_with_the_counts_in_the_hover(self) -> None:
        panel = self.panel()
        assert panel.color_scale == "diverging"
        person_2020 = next(m for m in panel.marks if m.key == "PERSON|2020s")
        assert person_2020.value is not None and person_2020.value > 2
        assert "observed 45" in person_2020.evidence.describe

    def test_decades_run_in_order(self) -> None:
        assert self.panel().x_categories == ("1950s", "1990s", "2020s")

    def test_each_cell_resolves_to_one_row(self) -> None:
        panel = self.panel()
        for mark in panel.marks:
            selected = panel.data
            for column, value in mark.evidence.filters:
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == 1

    def test_sparse_tables_are_warned_about(self) -> None:
        small = entities().groupby(["NER Tag", "Decade"]).head(2)
        obs_exp = via_csv(chi_square(small, "NER Tag", "Decade", alpha=0.05).unwrap().obs_exp)
        assert any(d.code == "PANEL_SMALL_EXPECTED" for d in build("table_chi2_residuals", obs_exp).diagnostics)


class TestCrosstab:
    def test_every_row_is_stretched_to_one_length(self) -> None:
        counts = via_csv(crosstab(entities(), "NER Tag", "Decade").unwrap().counts)
        panel = build("table_crosstab_composition", counts).unwrap()
        for row in {m.y for m in panel.marks}:
            segments = [m for m in panel.marks if m.y == row]
            assert sum(m.size or 0 for m in segments) == pytest.approx(1.0)


class TestGroupsAndTrend:
    def readings(self) -> pd.DataFrame:
        years = list(range(1934, 2025, 3))
        return pd.DataFrame(
            {
                "Date": [f"{year}-01-05" for year in years],
                "Decade": [f"{year // 10 * 10}s" for year in years],
                "Flesch": [45 + (year - 1934) * 0.2 + (3 if year % 12 else -3) for year in years],
            }
        )

    def test_the_medians_figure_says_the_values_are_not_in_the_result(self) -> None:
        medians = via_csv(
            kruskal_wallis(self.readings(), "Flesch", "Decade", alpha=0.05, posthoc_method="holm").unwrap().medians
        )
        panel = build("table_kw_medians", medians).unwrap()
        assert any("not every value" in note for note in panel.notes)

    def test_the_trend_draws_points_and_the_fitted_line(self) -> None:
        trend = via_csv(mann_kendall(self.readings(), "Date", "Flesch").unwrap().trend)
        panel = build("table_trend_trend", trend).unwrap()
        assert panel.points_only == ("Observations",)
        assert {m.group for m in panel.marks} == {"Observations", "Sen's slope"}
        line = sorted((m for m in panel.marks if m.group == "Sen's slope"), key=lambda m: m.x)
        assert line[0].y < line[-1].y, "Flesch rises in the fixture"


def test_both_doors_to_each_engine_have_the_figures() -> None:
    names = set(BY_NAME)
    assert {"table_chi2_residuals", "stats_categorical_residuals"} <= names
    assert {"table_kw_posthoc", "stats_groups_posthoc"} <= names
    assert {"table_trend_trend", "stats_trends_trend"} <= names


class TestCorrelationMatrix:
    """The heatmap needs a square; the engine now writes one (``correlation_matrix.csv``)."""

    def matrix(self) -> pd.DataFrame:
        from core.analysis.csv_stats import correlation_matrix

        frame = pd.DataFrame(
            {
                "year": list(range(1930, 1990, 10)),
                "words": [100, 240, 180, 300, 260, 310],
                "sentences": [8, 15, 12, 19, 17, 20],
                "label": [f"d{y}" for y in range(1930, 1990, 10)],
            }
        )
        return via_csv(correlation_matrix(frame, method="pearson").unwrap())

    def test_square_cells_on_a_diverging_scale(self) -> None:
        result = build("csv_stats_correlations", self.matrix())
        assert result.ok, [str(d) for d in result.diagnostics]
        panel = result.unwrap()
        assert panel.color_scale == "diverging"
        assert panel.x_categories == ("year", "words", "sentences")
        assert panel.y_categories == panel.x_categories
        # 3 x 3: the diagonal is included, written by the engine.
        assert len(panel.marks) == 9

    def test_each_cell_resolves_to_its_row(self) -> None:
        panel = build("csv_stats_correlations", self.matrix()).unwrap()
        for mark in panel.marks:
            selected = panel.data
            for column, value in mark.evidence.filters:
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == 1

    def test_a_non_square_table_is_refused(self) -> None:
        broken = self.matrix().drop(index=0)
        result = build("csv_stats_correlations", broken)
        assert not result.ok
        assert any(d.code == "PANEL_NO_DATA" for d in result.diagnostics)
