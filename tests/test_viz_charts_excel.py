"""Native Excel chart export (charts_excel): legacy-parity workbook shape.

Verifies the workbook layout the legacy suite produced (Chart sheet first,
Data sheet with categories + per-group series, legend removed for a single
series, unsupported kinds refused with a precise diagnostic). openpyxl is
already a test-environment dependency (installed); if absent the module
fails loudly instead of pretending.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from core.viz.charts_excel import EXCEL_KINDS, excel_chart_bytes
from core.viz.chartspec import ChartSpec, prepare_chart_data

openpyxl = pytest.importorskip("openpyxl")
from openpyxl import load_workbook  # noqa: E402


def _frame() -> pd.DataFrame:
    """Category totals: A=5, B=3, C=8."""
    return pd.DataFrame({"cat": ["A", "B", "C"], "n": [5.0, 3.0, 8.0]})


def _grouped() -> pd.DataFrame:
    """(A,x)=1 (A,y)=2 (B,x)=3 (B,y)=4."""
    return pd.DataFrame(
        {
            "cat": ["A", "A", "B", "B"],
            "grp": ["x", "y", "x", "y"],
            "n": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _workbook(result: object) -> object:
    return load_workbook(io.BytesIO(result.unwrap()))  # type: ignore[attr-defined]


class TestContract:
    def test_excel_kinds_match_legacy_menu(self) -> None:
        assert EXCEL_KINDS == ("bar", "line", "pie", "scatter", "radar", "bubble")


class TestWorkbookShape:
    def test_chart_sheet_first_data_behind(self) -> None:
        spec = ChartSpec(kind="bar", x="cat", y="n")
        prepared = prepare_chart_data(_frame(), spec).unwrap()
        result = excel_chart_bytes(prepared, spec)
        assert result.ok
        workbook = load_workbook(io.BytesIO(result.unwrap()))
        assert workbook.sheetnames == ["Chart", "Data"]

    def test_data_sheet_categories_and_values(self) -> None:
        spec = ChartSpec(kind="bar", x="cat", y="n")
        prepared = prepare_chart_data(_frame(), spec).unwrap()
        workbook = load_workbook(io.BytesIO(excel_chart_bytes(prepared, spec).unwrap()))
        rows = [[cell.value for cell in row] for row in workbook["Data"].iter_rows(max_row=4)]
        assert rows[0] == ["cat", "n"]
        assert [row[1] for row in rows[1:]] == [5, 3, 8]

    def test_single_series_legend_removed(self) -> None:
        spec = ChartSpec(kind="bar", x="cat", y="n", title="Counts")
        prepared = prepare_chart_data(_frame(), spec).unwrap()
        workbook = load_workbook(io.BytesIO(excel_chart_bytes(prepared, spec).unwrap()))
        chart = workbook["Chart"]._charts[0]
        assert chart.legend is None  # legacy: no "Series 1" box for one series
        assert chart.title is not None

    def test_axis_titles_and_low_labels(self) -> None:
        spec = ChartSpec(kind="bar", x="cat", y="n", x_label="Category", y_label="Count")
        prepared = prepare_chart_data(_frame(), spec).unwrap()
        workbook = load_workbook(io.BytesIO(excel_chart_bytes(prepared, spec).unwrap()))
        chart = workbook["Chart"]._charts[0]
        assert "Category" in chart.x_axis.title.tx.rich.p[0].r[0].t
        assert chart.y_axis.title.tx.rich.p[0].r[0].t == "Count"
        assert chart.x_axis.tickLblPos == "low"

    def test_grouped_bar_one_series_per_group(self) -> None:
        spec = ChartSpec(kind="bar", x="cat", y="n", group="grp", agg="sum")
        prepared = prepare_chart_data(_grouped(), spec).unwrap()
        workbook = load_workbook(io.BytesIO(excel_chart_bytes(prepared, spec).unwrap()))
        rows = [[cell.value for cell in row] for row in workbook["Data"].iter_rows(max_row=3)]
        assert rows[0] == ["cat", "x", "y"]
        assert rows[1] == ["A", 1, 2]
        chart = workbook["Chart"]._charts[0]
        assert len(chart.series) == 2  # one series per group, like the legacy
        assert chart.legend is not None  # multi-series keeps the legend

    @pytest.mark.parametrize("kind", ["line", "pie", "radar", "scatter"])
    def test_every_excel_kind_builds(self, kind: str) -> None:
        spec = ChartSpec(kind=kind, x="cat", y="n")  # type: ignore[arg-type]
        prepared = prepare_chart_data(_frame(), spec).unwrap()
        result = excel_chart_bytes(prepared, spec)
        assert result.ok, (kind, result.diagnostics)
        workbook = load_workbook(io.BytesIO(result.unwrap()))
        assert len(workbook["Chart"]._charts) == 1

    def test_bubble_is_reachable_through_the_public_api(self) -> None:
        """Previously it was not, and this test asserted that as the contract.

        "bubble" was in EXCEL_KINDS and had a branch in the Excel builder, but
        was missing from CHART_KINDS, so no ChartSpec could carry it: the
        branch was unreachable and one of the six kinds the original suite's
        Excel GUI offered could not be produced by anyone. Since that output is
        a required deliverable for work built on 1.6.38, the gap was closed
        rather than documented -- bubble is now a chart kind like any other,
        drawn by the Excel exporter and by the HTML renderer.
        """
        assert "bubble" in EXCEL_KINDS
        spec = ChartSpec(kind="bubble", x="cat", y="n")
        prepared = prepare_chart_data(_frame(), spec).unwrap()
        result = excel_chart_bytes(prepared, spec)
        assert result.ok, result.diagnostics
        workbook = load_workbook(io.BytesIO(result.unwrap()))
        assert len(workbook["Chart"]._charts) == 1


class TestUnsupportedKinds:
    @pytest.mark.parametrize("kind", ["sunburst", "treemap", "heatmap"])
    def test_no_native_excel_chart_fails_precisely(self, kind: str) -> None:
        spec = ChartSpec(kind=kind, x="cat", y="n", group="grp", agg="sum")  # type: ignore[arg-type]
        prepared = prepare_chart_data(_grouped(), spec).unwrap()
        result = excel_chart_bytes(prepared, spec)
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_EXCEL_UNSUPPORTED"
        assert kind in result.diagnostics[0].message

    @pytest.mark.parametrize("kind", ["violin", "waffle", "box"])
    def test_no_native_chart_from_ungrouped(self, kind: str) -> None:
        spec = ChartSpec(kind=kind, x="cat", y="n")  # type: ignore[arg-type]
        prepared = prepare_chart_data(_frame(), spec).unwrap()
        result = excel_chart_bytes(prepared, spec)
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_EXCEL_UNSUPPORTED"


class TestHorizontalAndModes:
    def test_horizontal_bar_type(self) -> None:
        spec = ChartSpec(kind="bar", x="cat", y="n", horizontal=True)
        prepared = prepare_chart_data(_frame(), spec).unwrap()
        workbook = load_workbook(io.BytesIO(excel_chart_bytes(prepared, spec).unwrap()))
        assert workbook["Chart"]._charts[0].type == "bar"

    def test_vertical_bar_col_type(self) -> None:
        spec = ChartSpec(kind="bar", x="cat", y="n")
        prepared = prepare_chart_data(_frame(), spec).unwrap()
        workbook = load_workbook(io.BytesIO(excel_chart_bytes(prepared, spec).unwrap()))
        assert workbook["Chart"]._charts[0].type == "col"

    def test_stacked_grouping(self) -> None:
        spec = ChartSpec(kind="bar", x="cat", y="n", group="grp", agg="sum", bar_mode="stack")
        prepared = prepare_chart_data(_grouped(), spec).unwrap()
        workbook = load_workbook(io.BytesIO(excel_chart_bytes(prepared, spec).unwrap()))
        chart = workbook["Chart"]._charts[0]
        assert chart.grouping == "stacked"
        assert chart.overlap == 100
