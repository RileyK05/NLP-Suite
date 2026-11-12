"""Extended chart kinds (pie/sunburst/treemap/violin/radar/waffle/calendar).

Hand-computed semantics for the legacy-parity visualizations ported into the
typed charts pipeline, plus renderer construction checks (no disk, no net).
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.viz.chartspec import CHART_KINDS, ChartSpec, prepare_chart_data

try:
    import plotly  # noqa: F401

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover - exercised only without plotly
    HAS_PLOTLY = False


def _hierarchy_frame() -> pd.DataFrame:
    """Outer categories A/B, inner groups x/y; hand totals:
    A: x=3, y=4 (7) | B: x=1, y=2 (3)."""
    return pd.DataFrame(
        {
            "cat": ["A", "B", "A", "B"],
            "grp": ["x", "x", "y", "y"],
            "n": [3.0, 1.0, 4.0, 2.0],
        }
    )


def _date_frame() -> pd.DataFrame:
    """Jan 2024: Mon Jan 1 (1.0), Wed Jan 3 (2.0), Fri Feb 2 (5.0)."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-02-02"]),
            "n": [1.0, 2.0, 5.0],
        }
    )


class TestContract:
    def test_kinds_extended(self) -> None:
        assert "pie" in CHART_KINDS and "sunburst" in CHART_KINDS and "treemap" in CHART_KINDS
        assert "violin" in CHART_KINDS and "radar" in CHART_KINDS
        assert "waffle" in CHART_KINDS and "calendar" in CHART_KINDS

    def test_old_kinds_unchanged(self) -> None:
        assert CHART_KINDS[:6] == ("bar", "line", "scatter", "histogram", "box", "heatmap")


def test_ecdf_keeps_ties_and_uses_each_groups_own_denominator() -> None:
    frame = pd.DataFrame({"n": [1, 1, 4, 1, 5], "grp": ["a", "a", "a", "b", "b"]})
    spec = ChartSpec(kind="ecdf", x="n", y="n", group="grp")
    prepared = prepare_chart_data(frame, spec).unwrap()
    assert list(prepared.data["x"]) == [1, 4, 1, 5]
    assert list(prepared.data["frequency"]) == [2, 1, 1, 1]
    assert list(prepared.data["n"]) == [2, 3, 1, 2]
    assert list(prepared.data["y"]) == pytest.approx([200 / 3, 100, 50, 100])
    assert prepared.x_label == "n" and prepared.y_label == "At or below (%)"
    if HAS_PLOTLY:
        from core.viz.plotters import build_figure

        fig = build_figure(prepared, spec).unwrap()
        assert fig.data[0].line.shape == "hv"
        assert list(fig.data[0].x) == [1, 1, 4, 5]
        assert list(fig.data[1].customdata) == [2, 3]


def test_calendar_aggregates_timestamps_by_date() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-29 08:00", "2024-02-29 09:00"]),
            "n": [2, 5],
        }
    )
    assert not prepare_chart_data(frame, ChartSpec(kind="calendar", x="date", y="n")).ok
    prepared = prepare_chart_data(frame, ChartSpec(kind="calendar", x="date", y="n", agg="sum")).unwrap()
    assert list(prepared.data["x"]) == [pd.Timestamp("2024-02-29")]
    assert list(prepared.data["y"]) == [7]


class TestHierarchy:
    def test_sunburst_requires_group(self) -> None:
        with pytest.raises(ValueError, match="sunburst needs --group"):
            ChartSpec(kind="sunburst", x="cat", y="n")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="treemap needs --group"):
            ChartSpec(kind="treemap", x="cat", y="n")  # type: ignore[arg-type]

    def test_pie_rejects_group(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="pie", x="cat", y="n", group="grp"))
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_UNSUPPORTED"
        assert "sunburst" in result.diagnostics[0].message

    def test_pie_deduplicates_with_agg(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="pie", x="cat", y="n", agg="sum"))
        assert result.ok
        prepared = result.unwrap()
        # A = 3+4 = 7, B = 1+2 = 3
        assert dict(zip(prepared.data["x"], prepared.data["y"], strict=True)) == {"A": 7.0, "B": 3.0}

    def test_sunburst_two_levels(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="sunburst", x="cat", y="n", group="grp"))
        assert result.ok
        prepared = result.unwrap()
        assert prepared.layout == "hierarchy"
        assert len(prepared.data) == 4  # (A,x) (B,x) (A,y) (B,y)
        assert list(prepared.data.columns) == ["x", "group", "y"]

    def test_sunburst_duplicates_refused_without_agg(self) -> None:
        doubled = pd.concat([_hierarchy_frame(), _hierarchy_frame()], ignore_index=True)
        result = prepare_chart_data(doubled, ChartSpec(kind="sunburst", x="cat", y="n", group="grp"))
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_AMBIGUOUS"
        assert "sunburst" in result.diagnostics[0].message

    def test_negative_slices_refused(self) -> None:
        frame = _hierarchy_frame()
        frame.loc[0, "n"] = -3.0
        result = prepare_chart_data(frame, ChartSpec(kind="sunburst", x="cat", y="n", group="grp"))
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_NEGATIVE_SLICE"

    def test_hierarchy_rejects_rate_and_bins(self) -> None:
        spec = ChartSpec(kind="treemap", x="cat", y="n", group="grp", bins=5)  # type: ignore[arg-type]
        result = prepare_chart_data(_hierarchy_frame(), spec)
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_UNSUPPORTED"

    def test_sunburst_top_n_keeps_whole_outer(self) -> None:
        result = prepare_chart_data(
            _hierarchy_frame(), ChartSpec(kind="sunburst", x="cat", y="n", group="grp", top_n=1)
        )
        assert result.ok
        assert set(result.unwrap().data["x"]) == {"A"}  # A total 7 > B total 3


class TestViolinRadar:
    def test_violin_keeps_observations(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="violin", x="cat", y="n"))
        assert result.ok
        assert len(result.unwrap().data) == 4  # raw rows, no aggregation
        assert result.unwrap().layout == "wide"

    def test_violin_rejects_agg_and_normalize(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="violin", x="cat", y="n", agg="sum"))
        assert result.value is None
        assert result.diagnostics[0].context["param"] == "agg"
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="violin", x="cat", y="n", normalize="percent"))
        assert result.value is None
        assert result.diagnostics[0].context["param"] == "normalize"

    def test_radar_aggregates_per_spoke(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="radar", x="cat", y="n", agg="sum"))
        assert result.ok
        prepared = result.unwrap()
        # Ungrouped radar: one spoke per x category (A=7, B=3).
        assert len(prepared.data) == 2
        assert dict(zip(prepared.data["x"], prepared.data["y"], strict=True)) == {"A": 7.0, "B": 3.0}
        assert prepared.layout == "wide"

    def test_radar_grouped_spokes(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="radar", x="cat", y="n", group="grp", agg="sum"))
        assert result.ok
        assert len(result.unwrap().data) == 4

    def test_radar_rejects_bins(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="radar", x="cat", y="n", bins=5))  # type: ignore[arg-type]
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_UNSUPPORTED"


class TestWaffle:
    def test_waffle_aggregates(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="waffle", x="cat", y="n", agg="sum"))
        assert result.ok
        prepared = result.unwrap()
        assert prepared.layout == "grid"
        assert prepared.prepared_by["waffle_grid"] == "10x10 (100 squares)"
        assert dict(zip(prepared.data["x"], prepared.data["y"], strict=True)) == {"A": 7.0, "B": 3.0}

    def test_waffle_refuses_negatives(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B", "C"], "n": [5.0, -3.0, 8.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="waffle", x="cat", y="n"))
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_NEGATIVE_SLICE"

    def test_waffle_top_n(self) -> None:
        result = prepare_chart_data(_hierarchy_frame(), ChartSpec(kind="waffle", x="cat", y="n", agg="sum", top_n=1))
        assert result.ok
        assert set(result.unwrap().data["x"]) == {"A"}


class TestCalendar:
    def test_calendar_needs_datetime(self) -> None:
        result = prepare_chart_data(
            _hierarchy_frame().rename(columns={"cat": "date"}), ChartSpec(kind="calendar", x="date", y="n")
        )
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_NOT_DATETIME"

    def test_calendar_spans_recorded(self) -> None:
        result = prepare_chart_data(_date_frame(), ChartSpec(kind="calendar", x="date", y="n"))
        assert result.ok
        assert result.unwrap().prepared_by["calendar_span"] == "2024-01-01..2024-02-02"
        assert result.unwrap().layout == "calendar"

    def test_calendar_duplicate_dates_refused(self) -> None:
        doubled = pd.concat([_date_frame(), _date_frame()], ignore_index=True)
        result = prepare_chart_data(doubled, ChartSpec(kind="calendar", x="date", y="n"))
        assert result.value is None
        assert result.diagnostics[0].code == "CHART_AMBIGUOUS"

    def test_calendar_rejects_normalize(self) -> None:
        result = prepare_chart_data(_date_frame(), ChartSpec(kind="calendar", x="date", y="n", normalize="percent"))
        assert result.value is None
        assert result.diagnostics[0].context["param"] == "normalize"


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestRenderers:
    def test_pie_sunburst_treemap_build(self) -> None:
        import plotly.express as px  # noqa: F401

        from core.viz.plotters import build_figure

        for kind, kwargs in (
            ("pie", {}),
            ("sunburst", {"group": "grp"}),
            ("treemap", {"group": "grp"}),
        ):
            spec = ChartSpec(kind=kind, x="cat", y="n", agg="sum", **kwargs)  # type: ignore[arg-type]
            prepared = prepare_chart_data(_hierarchy_frame(), spec).unwrap()
            fig = build_figure(prepared, spec).unwrap()
            assert len(fig.data) == 1, kind

    def test_violin_radar_build(self) -> None:
        from core.viz.plotters import build_figure

        violin = ChartSpec(kind="violin", x="cat", y="n")
        prepared = prepare_chart_data(_hierarchy_frame(), violin).unwrap()
        fig = build_figure(prepared, violin).unwrap()
        assert len(fig.data) == 1

        radar = ChartSpec(kind="radar", x="cat", y="n", agg="sum")
        prepared = prepare_chart_data(_hierarchy_frame(), radar).unwrap()
        fig = build_figure(prepared, radar).unwrap()
        assert len(fig.data) == 1

    def test_waffle_100_squares(self) -> None:
        from core.viz.plotters import build_figure

        spec = ChartSpec(kind="waffle", x="cat", y="n", agg="sum")  # type: ignore[arg-type]
        prepared = prepare_chart_data(_hierarchy_frame(), spec).unwrap()
        fig = build_figure(prepared, spec).unwrap()
        squares = sum(len(trace["x"]) for trace in fig.data)
        assert squares == 100

    def test_calendar_rows_are_months(self) -> None:
        from core.viz.plotters import build_figure

        spec = ChartSpec(kind="calendar", x="date", y="n", width=1100, title="Daily counts")
        frame = pd.concat(
            [
                _date_frame(),
                pd.DataFrame({"date": pd.to_datetime(["2024-01-08", "2024-01-15"]), "n": [3.0, 0.0]}),
            ]
        )
        prepared = prepare_chart_data(frame, spec).unwrap()
        fig = build_figure(prepared, spec).unwrap()
        assert list(fig.data[0].y) == ["2024-01", "2024-02"]
        assert list(fig.data[0].x) == list(range(1, 32))
        january, february = fig.data[0].z
        assert january[0] == 1.0 and january[7] == 3.0 and january[14] == 0.0
        assert january[1] is None and february[1] == 5.0
        assert fig.layout.title.text == "Daily counts"
        assert fig.layout.width == 1100
