"""The five shapes added for 0.4.0, and the rules each one must share with the app.

Every rule here has a TypeScript twin in ``desktop/src/panelLayout.ts`` and a
test there on the same sample (``panelLayout.test.ts``);
``tests/test_panel_parity.py`` checks the two test files pin the same
numbers. The fixtures are small, hand-checked and built through
``PreparedPanel`` itself, so they satisfy the contract's validation the way a
real builder's output must.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from core.viz.panel_plotters import (
    _box_stats,
    _edge_points,
    _edge_width,
    _facet_grid,
    _heat_color,
    _heat_domain,
    _jitter,
    _jitter_offsets,
    _quantile,
    build_panel_figure,
    panel_html,
)
from core.viz.panelspec import COLOR_SCALES, Evidence, PanelEdge, PanelMark, PreparedPanel, Provenance


def _evidence(key: str) -> Evidence:
    return Evidence(scope="rows", filters=(("k", key),), count=1, describe=f"{key} stands for something")


def _mark(key: str, x: float, y: float, **extra: Any) -> PanelMark:
    return PanelMark(key=key, label=extra.pop("label", key), x=x, y=y, evidence=_evidence(key), **extra)


def _panel(shape: str, marks: tuple[PanelMark, ...], **extra: Any) -> PreparedPanel:
    return PreparedPanel(
        panel="p",
        shape=shape,  # type: ignore[arg-type]
        title="t",
        marks=marks,
        x_label="x",
        y_label="y",
        provenance=Provenance(tool="t", panel="p", source="s.csv"),
        data=pd.DataFrame({"k": [mark.key for mark in marks]}),
        **extra,
    )


# ------------------------------------------------------------------ heatmap --


class TestHeatmapColour:
    """Linear in RGB between evenly spaced stops, half-up rounding."""

    def test_the_ends_and_the_middle_are_the_stops(self) -> None:
        stops = COLOR_SCALES["sequential"]
        assert _heat_color(0.0, 0.0, 1.0, stops) == stops[0]
        assert _heat_color(0.5, 0.0, 1.0, stops) == stops[2]
        assert _heat_color(1.0, 0.0, 1.0, stops) == stops[-1]

    def test_between_two_stops_is_the_rgb_midpoint_rounded_half_up(self) -> None:
        """1/8 of the way is halfway from #f7fbff (247,251,255) to #c6dbef
        (198,219,239): (222.5, 235, 247). Half up gives 223 = 0xdf; Python's
        round() would give 222 and disagree with the app."""
        assert _heat_color(0.125, 0.0, 1.0, COLOR_SCALES["sequential"]) == "#dfebf7"

    def test_a_one_value_domain_is_the_middle_colour(self) -> None:
        stops = COLOR_SCALES["diverging"]
        assert _heat_color(3.0, 3.0, 3.0, stops) == stops[2]

    def test_values_outside_the_domain_clamp_to_the_ends(self) -> None:
        stops = COLOR_SCALES["sequential"]
        assert _heat_color(-5.0, 0.0, 1.0, stops) == stops[0]
        assert _heat_color(5.0, 0.0, 1.0, stops) == stops[-1]

    def test_sequential_spans_min_to_max(self) -> None:
        assert _heat_domain([0.2, -0.4, 0.9], "sequential") == (-0.4, 0.9)

    def test_diverging_is_symmetric_around_zero(self) -> None:
        """-0.4 and 0.9 give [-0.9, 0.9]: zero is the neutral centre stop
        whatever the matrix holds."""
        assert _heat_domain([0.2, -0.4, 0.9], "diverging") == (-0.9, 0.9)
        low, high = _heat_domain([0.2, -0.4, 0.9], "diverging")
        assert _heat_color(0.0, low, high, COLOR_SCALES["diverging"]) == "#f7f7f7"


def _heatmap_panel(scale: str = "sequential") -> PreparedPanel:
    # Two rows by three columns, the (row 1, column 2) cell left out on purpose.
    cells = [(0, 0, 0.1), (1, 0, 0.5), (2, 0, 0.9), (0, 1, 0.3), (1, 1, 0.7)]
    return _panel(
        "heatmap",
        tuple(_mark(f"r{y}c{x}", float(x), float(y), value=value) for x, y, value in cells),
        x_categories=("alpha", "beta", "a column label much longer than eighteen"),
        y_categories=("first row", "second row"),
        color_scale=scale,
    )


class TestHeatmapRendering:
    def test_the_matrix_is_built_from_the_marks_with_a_gap_where_none_is(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_heatmap_panel()).unwrap()
        heat = figure.data[0]
        assert heat.type == "heatmap"
        assert [list(row) for row in heat.z] == [[0.1, 0.5, 0.9], [0.3, 0.7, None]]
        assert [list(row) for row in heat.hovertext][1][:2] == [
            "r1c0 stands for something",
            "r1c1 stands for something",
        ]

    def test_row_zero_is_at_the_top(self) -> None:
        """Plotly's y axis grows upward; the app draws row 0 first."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(_heatmap_panel()).unwrap()
        assert figure.layout.yaxis.autorange == "reversed"
        assert list(figure.layout.yaxis.ticktext) == ["first row", "second row"]

    def test_the_colourscale_is_the_shared_stops_over_the_shared_domain(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_heatmap_panel("diverging")).unwrap()
        heat = figure.data[0]
        assert [stop for _, stop in heat.colorscale] == list(COLOR_SCALES["diverging"])
        assert [position for position, _ in heat.colorscale] == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert (heat.zmin, heat.zmax) == (-0.9, 0.9)

    def test_long_column_labels_are_cut_like_the_apps(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_heatmap_panel()).unwrap()
        assert list(figure.layout.xaxis.ticktext)[2] == "a column label mu…"


# ------------------------------------------------------------- distribution --


class TestDistributionRules:
    def test_quartiles_by_linear_interpolation(self) -> None:
        """The sample panelLayout.test.ts pins too: [1, 2, 3, 4, 10]."""
        assert _box_stats([10, 1, 3, 2, 4]) == {"min": 1, "q1": 2, "median": 3, "q3": 4, "max": 10}

    def test_quartiles_between_observations(self) -> None:
        """[1, 2, 3, 4]: positions 0.75, 1.5, 2.25 -- numpy's default."""
        stats = _box_stats([1, 2, 3, 4])
        assert (stats["q1"], stats["median"], stats["q3"]) == (1.75, 2.5, 3.25)
        numpy = pytest.importorskip("numpy")
        assert _quantile([1.0, 2.0, 3.0, 4.0], 0.25) == float(numpy.quantile([1, 2, 3, 4], 0.25))

    def test_whiskers_reach_the_extremes_not_one_and_a_half_iqr(self) -> None:
        assert _box_stats([1, 2, 3, 4, 10])["max"] == 10

    def test_jitter_is_deterministic_and_within_a_quarter_row(self) -> None:
        assert _jitter(0) == -0.25
        assert _jitter(1) == pytest.approx(0.059017, abs=1e-6)
        assert _jitter(2) == pytest.approx(-0.131966, abs=1e-6)
        assert all(-0.25 <= _jitter(index) < 0.25 for index in range(200))

    def test_jitter_is_numbered_within_each_row(self) -> None:
        marks = (_mark("a", 1.0, 0.0), _mark("b", 2.0, 1.0), _mark("c", 3.0, 0.0))
        assert _jitter_offsets(marks) == [_jitter(0), _jitter(0), _jitter(1)]


def _distribution_panel() -> PreparedPanel:
    values = (1.0, 2.0, 3.0, 4.0, 10.0)
    marks = tuple(_mark(f"a{index}", value, 0.0, group="G") for index, value in enumerate(values))
    marks += (_mark("b0", 5.0, 1.0, group="H"), _mark("b1", 7.0, 1.0, group="H"))
    return _panel("distribution", marks, y_categories=("row a", "row b"), groups=("G", "H"))


class TestDistributionRendering:
    def test_the_box_is_handed_to_plotly_precomputed(self) -> None:
        """So plotly's own quartile method never decides a box edge."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(_distribution_panel()).unwrap()
        box = figure.data[0]
        assert box.type == "box" and box.orientation == "h"
        assert list(box.q1) == [2.0, 5.5]
        assert list(box.median) == [3.0, 6.0]
        assert list(box.q3) == [4.0, 6.5]
        assert list(box.lowerfence) == [1.0, 5.0]
        assert list(box.upperfence) == [10.0, 7.0]

    def test_every_point_is_drawn_at_its_row_plus_its_jitter(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_distribution_panel()).unwrap()
        points = {trace.name: trace for trace in figure.data if trace.type == "scatter"}
        assert list(points) == ["G", "H"], "one trace per group, declared order"
        assert list(points["G"].x) == [1.0, 2.0, 3.0, 4.0, 10.0]
        assert list(points["G"].y) == [_jitter(index) for index in range(5)]
        assert list(points["H"].y) == [1.0 + _jitter(0), 1.0 + _jitter(1)]
        assert figure.layout.yaxis.autorange == "reversed"


# ---------------------------------------------------------------- positions --


class TestPositionsRendering:
    def _panel(self) -> PreparedPanel:
        marks = (
            _mark("d0:1", 0.1, 0.0, group="war"),
            _mark("d0:2", 0.4, 0.0, group="peace"),
            _mark("d1:1", 0.2, 1.0, group="war"),
        )
        return _panel("positions", marks, y_categories=("1934", "1942"), groups=("war", "peace"))

    def test_ticks_per_group_in_their_rows(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert [trace.name for trace in figure.data] == ["war", "peace"]
        assert all(trace.marker.symbol == "line-ns-open" for trace in figure.data)
        assert list(figure.data[0].y) == [0.0, 1.0]
        assert figure.layout.yaxis.autorange == "reversed"

    def test_the_axis_spans_the_whole_document(self) -> None:
        """Hits only in the first half still show an empty second half."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert list(figure.layout.xaxis.range) == [0.0, 1.0]


# ------------------------------------------------------------------ network --


def _network_panel(style: str = "straight") -> PreparedPanel:
    nodes = (
        _mark("war", 0.0, 0.0, group="A", size=40.0, labelled=True),
        _mark("peace", 1.0, 1.0, group="A", size=10.0),
        _mark("treaty", 0.5, 0.2, group="B"),
    )
    edges = (
        PanelEdge(key="war~peace", source="war", target="peace", weight=2.0, evidence=_evidence("war~peace")),
        PanelEdge(key="war~treaty", source="war", target="treaty", weight=4.0, evidence=_evidence("war~treaty")),
    )
    return _panel("network", nodes, edges=edges, edge_style=style, groups=("A", "B"))


class TestNetworkRules:
    def test_edge_width_is_linear_on_weight_over_the_heaviest(self) -> None:
        assert [_edge_width(weight, 4.0) for weight in (0.0, 2.0, 4.0)] == [1.0, 3.5, 6.0]
        assert _edge_width(3.0, 0.0) == 1.0, "no weight anywhere is the thinnest line, not a division by zero"

    def test_an_elbow_turns_at_the_source_x_and_the_target_y(self) -> None:
        source, target = _mark("s", 0.2, 0.9), _mark("t", 0.7, 0.1)
        assert _edge_points(source, target, "elbow") == [(0.2, 0.9), (0.2, 0.1), (0.7, 0.1)]
        assert _edge_points(source, target, "straight") == [(0.2, 0.9), (0.7, 0.1)]


class TestNetworkRendering:
    def test_edges_are_drawn_beneath_nodes_with_widths_from_weight(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_network_panel()).unwrap()
        lines = [trace for trace in figure.data if trace.mode == "lines"]
        assert [trace.line.width for trace in lines] == [3.5, 6.0]
        first_node = next(index for index, trace in enumerate(figure.data) if trace.name == "A")
        assert all(figure.data.index(trace) < first_node for trace in lines), "edges under nodes"

    def test_an_edge_hovers_with_its_own_evidence(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_network_panel()).unwrap()
        links = next(trace for trace in figure.data if trace.name == "links")
        assert list(links.hovertext) == ["war~peace stands for something", "war~treaty stands for something"]
        # A straight edge's hover point is its midpoint.
        assert (links.x[0], links.y[0]) == (0.5, 0.5)

    def test_an_elbow_edge_is_three_points_and_hovers_at_its_corner(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_network_panel("elbow")).unwrap()
        first = next(trace for trace in figure.data if trace.mode == "lines")
        assert list(first.x) == [0.0, 0.0, 1.0] and list(first.y) == [0.0, 1.0, 1.0]
        links = next(trace for trace in figure.data if trace.name == "links")
        assert (links.x[0], links.y[0]) == (0.0, 1.0)

    def test_the_layout_is_not_an_axis_and_y_zero_is_the_top(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_network_panel()).unwrap()
        assert figure.layout.xaxis.showticklabels is False
        assert list(figure.layout.yaxis.range) == [1.05, -0.05]

    def test_only_labelled_nodes_carry_text(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_network_panel()).unwrap()
        written = [text for trace in figure.data if trace.name in ("A", "B") for text in trace.text if text]
        assert written == ["war"]


# ---------------------------------------------------------- small multiples --


def _small_multiples_panel() -> PreparedPanel:
    marks: list[PanelMark] = []
    for facet, scale in (("Words", 1000.0), ("Sentences", 40.0), ("Words per sentence", 25.0), ("Commas", 3.0)):
        for year in (2001.0, 2002.0, 2005.0):
            marks.append(_mark(f"{facet}:{year}", year, scale * (year - 2000), facet=facet, group="median"))
        marks.append(_mark(f"{facet}:doc", 2003.5, scale * 2.5, facet=facet, group="speech"))
    return _panel(
        "small_multiples",
        tuple(marks),
        facets=("Words", "Sentences", "Words per sentence", "Commas"),
        groups=("median", "speech"),
        points_only=("speech",),
        line_gap=2.0,
    )


class TestSmallMultiples:
    def test_at_most_three_across(self) -> None:
        assert _facet_grid(1) == (1, 1)
        assert _facet_grid(3) == (1, 3)
        assert _facet_grid(4) == (2, 3)
        assert _facet_grid(7) == (3, 3)

    def test_x_is_shared_and_each_y_is_its_own(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_small_multiples_panel()).unwrap()
        layout = figure.layout
        anchors = {layout[name].matches or name.replace("axis", "") for name in ("xaxis", "xaxis2", "xaxis3", "xaxis4")}
        assert len(anchors) == 1, f"every x axis follows one: {anchors}"
        assert all(layout[name].matches is None for name in ("yaxis", "yaxis2", "yaxis3", "yaxis4"))

    def test_one_titled_subplot_per_facet_in_declared_order(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_small_multiples_panel()).unwrap()
        titles = [a.text for a in figure.layout.annotations][:4]
        assert titles == ["Words", "Sentences", "Words per sentence", "Commas"]

    def test_line_gap_breaks_lines_and_points_only_groups_are_never_joined(self) -> None:
        """2002 to 2005 is wider than line_gap=2: a break. The single speech
        is a point, not part of a line."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(_small_multiples_panel()).unwrap()
        words = [trace for trace in figure.data if trace.xaxis == "x"]
        median = next(trace for trace in words if trace.name == "median")
        assert list(median.x) == [2001.0, 2002.0, None, 2005.0]
        speech = next(trace for trace in words if trace.name == "speech")
        assert speech.mode == "markers"

    def test_a_group_appears_once_in_the_legend(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(_small_multiples_panel()).unwrap()
        shown = [trace.name for trace in figure.data if trace.showlegend]
        assert shown == ["median", "speech"]


# -------------------------------------------------------------- line_series --


class TestLineSeriesGap:
    def _panel(self, **extra: Any) -> PreparedPanel:
        marks = tuple(
            _mark(f"m{index}", x, float(index), group="war") for index, x in enumerate((1990.1, 1990.6, 1992.0))
        )
        return _panel("line_series", marks, groups=("war",), **extra)

    def test_the_default_gap_is_one(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert list(figure.data[0].x) == [1990.1, 1990.6, None, 1992.0]

    def test_a_builder_may_widen_it(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel(line_gap=2.0)).unwrap()
        assert list(figure.data[0].x) == [1990.1, 1990.6, 1992.0]

    def test_points_only_groups_are_markers(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel(points_only=("war",))).unwrap()
        assert figure.data[0].mode == "markers"
        assert None not in list(figure.data[0].x)


@pytest.mark.parametrize(
    "panel",
    [_heatmap_panel(), _distribution_panel(), _network_panel(), _small_multiples_panel()],
    ids=["heatmap", "distribution", "network", "small_multiples"],
)
def test_every_new_shape_renders_html_with_its_caption(panel: PreparedPanel) -> None:
    pytest.importorskip("plotly")
    result = panel_html(panel, offline=False)
    assert result.ok, [str(d) for d in result.diagnostics]
    assert "s.csv" in result.unwrap()
