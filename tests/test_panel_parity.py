"""The desktop's panel drawing and the engine's must agree about the same facts.

``desktop/src/panelLayout.ts`` is a second implementation of panel geometry:
the app draws marks itself (plain SVG, so every mark stays clickable) while
the engine renders the published figure with plotly. Two copies of one fact
across a language boundary is how this project has been bitten before — a
chart kind listed twice drifted apart until ``test_chart_kind_parity.py``
started reading the TypeScript rather than trusting it. These tests are the
panel answer to the same failure, and they read the TypeScript too.

Three facts are load-bearing enough to check:

* **The shape vocabulary.** The app draws exactly what
  ``IMPLEMENTED_SHAPES`` names — no more (a shape it drew but the engine
  refused would publish a figure the engine cannot reproduce) and no fewer
  (a shape nothing draws is refused up front, never half-served).
* **The palette.** Both sides colour groups from ``OKABE_ITO`` by index into
  the panel's declared draw order. The chart layer's copy is already pinned
  by ``test_chart_kind_parity.py::test_the_palette_is_the_engine_palette``;
  this extends the same rule to the panel layer, which re-uses that constant
  rather than declaring its own.
* **The stream stacking rule.** A group missing at an x value contributes
  zero rather than creating a gap — because a straight line through a missing
  period invents data and, on a stack, lifts every band above it. Python-side
  it is pinned by ``TestStream``; TypeScript-side by ``panelLayout.test.ts``;
  here it is pinned as *the same rule in both files*, so a change to either
  side without the other fails here.

The shapes added later each bring a rule of the same kind — a heatmap's
colour stops, interpolation and domain; a distribution's quartiles and
jitter; a network's edge widths and elbow path; the small-multiples grid;
the line gap. For each, the Python behaviour is asserted here, the
TypeScript is read for the same rule, and the TypeScript *test* is read for
the same sample, so neither side can move alone.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from core.viz.panel_plotters import IMPLEMENTED_SHAPES
from core.viz.panelspec import PANEL_SHAPES

SOURCE = Path(__file__).resolve().parents[1] / "desktop" / "src"
LAYOUT = SOURCE / "panelLayout.ts"
CANVAS = SOURCE / "PanelCanvas.tsx"
CANVAS_TESTS = SOURCE / "PanelCanvas.test.tsx"


def ts(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"{path.name} read as empty; the parity reader is broken, not the source"
    return text


def ts_body(source: str, start_marker: str, end: str = "];") -> str:
    """The body of a top-level declaration, asserting the slice found content."""
    assert start_marker in source, f"{start_marker!r} not found; the extractor needs updating"
    body = source.split(start_marker, 1)[1].split(end, 1)[0]
    assert body.strip(), f"the slice from {start_marker!r} is empty; the extractor needs updating"
    return body


class TestTheTwoShapeListsAgree:
    """The shapes the app draws are the shapes the engine renders. Both."""

    def _dispatch_shapes(self) -> set[str]:
        """Shape names the TS layout dispatches on, from its switch."""
        source = ts(LAYOUT)
        start = source.index("switch (prepared.shape)")
        body = source[start : source.index("default:", start)]
        return set(re.findall(r'case "([a-z_]+)":', body))

    def _canvas_shapes(self) -> set[str]:
        """Shape names the TS canvas renders, from its conditionals."""
        source = ts(CANVAS)
        return set(re.findall(r'layout\.shape === "([a-z_]+)"', source))

    def test_the_app_draws_every_shape_the_engine_draws(self) -> None:
        drawn = self._dispatch_shapes()
        assert drawn == set(IMPLEMENTED_SHAPES), (
            f"panelLayout draws {sorted(drawn)} but the engine implements {sorted(IMPLEMENTED_SHAPES)}; "
            "a shape missing from the app is a figure a reader is offered and cannot see"
        )

    def test_the_canvas_has_a_figure_for_every_layout_shape(self) -> None:
        assert self._canvas_shapes() == set(IMPLEMENTED_SHAPES), (
            "PanelCanvas renders a subset of what panelLayout decides; a shape "
            "prepared but never drawn is the half-served panel the contract forbids"
        )

    def test_the_app_claims_nothing_beyond_the_engine(self) -> None:
        """A shape the vocabulary names but nothing implements must reach the
        app as a refusal, so the app must not pretend to draw it."""
        drawn = self._dispatch_shapes()
        unimplemented = set(PANEL_SHAPES) - set(IMPLEMENTED_SHAPES)
        assert not (drawn & unimplemented), (
            f"panelLayout draws {sorted(drawn & unimplemented)}, which no renderer draws; "
            "unimplemented shapes must stay refused, not half-served"
        )

    def test_the_refusal_stand_in_is_a_shape_nothing_draws(self) -> None:
        """The app's refusal tests need a shape nothing draws. Every declared
        shape is now implemented, so they use one outside the vocabulary; if
        the vocabulary ever grows a shape of that name, this says to pick
        another stand-in rather than quietly testing a case that cannot
        happen."""
        stand_in = "sankey"
        assert stand_in not in PANEL_SHAPES and stand_in not in IMPLEMENTED_SHAPES
        for path in (LAYOUT.parent / "panelLayout.test.ts", CANVAS_TESTS):
            assert f'shape: "{stand_in}"' in ts(path), f"{path.name} no longer tests the refusal with {stand_in!r}"


class TestThePaletteIsShared:
    """One palette across the boundary, by declared group order on both sides."""

    def test_the_panel_layer_reuses_the_chart_palette(self) -> None:
        """panelLayout imports OKABE_ITO rather than restating it — the same
        rule test_chart_kind_parity pins for chartLayout."""
        source = ts(LAYOUT)
        assert re.search(r'import \{[^}]*OKABE_ITO[^}]*\} from "\./chartLayout"', source), (
            "panelLayout should import the chart layer's OKABE_ITO instead of declaring a second copy that can drift"
        )

    def test_the_canvas_test_file_pins_the_same_palette(self) -> None:
        source = ts(LAYOUT)
        body = ts_body(source, "export function groupColors", end="\n}\n")
        # Colour by index into the declared order, wrapping the palette — the
        # same rule panel_plotters._group_color_map applies.
        assert "OKABE_ITO" in body
        assert "% OKABE_ITO.length" in body

    def test_an_ungrouped_panel_gets_the_first_colour_on_both_sides(self) -> None:
        """Behavioural on the Python side, structural on the TypeScript side
        (panelLayout.test.ts pins the TS behaviour). The textual checks above
        passed while the two sides disagreed about exactly this case: the app
        drew the intertopic map orange and its export blue."""
        from core.viz.panel_plotters import _group_color_map
        from core.viz.plotters import OKABE_ITO

        assert _group_color_map(()) == {"(all)": OKABE_ITO[0]}
        body = ts_body(ts(LAYOUT), "export function groupColors", end="\n}\n")
        assert "UNGROUPED" in body, "groupColors must colour the implicit (all) group, as the engine does"
        assert not re.search(r"hash\(", ts(LAYOUT)), (
            "undeclared groups fall back to the first colour, as the engine does"
        )

    def test_the_engine_palette_is_what_both_sides_name(self) -> None:
        """The chart-layer copy is pinned against the engine by
        test_chart_kind_parity; assert the import chain still holds so the
        panel layer inherits that pin rather than bypassing it."""
        layout = ts(SOURCE / "chartLayout.ts")
        assert re.search(r"export const OKABE_ITO = \[", layout)


class TestStreamStackingAgrees:
    """Zero-fill for a missing bucket, stacked in declared group order."""

    def test_the_python_side_fills_missing_buckets_with_zero(self) -> None:
        """The engine renderer fills gaps with 0.0 — asserted behaviourally,
        on a real prepared panel, so this test dies if the rule ever changes
        rather than passing against a stale quotation."""
        # Mirrors TestStream's fixture: Topic 1 is absent in 1940 on purpose.
        import pandas as pd

        from core.viz.panel_plotters import build_panel_figure
        from core.viz.panelspec import Evidence, PanelMark, PreparedPanel, Provenance

        def mark(key: str, x: float, y: float, group: str) -> PanelMark:
            return PanelMark(
                key=key,
                label=group,
                x=x,
                y=y,
                group=group,
                evidence=Evidence(scope="rows", filters=(("k", key),), count=1, describe=key),
            )

        prepared = PreparedPanel(
            panel="p",
            shape="stream",
            title="t",
            marks=(
                mark("1930:0", 1930.0, 0.6, "Topic 0"),
                mark("1940:0", 1940.0, 0.4, "Topic 0"),
                mark("1930:1", 1930.0, 0.4, "Topic 1"),
            ),
            x_label="x",
            y_label="y",
            provenance=Provenance(tool="t", panel="p", source="s.csv"),
            data=pd.DataFrame({"k": ["1930:0", "1940:0", "1930:1"]}),
            groups=("Topic 0", "Topic 1"),
        )
        figure = build_panel_figure(prepared).unwrap()
        topic_one = next(trace for trace in figure.data if trace.name == "Topic 1")
        # The missing 1940 cell contributes zero; the band spans the axis.
        assert list(topic_one.y) == [0.4, 0.0]
        assert list(topic_one.x) == [1930.0, 1940.0]

    def test_the_typescript_side_states_the_same_rule(self) -> None:
        """panelLayout.ts must fill a missing (bucket, group) cell with zero
        and stack in the declared order. Read structurally: the stacking loop
        reads the declared group order, and a missing mark contributes 0."""
        source = ts(LAYOUT)
        start = source.index("function layoutStream(")
        body = source[start : source.index("\n}", source.index("return { shape:", start))]
        # Stacks in the declared order: the loop runs over `ordered`, which
        # groupOrder builds from prepared.groups.
        assert "for (const group of ordered)" in body
        # A missing mark contributes zero, never a gap.
        assert "?? null" in body
        assert "contribution = mark ? mark.y : 0" in body
        # The band spans the whole axis (one slot per axis value).
        assert "for (let index = 0; index < axis.length; index++)" in body

    def test_the_typescript_test_pins_the_same_fixture(self) -> None:
        """panelLayout.test.ts exercises the gap case with the same shape of
        fixture TestStream uses (a group absent at one x). If the TS test
        stops covering the gap, this fails — the parity of the *tests* is
        part of the parity of the rule."""
        source = ts(LAYOUT.parent / "panelLayout.test.ts")
        assert "fills a missing bucket with zero" in source, (
            "the zero-fill test was renamed or removed; the TS side of the stacking rule must stay pinned"
        )
        assert "?? null" in source or "marks[1]).toBeNull" in source


class TestRankedBarsOrderStaysSVGNative:
    """Rank 0 is the first row on the SVG side — deliberately unlike Plotly.

    The Python renderer reverses its category array because Plotly's category
    axis grows upward. SVG's y grows downward, so copying that reversal here
    would flip the panel: the best term at the bottom, reading as the worst.
    """

    def test_the_svg_layout_does_not_copy_the_plotly_reversal(self) -> None:
        source = ts(LAYOUT)
        start = source.index("function layoutRankedBars(")
        body = source[start : source.index("\n}", source.index("return { shape:", start))]
        # Rows are sorted ascending by rank (rank 0 first)…
        assert re.search(r"\.sort\(\(a, b\) => a\.rank - b\.rank", body)
        # …and nothing re-sorts descending anywhere in the ranked-bars path.
        assert "b.rank - a.rank" not in body, (
            "the SVG side reversed the ranking; rank 0 would draw at the "
            "bottom, which is the Plotly axis convention and wrong here"
        )

    def test_the_canvas_comments_state_the_rule(self) -> None:
        """The reversal trap is documented where the drawing happens, so the
        next reader who reaches for the Plotly answer meets the warning."""
        source = ts(CANVAS)
        assert "no reversal" in source, (
            "PanelCanvas should state, where bars are drawn, that rank order "
            "is draw order on an SVG axis — the trap the Plotly renderer had "
            "to solve with a reversal"
        )


class TestRibbonGeometryAgrees:
    """A ribbon's x is where a segment starts and its size is how wide it
    is; row 0 is the band at the top on both sides."""

    def test_both_sides_state_the_segment_geometry(self) -> None:
        # Python: the plotly trace reads `base` (start) and `x` (width).
        from core.viz import panel_plotters as plotters

        source = plotters.__dict__.get("_ribbon").__doc__ or ""
        assert "start" in source and "size" in source
        # TypeScript: the layout places segments by (x, size) pairs.
        layout_source = ts(LAYOUT)
        start = layout_source.index("function layoutRibbon(")
        body = layout_source[start : layout_source.index("\n}", start)]
        assert "mark.size" in body, "the ribbon layout must read the segment's size"
        assert "a.x - b.x" in body, "segments are drawn in position order within a band"

    def test_the_svg_side_draws_row_order_directly(self) -> None:
        layout_source = ts(LAYOUT)
        start = layout_source.index("function layoutRibbon(")
        body = layout_source[start : layout_source.index("return { shape:", start)]
        assert ".sort((a, b) => a.row - b.row)" in body, (
            "ribbon bands must be drawn row-ascending (row 0 top); the reversal is Plotly's problem, not the SVG's"
        )


LAYOUT_TESTS = LAYOUT.parent / "panelLayout.test.ts"


def ts_function(source: str, name: str) -> str:
    """The text of one top-level TypeScript function, by name."""
    start = source.index(f"function {name}(")
    return source[start : source.index("\n}\n", start)]


class TestHeatmapColoursAgree:
    """One scale, one interpolation, one domain rule, on both sides."""

    def test_the_app_declares_the_engines_stops(self) -> None:
        from core.viz.panelspec import COLOR_SCALES

        body = ts_body(ts(LAYOUT), "export const COLOR_SCALES", end="};")
        declared = {
            name: re.findall(r'"(#[0-9a-fA-F]{6})"', stops) for name, stops in re.findall(r"(\w+): \[([^\]]*)\]", body)
        }
        assert declared == {name: list(stops) for name, stops in COLOR_SCALES.items()}, (
            "panelLayout.COLOR_SCALES drifted from panelspec.COLOR_SCALES; the app and the export "
            "would colour the same cell differently"
        )

    def test_both_sides_round_channels_half_up(self) -> None:
        """Python's round() is half-to-even; the app's Math.floor(c + 0.5)
        is half-up. The engine must use the app's rule, and the halfway
        sample both test files pin proves it does."""
        from core.viz.panel_plotters import _heat_color
        from core.viz.panelspec import COLOR_SCALES

        assert "Math.floor(a + (end[channel] - a) * fraction + 0.5)" in ts_function(ts(LAYOUT), "heatColor")
        assert _heat_color(0.125, 0.0, 1.0, COLOR_SCALES["sequential"]) == "#dfebf7"
        assert '"#dfebf7"' in ts(LAYOUT_TESTS), "the TS test must pin the same halfway colour"

    def test_the_diverging_domain_is_symmetric_on_both_sides(self) -> None:
        from core.viz.panel_plotters import _heat_domain

        assert _heat_domain([0.2, -0.4, 0.9], "diverging") == (-0.9, 0.9)
        body = ts_function(ts(LAYOUT), "heatDomain")
        assert "return [-largest, largest]" in body
        assert "[-0.9, 0.9]" in ts(LAYOUT_TESTS)


class TestDistributionRulesAgree:
    def test_both_sides_pin_the_same_quartile_sample(self) -> None:
        from core.viz.panel_plotters import _box_stats

        assert _box_stats([1, 2, 3, 4, 10]) == {"min": 1, "q1": 2, "median": 3, "q3": 4, "max": 10}
        tests = ts(LAYOUT_TESTS)
        assert "boxStats([1, 2, 3, 4, 10])" in tests
        assert "q1: 1.75" in tests and "q3: 3.25" in tests, "the between-observations sample [1, 2, 3, 4]"

    def test_both_sides_interpolate_at_n_minus_one_times_p(self) -> None:
        assert "(ordered.length - 1) * p" in ts_function(ts(LAYOUT), "quantile")

    def test_the_jitter_constant_is_the_same_double(self) -> None:
        from core.viz import panel_plotters

        match = re.search(r"const JITTER_STEP = ([0-9.]+);", ts(LAYOUT))
        assert match, "panelLayout.ts no longer declares JITTER_STEP"
        assert float(match.group(1)) == panel_plotters._JITTER_STEP
        assert "Math.random" not in ts(LAYOUT), "jitter must be deterministic"

    def test_both_sides_pin_the_same_jitter_values(self) -> None:
        from core.viz.panel_plotters import _jitter

        tests = ts(LAYOUT_TESTS)
        for index, literal in ((1, "0.059017"), (2, "-0.131966")):
            assert round(_jitter(index), 6) == float(literal)
            assert literal in tests, f"panelLayout.test.ts does not pin jitter({index}) = {literal}"


class TestNetworkRulesAgree:
    def test_the_elbow_turns_at_source_x_target_y_on_both_sides(self) -> None:
        from core.viz import panel_plotters

        assert "(source.x, target.y)" in (panel_plotters._edge_points.__doc__ or "")
        assert "{ x: source.x, y: target.y }" in ts_function(ts(LAYOUT), "edgePoints")

    def test_both_sides_pin_the_same_edge_widths(self) -> None:
        from core.viz.panel_plotters import _edge_width

        assert [_edge_width(weight, 4.0) for weight in (0.0, 2.0, 4.0)] == [1.0, 3.5, 6.0]
        # Whitespace-insensitive: prettier wraps a short array across lines
        # (with a trailing comma) when the surrounding call is long.
        assert re.search(r"\[\s*1,\s*3\.5,\s*6,?\s*\]", ts(LAYOUT_TESTS)), (
            "panelLayout.test.ts does not pin edge widths [1, 3.5, 6]"
        )


class TestFacetGridAgrees:
    def test_at_most_three_across_on_both_sides(self) -> None:
        from core.viz.panel_plotters import _facet_grid

        assert _facet_grid(4) == (2, 3)
        assert "Math.min(3, count)" in ts_function(ts(LAYOUT), "facetGrid")
        assert "facetGrid(4)" in ts(LAYOUT_TESTS)


class TestLineGapAgrees:
    def test_neither_side_hard_codes_a_gap_of_one(self) -> None:
        """The gap was ``> 1`` in both renderers; a builder over dated
        documents needs its own, and both must read the same field."""
        body = ts_function(ts(LAYOUT), "lineSeries")
        assert "prepared.lineGap ?? 1" in body
        assert "mark.x - previous > gap" in body
        assert "prepared.pointsOnly ?? []" in body
        from core.viz import panel_plotters

        source = Path(panel_plotters.__file__).read_text(encoding="utf-8")
        assert "mark.x - previous > gap" in source
        assert "mark.x - previous > 1" not in source


class TestEvidenceReachesTheDrawing:
    """A click has to be answerable without a second round trip."""

    def test_the_layout_carries_evidence_through(self) -> None:
        source = ts(LAYOUT)
        # PlacedMark carries the evidence object, not a copy of its words.
        assert re.search(r"evidence: PanelEvidence", source)
        start = source.index("function placed(")
        body = source[start : source.index("\n}", start)]
        assert "evidence: mark.evidence" in body, (
            "the placed mark must carry the engine's own evidence object; "
            "recomposing the description in TypeScript would create a second "
            "wording that can drift from the hover and the export"
        )

    def test_the_canvas_uses_the_engine_description_for_aria(self) -> None:
        source = ts(CANVAS)
        assert '"aria-label": mark.evidence.describe' in source, (
            "a screen reader must hear the same sentence a hover shows, written once engine-side"
        )

    def test_the_canvas_draws_the_engine_caption(self) -> None:
        source = ts(CANVAS)
        assert re.search(r"\{prepared\.caption\}", source), (
            "the provenance caption must reach the page; the engine wrote it and the app does not get to paraphrase it"
        )


class TestNothingRestatesTheGate:
    """The engine refuses unimplemented shapes and bad params up front; the
    app must rely on that rather than growing its own list."""

    def test_the_app_does_not_keep_its_own_implemented_shapes_tuple(self) -> None:
        source = ts(LAYOUT)
        assert not re.search(r"(const|let)\s+IMPLEMENTED_SHAPES", source), (
            "panelLayout must dispatch on the shape the engine sent, not keep "
            "a second list of drawable shapes that can drift from the registry"
        )

    def test_the_app_does_not_hard_code_panel_names(self) -> None:
        for path in (LAYOUT, CANVAS, SOURCE / "PanelSection.tsx"):
            source = ts(path)
            assert "keyness_volcano" not in source, (
                f"{path.name} names a specific panel; controls come from the "
                "declarations, and a panel added to the engine must need no "
                "edit here"
            )


@pytest.mark.parametrize("shape", sorted(IMPLEMENTED_SHAPES))
def test_every_implemented_shape_has_a_python_and_a_typescript_renderer(shape: str) -> None:
    """Each side names every implemented shape in its own source, so adding a
    shape to one language without the other fails here by name."""
    assert shape in ts(LAYOUT), f"panelLayout.ts does not draw {shape}"
    assert shape in ts(CANVAS), f"PanelCanvas.tsx does not draw {shape}"
    import core.viz.panel_plotters as plotters

    assert shape in plotters.__dict__.get("IMPLEMENTED_SHAPES", ()), shape
