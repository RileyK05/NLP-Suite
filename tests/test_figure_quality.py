"""Every figure reads correctly, checked across the whole registry.

The problems a reader found by looking -- "-0" ticks, labels printed over
labels, a caption naming a measure nobody picked, "sentence 7" as a label,
eighty rows in five hundred pixels -- came in classes
(docs/FIGURE_QUALITY_PLAN.md, C1-C8). These tests check the classes, not the
instances, over every registered panel with every choice of every parameter,
so a new panel is covered the day it is registered.

Examples are the real 87-speech outputs trimmed to twelve speeches
(``tests/fixtures/figures``). A panel whose tool could not run on the audit
machine is listed in ``NO_EXAMPLE`` with the reason; its family test covers
its logic, and the list is checked so it cannot hide a panel that does have
an example.
"""

from __future__ import annotations

from functools import cache
import io
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from core.viz.figure_lint import lint_definition, lint_prepared
from core.viz.panels import PANELS, best_table, fit_to_rows
from core.viz.panelspec import PanelDefinition, Provenance

FIXTURES = Path(__file__).parent / "fixtures" / "figures"

#: Panels with no real example, and why. Each is covered by its family test.
NO_EXAMPLE: dict[str, str] = {
    **{
        name: "neural sentiment backends need models not installed on the audit machine"
        for name in (
            f"sentiment_neural_{backend}_{kind}"
            for backend in ("bert", "spacy", "stanza", "corenlp")
            for kind in ("sentences", "documents", "tone_over_time", "tone_by_group", "tone_all_measures")
        )
    },
    **{
        name: "lexicon or model data not installed on the audit machine"
        for name in (
            "sentiment_anew_valence",
            "sentiment_anew_over_time",
            "sentiment_anew_by_group",
            "sentiment_anew_all_measures",
            "sentiment_swn_over_time",
            "sentiment_swn_by_group",
            "sentiment_swn_all_measures",
            "sentiment_hedonometer_over_time",
            "sentiment_hedonometer_by_group",
            "style_concreteness_by_group",
            "style_iconicity_by_group",
            "verbnet_ranked_classes",
            "framenet_ranked_frames",
            "symbolic_ranked_spaces",
            "symbolic_ranked_actors",
            "wordnet_ranked_categories",
            "gender_annotator_mentions_over_time",
            "word2vec_bert_neighbours",
            "word2vec_gensim_neighbours",
            "mallet_document_topics",
            "mallet_topic_terms",
        )
    },
    **{
        name: "a table tool: its input is a user's own CSV, so there is no corpus run to trim"
        for name in (
            "table_chi2_residuals",
            "table_crosstab_counts",
            "table_crosstab_composition",
            "table_keyness_keyness",
            "stats_categorical_residuals",
            "stats_categorical_counts",
            "stats_categorical_composition",
            "stats_categorical_keyness",
            "table_mw_medians",
            "table_kw_medians",
            "table_kw_posthoc",
            "stats_groups_medians",
            "stats_groups_posthoc",
            "table_trend_trend",
            "stats_trends_trend",
            "csv_stats_correlations",
        )
    },
    "doc_duplicates_graph": "the real corpus has no near-duplicate pair at the default threshold",
}


@cache
def _examples() -> dict[str, list[pd.DataFrame]]:
    by_tool: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(FIXTURES.glob("*.parquet")):
        tool = path.stem.split("__", 1)[0]
        by_tool.setdefault(tool, []).append(pd.read_parquet(path))
    return by_tool


def example_for(definition: PanelDefinition) -> pd.DataFrame | None:
    """The fixture the app would choose (``panels.best_table``)."""
    frames = [frame for frame in _examples().get(definition.tool, []) if not frame.empty]
    name = best_table(definition, [(str(index), list(frame.columns)) for index, frame in enumerate(frames)])
    return frames[int(name)] if name is not None else None


def variants(definition: PanelDefinition) -> list[dict[str, Any]]:
    """Defaults, then each other choice of each choice parameter, then each
    boolean flipped: the settings a reader can actually reach."""
    out: list[dict[str, Any]] = [{}]
    for param in definition.params:
        if param.type == "choice":
            out += [{param.name: choice} for choice in param.choices if choice != param.default]
        elif param.type == "bool":
            out.append({param.name: not param.default})
    return out


WITH_EXAMPLE = [d for d in PANELS if d.name not in NO_EXAMPLE]


def test_every_panel_has_an_example_or_a_stated_reason() -> None:
    missing = [d.name for d in WITH_EXAMPLE if example_for(d) is None]
    assert not missing, f"no fixture for {missing}; add one to tests/fixtures/figures or to NO_EXAMPLE"
    stale = [name for name in NO_EXAMPLE if name not in {d.name for d in PANELS}]
    assert not stale, f"NO_EXAMPLE names panels that no longer exist: {stale}"
    hidden = [
        d.name
        for d in PANELS
        if d.name in NO_EXAMPLE and example_for(d) is not None and d.name != "doc_duplicates_graph"
    ]
    assert not hidden, f"these have an example now; take them out of NO_EXAMPLE: {hidden}"


@pytest.mark.parametrize("definition", PANELS, ids=lambda d: d.name)
def test_static_text_describes_every_setting(definition: PanelDefinition) -> None:
    """C5: the summary shown before drawing must not describe the default as
    if it were the figure (it said "MTLD" with TTR picked)."""
    assert [str(f) for f in lint_definition(definition)] == []


@pytest.mark.parametrize("definition", WITH_EXAMPLE, ids=lambda d: d.name)
def test_every_reachable_setting_draws_a_readable_figure(definition: PanelDefinition) -> None:
    """C5, C6, C8 over every setting: text follows the parameters, labels
    name things, rows are tall enough, nothing is NaN, extremes show both
    ends."""
    frame = example_for(definition)
    assert frame is not None
    problems = []
    for variant in variants(definition):
        params = definition.defaults() | variant
        result = definition.build(frame, params, Provenance(tool=definition.tool, panel=definition.name, source="t"))
        if result.value is None:
            problems.append(f"{variant}: refused: {'; '.join(d.message for d in result.diagnostics)[:160]}")
            continue
        prepared = fit_to_rows(result.unwrap())
        problems += [
            f"{variant}: {finding}"
            for finding in lint_prepared(prepared, definition, params, diagnostics=result.diagnostics)
        ]
    assert problems == []


@pytest.mark.parametrize("definition", WITH_EXAMPLE, ids=lambda d: d.name)
def test_the_publication_figure_has_no_overprinted_text(definition: PanelDefinition) -> None:
    """C1, C3, C4 measured on the drawing itself: no text over text, no "-0"."""
    pytest.importorskip("seaborn")
    from core.viz.static import render_static

    frame = example_for(definition)
    assert frame is not None
    built = definition.build(frame, definition.defaults(), Provenance(tool=definition.tool, panel=definition.name))
    image = render_static(fit_to_rows(built.unwrap()), "png", dpi=60)
    assert image.value is not None, [str(d) for d in image.diagnostics]
    assert image.unwrap()[:8] == b"\x89PNG\r\n\x1a\n"
    assert [str(d) for d in image.diagnostics if d.code.startswith("STATIC_")] == []


# ------------------------------------------------------------- the pieces --


class TestTicks:
    def test_no_negative_zero_and_one_precision(self) -> None:
        from core.viz.static.text import tick_labels

        assert tick_labels([-0.5, -0.0, 0.5, 1.0]) == ["-0.5", "0", "0.5", "1.0"]
        assert tick_labels([1.5, 2.0, 2.5]) == ["1.5", "2.0", "2.5"]
        assert tick_labels([1980, 2000, 2020], plain=True) == ["1980", "2000", "2020"]
        assert tick_labels([0, 500_000, 1_000_000]) == ["0", "0.5M", "1.0M"]

    def test_counts_print_as_counts(self) -> None:
        from core.viz.static.text import format_value

        assert [format_value(v) for v in (85.0, 9.0, 0.25, -0.0)] == ["85", "9", "0.25", "0"]

    def test_log_ticks_are_round_quantities(self) -> None:
        pytest.importorskip("matplotlib")
        from core.viz.static.shapes import log_tick_values

        assert log_tick_values(1.0, 2.0) == [10.0, 20.0, 50.0, 100.0]
        assert log_tick_values(0.5, 4.2) == [10.0, 100.0, 1000.0, 10000.0]


class TestSpread:
    def test_the_band_uses_the_rolling_medians_windows(self) -> None:
        from core.viz.panel_helpers import rolling_median
        from core.viz.static.stats import rolling_band

        xs = list(range(9))
        ys = [1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 9.0]
        band = rolling_band(xs, ys, 3)
        medians = rolling_median(xs, ys, 3)
        assert len(band) == len(medians) == 9
        assert all(lo <= median <= hi for (_, lo, hi), (_, median) in zip(band, medians, strict=True))

    def test_decades_come_from_short_labels(self) -> None:
        from core.viz.static.stats import decade_of

        assert decade_of("1934 Roosevelt") == "1930s"
        assert decade_of("notes") == ""


class TestColours:
    def test_a_remainder_group_is_grey_and_takes_no_colour(self) -> None:
        from core.viz.static.style import OKABE_ITO, REMAINDER, group_colors

        colors = group_colors(["war · peace", "(smaller clusters)", "tax · budget"])
        assert colors["(smaller clusters)"] == REMAINDER
        assert colors["tax · budget"] == OKABE_ITO[1], "the next real group keeps the next colour"


class TestCommunities:
    def test_linked_words_share_a_community_named_by_its_heaviest(self) -> None:
        from core.viz.panel_helpers import OTHER_COMMUNITY, communities

        nodes = ["war", "peace", "army", "tax", "budget", "spending", "alone"]
        edges = [("war", "peace", 5.0), ("war", "army", 4.0), ("peace", "army", 3.0)]
        edges += [("tax", "budget", 5.0), ("tax", "spending", 4.0), ("budget", "spending", 3.0)]
        mass = {"war": 30, "peace": 20, "army": 10, "tax": 25, "budget": 15, "spending": 5, "alone": 1}
        names = communities(nodes, edges, mass)
        assert names["war"] == names["army"] == "war · peace"
        assert names["tax"] == names["spending"] == "tax · budget"
        assert names["alone"] == OTHER_COMMUNITY
        assert names == communities(nodes, edges, mass), "deterministic"


class TestRenderer:
    @pytest.fixture
    def panel(self) -> Any:
        from core.viz.panelspec import Evidence, PanelMark, PreparedPanel

        evidence = Evidence(scope="rows", filters=(("k", "v"),), count=1, describe="d")
        marks = tuple(
            PanelMark(key=f"m{i}", label=f"word {i}", x=float(i), y=float(i % 3), evidence=evidence, labelled=True)
            for i in range(8)
        )
        return PreparedPanel(
            panel="p",
            shape="scatter_labelled",
            title="A title",
            subtitle="a subtitle",
            marks=marks,
            x_label="x",
            y_label="y",
            provenance=Provenance(tool="t", panel="p", source="s.csv"),
            data=pd.DataFrame({"k": ["v"]}),
        )

    @pytest.mark.parametrize(("fmt", "magic"), [("png", b"\x89PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")])
    def test_every_format(self, panel: Any, fmt: str, magic: bytes) -> None:
        pytest.importorskip("seaborn")
        from core.viz.static import render_static

        assert render_static(panel, fmt, dpi=50).unwrap()[: len(magic)] == magic

    def test_svg_is_deterministic(self, panel: Any) -> None:
        pytest.importorskip("seaborn")
        from core.viz.static import render_static

        assert render_static(panel, "svg").unwrap() == render_static(panel, "svg").unwrap()

    def test_the_caption_names_the_source_and_this_renderer(self, panel: Any) -> None:
        pytest.importorskip("seaborn")
        from core.viz.static import render_static

        svg = render_static(panel, "svg").unwrap().decode("utf-8")
        assert "s.csv" in svg
        assert "publication renderer" in svg

    def test_an_unknown_format_is_refused(self, panel: Any) -> None:
        from core.viz.static import render_static

        assert render_static(panel, "gif").diagnostics[0].code == "STATIC_BAD_FORMAT"

    def test_the_lint_sees_overprinted_text(self) -> None:
        pytest.importorskip("matplotlib")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from core.viz.static.lint import lint_figure

        fig = Figure(figsize=(3, 2))
        FigureCanvasAgg(fig)
        fig.text(0.1, 0.5, "one label here")
        fig.text(0.12, 0.5, "another label")
        fig.canvas.draw()
        assert [p.code for p in lint_figure(fig)] == ["TEXT_OVERLAP"]
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
