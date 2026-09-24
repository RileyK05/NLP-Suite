"""The panel seam: what the registry refuses, and what a prepared panel promises.

Two things are worth more than the rest here and are tested hardest.

**Evidence must resolve.** It is easy to attach a plausible-looking filter to
a mark and never check it selects anything. ``TestEvidenceResolves`` takes
each mark's filters back to the source frame and asserts they find exactly
the row the mark was built from — so the protocol is load-bearing rather
than decorative, which is the whole reason it exists.

**The caption must survive export.** A provenance line rendered in
surrounding HTML disappears the moment someone exports a PNG, which is
exactly when a reader needs it. ``TestCaptionSurvivesExport`` asserts it is
an annotation inside the figure and reaches ``to_plotly_json`` — what
kaleido actually renders from.

The rest is the gate in front of builders: unknown panels, missing columns,
misspelled parameters and out-of-range values must each fail with their own
named diagnostic, because "the figure did not appear" is not a bug report
anyone can act on.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from core.result import Result, Severity
from core.viz.panel_plotters import IMPLEMENTED_SHAPES, build_panel_figure, panel_html, panel_image_bytes
from core.viz.panels import PANELS, get_panel, panel_names, panels_for_tool, prepare_panel, validate_registry
from core.viz.panels_keyness import FREQ_A, FREQ_B, G2, LOG_RATIO, OVERREPRESENTED, WORD
from core.viz.panelspec import (
    PANEL_SHAPES,
    Annotation,
    Evidence,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
    Source,
)

PANEL = "keyness_volcano"

# freedom: strong evidence AND a large effect -- the upper corner.
# the:     strong evidence, no effect       -- the trap a G2 bar chart sets.
# tariff:  large effect, weak evidence      -- the other trap.
# war:     moderate on both, other group.
_ROWS: list[dict[str, Any]] = [
    {"word": "freedom", "a": 120, "b": 12, "g2": 88.4, "lr": 2.31, "in": "Group A"},
    {"word": "the", "a": 9000, "b": 8700, "g2": 41.2, "lr": 0.04, "in": "Group A"},
    {"word": "tariff", "a": 1, "b": 19, "g2": 2.1, "lr": -3.20, "in": "Group B"},
    {"word": "war", "a": 60, "b": 140, "g2": 22.8, "lr": -1.20, "in": "Group B"},
]


def keyness_frame(rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                WORD: row["word"],
                FREQ_A: row["a"],
                FREQ_B: row["b"],
                G2: row["g2"],
                "p-value": 0.0,
                LOG_RATIO: row["lr"],
                OVERREPRESENTED: row["in"],
            }
            for row in (rows if rows is not None else _ROWS)
        ]
    )


def prepared(params: dict[str, Any] | None = None, source: Source | None = None) -> PreparedPanel:
    result = prepare_panel(PANEL, keyness_frame(), params or {}, source=source or Source(path="keyness.csv"))
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def codes(result: Result[Any]) -> list[str]:
    return [d.code for d in result.diagnostics]


def find(result: Result[Any], code: str) -> Any:
    """The one diagnostic with this code. Panels routinely carry more than
    one (a filter notice beside a provenance warning), so a test that wants
    a specific message must ask for it by code rather than by position."""
    matching = [d for d in result.diagnostics if d.code == code]
    assert len(matching) == 1, f"expected one {code}, got {codes(result)}"
    return matching[0]


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_every_registered_panel_is_drawable(self) -> None:
        """A panel nothing can draw is a listing that fails when clicked."""
        assert validate_registry() == []

    def test_lookup_by_name_and_by_tool(self) -> None:
        assert get_panel(PANEL) is not None
        assert get_panel("no_such_panel") is None
        assert PANEL in panel_names()
        assert get_panel(PANEL) in panels_for_tool("keyness")
        assert panels_for_tool("nothing_registers_this") == ()

    def test_an_unknown_panel_names_the_ones_that_exist(self) -> None:
        result = prepare_panel("volcano", keyness_frame())
        assert not result.ok
        assert codes(result) == ["PANEL_UNKNOWN"]
        # A typo should be one step from a fix, not a dead end.
        assert PANEL in result.diagnostics[0].message

    def test_every_panel_belongs_to_a_tool_that_exists(self) -> None:
        """The desktop offers a panel by the tool name in a run's envelope.
        A panel naming a tool nothing registers is never offered anywhere,
        and nothing fails: three LDA panels shipped naming ``topic_model``
        (the live-bench name) instead of ``lda_gensim`` (the tool that writes
        runs), and every test passed while the desktop offered them to no
        one. The check lives here rather than in ``validate_registry`` because
        the tool registry imports this one, and the reverse import would
        cycle."""
        from core.profiler.registry import tool_names
        from desktop_backend.tables import TABLE_TOOLS

        # The desktop's table workflows write their envelopes under their own
        # names (``OutputWriter(tool="table_chi2")`` in ``run_table``), so a
        # panel for a table tool is reachable by that name too.
        registered = set(tool_names()) | set(TABLE_TOOLS)
        for definition in PANELS:
            assert definition.tool in registered, (
                f"{definition.name} names tool {definition.tool!r}, which is not registered; "
                "a run's envelope will never carry that name, so the panel is unreachable"
            )

    def test_every_panel_declares_what_it_cannot_show(self) -> None:
        for definition in PANELS:
            assert definition.notes, f"{definition.name} states no limits"
            assert definition.requires, f"{definition.name} validates no input"
            for param in definition.params:
                assert param.help.strip(), f"{definition.name}.{param.name} has no help"


# ---------------------------------------------------------------------------
# The gate in front of builders
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_an_empty_table_is_refused(self) -> None:
        result = prepare_panel(PANEL, keyness_frame().iloc[0:0])
        assert codes(result) == ["PANEL_EMPTY"]

    def test_a_missing_column_is_named(self) -> None:
        """Pointed at the wrong artifact, a panel says which column is absent
        rather than raising KeyError from somewhere inside pandas."""
        result = prepare_panel(PANEL, keyness_frame().drop(columns=[LOG_RATIO]))
        assert codes(result) == ["PANEL_MISSING_COLUMN"]
        assert LOG_RATIO in result.diagnostics[0].message
        assert result.diagnostics[0].context["missing"] == [LOG_RATIO]

    def test_a_misspelled_parameter_is_refused_not_ignored(self) -> None:
        """The quietest possible failure is a setting that does nothing."""
        result = prepare_panel(PANEL, keyness_frame(), {"label_top": 5})
        assert codes(result) == ["PANEL_UNKNOWN_PARAM"]
        assert "label-top" in result.diagnostics[0].message

    @pytest.mark.parametrize(
        ("params", "why"),
        [
            ({"label-top": "20"}, "text where a count belongs"),
            ({"label-top": True}, "a bool is an int in Python; it must not pass as one"),
            ({"label-top": -1}, "below the declared minimum"),
            ({"label-top": 10_000}, "above the declared maximum"),
            ({"significance": "0.5"}, "not one of the declared choices"),
            ({"label-by": "vibes"}, "not one of the declared choices"),
            ({"min-frequency": 1.5}, "a fraction where a count belongs"),
        ],
    )
    def test_declared_bounds_are_enforced_centrally(self, params: dict[str, Any], why: str) -> None:
        result = prepare_panel(PANEL, keyness_frame(), params)
        assert codes(result) == ["PANEL_BAD_PARAM"], why
        assert result.diagnostics[0].context["param"] == next(iter(params))

    def test_defaults_are_filled_in(self) -> None:
        panel = prepared()
        definition = get_panel(PANEL)
        assert definition is not None
        assert panel.provenance.params == definition.defaults()

    def test_a_builder_that_raises_is_contained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Panels are the newest surface here; a broken one must not take down
        a run that also produced good results."""
        from core.viz import panels as registry

        definition = get_panel(PANEL)
        assert definition is not None

        def explode(*_: Any, **__: Any) -> Any:
            raise RuntimeError("builder is on fire")

        monkeypatch.setattr(registry, "PANELS", (type(definition)(**{**_fields(definition), "build": explode}),))
        result = prepare_panel(PANEL, keyness_frame())
        assert not result.ok
        assert "on fire" in find(result, "PANEL_BUILD_FAILED").message


def _fields(definition: Any) -> dict[str, Any]:
    return {slot: getattr(definition, slot) for slot in definition.__slots__}


# ---------------------------------------------------------------------------
# Evidence -- the part that makes a figure a finding
# ---------------------------------------------------------------------------


class TestEvidenceResolves:
    def test_every_mark_carries_evidence(self) -> None:
        panel = prepared()
        assert panel.marks
        for mark in panel.marks:
            assert mark.evidence.count >= 0
            assert mark.evidence.describe.strip()

    def test_a_marks_filters_select_exactly_the_row_it_came_from(self) -> None:
        """The test this protocol exists for: filters taken back to the source
        table must find the one row the mark was built from. A plausible
        filter that selects nothing looks identical until someone clicks."""
        frame = keyness_frame()
        panel = prepared()
        for mark in panel.marks:
            selected = frame
            for column, value in mark.evidence.filters:
                assert column in frame.columns, f"{mark.key}: filter names a column the table lacks"
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == 1, f"{mark.key}: filters selected {len(selected)} rows, expected 1"
            row = selected.iloc[0]
            assert float(row[LOG_RATIO]) == pytest.approx(mark.x)
            assert float(row[G2]) == pytest.approx(mark.y)

    def test_evidence_counts_the_tokens_behind_the_point(self) -> None:
        panel = prepared()
        freedom = panel.evidence_for("freedom")
        assert freedom is not None
        assert freedom.count == 132, "120 in group A plus 12 in group B"
        assert freedom.scope == "terms", "a word is something you can look up in a concordance"

    def test_evidence_refuses_to_be_built_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Evidence(scope="rows", filters=(), count=1, describe="x")
        with pytest.raises(ValueError, match="describe"):
            Evidence(scope="rows", filters=(("Word", "a"),), count=1, describe="  ")
        with pytest.raises(ValueError, match="scope"):
            Evidence(scope="vibes", filters=(("Word", "a"),), count=1, describe="x")  # type: ignore[arg-type]

    def test_as_query_is_what_a_caller_would_filter_with(self) -> None:
        panel = prepared()
        evidence = panel.evidence_for("war")
        assert evidence is not None
        assert evidence.as_query() == {WORD: "war"}


# ---------------------------------------------------------------------------
# What the volcano actually claims
# ---------------------------------------------------------------------------


class TestVolcanoMeaning:
    def test_the_axes_are_effect_against_evidence(self) -> None:
        """The whole point: a G2 bar chart cannot separate 'the' from
        'freedom', and this must."""
        panel = prepared()
        marks = {mark.key: mark for mark in panel.marks}
        assert marks["freedom"].x > 2.0 and marks["freedom"].y > 50, "large effect, strong evidence"
        assert marks["the"].x < 0.1 and marks["the"].y > 20, "no effect, strong evidence"
        assert marks["tariff"].x < -3.0 and marks["tariff"].y < 5, "large effect, weak evidence"

    def test_groups_have_a_declared_draw_order(self) -> None:
        assert prepared().groups == ("Group A", "Group B")

    def test_the_threshold_line_explains_itself(self) -> None:
        """A line at 3.84 means nothing; 'p < 0.05 at 1 degree of freedom'
        means something."""
        panel = prepared({"significance": "0.01"})
        lines = {a.kind: a for a in panel.annotations}
        assert lines["hline"].value == 6.63
        assert "0.01" in lines["hline"].label
        assert "degree of freedom" in lines["hline"].note
        assert "conventional cut" in lines["hline"].note, "a threshold is not a decision"
        assert lines["vline"].value == 0.0
        assert "doubling" in lines["vline"].note

    def test_labels_go_to_the_strongest_evidence_by_default(self) -> None:
        panel = prepared({"label-top": 2})
        labelled = [m.key for m in panel.marks if m.labelled]
        assert labelled[0] == "freedom"
        assert "the" not in labelled, "a function word stays a point; its label goes to a content word"
        assert "the" in {m.key for m in panel.marks}

    def test_labelling_by_effect_picks_different_words(self) -> None:
        """The two orders answer different questions, so the panel offers the
        choice rather than inventing a blend of them."""
        panel = prepared({"label-top": 2, "label-by": "effect"})
        assert {m.key for m in panel.marks if m.labelled} == {"tariff", "freedom"}

    def test_labelling_none_is_allowed(self) -> None:
        assert not any(m.labelled for m in prepared({"label-top": 0}).marks)

    def test_size_carries_how_common_the_word_is(self) -> None:
        marks = {mark.key: mark for mark in prepared().marks}
        assert marks["the"].size == 17700
        assert marks["tariff"].size == 20

    def test_the_notes_warn_about_the_reading_g2_invites(self) -> None:
        notes = " ".join(prepared().notes).lower()
        assert "not how big" in notes, "G2 is evidence, not effect size"
        assert "corpus" in notes or "frequent word" in notes
        assert "concordance" in notes, "a score is a reason to go and read"


class TestVolcanoFiltering:
    def test_unplaceable_numbers_are_dropped_with_a_count_not_silently(self) -> None:
        rows = [*_ROWS, {"word": "broken", "a": 5, "b": 5, "g2": float("nan"), "lr": 1.0, "in": "Group A"}]
        result = prepare_panel(PANEL, keyness_frame(rows), source=Source(path="k.csv"))
        assert result.ok
        assert "PANEL_BAD_NUMERIC" in codes(result)
        panel = result.unwrap()
        assert "broken" not in {mark.key for mark in panel.marks}

    def test_minimum_frequency_reports_what_it_removed(self) -> None:
        result = prepare_panel(PANEL, keyness_frame(), {"min-frequency": 100}, source=Source(path="k.csv"))
        assert result.ok
        assert "PANEL_FREQUENCY_FILTERED" in codes(result)
        panel = result.unwrap()
        assert "tariff" not in {mark.key for mark in panel.marks}, "20 tokens is below 100"
        assert "freedom" in {mark.key for mark in panel.marks}

    def test_filtering_everything_away_fails_with_advice(self) -> None:
        result = prepare_panel(PANEL, keyness_frame(), {"min-frequency": 1_000_000})
        assert not result.ok
        assert "min-frequency" in find(result, "PANEL_NO_DATA").message

    def test_a_pre_truncated_table_says_so_rather_than_implying_significance(self) -> None:
        """Found on the real corpus: `keyness --top-n 300` leaves every word
        above the line, and a plot reporting "294 of 294 significant" reads as
        a finding about the corpus when it is a fact about the input."""
        rows = [
            {"word": f"w{index}", "a": 50 + index, "b": 5, "g2": 40.0 + index, "lr": 2.0, "in": "Group A"}
            for index in range(25)
        ]
        result = prepare_panel(PANEL, keyness_frame(rows), source=Source(path="k.csv"))
        assert result.ok
        notice = find(result, "PANEL_ALL_ABOVE_THRESHOLD")
        assert "--top-n" in notice.message, "the notice must say how to see the whole shape"
        assert "already cut to the strongest" in result.unwrap().subtitle

    def test_a_small_table_where_all_pass_is_not_accused_of_truncation(self) -> None:
        """Four words all clearing the line is not evidence of anything."""
        rows = [{"word": w, "a": 50, "b": 5, "g2": 40.0, "lr": 2.0, "in": "Group A"} for w in ("a", "b", "c")]
        result = prepare_panel(PANEL, keyness_frame(rows), source=Source(path="k.csv"))
        assert "PANEL_ALL_ABOVE_THRESHOLD" not in codes(result)

    def test_the_table_beside_the_figure_is_the_rows_that_were_drawn(self) -> None:
        panel = prepared({"min-frequency": 100})
        assert set(panel.data[WORD]) == {mark.key for mark in panel.marks}
        assert "_total" not in panel.data.columns, "working columns do not leak into the published table"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestReadingInContext:
    """A terms mark says what to search the text for, and how the run counted.

    The filters say which row a mark came from. They cannot say what to look
    up in the documents, so the builder does -- and only the builder knows the
    run was counted over lemmas, in which case "be" must also find "is"."""

    def test_a_keyness_mark_looks_up_its_own_word(self) -> None:
        panel = prepared()
        for mark in panel.marks:
            assert mark.evidence.phrase == mark.key

    def test_lemma_matching_follows_the_field_the_run_counted(self) -> None:
        on_lemmas = prepared(source=Source(path="k.csv", settings={"field": "lemma"}))
        on_forms = prepared(source=Source(path="k.csv", settings={"field": "form"}))
        assert all(mark.evidence.lemma for mark in on_lemmas.marks)
        assert not any(mark.evidence.lemma for mark in on_forms.marks)

    def test_with_no_recorded_field_the_tools_default_is_assumed(self) -> None:
        """A CLI panel has no envelope; keyness counts lemmas by default."""
        assert all(mark.evidence.lemma for mark in prepared().marks)

    def test_the_lookup_crosses_to_the_app(self) -> None:
        evidence = prepared().marks[0].evidence.to_dict()
        assert evidence["phrase"] == prepared().marks[0].key
        assert evidence["lemma"] is True

    def test_only_terms_evidence_may_carry_a_phrase(self) -> None:
        """Documents and sentences are found by filter, not by search; a
        phrase there would be a second, disagreeing account of the mark."""
        with pytest.raises(ValueError, match="only terms evidence"):
            Evidence(scope="documents", filters=(("Period", "1930"),), count=1, describe="d", phrase="war")

    def test_lemma_matching_needs_a_phrase_to_describe(self) -> None:
        with pytest.raises(ValueError, match="lemma matching"):
            Evidence(scope="terms", filters=(("Word", "a"),), count=1, describe="a", lemma=True)


class TestProvenance:
    def test_a_panel_without_a_source_says_so(self) -> None:
        result = prepare_panel(PANEL, keyness_frame())
        assert result.ok, "a live preview has no artifact yet; that is not a failure"
        assert "PANEL_PROVENANCE_INCOMPLETE" in codes(result)
        assert result.diagnostics[0].severity is Severity.WARNING
        assert "source not recorded" in result.unwrap().caption

    def test_a_panel_with_a_source_has_a_clean_caption(self) -> None:
        result = prepare_panel(
            PANEL,
            keyness_frame(),
            source=Source(
                path="runs/keyness/keyness.csv",
                sha256="0123456789abcdef" * 4,
                settings={"field": "Lemma", "smoothing": 0.5},
                filters="1930-1950",
                library="plotly 6.8.0",
            ),
        )
        assert "PANEL_PROVENANCE_INCOMPLETE" not in codes(result)
        caption = result.unwrap().caption
        for expected in (
            "keyness_volcano",
            "runs/keyness/keyness.csv",
            "0123456789ab",
            "field=Lemma",
            "smoothing=0.5",
            "1930-1950",
            "plotly 6.8.0",
        ):
            assert expected in caption, f"caption omits {expected}: {caption}"

    def test_the_caption_is_stable_for_one_run(self) -> None:
        """Settings render in key order, so two captions of the same run do
        not differ by dict ordering."""
        provenance = Provenance(
            tool="keyness",
            panel=PANEL,
            source="k.csv",
            settings={"b": 2, "a": 1},
            generated="2026-01-01T00:00:00+00:00",
        )
        assert "a=1 b=2" in provenance.caption()
        assert provenance.caption() == provenance.caption()


# ---------------------------------------------------------------------------
# The prepared shape's own invariants
# ---------------------------------------------------------------------------


class TestPreparedPanelInvariants:
    def _panel(self, **overrides: Any) -> PreparedPanel:
        evidence = Evidence(scope="rows", filters=(("Word", "a"),), count=1, describe="a")
        base: dict[str, Any] = {
            "panel": "p",
            "shape": "scatter_labelled",
            "title": "t",
            "marks": (PanelMark(key="a", label="a", x=0.0, y=0.0, evidence=evidence),),
            "x_label": "x",
            "y_label": "y",
            "provenance": Provenance(tool="t", panel="p"),
            "data": pd.DataFrame({"Word": ["a"]}),
        }
        return PreparedPanel(**(base | overrides))

    def test_mark_keys_must_be_unique(self) -> None:
        evidence = Evidence(scope="rows", filters=(("Word", "a"),), count=1, describe="a")
        duplicate = (
            PanelMark(key="a", label="a", x=0.0, y=0.0, evidence=evidence),
            PanelMark(key="a", label="b", x=1.0, y=1.0, evidence=evidence),
        )
        with pytest.raises(ValueError, match="duplicate mark key"):
            self._panel(marks=duplicate)

    def test_marks_may_not_use_an_undeclared_group(self) -> None:
        """Otherwise a group silently loses its colour and its legend entry."""
        evidence = Evidence(scope="rows", filters=(("Word", "a"),), count=1, describe="a")
        with pytest.raises(ValueError, match="draw order"):
            self._panel(marks=(PanelMark(key="a", label="a", x=0.0, y=0.0, group="ghost", evidence=evidence),))

    def test_an_unknown_shape_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            self._panel(shape="interpretive_dance")

    def test_a_panel_cannot_be_smaller_than_it_can_be_read_at(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            self._panel(width=50)

    def test_a_parameter_must_explain_itself(self) -> None:
        with pytest.raises(ValueError, match="help"):
            PanelParam(name="n", type="int", default=1, help="")

    def test_a_choice_parameter_must_offer_choices(self) -> None:
        with pytest.raises(ValueError, match="choices"):
            PanelParam(name="n", type="choice", default="a", help="h")
        with pytest.raises(ValueError, match="only a choice"):
            PanelParam(name="n", type="int", default=1, help="h", choices=("a",))

    def test_an_annotation_kind_is_closed(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            Annotation(kind="squiggle", value=1.0, label="l")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Rendering and export
# ---------------------------------------------------------------------------


class TestRendering:
    def test_one_trace_per_group_in_the_declared_order(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(prepared()).unwrap()
        assert [trace.name for trace in figure.data] == ["Group A", "Group B"]

    def test_only_the_chosen_marks_are_labelled_on_the_page(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(prepared({"label-top": 1})).unwrap()
        written = [text for trace in figure.data for text in trace.text if text]
        assert written == ["freedom"]

    def test_hover_text_is_the_marks_own_evidence(self) -> None:
        """What the figure says about a point and what a click would retrieve
        are one sentence, written once."""
        pytest.importorskip("plotly")
        panel = prepared()
        figure = build_panel_figure(panel).unwrap()
        hovers = {text for trace in figure.data for text in trace.hovertext}
        assert hovers == {mark.evidence.describe for mark in panel.marks}

    def test_reference_lines_are_drawn(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(prepared()).unwrap()
        assert len(figure.layout.shapes) == 2, "the no-difference line and the significance line"

    def test_a_panel_with_no_marks_is_not_drawn_blank(self) -> None:
        pytest.importorskip("plotly")
        evidence = Evidence(scope="rows", filters=(("Word", "a"),), count=1, describe="a")
        del evidence
        panel = PreparedPanel(
            panel="p",
            shape="scatter_labelled",
            title="t",
            marks=(),
            x_label="x",
            y_label="y",
            provenance=Provenance(tool="t", panel="p"),
            data=pd.DataFrame(),
        )
        result = build_panel_figure(panel)
        assert not result.ok
        assert codes(result) == ["PANEL_NO_MARKS"]

    def test_html_carries_the_caption_and_the_limits(self) -> None:
        pytest.importorskip("plotly")
        result = panel_html(prepared())
        assert result.ok
        html = result.unwrap()
        assert "keyness.csv" in html, "the caption"
        assert "What this can and cannot show" in html, "the notes"
        assert "<figure" in html and "<figcaption" in html


class TestCaptionSurvivesExport:
    def test_the_caption_is_inside_the_figure_not_around_it(self) -> None:
        """A caption in surrounding HTML is lost the moment anyone exports a
        PNG -- which is exactly when the reader needs it."""
        pytest.importorskip("plotly")
        panel = prepared(source=Source(path="runs/k/keyness.csv"))
        figure = build_panel_figure(panel).unwrap()
        captions = [a.text for a in figure.layout.annotations if a.text and "keyness.csv" in a.text]
        assert captions, "no provenance annotation inside the figure"

    def test_the_caption_reaches_what_kaleido_renders_from(self) -> None:
        """kaleido renders to_plotly_json(), so anything not serialized there
        is absent from every exported image."""
        pytest.importorskip("plotly")
        panel = prepared(source=Source(path="runs/k/keyness.csv"))
        payload = build_panel_figure(panel).unwrap().to_plotly_json()
        texts = [a.get("text", "") for a in payload["layout"].get("annotations", ())]
        assert any("keyness.csv" in text for text in texts)


class TestImageExport:
    def test_an_unsupported_format_is_refused_before_any_work(self) -> None:
        result = panel_image_bytes(prepared(), "gif")
        assert codes(result) == ["PANEL_BAD_FORMAT"]

    @pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
    def test_every_declared_format_reaches_kaleido_with_the_panels_size(
        self, monkeypatch: pytest.MonkeyPatch, fmt: str
    ) -> None:
        """Verifies the export path itself rather than the optional
        dependency: kaleido is not installed everywhere, but the call it
        would receive is still ours to get right."""
        pytest.importorskip("plotly")
        from core.viz import panel_plotters

        seen: dict[str, Any] = {}

        def fake(fig: Any, panel: PreparedPanel, out_fmt: str, browser: str | None) -> bytes:
            seen.update(format=out_fmt, width=panel.width, height=panel.height, browser=browser)
            return b"\x89PNG-pretend"

        monkeypatch.setattr(panel_plotters, "_kaleido_bytes", fake)
        panel = prepared()
        result = panel_plotters.panel_image_bytes(panel, fmt, browser_path="/usr/bin/chromium")
        assert result.ok, [str(d) for d in result.diagnostics]
        assert result.unwrap() == b"\x89PNG-pretend"
        assert seen == {
            "format": fmt,
            "width": panel.width,
            "height": panel.height,
            "browser": "/usr/bin/chromium",
        }

    def test_a_missing_kaleido_fails_loudly_and_never_substitutes_html(self) -> None:
        try:
            import kaleido  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("kaleido installed; the missing-dependency path is not reachable here")
        pytest.importorskip("plotly")
        result = panel_image_bytes(prepared(), "png")
        assert not result.ok
        assert codes(result) == ["PANEL_IMAGE_KALEIDO_MISSING"]
        assert result.diagnostics[0].context.get("fix"), "a missing extra must say how to install it"

    def test_a_browser_failure_is_told_apart_from_any_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("plotly")
        from core.viz import panel_plotters

        class ChromeNotFoundError(Exception):
            pass

        def missing_browser(*_: Any, **__: Any) -> bytes:
            raise ChromeNotFoundError("no chrome at /nope")

        monkeypatch.setattr(panel_plotters, "_kaleido_bytes", missing_browser)
        result = panel_plotters.panel_image_bytes(prepared(), "png")
        assert codes(result) == ["PANEL_IMAGE_BROWSER_NOT_FOUND"]
        assert "BROWSER_PATH" in result.diagnostics[0].message


class TestPlotlyMissingDegradesHonestly:
    def test_the_fallback_is_the_same_marks_in_the_same_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.viz import panel_plotters

        def no_plotly(_: PreparedPanel) -> Any:
            raise ImportError("no plotly")

        monkeypatch.setattr(panel_plotters, "build_panel_figure", no_plotly)
        panel = prepared()
        result = panel_plotters.panel_html(panel)
        assert result.ok, "a missing optional renderer degrades; it does not fail the run"
        assert codes(result) == ["PANEL_PLOTLY_UNAVAILABLE"]
        html = result.unwrap()
        assert html.index("freedom") < html.index("tariff"), "table order follows mark order"
        for mark in panel.marks:
            assert mark.label in html
        assert "keyness.csv" in html, "the caption survives the degradation"


def _mark(key: str, label: str, x: float, y: float, group: str) -> PanelMark:
    evidence = Evidence(scope="rows", filters=(("k", key),), count=1, describe=f"{label} at {x}")
    return PanelMark(key=key, label=label, x=x, y=y, group=group, evidence=evidence)


def _shaped(shape: str, marks: tuple[PanelMark, ...], groups: tuple[str, ...]) -> PreparedPanel:
    return PreparedPanel(
        panel="p",
        shape=shape,  # type: ignore[arg-type]
        title="t",
        marks=marks,
        x_label="x",
        y_label="y",
        provenance=Provenance(tool="t", panel="p", source="s.csv"),
        data=pd.DataFrame({"k": [m.key for m in marks]}),
        groups=groups,
    )


class TestRankedBars:
    """Horizontal bars whose order is the panel's decision, not plotly's."""

    def _panel(self) -> PreparedPanel:
        marks: list[PanelMark] = []
        for rank, word in enumerate(("freedom", "war", "shall")):
            marks.append(_mark(f"{word}:rel", word, 0.9 - rank * 0.2, float(rank), "Relevance"))
            marks.append(_mark(f"{word}:sal", word, 0.5 - rank * 0.1, float(rank), "Saliency"))
        return _shaped("ranked_bars", tuple(marks), ("Relevance", "Saliency"))

    def test_rank_zero_is_drawn_at_the_top(self) -> None:
        """Plotly's category axis counts upward, so the highest-ranked term
        lands at the bottom unless the array is built in reverse."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert list(figure.layout.yaxis.categoryarray) == ["shall", "war", "freedom"]

    def test_the_two_series_share_a_baseline(self) -> None:
        """Side-by-side half-width bars turn "how far apart are these two
        readings?" into a comparison across a gap."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert figure.layout.barmode == "overlay"
        assert [trace.name for trace in figure.data] == ["Relevance", "Saliency"]
        assert figure.data[1].opacity < figure.data[0].opacity, "the second series must not hide the first"

    def test_bars_are_horizontal_and_carry_their_evidence(self) -> None:
        pytest.importorskip("plotly")
        panel = self._panel()
        figure = build_panel_figure(panel).unwrap()
        assert all(trace.orientation == "h" for trace in figure.data)
        hovers = {text for trace in figure.data for text in trace.hovertext}
        assert hovers == {mark.evidence.describe for mark in panel.marks}


class TestStream:
    """Stacked bands over an ordered axis."""

    def _panel(self) -> PreparedPanel:
        marks = (
            _mark("1930:0", "T0", 1930.0, 0.6, "Topic 0"),
            _mark("1940:0", "T0", 1940.0, 0.4, "Topic 0"),
            # Topic 1 is absent in 1940 on purpose.
            _mark("1930:1", "T1", 1930.0, 0.4, "Topic 1"),
        )
        return _shaped("stream", marks, ("Topic 0", "Topic 1"))

    def test_a_missing_bucket_is_zero_not_a_line_across_the_gap(self) -> None:
        """Drawing straight from 1930 to 1950 through a missing 1940 invents
        data, and on a stack it also lifts every band above it."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        topic_one = next(trace for trace in figure.data if trace.name == "Topic 1")
        assert list(topic_one.x) == [1930.0, 1940.0]
        assert list(topic_one.y) == [0.4, 0.0]

    def test_bands_are_stacked_and_share_one_axis(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert all(trace.stackgroup == "one" for trace in figure.data)
        assert all(list(trace.x) == [1930.0, 1940.0] for trace in figure.data), "every band spans the whole axis"

    def test_the_axis_stays_numeric(self) -> None:
        """A decade axis sorted as text puts 1940 between 1930 and 2020, and
        a reader has no way to see that it happened."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert all(isinstance(value, float) for value in figure.data[0].x)

    def test_declared_group_order_fixes_the_band_colours(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert [trace.name for trace in figure.data] == ["Topic 0", "Topic 1"]


class TestRibbon:
    """Coloured segments along one axis: a document per band, a paragraph
    per segment, in the order the paragraphs occur."""

    def _panel(self) -> PreparedPanel:
        marks = (
            _mark("1934:1", "1934", 0.0, 0.0, "Topic 0"),
            _mark("1934:2", "1934", 0.5, 0.0, "Topic 1"),
            _mark("1942:1", "1942", 0.0, 1.0, "Topic 1"),
        )
        for mark_ in marks:
            object.__setattr__(mark_, "size", 0.5)
        return _shaped("ribbon", marks, ("Topic 0", "Topic 1"))

    def test_bands_are_horizontal_segments_with_explicit_starts(self) -> None:
        """base = segment start, x = width: the left edge is where the
        paragraph starts, not a bar's length from zero. One trace per
        group, so the two traces cover the marks between them."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        by_name = {trace.name: trace for trace in figure.data}
        assert all(trace.orientation == "h" for trace in figure.data)
        assert list(by_name["Topic 0"].base) == [0.0]
        assert list(by_name["Topic 0"].x) == [0.5]
        assert list(by_name["Topic 1"].base) == [0.5, 0.0]
        assert list(by_name["Topic 1"].x) == [0.5, 0.5]

    def test_row_zero_is_drawn_at_the_top(self) -> None:
        """Plotly's category axis counts upward, so the row array is built in
        reverse — the SVG renderer draws row order directly, no reversal."""
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert list(figure.layout.yaxis.categoryarray) == ["1942", "1934"]

    def test_declared_group_order_fixes_the_segment_colours(self) -> None:
        pytest.importorskip("plotly")
        figure = build_panel_figure(self._panel()).unwrap()
        assert [trace.name for trace in figure.data] == ["Topic 0", "Topic 1"]

    def test_hover_text_is_the_segments_own_evidence(self) -> None:
        pytest.importorskip("plotly")
        panel = self._panel()
        figure = build_panel_figure(panel).unwrap()
        hovers = {text for trace in figure.data for text in trace.hovertext}
        assert hovers == {mark.evidence.describe for mark in panel.marks}


class TestShapeVocabulary:
    def test_the_renderer_only_claims_shapes_it_draws(self) -> None:
        assert set(IMPLEMENTED_SHAPES) <= set(PANEL_SHAPES)

    def test_a_shape_nothing_draws_is_refused_rather_than_half_served(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Withdraw a shape from the renderer and ask for it: the refusal is
        named, not a KeyError. (Independent of which shapes happen to be
        unimplemented, so it still means something once all of them are.)"""
        import core.viz.panel_plotters as plotters

        monkeypatch.setattr(
            plotters, "IMPLEMENTED_SHAPES", tuple(s for s in IMPLEMENTED_SHAPES if s != "scatter_labelled")
        )
        panel = PreparedPanel(
            panel="p",
            shape="scatter_labelled",
            title="t",
            marks=(
                PanelMark(
                    key="a",
                    label="a",
                    x=0.0,
                    y=0.0,
                    evidence=Evidence(scope="rows", filters=(("Word", "a"),), count=1, describe="a"),
                ),
            ),
            x_label="x",
            y_label="y",
            provenance=Provenance(tool="t", panel="p"),
            data=pd.DataFrame(),
        )
        result = plotters.build_panel_figure(panel)
        assert codes(result) == ["PANEL_SHAPE_UNIMPLEMENTED"]
