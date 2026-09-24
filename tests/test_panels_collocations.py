"""Tests for ``core/viz/panels_collocations.py``.

This panel is not registered in ``core/viz/panels.py`` (that file is owned
centrally and out of scope here), so every test calls
``collocation_strength`` directly -- constructing ``Provenance`` by hand and
building params as ``COLLOCATION_STRENGTH.defaults() | overrides`` -- rather
than going through ``prepare_panel``. That means the central gate
(``PANEL_UNKNOWN_PARAM``, ``PANEL_BAD_PARAM``, ``PANEL_MISSING_COLUMN``,
provenance completeness, ...) is not exercised here; it lives in
``tests/test_panels.py`` and applies uniformly to every registered panel
once this one is added to ``PANELS``.

Two things matter most, in the same spirit as ``tests/test_panels.py``:

**Evidence must resolve.** ``TestEvidenceResolves`` takes each mark's
``(Word 1, value), (Word 2, value)`` filters back to the source frame and
asserts they select exactly the one row the mark was built from.

**The panel must be informative on real data, not just correct.** PMI's
rare-pair bias is easy to get numerically right and still mislead a reader;
``PANEL_RARE_PAIRS_DOMINATE`` is tested directly, including that it does
*not* fire for the other three measures or when no pairs are labelled.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from core.result import Result, Severity
from core.viz.panels_collocations import (
    COLLOCATION_STRENGTH,
    COOCCURRENCES,
    G2,
    LOG_DICE,
    PAIR,
    PMI,
    T_SCORE,
    WORD_1,
    WORD_1_FREQ,
    WORD_2,
    WORD_2_FREQ,
    collocation_strength,
)
from core.viz.panelspec import PreparedPanel, Provenance

# common_strong:  frequent, and strongly associated by every measure -- the
#                 pair worth reading in context.
# rare_pmi_spike: two rare words that always co-occur -- PMI's trap: a huge
#                 PMI score built from almost nothing.
# common_weak:    very frequent (a function-word-like pair), unremarkable
#                 once independence is accounted for -- weak on every
#                 measure except the ones sheer volume inflates (t-score,
#                 g2).
# moderate:       middling on everything, the boring/typical case.
_ROWS: list[dict[str, Any]] = [
    {
        "w1": "union",
        "w2": "strong",
        "cooc": 500,
        "f1": 600,
        "f2": 550,
        "pmi": 2.0,
        "t": 20.0,
        "g2": 300.0,
        "logdice": 10.5,
    },
    {
        "w1": "ephemeral",
        "w2": "zenith",
        "cooc": 2,
        "f1": 2,
        "f2": 2,
        "pmi": 11.5,
        "t": 1.2,
        "g2": 8.5,
        "logdice": 5.0,
    },
    {
        "w1": "of",
        "w2": "the",
        "cooc": 4000,
        "f1": 9000,
        "f2": 8000,
        "pmi": -0.3,
        "t": 35.0,
        "g2": 150.0,
        "logdice": 2.8,
    },
    {
        "w1": "free",
        "w2": "press",
        "cooc": 15,
        "f1": 20,
        "f2": 18,
        "pmi": 4.2,
        "t": 3.5,
        "g2": 28.0,
        "logdice": 7.8,
    },
]


def collocation_frame(rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                WORD_1: row["w1"],
                WORD_2: row["w2"],
                PAIR: f"{row['w1']} {row['w2']}",
                COOCCURRENCES: row["cooc"],
                WORD_1_FREQ: row["f1"],
                WORD_2_FREQ: row["f2"],
                PMI: row["pmi"],
                T_SCORE: row["t"],
                G2: row["g2"],
                LOG_DICE: row["logdice"],
            }
            for row in (rows if rows is not None else _ROWS)
        ]
    )


def build(
    params: dict[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
    source: str = "collocations.csv",
) -> Result[PreparedPanel]:
    values = COLLOCATION_STRENGTH.defaults() | (params or {})
    provenance = Provenance(tool="collocations", panel=COLLOCATION_STRENGTH.name, source=source)
    return collocation_strength(frame if frame is not None else collocation_frame(), values, provenance)


def prepared(params: dict[str, Any] | None = None, frame: pd.DataFrame | None = None) -> PreparedPanel:
    result = build(params, frame)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def codes(result: Result[Any]) -> list[str]:
    return [d.code for d in result.diagnostics]


def with_settings(settings: dict[str, Any]) -> PreparedPanel:
    """The panel as the desktop draws it: provenance carries the run's own
    envelope settings, which is where span and field come from."""
    provenance = Provenance(
        tool="collocations", panel=COLLOCATION_STRENGTH.name, source="collocations.csv", settings=settings
    )
    result = collocation_strength(collocation_frame(), COLLOCATION_STRENGTH.defaults(), provenance)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


class TestReadingInContext:
    def test_an_adjacent_pair_is_looked_up_as_a_phrase(self) -> None:
        panel = with_settings({"span": "adjacent", "field": "lemma"})
        for mark in panel.marks:
            first, second = (value for _, value in mark.evidence.filters)
            assert mark.evidence.phrase == f"{first} {second}"
            assert mark.evidence.lemma

    def test_a_windowed_pair_gets_no_lookup_rather_than_an_undercount(self) -> None:
        """Counted anywhere in a window, the pair is not a phrase: searching
        "w1 w2" would find only its adjacent cases and present that as the
        evidence."""
        panel = with_settings({"span": "window", "field": "lemma"})
        assert all(mark.evidence.phrase == "" for mark in panel.marks)
        assert not any(mark.evidence.lemma for mark in panel.marks)

    def test_a_form_count_is_looked_up_literally(self) -> None:
        panel = with_settings({"span": "adjacent", "field": "form"})
        assert not any(mark.evidence.lemma for mark in panel.marks)
        assert all(mark.evidence.phrase for mark in panel.marks)


def find(result: Result[Any], code: str) -> Any:
    matching = [d for d in result.diagnostics if d.code == code]
    assert len(matching) == 1, f"expected exactly one {code}, got {codes(result)}"
    return matching[0]


# ---------------------------------------------------------------------------
# The definition itself
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_the_panel_declares_itself(self) -> None:
        assert COLLOCATION_STRENGTH.name == "collocation_strength"
        assert COLLOCATION_STRENGTH.tool == "collocations"
        assert COLLOCATION_STRENGTH.shape == "scatter_labelled"
        assert COLLOCATION_STRENGTH.notes, "a panel must state what it cannot show"
        assert COLLOCATION_STRENGTH.requires
        for param in COLLOCATION_STRENGTH.params:
            assert param.help.strip(), f"{param.name} has no help"

    def test_defaults_match_the_spec(self) -> None:
        defaults = COLLOCATION_STRENGTH.defaults()
        assert defaults == {"measure": "log-dice", "label-top": 15, "min-count": 2}

    def test_measure_default_is_log_dice_because_it_is_frequency_robust(self) -> None:
        """``provenance.params`` is populated by the registry (``prepare_panel``),
        not by the builder under test here, so this checks the declaration
        itself plus the y axis it actually produces when no override is given."""
        assert COLLOCATION_STRENGTH.defaults()["measure"] == "log-dice"
        panel = prepared()
        assert "Log Dice" in panel.y_label


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
            assert mark.evidence.scope == "terms", "a word pair is something you can look up in a concordance"

    def test_a_marks_filters_select_exactly_the_row_it_came_from(self) -> None:
        """The test this protocol exists for: filters taken back to the source
        table must find the one row the mark was built from."""
        frame = collocation_frame()
        panel = prepared()
        for mark in panel.marks:
            selected = frame
            for column, value in mark.evidence.filters:
                assert column in frame.columns, f"{mark.key}: filter names a column the table lacks"
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == 1, f"{mark.key}: filters selected {len(selected)} rows, expected 1"
            row = selected.iloc[0]
            assert float(row[LOG_DICE]) == pytest.approx(mark.y)
            assert math.log10(float(row[COOCCURRENCES])) == pytest.approx(mark.x)
            assert mark.evidence.count == int(row[COOCCURRENCES])

    def test_filters_are_on_word_1_and_word_2_not_the_composite_pair_column(self) -> None:
        """``Pair`` is unique per row (it is built from the (Word 1, Word 2)
        key in the engine), but evidence must filter on the two word columns
        the task specifies, so a caller resolving evidence never has to
        parse a composite string back apart."""
        panel = prepared()
        for mark in panel.marks:
            columns = {column for column, _ in mark.evidence.filters}
            assert columns == {WORD_1, WORD_2}

    def test_evidence_describes_the_pair_counts_and_measure(self) -> None:
        panel = prepared({"measure": "pmi"})
        mark = next(m for m in panel.marks if m.key == "ephemeral zenith")
        assert mark.evidence.count == 2
        assert "ephemeral zenith" in mark.evidence.describe
        assert "2 co-occurrences" in mark.evidence.describe
        assert "PMI" in mark.evidence.describe

    def test_as_query_is_what_a_caller_would_filter_with(self) -> None:
        panel = prepared()
        mark = next(m for m in panel.marks if m.key == "union strong")
        assert mark.evidence.as_query() == {WORD_1: "union", WORD_2: "strong"}


# ---------------------------------------------------------------------------
# What the panel actually claims
# ---------------------------------------------------------------------------


class TestPanelMeaning:
    def test_x_is_log10_of_cooccurrences(self) -> None:
        panel = prepared()
        marks = {m.key: m for m in panel.marks}
        assert marks["union strong"].x == pytest.approx(math.log10(500))
        assert marks["ephemeral zenith"].x == pytest.approx(math.log10(2))
        # The axis says so, and renderers label ticks as counts (10, 100).
        assert panel.x_log10
        assert "log scale" in panel.x_label

    def test_y_is_the_chosen_measure(self) -> None:
        panel = prepared({"measure": "t-score"})
        marks = {m.key: m for m in panel.marks}
        assert marks["union strong"].y == pytest.approx(20.0)
        assert "T-score" in panel.y_label

    def test_size_carries_raw_cooccurrence_count_not_log(self) -> None:
        marks = {m.key: m for m in prepared().marks}
        assert marks["union strong"].size == 500.0
        assert marks["ephemeral zenith"].size == 2.0

    def test_no_groups_are_declared(self) -> None:
        panel = prepared()
        assert panel.groups == ()
        assert all(mark.group == "" for mark in panel.marks)

    def test_pmi_and_log_dice_disagree_about_which_pairs_matter(self) -> None:
        """The panel's whole argument: the two measures rank pairs
        differently enough that they label different points."""
        by_pmi = prepared({"measure": "pmi", "label-top": 1})
        by_logdice = prepared({"measure": "log-dice", "label-top": 1})
        pmi_pick = {m.key for m in by_pmi.marks if m.labelled}
        logdice_pick = {m.key for m in by_logdice.marks if m.labelled}
        assert pmi_pick == {"ephemeral zenith"}, "PMI is highest for the rare, self-consistent pair"
        assert logdice_pick == {"union strong"}, "log-dice is highest for the frequent, well-attested pair"
        assert pmi_pick != logdice_pick

    def test_labelling_ties_break_on_pair_ascending(self) -> None:
        """So a redraw labels the same point: two pairs tied on the chosen
        measure must not be ordered by frame arrival order."""
        rows = [
            {
                "w1": "zulu",
                "w2": "yankee",
                "cooc": 10,
                "f1": 12,
                "f2": 11,
                "pmi": 1.0,
                "t": 2.0,
                "g2": 5.0,
                "logdice": 6.0,
            },
            {
                "w1": "alpha",
                "w2": "bravo",
                "cooc": 10,
                "f1": 12,
                "f2": 11,
                "pmi": 1.0,
                "t": 2.0,
                "g2": 5.0,
                "logdice": 6.0,
            },
        ]
        panel = prepared({"measure": "g2", "label-top": 1}, frame=collocation_frame(rows))
        labelled = [m.key for m in panel.marks if m.labelled]
        assert labelled == ["alpha bravo"], "tied on G2; 'alpha bravo' sorts before 'zulu yankee'"

    def test_labelling_is_deterministic_across_rebuilds(self) -> None:
        first = {m.key for m in prepared({"label-top": 2}).marks if m.labelled}
        second = {m.key for m in prepared({"label-top": 2}).marks if m.labelled}
        assert first == second

    def test_labelling_none_is_allowed(self) -> None:
        assert not any(m.labelled for m in prepared({"label-top": 0}).marks)

    def test_the_notes_cover_the_required_claims(self) -> None:
        notes = " ".join(COLLOCATION_STRENGTH.notes).lower()
        assert "pmi" in notes and "rare" in notes, "PMI's bias toward rare pairs"
        assert "window" in notes or "min-count" in notes, "the collocations run's own settings change every number"
        assert "concordance" in notes, "a score is a reason to go and read"
        assert "log10" in notes, "the x axis is logarithmic"


# ---------------------------------------------------------------------------
# Annotations -- only where a reference value means something
# ---------------------------------------------------------------------------


class TestAnnotations:
    def test_pmi_reference_line_is_zero_and_explains_independence(self) -> None:
        panel = prepared({"measure": "pmi"})
        (annotation,) = panel.annotations
        assert annotation.kind == "hline"
        assert annotation.value == 0.0
        assert "independence" in annotation.note

    def test_t_score_reference_line_is_zero_and_explains_observed_equals_expected(self) -> None:
        panel = prepared({"measure": "t-score"})
        (annotation,) = panel.annotations
        assert annotation.value == 0.0
        assert "expected" in annotation.note

    def test_g2_reference_line_is_the_p_05_threshold(self) -> None:
        panel = prepared({"measure": "g2"})
        (annotation,) = panel.annotations
        assert annotation.value == pytest.approx(3.84)
        assert "0.05" in annotation.label
        assert "degree of freedom" in annotation.note

    def test_log_dice_draws_its_ceiling_not_an_independence_line(self) -> None:
        """Log Dice has no independence baseline, so no "no association" line.
        It does have a ceiling that real pairs reach -- two words that never
        occur apart -- and that is a fact about the data worth drawing."""
        panel = prepared({"measure": "log-dice"})
        (annotation,) = panel.annotations
        assert annotation.kind == "hline"
        assert annotation.value == 14.0
        assert "never occur apart" in annotation.note
        assert "not an independence baseline" in annotation.note

    def test_the_ceiling_matches_the_engines_offset(self) -> None:
        """Restated in the panel because the engine's name is private; this
        keeps the line on the value the engine actually caps at."""
        from core.analysis.collocations import _LOG_DICE_OFFSET
        from core.viz.panels_collocations import _LOG_DICE_CEILING

        assert _LOG_DICE_CEILING == _LOG_DICE_OFFSET


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_non_numeric_measure_is_dropped_with_a_count(self) -> None:
        rows = [
            *_ROWS,
            {
                "w1": "broken",
                "w2": "pair",
                "cooc": 5,
                "f1": 5,
                "f2": 5,
                "pmi": float("nan"),
                "t": 1.0,
                "g2": 1.0,
                "logdice": 1.0,
            },
        ]
        result = build({"measure": "pmi"}, frame=collocation_frame(rows))
        assert result.ok
        assert "PANEL_BAD_NUMERIC" in codes(result)
        assert find(result, "PANEL_BAD_NUMERIC").context["dropped"] == 1
        assert "broken pair" not in {m.key for m in result.unwrap().marks}

    def test_zero_cooccurrences_cannot_be_placed_on_a_log_axis(self) -> None:
        """A pair with zero co-occurrences has no log10 position; the task
        requires it be dropped with a diagnostic rather than placed at an
        invented x."""
        rows = [
            *_ROWS,
            {
                "w1": "never",
                "w2": "happens",
                "cooc": 0,
                "f1": 5,
                "f2": 5,
                "pmi": 0.0,
                "t": 0.0,
                "g2": 0.0,
                "logdice": 0.0,
            },
        ]
        result = build(frame=collocation_frame(rows))
        assert result.ok
        assert "PANEL_BAD_NUMERIC" in codes(result)
        assert "never happens" not in {m.key for m in result.unwrap().marks}

    def test_min_count_filter_reports_what_it_removed(self) -> None:
        result = build({"min-count": 10})
        assert result.ok
        assert "PANEL_FREQUENCY_FILTERED" in codes(result)
        removed_diag = find(result, "PANEL_FREQUENCY_FILTERED")
        assert removed_diag.context["removed"] == 1
        assert "ephemeral zenith" not in {m.key for m in result.unwrap().marks}, "cooc=2 is below min-count=10"

    def test_filtering_everything_away_fails_with_advice(self) -> None:
        result = build({"min-count": 1_000_000})
        assert not result.ok
        assert "min-count" in find(result, "PANEL_NO_DATA").message

    def test_too_many_pairs_are_capped_keeping_the_strongest_by_the_chosen_measure(self) -> None:
        rows = [
            {
                "w1": f"w{i}",
                "w2": f"x{i}",
                "cooc": 10 + i,
                "f1": 20,
                "f2": 20,
                "pmi": float(i),
                "t": float(i),
                "g2": float(i),
                "logdice": float(i),
            }
            for i in range(2100)
        ]
        result = build({"measure": "g2"}, frame=collocation_frame(rows))
        assert result.ok
        notice = find(result, "PANEL_TOO_MANY_MARKS")
        assert notice.context["drawn"] == 2000
        assert notice.context["available"] == 2100
        panel = result.unwrap()
        assert len(panel.marks) == 2000
        # Kept the strongest G2 (highest i), not the first 2000 by input order.
        kept = {m.key for m in panel.marks}
        assert "w2099 x2099" in kept
        assert "w0 x0" not in kept

    def test_rare_pairs_dominate_fires_only_for_pmi(self) -> None:
        rare_rows = [
            {
                "w1": f"rare{i}",
                "w2": f"pair{i}",
                "cooc": 2,
                "f1": 2,
                "f2": 2,
                "pmi": 10.0 + i,
                "t": 1.0,
                "g2": 5.0,
                "logdice": 3.0,
            }
            for i in range(4)
        ]
        common_row = {
            "w1": "common",
            "w2": "pair",
            "cooc": 500,
            "f1": 600,
            "f2": 550,
            "pmi": 1.0,
            "t": 20.0,
            "g2": 300.0,
            "logdice": 10.0,
        }
        frame = collocation_frame([*rare_rows, common_row])

        pmi_result = build({"measure": "pmi", "label-top": 4}, frame=frame)
        assert pmi_result.ok
        notice = find(pmi_result, "PANEL_RARE_PAIRS_DOMINATE")
        assert notice.severity is Severity.WARNING
        assert "log-dice" in notice.message

        for other in ("log-dice", "t-score", "g2"):
            other_result = build({"measure": other, "label-top": 4}, frame=frame)
            assert "PANEL_RARE_PAIRS_DOMINATE" not in codes(other_result), other

    def test_rare_pairs_dominate_does_not_fire_when_nothing_is_labelled(self) -> None:
        rare_rows = [
            {
                "w1": f"rare{i}",
                "w2": f"pair{i}",
                "cooc": 2,
                "f1": 2,
                "f2": 2,
                "pmi": 10.0 + i,
                "t": 1.0,
                "g2": 5.0,
                "logdice": 3.0,
            }
            for i in range(4)
        ]
        result = build({"measure": "pmi", "label-top": 0}, frame=collocation_frame(rare_rows))
        assert "PANEL_RARE_PAIRS_DOMINATE" not in codes(result)

    def test_rare_pairs_dominate_catches_pairs_just_above_a_fixed_floor(self) -> None:
        """Regression test for what the real 87-address State of the Union
        corpus showed: with the collocations engine's own default
        min_count=3, PMI's top-labelled pairs there sat at 3-6 occurrences
        each -- clearly rare -- but a threshold of "co-occurrences <= 3"
        only matched the table's exact floor, so 6 of those 15 pairs missed
        it and the diagnostic never fired. None of the pairs below are
        literally "<= 3", so a fixed-threshold version of this check would
        stay silent; the median-relative check must still catch them."""
        rows = [
            {"w1": "w1a", "w2": "w1b", "cooc": 4, "f1": 4, "f2": 4, "pmi": 20.0, "t": 1.0, "g2": 5.0, "logdice": 6.0},
            {"w1": "w2a", "w2": "w2b", "cooc": 5, "f1": 5, "f2": 5, "pmi": 19.0, "t": 1.0, "g2": 5.0, "logdice": 6.0},
            {"w1": "w3a", "w2": "w3b", "cooc": 5, "f1": 5, "f2": 5, "pmi": 18.0, "t": 1.0, "g2": 5.0, "logdice": 6.0},
            {"w1": "w4a", "w2": "w4b", "cooc": 6, "f1": 6, "f2": 6, "pmi": 17.0, "t": 1.0, "g2": 5.0, "logdice": 6.0},
            {
                "w1": "common",
                "w2": "pair",
                "cooc": 500,
                "f1": 600,
                "f2": 550,
                "pmi": 1.0,
                "t": 20.0,
                "g2": 300.0,
                "logdice": 10.0,
            },
        ]
        result = build({"measure": "pmi", "label-top": 4}, frame=collocation_frame(rows))
        assert result.ok
        notice = find(result, "PANEL_RARE_PAIRS_DOMINATE")
        assert notice.context["rare"] == 3
        assert notice.context["labelled"] == 4

    def test_rare_pairs_dominate_does_not_fire_when_the_top_pairs_are_common(self) -> None:
        """ "Rare" is relative to this table's own median co-occurrence (257.5
        here: the two middle values of [2, 15, 500, 4000]). Only 2 of the 4
        labelled pairs fall at or below that -- not more than half."""
        result = build({"measure": "pmi", "label-top": 4})
        assert "PANEL_RARE_PAIRS_DOMINATE" not in codes(result)

    def test_the_table_beside_the_figure_is_the_rows_that_were_drawn(self) -> None:
        panel = prepared({"min-count": 10})
        assert set(panel.data[PAIR]) == {m.key for m in panel.marks}


# ---------------------------------------------------------------------------
# Bounds (spec-declared; the central gate in tests/test_panels.py enforces
# these once the panel is registered, but the declarations themselves are
# this panel's own responsibility to get right)
# ---------------------------------------------------------------------------


class TestParamBounds:
    def test_label_top_bounds(self) -> None:
        param = next(p for p in COLLOCATION_STRENGTH.params if p.name == "label-top")
        assert param.default == 15
        assert param.minimum == 0
        assert param.maximum == 100

    def test_min_count_bounds(self) -> None:
        param = next(p for p in COLLOCATION_STRENGTH.params if p.name == "min-count")
        assert param.default == 2
        assert param.minimum == 1
        assert param.maximum == 1_000_000

    def test_measure_choices(self) -> None:
        param = next(p for p in COLLOCATION_STRENGTH.params if p.name == "measure")
        assert param.choices == ("pmi", "log-dice", "t-score", "g2")
        assert param.default == "log-dice"
