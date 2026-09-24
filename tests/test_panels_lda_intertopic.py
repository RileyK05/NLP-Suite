"""Tests for the Intertopic Distance Map panel (``panels_lda_intertopic.py``).

This panel is not registered in ``core/viz/panels.py`` -- that registry is
owned centrally and this module is deliberately not allowed to touch it (see
the task that produced this file). So unlike ``tests/test_panels.py``, which
drives everything through ``prepare_panel``, these tests call
``lda_intertopic`` directly: they build a ``Provenance`` themselves and pass
an already-defaulted, already-validated params dict, exactly what the
registry would have handed the builder if it were wired in. That is a
faithful test of the builder's contract -- the registry's own job (unknown
panel names, unknown params, type/bounds checking) is generic code tested
once in ``tests/test_panels.py`` and does not need re-proving here.

As in ``tests/test_panels.py``, two things matter most:

**Evidence must resolve.** ``TestEvidenceResolves`` takes each mark's
filters back to the source frame and asserts they select exactly the one
row the mark was built from -- a plausible filter that selects nothing looks
identical until someone clicks it.

**The axes must not be misread.** The MDS layout has no meaning along
either axis individually, only in the distance between points. Tests here
check that the panel says so in its annotation and in its notes, not just
that a chart of some kind comes out.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from core.result import Result, Severity
from core.viz.panels_lda_intertopic import LDA_INTERTOPIC, PREVALENCE, TOPIC, X, Y, lda_intertopic
from core.viz.panelspec import PreparedPanel, Provenance

# topic 0: prevalent, off in its own corner.
# topic 1: prevalent, close to topic 2 (shares vocabulary with it).
# topic 2: less prevalent, close to topic 1.
# topic 3: the least prevalent, far from everything.
_ROWS: list[dict[str, Any]] = [
    {"topic": 0, "x": -2.1, "y": 1.4, "prevalence": 1.0},
    {"topic": 1, "x": 0.3, "y": -0.2, "prevalence": 0.82},
    {"topic": 2, "x": 0.5, "y": -0.1, "prevalence": 0.41},
    {"topic": 3, "x": 3.7, "y": 2.9, "prevalence": 0.06},
]


def intertopic_frame(rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {TOPIC: row["topic"], X: row["x"], Y: row["y"], PREVALENCE: row["prevalence"]}
            for row in (rows if rows is not None else _ROWS)
        ]
    )


def provenance(**params: Any) -> Provenance:
    return Provenance(tool="lda_gensim", panel=LDA_INTERTOPIC.name, source="lda.csv", params=dict(params))


def build(
    frame: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
) -> Result[PreparedPanel]:
    """Call the builder directly, with a fully defaulted param dict -- what
    the registry would hand it, assembled by hand because the registry
    cannot be edited to register this panel."""
    values = {**LDA_INTERTOPIC.defaults(), **(params or {})}
    return lda_intertopic(frame if frame is not None else intertopic_frame(), values, provenance(**values))


def prepared(params: dict[str, Any] | None = None, frame: pd.DataFrame | None = None) -> PreparedPanel:
    result = build(frame, params)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def codes(result: Result[Any]) -> list[str]:
    return [d.code for d in result.diagnostics]


def find(result: Result[Any], code: str) -> Any:
    matching = [d for d in result.diagnostics if d.code == code]
    assert len(matching) == 1, f"expected one {code}, got {codes(result)}"
    return matching[0]


# ---------------------------------------------------------------------------
# The definition itself
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_names_and_shape(self) -> None:
        assert LDA_INTERTOPIC.name == "lda_intertopic"
        assert LDA_INTERTOPIC.tool == "lda_gensim"
        assert LDA_INTERTOPIC.shape == "scatter_labelled"

    def test_requires_the_columns_the_builder_reads(self) -> None:
        assert set(LDA_INTERTOPIC.requires) == {TOPIC, X, Y, PREVALENCE}

    def test_states_what_it_cannot_show(self) -> None:
        assert LDA_INTERTOPIC.notes
        assert any("meaning" in note.lower() for note in LDA_INTERTOPIC.notes), (
            "the axes-have-no-meaning caveat must be one of the notes, not just the docstring"
        )

    def test_every_param_has_help(self) -> None:
        for param in LDA_INTERTOPIC.params:
            assert param.help.strip(), f"{param.name} has no help text"

    def test_label_top_help_says_it_is_conditional(self) -> None:
        label_top = next(p for p in LDA_INTERTOPIC.params if p.name == "label-top")
        assert "largest" in label_top.help

    def test_defaults(self) -> None:
        assert LDA_INTERTOPIC.defaults() == {"label-topics": "all", "label-top": 10}


# ---------------------------------------------------------------------------
# Basic shape of a successful build
# ---------------------------------------------------------------------------


class TestBuild:
    def test_one_mark_per_topic(self) -> None:
        panel = prepared()
        assert len(panel.marks) == len(_ROWS)
        assert {mark.key for mark in panel.marks} == {"0", "1", "2", "3"}

    def test_marks_have_no_groups(self) -> None:
        """There is one series here, not several -- every mark's group is
        empty, and the panel declares no draw order because there is
        nothing to order."""
        panel = prepared()
        assert panel.groups == ()
        assert all(mark.group == "" for mark in panel.marks)

    def test_mark_position_and_size_come_from_the_row(self) -> None:
        panel = prepared()
        by_key = {mark.key: mark for mark in panel.marks}
        topic0 = by_key["0"]
        assert topic0.x == pytest.approx(-2.1)
        assert topic0.y == pytest.approx(1.4)
        assert topic0.size == pytest.approx(1.0)

    def test_shape_is_scatter_labelled(self) -> None:
        assert prepared().shape == "scatter_labelled"

    def test_data_carries_the_working_frame(self) -> None:
        panel = prepared()
        assert set(panel.data[TOPIC].astype(int)) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Evidence -- the part that makes a figure a finding
# ---------------------------------------------------------------------------


class TestEvidenceResolves:
    def test_every_mark_carries_evidence(self) -> None:
        panel = prepared()
        assert panel.marks
        for mark in panel.marks:
            assert mark.evidence.scope == "rows"
            assert mark.evidence.count >= 0
            assert mark.evidence.describe.strip()

    def test_a_marks_filters_select_exactly_the_row_it_came_from(self) -> None:
        """The test this protocol exists for: filters taken back to the
        source table must find the one row the mark was built from. A
        plausible filter that selects nothing looks identical until someone
        clicks."""
        frame = intertopic_frame()
        panel = prepared()
        for mark in panel.marks:
            selected = frame
            for column, value in mark.evidence.filters:
                assert column in frame.columns, f"{mark.key}: filter names a column the table lacks"
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == 1, f"{mark.key}: filters selected {len(selected)} rows, expected 1"
            row = selected.iloc[0]
            assert float(row[X]) == pytest.approx(mark.x)
            assert float(row[Y]) == pytest.approx(mark.y)
            assert float(row[PREVALENCE]) == pytest.approx(mark.size)

    def test_evidence_names_the_topic_and_its_prevalence(self) -> None:
        panel = prepared()
        topic0 = panel.evidence_for("0")
        assert topic0 is not None
        assert "0" in topic0.describe
        assert "1.0" in topic0.describe or "1.000" in topic0.describe

    def test_evidence_count_is_the_one_row_behind_the_bubble(self) -> None:
        panel = prepared()
        for mark in panel.marks:
            assert mark.evidence.count == 1


# ---------------------------------------------------------------------------
# label-topics / label-top
# ---------------------------------------------------------------------------


class TestLabelling:
    def test_all_labels_every_topic(self) -> None:
        panel = prepared({"label-topics": "all"})
        assert all(mark.labelled for mark in panel.marks)

    def test_none_labels_nothing(self) -> None:
        panel = prepared({"label-topics": "none"})
        assert not any(mark.labelled for mark in panel.marks)

    def test_largest_labels_only_the_top_n_by_prevalence(self) -> None:
        panel = prepared({"label-topics": "largest", "label-top": 2})
        labelled = {mark.key for mark in panel.marks if mark.labelled}
        # topics 0 and 1 are the two most prevalent (1.0, 0.82).
        assert labelled == {"0", "1"}

    def test_largest_respects_a_smaller_label_top(self) -> None:
        panel = prepared({"label-topics": "largest", "label-top": 1})
        labelled = {mark.key for mark in panel.marks if mark.labelled}
        assert labelled == {"0"}

    def test_label_top_is_ignored_when_label_topics_is_not_largest(self) -> None:
        all_labelled = prepared({"label-topics": "all", "label-top": 1})
        assert all(mark.labelled for mark in all_labelled.marks)


# ---------------------------------------------------------------------------
# The axes must not be misread
# ---------------------------------------------------------------------------


class TestAxesHaveNoIndependentMeaning:
    def test_a_note_annotation_says_so(self) -> None:
        panel = prepared()
        notes_annotations = [a for a in panel.annotations if a.kind == "note"]
        assert notes_annotations, "the axes-have-no-meaning warning must be an Annotation, not just prose"
        assert any("distance" in a.note.lower() for a in notes_annotations)

    def test_axis_labels_do_not_invent_a_dimension_name(self) -> None:
        panel = prepared()
        for label in (panel.x_label, panel.y_label):
            assert "mds" in label.lower()
            lowered = label.lower()
            assert "no meaning" in lowered or "no independent meaning" in lowered

    def test_notes_repeat_the_warning(self) -> None:
        panel = prepared()
        assert any("distance" in note.lower() and "meaning" in note.lower() for note in panel.notes)

    def test_notes_mention_prevalence_and_bubble_size(self) -> None:
        panel = prepared()
        assert any("prevalence" in note.lower() for note in panel.notes)

    def test_notes_mention_topic_numbers_are_not_stable(self) -> None:
        panel = prepared()
        assert any("seed" in note.lower() or "stable" in note.lower() for note in panel.notes)

    def test_notes_mention_2d_projection_distortion(self) -> None:
        panel = prepared()
        assert any("distortion" in note.lower() or "projection" in note.lower() for note in panel.notes)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_zero_topics_is_no_data(self) -> None:
        result = build(frame=intertopic_frame([]))
        assert not result.ok
        assert codes(result) == ["PANEL_NO_DATA"]

    def test_one_topic_is_no_data(self) -> None:
        """A distance map of one topic is not a thing: there is nothing to
        be relative to."""
        result = build(frame=intertopic_frame(_ROWS[:1]))
        assert not result.ok
        assert codes(result) == ["PANEL_NO_DATA"]

    def test_bad_numeric_rows_are_dropped_and_counted(self) -> None:
        rows = [dict(row) for row in _ROWS]
        rows[1]["x"] = float("nan")
        rows[2]["y"] = float("inf")
        result = build(frame=intertopic_frame(rows))
        assert result.ok, [str(d) for d in result.diagnostics]
        warning = find(result, "PANEL_BAD_NUMERIC")
        assert warning.severity == Severity.WARNING
        assert warning.context["dropped"] == 2
        panel = result.unwrap()
        assert len(panel.marks) == 2
        assert {mark.key for mark in panel.marks} == {"0", "3"}

    def test_dropping_bad_numeric_rows_down_to_one_topic_is_still_no_data(self) -> None:
        rows = [dict(row) for row in _ROWS[:2]]
        rows[1]["x"] = float("nan")
        result = build(frame=intertopic_frame(rows))
        assert not result.ok
        assert "PANEL_NO_DATA" in codes(result)
        assert "PANEL_BAD_NUMERIC" in codes(result)

    def test_identical_topics_emit_a_warning(self) -> None:
        rows = [
            {"topic": 0, "x": 0.0, "y": 0.0, "prevalence": 0.6},
            {"topic": 1, "x": 0.0, "y": 0.0, "prevalence": 0.4},
            {"topic": 2, "x": 0.0, "y": 0.0, "prevalence": 0.2},
        ]
        result = build(frame=intertopic_frame(rows))
        assert result.ok, [str(d) for d in result.diagnostics]
        warning = find(result, "PANEL_TOPICS_IDENTICAL")
        assert warning.severity == Severity.WARNING
        assert warning.context["topics"] == 3
        assert "too many topics" in warning.message or "more topics" in warning.message

    def test_distinct_topics_do_not_trigger_the_identical_warning(self) -> None:
        result = build()
        assert "PANEL_TOPICS_IDENTICAL" not in codes(result)


# ---------------------------------------------------------------------------
# Provenance passes through untouched
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_the_provenance_object_is_carried_through(self) -> None:
        given = provenance(**LDA_INTERTOPIC.defaults())
        result = lda_intertopic(intertopic_frame(), LDA_INTERTOPIC.defaults(), given)
        assert result.ok
        assert result.unwrap().provenance is given
