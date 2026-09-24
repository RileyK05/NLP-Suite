"""Tests for ``core/viz/panels_svo_agency.py`` -- who acts, who is acted upon.

**Why this file calls the builder directly instead of going through
``prepare_panel``.** ``svo_agency`` is not one of the panels listed in
``core/viz/panels.py`` -- that file is owned centrally and this change does
not touch it. Going through the registry would only ever produce
``PANEL_UNKNOWN``. These tests instead call :func:`svo_agency` as the plain
function it is: ``(frame, params, provenance) -> Result[PreparedPanel]``,
with ``params`` already the dict the registry would have produced --
``SVO_AGENCY.defaults()`` merged with whatever a test overrides -- and a
:class:`~core.viz.panelspec.Provenance` built by hand in place of what the
registry normally assembles. Same approach as
``tests/test_panels_lda_relevance.py``.

Two things carry most of the weight, same as everywhere else in this
package:

* **Evidence resolves against the published table.** An entity's identity is
  normalised (stripped, casefolded), so a single filter on the raw
  ``Subject``/``Object`` column cannot express "any spelling of this
  entity" -- the builder publishes ``Subject (normalised)`` and
  ``Object (normalised)`` columns instead, and ``TestEvidenceResolves`` takes
  every mark's filters back to ``panel.data`` and checks they land on exactly
  the count the mark claims. The fixture below is built so that at least one
  entity ("government") has two real spellings in the source data
  ("Government" and "government"), so this test is not able to pass by
  accident -- a builder that filtered on the raw column would undercount it.
* **The panel says what it means.** ``TestRanking`` checks that the four
  ``order-by`` choices are honoured as distinct, un-blended orderings
  (including a fixture built so 'subject' and 'balance' disagree, which a
  coincidence in the main fixture would not prove), and ``TestNotes`` checks
  the limits the task called out: the parser/heuristic origin of Subject and
  Object, the passive-voice failure mode as the extractor actually handles
  it, the ``Inferred_Subject_Passive`` placeholder, the pronoun/coreference
  caveat, and why ``balance`` needs ``min-count``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.result import Result, Severity
from core.viz.panels_svo_agency import (
    OBJECT,
    OBJECT_NORM,
    SUBJECT,
    SUBJECT_NORM,
    SVO_AGENCY,
    svo_agency,
)
from core.viz.panelspec import PreparedPanel, Provenance

# A small corpus of (Subject, Object) triples built to exercise every path:
#
# * "government": subject 5x (3x "Government", 2x "government"), object 2x
#   (both spelled "Government") -- the two-spellings-of-one-entity case
#   TestEvidenceResolves needs to be a real test rather than a trivial one.
# * "citizens": subject 2x, object 5x -- mostly acted upon.
# * "congress": subject 4x, object 4x -- even.
# * "president": subject 6x, object 1x -- mostly acts.
# * "rareword" and every filler object/subject below: total appearances of 1,
#   below the default min-count of 3, so they are filtered out and counted.
# * one row where Subject and Object are both ``None`` and one where both are
#   ``""`` -- the two blank spellings of "no triple here" PANEL_EMPTY_TRIPLES
#   must catch.
# * one row where only Object is blank (a real clause with no object) --
#   deliberately NOT counted as an empty triple.
_ROWS: list[tuple[str | None, str | None]] = [
    ("Government", "policy"),
    ("Government", "funding"),
    ("Government", "reform"),
    ("government", "budget"),
    ("government", "oversight"),
    ("citizens", "benefits"),
    ("Citizens", "healthcare"),
    ("congress", "education"),
    ("congress", "trade"),
    ("Congress", "immigration"),
    ("congress", "defense"),
    ("president", "congress"),
    ("President", "Government"),
    ("president", "citizens"),
    ("president", "budget2"),
    ("president", "address"),
    ("president", "nation"),
    ("senator1", "congress"),
    ("senator2", "congress"),
    ("senator3", "congress"),
    ("voter1", "citizens"),
    ("voter2", "citizens"),
    ("voter3", "citizens"),
    ("voter4", "citizens"),
    ("lobbyist1", "Government"),
    ("voter5", "president"),
    ("rareword", "fillerobjx"),
    (None, None),
    ("", ""),
    ("filler_subj_x", None),
]


def frame(rows: list[tuple[str | None, str | None]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                SUBJECT: subject,
                OBJECT: obj,
                "Verb": "acted",
                "Sentence ID": index,
                "Document ID": "d1",
                "Document": "doc.txt",
            }
            for index, (subject, obj) in enumerate(rows if rows is not None else _ROWS)
        ]
    )


def build(
    rows: list[tuple[str | None, str | None]] | None = None,
    params: dict[str, Any] | None = None,
    provenance: Provenance | None = None,
) -> Result[PreparedPanel]:
    settings = SVO_AGENCY.defaults() | (params or {})
    return svo_agency(
        frame(rows),
        settings,
        provenance or Provenance(tool="clause_svo", panel="svo_agency", source="svo.csv"),
    )


def prepared(
    rows: list[tuple[str | None, str | None]] | None = None,
    params: dict[str, Any] | None = None,
) -> PreparedPanel:
    result = build(rows, params)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def codes(result: Result[Any]) -> list[str]:
    return [d.code for d in result.diagnostics]


def find(result: Result[Any], code: str) -> Any:
    matching = [d for d in result.diagnostics if d.code == code]
    assert len(matching) == 1, f"expected one {code}, got {codes(result)}"
    return matching[0]


def subject_marks(panel: PreparedPanel) -> list[Any]:
    return [mark for mark in panel.marks if mark.group == "As subject"]


# ---------------------------------------------------------------------------
# The definition itself
# ---------------------------------------------------------------------------


class TestDefinition:
    def test_the_definition_declares_the_contract(self) -> None:
        assert SVO_AGENCY.name == "svo_agency"
        assert SVO_AGENCY.tool == "clause_svo"
        assert SVO_AGENCY.shape == "ranked_bars"
        assert set(SVO_AGENCY.requires) == {SUBJECT, OBJECT}

    def test_every_param_has_help_and_the_definition_has_notes(self) -> None:
        """This panel is not registered (a separate step, not this file's
        job), so the central check in ``tests/test_panels.py`` never runs
        against it -- worth asserting locally."""
        assert SVO_AGENCY.notes
        assert SVO_AGENCY.requires
        for param in SVO_AGENCY.params:
            assert param.help.strip(), f"{param.name} has no help"

    def test_params_match_the_task(self) -> None:
        by_name = {param.name: param for param in SVO_AGENCY.params}
        assert by_name["top-n"].default == 20
        assert by_name["top-n"].minimum == 1
        assert by_name["top-n"].maximum == 100
        assert by_name["order-by"].default == "total"
        assert set(by_name["order-by"].choices) == {"total", "subject", "object", "balance"}
        assert by_name["min-count"].default == 3
        assert by_name["min-count"].minimum == 1
        assert by_name["min-count"].maximum == 10_000


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_triples_with_neither_a_subject_nor_an_object_are_counted(self) -> None:
        result = build()
        assert result.ok
        notice = find(result, "PANEL_EMPTY_TRIPLES")
        assert notice.severity is Severity.INFO
        assert notice.context["count"] == 2, "the (None, None) row and the ('', '') row, and no others"

    def test_a_row_with_only_one_blank_side_is_not_counted_as_empty(self) -> None:
        """`('filler_subj_x', None)` is a real clause with no object -- it
        must not inflate PANEL_EMPTY_TRIPLES, which is specifically about
        triples with nothing usable on either side."""
        result = build()
        assert find(result, "PANEL_EMPTY_TRIPLES").context["count"] == 2

    def test_min_count_filtering_reports_what_it_removed(self) -> None:
        result = build()
        assert result.ok
        notice = find(result, "PANEL_MIN_COUNT_FILTERED")
        assert notice.severity is Severity.INFO
        assert notice.context["minimum"] == 3
        assert notice.context["removed"] > 0
        panel = result.unwrap()
        assert {mark.label for mark in panel.marks} == {"Government", "citizens", "congress", "president"}

    def test_filtering_everything_away_fails_with_advice(self) -> None:
        result = build(params={"min-count": 1000})
        assert not result.ok
        assert "min-count" in find(result, "PANEL_NO_DATA").message

    def test_a_table_of_only_blank_triples_fails_rather_than_drawing_nothing(self) -> None:
        result = build(rows=[(None, None), ("", "")])
        assert not result.ok
        assert codes(result) == ["PANEL_NO_DATA", "PANEL_EMPTY_TRIPLES"], (
            "both blank rows are named on the way to the failure, not swallowed by it"
        )

    def test_a_pronoun_heavy_selection_is_flagged(self) -> None:
        """Four entities tied at the top of the ranking, three of them
        pronouns -- the case PANEL_PRONOUN_HEAVY exists to name."""
        rows = [
            ("we", "policy"),
            ("we", "funding"),
            ("we", "reform"),
            ("it", "budget"),
            ("it", "oversight"),
            ("it", "x1"),
            ("they", "y1"),
            ("they", "y2"),
            ("they", "y3"),
            ("congress", "trade"),
            ("congress", "defense"),
            ("congress", "immigration"),
        ]
        result = build(rows=rows, params={"min-count": 1, "top-n": 4})
        assert result.ok
        notice = find(result, "PANEL_PRONOUN_HEAVY")
        assert notice.severity is Severity.WARNING
        assert notice.context == {"pronouns": 3, "entities": 4}
        assert "coreference" in notice.message

    def test_a_corpus_that_is_not_pronoun_heavy_is_not_flagged(self) -> None:
        assert "PANEL_PRONOUN_HEAVY" not in codes(build())


# ---------------------------------------------------------------------------
# Evidence -- the part that makes a figure a finding
# ---------------------------------------------------------------------------


class TestEvidenceResolves:
    def test_every_mark_carries_evidence(self) -> None:
        panel = prepared()
        assert panel.marks
        for mark in panel.marks:
            assert mark.evidence.scope == "sentences"
            assert mark.evidence.describe.strip()

    def test_a_marks_filters_select_exactly_the_rows_it_counted(self) -> None:
        """The load-bearing test. Every drawn entity in this fixture has a
        non-zero count on both sides, so this also checks the count is never
        zero for a real mark -- a filter that resolves to nothing looks
        identical to one that resolves correctly until someone clicks it."""
        panel = prepared()
        assert SUBJECT_NORM in panel.data.columns
        assert OBJECT_NORM in panel.data.columns
        for mark in panel.marks:
            selected = panel.data
            for column, value in mark.evidence.filters:
                assert column in panel.data.columns, f"{mark.key}: filter names a column the table lacks"
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == mark.evidence.count, (
                f"{mark.key}: filters found {len(selected)} rows but the mark claims {mark.evidence.count}"
            )
            assert len(selected) > 0

    def test_normalisation_is_genuinely_exercised_not_trivially_passed(self) -> None:
        """Control assertion: 'government' really does have two spellings in
        the source data. A filter on the raw Subject column alone would
        undercount it (3 of 5), which is exactly why the builder filters on
        the normalised column instead."""
        source = frame()
        raw_spellings = set(source.loc[source[SUBJECT].notna(), SUBJECT].astype(str))
        assert {"Government", "government"} <= raw_spellings, "fixture must carry two real spellings"

        panel = prepared()
        government = next(mark for mark in panel.marks if mark.label == "Government" and mark.group == "As subject")
        raw_only = int((source[SUBJECT] == "Government").sum())
        assert raw_only < government.evidence.count, "a raw-column filter would have undercounted this entity"
        assert government.evidence.count == 5, "3x 'Government' + 2x 'government' as subject"

    def test_display_form_is_the_most_frequent_spelling(self) -> None:
        panel = prepared()
        government_marks = [mark for mark in panel.marks if mark.key.startswith("government:")]
        assert {mark.label for mark in government_marks} == {"Government"}, (
            "'Government' (5 occurrences) outnumbers 'government' (2) across subject and object slots"
        )

    def test_describe_names_the_entity_the_role_the_count_and_the_total(self) -> None:
        panel = prepared()
        president_subject = next(
            mark for mark in panel.marks if mark.label == "president" and mark.group == "As subject"
        )
        assert "president" in president_subject.evidence.describe
        assert "subject" in president_subject.evidence.describe
        assert "6" in president_subject.evidence.describe, "the subject count"
        assert "7" in president_subject.evidence.describe, "the entity's total (6 subject + 1 object)"


# ---------------------------------------------------------------------------
# What the panel actually claims
# ---------------------------------------------------------------------------


class TestRanking:
    def test_default_order_is_by_total_ties_broken_alphabetically(self) -> None:
        """congress: 4+4=8. citizens, government (as 'Government'), and
        president are all tied at 7, so alphabetical order (by the
        normalised key: citizens < government < president) decides."""
        panel = prepared()
        assert [mark.label for mark in subject_marks(panel)] == ["congress", "citizens", "Government", "president"]

    def test_order_by_subject_ranks_the_most_frequent_actor_first(self) -> None:
        panel = prepared(params={"order-by": "subject"})
        assert [mark.label for mark in subject_marks(panel)] == ["president", "Government", "congress", "citizens"]

    def test_order_by_object_ranks_the_most_frequent_patient_first(self) -> None:
        panel = prepared(params={"order-by": "object"})
        assert [mark.label for mark in subject_marks(panel)] == ["citizens", "congress", "Government", "president"]

    def test_balance_and_subject_orderings_can_disagree(self) -> None:
        """The proof that 'balance' is its own ranking and not just 'subject'
        wearing a different name: an entity with a huge subject count but an
        almost-equal object count ('steady') loses to one with a small but
        pure subject count ('purist') once the ranking is a ratio."""
        rows = [("steady", f"obj{i}") for i in range(10)]
        rows += [(f"subj{i}", "steady") for i in range(9)]
        rows += [("purist", f"pobj{i}") for i in range(3)]

        by_subject = prepared(rows=rows, params={"order-by": "subject"})
        assert [mark.label for mark in subject_marks(by_subject)] == ["steady", "purist"]

        by_balance = prepared(rows=rows, params={"order-by": "balance"})
        assert [mark.label for mark in subject_marks(by_balance)] == ["purist", "steady"], (
            "purist: (3-0)/3 = 1.0 outranks steady: (10-9)/19 = 0.05"
        )

    def test_x_is_always_the_raw_count_never_the_balance_ratio(self) -> None:
        """Both series are counts on one scale -- that is what makes
        overlaying them legitimate -- so re-ranking by balance must not turn
        the bar length into the ratio itself."""
        rows = [("steady", f"obj{i}") for i in range(10)]
        rows += [(f"subj{i}", "steady") for i in range(9)]
        rows += [("purist", f"pobj{i}") for i in range(3)]
        panel = prepared(rows=rows, params={"order-by": "balance"})
        by_label = {mark.label: mark.x for mark in subject_marks(panel)}
        assert by_label["steady"] == 10.0
        assert by_label["purist"] == 3.0

    def test_rank_zero_is_drawn_at_the_top(self) -> None:
        panel = prepared()
        top = min(panel.marks, key=lambda mark: mark.y)
        assert top.label == "congress"
        assert top.y == 0.0

    def test_top_n_limits_how_many_entities_are_drawn(self) -> None:
        panel = prepared(params={"top-n": 2})
        assert {mark.label for mark in subject_marks(panel)} == {"congress", "citizens"}

    def test_two_marks_per_entity_share_one_row(self) -> None:
        panel = prepared()
        by_label: dict[str, list[Any]] = {}
        for mark in panel.marks:
            by_label.setdefault(mark.label, []).append(mark)
        assert len(by_label) == 4
        for label, marks in by_label.items():
            assert len(marks) == 2, label
            assert marks[0].y == marks[1].y, f"{label}: the two series must share a row"
            assert {marks[0].group, marks[1].group} == {"As subject", "As object"}

    def test_groups_have_a_declared_draw_order(self) -> None:
        assert prepared().groups == ("As subject", "As object")

    def test_both_series_are_counts_on_one_scale(self) -> None:
        """The reason overlaying them is legitimate in the first place."""
        panel = prepared()
        assert all(mark.x >= 0.0 for mark in panel.marks)

    def test_the_published_table_carries_both_normalised_columns(self) -> None:
        panel = prepared()
        assert SUBJECT_NORM in panel.data.columns
        assert OBJECT_NORM in panel.data.columns
        # The published table is not trimmed down to only the drawn
        # entities -- evidence for a filtered-out entity must still resolve.
        assert len(panel.data) > len(subject_marks(panel))

    def test_the_panel_definitions_notes_match_the_prepared_panels_notes(self) -> None:
        assert SVO_AGENCY.notes == prepared().notes


# ---------------------------------------------------------------------------
# Notes -- what the figure cannot show on its own
# ---------------------------------------------------------------------------


class TestNotes:
    def test_the_notes_state_the_parser_origin_and_the_passive_failure_mode(self) -> None:
        notes = " ".join(prepared().notes)
        assert "dependency parser" in notes
        assert "nsubj:pass" in notes or "nsubjpass" in notes, "the actual deprel the extractor looks for"
        assert "aux:pass" in notes, "the specific partial-tagging failure the extractor does not correct"

    def test_the_notes_name_the_inferred_subject_placeholder(self) -> None:
        """A real artefact of extract_svo -- an agentless passive gets the
        literal string 'Inferred_Subject_Passive' in Subject, not a word
        from the text -- and a reader needs to know that before treating a
        high subject count for it as a finding."""
        notes = " ".join(prepared().notes)
        assert "Inferred_Subject_Passive" in notes

    def test_the_notes_warn_about_pronouns_needing_coreference(self) -> None:
        notes = " ".join(prepared().notes).lower()
        assert "coreference" in notes
        assert "pronoun" in notes

    def test_the_notes_explain_why_balance_needs_min_count(self) -> None:
        notes = " ".join(prepared().notes).lower()
        assert "balance" in notes
        assert "min-count" in notes

    def test_the_notes_warn_that_frequent_mentions_can_still_look_minor(self) -> None:
        notes = " ".join(prepared().notes).lower()
        assert "triples" in notes


# ---------------------------------------------------------------------------
# Provenance passthrough
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_provenance_is_carried_through_unchanged(self) -> None:
        provenance = Provenance(tool="clause_svo", panel="svo_agency", source="runs/svo/svo.csv")
        result = build(provenance=provenance)
        assert result.ok
        assert result.unwrap().provenance is provenance
        assert "runs/svo/svo.csv" in result.unwrap().caption

    def test_severity_of_the_no_data_failure_is_error(self) -> None:
        result = build(params={"min-count": 1000})
        assert result.diagnostics[0].severity is Severity.ERROR
