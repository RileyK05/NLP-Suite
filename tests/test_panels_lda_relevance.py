"""The topic-term relevance panel: scores order the rows, counts draw them.

**The fixture comes from the engine, not from imagination.** The first
version of this file invented its rows -- relevance 5.2, 4.0, 3.5 -- and
every test passed, while on the real corpus the panel was misleading. The
engine can never produce a positive relevance: it is a sum of logs of
probabilities. Tests written from an impossible fixture agree with whatever
misconception wrote them. So the main fixture here is
:func:`core.analysis.lda.relevance_terms` run on a small model whose numbers
can be checked by hand, and it cannot hold values the engine would never
emit. A second, hand-built frame appears only where a test needs a case the
small model does not produce, and it keeps the engine's signs.

The builder is called directly: the registry's gate is tested in
``tests/test_panels.py``, and this file is about what the panel claims.

Three regressions are pinned, each found on the real 87-address model:

* bars are **counts**, never the log-scale scores (a negative score drawn
  from zero made the top-ranked word the shortest bar);
* ordering by saliency puts the most salient word first (the engine's
  saliency is negative, and sorting it as written put the least salient
  word on top);
* evidence resolves to exactly one row, with a control proving the Topic
  filter is load-bearing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from core.analysis.lda import relevance_terms
from core.result import Result, Severity
from core.viz.panels_lda_relevance import (
    CORPUS_FREQUENCY,
    LDA_RELEVANCE,
    RELEVANCE,
    SALIENCY,
    TOPIC,
    TOPIC_FREQUENCY,
    WORD,
    lda_relevance,
)
from core.viz.panelspec import PreparedPanel, Provenance

# A two-topic model small enough to check by hand. "the" is probable in both
# topics and frequent in both documents: the corpus-wide word. war/peace
# belong to topic 0; tax/job to topic 1.
VOCAB = ["war", "peace", "the", "tax", "job"]
TOKENS = {
    "d1": ["war"] * 6 + ["peace"] * 4 + ["the"] * 8 + ["tax"],
    "d2": ["tax"] * 6 + ["job"] * 4 + ["the"] * 8 + ["war"],
}
PHI = np.array(
    [
        [0.35, 0.22, 0.38, 0.03, 0.02],
        [0.03, 0.02, 0.38, 0.33, 0.24],
    ]
)
TOPIC_TOKENS = [19.0, 19.0]


def engine_frame() -> pd.DataFrame:
    """What ``lda_gensim`` would write to terms_by_relevance.csv."""
    return relevance_terms(PHI, VOCAB, TOKENS, lam=0.6, top_n=5, topic_tokens=TOPIC_TOKENS)


def build(frame: pd.DataFrame | None = None, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    return lda_relevance(
        engine_frame() if frame is None else frame,
        LDA_RELEVANCE.defaults() | (params or {}),
        Provenance(tool="lda_gensim", panel="lda_relevance", source="terms_by_relevance.csv"),
    )


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


def row_order(panel: PreparedPanel) -> list[str]:
    """Terms top to bottom: y is the row, 0 at the top."""
    return [m.label for m in sorted(panel.marks, key=lambda m: m.y) if m.group == "In this topic"]


class TestTheFixtureIsHonest:
    def test_the_engine_fixture_has_the_signs_real_models_have(self) -> None:
        """If this ever fails, the fixture stopped resembling real output and
        every test below is testing something that cannot happen."""
        frame = engine_frame()
        assert (frame[RELEVANCE] < 0).all()
        assert (frame[CORPUS_FREQUENCY] > 0).all()
        assert (frame[TOPIC_FREQUENCY] >= 0).all()


class TestBarsAreCounts:
    def test_no_bar_is_a_negative_score(self) -> None:
        """The regression. Relevance is always negative; drawn from zero, the
        best-ranked term got the shortest bar."""
        assert all(mark.x >= 0 for mark in prepared().marks)

    def test_each_bar_is_the_count_from_the_table(self) -> None:
        panel = prepared()
        table = engine_frame().set_index([TOPIC, WORD])
        for mark in panel.marks:
            row = table.loc[(0, mark.label)]
            expected = row[TOPIC_FREQUENCY] if mark.group == "In this topic" else row[CORPUS_FREQUENCY]
            assert mark.x == pytest.approx(float(expected))

    def test_the_gap_between_bars_is_what_the_panel_is_for(self) -> None:
        """A word common everywhere has a corpus bar that dwarfs its topic
        bar; a word the topic owns has two bars that nearly meet."""
        bars = {(m.label, m.group): m.x for m in prepared().marks}
        the_share = bars[("the", "In this topic")] / bars[("the", "In the whole corpus")]
        war_share = bars[("war", "In this topic")] / bars[("war", "In the whole corpus")]
        assert the_share < 0.5, "'the' is split across topics"
        assert war_share > 0.9, "'war' belongs to topic 0"

    def test_both_bars_of_a_term_share_its_row(self) -> None:
        by_label: dict[str, set[float]] = {}
        for mark in prepared().marks:
            by_label.setdefault(mark.label, set()).add(mark.y)
        assert all(len(rows) == 1 for rows in by_label.values())

    def test_the_topics_own_count_is_the_foreground_series(self) -> None:
        assert prepared().groups == ("In this topic", "In the whole corpus")

    def test_the_hover_says_the_share_in_words(self) -> None:
        war = next(m for m in prepared().marks if m.label == "war")
        assert "of its 7 occurrences" in war.evidence.describe
        assert "95%" in war.evidence.describe


class TestOrdering:
    def test_relevance_orders_as_the_engine_ranks_it(self) -> None:
        assert row_order(prepared({"order-by": "relevance"})) == ["the", "war", "peace", "tax", "job"]

    def test_saliency_puts_the_most_salient_word_first(self) -> None:
        """The engine's Saliency is the corrected Chuang et al. divergence:
        never negative, larger for more salient words, and ordering by it as
        written is correct. In this fixture the corpus-wide word 'the' has
        saliency exactly 0 (its p(t|w) equals the topic marginal for every
        topic), so the least salient word is 'the' -- not 'job'. The old
        inverted definition made this ordering impossible; this test dies if
        the definition ever regresses."""
        order = row_order(prepared({"order-by": "saliency"}))
        assert order[0] == "job", "the word with the highest saliency divergence leads"
        assert order[-1] == "the", "'the' is shared evenly by both topics and has zero saliency"

    def test_the_two_orderings_really_differ_on_this_topic(self) -> None:
        """Control for the test above: if the orders matched, it would pass
        without exercising the saliency rule at all."""
        assert row_order(prepared({"order-by": "relevance"})) != row_order(prepared({"order-by": "saliency"}))

    def test_top_n_keeps_the_head_of_the_chosen_ordering(self) -> None:
        assert row_order(prepared({"top-n": 2})) == ["the", "war"]

    def test_a_different_topic_draws_its_own_terms(self) -> None:
        assert row_order(prepared({"topic": 1}))[:3] == ["the", "tax", "job"]


class TestEvidenceResolves:
    def test_a_marks_filters_select_exactly_the_row_it_came_from(self) -> None:
        frame = engine_frame()
        for mark in prepared().marks:
            selected = frame
            for column, value in mark.evidence.filters:
                assert column in frame.columns
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == 1, f"{mark.key}: filters selected {len(selected)} rows"
            assert str(selected.iloc[0][WORD]) == mark.label

    def test_the_topic_filter_is_load_bearing(self) -> None:
        """Control: every word appears under both topics, so filtering by Word
        alone would select two rows. Without this, the test above could pass
        on a filter that never needed the Topic."""
        frame = engine_frame()
        assert len(frame[frame[WORD] == "the"]) == 2

    def test_a_topic_term_is_looked_up_across_its_inflections(self) -> None:
        """lda_gensim models lemmas, so reading 'war' in the text must also
        find 'wars'."""
        for mark in prepared().marks:
            assert mark.evidence.phrase == mark.label
            assert mark.evidence.lemma

    def test_surface_form_runs_are_searched_literally(self) -> None:
        provenance = Provenance(
            tool="lda_gensim",
            panel="lda_relevance",
            source="terms_by_relevance.csv",
            settings={"field": "form"},
        )
        panel = lda_relevance(engine_frame(), LDA_RELEVANCE.defaults(), provenance).unwrap()
        assert all(not mark.evidence.lemma for mark in panel.marks)

    def test_evidence_names_both_scores(self) -> None:
        describe = next(m for m in prepared().marks if m.label == "war").evidence.describe
        assert "Relevance" in describe and "Saliency" in describe
        assert "topic 0" in describe


class TestDiagnostics:
    def test_a_topic_not_in_the_frame_lists_the_ones_that_are(self) -> None:
        result = build(params={"topic": 7})
        assert not result.ok
        assert "[0, 1]" in find(result, "PANEL_NO_DATA").message

    def test_missing_topic_counts_say_to_re_run(self) -> None:
        """A table written before the counts existed has the columns but not
        the numbers; the message must point at the cure."""
        frame = engine_frame()
        frame.loc[frame[WORD] == "job", TOPIC_FREQUENCY] = np.nan
        result = build(frame)
        assert result.ok
        notice = find(result, "PANEL_BAD_NUMERIC")
        assert notice.severity is Severity.WARNING
        assert "re-run" in notice.message

    def test_a_topic_with_no_usable_rows_fails_rather_than_drawing_nothing(self) -> None:
        frame = engine_frame()
        frame[TOPIC_FREQUENCY] = np.nan
        result = build(frame)
        assert not result.ok
        assert "PANEL_NO_DATA" in codes(result)

    def test_agreeing_orderings_are_announced(self) -> None:
        """Hand-built with the engine's values: when both scores rank the
        terms the same way, order-by cannot change anything, and saying so is
        kinder than letting a reader hunt for a difference. Both scores order
        descending as written now, so agreement needs Saliency to follow the
        same shape as Relevance."""
        frame = pd.DataFrame(
            {
                TOPIC: [0, 0, 0],
                WORD: ["a", "b", "c"],
                RELEVANCE: [-1.0, -2.0, -3.0],
                SALIENCY: [0.30, 0.20, 0.10],
                CORPUS_FREQUENCY: [10, 8, 6],
                TOPIC_FREQUENCY: [9.0, 6.0, 3.0],
            }
        )
        assert find(build(frame), "PANEL_RANKINGS_AGREE").severity is Severity.INFO

    def test_disagreeing_orderings_are_not_announced(self) -> None:
        assert "PANEL_RANKINGS_AGREE" not in codes(build())


class TestDefinition:
    def test_the_panel_belongs_to_the_tool_that_writes_the_table(self) -> None:
        assert LDA_RELEVANCE.tool == "lda_gensim"
        assert LDA_RELEVANCE.shape == "ranked_bars"
        assert {CORPUS_FREQUENCY, TOPIC_FREQUENCY} <= set(LDA_RELEVANCE.requires)

    def test_every_param_explains_itself(self) -> None:
        for param in LDA_RELEVANCE.params:
            assert param.help.strip(), param.name

    def test_the_notes_say_the_bars_are_estimated_counts(self) -> None:
        notes = " ".join(LDA_RELEVANCE.notes).lower()
        assert "occurrences, not scores" in notes
        assert "estimate" in notes
        assert prepared().notes == LDA_RELEVANCE.notes

    def test_the_published_table_carries_the_counts_that_were_drawn(self) -> None:
        panel = prepared({"top-n": 3})
        assert list(panel.data[WORD]) == ["the", "war", "peace"]
        assert {TOPIC_FREQUENCY, CORPUS_FREQUENCY} <= set(panel.data.columns)
