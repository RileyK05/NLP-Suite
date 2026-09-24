"""The topic-stability panel: worst-seed and mean overlap, most stable first.

The main fixture is :func:`core.analysis.topic_stability.topic_stability`
run on a small corpus, not invented numbers -- ``tests/test_panels_lda_
relevance.py`` documents why: an earlier panel's fixture held values its
engine could never produce, every test passed, and the panel was misleading
on real data. The corpus below (8 topics fit over a 12-document, 2-theme
corpus) reliably produces a mix of Stable and not-Stable reference topics, so
the ordering and PANEL_MOSTLY_UNSTABLE tests are exercising a real mix rather
than a fixture built to make them pass. A second, hand-built frame appears
only where a test needs a specific case (an unusable value, a chosen stable
share) that the small model does not reliably produce, and it keeps Jaccard
in [0, 1] with Min <= Mean, as the engine always does.

The builder (``lda_stability``) is called directly. ``lda_stability`` is not
registered in ``core/viz/panels.py`` yet (that registration, and the tool
executor, belong to someone else), so these tests cannot go through
``prepare_panel`` -- they build ``LDA_STABILITY.defaults() | overrides`` and
a hand-made ``Provenance`` themselves, the way this file's sibling for
relevance does before its panel is registered.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from core.analysis.topic_stability import (
    MEAN_JACCARD as ENGINE_MEAN_JACCARD,
    MIN_JACCARD as ENGINE_MIN_JACCARD,
    REFERENCE_WORDS as ENGINE_REFERENCE_WORDS,
    SEEDS as ENGINE_SEEDS,
    STABLE as ENGINE_STABLE,
    STABLE_AT as ENGINE_STABLE_AT,
    TOPIC as ENGINE_TOPIC,
    topic_stability,
)
from core.result import Result, Severity
from core.viz.panels_lda_stability import (
    LDA_STABILITY,
    MEAN_JACCARD,
    MIN_JACCARD,
    REFERENCE_WORDS,
    SEEDS,
    STABLE,
    STABLE_AT,
    TOPIC,
    lda_stability,
)
from core.viz.panelspec import PreparedPanel, Provenance

# Engine and panel agree on the column names; asserted once so a rename in
# either module fails loudly here instead of silently in a column that
# looks right but isn't the one being read.
assert (TOPIC, REFERENCE_WORDS, MEAN_JACCARD, MIN_JACCARD, SEEDS, STABLE, STABLE_AT) == (
    ENGINE_TOPIC,
    ENGINE_REFERENCE_WORDS,
    ENGINE_MEAN_JACCARD,
    ENGINE_MIN_JACCARD,
    ENGINE_SEEDS,
    ENGINE_STABLE,
    ENGINE_STABLE_AT,
)

THEME_A = ["war", "army", "battle", "soldier", "troops", "combat"]
THEME_B = ["tax", "budget", "spending", "revenue", "deficit", "fiscal"]


def _doc(theme: list[str], offset: int, n: int = 30) -> list[str]:
    return [theme[(i * 7 + offset) % len(theme)] for i in range(n)]


def _corpus() -> dict[str, list[str]]:
    tokens: dict[str, list[str]] = {}
    for i in range(6):
        tokens[f"war{i}.txt"] = _doc(THEME_A, i)
    for i in range(6):
        tokens[f"tax{i}.txt"] = _doc(THEME_B, i)
    return tokens


DOC_TOKENS = _corpus()


def engine_summary() -> pd.DataFrame:
    """8 topics over a 12-document corpus: too many topics for the corpus to
    support all of them, which is exactly the mix (some Stable, some not)
    this panel needs to be tested honestly. Computed once per test session."""
    result = topic_stability(DOC_TOKENS, n_topics=8, seeds=(100, 101, 102), top_n=4, passes=15)
    assert result.ok, result.diagnostics
    return result.unwrap().summary


def build(frame: pd.DataFrame | None = None, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    return lda_stability(
        engine_summary() if frame is None else frame,
        LDA_STABILITY.defaults() | (params or {}),
        Provenance(tool="lda_stability", panel="lda_stability", source="topic_stability_summary.csv"),
    )


def prepared(params: dict[str, Any] | None = None, frame: pd.DataFrame | None = None) -> PreparedPanel:
    result = build(frame, params)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def codes(result: Result[Any]) -> list[str]:
    return [d.code for d in result.diagnostics]


def find(result: Result[Any], code: str) -> Any:
    matching = [d for d in result.diagnostics if d.code == code]
    assert len(matching) == 1, f"expected exactly one {code}, got {codes(result)}"
    return matching[0]


def topic_order(panel: PreparedPanel) -> list[int]:
    """Reference topics top to bottom, read from each mark's own evidence
    filter rather than parsed out of the label text."""
    by_y: dict[float, int] = {}
    for mark in panel.marks:
        topic = int(dict(mark.evidence.filters)[TOPIC])
        by_y[mark.y] = topic
    return [topic for _, topic in sorted(by_y.items())]


def hand_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """A hand-built summary with exactly the columns the rows carry: six for
    the shape an older engine wrote, seven when the rows include the
    Stable-at cutoff. Never pads columns the rows did not bring — the
    panel's tolerance for the missing cutoff is part of what gets tested."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gensim-backed: the engine fixture.
# ---------------------------------------------------------------------------
pytest.importorskip("gensim")


class TestTheFixtureIsHonest:
    def test_the_engine_fixture_has_both_a_stable_and_an_unstable_topic(self) -> None:
        """Control for the ordering and PANEL_MOSTLY_UNSTABLE tests below: if
        every topic landed on one side, those tests would pass without the
        corpus actually forcing a mix."""
        summary = engine_summary()
        assert summary[ENGINE_STABLE].any(), "need at least one Stable topic"
        assert not summary[ENGINE_STABLE].all(), "need at least one not-Stable topic"

    def test_min_never_exceeds_mean(self) -> None:
        summary = engine_summary()
        assert (summary[ENGINE_MIN_JACCARD] <= summary[ENGINE_MEAN_JACCARD] + 1e-9).all()

    def test_jaccard_values_are_in_range(self) -> None:
        summary = engine_summary()
        for column in (ENGINE_MEAN_JACCARD, ENGINE_MIN_JACCARD):
            assert (summary[column] >= 0.0).all()
            assert (summary[column] <= 1.0).all()


class TestOrdering:
    def test_rows_are_ordered_by_min_jaccard_descending(self) -> None:
        summary = engine_summary()
        # Computed independently of the builder, from the same summary the
        # builder was given -- this checks the builder's order against the
        # data, not against itself.
        expected = summary.sort_values([ENGINE_MIN_JACCARD, ENGINE_TOPIC], ascending=[False, True], kind="stable")[
            ENGINE_TOPIC
        ].tolist()
        assert topic_order(prepared()) == expected

    def test_the_most_stable_topic_leads(self) -> None:
        summary = engine_summary()
        best = summary.loc[summary[ENGINE_MIN_JACCARD].idxmax(), ENGINE_TOPIC]
        assert topic_order(prepared())[0] == int(best)

    def test_top_n_keeps_the_most_stable_end(self) -> None:
        summary = engine_summary()
        expected_all = summary.sort_values([ENGINE_MIN_JACCARD, ENGINE_TOPIC], ascending=[False, True], kind="stable")[
            ENGINE_TOPIC
        ].tolist()
        assert topic_order(prepared({"top-n": 3})) == expected_all[:3]


class TestBothSeriesShareOneScale:
    def test_no_bar_is_outside_zero_one(self) -> None:
        assert all(0.0 <= mark.x <= 1.0 for mark in prepared().marks)

    def test_worst_seed_never_exceeds_mean_overlap(self) -> None:
        by_topic_and_group = {(int(dict(m.evidence.filters)[TOPIC]), m.group): m.x for m in prepared().marks}
        topics = {topic for topic, _ in by_topic_and_group}
        for topic in topics:
            assert by_topic_and_group[(topic, "Worst seed")] <= by_topic_and_group[(topic, "Mean overlap")] + 1e-9

    def test_both_bars_of_a_topic_share_its_row(self) -> None:
        by_label: dict[str, set[float]] = {}
        for mark in prepared().marks:
            by_label.setdefault(mark.label, set()).add(mark.y)
        assert all(len(rows) == 1 for rows in by_label.values())

    def test_the_worst_seed_series_is_foreground(self) -> None:
        assert prepared().groups == ("Worst seed", "Mean overlap")


class TestLabelling:
    def test_each_row_is_labelled_with_the_topics_own_words(self) -> None:
        summary = engine_summary()
        by_topic = summary.set_index(ENGINE_TOPIC)[ENGINE_REFERENCE_WORDS]
        for mark in prepared().marks:
            topic = int(dict(mark.evidence.filters)[TOPIC])
            first_words = ", ".join(str(by_topic[topic]).split(", ")[:3])
            assert mark.label == f"Topic {topic}: {first_words}"


class TestEvidenceResolves:
    def test_a_marks_filters_select_exactly_the_row_it_came_from(self) -> None:
        panel = prepared()
        for mark in panel.marks:
            selected = panel.data
            for column, value in mark.evidence.filters:
                assert column in panel.data.columns
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == 1, f"{mark.key}: filters selected {len(selected)} rows"
            assert int(selected.iloc[0][TOPIC]) == int(dict(mark.evidence.filters)[TOPIC])

    def test_evidence_never_sets_a_lookup_phrase(self) -> None:
        """scope='rows' evidence has no honest term to look up; Evidence
        itself refuses a phrase outside scope='terms', and this pins that
        the builder never tries to set one."""
        assert all(mark.evidence.phrase == "" for mark in prepared().marks)

    def test_evidence_count_is_the_number_of_seeds(self) -> None:
        summary = engine_summary()
        by_topic = summary.set_index(ENGINE_TOPIC)[ENGINE_SEEDS]
        for mark in prepared().marks:
            topic = int(dict(mark.evidence.filters)[TOPIC])
            assert mark.evidence.count == int(by_topic[topic])

    def test_describe_names_the_words_and_both_scores(self) -> None:
        summary = engine_summary()
        topic = int(summary.iloc[0][ENGINE_TOPIC])
        mark = next(m for m in prepared().marks if int(dict(m.evidence.filters)[TOPIC]) == topic)
        words = str(summary.set_index(ENGINE_TOPIC).loc[topic, ENGINE_REFERENCE_WORDS])
        for word in words.split(", "):
            assert word in mark.evidence.describe
        assert "%" in mark.evidence.describe


class TestDiagnostics:
    def test_non_finite_jaccard_is_dropped_with_a_count(self) -> None:
        frame = hand_frame(
            [
                {TOPIC: 0, REFERENCE_WORDS: "a, b", MEAN_JACCARD: 0.9, MIN_JACCARD: 0.8, SEEDS: 2, STABLE: True},
                {
                    TOPIC: 1,
                    REFERENCE_WORDS: "c, d",
                    MEAN_JACCARD: float("nan"),
                    MIN_JACCARD: 0.2,
                    SEEDS: 2,
                    STABLE: False,
                },
                {
                    TOPIC: 2,
                    REFERENCE_WORDS: "e, f",
                    MEAN_JACCARD: 0.5,
                    MIN_JACCARD: float("inf"),
                    SEEDS: 2,
                    STABLE: False,
                },
            ]
        )
        result = build(frame)
        assert result.ok
        notice = find(result, "PANEL_BAD_NUMERIC")
        assert notice.severity is Severity.WARNING
        assert notice.context["dropped"] == 2
        assert list(result.unwrap().data[TOPIC]) == [0]

    def test_all_rows_unusable_fails_rather_than_drawing_nothing(self) -> None:
        frame = hand_frame(
            [
                {
                    TOPIC: 0,
                    REFERENCE_WORDS: "a, b",
                    MEAN_JACCARD: float("nan"),
                    MIN_JACCARD: 0.8,
                    SEEDS: 2,
                    STABLE: True,
                },
            ]
        )
        result = build(frame)
        assert not result.ok
        assert "PANEL_NO_DATA" in codes(result)

    def test_mostly_unstable_fires_on_the_engine_fixture(self) -> None:
        """This is the diagnostic that makes the panel informative rather than
        just decorative: on the real 8-topic fit most topics do not survive."""
        result = build()
        notice = find(result, "PANEL_MOSTLY_UNSTABLE")
        assert notice.severity is Severity.WARNING
        assert notice.context["unstable"] > notice.context["total"] / 2

    def test_mostly_unstable_does_not_fire_when_most_topics_are_stable(self) -> None:
        frame = hand_frame(
            [
                {TOPIC: 0, REFERENCE_WORDS: "a, b", MEAN_JACCARD: 0.9, MIN_JACCARD: 0.8, SEEDS: 2, STABLE: True},
                {TOPIC: 1, REFERENCE_WORDS: "c, d", MEAN_JACCARD: 0.9, MIN_JACCARD: 0.7, SEEDS: 2, STABLE: True},
                {TOPIC: 2, REFERENCE_WORDS: "e, f", MEAN_JACCARD: 0.9, MIN_JACCARD: 0.6, SEEDS: 2, STABLE: True},
                {TOPIC: 3, REFERENCE_WORDS: "g, h", MEAN_JACCARD: 0.4, MIN_JACCARD: 0.1, SEEDS: 2, STABLE: False},
            ]
        )
        result = build(frame)
        assert "PANEL_MOSTLY_UNSTABLE" not in codes(result)

    def test_mostly_unstable_fires_at_exactly_over_half(self) -> None:
        frame = hand_frame(
            [
                {TOPIC: 0, REFERENCE_WORDS: "a, b", MEAN_JACCARD: 0.9, MIN_JACCARD: 0.8, SEEDS: 2, STABLE: True},
                {TOPIC: 1, REFERENCE_WORDS: "c, d", MEAN_JACCARD: 0.4, MIN_JACCARD: 0.2, SEEDS: 2, STABLE: False},
                {TOPIC: 2, REFERENCE_WORDS: "e, f", MEAN_JACCARD: 0.4, MIN_JACCARD: 0.1, SEEDS: 2, STABLE: False},
            ]
        )
        assert "PANEL_MOSTLY_UNSTABLE" in codes(build(frame))

    def test_no_vline_annotation_is_invented_without_a_threshold(self) -> None:
        """A summary that predates the Stable-at column carries no numeric
        cutoff, and the panel must not invent one: it draws no line and says
        why (PANEL_THRESHOLD_MISSING) rather than inferring a threshold from
        the boolean split."""
        frame = hand_frame(
            [
                {TOPIC: 0, REFERENCE_WORDS: "a, b", MEAN_JACCARD: 0.9, MIN_JACCARD: 0.8, SEEDS: 2, STABLE: True},
                {TOPIC: 1, REFERENCE_WORDS: "c, d", MEAN_JACCARD: 0.4, MIN_JACCARD: 0.2, SEEDS: 2, STABLE: False},
            ]
        )
        result = build(frame)
        assert "PANEL_THRESHOLD_MISSING" in codes(result)
        panel = prepared(None, frame)
        assert panel.annotations == ()

    def test_the_run_s_own_threshold_is_drawn_when_the_summary_carries_it(self) -> None:
        """The corrected behaviour: the summary now carries Stable at, and
        the panel draws its line at exactly the run's own cutoff, labelled
        with the reason it is there."""
        frame = hand_frame(
            [
                {
                    TOPIC: 0,
                    REFERENCE_WORDS: "a, b",
                    MEAN_JACCARD: 0.9,
                    MIN_JACCARD: 0.8,
                    SEEDS: 2,
                    STABLE: True,
                    STABLE_AT: 0.5,
                }
            ]
        )
        result = build(frame)
        assert result.ok, [str(d) for d in result.diagnostics]
        panel = result.unwrap()
        assert len(panel.annotations) == 1
        line = panel.annotations[0]
        assert line.value == 0.5
        assert "0.5" in line.label
        assert "cutoff" in line.note
        # And the published table carries the run's own number.
        assert list(panel.data["Stable at"]) == [0.5]

    def test_conflicting_thresholds_are_named_not_averaged(self) -> None:
        frame = hand_frame(
            [
                {
                    TOPIC: 0,
                    REFERENCE_WORDS: "a, b",
                    MEAN_JACCARD: 0.9,
                    MIN_JACCARD: 0.8,
                    SEEDS: 2,
                    STABLE: True,
                    STABLE_AT: 0.5,
                },
                {
                    TOPIC: 1,
                    REFERENCE_WORDS: "c, d",
                    MEAN_JACCARD: 0.4,
                    MIN_JACCARD: 0.2,
                    SEEDS: 2,
                    STABLE: False,
                    STABLE_AT: 0.6,
                },
            ]
        )
        result = build(frame)
        notice = find(result, "PANEL_THRESHOLD_CONFLICT")
        assert "0.5" in notice.message, "the drawn threshold is named, not blended"


class TestDefinition:
    def test_the_panel_belongs_to_its_own_tool(self) -> None:
        assert LDA_STABILITY.tool == "lda_stability"
        assert LDA_STABILITY.name == "lda_stability"
        assert LDA_STABILITY.shape == "ranked_bars"
        assert set(LDA_STABILITY.requires) == {TOPIC, REFERENCE_WORDS, MEAN_JACCARD, MIN_JACCARD, SEEDS, STABLE}

    def test_every_param_explains_itself(self) -> None:
        for param in LDA_STABILITY.params:
            assert param.help.strip(), param.name

    def test_top_n_is_bounded(self) -> None:
        param = next(p for p in LDA_STABILITY.params if p.name == "top-n")
        assert param.default == 20
        assert param.minimum == 1
        assert param.maximum == 100

    def test_the_notes_cover_the_four_required_caveats(self) -> None:
        notes = " ".join(LDA_STABILITY.notes).lower()
        assert "top words" in notes and "tail" in notes
        assert "reproducible" in notes and "meaningful" in notes
        assert "small corpus" in notes
        assert "arbitrary" in notes and "differ between seeds" in notes
        assert prepared().notes == LDA_STABILITY.notes

    def test_the_published_table_carries_the_rows_that_were_drawn(self) -> None:
        panel = prepared({"top-n": 2})
        assert len(panel.data) == 2
        assert {TOPIC, REFERENCE_WORDS, MEAN_JACCARD, MIN_JACCARD, SEEDS, STABLE} <= set(panel.data.columns)
