"""Topic flow: each speech as a ribbon of its paragraphs' dominant topics.

The builder is called directly. The fixture is hand-built, but only with
values the engine can produce -- integer topics below k, contributions in
(0, 1], segment numbers from 1 to the speech's own count -- and an
engine-derived test lives beside ``fit_lda``'s segment scoring in
``tests/test_lda.py``. What these pin:

* the segmentation rule is always reported, and an unaligned speech is named;
* a paragraph too short to score is a gap, never recoloured;
* switches and runs are counted over scored paragraphs only;
* evidence resolves against the published table;
* colours are grouped exactly as the prevalence panel groups them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from core.result import Result, Severity
from core.viz.panels_lda_flow import LDA_FLOW, lda_flow
from core.viz.panelspec import PreparedPanel, Provenance


def speech(
    name: str, topics: list[int | None], tokens: list[int] | None = None, rule: str = "line", aligned: bool = True
) -> list[dict[str, Any]]:
    """One speech's flow rows. ``None`` is a paragraph the model could not score."""
    counts = tokens or [40] * len(topics)
    return [
        {
            "Document ID": name,
            "Document": name,
            "Segment": number,
            "Segments": len(topics),
            "Rule": rule,
            "Aligned": aligned,
            "Tokens": count if topic is not None else 0,
            "Dominant topic": topic,
            "Contribution": 0.7 if topic is not None else None,
            "Topic keywords": {0: "war, army, battle", 1: "tax, budget, spending", 2: "school, child, teacher"}.get(
                topic if topic is not None else -1, ""
            ),
            "Start": None,
            "End": None,
        }
        for number, (topic, count) in enumerate(zip(topics, counts, strict=True), start=1)
    ]


def flow_frame(*speeches: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([row for rows in speeches for row in rows])


def build(frame: pd.DataFrame, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    return lda_flow(
        frame, LDA_FLOW.defaults() | (params or {}), Provenance(tool="lda_gensim", panel="lda_flow", source="f.csv")
    )


def ok(frame: pd.DataFrame, params: dict[str, Any] | None = None) -> PreparedPanel:
    result = build(frame, params)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def find(result: Result[Any], code: str) -> Any:
    matching = [d for d in result.diagnostics if d.code == code]
    assert len(matching) == 1, f"expected one {code}, got {[d.code for d in result.diagnostics]}"
    return matching[0]


FDR = speech("1934-01-03_fdr_sotu.txt", [0, 0, 1, 1, 1, 2])
BIDEN = speech("2024-03-07_biden_sotu.txt", [1, 2, 1, 2])


class TestRibbons:
    def test_one_band_per_speech_in_chronological_order(self) -> None:
        panel = ok(flow_frame(BIDEN, FDR))
        # Short labels ("1934 ..."), as a reader names the speech.
        rows = {mark.label[:4]: mark.y for mark in panel.marks}
        assert rows["1934"] == 0.0
        assert rows["2024"] == 1.0
        assert not any(mark.label.endswith(".txt") for mark in panel.marks)

    def test_relative_alignment_stretches_every_speech_to_one_length(self) -> None:
        panel = ok(flow_frame(FDR, BIDEN), {"align": "relative"})
        for year in ("1934", "2024"):
            marks = sorted((m for m in panel.marks if m.label.startswith(year)), key=lambda m: m.x)
            assert marks[0].x == 0.0
            assert marks[-1].x + (marks[-1].size or 0) == pytest.approx(1.0)

    def test_absolute_alignment_gives_each_paragraph_one_unit(self) -> None:
        panel = ok(flow_frame(FDR, BIDEN), {"align": "absolute"})
        fdr = sorted((m for m in panel.marks if m.label.startswith("1934")), key=lambda m: m.x)
        assert [m.x for m in fdr] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert {m.size for m in fdr} == {1.0}

    def test_topics_are_grouped_as_the_prevalence_panel_groups_them(self) -> None:
        """Same group names, same order -> the same colour for a topic in both figures."""
        panel = ok(flow_frame(FDR, BIDEN))
        # Named by their words, numbered in order: "Topic 0: war, army, battle".
        assert [group.split(":")[0] for group in panel.groups] == ["Topic 0", "Topic 1", "Topic 2"]
        assert {m.group for m in panel.marks} <= set(panel.groups)

    def test_the_panel_declares_the_ribbon_shape(self) -> None:
        assert ok(flow_frame(FDR)).shape == "ribbon"


class TestGaps:
    def test_an_unscorable_paragraph_is_a_gap_not_a_colour(self) -> None:
        rows = speech("1950_truman_sotu.txt", [0, None, 1])
        panel = ok(flow_frame(rows))
        assert sorted(int(m.key.rsplit(":", 1)[1]) for m in panel.marks) == [1, 3]

    def test_a_short_paragraph_is_a_gap_and_is_counted(self) -> None:
        rows = speech("1950_truman_sotu.txt", [0, 0, 1], tokens=[40, 2, 40])
        result = build(flow_frame(rows), {"min-tokens": 5})
        notice = find(result, "PANEL_SHORT_SEGMENTS")
        assert notice.severity is Severity.INFO
        assert notice.context["segments"] == 1
        assert len(result.unwrap().marks) == 2

    def test_lowering_min_tokens_brings_the_short_paragraph_back(self) -> None:
        rows = speech("1950_truman_sotu.txt", [0, 0, 1], tokens=[40, 2, 40])
        assert len(ok(flow_frame(rows), {"min-tokens": 1}).marks) == 3


class TestSwitches:
    def test_switches_and_the_longest_run_are_in_the_hover(self) -> None:
        # 0 0 1 1 1 2 -> two switches, longest run three paragraphs.
        panel = ok(flow_frame(FDR))
        describe = panel.marks[0].evidence.describe
        assert "changes topic 2 time(s)" in describe
        assert "longest run is 3 paragraph(s)" in describe

    def test_a_gap_is_not_counted_as_a_switch(self) -> None:
        """A salutation line between two paragraphs on the same topic must
        not read as the speech changing subject twice."""
        rows = speech("1950_truman_sotu.txt", [1, None, 1, 1])
        describe = ok(flow_frame(rows)).marks[0].evidence.describe
        assert "changes topic 0 time(s)" in describe
        assert "longest run is 3 paragraph(s)" in describe

    def test_ordering_by_switches_puts_the_most_fragmented_first(self) -> None:
        # BIDEN: 1 2 1 2 -> three switches; FDR: two.
        panel = ok(flow_frame(FDR, BIDEN), {"order": "switches"})
        first = next(m for m in panel.marks if m.y == 0.0)
        assert first.label.startswith("2024")


class TestSegmentationIsStated:
    def test_the_rule_is_reported_even_when_nothing_went_wrong(self) -> None:
        notice = find(build(flow_frame(FDR, BIDEN)), "PANEL_SEGMENTATION")
        assert "2 one paragraph per line" in notice.message
        assert notice.context["rules"] == {"line": 2}

    def test_an_unaligned_speech_is_named(self) -> None:
        odd = speech("1999_x_sotu.txt", [0, 1], rule="sentences", aligned=False)
        notice = find(build(flow_frame(FDR, odd)), "PANEL_SEGMENTATION")
        assert "1999_x_sotu.txt" in notice.message
        assert "sentence windows" in notice.message


class TestEvidenceResolves:
    def test_each_mark_selects_exactly_its_paragraph_in_the_published_table(self) -> None:
        panel = ok(flow_frame(FDR, BIDEN))
        for mark in panel.marks:
            selected = panel.data
            for column, value in mark.evidence.filters:
                assert column in panel.data.columns
                selected = selected[selected[column].astype(str) == value]
            assert len(selected) == 1, f"{mark.key}: {len(selected)} rows"

    def test_the_segment_filter_is_load_bearing(self) -> None:
        """Control: every speech has several paragraphs, so a Document filter
        alone would select more than one row."""
        panel = ok(flow_frame(FDR))
        assert (panel.data["Document"] == "1934-01-03_fdr_sotu.txt").sum() > 1

    def test_the_hover_names_the_paragraph_and_its_topic_words(self) -> None:
        mark = ok(flow_frame(FDR)).marks[0]
        assert "paragraph 1 of 6" in mark.evidence.describe
        assert "war, army, battle" in mark.evidence.describe


class TestLimits:
    def test_too_many_speeches_are_capped_and_said_so(self) -> None:
        many = [speech(f"{1900 + i}_x_sotu.txt", [0, 1]) for i in range(5)]
        result = build(flow_frame(*many), {"documents": 3})
        assert find(result, "PANEL_DOCUMENTS_CAPPED").context == {"drawn": 3, "available": 5}
        assert {m.y for m in result.unwrap().marks} == {0.0, 1.0, 2.0}

    def test_a_table_with_nothing_to_score_fails_with_advice(self) -> None:
        rows = speech("1950_truman_sotu.txt", [None, None])
        result = build(flow_frame(rows))
        assert not result.ok
        assert "min-tokens" in find(result, "PANEL_NO_DATA").message

    def test_the_notes_warn_against_reading_switches_as_quality(self) -> None:
        notes = " ".join(LDA_FLOW.notes).lower()
        assert "not incoherence" in notes
        assert "noisier" in notes
