"""Per-document measures: trends, groups, length checks, all measures at once.

One factory serves readability, lexical diversity, corpus and text
statistics, sentence complexity, verb analysis, nominalization and style, so
these tests pin the shared rules on the shared factory and then the few
places a tool differs (sentence rows aggregated to documents; verb counts
turned into rates). Column names are the real tables' (see the audit of the
87-speech corpus); values are ones the engines produce.

What they pin:

* a trend draws every document as an unjoined point plus a rolling median,
  never a line through single speeches, and refuses under three dated
  documents;
* evidence selects exactly the document behind a mark, in the published data;
* groups are chronological for decades and years, by first appearance for
  speakers;
* sentence-level tables become one value per document, with the sentence
  count kept;
* a length-sensitive measure is offered its length check.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from core.result import Result
from core.viz.panels_document_measures import DOCUMENT_MEASURES_PANELS
from core.viz.panelspec import PanelDefinition, PreparedPanel, Provenance

BY_NAME = {panel.name: panel for panel in DOCUMENT_MEASURES_PANELS}

SPEECHES = [
    ("1", "1934-01-03_franklin d roosevelt_sotu.txt", "1934-01-03"),
    ("2", "1945-01-06_franklin d roosevelt_sotu.txt", "1945-01-06"),
    ("3", "1953-01-07_harry s truman_sotu.txt", "1953-01-07"),
    ("4", "1953-02-02_dwight d eisenhower_sotu.txt", "1953-02-02"),
    ("5", "1962-01-11_john f kennedy_sotu.txt", "1962-01-11"),
    ("6", "1985-02-06_ronald reagan_sotu.txt", "1985-02-06"),
    ("7", "2024-03-07_joseph r biden_sotu.txt", "2024-03-07"),
]


def readability_frame() -> pd.DataFrame:
    """readability.csv's columns; Flesch rising over time as in the real run."""
    flesch = [50.1, 52.3, 49.8, 55.0, 58.2, 61.4, 66.0]
    words = [3400, 8200, 9800, 7100, 6500, 4900, 7600]
    return pd.DataFrame(
        [
            {
                "Document ID": doc_id,
                "Document": name,
                "Date": date,
                "Year": int(date[:4]),
                "Sentences": words[i] // 25,
                "Words": words[i],
                "Syllables": int(words[i] * 1.5),
                "Polysyllabic Words": words[i] // 9,
                "Characters": words[i] * 5,
                "Flesch Reading Ease": flesch[i],
                "Flesch-Kincaid Grade": round(20 - flesch[i] / 5, 2),
                "Gunning Fog Index": round(22 - flesch[i] / 5, 2),
                "Coleman-Liau Index": round(19 - flesch[i] / 6, 2),
                "Automated Readability Index": round(21 - flesch[i] / 5, 2),
                "SMOG Index": round(18 - flesch[i] / 7, 2),
                "Interpretation": "Fairly difficult",
            }
            for i, (doc_id, name, date) in enumerate(SPEECHES)
        ]
    )


def sentence_frame() -> pd.DataFrame:
    """sentence_complexity.csv: one row per sentence, three per speech."""
    rows = []
    for doc_id, name, date in SPEECHES:
        for sentence, (distance, tokens) in enumerate([(2.5, 18), (3.1, 25), (2.8, 22)], start=1):
            rows.append(
                {
                    "Sentence ID": sentence,
                    "Document ID": doc_id,
                    "Document": name,
                    "Date": date,
                    "Year": int(date[:4]),
                    "Sentence": "We shall act.",
                    "Tokens": tokens,
                    "Mean Dependency Distance": distance,
                    "Max Dependency Distance": distance * 3,
                    "Depth": 4,
                    "Subordinate Clauses": 1,
                }
            )
    return pd.DataFrame(rows)


def build(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    definition: PanelDefinition = BY_NAME[name]
    return definition.build(
        frame, definition.defaults() | (params or {}), Provenance(tool=definition.tool, panel=name, source="t.csv")
    )


def ok(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> PreparedPanel:
    result = build(name, frame, params)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def rows_behind(panel: PreparedPanel, filters: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    selected = panel.data
    for column, value in filters:
        selected = selected[selected[column].astype(str) == value]
    return selected


class TestEveryToolIsServed:
    @pytest.mark.parametrize(
        "tool",
        [
            "readability",
            "lexical_diversity",
            "corpus_statistics",
            "text_statistics",
            "sentence_complexity",
            "verb_analysis",
            "nominalization",
            "style",
        ],
    )
    def test_the_tool_has_figures_that_ask_questions(self, tool: str) -> None:
        panels = [panel for panel in DOCUMENT_MEASURES_PANELS if panel.tool == tool]
        assert panels
        assert all(panel.question and panel.notes for panel in panels)


class TestTrend:
    def test_documents_are_points_and_the_median_is_the_line(self) -> None:
        panel = ok("readability_over_time", readability_frame(), {"window": 3})
        assert panel.points_only == ("Documents",)
        documents = [m for m in panel.marks if m.group == "Documents"]
        median = [m for m in panel.marks if m.group != "Documents"]
        assert len(documents) == 7
        assert len(median) == 7, "the line reaches both ends, each point a median of 3 documents"

    def test_the_median_line_is_not_broken_by_ordinary_yearly_spacing(self) -> None:
        panel = ok("readability_over_time", readability_frame(), {"window": 3})
        assert panel.line_gap > 1

    def test_x_is_the_date_as_a_position_in_the_year(self) -> None:
        panel = ok("readability_over_time", readability_frame(), {"window": 3})
        first = next(m for m in panel.marks if m.label == "1934 Roosevelt")
        assert first.x == pytest.approx(1934.005, abs=0.001)

    def test_evidence_selects_exactly_the_document(self) -> None:
        panel = ok("readability_over_time", readability_frame(), {"window": 3})
        for mark in (m for m in panel.marks if m.group == "Documents"):
            assert len(rows_behind(panel, mark.evidence.filters)) == 1

    def test_the_filter_is_load_bearing(self) -> None:
        panel = ok("readability_over_time", readability_frame(), {"window": 3})
        assert len(panel.data) > 1

    def test_the_hover_names_the_document_value_and_length(self) -> None:
        panel = ok("readability_over_time", readability_frame(), {"window": 3})
        mark = next(m for m in panel.marks if m.label == "2024 Biden")
        assert "66" in mark.evidence.describe
        assert "7,600" in mark.evidence.describe or "7600" in mark.evidence.describe

    def test_too_few_dated_documents_is_refused_with_a_reason(self) -> None:
        frame = readability_frame().head(2)
        result = build("readability_over_time", frame)
        assert not result.ok
        assert any(d.code == "PANEL_TOO_FEW_DATED" for d in result.diagnostics)

    def test_undated_documents_are_counted_not_hidden(self) -> None:
        frame = readability_frame()
        frame.loc[0, "Date"] = None
        result = build("readability_over_time", frame, {"window": 3})
        assert any(d.code == "PANEL_UNDATED_DROPPED" for d in result.diagnostics)

    def test_an_even_window_is_refused(self) -> None:
        assert not build("readability_over_time", readability_frame(), {"window": 4}).ok


class TestGroups:
    def test_decades_run_in_order(self) -> None:
        panel = ok("readability_by_group", readability_frame())
        assert panel.y_categories == ("1930s", "1940s", "1950s", "1960s", "1980s", "2020s")

    def test_speakers_come_from_the_file_names(self) -> None:
        panel = ok("readability_by_group", readability_frame(), {"group-by": "speaker"})
        assert panel.y_categories[0] == "Franklin D Roosevelt"
        assert "Joseph R Biden" in panel.y_categories

    def test_every_document_is_a_point_in_its_row(self) -> None:
        panel = ok("readability_by_group", readability_frame())
        assert len(panel.marks) == 7
        fifties = panel.y_categories.index("1950s")
        assert sum(1 for m in panel.marks if m.y == fifties) == 2, "Truman and Eisenhower, 1953"


class TestSentenceTables:
    def test_sentences_become_one_median_per_document(self) -> None:
        panel = ok("sentence_complexity_over_time", sentence_frame(), {"window": 3})
        documents = [m for m in panel.marks if m.group == "Documents"]
        assert len(documents) == 7
        assert {m.y for m in documents} == {2.8}, "median of 2.5, 3.1, 2.8"
        assert len(panel.data) == 7, "the published data is the per-document aggregate"

    def test_the_length_check_uses_sentence_length(self) -> None:
        panel = ok("sentence_complexity_length_check", sentence_frame())
        assert "sentence" in panel.x_label.lower()


class TestLengthCheckAndAllMeasures:
    def test_a_length_sensitive_measure_has_its_check(self) -> None:
        assert "lexical_diversity_length_check" in BY_NAME
        assert "corpus_statistics_length_check" in BY_NAME

    def test_all_measures_gives_each_its_own_facet(self) -> None:
        panel = ok("readability_all_measures", readability_frame(), {"window": 3})
        assert panel.shape == "small_multiples"
        assert "Flesch Reading Ease" in panel.facets
        assert "Flesch-Kincaid Grade" in panel.facets
        assert {m.facet for m in panel.marks} == set(panel.facets)
