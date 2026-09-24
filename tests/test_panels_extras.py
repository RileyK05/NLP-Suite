"""SVO triples and quote speakers: the rankings, and what they leave out.

Columns are the real tables' (svo.csv, quotes.csv, quotes_summary.csv from
the 87-speech run). What these pin: triples are counted case-insensitively
with how many speeches each comes from; the extractor's inferred subjects are
out by default; quote speakers are one speaker whatever their case, and a
pronoun "speaker" (the real top three were he, I, He) is hidden and counted;
and the quote figures never divide by quoted words as if they were speech
length.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.result import Result
from core.viz.panels_extras import EXTRAS_PANELS
from core.viz.panelspec import PanelDefinition, PreparedPanel, Provenance

BY_NAME = {panel.name: panel for panel in EXTRAS_PANELS}


def build(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    definition: PanelDefinition = BY_NAME[name]
    return definition.build(
        frame, definition.defaults() | (params or {}), Provenance(tool=definition.tool, panel=name, source="t.csv")
    )


def svo() -> pd.DataFrame:
    rows = [
        ("I", "ask", "Congress", "1"),
        ("i", "ask", "congress", "2"),
        ("I", "ask", "Congress", "2"),
        ("Inferred_Subject_Passive", "selected", "who", "1"),
        ("We", "do", "it", "3"),
    ]
    return pd.DataFrame(
        [
            {"Subject": s, "Verb": v, "Object": o, "Sentence ID": 1, "Document ID": d, "Document": f"{d}.txt"}
            for s, v, o, d in rows
        ]
    )


class TestTriples:
    def test_triples_are_counted_whatever_their_case_with_speech_coverage(self) -> None:
        panel = build("clause_svo_top_triples", svo()).unwrap()
        top = panel.marks[0]
        assert top.label == "i / ask / congress"
        assert top.x == 3.0
        assert "in 2 of 3 speeches" in top.evidence.describe

    def test_inferred_subjects_are_out_by_default_and_said_so(self) -> None:
        result = build("clause_svo_top_triples", svo())
        assert not any("inferred" in m.label for m in result.unwrap().marks)
        assert any(d.code == "PANEL_TRIPLES_FILTERED" for d in result.diagnostics)

    def test_evidence_resolves_to_the_triple(self) -> None:
        panel = build("clause_svo_top_triples", svo()).unwrap()
        for mark in panel.marks:
            ((column, value),) = mark.evidence.filters
            assert (panel.data[column] == value).sum() == 1


class TestSpeakers:
    def quotes(self) -> pd.DataFrame:
        speakers = ["he", "He", "I", "Roosevelt", "roosevelt", "Lincoln", "none"]
        return pd.DataFrame(
            [
                {"Document ID": str(i % 3), "Quote ID": i, "Quote": "q", "Speaker": s, "Cue": "said", "Sentence ID": i}
                for i, s in enumerate(speakers)
            ]
        )

    def test_one_speaker_whatever_the_case_and_pronouns_hidden(self) -> None:
        result = build("quote_annotator_speakers", self.quotes())
        labels = [m.label for m in result.unwrap().marks]
        assert labels == ["Roosevelt", "Lincoln"]
        notice = next(d for d in result.diagnostics if d.code == "PANEL_UNATTRIBUTED")
        assert "3 are attributed to a pronoun" in notice.message


def test_no_quote_figure_divides_by_quoted_words() -> None:
    for name, definition in BY_NAME.items():
        if name.startswith("quote_annotator_rate"):
            measure = next((p for p in definition.params if p.name == "measure"), None)
            if measure is not None:
                assert all("per 1,000" not in choice for choice in measure.choices)
