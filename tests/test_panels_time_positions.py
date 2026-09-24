"""Time and position figures: arcs through a speech, entities and dates over time.

Fixtures use the real tables' columns (see the audit of the 87-speech
corpus). What these pin, most of it found by drawing every figure from the
real tables:

* the relative-position binning works on a real document (it crashed on
  every document with more than one sentence: an Index has no ``clip``);
* labels go to things worth a look: no function words among "concentrated"
  words, no bare numbers among "dates mentioned";
* an entity named in nearly every speech ("Mr. Speaker") is not a default
  timeline -- its line is flat -- and places are the default tag;
* every hit of a concordance is a tick in its document's row.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.result import Result
from core.viz.panels_time_positions import TIME_POSITIONS_PANELS
from core.viz.panelspec import PanelDefinition, PreparedPanel, Provenance

BY_NAME = {panel.name: panel for panel in TIME_POSITIONS_PANELS}
YEARS = [1934, 1945, 1953, 1962, 1985, 2001, 2024]


def build(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    definition: PanelDefinition = BY_NAME[name]
    return definition.build(
        frame, definition.defaults() | (params or {}), Provenance(tool=definition.tool, panel=name, source="t.csv")
    )


def ok(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> PreparedPanel:
    result = build(name, frame, params)
    assert result.ok, [str(d) for d in result.diagnostics]
    return result.unwrap()


def emotion_arc() -> pd.DataFrame:
    """emotion_arc.csv: one row per sentence; tone falls then recovers."""
    rows = []
    for doc, year in enumerate(YEARS[:3], start=1):
        for sentence in range(1, 21):
            rows.append(
                {
                    "Document ID": str(doc),
                    "Date": f"{year}-01-05",
                    "Year": year,
                    "Sentence ID": sentence,
                    "Compound": round(0.5 - abs(sentence - 10) * -0.03 - 0.4, 3),
                    "Sentence": "We shall act.",
                }
            )
    return pd.DataFrame(rows)


class TestArcs:
    def test_a_real_multi_sentence_document_is_binned(self) -> None:
        panel = ok("narrative_emotion_arc", emotion_arc(), {"bins": 10})
        groups = {m.group for m in panel.marks}
        assert len(groups) == 4, "three documents and the median across all of them"
        per_document = [m for m in panel.marks if m.group == "1934-01-05 (doc 1)"]
        assert len(per_document) == 10, "twenty sentences into ten bins, none empty"
        assert all(0.0 < m.x < 1.0 for m in per_document), "bins are placed at their centres"


class TestDispersion:
    def frame(self) -> pd.DataFrame:
        words = [
            ("the", 9000, 0.05),
            ("he", 800, 0.62),
            ("energy", 1000, 0.71),
            ("war", 1200, 0.44),
            ("year", 900, 0.1),
        ]
        return pd.DataFrame([{"Term": w, "Frequency": f, "Range": 50, "Gries DP": dp} for w, f, dp in words])

    def test_function_words_are_never_labelled_as_concentrated(self) -> None:
        panel = ok("dispersion_frequency_vs_concentration", self.frame(), {"label-top": 3})
        labelled = {m.label for m in panel.marks if m.labelled}
        assert "he" not in labelled
        assert "energy" in labelled


class TestNerTimeline:
    def frame(self) -> pd.DataFrame:
        rows = []
        for doc, year in enumerate(YEARS, start=1):
            mentions = [("Speaker", "PERSON", 2), ("America", "GPE", 5)]
            if year < 1990:
                mentions.append(("the Soviet Union", "GPE", 3))
            if year > 1990:
                mentions.append(("China", "GPE", 4))
            for entity, tag, count in mentions:
                rows.append(
                    {
                        "Entity": entity,
                        "NER Tag": tag,
                        "Count": count,
                        "First Sentence": 1,
                        "Last Sentence": 9,
                        "Document ID": str(doc),
                        "Document": f"{year}-01-05_x y_sotu.txt",
                        "Date": f"{year}-01-05",
                        "Year": year,
                    }
                )
        return pd.DataFrame(rows)

    def test_places_are_the_default_and_ubiquitous_names_are_not_chosen(self) -> None:
        panel = ok("ner_entity_timeline", self.frame())
        assert set(panel.groups) == {"the Soviet Union", "China"}, "America is in every speech: a flat line"

    def test_a_named_entity_is_still_drawn_on_request(self) -> None:
        panel = ok("ner_entity_timeline", self.frame(), {"entities": "America"})
        assert panel.groups == ("America",)


class TestDates:
    def frame(self) -> pd.DataFrame:
        mentions = [
            ("1", 1962, "1776", "year"),
            ("2", 1985, "1776", "year"),
            ("3", 2001, "1500", "year"),
            ("4", 2024, "September 17th, 1787", "date"),
        ]
        return pd.DataFrame(
            [
                {
                    "Document": f"{year}-01-05_x y_sotu.txt",
                    "Date": f"{year}-01-05",
                    "Year": year,
                    "Document ID": doc,
                    "Date ID": str(index),
                    "Surface": surface,
                    "Normalized": "1787-09-17" if kind == "date" else surface,
                    "Type": kind,
                }
                for index, (doc, year, surface, kind) in enumerate(mentions, start=1)
            ]
        )

    def test_a_bare_number_seen_once_is_not_labelled_as_a_date(self) -> None:
        panel = ok("date_annotator_timeline", self.frame(), {"label-top": 5})
        labelled = {m.label for m in panel.marks if m.labelled}
        assert "1500" not in labelled, "a number read as a year, in one speech"
        assert "1776" in labelled, "a year that recurs across speeches is an anchor"


class TestKwic:
    def test_every_hit_is_a_tick_in_its_documents_row(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "Document ID": str(doc),
                    "Document": f"{year}-01-05_x y_sotu.txt",
                    "Date": f"{year}-01-05",
                    "Year": year,
                    "Sentence ID": sentence,
                    "Left Context": "the cause of",
                    "Hit": "freedom",
                    "Right Context": "is ours",
                }
                for doc, year in enumerate(YEARS[:3], start=1)
                for sentence in (3, 40)
            ]
        )
        panel = ok("kwic_hit_positions", frame)
        assert panel.shape == "positions"
        assert len(panel.marks) == 6
        assert len(panel.y_categories) == 3
