"""Sentiment and style trends drawn with spread, through the shared factory.

The annual-mean sentiment lines were an average with no spread, joined
through single speeches; these views replace them in the registry. What
these pin: every speech is a point and the line is a rolling median; the
neural sentence counts become shares of the speech; the writing-style
guesser says what it is not; and the annual means are out of the registry.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.result import Result
from core.viz.panels import PANELS
from core.viz.panels_sentiment import ANNUAL_SENTIMENT_PANELS
from core.viz.panels_sentiment_views import SENTIMENT_VIEWS_PANELS
from core.viz.panelspec import PanelDefinition, PreparedPanel, Provenance

BY_NAME = {panel.name: panel for panel in SENTIMENT_VIEWS_PANELS}
YEARS = [1934, 1945, 1953, 1962, 1985, 2001, 2024]


def build(name: str, frame: pd.DataFrame, params: dict[str, Any] | None = None) -> Result[PreparedPanel]:
    definition: PanelDefinition = BY_NAME[name]
    return definition.build(
        frame, definition.defaults() | (params or {}), Provenance(tool=definition.tool, panel=name, source="t.csv")
    )


def documents(extra: dict[str, list[Any]]) -> pd.DataFrame:
    base = {
        "Document ID": [str(i) for i in range(1, len(YEARS) + 1)],
        "Document": [f"{year}-01-05_x y_sotu.txt" for year in YEARS],
        "Date": [f"{year}-01-05" for year in YEARS],
        "Year": YEARS,
    }
    return pd.DataFrame(base | extra)


class TestVader:
    def frame(self) -> pd.DataFrame:
        return documents(
            {
                "Sentences": [73, 139, 159, 120, 200, 180, 240],
                "Neg": [0.06, 0.05, 0.07, 0.05, 0.06, 0.08, 0.09],
                "Neu": [0.82, 0.82, 0.80, 0.81, 0.80, 0.80, 0.79],
                "Pos": [0.12, 0.13, 0.13, 0.14, 0.14, 0.12, 0.12],
                "Compound": [0.25, 0.27, 0.23, 0.26, 0.24, 0.18, 0.14],
            }
        )

    def test_every_speech_is_a_point_and_the_line_is_a_rolling_median(self) -> None:
        panel = build("sentiment_vader_tone_over_time", self.frame(), {"window": 3}).unwrap()
        assert panel.points_only == ("Documents",)
        assert sum(1 for m in panel.marks if m.group == "Documents") == 7


class TestNeural:
    def test_sentence_counts_become_shares_of_the_speech(self) -> None:
        frame = documents(
            {
                "Sentences": [100, 200, 100, 100, 100, 100, 100],
                "Mean Compound": [0.2] * 7,
                "Positive": [50, 50, 50, 50, 50, 50, 50],
                "Negative": [10] * 7,
                "Neutral": [40, 140, 40, 40, 40, 40, 40],
            }
        )
        panel = build("sentiment_neural_bert_tone_by_group", frame, {"measure": "% Positive"}).unwrap()
        by_document = {m.evidence.filters[0][1]: m.x for m in panel.marks}
        assert by_document["2"] == 25.0, "50 of 200 sentences is 25%, not 50"
        assert by_document["1"] == 50.0


class TestWritingStyle:
    def test_the_notes_say_it_is_not_about_the_person(self) -> None:
        notes = " ".join(BY_NAME["gender_guess_style_by_group"].notes)
        assert "WRITING STYLE" in notes and "not a person" in notes


def test_the_annual_means_are_out_of_the_registry() -> None:
    registered = {panel.name for panel in PANELS}
    assert not {panel.name for panel in ANNUAL_SENTIMENT_PANELS} & registered
    assert "sentiment_vader_tone_over_time" in registered
