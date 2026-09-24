"""Sentiment panels use the score tables written by the public tools."""

from __future__ import annotations

import pandas as pd
import pytest

from core.viz.panels_sentiment import (
    COMPOUND,
    DOC,
    DOC_ID,
    MEAN_COMPOUND,
    SENT_ID,
    SENTENCE,
    SENTIMENT_BERT_OVER_TIME,
    SENTIMENT_PANELS,
    VALENCE,
)
from core.viz.panelspec import Provenance


def result_frame(definition) -> pd.DataFrame:
    """Construct rows following a definition's declared public output schema."""
    rows = []
    # VADER and neural document files use different published score names.
    if definition.name == "sentiment_anew_valence":
        score_column = VALENCE
    elif ("neural_" in definition.name and definition.name.endswith("_documents")) or (
        definition.shape == "line_series" and "neural_" in definition.name
    ):
        score_column = MEAN_COMPOUND
    else:
        score_column = COMPOUND
    for index, score in enumerate((0.2, 0.85), start=1):
        row = {
            DOC_ID: f"doc-{index}",
            DOC: "Repeated title",
            SENT_ID: "1",
            SENTENCE: f"sentence {index}",
            "Label": "positive",
            "Score": 0.85,
            "Compound": score,
            "Mean Compound": score,
            "Valence": 3.0 if index == 1 else 8.0,
            "Hits": 2,
            "Sentences": 1,
            "Positive": 1,
            "Negative": 0,
            "Neutral": 0,
            "Year": 2024,
        }
        # Restrict to the declared columns just like the artifact, keeping
        # values from the per-tool result schema above.
        rows.append({column: row.get(column, f"{column}-{index}") for column in definition.requires})
        rows[-1][score_column] = score if score_column != VALENCE else row[VALENCE]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("definition", SENTIMENT_PANELS, ids=lambda definition: definition.name)
def test_panel_marks_rank_and_map_back_to_one_source_row(definition) -> None:
    frame = result_frame(definition)
    result = definition.build(frame, definition.defaults(), Provenance(tool=definition.tool, panel=definition.name))
    prepared = result.unwrap()

    if definition.shape == "line_series":
        assert prepared.shape == "line_series"
        assert len(prepared.marks) == 1
        mark = prepared.marks[0]
        matches = frame[frame["Year"].astype(str) == mark.evidence.filters[0][1]]
        score_column = COMPOUND if definition.name == "sentiment_vader_anew_over_time" else MEAN_COMPOUND
        assert mark.x == 2024
        assert mark.y == pytest.approx(matches[score_column].astype(float).mean())
        assert mark.evidence.count == len(matches) == 2
        assert "one document" not in mark.evidence.describe
        return

    assert prepared.shape == "ranked_bars"
    assert len(prepared.marks) == 2
    assert len({mark.label for mark in prepared.marks}) == 2
    assert "(neutral)" in prepared.subtitle
    assert not any(mark.label.startswith("doc-") and "sentence" in mark.label for mark in prepared.marks)
    for mark in prepared.marks:
        matches = frame
        for column, value in mark.evidence.filters:
            matches = matches[matches[column].astype(str) == value]
        assert mark.evidence.scope == "rows"
        assert len(matches) == mark.evidence.count == 1
        assert mark.x == pytest.approx(
            float(
                matches.iloc[0][
                    VALENCE
                    if definition.name == "sentiment_anew_valence"
                    else MEAN_COMPOUND
                    if "neural_" in definition.name and definition.name.endswith("_documents")
                    else COMPOUND
                ]
            )
        )


def test_backend_notes_explain_score_meaning_and_date_prerequisite() -> None:
    by_name = {definition.name: definition for definition in SENTIMENT_PANELS}
    assert "not a calibrated probability" in " ".join(by_name["sentiment_vader_sentences"].notes)
    assert "-1, 0, or +1" in " ".join(by_name["sentiment_neural_stanza_sentences"].notes)
    assert "Year column attached to dated document rows" in " ".join(by_name["sentiment_neural_bert_documents"].notes)


def test_annual_means_keep_one_document_year_and_report_undated_rows() -> None:
    frame = pd.DataFrame(
        [
            {DOC_ID: "a", "Year": 2020, MEAN_COMPOUND: -0.2},
            {DOC_ID: "b", "Year": 2020, MEAN_COMPOUND: 0.8},
            {DOC_ID: "c", "Year": 2021, MEAN_COMPOUND: 0.4},
            {DOC_ID: "d", "Year": None, MEAN_COMPOUND: 1.0},
        ]
    )
    result = SENTIMENT_BERT_OVER_TIME.build(
        frame,
        {},
        Provenance(tool=SENTIMENT_BERT_OVER_TIME.tool, panel=SENTIMENT_BERT_OVER_TIME.name),
    )
    prepared = result.unwrap()
    by_year = {int(mark.x): mark for mark in prepared.marks}

    assert by_year[2020].y == pytest.approx(0.3)
    assert by_year[2020].evidence.count == 2
    assert by_year[2021].y == pytest.approx(0.4)
    assert by_year[2021].evidence.count == 1
    assert "one document" in by_year[2021].evidence.describe
    assert "PANEL_UNDATED_ROWS" in [diagnostic.code for diagnostic in result.diagnostics]
    assert len(frame[frame["Year"] == int(by_year[2020].x)]) == by_year[2020].evidence.count


def test_anew_zero_valence_without_lexicon_hits_is_not_drawn_as_negative() -> None:
    definition = next(panel for panel in SENTIMENT_PANELS if panel.name == "sentiment_anew_valence")
    frame = result_frame(definition)
    frame.loc[0, "Hits"] = 0
    frame.loc[0, "Valence"] = 0.0  # analysis records this sentinel when no terms matched

    result = definition.build(frame, definition.defaults(), Provenance(tool=definition.tool, panel=definition.name))
    prepared = result.unwrap()

    assert [mark.label for mark in prepared.marks] == ["Repeated title"], "unique once the other is left out"
    assert "PANEL_NO_LEXICON_MATCHES" in [diagnostic.code for diagnostic in result.diagnostics]


def _sentences(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                DOC_ID: str(index),
                DOC: "1934-01-03_franklin d roosevelt_sotu.txt",
                SENT_ID: str(index),
                SENTENCE: f"We have seen the {index} of many things in this long year of ours",
                COMPOUND: score,
                "Label": "x",
            }
            for index, score in enumerate(scores)
        ]
    )


def _vader_sentences():
    return next(panel for panel in SENTIMENT_PANELS if panel.name == "sentiment_vader_sentences")


def test_both_ends_are_shown_on_a_mostly_positive_table() -> None:
    """Regression: ranking by |compound| on the real corpus gave nineteen
    bars at +0.99 and one below zero."""
    definition = _vader_sentences()
    frame = _sentences([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, -0.3, -0.5])
    prepared = definition.build(frame, {"top-n": 4, "direction": "both"}, Provenance(tool="t", panel="p")).unwrap()
    values = [mark.x for mark in sorted(prepared.marks, key=lambda m: m.y)]
    assert values == [0.99, 0.98, -0.3, -0.5], "most positive at the top, most negative at the bottom"
    assert {mark.group for mark in prepared.marks} == {"Above neutral", "Below neutral"}


def test_one_end_only_when_asked() -> None:
    definition = _vader_sentences()
    frame = _sentences([0.9, 0.5, -0.3, -0.5])
    prepared = definition.build(frame, {"top-n": 4, "direction": "negative"}, Provenance(tool="t", panel="p")).unwrap()
    assert sorted(mark.x for mark in prepared.marks) == [-0.5, -0.3]


def test_a_sentence_is_labelled_by_its_speech_and_its_words() -> None:
    definition = _vader_sentences()
    prepared = definition.build(
        _sentences([0.9, -0.2]), definition.defaults(), Provenance(tool="t", panel="p")
    ).unwrap()
    label = prepared.marks[0].label
    assert label.startswith("1934 Roosevelt — “We have seen the")
    assert label.endswith("…”")
