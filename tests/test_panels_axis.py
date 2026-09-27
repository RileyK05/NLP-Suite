"""Every figure about change works along the corpus's axis, not only along time (docs/PLAN_0.5.0.md 1.7).

The class fix for "the suite only works for time series". These are the 61
panels whose requirements named ``Date`` or ``Year`` at ``ab627c4``, before
the axis existed; each is built from its real fixture three ways:

(a) dated, as it always was: the output must not change by a single mark;
(b) as a book: dates removed, each document a chapter in date order. The
    figure must draw, and say chapter, not year or time;
(c) with nothing placing the documents: it draws something that needs no
    axis, or refuses and says to add a date or an order on the Corpus page.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from core.viz.panels import PANELS, prepare_panel
from core.viz.panelspec import Source
from tests.test_figure_quality import example_for, variants

#: The 61 panels that required Date or Year at ab627c4 (listed from PANELS then).
DATED_AT_START = (
    "readability_over_time",
    "readability_by_group",
    "readability_length_check",
    "readability_all_measures",
    "lexical_diversity_over_time",
    "lexical_diversity_by_group",
    "lexical_diversity_length_check",
    "lexical_diversity_all_measures",
    "corpus_statistics_over_time",
    "corpus_statistics_by_group",
    "corpus_statistics_length_check",
    "corpus_statistics_all_measures",
    "text_statistics_over_time",
    "text_statistics_by_group",
    "text_statistics_all_measures",
    "sentence_complexity_over_time",
    "sentence_complexity_by_group",
    "sentence_complexity_length_check",
    "sentence_complexity_all_measures",
    "verb_analysis_over_time",
    "verb_analysis_by_group",
    "verb_analysis_all_measures",
    "nominalization_over_time",
    "nominalization_by_group",
    "sentiment_vader_tone_over_time",
    "sentiment_vader_tone_by_group",
    "sentiment_vader_tone_all_measures",
    "sentiment_anew_over_time",
    "sentiment_anew_by_group",
    "sentiment_anew_all_measures",
    "sentiment_neural_bert_tone_over_time",
    "sentiment_neural_bert_tone_by_group",
    "sentiment_neural_bert_tone_all_measures",
    "sentiment_neural_spacy_tone_over_time",
    "sentiment_neural_spacy_tone_by_group",
    "sentiment_neural_spacy_tone_all_measures",
    "sentiment_neural_stanza_tone_over_time",
    "sentiment_neural_stanza_tone_by_group",
    "sentiment_neural_stanza_tone_all_measures",
    "sentiment_neural_corenlp_tone_over_time",
    "sentiment_neural_corenlp_tone_by_group",
    "sentiment_neural_corenlp_tone_all_measures",
    "sentiment_swn_over_time",
    "sentiment_swn_by_group",
    "sentiment_swn_all_measures",
    "sentiment_hedonometer_over_time",
    "sentiment_hedonometer_by_group",
    "gender_guess_style_over_time",
    "gender_guess_style_by_group",
    "quote_annotator_rate_over_time",
    "quote_annotator_rate_by_group",
    "shape_hc_eras",
    "bert_topics_eras",
    "ngram_frequency_over_time",
    "date_annotator_timeline",
    "gender_annotator_mentions_over_time",
    "narrative_emotion_arc",
    "narrative_character_positions",
    "shapes_story_arc",
    "ner_entity_timeline",
    "kwic_hit_positions",
)
BY_NAME = {panel.name: panel for panel in PANELS}
#: Digests of every dated output at ab627c4, one per panel and parameter choice.
SNAPSHOT = Path(__file__).parent / "fixtures" / "axis" / "dated_outputs.json"
WITH_EXAMPLE = [name for name in DATED_AT_START if example_for(BY_NAME[name]) is not None]
#: Words that say "time" where a book has chapters.
TIME_WORDS = ("year", "over time", "speech date", "decade")


def _prepare(name: str, frame: pd.DataFrame, params: dict[str, Any]) -> Any:
    return prepare_panel(name, frame, params, source=Source(path="fixture.csv", sha256="x"))


def _entry(result: Any) -> dict[str, Any]:
    if result.value is None:
        return {"refused": [d.code for d in result.diagnostics]}
    prepared = result.unwrap()
    return {
        "title": prepared.title,
        "subtitle": prepared.subtitle,
        "x": prepared.x_label,
        "y": prepared.y_label,
        "line_gap": prepared.line_gap,
        "codes": sorted(d.code for d in result.diagnostics),
        "marks": [
            [
                mark.key,
                round(mark.x, 9) if mark.x is not None else None,
                round(mark.y, 9) if mark.y is not None else None,
                mark.group,
                mark.label,
                mark.evidence.describe if mark.evidence else "",
            ]
            for mark in prepared.marks
        ],
    }


def _digest(entry: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def test_the_frozen_list_is_every_panel_that_needed_a_date() -> None:
    assert len(DATED_AT_START) == 61
    gone = [name for name in DATED_AT_START if name not in BY_NAME]
    assert not gone, f"renamed or removed: {gone}; update the list with the reason"


@pytest.mark.parametrize("name", WITH_EXAMPLE)
def test_a_dated_corpus_draws_exactly_what_it_drew_before(name: str) -> None:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    frame = example_for(BY_NAME[name])
    for params in variants(BY_NAME[name]):
        key = f"{name} {json.dumps(params, sort_keys=True)}"
        assert _digest(_entry(_prepare(name, frame, params))) == expected[key], f"{key} changed"


def as_chapters(frame: pd.DataFrame) -> pd.DataFrame:
    """The same table from a book: no dates, each document a chapter, in the order the dates gave."""
    out = frame.copy()
    when = "Date" if "Date" in out.columns else "Year" if "Year" in out.columns else None
    key = "Document ID" if "Document ID" in out.columns else "Document" if "Document" in out.columns else None
    if when is not None and key is not None:
        firsts = out.drop_duplicates(key)[[key, when]].copy()
        firsts["_when"] = firsts[when].astype(str)
        ordered = firsts.sort_values("_when", kind="stable")[key].astype(str)
        chapter = {doc: index for index, doc in enumerate(ordered, 1)}
        out["Position"] = out[key].astype(str).map(chapter).astype(float)
    elif when is not None:
        # A tool's own per-year table (the n-gram viewer): each year becomes a chapter.
        steps = {value: index for index, value in enumerate(sorted(out[when].dropna().unique()), 1)}
        out["Position"] = out[when].map(steps).astype(float)
    else:
        return out
    out["Position label"] = [f"Chapter {int(value)}" if pd.notna(value) else None for value in out["Position"]]
    return out.drop(columns=[column for column in ("Date", "Year") if column in out.columns])


def without_axis(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in ("Date", "Year", "Position", "Position label") if c in frame.columns])


#: Panels whose y axis is calendar years in any corpus, and why that is right.
CALENDAR_Y = {
    "date_annotator_timeline": "y is the year each date written in the text names: a book mentions years too",
}


def _says_time(result: Any) -> list[str]:
    prepared = result.unwrap()
    y_label = "" if prepared.panel in CALENDAR_Y else prepared.y_label
    words = " ".join((prepared.title, prepared.subtitle, prepared.x_label, y_label)).lower()
    return [word for word in TIME_WORDS if word in words]


@pytest.mark.parametrize("name", WITH_EXAMPLE)
def test_a_book_draws_along_its_chapters(name: str) -> None:
    frame = as_chapters(example_for(BY_NAME[name]))
    result = _prepare(name, frame, {})
    assert result.value is not None, [d.message for d in result.diagnostics]
    assert not _says_time(result), f"{name} still says {_says_time(result)} for a book"
    prepared = result.unwrap()
    if prepared.shape in ("line_series", "small_multiples", "positions", "stream") and "Relative" not in (
        prepared.x_label
    ):
        described = f"{prepared.title} {prepared.x_label} {prepared.subtitle} {prepared.y_label}".lower()
        assert "chapter" in described, f"{name}: a figure along the axis should say chapter: {described}"


def test_no_tool_is_left_with_nothing_to_draw_for_a_book() -> None:
    drawn: dict[str, int] = {}
    for name in WITH_EXAMPLE:
        tool = BY_NAME[name].tool
        result = _prepare(name, as_chapters(example_for(BY_NAME[name])), {})
        drawn[tool] = drawn.get(tool, 0) + (result.value is not None)
    empty = sorted(tool for tool, count in drawn.items() if not count)
    assert not empty, f"these tools draw nothing for a book: {empty}"


@pytest.mark.parametrize("name", WITH_EXAMPLE)
def test_with_no_axis_a_figure_draws_without_one_or_says_where_to_add_it(name: str) -> None:
    result = _prepare(name, without_axis(example_for(BY_NAME[name])), {})
    if result.value is not None:
        assert not _says_time(result), f"{name} draws without an axis but still says {_says_time(result)}"
        return
    message = " ".join(d.message for d in result.diagnostics)
    assert "Corpus page" in message and ("date" in message or "order" in message), (
        f"{name} refuses without saying where to add a date or an order: {message}"
    )
