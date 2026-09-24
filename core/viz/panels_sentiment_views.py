"""Sentiment and style scores per document: trends with spread, and groups.

The ranked panels in ``panels_sentiment.py`` answer "which speeches score
highest"; what they lacked was the question a corpus of ninety years exists
for -- has the tone changed? -- asked without the two faults of the annual
mean they offered: a line through single speeches, and an average with no
spread. These views reuse the per-document measure factory
(``panels_document_measures.measure_panels``), so a sentiment trend is drawn
exactly as a readability trend is: every speech a point, a rolling median
through them, and groups by decade or speaker as boxes with every point.

Tools: VADER (and ANEW where its lexicon is installed), the four neural
backends, SentiWordNet with the hedonometer, and the writing-style gender
guesser -- whose notes say plainly what it is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.result import Result
from core.viz.panels_document_measures import MeasureTool, measure_panels, prepare_passthrough

__all__ = ["SENTIMENT_VIEWS_PANELS"]

DOC_ID = "Document ID"
DOC = "Document"
DATE = "Date"

_VADER_NOTES = (
    "VADER's compound score runs from -1 to +1 and is a rule-based reading of the words, not a human rating. "
    "Formal speeches score mildly positive almost everywhere, so a change of 0.05 is already large here.",
)
_NEURAL_NOTES = (
    "A neural model's score is only comparable with the same model's other scores; do not set one model's "
    "line beside another's as though they shared a unit.",
    "Positive, negative and neutral are shares of the speech's sentences, so a long speech does not weigh "
    "more than a short one.",
)


def _neural_prepare(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """Sentence counts per label become shares of the speech's sentences."""
    working = frame.copy()
    sentences = pd.to_numeric(working["Sentences"], errors="coerce")
    for label in ("Positive", "Negative", "Neutral"):
        counts = pd.to_numeric(working[label], errors="coerce")
        working[f"% {label}"] = np.where(sentences > 0, 100.0 * counts / sentences, np.nan)
    return Result.success(working)


def _vader() -> MeasureTool:
    return MeasureTool(
        tool="sentiment_vader_anew",
        requires=(DOC_ID, DOC, DATE, "Sentences", "Compound", "Pos", "Neg", "Neu"),
        prepare=prepare_passthrough((DOC_ID, DOC, "Compound", "Sentences")),
        measures=("Compound", "Pos", "Neg", "Neu"),
        default_measure="Compound",
        size_column="Sentences",
        size_unit="sentences",
        facet_measures=("Compound", "Pos", "Neg"),
        notes_by_measure={
            "Pos": "Pos, Neg and Neu are the average shares of positive, negative and neutral words per sentence.",
        },
        general_notes=_VADER_NOTES,
    )


def _anew() -> MeasureTool:
    return MeasureTool(
        tool="sentiment_vader_anew",
        requires=(DOC_ID, DOC, DATE, "Hits", "Valence", "Arousal", "Dominance"),
        prepare=prepare_passthrough((DOC_ID, DOC, "Valence", "Hits")),
        measures=("Valence", "Arousal", "Dominance"),
        default_measure="Valence",
        size_column="Hits",
        size_unit="rated words",
        facet_measures=("Valence", "Arousal", "Dominance"),
        general_notes=(
            "ANEW rates words from 1 to 9 (5 is neutral); a document's score is the mean over the words the "
            "lexicon knows, so a speech with few rated words has a noisy score -- the hover gives the count.",
        ),
    )


def _neural(tool: str) -> MeasureTool:
    return MeasureTool(
        tool=tool,
        requires=(DOC_ID, DOC, DATE, "Sentences", "Mean Compound", "Positive", "Negative", "Neutral"),
        prepare=_neural_prepare,
        measures=("Mean Compound", "% Positive", "% Negative", "% Neutral"),
        default_measure="Mean Compound",
        size_column="Sentences",
        size_unit="sentences",
        facet_measures=("Mean Compound", "% Positive", "% Negative"),
        general_notes=_NEURAL_NOTES,
    )


def _swn() -> MeasureTool:
    return MeasureTool(
        tool="sentiment_swn_hedono",
        requires=(DOC_ID, DOC, DATE, "Tokens", "Hits", "Pos", "Neg", "Obj", "Net"),
        prepare=prepare_passthrough((DOC_ID, DOC, "Net", "Tokens")),
        measures=("Net", "Pos", "Neg", "Obj"),
        default_measure="Net",
        size_column="Tokens",
        size_unit="tokens",
        facet_measures=("Net", "Pos", "Neg"),
        general_notes=(
            "SentiWordNet scores each word's first sense; Net is positive minus negative, averaged over the words "
            "it knows. Most words are rated mostly objective, so Net sits near zero and small shifts matter.",
        ),
    )


def _hedonometer() -> MeasureTool:
    return MeasureTool(
        tool="sentiment_swn_hedono",
        requires=(DOC_ID, DOC, DATE, "Tokens", "Hits", "Happiness"),
        prepare=prepare_passthrough((DOC_ID, DOC, "Happiness", "Tokens")),
        measures=("Happiness",),
        default_measure="Happiness",
        size_column="Hits",
        size_unit="rated words",
        general_notes=(
            "The hedonometer rates words from 1 to 9 for happiness; a document's score is the mean of its rated "
            "words, so it moves slowly -- a change of a tenth of a point is large for this measure.",
        ),
    )


def _gender_guess() -> MeasureTool:
    return MeasureTool(
        tool="gender_guess",
        requires=(DOC_ID, DOC, DATE, "Words", "Masculine", "Feminine", "Label"),
        prepare=prepare_passthrough((DOC_ID, DOC, "Masculine", "Feminine", "Words")),
        measures=("Masculine", "Feminine"),
        default_measure="Feminine",
        size_column="Words",
        size_unit="words",
        general_notes=(
            "This classifies WRITING STYLE, not a person: it scores function words (prepositions, articles, "
            "pronouns) on a formal-to-involved axis coded male/female after Koppel et al. (2002). It says nothing "
            "about the speaker's gender.",
            "A State of the Union is drafted by many hands, so a score describes the speechwriting office's "
            "register as much as the president's. A formal genre scores 'female' throughout for that reason.",
        ),
    )


SENTIMENT_VIEWS_PANELS = (
    *measure_panels(_vader(), title_word="VADER tone", prefix="sentiment_vader_tone"),
    *measure_panels(_anew(), title_word="ANEW ratings", prefix="sentiment_anew"),
    *measure_panels(_neural("sentiment_neural_bert"), title_word="BERT tone", prefix="sentiment_neural_bert_tone"),
    *measure_panels(_neural("sentiment_neural_spacy"), title_word="spaCy tone", prefix="sentiment_neural_spacy_tone"),
    *measure_panels(
        _neural("sentiment_neural_stanza"), title_word="Stanza tone", prefix="sentiment_neural_stanza_tone"
    ),
    *measure_panels(
        _neural("sentiment_neural_corenlp"), title_word="CoreNLP tone", prefix="sentiment_neural_corenlp_tone"
    ),
    *measure_panels(_swn(), title_word="SentiWordNet tone", prefix="sentiment_swn"),
    *measure_panels(_hedonometer(), title_word="Hedonometer happiness", prefix="sentiment_hedonometer"),
    *measure_panels(_gender_guess(), title_word="Writing-style score", prefix="gender_guess_style"),
)
