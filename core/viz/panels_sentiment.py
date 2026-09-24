"""Ranked sentiment panels for the public lexicon and neural tools.

These panels show recorded sentence/document scores and annual means where
the profiler attached a year to dated document rows.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels
from core.viz.panelspec import Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

__all__ = [
    "ANNUAL_SENTIMENT_PANELS",
    "SENTIMENT_ANEW_VALENCE",
    "SENTIMENT_BERT_DOCUMENTS",
    "SENTIMENT_BERT_OVER_TIME",
    "SENTIMENT_BERT_SENTENCES",
    "SENTIMENT_CORENLP_DOCUMENTS",
    "SENTIMENT_CORENLP_OVER_TIME",
    "SENTIMENT_CORENLP_SENTENCES",
    "SENTIMENT_PANELS",
    "SENTIMENT_SPACY_DOCUMENTS",
    "SENTIMENT_SPACY_OVER_TIME",
    "SENTIMENT_SPACY_SENTENCES",
    "SENTIMENT_STANZA_DOCUMENTS",
    "SENTIMENT_STANZA_OVER_TIME",
    "SENTIMENT_STANZA_SENTENCES",
    "SENTIMENT_VADER_DOCUMENTS",
    "SENTIMENT_VADER_OVER_TIME",
    "SENTIMENT_VADER_SENTENCES",
    "annual_sentiment_scores",
    "sentiment_scores",
]

DOC_ID = "Document ID"
DOC = "Document"
SENT_ID = "Sentence ID"
SENTENCE = "Sentence"
COMPOUND = "Compound"
MEAN_COMPOUND = "Mean Compound"
VALENCE = "Valence"

_TOP_N = PanelParam(
    name="top-n",
    type="int",
    default=20,
    minimum=1,
    maximum=200,
    label="Rows to show",
    help="How many of the most extreme document or sentence scores to show.",
)
_DIRECTION = PanelParam(
    name="direction",
    type="choice",
    default="both",
    choices=("both", "positive", "negative"),
    label="Which end",
    help=(
        "both: the top half of the rows from each end of the scale; positive or negative: one end only. "
        "Ranking by distance from neutral alone let one end fill every row on a mostly positive corpus."
    ),
)
#: The two ends of a signed score, as the figure's legend names them.
_ABOVE = "Above neutral"
_BELOW = "Below neutral"
#: A sentence label: the speech, then the sentence's opening words.
_SNIPPET_CHARS = 38
_VADER_NOTES = (
    "Compound is VADER's normalized signed score in [-1, +1]; negative is less positive, positive is more positive, and values near zero are neutral-ish.",
    "VADER is a lexicon and rule-based instrument. Its compound score is not a calibrated probability or a human rating.",
    "Annual views require a Year column attached to dated document rows. Undated documents remain in the score table and are omitted from annual means.",
)
_ANEW_NOTES = (
    "ANEW Valence is the mean human-rated pleasantness score of matched lexicon words, on the lexicon's 1-9 scale; the midpoint is around 5.",
    "This is a document-level lexicon average over matched words, not a model confidence or a sentence score.",
    "Annual views require a Year column attached to dated document rows. Undated documents remain in the score table and are omitted from annual means.",
)
_NEURAL_LIMITS = {
    "sentiment_neural_bert": (
        "Compound is the signed winning-class confidence in [-1, +1] from the selected Hugging Face classifier. Confidence is model-specific and not necessarily calibrated.",
        "The model classifies sentence text as a whole; it is not a word-level explanation. Different model checkpoints can produce different scores.",
    ),
    "sentiment_neural_spacy": (
        "Compound is the signed confidence from the loaded spaCy textcat head in [-1, +1]. The pipeline must contain a trained sentiment head; vanilla en_core_web_sm has none.",
        "The score reflects this textcat model and is not calibrated against the other sentiment backends.",
    ),
    "sentiment_neural_stanza": (
        "Stanza emits a hard negative, neutral, or positive class. Compound is therefore only -1, 0, or +1; Score is not a graded confidence.",
        "This three-way output is coarser than confidence-based classifiers and should not be read as equally detailed evidence.",
    ),
    "sentiment_neural_corenlp": (
        "CoreNLP's RNTN five-class sentiment is mapped to Compound values from -1 to +1 in half-step increments.",
        "The score is specific to the CoreNLP model and server version; it is not calibrated against the other sentiment backends.",
    ),
}


def sentiment_scores(  # noqa: PLR0913 - tool-specific score panels share one explicit schema adapter
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    definition: PanelDefinition,
    score_column: str,
    identity_columns: tuple[str, ...],
    label_column: str,
    unit: str,
    center: float = 0.0,
    notes: tuple[str, ...],
) -> Result[PreparedPanel]:
    """Build a ranked horizontal score panel with exact source-row evidence."""
    top_n = int(params["top-n"])
    working = frame.copy()
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    usable = working[score_column].map(_is_finite)
    dropped = int((~usable).sum())
    working = working[usable].copy()
    diagnostics: list[Diagnostic] = []
    if dropped:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{dropped} row(s) had a non-numeric or non-finite {score_column} score and were left out.",
                dropped=dropped,
            )
        )
    if score_column == VALENCE and "Hits" in working.columns:
        no_hits = pd.to_numeric(working["Hits"], errors="coerce").fillna(0) <= 0
        missing_hits = int(no_hits.sum())
        working = working[~no_hits].copy()
        if missing_hits:
            diagnostics.append(
                Diagnostic.info(
                    "PANEL_NO_LEXICON_MATCHES",
                    f"{missing_hits} document(s) had no ANEW word matches and were left out; the table records their valence as 0.",
                    dropped=missing_hits,
                )
            )
    if working.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", f"no rows have a usable {score_column} score"),
            *diagnostics,
        )

    direction = str(params.get("direction", "both"))
    names = document_labels(working[DOC].astype(str)) if DOC in working.columns else {}
    working["_stable_label"] = _unique(
        [_mark_label(row, label_column, identity_columns, names) for _, row in working.iterrows()],
        [" · ".join(str(row[column]) for column in identity_columns) for _, row in working.iterrows()],
    )
    drawn, one_sided = _extremes(working, score_column, center, top_n, direction)
    if one_sided:
        diagnostics.append(Diagnostic.info("PANEL_ONE_SIDED", one_sided))
    marks: list[PanelMark] = []
    for rank, (_, row) in enumerate(drawn.iterrows()):
        label, score = str(row["_stable_label"]), float(row[score_column])
        filters = tuple((column, str(row[column])) for column in identity_columns)
        detail = f"{score_column} {score:g} {unit}"
        if SENTENCE in row.index:
            sentence = str(row[SENTENCE])
            excerpt = sentence if len(sentence) <= 180 else sentence[:177] + "..."
            detail += f"; sentence: {excerpt!r}"
        evidence = Evidence(
            scope="rows",
            filters=filters,
            count=1,
            describe=f"{label}: {detail}; source row {', '.join(f'{column}={row[column]}' for column in identity_columns)}",
        )
        marks.append(
            PanelMark(
                key="|".join(value for _, value in filters),
                label=label,
                # Bars grow from the neutral point, so a valence of 4.2 on a
                # 1-9 scale reads as a short bar below neutral, not a long
                # bar above zero.
                x=score,
                y=float(rank),
                group=_ABOVE if score >= center else _BELOW,
                evidence=evidence,
            )
        )

    ends = {"both": "the most extreme at each end", "positive": "the highest", "negative": "the lowest"}[direction]
    subtitle = f"{len(drawn)} row(s) · {ends} of {score_column}, from {center:g} (neutral) · scale: {unit}"
    if one_sided:
        subtitle += f" · {one_sided}"
    prepared = PreparedPanel(
        panel=definition.name,
        shape="ranked_bars",
        title=definition.title,
        subtitle=subtitle,
        marks=tuple(marks),
        x_label=f"{score_column} ({unit})",
        y_label="",
        provenance=provenance,
        data=drawn.drop(columns=["_stable_label"]).reset_index(drop=True),
        groups=tuple(group for group in (_ABOVE, _BELOW) if any(m.group == group for m in marks)),
        notes=notes,
    )
    return Result.success(prepared, *diagnostics)


def _extremes(
    working: pd.DataFrame, score_column: str, center: float, top_n: int, direction: str
) -> tuple[pd.DataFrame, str]:
    """The rows to rank, most positive first and most negative last.

    ``both`` takes the top half from each end, so a corpus that is mostly
    positive still shows its most negative sentences -- ranking by distance
    from neutral alone gave nineteen bars at +0.99 and one below zero. When
    one end has fewer rows than its half, the other end fills the space, and
    the returned sentence says so.
    """
    above = working[working[score_column] >= center].sort_values(
        [score_column, "_stable_label"], ascending=[False, True], kind="stable"
    )
    below = working[working[score_column] < center].sort_values(
        [score_column, "_stable_label"], ascending=[True, True], kind="stable"
    )
    note = ""
    if direction == "positive":
        chosen_above, chosen_below = above.head(top_n), below.head(0)
    elif direction == "negative":
        chosen_above, chosen_below = above.head(0), below.head(top_n)
    else:
        want_below = top_n // 2
        want_above = top_n - want_below
        take_below = min(want_below, len(below))
        take_above = min(len(above), want_above + (want_below - take_below))
        take_below = min(len(below), top_n - take_above)
        chosen_above, chosen_below = above.head(take_above), below.head(take_below)
        if not len(below) or not len(above):
            side = "below" if not len(below) else "above"
            note = f"No row scores {side} neutral, so every row shown is from the other end."
    # Positive rows top down from the highest; then negative rows ending at
    # the lowest, so the most negative sits at the bottom of the figure.
    drawn = pd.concat([chosen_above, chosen_below.iloc[::-1]])
    return drawn.copy(), note


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def annual_sentiment_scores(  # noqa: PLR0913 - annual builder inputs are explicit schema and provenance
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    definition: PanelDefinition,
    score_column: str,
    unit: str,
    notes: tuple[str, ...],
) -> Result[PreparedPanel]:
    """Average document scores by dated year, retaining sample size per mark."""
    working = frame.copy()
    years = pd.to_numeric(working["Year"], errors="coerce")
    scores = pd.to_numeric(working[score_column], errors="coerce")
    valid_year = years.map(_is_finite) & years.map(
        lambda value: float(value).is_integer() if _is_finite(value) else False
    )
    valid_score = scores.map(_is_finite)
    valid = valid_year & valid_score
    undated = int((~valid_year).sum())
    bad_score = int((valid_year & ~valid_score).sum())
    working = working.loc[valid].copy()
    working["Year"] = years.loc[valid].astype(int)
    working[score_column] = scores.loc[valid].astype(float)

    diagnostics: list[Diagnostic] = []
    if undated:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_UNDATED_ROWS",
                f"{undated} document row(s) have no valid Year and were omitted from the annual view; their source scores remain in the table.",
                dropped=undated,
            )
        )
    if bad_score:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{bad_score} dated document row(s) had a non-numeric or non-finite {score_column} and were omitted.",
                dropped=bad_score,
            )
        )
    if working.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no dated document scores are available for an annual mean"),
            *diagnostics,
        )

    annual = working.groupby("Year", as_index=False, sort=True).agg(
        **{"Mean score": (score_column, "mean"), "Documents": (DOC_ID, "count")}
    )
    marks: list[PanelMark] = []
    for _, row in annual.iterrows():
        year, mean, count = int(row["Year"]), float(row["Mean score"]), int(row["Documents"])
        marks.append(
            PanelMark(
                key=str(year),
                label="Mean document score",
                x=float(year),
                y=mean,
                group=definition.tool,
                evidence=Evidence(
                    scope="rows",
                    filters=(("Year", str(year)),),
                    count=count,
                    describe=(
                        f"{year}: mean {score_column} {mean:.4f} across {count} dated document row(s)"
                        + (" (one document; interpret cautiously)" if count == 1 else "")
                    ),
                ),
            )
        )
    prepared = PreparedPanel(
        panel=definition.name,
        shape="line_series",
        title=definition.title,
        subtitle=f"Annual arithmetic mean of document scores · {len(marks)} year(s) · sample size varies by year",
        marks=tuple(marks),
        x_label="Year",
        y_label=f"Mean {score_column} ({unit})",
        provenance=provenance,
        data=annual,
        groups=(definition.tool,),
        notes=(
            *notes,
            "Each point is an arithmetic mean over dated document rows for that year; hover evidence reports the number of documents.",
            "Years with one document remain visible and should be interpreted cautiously. Undated documents and rows with invalid scores are omitted, not assigned to a year.",
            "An empty year has no dated document rows and remains a gap in the line; missing scores are not imputed.",
        ),
    )
    return Result.success(prepared, *diagnostics)


def _mark_label(row: pd.Series, label_column: str, identity_columns: tuple[str, ...], names: Mapping[str, str]) -> str:
    """What a bar is, in words: the speech, and for a sentence its opening.

    "58 · sentence 7" was unique and told the reader nothing; the row id is
    still in the evidence, and the full sentence in the hover.
    """
    document = names.get(str(row[DOC]), str(row[DOC])) if DOC in row.index else f"Document {row[DOC_ID]}"
    if SENT_ID in identity_columns:
        if SENTENCE in row.index and str(row[SENTENCE]).strip():
            return f"{document} — “{_snippet(str(row[SENTENCE]))}”"
        return f"{document}, sentence {row[SENT_ID]}"
    if label_column == DOC:
        return document
    return str(row[label_column])


def _unique(labels: list[str], ids: list[str]) -> list[str]:
    """Labels made unique by adding the row's id only where two collide
    (two speeches with one title, one opening line said twice)."""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return [f"{label} ({row_id})" if counts[label] > 1 else label for label, row_id in zip(labels, ids, strict=True)]


_DETOKENIZE = (
    (re.compile(r"\s+([,.;:!?%)\]])"), r"\1"),
    (re.compile(r"([(\[$])\s+"), r"\1"),
    (re.compile(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b"), r"\1"),
)


def _snippet(sentence: str) -> str:
    """A sentence's first words, cut at a word boundary.

    The sentence table stores tokens joined by spaces ("the 1980 's",
    "[ Laughter ] Well ,"); a label reads as prose, so the spaces a
    tokenizer added before punctuation and clitics are taken back out.
    """
    text = " ".join(sentence.split())
    for pattern, replacement in _DETOKENIZE:
        text = pattern.sub(replacement, text)
    if len(text) <= _SNIPPET_CHARS:
        return text
    cut = text[:_SNIPPET_CHARS].rsplit(" ", 1)[0].rstrip(",;:")
    return f"{cut}…"


def _builder(  # noqa: PLR0913 - declaration maps each public table schema to a panel
    *,
    name: str,
    tool: str,
    title: str,
    score_column: str,
    identity_columns: tuple[str, ...],
    label_column: str,
    unit: str,
    center: float = 0.0,
    notes: tuple[str, ...],
    requires: tuple[str, ...],
    question: str,
) -> PanelDefinition:
    # Bind every schema choice in the closure to keep the builder simple and
    # shareable while each registered definition remains tool-specific.
    holder: dict[str, PanelDefinition] = {}

    def build(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        return sentiment_scores(
            frame,
            params,
            provenance,
            definition=holder["definition"],
            score_column=score_column,
            identity_columns=identity_columns,
            label_column=label_column,
            unit=unit,
            center=center,
            notes=notes,
        )

    build.__name__ = f"build_{name}"
    definition = PanelDefinition(
        name=name,
        title=title,
        tool=tool,
        shape="ranked_bars",
        summary=f"The rows scoring furthest from neutral on {score_column}, from each end of the scale.",
        requires=requires,
        params=(_TOP_N, _DIRECTION),
        build=build,
        notes=notes,
        question=question,
    )
    holder["definition"] = definition
    return definition


SENTIMENT_VADER_SENTENCES = _builder(
    name="sentiment_vader_sentences",
    tool="sentiment_vader_anew",
    title="VADER sentence sentiment",
    question="Which sentences does VADER score as the most positive and the most negative?",
    score_column=COMPOUND,
    identity_columns=(DOC_ID, SENT_ID),
    label_column=SENT_ID,
    unit="VADER compound, [-1, +1]",
    notes=_VADER_NOTES,
    requires=(DOC_ID, SENT_ID, SENTENCE, COMPOUND, "Label"),
)
SENTIMENT_VADER_DOCUMENTS = _builder(
    name="sentiment_vader_documents",
    tool="sentiment_vader_anew",
    title="VADER document sentiment",
    question="Which documents does VADER score as the most positive and the most negative overall?",
    score_column=COMPOUND,
    identity_columns=(DOC_ID,),
    label_column=DOC,
    unit="mean VADER compound, [-1, +1]",
    notes=_VADER_NOTES,
    requires=(DOC_ID, DOC, COMPOUND),
)
SENTIMENT_ANEW_VALENCE = _builder(
    name="sentiment_anew_valence",
    tool="sentiment_vader_anew",
    title="ANEW document valence",
    question="Which documents use the most pleasant and the most unpleasant words, by ANEW's ratings?",
    score_column=VALENCE,
    identity_columns=(DOC_ID,),
    label_column=DOC,
    unit="ANEW valence, 1-9",
    center=5.0,
    notes=_ANEW_NOTES,
    requires=(DOC_ID, DOC, VALENCE, "Hits"),
)


def _neural(tool: str, short: str) -> tuple[PanelDefinition, PanelDefinition]:
    notes = _NEURAL_LIMITS[tool] + (
        "Neural Compound is a signed model output, not a universal sentiment unit; compare only with the same model/backend settings.",
        "Annual views require a Year column attached to dated document rows. Undated documents remain in the score table and are omitted from annual means.",
    )
    sentences = _builder(
        name=f"{tool}_sentences",
        tool=tool,
        title=f"{short} sentence sentiment",
        question=f"Which sentences does {short} score as the most positive and the most negative?",
        score_column=COMPOUND,
        identity_columns=(DOC_ID, SENT_ID),
        label_column=SENT_ID,
        unit="signed Compound, [-1, +1]",
        notes=notes,
        requires=(DOC_ID, SENT_ID, SENTENCE, "Label", "Score", COMPOUND),
    )
    documents = _builder(
        name=f"{tool}_documents",
        tool=tool,
        title=f"{short} document sentiment",
        question=f"Which documents does {short} score as the most positive and the most negative overall?",
        score_column=MEAN_COMPOUND,
        identity_columns=(DOC_ID,),
        label_column=DOC,
        unit="mean signed Compound, [-1, +1]",
        notes=notes,
        requires=(DOC_ID, DOC, "Sentences", MEAN_COMPOUND, "Positive", "Negative", "Neutral"),
    )
    return sentences, documents


def _annual_builder(tool: str, title: str, score_column: str, unit: str, notes: tuple[str, ...]) -> PanelDefinition:
    holder: dict[str, PanelDefinition] = {}

    def build(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        return annual_sentiment_scores(
            frame,
            params,
            provenance,
            definition=holder["definition"],
            score_column=score_column,
            unit=unit,
            notes=notes,
        )

    name = f"{tool}_over_time"
    definition = PanelDefinition(
        name=name,
        title=title,
        tool=tool,
        shape="line_series",
        summary=f"Annual arithmetic means of dated document-level {score_column} scores, with row counts.",
        requires=(DOC_ID, "Year", score_column),
        params=(),
        build=build,
        notes=notes,
        question=f"Has the tone of these documents changed over time, by {title.split(maxsplit=1)[0]}'s measure?",
    )
    holder["definition"] = definition
    return definition


SENTIMENT_BERT_SENTENCES, SENTIMENT_BERT_DOCUMENTS = _neural("sentiment_neural_bert", "BERT")
SENTIMENT_SPACY_SENTENCES, SENTIMENT_SPACY_DOCUMENTS = _neural("sentiment_neural_spacy", "spaCy")
SENTIMENT_STANZA_SENTENCES, SENTIMENT_STANZA_DOCUMENTS = _neural("sentiment_neural_stanza", "Stanza")
SENTIMENT_CORENLP_SENTENCES, SENTIMENT_CORENLP_DOCUMENTS = _neural("sentiment_neural_corenlp", "CoreNLP")

SENTIMENT_VADER_OVER_TIME = _annual_builder(
    "sentiment_vader_anew",
    "VADER document sentiment over time",
    COMPOUND,
    "VADER compound, [-1, +1]",
    _VADER_NOTES,
)
SENTIMENT_BERT_OVER_TIME = _annual_builder(
    "sentiment_neural_bert",
    "BERT document sentiment over time",
    MEAN_COMPOUND,
    "signed Compound, [-1, +1]",
    _NEURAL_LIMITS["sentiment_neural_bert"],
)
SENTIMENT_SPACY_OVER_TIME = _annual_builder(
    "sentiment_neural_spacy",
    "spaCy document sentiment over time",
    MEAN_COMPOUND,
    "signed Compound, [-1, +1]",
    _NEURAL_LIMITS["sentiment_neural_spacy"],
)
SENTIMENT_STANZA_OVER_TIME = _annual_builder(
    "sentiment_neural_stanza",
    "Stanza document sentiment over time",
    MEAN_COMPOUND,
    "signed Compound, [-1, +1]",
    _NEURAL_LIMITS["sentiment_neural_stanza"],
)
SENTIMENT_CORENLP_OVER_TIME = _annual_builder(
    "sentiment_neural_corenlp",
    "CoreNLP document sentiment over time",
    MEAN_COMPOUND,
    "signed Compound, [-1, +1]",
    _NEURAL_LIMITS["sentiment_neural_corenlp"],
)

ANNUAL_SENTIMENT_PANELS = (
    SENTIMENT_VADER_OVER_TIME,
    SENTIMENT_BERT_OVER_TIME,
    SENTIMENT_SPACY_OVER_TIME,
    SENTIMENT_STANZA_OVER_TIME,
    SENTIMENT_CORENLP_OVER_TIME,
)

SENTIMENT_PANELS = (
    SENTIMENT_VADER_SENTENCES,
    SENTIMENT_VADER_DOCUMENTS,
    SENTIMENT_ANEW_VALENCE,
    SENTIMENT_BERT_SENTENCES,
    SENTIMENT_BERT_DOCUMENTS,
    SENTIMENT_SPACY_SENTENCES,
    SENTIMENT_SPACY_DOCUMENTS,
    SENTIMENT_STANZA_SENTENCES,
    SENTIMENT_STANZA_DOCUMENTS,
    SENTIMENT_CORENLP_SENTENCES,
    SENTIMENT_CORENLP_DOCUMENTS,
    *ANNUAL_SENTIMENT_PANELS,
)
