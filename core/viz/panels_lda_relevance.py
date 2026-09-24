"""Topic-term relevance — pyLDAvis's right-hand chart, natively.

``core/analysis/lda.py::relevance_terms`` scores every (topic, word) pair two
ways, after Chuang et al./pyLDAvis:

* **Relevance** -- ``lam * log p(word|topic) + (1 - lam) * log p(word)``.
  Lowering lam below 1 discounts a word by how common it is in the whole
  corpus, which lifts terms that are distinctive of the topic even when they
  are individually rare.
* **Saliency** -- ``p(word|topic) * (log p(word|topic) - reference)``, where
  the reference is how much the word distinguishes topics at all. It is NOT
  relevance at lam = 1: that would be ``log p(word|topic)`` alone, and
  saliency is computed independently of lam.

**Neither number can be a bar length**, and the first version of this panel
learned that on the real corpus. Both are log-scale scores. Relevance is a
sum of logs of probabilities, so it is always negative, and the best-ranked
term has the *smallest* magnitude: drawn from zero, the top word got the
shortest bar, and rank and length pointed in opposite directions. Saliency
ran about a hundred times smaller than relevance and would have been drawn on
the same axis as slivers.

So this panel does what pyLDAvis does. The scores **order** the rows; the
bars are **counts**, which are positive and share one scale:

* ``In this topic`` -- ``Topic frequency``, the model's estimate of how many
  of the word's occurrences belong to this topic.
* ``In the whole corpus`` -- ``Corpus frequency``, how many times it occurs.

The gap between the two bars is the finding. A word whose corpus count dwarfs
its topic count is common everywhere and says little about this topic; a word
whose two bars nearly meet is one this topic owns. A single ranking hides that
case, and two log scores on one axis hide it too.

Ranking is the caller's choice (``order-by``): relevance and saliency answer
different questions, and blending them would put a number on the page that no
column holds. The keyness volcano makes the same refusal for ``label-by``.

Pure function of (frame, params, provenance): no plotly, no filesystem, no
model access.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = ["LDA_RELEVANCE", "lda_relevance"]

TOPIC = "Topic"
WORD = "Word"
RELEVANCE = "Relevance"
SALIENCY = "Saliency"
CORPUS_FREQUENCY = "Corpus frequency"
TOPIC_FREQUENCY = "Topic frequency"

# Foreground first: both renderers draw the first series opaque and the second
# translucent over it, so the topic's own share reads as the solid bar and the
# corpus total as the lighter one it sits inside.
_TOPIC_GROUP = "In this topic"
_CORPUS_GROUP = "In the whole corpus"


def lda_relevance(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Prepare one topic's terms: ordered by a score, drawn as counts."""
    topic = int(params["topic"])
    top_n = int(params["top-n"])
    order_by = str(params["order-by"])
    rank_column = RELEVANCE if order_by == "relevance" else SALIENCY

    diagnostics: list[Diagnostic] = []
    working = frame.copy()

    # Topic ids come from the fitted model (0-based, Gensim-native), but a
    # frame from a different run may simply not contain the requested topic.
    # Say which ones it does contain rather than drawing an empty figure.
    topic_numbers = pd.to_numeric(working[TOPIC], errors="coerce")
    available = sorted({int(value) for value in topic_numbers.dropna().unique()})
    if topic not in available:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                f"topic {topic} is not in this run; topics present: {available}",
                requested=topic,
                available=available,
            )
        )

    selected = working[topic_numbers == topic].copy()

    # Every column the row is placed or ordered by must be a real, finite
    # number. A term with a missing count is left out with a count, never
    # drawn at zero -- a zero bar asserts the word never occurs.
    numeric = (RELEVANCE, SALIENCY, CORPUS_FREQUENCY, TOPIC_FREQUENCY)
    for column in numeric:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    placeable = selected[list(numeric)].apply(lambda column: column.map(_is_finite)).all(axis=1)
    unplaceable = int((~placeable).sum())
    if unplaceable:
        missing_counts = int(selected[TOPIC_FREQUENCY].isna().sum())
        why = (
            " The topic counts are missing, which means this table was written without the topic's "
            "token mass -- re-run the topic model to get them."
            if missing_counts == unplaceable
            else ""
        )
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{unplaceable} row(s) for topic {topic} had a missing or non-numeric score or count and "
                f"were left out of the plot.{why}",
                dropped=unplaceable,
            )
        )
    selected = selected[placeable]

    if selected.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                f"topic {topic} has no rows with usable scores and counts; topics present: {available}",
                requested=topic,
                available=available,
            ),
            *diagnostics,
        )

    ranked = selected.assign(_rank=_score(selected, rank_column)).sort_values(
        ["_rank", WORD], ascending=[False, True], kind="stable"
    )
    drawn = ranked.head(top_n)

    if _same_order(drawn, RELEVANCE) == _same_order(drawn, SALIENCY):
        # The two scores exist precisely to disagree sometimes. When they do
        # not -- for these terms -- switching order-by changes nothing, and
        # saying so saves a reader from looking for a difference.
        diagnostics.append(
            Diagnostic.info(
                "PANEL_RANKINGS_AGREE",
                f"Relevance and Saliency rank these {len(drawn)} terms in the same order for topic {topic}, "
                "so switching order-by will not change this panel.",
                topic=topic,
                terms=len(drawn),
            )
        )

    marks: list[PanelMark] = []
    # iterrows rather than itertuples: two of these columns have spaces in
    # their names, which itertuples silently renames to positional fields.
    for position, (_, record) in enumerate(drawn.iterrows()):
        word = str(record[WORD])
        in_topic = float(record[TOPIC_FREQUENCY])
        in_corpus = float(record[CORPUS_FREQUENCY])
        relevance_value = float(record[RELEVANCE])
        saliency_value = float(record[SALIENCY])
        share = f"{in_topic / in_corpus:.0%}" if in_corpus > 0 else "n/a"
        describe = (
            f"{word} in topic {topic}: about {in_topic:,.0f} of its {in_corpus:,.0f} occurrences ({share}) "
            f"-- Relevance {relevance_value:.4f}, Saliency {saliency_value:.4f}"
        )
        evidence = Evidence(
            scope="terms",
            filters=((TOPIC, str(topic)), (WORD, word)),
            count=1,
            describe=describe,
            # Read the training field from the run so a surface-form model is
            # searched literally while a lemma model includes its inflections.
            phrase=word,
            lemma=str(provenance.settings.get("field", "lemma")) == "lemma",
        )
        marks.append(
            PanelMark(
                key=f"{word}:topic",
                label=word,
                x=in_topic,
                y=float(position),
                group=_TOPIC_GROUP,
                evidence=evidence,
            )
        )
        marks.append(
            PanelMark(
                key=f"{word}:corpus",
                label=word,
                x=in_corpus,
                y=float(position),
                group=_CORPUS_GROUP,
                evidence=evidence,
            )
        )

    prepared = PreparedPanel(
        panel=LDA_RELEVANCE.name,
        shape="ranked_bars",
        title=f"Topic {topic}: the terms that define it",
        subtitle=f"{len(drawn)} terms, ordered by {order_by} · bars are occurrences",
        marks=tuple(marks),
        x_label="Occurrences (in this topic, estimated, against the whole corpus)",
        y_label=f"Terms, ordered by {order_by}",
        provenance=provenance,
        data=drawn[[TOPIC, WORD, RELEVANCE, SALIENCY, TOPIC_FREQUENCY, CORPUS_FREQUENCY]].reset_index(drop=True),
        groups=(_TOPIC_GROUP, _CORPUS_GROUP),
        notes=_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def _is_finite(value: Any) -> bool:
    """True for a real, finite number. NaN and +/-inf cannot be placed."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _score(frame: pd.DataFrame, column: str) -> pd.Series:
    """The value a column is ordered by: higher is more important.

    Both score columns now order as written. Relevance is higher (closer to
    zero) for more relevant terms; Saliency is the corrected Chuang et al.
    KL divergence, never negative and larger for more salient terms -- see
    ``relevance_terms`` in ``core/analysis/lda.py``, and
    ``TestRelevanceLambda`` for the definitional checks. The old engine's
    inverted saliency (and this function's former magnitude fallback) are
    both gone, so an ordering that looks wrong on real data is a real bug
    again rather than something two conventions can hide.
    """
    return frame[column]


def _same_order(frame: pd.DataFrame, column: str) -> list[str]:
    """The drawn terms' Word order under one score, ties broken by Word."""
    ordered = frame.assign(_order_rank=_score(frame, column)).sort_values(
        ["_order_rank", WORD], ascending=[False, True], kind="stable"
    )
    return [str(word) for word in ordered[WORD]]


_NOTES: tuple[str, ...] = (
    "The bars are occurrences, not scores. The rows are ordered by Relevance or Saliency, which are "
    "log-scale numbers that cannot be drawn as lengths; the counts can.",
    "The topic count is the model's estimate -- the word's probability in the topic times the topic's "
    "share of all tokens -- not an observed count. A word is never assigned to one topic outright.",
    "A word whose corpus bar dwarfs its topic bar is common everywhere and says little about this "
    "topic. A word whose two bars nearly meet is one this topic owns.",
    "Relevance depends on the lambda chosen when the model was fitted. This panel shows the table "
    "as fitted; it does not recompute Relevance for a different lambda.",
    "A topic is a distribution over words, not a subject. The name a reader gives it ('economy', "
    "'foreign policy') is an interpretation of these terms, not something the model outputs.",
)


LDA_RELEVANCE = PanelDefinition(
    name="lda_relevance",
    title="Topic-term relevance",
    question="Which words define each topic?",
    tool="lda_gensim",
    shape="ranked_bars",
    summary=(
        "One topic's defining terms, ordered by Relevance or Saliency, each drawn as its estimated "
        "count in the topic against its count in the whole corpus."
    ),
    requires=(TOPIC, WORD, RELEVANCE, SALIENCY, CORPUS_FREQUENCY, TOPIC_FREQUENCY),
    params=(
        PanelParam(
            name="topic",
            type="int",
            default=0,
            minimum=0,
            label="Topic",
            help="Which topic (0-based, as the model numbered it) to show terms for.",
        ),
        PanelParam(
            name="top-n",
            type="int",
            default=15,
            minimum=1,
            maximum=50,
            label="Terms to show",
            help="How many terms to draw, taken from the top of the chosen ordering.",
        ),
        PanelParam(
            name="order-by",
            type="choice",
            default="relevance",
            choices=("relevance", "saliency"),
            label="Order by",
            help=(
                "Which score orders the rows: 'relevance' (lam-weighted distinctiveness against the corpus) "
                "or 'saliency' (importance to this topic, weighted by how much the word tells topics apart). "
                "There is no blended default."
            ),
        ),
    ),
    build=lda_relevance,
    notes=_NOTES,
)
