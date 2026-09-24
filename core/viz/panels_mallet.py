"""MALLET document-topic composition and ordered topic-term views."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panelspec import Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

DOCUMENT = "Document"
PROPORTIONS = "Topic proportions"
TOPIC = "Topic"
WORDS = "Words"
_NOTES = (
    "Each coloured segment is the published MALLET topic proportion, not a span of text assigned to that topic.",
    "A topic number is a model component; interpreting or naming it requires reading its terms and source documents.",
    "Rounding may leave a very small uncoloured remainder; shares are not silently renormalized.",
)


def _parts(value: object) -> list[tuple[int, float]] | None:
    if not isinstance(value, str):
        return None
    found: list[tuple[int, float]] = []
    seen: set[int] = set()
    for part in value.split(","):
        topic, separator, share = part.strip().partition(":")
        if not separator:
            return None
        try:
            identifier, fraction = int(topic), float(share)
        except ValueError:
            return None
        if identifier < 0 or identifier in seen or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            return None
        seen.add(identifier)
        found.append((identifier, fraction))
    return sorted(found) if found and sum(share for _, share in found) <= 1.01 else None


def mallet_document_topics(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """One horizontal composition ribbon per document, using saved shares."""
    limit = int(params["documents"])
    working = frame.head(limit).copy()
    if working[DOCUMENT].astype(str).duplicated().any():
        return Result.failure(
            Diagnostic.error("PANEL_AMBIGUOUS_EVIDENCE", "duplicate document labels cannot be linked to unique rows")
        )
    marks: list[PanelMark] = []
    topics: set[int] = set()
    bad: list[str] = []
    for row_number, (_, row) in enumerate(working.iterrows()):
        name = str(row[DOCUMENT])
        shares = _parts(row[PROPORTIONS])
        if shares is None:
            bad.append(name)
            continue
        start = 0.0
        for topic, share in shares:
            topics.add(topic)
            marks.append(
                PanelMark(
                    key=f"{name}:{topic}",
                    label=name,
                    x=start,
                    y=float(row_number),
                    size=share,
                    group=f"Topic {topic}",
                    evidence=Evidence(
                        scope="rows",
                        filters=((DOCUMENT, name),),
                        count=1,
                        describe=f"{name}: topic {topic} accounts for {share:.1%} of the document's modelled topic mixture",
                    ),
                )
            )
            start += share
    if not marks:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no valid document-topic proportions are available"))
    diagnostics: list[Diagnostic] = []
    if bad:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_PROPORTIONS",
                f"{len(bad)} document(s) have invalid topic proportions and were omitted",
                documents=bad[:5],
            )
        )
    if len(frame) > limit:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_DOCUMENTS_CAPPED",
                f"Showing {limit} of {len(frame)} documents; raise the document limit to see more",
                shown=limit,
                available=len(frame),
            )
        )
    return Result.success(
        PreparedPanel(
            panel=MALLET_DOCUMENT_TOPICS.name,
            shape="ribbon",
            title="MALLET topic mixture by document",
            subtitle=f"{len(working) - len(bad)} documents · topic shares from the fitted model",
            marks=tuple(marks),
            x_label="Share of document topic mixture",
            y_label="Documents",
            provenance=provenance,
            data=working.loc[~working[DOCUMENT].astype(str).isin(bad)].reset_index(drop=True),
            groups=tuple(f"Topic {topic}" for topic in sorted(topics)),
            notes=_NOTES,
        ),
        *diagnostics,
    )


def mallet_topic_terms(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    """Show MALLET's ranked words without inventing per-word probabilities."""
    raw_topic = pd.to_numeric(frame[TOPIC], errors="coerce")
    topic = int(params["topic"])
    selected = frame.loc[raw_topic == topic]
    if selected.empty:
        available = sorted(int(value) for value in raw_topic.dropna().unique())
        return Result.failure(
            Diagnostic.error(
                "PANEL_TOPIC_NOT_FOUND",
                f"topic {topic} is not in this MALLET output; topics present: {available}",
                topic=topic,
                available=available,
            )
        )
    if len(selected) != 1:
        return Result.failure(
            Diagnostic.error(
                "PANEL_AMBIGUOUS_EVIDENCE",
                f"topic {topic} appears in {len(selected)} rows; topic terms cannot be linked uniquely",
                topic=topic,
            )
        )
    source_words = str(selected.iloc[0][WORDS]).strip()
    terms = [word for word in source_words.split() if word]
    limit = int(params["top-n"])
    terms = terms[:limit]
    if not terms:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"MALLET topic {topic} has no listed terms"))

    categories = tuple(terms)
    marks: list[PanelMark] = []
    for rank, word in enumerate(terms, start=1):
        marks.append(
            PanelMark(
                key=f"topic-{topic}:{rank}:{word}",
                label=word,
                x=float(rank),
                y=float(rank - 1),
                labelled=True,
                evidence=Evidence(
                    scope="terms",
                    filters=((TOPIC, str(topic)), ("Rank", str(rank)), ("Word", word)),
                    count=1,
                    describe=f"{word} is ranked {rank} in MALLET topic {topic}; the output gives rank order, not a per-word probability",
                    phrase=word,
                ),
            )
        )
    data = pd.DataFrame(
        [{TOPIC: topic, "Rank": rank, "Word": word, WORDS: source_words} for rank, word in enumerate(terms, start=1)],
        columns=[TOPIC, "Rank", "Word", WORDS],
    )
    return Result.success(
        PreparedPanel(
            panel=MALLET_TOPIC_TERMS.name,
            shape="positions",
            title=f"MALLET topic {topic}: top terms",
            subtitle=f"{len(terms)} terms in MALLET's listed order",
            marks=tuple(marks),
            x_label="Rank in the topic-keys list (lower is earlier)",
            y_label="Top terms",
            provenance=provenance,
            data=data,
            y_categories=categories,
            notes=(
                "MALLET's topic-keys file gives an ordered list of words, not a probability for each word. This figure preserves that order and does not turn rank into a score.",
                "Topic IDs are model labels, not meanings. Read the terms and the linked source passages before naming a topic.",
                "The separate Weight column is topic-level output; it is not used here as a word weight.",
            ),
        )
    )


MALLET_DOCUMENT_TOPICS = PanelDefinition(
    name="mallet_document_topics",
    title="Topic mixture by document",
    question="What mix of topics is each document made of?",
    tool="lda_mallet",
    shape="ribbon",
    summary="One document per ribbon; colours show the fitted MALLET topic proportions.",
    requires=(DOCUMENT, PROPORTIONS),
    params=(
        PanelParam(
            name="documents",
            type="int",
            default=25,
            minimum=1,
            maximum=100,
            label="Documents",
            help="Maximum documents to show, in result order.",
        ),
    ),
    build=mallet_document_topics,
    notes=_NOTES,
)

MALLET_TOPIC_TERMS = PanelDefinition(
    name="mallet_topic_terms",
    title="MALLET topic terms in ranked order",
    question="Which words did MALLET list for a topic, and in what order?",
    tool="lda_mallet",
    shape="positions",
    summary="Show the topic-keys list as rank order; the output does not provide per-word probabilities.",
    requires=(TOPIC, WORDS),
    params=(
        PanelParam(
            name="topic",
            type="int",
            default=0,
            minimum=0,
            maximum=1000,
            label="Topic number",
            help="Choose a topic ID from topics.csv; topic IDs are model labels.",
        ),
        PanelParam(
            name="top-n",
            type="int",
            default=15,
            minimum=1,
            maximum=50,
            label="Terms to show",
            help="Show the first N words in MALLET's ordered topic-keys list.",
        ),
    ),
    build=mallet_topic_terms,
    notes=(
        "MALLET's topic-keys file gives an ordered list of words, not a probability for each word. This figure preserves that order and does not turn rank into a score.",
        "Topic IDs are model labels, not meanings. Read the terms and the linked source passages before naming a topic.",
        "The separate Weight column is topic-level output; it is not used here as a word weight.",
    ),
)
