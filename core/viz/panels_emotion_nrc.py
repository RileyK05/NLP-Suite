"""NRC emotion panels -- corpus emotion profile, trajectories over time, and distribution.

The emotion_nrc tool (``core/analysis/emotion_nrc.py``) computes the proportions of
eight primary Plutchik emotions (fear, anger, anticipation, trust, surprise, sadness,
disgust, joy) plus positive/negative valence for each document.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels, no_axis, positioned
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = [
    "NRC_EMOTION_DISTRIBUTION",
    "NRC_EMOTION_OVER_TIME",
    "NRC_EMOTION_PANELS",
    "NRC_EMOTION_PROFILE",
    "nrc_emotion_distribution",
    "nrc_emotion_over_time",
    "nrc_emotion_profile",
]

_DOC_ID = "Document ID"
_DOC = "Document"
_TOKENS = "Tokens"
_HITS = "Hits"

_EIGHT_EMOTIONS: tuple[str, ...] = (
    "fear",
    "anger",
    "anticipation",
    "trust",
    "surprise",
    "sadness",
    "disgust",
    "joy",
)

_EMOTION_COLORS: dict[str, str] = {
    "joy": "#E69F00",  # orange / gold
    "trust": "#009E73",  # green
    "fear": "#56B4E9",  # sky blue
    "surprise": "#F0E442",  # yellow
    "sadness": "#0072B2",  # blue
    "disgust": "#CC79A7",  # purple/pink
    "anger": "#D55E00",  # vermillion
    "anticipation": "#999999",  # neutral / grey
}


def nrc_emotion_profile(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Ranked bars of corpus-wide prevalence across the eight Plutchik emotions."""
    _ = params
    working = frame.copy()
    missing = [e for e in _EIGHT_EMOTIONS if e not in working.columns]
    if missing:
        return Result.failure(Diagnostic.error("PANEL_MISSING_COLUMN", f"missing emotion columns: {missing}"))

    for e in _EIGHT_EMOTIONS:
        working[e] = pd.to_numeric(working[e], errors="coerce").fillna(0.0)

    means = {e: float(working[e].mean()) for e in _EIGHT_EMOTIONS}
    ranked = sorted(means.items(), key=lambda item: item[1], reverse=True)

    marks: list[PanelMark] = []
    for rank, (emotion, mean_val) in enumerate(ranked):
        pct = mean_val * 100.0
        marks.append(
            PanelMark(
                key=emotion,
                label=emotion.capitalize(),
                x=pct,
                y=float(rank),
                evidence=Evidence(
                    scope="rows",
                    filters=(("Emotion", emotion),),
                    count=int(working[_HITS].sum()) if _HITS in working.columns else len(working),
                    describe=f"{emotion.capitalize()}: {pct:.1f}% mean share of scored emotion hits across corpus",
                ),
            )
        )

    summary_df = pd.DataFrame([{"Emotion": emotion, "MeanSharePct": mean_val * 100.0} for emotion, mean_val in ranked])

    prepared = PreparedPanel(
        panel=NRC_EMOTION_PROFILE.name,
        shape="ranked_bars",
        title=NRC_EMOTION_PROFILE.title,
        subtitle=f"Corpus-wide average prevalence across {len(_EIGHT_EMOTIONS)} Plutchik emotions",
        marks=tuple(marks),
        x_label="Mean Share of Emotion Hits (%)",
        y_label="",
        provenance=provenance,
        data=summary_df,
        notes=NRC_EMOTION_PROFILE.notes,
    )
    return Result.success(prepared)


def nrc_emotion_over_time(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Trajectories of key emotions across the corpus timeline or sequence."""
    _ = params
    working = frame.copy()
    working[_DOC] = working[_DOC].astype(str)
    missing = [e for e in _EIGHT_EMOTIONS if e not in working.columns]
    if missing:
        return Result.failure(Diagnostic.error("PANEL_MISSING_COLUMN", f"missing emotion columns: {missing}"))

    for e in _EIGHT_EMOTIONS:
        working[e] = pd.to_numeric(working[e], errors="coerce").fillna(0.0)

    places, axis = positioned(working)
    # Like every figure about change: no dates and no order, no line. File
    # order is usually alphabetical, and a line over it reads as a trend.
    if axis.kind == "none":
        return Result.failure(no_axis("Following emotions across the corpus"))
    x_label, along = axis.x_label, axis.along

    working["_pos"] = places.astype(float)
    working = working.dropna(subset=["_pos"]).sort_values("_pos", kind="stable").reset_index(drop=True)
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no documents with valid timeline positions"))

    labels = document_labels(working[_DOC].tolist())

    # Focus on top contrasting primary emotions: joy, trust, fear, anger, sadness
    displayed_emotions = ("joy", "trust", "fear", "anger", "sadness")

    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        doc = str(row[_DOC])
        doc_disp = labels.get(doc, doc)
        pos = float(row["_pos"])
        for emotion in displayed_emotions:
            val = float(row[emotion]) * 100.0
            marks.append(
                PanelMark(
                    key=f"{doc}:{emotion}",
                    label=emotion.capitalize(),
                    group=emotion.capitalize(),
                    x=pos,
                    y=val,
                    evidence=Evidence(
                        scope="rows",
                        filters=((_DOC, doc),),
                        count=1,
                        describe=f"{doc_disp}: {emotion} = {val:.1f}% of emotion hits",
                    ),
                )
            )

    prepared = PreparedPanel(
        panel=NRC_EMOTION_OVER_TIME.name,
        shape="line_series",
        title=NRC_EMOTION_OVER_TIME.title,
        subtitle=f"Primary emotions across {len(working)} document(s) {along}",
        marks=tuple(marks),
        x_label=x_label,
        x_axis=axis.x_axis,
        x_noun=axis.x_noun,
        y_label="Share of Document Hits (%)",
        provenance=provenance,
        data=working,
        groups=tuple(e.capitalize() for e in displayed_emotions),
        notes=NRC_EMOTION_OVER_TIME.notes,
    )
    return Result.success(prepared)


def nrc_emotion_distribution(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Box and point distribution of document scores for each of the eight emotions."""
    _ = params
    working = frame.copy()
    missing = [e for e in _EIGHT_EMOTIONS if e not in working.columns]
    if missing:
        return Result.failure(Diagnostic.error("PANEL_MISSING_COLUMN", f"missing emotion columns: {missing}"))

    for e in _EIGHT_EMOTIONS:
        working[e] = pd.to_numeric(working[e], errors="coerce").fillna(0.0)

    doc_col = _DOC if _DOC in working.columns else _DOC_ID
    labels = document_labels(working[doc_col].astype(str).tolist()) if doc_col in working.columns else {}

    y_categories = tuple(e.capitalize() for e in _EIGHT_EMOTIONS)
    emotion_to_idx = {e: i for i, e in enumerate(_EIGHT_EMOTIONS)}

    marks: list[PanelMark] = []
    for row_idx, row in working.iterrows():
        doc_id = str(row.get(_DOC_ID, f"doc_{row_idx}"))
        doc_name = str(row.get(_DOC, doc_id))
        doc_disp = labels.get(doc_name, doc_name)
        for emotion in _EIGHT_EMOTIONS:
            val = float(row[emotion]) * 100.0
            y_idx = emotion_to_idx[emotion]
            marks.append(
                PanelMark(
                    key=f"{doc_id}:{emotion}",
                    label=doc_disp,
                    x=val,
                    y=float(y_idx),
                    evidence=Evidence(
                        scope="rows",
                        filters=((_DOC_ID, doc_id),) if _DOC_ID in working.columns else (),
                        count=1,
                        describe=f"{doc_disp}: {emotion} = {val:.1f}% of emotion hits",
                    ),
                )
            )

    prepared = PreparedPanel(
        panel=NRC_EMOTION_DISTRIBUTION.name,
        shape="distribution",
        title=NRC_EMOTION_DISTRIBUTION.title,
        subtitle=f"Distribution of emotion shares across {len(working)} document(s)",
        marks=tuple(marks),
        x_label="Share of Document Hits (%)",
        y_label="Emotion",
        provenance=provenance,
        data=working,
        y_categories=y_categories,
        notes=NRC_EMOTION_DISTRIBUTION.notes,
    )
    return Result.success(prepared)


NRC_EMOTION_PROFILE = PanelDefinition(
    name="nrc_emotion_profile",
    title="Corpus emotion profile (8 Plutchik emotions)",
    question="Which emotions are most prevalent across the corpus as a whole?",
    tool="emotion_nrc",
    shape="ranked_bars",
    summary="Corpus-wide average share across the eight Plutchik emotions.",
    requires=(_DOC_ID, *_EIGHT_EMOTIONS),
    params=(),
    build=nrc_emotion_profile,
    notes=(
        "Prevalence is the unweighted average across documents of each emotion's share of hits.",
        "Lexicon coverage varies across texts; documents with very few total emotion words carry equal weight here.",
    ),
)

NRC_EMOTION_OVER_TIME = PanelDefinition(
    name="nrc_emotion_over_time",
    title="Emotion trajectories across the corpus",
    question="How do primary emotions trend over the corpus timeline or document sequence?",
    tool="emotion_nrc",
    shape="line_series",
    summary="Trajectories of primary emotions (joy, trust, fear, anger, sadness) over timeline or order.",
    requires=(_DOC, *_EIGHT_EMOTIONS),
    params=(),
    build=nrc_emotion_over_time,
    notes=(
        "Positions are recovered from document dates or sequential corpus order.",
        "Lines show individual document percentages; look for broad shifts rather than single-document spikes.",
    ),
)

NRC_EMOTION_DISTRIBUTION = PanelDefinition(
    name="nrc_emotion_distribution",
    title="Emotion score distribution across documents",
    question="How variable is each emotion across the corpus?",
    tool="emotion_nrc",
    shape="distribution",
    summary="Box and point distribution of document scores for each of the eight emotions.",
    requires=(_DOC_ID, *_EIGHT_EMOTIONS),
    params=(),
    build=nrc_emotion_distribution,
    notes=(
        "Each dot represents one document's proportion for that emotion, with box showing quartiles and median.",
        "Emotions with wide spreads highlight stylistic or thematic divergence across authors or eras.",
    ),
)

NRC_EMOTION_PANELS: tuple[PanelDefinition, ...] = (
    NRC_EMOTION_PROFILE,
    NRC_EMOTION_OVER_TIME,
    NRC_EMOTION_DISTRIBUTION,
)
