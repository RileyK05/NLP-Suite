"""Spellcheck panels -- error frequencies, document error rates, and correction confidence.

The spellcheck table (``core/analysis/spellcheck.py``) records every unknown word
encountered in the corpus, its occurrence count, the dictionary suggestion, and the
edit distance / similarity score.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels
from core.viz.panelspec import (
    Annotation,
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = [
    "SPELLCHECK_CONFIDENCE_SCATTER",
    "SPELLCHECK_DOCUMENT_COUNTS",
    "SPELLCHECK_PANELS",
    "SPELLCHECK_TOP_ERRORS",
    "spellcheck_confidence_scatter",
    "spellcheck_document_counts",
    "spellcheck_top_errors",
]

_WORD = "Word"
_COUNT = "Count"
_SUGGESTION = "Suggestion"
_SCORE = "Score"
_DOC = "Document"


def spellcheck_top_errors(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Ranked bars of the most frequent misspelled / unknown words."""
    top_n = int(params.get("top-n", 25))
    working = frame.copy()
    working[_WORD] = working[_WORD].astype(str).str.strip()
    working[_COUNT] = pd.to_numeric(working[_COUNT], errors="coerce").fillna(0)
    working = working[working[_COUNT] > 0]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no spelling errors with positive count"))

    # Group by Word
    grouped = (
        working.groupby(_WORD, sort=False)
        .agg(
            TotalCount=(_COUNT, "sum"),
            TopSuggestion=(_SUGGESTION, "first"),
            TopScore=(_SCORE, "first") if _SCORE in working.columns else (_COUNT, lambda _: 1.0),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["TotalCount", _WORD], ascending=[False, True], kind="stable").head(top_n)

    marks: list[PanelMark] = []
    for rank, (_, row) in enumerate(grouped.iterrows()):
        word = str(row[_WORD])
        cnt = int(row["TotalCount"])
        sug = str(row["TopSuggestion"])
        score_val = float(row["TopScore"])
        marks.append(
            PanelMark(
                key=word,
                label=word,
                x=float(cnt),
                y=float(rank),
                evidence=Evidence(
                    scope="terms",
                    filters=((_WORD, word),),
                    count=cnt,
                    phrase=word,
                    describe=f"'{word}': {cnt} occurrence(s) across corpus -- suggestion: '{sug}' (score {score_val:.2f})",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=SPELLCHECK_TOP_ERRORS.name,
        shape="ranked_bars",
        title=SPELLCHECK_TOP_ERRORS.title,
        subtitle=f"Top {len(marks)} misspelled words by corpus frequency",
        marks=tuple(marks),
        x_label="Total Occurrences",
        y_label="",
        provenance=provenance,
        data=grouped,
        notes=SPELLCHECK_TOP_ERRORS.notes,
    )
    return Result.success(prepared)


def spellcheck_document_counts(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Ranked bars of total spelling errors per document."""
    top_n = int(params.get("top-n", 25))
    working = frame.copy()
    working[_DOC] = working[_DOC].astype(str).str.strip()
    working[_COUNT] = pd.to_numeric(working[_COUNT], errors="coerce").fillna(0)
    working = working[working[_COUNT] > 0]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no documents with spelling errors"))

    grouped = (
        working.groupby(_DOC, sort=False)
        .agg(
            TotalErrors=(_COUNT, "sum"),
            UniqueErrors=(_WORD, "nunique") if _WORD in working.columns else (_COUNT, "count"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["TotalErrors", _DOC], ascending=[False, True], kind="stable").head(top_n)

    labels = document_labels(grouped[_DOC].tolist())

    marks: list[PanelMark] = []
    for rank, (_, row) in enumerate(grouped.iterrows()):
        doc = str(row[_DOC])
        errs = int(row["TotalErrors"])
        unq = int(row["UniqueErrors"])
        doc_disp = labels.get(doc, doc)
        marks.append(
            PanelMark(
                key=doc,
                label=doc_disp,
                x=float(errs),
                y=float(rank),
                evidence=Evidence(
                    scope="rows",
                    filters=((_DOC, doc),),
                    count=errs,
                    describe=f"{doc_disp}: {errs} total spelling error(s) across {unq} distinct word(s)",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=SPELLCHECK_DOCUMENT_COUNTS.name,
        shape="ranked_bars",
        title=SPELLCHECK_DOCUMENT_COUNTS.title,
        subtitle=f"Top {len(marks)} documents by spelling error count",
        marks=tuple(marks),
        x_label="Total Errors",
        y_label="",
        provenance=provenance,
        data=grouped,
        notes=SPELLCHECK_DOCUMENT_COUNTS.notes,
    )
    return Result.success(prepared)


def spellcheck_confidence_scatter(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Scatter of suggestion confidence score vs error frequency."""
    label_top = int(params.get("label-top", 20))
    working = frame.copy()
    working[_WORD] = working[_WORD].astype(str).str.strip()
    working[_COUNT] = pd.to_numeric(working[_COUNT], errors="coerce").fillna(0)
    working[_SCORE] = pd.to_numeric(working[_SCORE], errors="coerce").fillna(0.0)
    working = working[working[_COUNT] > 0]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no spelling errors to plot"))

    # Aggregate by Word
    grouped = (
        working.groupby(_WORD, sort=False)
        .agg(
            TotalCount=(_COUNT, "sum"),
            Score=(_SCORE, "mean"),
            Suggestion=(_SUGGESTION, "first") if _SUGGESTION in working.columns else (_SCORE, lambda _: ""),
        )
        .reset_index()
    )

    # Sort to find top items to label
    sorted_top = grouped.sort_values("TotalCount", ascending=False).head(label_top)
    labelled_set = set(sorted_top[_WORD])

    marks: list[PanelMark] = []
    for _, row in grouped.iterrows():
        word = str(row[_WORD])
        cnt = int(row["TotalCount"])
        sc = float(row["Score"])
        sug = str(row["Suggestion"])
        marks.append(
            PanelMark(
                key=word,
                label=word,
                x=sc,
                y=math.log10(cnt) if cnt > 0 else 0.0,
                size=float(cnt),
                labelled=word in labelled_set,
                evidence=Evidence(
                    scope="terms",
                    filters=((_WORD, word),),
                    count=cnt,
                    phrase=word,
                    describe=f"'{word}': {cnt} count(s), score {sc:.2f} -> suggestion: '{sug}'",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=SPELLCHECK_CONFIDENCE_SCATTER.name,
        shape="scatter_labelled",
        title=SPELLCHECK_CONFIDENCE_SCATTER.title,
        subtitle=f"{len(marks)} unknown word(s) -- x: confidence score, y: log10 frequency",
        marks=tuple(marks),
        x_label="Suggestion Confidence Score",
        y_label="Log10 Total Occurrences",
        provenance=provenance,
        data=grouped,
        annotations=(
            Annotation(
                kind="vline",
                value=0.75,
                label="High confidence (>= 0.75)",
                note="Scores above 0.75 are typically straightforward typographical errors.",
            ),
        ),
        notes=SPELLCHECK_CONFIDENCE_SCATTER.notes,
    )
    return Result.success(prepared)


SPELLCHECK_TOP_ERRORS = PanelDefinition(
    name="spellcheck_top_errors",
    title="Most frequent spelling errors",
    question="Which misspelled or unknown words appear most frequently in the corpus?",
    tool="spellcheck",
    shape="ranked_bars",
    summary="Most frequent misspelled words across the corpus with their top suggestions.",
    requires=(_WORD, _COUNT, _SUGGESTION),
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=25,
            minimum=5,
            maximum=200,
            label="Errors to show",
            help="Number of top spelling errors to display.",
        ),
    ),
    build=spellcheck_top_errors,
    notes=(
        "Counts reflect raw occurrences of unknown words across the corpus.",
        "Unknown words may include proper names, technical terminology, or dialect rather than genuine errors.",
    ),
)

SPELLCHECK_DOCUMENT_COUNTS = PanelDefinition(
    name="spellcheck_document_counts",
    title="Spelling errors by document",
    question="Which documents in the corpus contain the highest number of spelling errors or unknown words?",
    tool="spellcheck",
    shape="ranked_bars",
    summary="Total spelling error count per document across the corpus.",
    requires=(_DOC, _COUNT),
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=25,
            minimum=5,
            maximum=200,
            label="Documents to show",
            help="Number of documents to display.",
        ),
    ),
    build=spellcheck_document_counts,
    notes=(
        "Longer documents naturally accumulate more unknown words; check raw counts against document length.",
        "Unrecognized words reflect absence from the reference wordlist rather than definitive spelling mistakes.",
    ),
)

SPELLCHECK_CONFIDENCE_SCATTER = PanelDefinition(
    name="spellcheck_confidence_scatter",
    title="Error frequency vs. suggestion confidence",
    question="Which spelling errors have high-confidence automatic fixes versus low-confidence anomalies?",
    tool="spellcheck",
    shape="scatter_labelled",
    summary="Suggestion match score against log frequency for unknown words.",
    requires=(_WORD, _COUNT, _SCORE),
    params=(
        PanelParam(
            name="label-top",
            type="int",
            default=20,
            minimum=0,
            maximum=100,
            label="Words to label",
            help="How many words get their name written on the plot.",
        ),
    ),
    build=spellcheck_confidence_scatter,
    notes=(
        "Score reflects character similarity / edit distance to the dictionary suggestion.",
        "High-frequency words with low confidence scores typically indicate domain vocabulary or names not in the dictionary.",
    ),
)

SPELLCHECK_PANELS: tuple[PanelDefinition, ...] = (
    SPELLCHECK_TOP_ERRORS,
    SPELLCHECK_DOCUMENT_COUNTS,
    SPELLCHECK_CONFIDENCE_SCATTER,
)
