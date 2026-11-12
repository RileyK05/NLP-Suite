"""K-sentences panels -- bookend repetitions and opening vs closing content density.

The k_sentences tool (``core/analysis/k_sentences.py``) examines the first K and last K
sentences of documents to uncover framing, rhetorical bookends, and opening vs closing
structure.
"""

from __future__ import annotations

from collections.abc import Mapping
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

__all__ = [
    "K_SENTENCES_PANELS",
    "K_SENTENCES_SECTION_DENSITY",
    "K_SENTENCES_TOP_REPETITIONS",
    "k_sentences_section_density",
    "k_sentences_top_repetitions",
]

_LEMMA = "Lemma"
_DOC_FREQ = "Doc Frequency"
_DOC_ID = "Document ID"
_SECTION = "Section"
_TOKENS = "Tokens"
_CONTENT_TOKENS = "Content Tokens"


def k_sentences_top_repetitions(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Ranked bars of lemmas that repeat across first-K and last-K sentences."""
    top_n = int(params.get("top-n", 25))
    working = frame.copy()
    working[_LEMMA] = working[_LEMMA].astype(str).str.strip().str.lower()
    working[_DOC_FREQ] = pd.to_numeric(working[_DOC_FREQ], errors="coerce").fillna(0)
    working = working[working[_DOC_FREQ] > 0]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no repeated lemmas with positive document frequency"))

    grouped = (
        working.groupby(_LEMMA, sort=False)
        .agg(
            DocFrequency=(_DOC_FREQ, "max"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["DocFrequency", _LEMMA], ascending=[False, True], kind="stable").head(top_n)

    marks: list[PanelMark] = []
    for rank, (_, row) in enumerate(grouped.iterrows()):
        lemma = str(row[_LEMMA])
        freq = int(row["DocFrequency"])
        marks.append(
            PanelMark(
                key=lemma,
                label=lemma,
                x=float(freq),
                y=float(rank),
                evidence=Evidence(
                    scope="terms",
                    filters=((_LEMMA, lemma),),
                    count=freq,
                    phrase=lemma,
                    describe=f"'{lemma}': appears in both opening and closing K-sentences across {freq} document(s)",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=K_SENTENCES_TOP_REPETITIONS.name,
        shape="ranked_bars",
        title=K_SENTENCES_TOP_REPETITIONS.title,
        subtitle=f"Top {len(marks)} content words repeated between first-K and last-K sentences",
        marks=tuple(marks),
        x_label="Document Frequency",
        y_label="",
        provenance=provenance,
        data=grouped,
        notes=K_SENTENCES_TOP_REPETITIONS.notes,
    )
    return Result.success(prepared)


def k_sentences_section_density(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Distribution of content word density in opening (first-K) vs closing (last-K) sentences."""
    _ = params
    working = frame.copy()
    working[_SECTION] = working[_SECTION].astype(str).str.strip().str.lower()
    working[_TOKENS] = pd.to_numeric(working[_TOKENS], errors="coerce").fillna(0)
    working[_CONTENT_TOKENS] = pd.to_numeric(working[_CONTENT_TOKENS], errors="coerce").fillna(0)
    working = working[working[_TOKENS] > 0]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no section rows with tokens"))

    working["_density"] = working[_CONTENT_TOKENS] / working[_TOKENS]
    sections = [s for s in ("first", "last") if s in working[_SECTION].unique()]
    if not sections:
        sections = sorted(working[_SECTION].unique())

    y_categories = tuple(
        f"Opening ({s})" if s == "first" else (f"Closing ({s})" if s == "last" else s) for s in sections
    )
    section_to_idx = {s: i for i, s in enumerate(sections)}

    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        sec = str(row[_SECTION])
        if sec not in section_to_idx:
            continue
        row_idx = section_to_idx[sec]
        ratio = float(row["_density"])
        doc_id = str(row.get(_DOC_ID, ""))
        tokens = int(row[_TOKENS])
        content = int(row[_CONTENT_TOKENS])
        marks.append(
            PanelMark(
                key=f"{doc_id}:{sec}",
                label=f"{doc_id} ({sec})",
                x=ratio,
                y=float(row_idx),
                evidence=Evidence(
                    scope="rows",
                    filters=((_DOC_ID, doc_id), (_SECTION, sec)),
                    count=content,
                    describe=f"{doc_id} ({sec} K-sentences): {content}/{tokens} content tokens ({ratio:.1%})",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=K_SENTENCES_SECTION_DENSITY.name,
        shape="distribution",
        title=K_SENTENCES_SECTION_DENSITY.title,
        subtitle="Distribution of content word proportions: opening vs closing sentences across documents",
        marks=tuple(marks),
        x_label="Content Token Ratio (Content / Total Tokens)",
        y_label="Section",
        provenance=provenance,
        data=working,
        y_categories=y_categories,
        notes=K_SENTENCES_SECTION_DENSITY.notes,
    )
    return Result.success(prepared)


K_SENTENCES_TOP_REPETITIONS = PanelDefinition(
    name="k_sentences_top_repetitions",
    title="Bookend repetitions (first-K vs last-K sentences)",
    question="Which content words recur between the opening and closing sentences of documents across the corpus?",
    tool="k_sentences",
    shape="ranked_bars",
    summary="Content words appearing in both first-K and last-K sentences, ranked by document frequency.",
    requires=(_LEMMA, _DOC_FREQ),
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=25,
            minimum=5,
            maximum=200,
            label="Lemmas to show",
            help="Number of repeated lemmas to display.",
        ),
    ),
    build=k_sentences_top_repetitions,
    notes=(
        "Repetitions reflect content words that appear in both the initial and final K-sentence windows of the same document.",
        "High-frequency bookends often signal key rhetorical themes, deliberate framing, or structural echoes.",
    ),
)

K_SENTENCES_SECTION_DENSITY = PanelDefinition(
    name="k_sentences_section_density",
    title="Content word density: opening vs closing sentences",
    question="How does content word concentration compare between opening (first-K) and closing (last-K) sentences?",
    tool="k_sentences",
    shape="distribution",
    summary="Distribution of content word proportions across documents in opening vs closing sections.",
    requires=(_DOC_ID, _SECTION, _TOKENS, _CONTENT_TOKENS),
    params=(),
    build=k_sentences_section_density,
    notes=(
        "Content words include nouns, verbs, adjectives, and adverbs, excluding grammatical function words.",
        "Differences in density show whether authors open or close with denser informational content or rhetorical formulas.",
    ),
)

K_SENTENCES_PANELS: tuple[PanelDefinition, ...] = (
    K_SENTENCES_TOP_REPETITIONS,
    K_SENTENCES_SECTION_DENSITY,
)
