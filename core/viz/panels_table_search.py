"""Table search panels -- hit frequencies and hit positions across documents.

The table_search tool (``core/analysis/search.py`` / ``tools/table_search.py``) filters
CoNLL parse tables and finds occurrences matching linguistic queries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = [
    "TABLE_SEARCH_HITS_BY_DOCUMENT",
    "TABLE_SEARCH_HIT_POSITIONS",
    "TABLE_SEARCH_PANELS",
    "table_search_hit_positions",
    "table_search_hits_by_document",
]

_DOC = "Document"
_SENTENCE_ID = "Sentence ID"


def table_search_hits_by_document(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Ranked bars of search matches per document."""
    top_n = int(params.get("top-n", 25))
    working = frame.copy()
    working[_DOC] = working[_DOC].astype(str).str.strip()
    working = working[working[_DOC] != ""]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no search hits to display"))

    counts = working.groupby(_DOC, sort=False).size().reset_index(name="Hits")
    counts = counts.sort_values(["Hits", _DOC], ascending=[False, True], kind="stable").head(top_n)

    labels = document_labels(counts[_DOC].tolist())

    marks: list[PanelMark] = []
    for rank, (_, row) in enumerate(counts.iterrows()):
        doc = str(row[_DOC])
        hits = int(row["Hits"])
        doc_disp = labels.get(doc, doc)
        marks.append(
            PanelMark(
                key=doc,
                label=doc_disp,
                x=float(hits),
                y=float(rank),
                evidence=Evidence(
                    scope="rows",
                    filters=((_DOC, doc),),
                    count=hits,
                    describe=f"{doc_disp}: {hits} search match(es)",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=TABLE_SEARCH_HITS_BY_DOCUMENT.name,
        shape="ranked_bars",
        title=TABLE_SEARCH_HITS_BY_DOCUMENT.title,
        subtitle=f"Top {len(marks)} documents by number of search matches",
        marks=tuple(marks),
        x_label="Search Matches",
        y_label="",
        provenance=provenance,
        data=counts,
        notes=TABLE_SEARCH_HITS_BY_DOCUMENT.notes,
    )
    return Result.success(prepared)


def table_search_hit_positions(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Tick per search match along each document's sentence sequence."""
    _ = params
    working = frame.copy()
    working[_DOC] = working[_DOC].astype(str).str.strip()
    working[_SENTENCE_ID] = pd.to_numeric(working[_SENTENCE_ID], errors="coerce")
    working = working.dropna(subset=[_SENTENCE_ID])
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no search matches with valid sentence IDs"))

    ordered_docs = sorted(working[_DOC].unique())
    labels = document_labels(ordered_docs)
    y_categories = tuple(labels[d] for d in ordered_docs)
    row_index = {doc: i for i, doc in enumerate(ordered_docs)}

    marks: list[PanelMark] = []
    for idx, (_, row) in enumerate(working.iterrows()):
        doc = str(row[_DOC])
        doc_disp = labels.get(doc, doc)
        sent_id = int(row[_SENTENCE_ID])
        row_idx = row_index[doc]
        marks.append(
            PanelMark(
                key=f"{doc}␟{sent_id}␟{idx}",
                label=f"Sentence {sent_id}",
                x=float(sent_id),
                y=float(row_idx),
                evidence=Evidence(
                    scope="rows",
                    filters=((_DOC, doc), (_SENTENCE_ID, str(sent_id))),
                    count=1,
                    describe=f"{doc_disp}, sentence {sent_id}: search match",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=TABLE_SEARCH_HIT_POSITIONS.name,
        shape="positions",
        title=TABLE_SEARCH_HIT_POSITIONS.title,
        subtitle=f"{len(marks)} match(es) across {len(y_categories)} document(s)",
        marks=tuple(marks),
        x_label="Sentence Index in Document",
        y_label="Document",
        provenance=provenance,
        data=working,
        y_categories=y_categories,
        notes=TABLE_SEARCH_HIT_POSITIONS.notes,
    )
    return Result.success(prepared)


TABLE_SEARCH_HITS_BY_DOCUMENT = PanelDefinition(
    name="table_search_hits_by_document",
    title="Search hits by document",
    question="Which documents contain the most search matches?",
    tool="table_search",
    shape="ranked_bars",
    summary="Total search hit count per document across the corpus.",
    requires=(_DOC, _SENTENCE_ID),
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
    build=table_search_hits_by_document,
    notes=(
        "Raw search counts naturally correlate with document length; check long speeches against word totals.",
        "Each mark links directly to the search matches in that document.",
    ),
)

TABLE_SEARCH_HIT_POSITIONS = PanelDefinition(
    name="table_search_hit_positions",
    title="Search hit positions across documents",
    question="Where do search hits occur within the structure of each document?",
    tool="table_search",
    shape="positions",
    summary="Positions of matching search hits across document sentences.",
    requires=(_DOC, _SENTENCE_ID),
    params=(),
    build=table_search_hit_positions,
    notes=(
        "Ticks mark the sentence number where a search query hit occurs in each document.",
        "Clusters of ticks indicate passages or sections with concentrated discussion of the queried term.",
    ),
)

TABLE_SEARCH_PANELS: tuple[PanelDefinition, ...] = (
    TABLE_SEARCH_HITS_BY_DOCUMENT,
    TABLE_SEARCH_HIT_POSITIONS,
)
