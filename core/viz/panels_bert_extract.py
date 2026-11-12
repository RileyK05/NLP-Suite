"""BERT extractive summarization panels -- summary sentence positions across documents.

The bert_extract tool (``core/analysis/bert_extract.py``) uses transformer sentence
embeddings and centroid ranking to extract the most representative sentences from each
document.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelMark,
    PreparedPanel,
    Provenance,
)

__all__ = [
    "BERT_EXTRACT_PANELS",
    "BERT_EXTRACT_SENTENCE_POSITIONS",
    "bert_extract_sentence_positions",
]

_DOC = "Document"
_SENTENCE_IDS = "Sentence IDs"


def bert_extract_sentence_positions(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Positions of selected summary sentences across each document's text."""
    _ = params
    working = frame.copy()
    working[_DOC] = working[_DOC].astype(str).str.strip()
    working[_SENTENCE_IDS] = working[_SENTENCE_IDS].astype(str).str.strip()
    working = working[working[_SENTENCE_IDS] != ""]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no documents with summary sentences"))

    ordered_docs = sorted(working[_DOC].unique())
    labels = document_labels(ordered_docs)
    y_categories = tuple(labels[d] for d in ordered_docs)
    row_index = {doc: i for i, doc in enumerate(ordered_docs)}

    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        doc = str(row[_DOC])
        doc_disp = labels.get(doc, doc)
        row_idx = row_index[doc]
        raw_ids = re.findall(r"\d+", str(row[_SENTENCE_IDS]))
        for rank, sid_str in enumerate(raw_ids):
            sent_id = int(sid_str)
            marks.append(
                PanelMark(
                    key=f"{doc}␟{sent_id}␟{rank}",
                    label=f"Sentence {sent_id}",
                    x=float(sent_id),
                    y=float(row_idx),
                    evidence=Evidence(
                        scope="rows",
                        filters=((_DOC, doc),),
                        count=1,
                        describe=f"{doc_disp}, sentence {sent_id} selected in extractive summary (rank {rank + 1})",
                    ),
                )
            )

    if not marks:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no sentence IDs parsed from summary table"))

    prepared = PreparedPanel(
        panel=BERT_EXTRACT_SENTENCE_POSITIONS.name,
        shape="positions",
        title=BERT_EXTRACT_SENTENCE_POSITIONS.title,
        subtitle=f"{len(marks)} summary sentence(s) selected across {len(y_categories)} document(s)",
        marks=tuple(marks),
        x_label="Sentence Index in Document",
        y_label="Document",
        provenance=provenance,
        data=working,
        y_categories=y_categories,
        notes=BERT_EXTRACT_SENTENCE_POSITIONS.notes,
    )
    return Result.success(prepared)


BERT_EXTRACT_SENTENCE_POSITIONS = PanelDefinition(
    name="bert_extract_sentence_positions",
    title="Summary sentence positions in narrative",
    question="Where in each document do the extractive summary sentences fall (lead bias vs distributed throughout)?",
    tool="bert_extract",
    shape="positions",
    summary="Positions of centroid-selected summary sentences along each document's text flow.",
    requires=(_DOC, _SENTENCE_IDS),
    params=(),
    build=bert_extract_sentence_positions,
    notes=(
        "Extractive summarization often exhibits lead bias by favoring introductory sentences.",
        "Ticks show where each extracted sentence sits along the document's sentence progression.",
    ),
)

BERT_EXTRACT_PANELS: tuple[PanelDefinition, ...] = (BERT_EXTRACT_SENTENCE_POSITIONS,)
