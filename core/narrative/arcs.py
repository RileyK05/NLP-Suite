"""Narrative & arcs — emotion arcs and sentence-length arcs."""

from __future__ import annotations

import pandas as pd

from core.analysis.polarity_vader import vader_sentences
from core.conll.schema import Col, validate_columns
from core.result import Result

__all__ = ["emotion_arc", "length_arc"]


def emotion_arc(frame: pd.DataFrame, *, field: Col = Col.FORM) -> Result[pd.DataFrame]:
    """Per-sentence VADER compound as an emotion arc."""
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Document ID", "Sentence ID", "Compound", "Sentence"]))

    # One call scores every sentence with one analyzer: building VADER's
    # analyzer parses its lexicon, so doing it per sentence cost ~9 ms each
    # (docs/internal/PERFORMANCE.md's 331 s narrative row). Sentences come back
    # in the same (Document ID, Sentence ID) order this groupby walks.
    scored = vader_sentences(frame, field=field)
    if scored.value is None:
        # Say *why* the arc could not be drawn rather than emitting 0.0 points.
        return Result[pd.DataFrame](None, scored.diagnostics)
    compounds = scored.unwrap()["Compound"].tolist()
    rows: list[dict[str, object]] = []
    groups = frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False)
    for ((did, sid), g), comp in zip(groups, compounds, strict=True):
        sent_text = " ".join(str(x) for x in g[Col.FORM.value].tolist())
        rows.append(
            {
                "Document ID": str(did),
                "Sentence ID": sid,
                "Compound": round(float(comp), 4) + 0.0,  # + 0.0: a neutral sentence reads 0.0, never -0.0
                "Sentence": sent_text,
            }
        )
    df = pd.DataFrame(rows, columns=["Document ID", "Sentence ID", "Compound", "Sentence"])
    return Result.success(df, *scored.diagnostics)


def length_arc(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """Per-sentence token length arc."""
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Document ID", "Sentence ID", "Tokens", "Sentence"]))
    rows: list[dict[str, object]] = []
    for (did, sid), g in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        toks = g[Col.FORM.value].tolist()
        rows.append(
            {
                "Document ID": str(did),
                "Sentence ID": sid,
                "Tokens": len(toks),
                "Sentence": " ".join(str(x) for x in toks),
            }
        )
    df = pd.DataFrame(rows, columns=["Document ID", "Sentence ID", "Tokens", "Sentence"])
    # groupby(sort=False) already yields rows in first-appearance order, which
    # is the true document order; sorting the string Document ID here would
    # scramble any corpus past nine documents ("10" < "2" lexicographically).
    return Result.success(df)
