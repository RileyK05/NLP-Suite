"""Narrative & arcs — emotion arcs and sentence-length arcs."""

from __future__ import annotations

import pandas as pd

from core.analysis.sentiment_vader_anew import vader
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["emotion_arc", "length_arc"]


def emotion_arc(frame: pd.DataFrame, *, field: Col = Col.FORM) -> Result[pd.DataFrame]:
    """Per-sentence VADER compound as an emotion arc."""
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Document ID", "Sentence ID", "Compound", "Sentence"]))

    # Build per-sentence subframes and score each with vader
    rows: list[dict[str, object]] = []
    failures: list[Diagnostic] = []
    for (did, sid), g in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        # Vader groups by Document ID; give the one-sentence slice a synthetic
        # document id so it is scored as a unit, then map back.
        sub = g.copy()
        res = vader(sub, field=field)
        if res.ok:
            vdf = res.unwrap()
            comp = float(vdf["Compound"].iloc[0]) if not vdf.empty else 0.0
        else:
            # Propagate *why* the sentence could not be scored rather than
            # silently emitting a 0.0 arc point.
            comp = 0.0
            failures.extend(res.diagnostics)
        # Reconstruct sentence text
        sent_text = " ".join(str(x) for x in g[Col.FORM.value].tolist())
        rows.append(
            {
                "Document ID": str(did),
                "Sentence ID": sid,
                "Compound": round(comp, 4),
                "Sentence": sent_text,
            }
        )
    if any(d.severity.value == "ERROR" for d in failures):
        return Result[pd.DataFrame](None, tuple(failures))
    df = pd.DataFrame(rows, columns=["Document ID", "Sentence ID", "Compound", "Sentence"])
    return Result.success(df, *failures)


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
