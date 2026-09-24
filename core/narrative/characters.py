"""Character mentions + emotion arcs (FR-6.2).

Characters are PERSON entities from the canonical NER column: consecutive
PERSON tokens in one sentence form one mention span (span boundaries do
not survive canonicalization, so adjacency is the documented
reconstruction). Mentions group by case-insensitive name.

``character_arcs`` joins each mention's sentence to its VADER compound
(the same sentence scores as ``vader_sentences``): one row per
(character, sentence). Sentence-binning + HTML are the documented
remainder, not silent behavior.
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from core.analysis.sentiment_vader_anew import vader_sentences
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["character_arcs", "character_mentions"]

_MENTION_COLUMNS = ["Character", "Mentions", "First Sentence", "Last Sentence", "Document ID", "Document"]
_ARC_COLUMNS = ["Character", "Sentence ID", "Compound", "Sentence", "Document ID", "Document"]


class _Span(TypedDict):
    tokens: list[str]
    key: tuple[object, object]
    doc: str


class _Mention(TypedDict):
    name: str
    count: int
    first: object
    last: object
    doc: str


def _check(frame: pd.DataFrame) -> list[Diagnostic]:
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return list(checked.diagnostics)
    for column in (Col.FORM.value, Col.NER.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value):
        if column not in frame.columns:
            return [Diagnostic.error("CHARACTER_MISSING_COLUMN", f"parse table needs column {column!r}")]
    return []


def _spans(frame: pd.DataFrame) -> list[_Span]:
    """One mention span per run of consecutive PERSON tokens (per sentence)."""
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    spans: list[_Span] = []
    open_tokens: list[str] = []
    open_key: tuple[object, object] | None = None
    open_doc = ""
    for (doc_id, sent_id), group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        for _, row in group.iterrows():
            if str(row[Col.NER.value]).upper() == "PERSON":
                open_tokens.append(str(row[Col.FORM.value]))
                open_key = (doc_id, sent_id)
                open_doc = doc
            elif open_tokens and open_key is not None:
                spans.append(_Span(tokens=list(open_tokens), key=open_key, doc=open_doc))
                open_tokens = []
                open_key = None
        if open_tokens and open_key is not None:
            spans.append(_Span(tokens=list(open_tokens), key=open_key, doc=open_doc))
            open_tokens = []
            open_key = None
    return spans


def character_mentions(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """One row per character: name, mention count, first/last sentence, document."""
    diags = _check(frame)
    if diags:
        return Result.failure(*diags)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_MENTION_COLUMNS))
    grouped: dict[tuple[str, object, str], _Mention] = {}
    for span in _spans(frame):
        name = " ".join(span["tokens"])
        doc_id, sent_id = span["key"]
        entry_key = (name.lower(), doc_id, span["doc"])
        entry = grouped.setdefault(
            entry_key, _Mention(name=name, count=0, first=sent_id, last=sent_id, doc=span["doc"])
        )
        entry["count"] += 1
        entry["last"] = sent_id
    rows = [
        {
            "Character": entry["name"],
            "Mentions": entry["count"],
            "First Sentence": entry["first"],
            "Last Sentence": entry["last"],
            "Document ID": doc_id,
            "Document": entry["doc"],
        }
        for (_norm, doc_id, _doc), entry in sorted(grouped.items(), key=lambda kv: str(kv[0]))
    ]
    return Result.success(pd.DataFrame(rows, columns=_MENTION_COLUMNS))


def character_arcs(
    frame: pd.DataFrame,
    *,
    field: Col = Col.FORM,
    analyzer: object = None,
) -> Result[pd.DataFrame]:
    """VADER compound of every sentence containing each character."""
    diags = _check(frame)
    if diags:
        return Result.failure(*diags)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_ARC_COLUMNS))
    scored = vader_sentences(frame, field=field, analyzer=analyzer)
    if scored.value is None:
        return Result.failure(*scored.diagnostics)
    # Join keys normalized to str: groupby yields numpy scalars while the
    # sentence frame may hold Python ints or strings per column dtype.
    compounds = {
        (str(row["Document ID"]), str(row["Sentence ID"])): (row["Compound"], row["Sentence"])
        for _, row in scored.unwrap().iterrows()
    }
    seen: set[tuple[str, object, object]] = set()
    display: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for span in _spans(frame):
        surface = " ".join(span["tokens"])
        doc_id, sent_id = span["key"]
        # First-surface-wins display name, matching character_mentions.
        name = display.setdefault(surface.lower(), surface)
        if (name.lower(), doc_id, sent_id) in seen:
            continue
        seen.add((name.lower(), doc_id, sent_id))
        hit = compounds.get((str(doc_id), str(sent_id)))
        if hit is None:
            continue
        compound, sentence = hit
        rows.append(
            {
                "Character": name,
                "Sentence ID": sent_id,
                "Compound": compound,
                "Sentence": sentence,
                "Document ID": doc_id,
                "Document": str(span["doc"]),
            }
        )
    ordered = sorted(rows, key=lambda r: (str(r["Document ID"]), str(r["Character"]), str(r["Sentence ID"])))
    return Result.success(pd.DataFrame(ordered, columns=_ARC_COLUMNS), *scored.diagnostics)
