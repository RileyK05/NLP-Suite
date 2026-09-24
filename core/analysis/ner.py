"""NER — entity timeline and location tracking."""

from __future__ import annotations

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["entity_timeline", "location_tracking"]

_LOCATION_TAGS = frozenset(["LOC", "GPE", "LOCATION", "FAC", "FACILITY"])


def _sid_key(sid: object) -> tuple[int, str, int]:
    """Order sentence ids numerically when possible, lexicographically otherwise.

    ``min(..., key=str)`` would report sentence 9 as the first of a
    12-sentence document, because "12" sorts before "9" as strings.
    """
    text = str(sid)
    try:
        return (0, "", int(text))
    except ValueError:
        return (1, text, 0)


def _is_entity(tag: str) -> bool:
    t = tag.strip()
    if not t or t in ("O", "_", "nan", "None"):
        return False
    # Strip BIO prefix
    if "-" in t:
        t = t.split("-", 1)[1]
    return t.upper() not in ("O",)


def _mentions_from_spans(g: pd.DataFrame) -> dict[tuple[str, str], list[object]]:
    """Whole-mention aggregation over one document's tokens.

    Review finding: grouping by (token text, tag) splits multiword mentions —
    "New York" became two entities. Mentions are spans: consecutive tokens of
    the same entity type within a sentence form ONE mention. BIO/BIOES
    boundaries are honored when present (B-/I-/E-/S- prefixes decide chunk
    edges); bare tags (spaCy convention) coalesce a maximal run of same-type
    tokens inside a sentence, but never across an ``O`` or a sentence break.
    """
    mentions: dict[tuple[str, str], list[object]] = {}
    if Col.RECORD_ID.value in g.columns:
        ordered = g.sort_values(Col.RECORD_ID.value, kind="stable")
    else:
        ordered = g
    sentence_col = Col.SENTENCE_ID.value
    current_text: list[str] = []
    current_tag = ""
    current_sids: list[object] = []
    current_sid: object = None
    current_sentence: object = None

    def flush() -> None:
        nonlocal current_text, current_tag, current_sids, current_sentence
        if current_text:
            key = (" ".join(current_text), current_tag)
            # One mention per span: the sentence it occurred in, once.
            mentions.setdefault(key, []).append(current_sids[0])
            current_text, current_tag, current_sids = [], "", []

    for form, ner, sid in zip(
        ordered[Col.FORM.value],
        ordered[Col.NER.value],
        ordered[Col.SENTENCE_ID.value],
        strict=False,
    ):
        raw = str(ner).strip()
        if not _is_entity(raw):
            flush()
            current_sentence = None
            continue
        norm = raw.split("-", 1)[1] if "-" in raw else raw
        norm = norm.upper()
        prefix = raw.split("-", 1)[0].upper() if "-" in raw else ""
        new_sentence = current_sentence is not None and str(sid) != str(current_sentence)
        if new_sentence:
            flush()
        boundary = (
            prefix in ("B", "S")  # explicit chunk start (BIO/BIOES)
            or (norm != current_tag and current_text)  # bare-tag type change
            or (prefix not in ("I", "E") and not current_text and norm != current_tag)  # chunk start
        )
        if current_text and (boundary or new_sentence):
            flush()
        if not current_text:
            current_sentence = sid
        current_text.append(str(form))
        current_tag = norm
        current_sids.append(sid)
        if prefix in ("E", "S"):
            flush()

    flush()
    return mentions


def entity_timeline(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """One row per (entity text, NER tag, document)."""
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    for need in [Col.FORM.value, Col.NER.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value]:
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("NER_MISSING_COLUMN", f"missing {need!r}"))
    if frame.empty:
        return Result.success(
            pd.DataFrame(
                columns=["Entity", "NER Tag", "Count", "First Sentence", "Last Sentence", "Document ID", "Document"]
            )
        )

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None

    # Collect per-document entity mentions (whole spans, not single tokens)
    rows: list[dict[str, object]] = []
    for doc_id, g in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(g[doc_col].iloc[0]) if doc_col is not None else ""
        mentions = _mentions_from_spans(g)
        for (entity, tag), sids in mentions.items():
            rows.append(
                {
                    "Entity": entity,
                    "NER Tag": tag,
                    "Count": len(sids),
                    "First Sentence": min(sids, key=_sid_key),
                    "Last Sentence": max(sids, key=_sid_key),
                    "Document ID": str(doc_id),
                    "Document": doc,
                }
            )

    if not rows:
        df = pd.DataFrame(
            columns=["Entity", "NER Tag", "Count", "First Sentence", "Last Sentence", "Document ID", "Document"]
        )
    else:
        df = pd.DataFrame(
            rows,
            columns=["Entity", "NER Tag", "Count", "First Sentence", "Last Sentence", "Document ID", "Document"],
        )
        df = df.sort_values(["Document ID", "Count"], ascending=[True, False]).reset_index(drop=True)
    return Result.success(df)


def location_tracking(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """Only location entities (LOC/GPE)."""
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    for need in [Col.FORM.value, Col.NER.value, Col.DOCUMENT_ID.value]:
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("NER_MISSING_COLUMN", f"missing {need!r}"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Location", "NER Tag", "Count", "Document ID", "Document"]))

    timeline = entity_timeline(frame)
    if not timeline.ok:
        return Result[pd.DataFrame](None, timeline.diagnostics)
    df = timeline.unwrap()
    if df.empty:
        return Result.success(pd.DataFrame(columns=["Location", "NER Tag", "Count", "Document ID", "Document"]))
    loc = df[df["NER Tag"].isin(_LOCATION_TAGS)].copy()
    if loc.empty:
        return Result.success(pd.DataFrame(columns=["Location", "NER Tag", "Count", "Document ID", "Document"]))
    # Rename Entity -> Location for clarity
    loc = loc.rename(columns={"Entity": "Location"})
    loc = loc[["Location", "NER Tag", "Count", "Document ID", "Document"]].reset_index(drop=True)
    return Result.success(loc)
