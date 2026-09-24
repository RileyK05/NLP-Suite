"""Location extraction — place spans from a parsed corpus (HW4 GIS pipeline).

The NER column answers "which tokens are entities"; a pin map needs *places*:
whole multi-word spans (``New York``, not ``New`` and ``York``) with the
document provenance the map tools join on. Span rules follow
:mod:`core.analysis.ner` exactly (BIO prefixes honored, bare tags coalesced
per sentence), reusing its ``_mentions_from_spans`` so the two tools can never
disagree on where a mention begins and ends.

Why this is not just ``ner.location_tracking``: that table is one row per
(document, entity). Geocoding wants the corpus-wide place list ranked by how
often it is mentioned; the SVO map wants per-document provenance. Both views
live here so neither tool re-derives span boundaries.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.analysis.ner import _mentions_from_spans
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["extract_locations", "locations_by_document"]

#: Tolerant across tagsets: CoreNLP GPE/LOCATION/FACILITY/CITY/COUNTRY/STATE,
#: spaCy GPE/LOC/FAC, and the BIO-prefixed variants ``_mentions_from_spans``
#: has already stripped and uppercased.
_LOCATION_TAGS = frozenset(
    {"LOC", "GPE", "LOCATION", "FAC", "FACILITY", "CITY", "COUNTRY", "STATE", "STATE_OR_PROVINCE", "PROVINCE"}
)

_EXTRACT_COLUMNS = ["Place", "Mentions", "Documents", "First document"]
_BY_DOC_COLUMNS = ["Document", "Document ID", "Place", "Mentions"]


def _check_frame(frame: pd.DataFrame) -> Diagnostic | None:
    """CoNLL validation shared by both views (one failure shape)."""
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return checked.diagnostics[0]
    for need in (Col.FORM.value, Col.NER.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value):
        if need not in frame.columns:
            return Diagnostic.error("LOCS_MISSING_COLUMN", f"missing {need!r}")
    return None


def _location_mentions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One record per (document, place span), in first-appearance order.

    Documents appear in the order the frame first mentions them, places in the
    order each document first names them (``_mentions_from_spans`` preserves
    span order). That order is the "first appearance" tie-break downstream.
    """
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else str(doc_id)
        for (text, tag), sids in _mentions_from_spans(group).items():
            if tag not in _LOCATION_TAGS:
                continue
            rows.append(
                {
                    "Place": text,
                    "Mentions": len(sids),
                    "Document ID": str(doc_id),
                    "Document": doc,
                }
            )
    return rows


def _min_count_diagnostic(dropped: int, min_count: int) -> list[Diagnostic]:
    if not dropped:
        return []
    return [
        Diagnostic.info(
            "LOCS_MIN_COUNT_DROPPED",
            f"{dropped} place(s) with fewer than {min_count} mention(s) were dropped",
            count=dropped,
            min_count=min_count,
        )
    ]


def extract_locations(frame: pd.DataFrame, *, min_count: int = 1) -> Result[pd.DataFrame]:
    """Corpus-wide place aggregate (Place, Mentions, Documents, First document).

    ``Mentions`` counts whole spans, ``Documents`` counts distinct documents,
    ``First document`` names the document where the place first appears (its
    provenance anchor; the per-document breakdown is
    :func:`locations_by_document`). Rows sort by Mentions descending, ties by
    first appearance. ``min_count`` is a corpus-wide floor on Mentions and
    reports how many rare places it dropped.
    """
    if min_count < 1:
        return Result.failure(Diagnostic.error("LOCS_BAD_PARAM", f"min_count must be >=1, got {min_count}"))
    problem = _check_frame(frame)
    if problem is not None:
        return Result[pd.DataFrame](None, (problem,))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_EXTRACT_COLUMNS))

    aggregate: dict[str, dict[str, Any]] = {}
    for row in _location_mentions(frame):
        place = str(row["Place"])
        entry = aggregate.get(place)
        if entry is None:
            aggregate[place] = {
                "Place": place,
                "Mentions": int(row["Mentions"]),
                "docs": {str(row["Document ID"])},
                "First document": str(row["Document"]),
            }
        else:
            entry["Mentions"] = int(entry["Mentions"]) + int(row["Mentions"])
            entry["docs"].add(str(row["Document ID"]))

    # dict order is first appearance, and sorted() is stable: equal Mention
    # counts keep their first-appearance order as the tie-break.
    ordered = sorted(aggregate.values(), key=lambda entry: -int(entry["Mentions"]))
    kept = [entry for entry in ordered if int(entry["Mentions"]) >= min_count]
    out = pd.DataFrame(
        [
            {
                "Place": entry["Place"],
                "Mentions": entry["Mentions"],
                "Documents": len(entry["docs"]),
                "First document": entry["First document"],
            }
            for entry in kept
        ],
        columns=_EXTRACT_COLUMNS,
    )
    return Result.success(out, *_min_count_diagnostic(len(ordered) - len(kept), min_count))


def locations_by_document(frame: pd.DataFrame, *, min_count: int = 1) -> Result[pd.DataFrame]:
    """Per-document place mentions (Document, Document ID, Place, Mentions).

    The provenance table the SVO map joins on. ``min_count`` is the same
    corpus-wide floor as :func:`extract_locations`, so both views agree on
    which places exist.
    """
    if min_count < 1:
        return Result.failure(Diagnostic.error("LOCS_BAD_PARAM", f"min_count must be >=1, got {min_count}"))
    problem = _check_frame(frame)
    if problem is not None:
        return Result[pd.DataFrame](None, (problem,))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_BY_DOC_COLUMNS))

    rows = _location_mentions(frame)
    totals: dict[str, int] = {}
    for row in rows:
        place = str(row["Place"])
        totals[place] = totals.get(place, 0) + int(row["Mentions"])
    kept = [row for row in rows if totals[str(row["Place"])] >= min_count]
    out = pd.DataFrame(
        [
            {
                "Document": row["Document"],
                "Document ID": row["Document ID"],
                "Place": row["Place"],
                "Mentions": row["Mentions"],
            }
            for row in kept
        ],
        columns=_BY_DOC_COLUMNS,
    )
    return Result.success(out, *_min_count_diagnostic(len(rows) - len(kept), min_count))
