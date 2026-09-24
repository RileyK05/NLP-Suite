"""SVO compare — pairwise Jaccard over SVO sets per document."""

from __future__ import annotations

import itertools

import pandas as pd

from core.analysis.clause_svo import extract_svo
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["compare"]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return round(inter / union, 4) if union else 0.0


def compare(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """Pairwise SVO similarity per document.

    Uses :func:`core.analysis.clause_svo.extract_svo` to get triples, then
    groups by Document ID and computes Jaccard on subject, verb, object, and
    full triple sets.
    """
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if Col.DOCUMENT_ID.value not in frame.columns:
        return Result.failure(Diagnostic.error("SVO_COMPARE_MISSING_COLUMN", "missing Document ID"))

    if frame.empty:
        return Result.success(
            pd.DataFrame(
                columns=[
                    "Doc A",
                    "Doc B",
                    "Common Triples",
                    "Subject Jaccard",
                    "Verb Jaccard",
                    "Object Jaccard",
                    "Triple Jaccard",
                ]
            )
        )

    svo_result = extract_svo(frame)
    if not svo_result.ok:
        return Result[pd.DataFrame](None, svo_result.diagnostics)
    triples = svo_result.unwrap()
    if not triples:
        # No SVOs — emit one row per pair with zeros, or empty if <2 docs.
        docs = sorted({str(x) for x in frame[Col.DOCUMENT_ID.value].unique()})
        if len(docs) < 2:
            return Result.success(
                pd.DataFrame(
                    columns=[
                        "Doc A",
                        "Doc B",
                        "Common Triples",
                        "Subject Jaccard",
                        "Verb Jaccard",
                        "Object Jaccard",
                        "Triple Jaccard",
                    ]
                )
            )
        rows: list[dict[str, object]] = []
        for a, b in itertools.combinations(docs, 2):
            rows.append(
                {
                    "Doc A": a,
                    "Doc B": b,
                    "Common Triples": 0,
                    "Subject Jaccard": 0.0,
                    "Verb Jaccard": 0.0,
                    "Object Jaccard": 0.0,
                    "Triple Jaccard": 0.0,
                }
            )
        return Result.success(pd.DataFrame(rows))

    # Build per-doc sets
    doc_subjects: dict[str, set[str]] = {}
    doc_verbs: dict[str, set[str]] = {}
    doc_objects: dict[str, set[str]] = {}
    doc_triples: dict[str, set[str]] = {}

    for t in triples:
        did = str(t.document_id)
        doc_subjects.setdefault(did, set()).add(t.subject.lower())
        doc_verbs.setdefault(did, set()).add(t.verb.lower())
        doc_objects.setdefault(did, set()).add(t.obj.lower())
        doc_triples.setdefault(did, set()).add(f"{t.subject.lower()}|{t.verb.lower()}|{t.obj.lower()}")

    # Ensure every doc in frame appears even if it had no SVOs
    all_docs = sorted({str(x) for x in frame[Col.DOCUMENT_ID.value].unique()})
    for d in all_docs:
        doc_subjects.setdefault(d, set())
        doc_verbs.setdefault(d, set())
        doc_objects.setdefault(d, set())
        doc_triples.setdefault(d, set())

    if len(all_docs) < 2:
        return Result.success(
            pd.DataFrame(
                columns=[
                    "Doc A",
                    "Doc B",
                    "Common Triples",
                    "Subject Jaccard",
                    "Verb Jaccard",
                    "Object Jaccard",
                    "Triple Jaccard",
                ]
            )
        )

    rows = []
    for a, b in itertools.combinations(all_docs, 2):
        common = len(doc_triples[a] & doc_triples[b])
        rows.append(
            {
                "Doc A": a,
                "Doc B": b,
                "Common Triples": common,
                "Subject Jaccard": _jaccard(doc_subjects[a], doc_subjects[b]),
                "Verb Jaccard": _jaccard(doc_verbs[a], doc_verbs[b]),
                "Object Jaccard": _jaccard(doc_objects[a], doc_objects[b]),
                "Triple Jaccard": _jaccard(doc_triples[a], doc_triples[b]),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Doc A",
            "Doc B",
            "Common Triples",
            "Subject Jaccard",
            "Verb Jaccard",
            "Object Jaccard",
            "Triple Jaccard",
        ],
    )
    return Result.success(df)
