"""Knowledge graphs — DBpedia/YAGO/Wikipedia stub."""

from __future__ import annotations

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["build"]

# Tiny baked knowledge base for tests and offline demo.
_KB: dict[str, list[tuple[str, str]]] = {
    "paris": [("isA", "City"), ("country", "France"), ("type", "GPE")],
    "london": [("isA", "City"), ("country", "United Kingdom"), ("type", "GPE")],
    "barack obama": [("isA", "Person"), ("role", "President"), ("country", "USA")],
    "obama": [("isA", "Person"), ("role", "President")],
    "einstein": [("isA", "Person"), ("field", "Physics"), ("type", "PERSON")],
    "google": [("isA", "Organization"), ("field", "Technology")],
}


def build(
    frame: pd.DataFrame,
    *,
    field: Col = Col.FORM,
    source: str = "stub",
) -> Result[pd.DataFrame]:
    """Emit (subject, predicate, object, source) triples for known entities.

    The legacy queried DBpedia/YAGO/Wikipedia at runtime. This offline stub
    returns triples for entities in the baked KB and an INFO diagnostic for
    unknown entities; it never touches the network.
    """
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("KG_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if field.value not in frame.columns:
        return Result.failure(Diagnostic.error("KG_MISSING_COLUMN", f"missing {field.value!r}"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Subject", "Predicate", "Object", "Source"]))

    # Collect unique entity strings lower-cased, but preserve first Form for subject label
    seen: dict[str, str] = {}
    for form in frame[field.value].tolist():
        txt = str(form).strip()
        if not txt or txt.lower() in ("nan", "none"):
            continue
        key = txt.lower()
        # For multi-word entities the CoNLL gives one token each; the caller
        # should have supplied already-merged entities via NER timeline. For a
        # plain CoNLL we just check unigrams against the KB.
        seen.setdefault(key, txt)

    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    for key, label in sorted(seen.items()):
        triples = _KB.get(key)
        if triples is None:
            diags.append(Diagnostic.info("KG_UNKNOWN_ENTITY", f"{label!r} not in stub KB", entity=label, source=source))
            continue
        for pred, obj in triples:
            rows.append({"Subject": label, "Predicate": pred, "Object": obj, "Source": source})

    if not rows:
        df = pd.DataFrame(columns=["Subject", "Predicate", "Object", "Source"])
    else:
        df = pd.DataFrame(rows, columns=["Subject", "Predicate", "Object", "Source"])
        df = df.sort_values(["Subject", "Predicate"]).reset_index(drop=True)

    if diags:
        return Result.success(df, *diags)
    return Result.success(df)
