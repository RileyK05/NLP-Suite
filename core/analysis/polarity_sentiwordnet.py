"""Polarity II — SentiWordNet (NLTK backend).

SentiWordNet ports the legacy semantics: first sense only, universal
NOUN/ADJ/ADV scored (verbs and the rest are skipped, as in the legacy
filter), per-document mean of pos-minus-neg. The synset resolver is injectable
so unit tests use a hand-built map; production resolves through NLTK.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "nltk_data_path_insert",
    "sentiwordnet",
]

# (lemma, wordnet-pos) -> (synset name, pos_score, neg_score); None = no sense.
SwnResolver = Callable[[str, str], tuple[str, float, float] | None]

_UNIVERSAL_TO_WN = {"NOUN": "n", "ADJ": "a", "ADV": "r"}
_SCORED_POS = frozenset({"NOUN", "ADJ", "ADV"})


def nltk_data_path_insert(entry: str | Path) -> None:
    """Prepend an NLTK data root (tests point at scratch corpora)."""
    import nltk

    nltk.data.path.insert(0, str(entry))


def _nltk_resolver() -> tuple[SwnResolver | None, list[Diagnostic]]:
    try:
        import nltk
        from nltk.corpus import sentiwordnet as swn
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None, [
            Diagnostic.error(
                "SWN_BACKEND_MISSING",
                "NLTK is not installed; SentiWordNet needs the optional 'wordnet' extra",
                fix="python -m pip install nlp-suite-ng[wordnet]",
            )
        ]
    try:
        wn.synsets("dog", pos=wn.NOUN)
        swn.senti_synset("dog.n.01")
    except LookupError:
        return None, [
            Diagnostic.error(
                "SWN_DATA_MISSING",
                "wordnet and/or sentiwordnet corpora not found on nltk.data.path",
                fix="python -m nltk.downloader wordnet sentiwordnet",
            )
        ]

    def resolve(lemma: str, pos: str) -> tuple[str, float, float] | None:
        native = {"n": wn.NOUN, "a": wn.ADJ, "r": wn.ADV}[pos]
        senses = wn.synsets(lemma, pos=native)
        if not senses:
            return None
        scored = swn.senti_synset(senses[0].name())
        return (senses[0].name(), float(scored.pos_score()), float(scored.neg_score()))

    return resolve, []


def sentiwordnet(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    resolver: SwnResolver | None = None,
) -> Result[pd.DataFrame]:
    """Per-document SentiWordNet means (first sense, NOUN/ADJ/ADV only)."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("SWN_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if (
        field.value not in frame.columns
        or Col.DOCUMENT_ID.value not in frame.columns
        or Col.POS.value not in frame.columns
    ):
        return Result.failure(Diagnostic.error("SWN_MISSING_COLUMN", "missing field, Document ID, or POS"))
    if frame.empty:
        return Result.success(
            pd.DataFrame(columns=["Document ID", "Document", "Tokens", "Hits", "Pos", "Neg", "Obj", "Net"])
        )
    resolve = resolver
    diags: list[Diagnostic] = []
    if resolve is None:
        resolve, diags = _nltk_resolver()
        if resolve is None:
            return Result.failure(*diags)

    from core.conll.frames import col_str_list

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    doc_ids = frame[Col.DOCUMENT_ID.value].tolist()
    doc_names = col_str_list(frame, doc_col) if doc_col is not None else [""] * len(frame)
    raw_fields = frame[field.value].tolist()
    pos_list = [v.strip().upper() for v in col_str_list(frame, Col.POS)]

    doc_buckets: dict[object, list[int]] = {}
    for idx, did in enumerate(doc_ids):
        doc_buckets.setdefault(did, []).append(idx)

    swn_memo: dict[tuple[str, str], tuple[str, float, float] | None] = {}
    rows: list[dict[str, object]] = []
    for doc_id, indices in doc_buckets.items():
        doc = doc_names[indices[0]] if doc_col is not None else ""
        pos_sum = 0.0
        neg_sum = 0.0
        hits = 0
        tokens = 0
        for idx in indices:
            raw = raw_fields[idx]
            if raw is None or raw is pd.NA:
                continue
            if isinstance(raw, float) and math.isnan(raw):
                continue
            text = str(raw).strip().lower()
            if not text or not text.isalpha():
                continue
            tokens += 1
            pos = pos_list[idx]
            wn_pos = _UNIVERSAL_TO_WN.get(pos)
            if wn_pos is None:
                continue
            mkey = (text, wn_pos)
            if mkey in swn_memo:
                hit = swn_memo[mkey]
            else:
                hit = resolve(text, wn_pos)
                swn_memo[mkey] = hit
            if hit is not None:
                pos_sum += hit[1]
                neg_sum += hit[2]
                hits += 1
        if hits:
            pos_mean = pos_sum / hits
            neg_mean = neg_sum / hits
            obj = 1.0 - pos_mean - neg_mean
            net = pos_mean - neg_mean
        else:
            pos_mean = neg_mean = net = 0.0
            obj = 1.0 if tokens else 0.0
        rows.append(
            {
                "Document ID": str(doc_id),
                "Document": doc,
                "Tokens": tokens,
                "Hits": hits,
                "Pos": round(pos_mean, 4),
                "Neg": round(neg_mean, 4),
                "Obj": round(obj, 4),
                "Net": round(net, 4),
            }
        )
    df = pd.DataFrame(rows, columns=["Document ID", "Document", "Tokens", "Hits", "Pos", "Neg", "Obj", "Net"])
    return Result.success(df, *diags)
