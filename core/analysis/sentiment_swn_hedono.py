"""Sentiment II — SentiWordNet (NLTK backend) and hedonometer (full asset).

SentiWordNet ports the legacy semantics: first sense only, universal
NOUN/ADJ/ADV scored (verbs and the rest are skipped, as in the legacy
filter), per-document mean of pos-minus-neg. The synset resolver is injectable so
unit tests use a hand-built map; production resolves through NLTK.

Hedonometer happiness is the mean over a full labMT asset
(``hedonometer.json``); the baked sample map is gone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path

import pandas as pd

from core.analysis.sentiment_vader_anew import _resolve_asset_path
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "hedonometer",
    "load_hedonometer_lexicon",
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

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        scored: list[tuple[float, float]] = []
        tokens = 0
        for _, row in group.iterrows():
            raw = row[field.value]
            if raw is None:
                continue
            try:
                if bool(pd.isna(raw)):
                    continue
            except (TypeError, ValueError):
                pass
            text = str(raw).strip().lower()
            if not text or not text.isalpha():
                continue
            tokens += 1
            pos = str(row[Col.POS.value]).strip().upper()
            if pos not in _SCORED_POS:
                continue
            hit = resolve(text, _UNIVERSAL_TO_WN[pos])
            if hit is not None:
                scored.append((hit[1], hit[2]))
        if scored:
            pos_mean = sum(p for p, _n in scored) / len(scored)
            neg_mean = sum(n for _p, n in scored) / len(scored)
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
                "Hits": len(scored),
                "Pos": round(pos_mean, 4),
                "Neg": round(neg_mean, 4),
                "Obj": round(obj, 4),
                "Net": round(net, 4),
            }
        )
    df = pd.DataFrame(rows, columns=["Document ID", "Document", "Tokens", "Hits", "Pos", "Neg", "Obj", "Net"])
    return Result.success(df, *diags)


def load_hedonometer_lexicon(path: str | Path) -> dict[str, float]:
    """Parse a labMT JSON asset ({"objects": [{"word", "happs"}]})."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    objects = payload.get("objects", []) if isinstance(payload, dict) else []
    lexicon: dict[str, float] = {}
    for entry in objects:
        try:
            lexicon[str(entry["word"]).strip().lower()] = float(entry["happs"])
        except (KeyError, TypeError, ValueError):
            continue
    return lexicon


def hedonometer(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    lexicon: Mapping[str, float] | str | Path | None = None,
) -> Result[pd.DataFrame]:
    """Per-document mean happiness over the full asset (5.0 when no hits)."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("HEDONO_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if field.value not in frame.columns or Col.DOCUMENT_ID.value not in frame.columns:
        return Result.failure(Diagnostic.error("HEDONO_MISSING_COLUMN", "missing field or Document ID"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Document ID", "Document", "Tokens", "Hits", "Happiness"]))
    table: Mapping[str, float] | None = lexicon if isinstance(lexicon, Mapping) else None
    diags: list[Diagnostic] = []
    if table is None:
        target, diags = _resolve_asset_path("hedonometer", lexicon)
        if target is None:
            return Result.failure(*diags)
        table = load_hedonometer_lexicon(target)

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        words = [str(x).lower() for x in group[field.value].tolist() if str(x).strip() and str(x) != "nan"]
        words = [w for w in words if w.isalpha()]
        vals = [table[w] for w in words if w in table]
        happy = round(sum(vals) / len(vals), 3) if vals else 5.0
        rows.append(
            {
                "Document ID": str(doc_id),
                "Document": doc,
                "Tokens": len(words),
                "Hits": len(vals),
                "Happiness": happy,
            }
        )
    df = pd.DataFrame(rows, columns=["Document ID", "Document", "Tokens", "Hits", "Happiness"])
    return Result.success(df, *diags)
