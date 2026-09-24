"""Verb modality, tense and voice (FR-5.10) over the canonical CoNLL frame.

Week 8 of the syllabus, one table per verb token: "Verb modality: Ability,
possibility, permission, and obligation. Verb tense: past, future, gerundive.
Verb voice: Active and passive verb forms."

Works with whatever morphology the parse actually carries and degrades with a
WARNING when it does not:

- **Tense** reads ``Tense=``/``VerbForm=`` from the ``feats`` column (CoNLL-U,
  kept as a non-canonical extra) when present; else the Penn POS tag
  (VBD past, VBG gerund, VBN participle, VBZ/VBP present, VB infinitive); else
  the word's ending (-ed past, -ing gerund). FUTURE is decided first and is
  lexical: a ``will``/``shall`` auxiliary governing the verb (its ``Head`` is
  the verb) or standing immediately before it in the same sentence. Values:
  past | present | future | gerund | infinitive | participle | unknown.
- **Voice** is passive when the verb's own DEPREL or one of its dependents'
  DEPREL carries a passive marker (``nsubj:pass``, ``nsubjpass``,
  ``aux:pass``, ``auxpass``, and the ``csubj``/``acl`` variants); degraded
  (empty DEPREL) a ``be``-lemma auxiliary immediately before a past
  participle also counts. Values: active | passive | unknown.
- **Modality** comes from the modal auxiliary governing or immediately
  preceding the verb: can/could -> ability; may/might -> possibility; must,
  should, ought -> obligation; will/shall/would -> none (bare future or
  conditional, so the tense column carries the signal); dare/need -> unknown.
  NOTE: "may" can encode PERMISSION as well as possibility, and one column
  cannot tell the two readings apart — the modal word itself is always kept in
  the ``Modal`` column so the ambiguity stays visible. Values: ability |
  possibility | obligation | none | unknown. ``Modal`` names the modal word.

``analyze_verbs`` keeps every verb row and every column whatever ``analysis``
says. ``analysis`` only chooses which COUNT columns ``summarize_verbs``
populates; the others are left at 0, so one table contract serves all four
facets (documented here because the alternative — dropping rows or columns per
facet — breaks any downstream comparison).

A verb token is anything tagged VERB/AUX (Universal) or VB* (Penn), so
modals and auxiliaries are rows too when the tagset is Universal.
"""

from __future__ import annotations

import pandas as pd

from core.conll.normalize import parse_feats
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "ANALYSIS_CHOICES",
    "SUMMARY_COLUMNS",
    "VERB_COLUMNS",
    "analyze_verbs",
    "summarize_verbs",
]

VERB_COLUMNS = ["Document", "Document ID", "Sentence ID", "Verb", "Lemma", "Tense", "Modality", "Modal", "Voice"]
SUMMARY_COLUMNS = [
    "Document",
    "Document ID",
    "Verbs",
    "Past",
    "Present",
    "Future",
    "Gerund",
    "Passive",
    "Active",
    "Ability",
    "Possibility",
    "Obligation",
    "No modality",
]
ANALYSIS_CHOICES = ("all", "modality", "tense", "voice")

_FUTURE_MODALS = frozenset({"will", "shall"})
_MODAL_MAP: dict[str, str] = {
    "can": "ability",
    "could": "ability",
    "may": "possibility",
    "might": "possibility",
    "must": "obligation",
    "should": "obligation",
    "ought": "obligation",
    "will": "none",
    "shall": "none",
    "would": "none",
    "dare": "unknown",
    "need": "unknown",
}
_PASSIVE_MARKERS = ("nsubj:pass", "nsubjpass", "aux:pass", "auxpass", "csubj:pass", "csubjpass", "acl:pass", "aclpass")
_BE_FORMS = frozenset({"be", "am", "is", "are", "was", "were", "been", "being"})
_POS_TENSE = {
    "VBD": "past",
    "VBG": "gerund",
    "VBN": "participle",
    "VBZ": "present",
    "VBP": "present",
    "VB": "infinitive",
}
_SUMMARY_FACETS: dict[str, tuple[str, ...]] = {
    "tense": ("Past", "Present", "Future", "Gerund"),
    "voice": ("Passive", "Active"),
    "modality": ("Ability", "Possibility", "Obligation", "No modality"),
}


def _is_verb(pos: str) -> bool:
    return pos in ("VERB", "AUX") or pos.startswith("VB")


def _key(value: object) -> str:
    """ID/Head join key — tolerant of float-cast ids ("3.0" vs "3")."""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _morph_column(frame: pd.DataFrame) -> str | None:
    for name in ("feats", "FEATS", "Morph", "morph", "Morphology"):
        if name in frame.columns:
            return name
    return None


def _word(row: pd.Series) -> str:
    return str(row[Col.FORM.value]).strip().lower()


def _lemma(row: pd.Series) -> str:
    raw = row[Col.LEMMA.value] if Col.LEMMA.value in row.index else ""
    text = "" if raw is None else str(raw).strip().lower()
    return text or _word(row)


def _deprel(row: pd.Series) -> str:
    return str(row[Col.DEPREL.value]).strip().lower()


def _tense_from_morph(feats: dict[str, str]) -> str:
    verb_form = feats.get("VerbForm", "")
    if verb_form == "Ger":
        return "gerund"
    if verb_form == "Part":
        return "participle"
    if verb_form == "Inf":
        return "infinitive"
    if feats.get("Tense", "") == "Past":
        return "past"
    if feats.get("Tense", "") == "Present":
        return "present"
    return ""


def _tense_from_form(form: str) -> str:
    lower = form.lower()
    if lower.endswith("ed"):
        return "past"
    if lower.endswith("ing"):
        return "gerund"
    return ""


def analyze_verbs(frame: pd.DataFrame, *, analysis: str = "all") -> Result[pd.DataFrame]:
    """Per-verb tense, modality and voice rows from a canonical CoNLL frame.

    Empty input yields an empty frame with the declared columns. ``analysis``
    is validated here (``VERB_BAD_ANALYSIS``) but every row and column is
    always produced; it only steers :func:`summarize_verbs`.
    """
    if analysis not in ANALYSIS_CHOICES:
        return Result.failure(
            Diagnostic.error(
                "VERB_BAD_ANALYSIS",
                f"analysis must be one of {list(ANALYSIS_CHOICES)}, got {analysis!r}",
                analysis=analysis,
            )
        )
    for need in (Col.FORM.value, Col.LEMMA.value, Col.POS.value, Col.DEPREL.value, Col.HEAD.value, Col.ID.value):
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("VERB_MISSING_COLUMN", f"missing {need!r}", missing=need))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=VERB_COLUMNS))

    diags: list[Diagnostic] = []
    morph_col = _morph_column(frame)
    if morph_col is None:
        diags.append(
            Diagnostic.warning(
                "VERB_NO_MORPHOLOGY",
                "no feats/Morph column in this parse; tense falls back to POS tags and word endings",
            )
        )

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for (doc_id, sent_id), sent in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        tokens = [row for _, row in sent.iterrows()]
        name = str(tokens[0][doc_col]) if doc_col is not None else ""
        for position, row in enumerate(tokens):
            pos = str(row[Col.POS.value]).strip().upper()
            if not _is_verb(pos):
                continue
            form = str(row[Col.FORM.value]).strip()
            lemma = _lemma(row)
            feats = parse_feats(row[morph_col]) if morph_col is not None else {}
            dependents = [
                other for other in tokens if _key(other[Col.HEAD.value]) == _key(row[Col.ID.value]) and other is not row
            ]
            preceding = tokens[position - 1] if position > 0 else None
            nearby = [*dependents, *([preceding] if preceding is not None else [])]

            # Future first: a will/shall aux overrides whatever morphology says
            # about the lexical verb it governs ("will have walked" is future).
            future_modal = ""
            for other in nearby:
                word = _word(other)
                if word in _FUTURE_MODALS or _lemma(other) in _FUTURE_MODALS:
                    future_modal = word
                    break
            modal_word = ""
            modality = "none"
            for other in nearby:
                word = _word(other)
                lemma_word = _lemma(other)
                for candidate in (word, lemma_word):
                    if candidate in _MODAL_MAP:
                        modal_word = word or candidate
                        modality = _MODAL_MAP[candidate]
                        break
                if modal_word:
                    break

            if future_modal:
                tense = "future"
            else:
                tense = _tense_from_morph(feats) or _POS_TENSE.get(pos, "") or _tense_from_form(form) or "unknown"

            own_rel = _deprel(row)
            passive = any(marker in own_rel for marker in _PASSIVE_MARKERS) or any(
                marker in _deprel(other) for marker in _PASSIVE_MARKERS for other in dependents
            )
            # Degraded passive cue: "was eaten" with no usable DEPREL.
            if (
                not passive
                and tense == "participle"
                and preceding is not None
                and (_lemma(preceding) in _BE_FORMS or _word(preceding) in _BE_FORMS)
            ):
                passive = True
            if passive:
                voice = "passive"
            elif any(_deprel(other) for other in tokens) or pos:
                voice = "active"
            else:
                voice = "unknown"

            rows.append(
                {
                    "Document": name,
                    "Document ID": str(doc_id),
                    "Sentence ID": sent_id,
                    "Verb": form,
                    "Lemma": lemma,
                    "Tense": tense,
                    "Modality": modality,
                    "Modal": modal_word,
                    "Voice": voice,
                }
            )
    return Result.success(pd.DataFrame(rows, columns=VERB_COLUMNS), *diags)


def summarize_verbs(annotated: pd.DataFrame, *, analysis: str = "all") -> Result[pd.DataFrame]:
    """Per-document verb counts; ``analysis`` zeroes the count columns outside its facet.

    The count column contract is fixed (infinitive/participle verbs count in
    ``Verbs`` but have no dedicated column here). ``analysis`` defaults to
    "all", so the single-argument call fills everything.
    """
    if analysis not in ANALYSIS_CHOICES:
        return Result.failure(
            Diagnostic.error(
                "VERB_BAD_ANALYSIS",
                f"analysis must be one of {list(ANALYSIS_CHOICES)}, got {analysis!r}",
                analysis=analysis,
            )
        )
    if annotated.empty or "Tense" not in annotated.columns:
        return Result.success(pd.DataFrame(columns=SUMMARY_COLUMNS))
    populated = (
        {column for columns in _SUMMARY_FACETS.values() for column in columns}
        if analysis == "all"
        else set(_SUMMARY_FACETS[analysis])
    )
    rows: list[dict[str, object]] = []
    for (doc_id, name), group in annotated.groupby(["Document ID", "Document"], sort=False):
        tense = group["Tense"].astype(str)
        voice = group["Voice"].astype(str)
        modality = group["Modality"].astype(str)
        counts = {
            "Verbs": len(group),
            "Past": int((tense == "past").sum()),
            "Present": int((tense == "present").sum()),
            "Future": int((tense == "future").sum()),
            "Gerund": int((tense == "gerund").sum()),
            "Passive": int((voice == "passive").sum()),
            "Active": int((voice == "active").sum()),
            "Ability": int((modality == "ability").sum()),
            "Possibility": int((modality == "possibility").sum()),
            "Obligation": int((modality == "obligation").sum()),
            "No modality": int((modality == "none").sum()),
        }
        for column in SUMMARY_COLUMNS[2:]:
            if column != "Verbs" and column not in populated:
                counts[column] = 0
        rows.append({"Document": name, "Document ID": doc_id, **counts})
    return Result.success(pd.DataFrame(rows, columns=SUMMARY_COLUMNS))
