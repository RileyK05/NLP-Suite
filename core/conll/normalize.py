"""Normalize any parser's CoNLL output to the canonical table.

Ported from the legacy ``CoNLL_util.universal_to_penn`` and
``normalize_to_canonical`` — the parts of the old CoNLL stack the defect review
called correct. The rewrite changes three things:

1. It operates on a DataFrame addressed **by column name** (R9), not on
   positional indices that silently shift when a parser adds a column.
2. It returns ``Result[DataFrame]`` rather than raising, printing, and showing
   a messagebox before returning ``None`` (R4, R7).
3. Tagset conversion happens **once**, here, and the resulting tagset is
   stamped on the sidecar. The legacy analyzer converted in memory and then
   re-read the raw file, filtering Universal tags with Penn patterns — which is
   how it produced silently empty outputs.
"""

from __future__ import annotations

import pandas as pd

from core.conll.schema import Col, PosTagset, canonical_columns, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "UNIVERSAL_POS",
    "detect_source_tagset",
    "normalize_table",
    "parse_feats",
    "pos_column",
    "universal_to_penn",
]

UNIVERSAL_POS: frozenset[str] = frozenset(
    {
        "NOUN",
        "PROPN",
        "VERB",
        "AUX",
        "ADJ",
        "ADV",
        "ADP",
        "DET",
        "PRON",
        "CCONJ",
        "SCONJ",
        "CONJ",
        "NUM",
        "PART",
        "INTJ",
        "SYM",
        "X",
        "PUNCT",
    }
)


def parse_feats(feats: object) -> dict[str, str]:
    """Parse a CoNLL-U feats string (``Number=Plur|Tense=Past``) into a dict.

    ``''``, ``'_'`` and ``None`` all yield ``{}``.
    """
    if feats is None:
        return {}
    text = str(feats)
    if text in ("", "_", "nan"):
        return {}
    parsed: dict[str, str] = {}
    for pair in text.split("|"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def universal_to_penn(pos: object, feats: object = "") -> str:
    """Map one Universal POS tag (+ optional feats) to the closest Penn tag.

    Tags that are not Universal — already Penn, or unknown — pass through
    unchanged, so the function is idempotent and safe to apply to mixed tables.
    """
    tag = "" if pos is None else str(pos)
    if tag not in UNIVERSAL_POS:
        return tag

    f = parse_feats(feats)
    number = f.get("Number", "")

    if tag == "NOUN":
        return "NNS" if number == "Plur" else "NN"
    if tag == "PROPN":
        return "NNPS" if number == "Plur" else "NNP"
    if tag in ("VERB", "AUX"):
        verb_form = f.get("VerbForm", "")
        tense = f.get("Tense", "")
        if verb_form == "Ger" or (verb_form == "Part" and tense == "Pres"):
            return "VBG"
        if verb_form == "Part" and tense == "Past":
            return "VBN"
        if verb_form == "Inf":
            return "VB"
        if verb_form == "Fin":
            if tense == "Past":
                return "VBD"
            if f.get("Person", "") == "3" and number == "Sing":
                return "VBZ"
            return "VBP"
        return "VB"
    if tag == "ADJ":
        return {"Cmp": "JJR", "Sup": "JJS"}.get(f.get("Degree", ""), "JJ")
    if tag == "ADV":
        return {"Cmp": "RBR", "Sup": "RBS"}.get(f.get("Degree", ""), "RB")
    if tag == "ADP":
        return "IN"
    if tag == "DET":
        return "DT"
    if tag == "PRON":
        pron_type = f.get("PronType", "")
        poss = f.get("Poss", "")
        if pron_type in ("Int", "Rel"):
            return "WP$" if poss == "Yes" else "WP"
        return "PRP$" if poss == "Yes" else "PRP"
    if tag in ("CCONJ", "CONJ"):
        return "CC"
    if tag == "SCONJ":
        return "IN"
    if tag == "NUM":
        return "CD"
    if tag == "PART":
        return "RP"
    if tag == "INTJ":
        return "UH"
    if tag == "PUNCT":
        return "."
    return tag


def detect_source_tagset(columns: list[str]) -> PosTagset:
    """Heuristic: which tagset a raw parser table is carrying.

    Only for tables arriving straight from a parser with no sidecar. Stanza and
    spaCy emit CoNLL-U ``feats`` and Universal tags; CoreNLP emits Penn. Once a
    table has been normalized the sidecar is authoritative — don't guess.
    """
    return "universal" if "feats" in columns else "penn"


def pos_column(frame: pd.DataFrame) -> pd.Series:
    """The POS column reached by name (R9). Raises KeyError if absent."""
    return frame[Col.POS.value]


def normalize_table(
    frame: pd.DataFrame,
    *,
    source_tagset: PosTagset | None = None,
) -> Result[pd.DataFrame]:
    """Reorder and retag a parser's CoNLL output into the canonical table.

    * Converts Universal POS to Penn while the ``feats`` column is still
      available, so number/tense/degree survive the mapping.
    * Fills auto-generated columns (``Deps``, ``Clause Tag``) with empty values.
    * Assigns a dense 1..N ``Record ID`` when the parser did not supply one.
    * Moves non-canonical columns to the end rather than dropping them.

    Returns diagnostics for everything it had to invent, so a downstream reader
    can tell "the parser said so" from "we filled this in".
    """
    columns = [str(c) for c in frame.columns]
    checked = validate_columns(columns)
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)

    diagnostics: list[Diagnostic] = list(checked.diagnostics)
    tagset = source_tagset if source_tagset is not None else detect_source_tagset(columns)
    work = frame.copy()

    pos_name = Col.POS.value
    if tagset == "universal" and pos_name in work.columns:
        feats_values = work["feats"].tolist() if "feats" in work.columns else ["" for _ in range(len(work))]
        before = [str(v) for v in pos_column(work).tolist()]
        after = [universal_to_penn(tag, feats) for tag, feats in zip(before, feats_values, strict=True)]
        changed = sum(1 for old, new in zip(before, after, strict=True) if old != new)
        if changed:
            work[pos_name] = after
            diagnostics.append(
                Diagnostic.info(
                    "POS_TAGSET_CONVERTED",
                    f"converted {changed} Universal POS tag(s) to Penn",
                    converted=changed,
                )
            )
    elif tagset == "universal":
        diagnostics.append(
            Diagnostic.warning(
                "POS_CONVERSION_SKIPPED",
                "source tagset is Universal but the table has no POS column",
            )
        )

    canonical = canonical_columns()

    # validate_columns has already rejected any table missing a required
    # column, so the only columns that can still be absent here are the
    # auto-generated ones — filling them is normal operation, not an event.
    for column in canonical:
        if column not in work.columns:
            work[column] = ""

    record_id_name = Col.RECORD_ID.value
    if work[record_id_name].isna().all() or (work[record_id_name].astype(str).str.strip() == "").all():
        work[record_id_name] = [str(i) for i in range(1, len(work) + 1)]
        diagnostics.append(Diagnostic.info("RECORD_ID_ASSIGNED", f"assigned dense Record ID 1..{len(work)}"))

    extras = [c for c in work.columns if str(c) not in set(canonical)]
    ordered = [c for c in canonical if c in work.columns] + extras
    return Result.success(work[ordered], *diagnostics)
