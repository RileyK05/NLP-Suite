"""Name-gender annotation from a choice of four legacy dictionaries (FR-4.8).

CoreNLP's gender annotator labels person names male/female from name lists.
The 446W syllabus's graded question is the COMPARISON: what do the US census,
Carnegie Mellon, NLTK and US Social Security lists each say about the same
corpus? So ``dictionary`` is a parameter, a missing list fails loudly (never
a silent substitution -- the MALLET pattern: the comparison is the point),
and every output row names the dictionary that produced it.

Intentional differences from the legacy (defect-grade, recorded):

- the legacy shipped four lists under ``lib\\namesGender``; here each is
  ``<names-dir>/<dictionary>/{male,female}.txt`` (one name per line, ``#``
  comments allowed), and only the NLTK list travels with a package;
- a name sitting in BOTH lists is gender ``both`` plus a WARNING diagnostic,
  not a silent winner;
- lookup is case-insensitive, and a multi-token name falls back to its first
  token (these lists index given names, so "John Smith" is judged by "John");
- coverage (how much of the document this dictionary knew) is reported per
  document, because dictionary reach is half of the comparison.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["annotate_names", "load_name_dictionary", "summarize_names"]

#: The four legacy lists, by folder name. Never substituted for one another.
DICTIONARIES: frozenset[str] = frozenset({"census", "carnegie_mellon", "nltk", "social_security"})

_NAME_COLUMNS = ["Document", "Document ID", "Name", "Gender", "Dictionary", "Mentions"]
_SUMMARY_COLUMNS = ["Document", "Document ID", "Male", "Female", "Both", "Unknown", "Coverage"]
_GENDERS: frozenset[str] = frozenset({"male", "female", "both", "unknown"})

_DICT_FIX = (
    "download the legacy name lists and place them as <names-dir>/census/male.txt + female.txt "
    "(same for carnegie_mellon, social_security); the nltk dictionary ships with 'pip install nltk' "
    "and a one-time nltk.download('names')"
)


def _read_name_file(path: Path) -> set[str]:
    """Case-folded names, one per line; blanks and ``#`` comments skipped."""
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        word = line.strip()
        if not word or word.startswith("#"):
            continue
        names.add(word.casefold())
    return names


def _load_from_dir(folder: Path, *, dictionary: str) -> Result[dict[str, set[str]]]:
    """``<folder>/male.txt`` + ``female.txt``; anything missing is loud."""
    male_path = folder / "male.txt"
    female_path = folder / "female.txt"
    if not male_path.is_file() or not female_path.is_file():
        return Result.failure(
            Diagnostic.error(
                "GENDER_DICT_MISSING",
                f"name lists not found under {folder} (need male.txt and female.txt)",
                fix=_DICT_FIX,
                dictionary=dictionary,
                path=str(folder),
            )
        )
    try:
        return Result.success({"male": _read_name_file(male_path), "female": _read_name_file(female_path)})
    except OSError as exc:
        return Result.failure(
            Diagnostic.error(
                "GENDER_DICT_MISSING",
                f"could not read name lists under {folder}: {exc}",
                fix=_DICT_FIX,
                dictionary=dictionary,
                path=str(folder),
            )
        )


def _load_from_nltk() -> Result[dict[str, set[str]]]:
    """The real NLTK ``names`` corpus (lazy import: optional backend)."""
    try:
        from nltk.corpus import names
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "GENDER_DICT_MISSING",
                "nltk is not installed; the nltk name dictionary needs it",
                fix=_DICT_FIX,
                dictionary="nltk",
            )
        )
    try:
        male_words = names.words("male.txt")
        female_words = names.words("female.txt")
    except LookupError as exc:
        # The corpus is downloaded separately from the package.
        return Result.failure(
            Diagnostic.error(
                "GENDER_DICT_MISSING",
                f"nltk 'names' corpus is not downloaded: {exc}",
                fix=_DICT_FIX,
                dictionary="nltk",
            )
        )
    return Result.success(
        {
            "male": {str(word).casefold() for word in male_words},
            "female": {str(word).casefold() for word in female_words},
        }
    )


def load_name_dictionary(dictionary: str, names_dir: str | Path | None = None) -> Result[dict[str, set[str]]]:
    """``{"male": ..., "female": ...}`` case-folded name sets for *dictionary*.

    Resolution order: (1) ``names_dir`` when given -- ``<names_dir>/<dictionary>/male.txt``
    and ``female.txt`` (one name per line, ``#`` comments allowed) and nothing
    else; (2) the NLTK ``names`` corpus when ``dictionary == "nltk"``. Every
    other case fails as ``GENDER_DICT_MISSING`` with the install pointer --
    one dictionary is never substituted for another, because comparing their
    answers is the whole point (the MALLET pattern).
    """
    key = str(dictionary).strip().lower()
    if names_dir is not None:
        return _load_from_dir(Path(names_dir) / key, dictionary=key)
    if key == "nltk":
        return _load_from_nltk()
    return Result.failure(
        Diagnostic.error(
            "GENDER_DICT_MISSING",
            f"name dictionary {dictionary!r} has no bundled lists; it needs a names dir",
            fix=_DICT_FIX,
            dictionary=key,
        )
    )


def _coerce_name_set(value: object) -> set[str] | None:
    """Case-fold an injectable name collection, or None when malformed."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return None
    return {str(word).casefold() for word in value}


def _resolve_names(
    dictionary: str,
    names_dir: str | Path | None,
    names: Mapping[str, object] | None,
) -> Result[dict[str, set[str]]]:
    """Injected override when given, else the resolved dictionary."""
    if names is None:
        return load_name_dictionary(dictionary, names_dir)
    male = _coerce_name_set(names.get("male"))
    female = _coerce_name_set(names.get("female"))
    if male is None or female is None:
        return Result.failure(
            Diagnostic.error(
                "GENDER_BAD_NAMES",
                "names override must be {'male': set[str], 'female': set[str]}",
                dictionary=str(dictionary),
            )
        )
    return Result.success({"male": male, "female": female})


def _entity_tag(raw: object) -> str:
    """NER tag without its BIO/BIOES prefix, uppercased."""
    text = str(raw).strip()
    if "-" in text:
        text = text.split("-", 1)[1]
    return text.upper()


def _ner_names(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(document_id, document, name) person mentions in reading order.

    Consecutive PERSON tokens in one sentence form one mention span: span
    boundaries do not survive canonicalization, so adjacency is the
    documented reconstruction (the core.narrative.characters rule).
    """
    out: list[tuple[str, str, str]] = []
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    open_tokens: list[str] = []
    open_doc_id = ""
    open_doc = ""
    open_sent: object = None

    def flush() -> None:
        nonlocal open_tokens, open_doc_id, open_doc, open_sent
        if open_tokens:
            out.append((open_doc_id, open_doc, " ".join(open_tokens)))
        open_tokens, open_doc_id, open_doc, open_sent = [], "", "", None

    for _, row in frame.iterrows():
        doc_id = str(row[Col.DOCUMENT_ID.value])
        doc = str(row[doc_col]) if doc_col is not None else ""
        sent = row[Col.SENTENCE_ID.value]
        form = str(row[Col.FORM.value]).strip()
        if _entity_tag(row[Col.NER.value]) != "PERSON" or not form:
            flush()
            continue
        if open_tokens and (doc_id != open_doc_id or str(sent) != str(open_sent)):
            flush()
        if not open_tokens:
            open_doc_id, open_doc, open_sent = doc_id, doc, sent
        open_tokens.append(form)
    flush()
    return out


def _propn_names(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(document_id, document, name) one per PROPN token (source="propn").

    Single tokens on purpose: this source exists for single-token given
    names, and joining "New York" into one candidate helps nobody.
    """
    out: list[tuple[str, str, str]] = []
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    for _, row in frame.iterrows():
        if str(row[Col.POS.value]).strip().upper() != "PROPN":
            continue
        form = str(row[Col.FORM.value]).strip()
        if not form:
            continue
        doc_id = str(row[Col.DOCUMENT_ID.value])
        doc = str(row[doc_col]) if doc_col is not None else ""
        out.append((doc_id, doc, form))
    return out


def _lookup_keys(name: str) -> tuple[str, ...]:
    """Case-folded lookup keys: the whole name, then its first token.

    The four lists index given names, so "John Smith" is judged by "John"
    when the full string is unknown to the dictionary.
    """
    key = name.strip().casefold()
    if not key:
        return ()
    if " " in key:
        return (key, key.split(" ", 1)[0])
    return (key,)


def _gender_of(name: str, male: set[str], female: set[str]) -> str:
    for key in _lookup_keys(name):
        in_male, in_female = key in male, key in female
        if in_male and in_female:
            return "both"
        if in_male:
            return "male"
        if in_female:
            return "female"
    return "unknown"


def _required_columns(source: str) -> tuple[str, ...]:
    if source == "propn":
        return (Col.FORM.value, Col.POS.value, Col.DOCUMENT_ID.value)
    return (Col.FORM.value, Col.NER.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value)


def annotate_names(
    frame: pd.DataFrame,
    *,
    dictionary: str = "nltk",
    names_dir: str | Path | None = None,
    source: str = "ner",
    names: Mapping[str, object] | None = None,
) -> Result[pd.DataFrame]:
    """Person names in *frame*, classified against one name dictionary.

    ``source="ner"`` keeps NER PERSON spans whole ("John Smith"); ``source="propn"``
    takes each PROPN token as one name. ``names`` injects ``{"male": ...,
    "female": ...}`` (the test seam) and is labelled in the output with
    ``dictionary``. Rows: Document, Document ID, Name, Gender (male|female|both|unknown),
    Dictionary, Mentions.
    """
    if source not in ("ner", "propn"):
        return Result.failure(
            Diagnostic.error(
                "GENDER_BAD_SOURCE", f"source must be 'ner' or 'propn', got {source!r}", source=str(source)
            )
        )
    for need in _required_columns(source):
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("GENDER_MISSING_COLUMN", f"missing {need!r}", missing=need))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_NAME_COLUMNS))
    resolved = _resolve_names(dictionary, names_dir, names)
    if resolved.value is None:
        return Result.failure(*resolved.diagnostics)
    male = resolved.value["male"]
    female = resolved.value["female"]

    occurrences = _propn_names(frame) if source == "propn" else _ner_names(frame)
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for doc_id, doc, name in occurrences:
        key = (doc_id, name.casefold())
        entry = aggregated.setdefault(key, {"document": doc, "name": name, "count": 0})
        entry["count"] = int(entry["count"]) + 1

    rows: list[dict[str, object]] = []
    both_names: list[str] = []
    for (doc_id, _key), entry in aggregated.items():
        surface = str(entry["name"])
        gender = _gender_of(surface, male, female)
        if gender == "both" and surface not in both_names:
            both_names.append(surface)
        rows.append(
            {
                "Document": entry["document"],
                "Document ID": doc_id,
                "Name": surface,
                "Gender": gender,
                "Dictionary": str(dictionary),
                "Mentions": int(entry["count"]),
            }
        )
    diags = [
        Diagnostic.warning(
            "GENDER_NAME_BOTH",
            f"{name!r} is in both the male and the female list of {dictionary!r}; counted as 'both'",
            name=name,
            dictionary=str(dictionary),
        )
        for name in both_names
    ]
    return Result.success(pd.DataFrame(rows, columns=_NAME_COLUMNS), *diags)


def summarize_names(annotated: pd.DataFrame) -> Result[pd.DataFrame]:
    """Per-document gender tally over the annotate_names output.

    Male/Female/Both/Unknown sum the Mentions column; Coverage is the share
    of mentions the dictionary resolved (male, female or both), rounded to
    4 dp and 0.0 for a document with no mentions.
    """
    for need in ("Document", "Document ID", "Gender", "Mentions"):
        if need not in annotated.columns:
            return Result.failure(Diagnostic.error("GENDER_MISSING_COLUMN", f"missing {need!r}", missing=need))
    if annotated.empty:
        return Result.success(pd.DataFrame(columns=_SUMMARY_COLUMNS))
    rows: list[dict[str, object]] = []
    grouped = annotated.groupby(["Document ID", "Document"], sort=False)
    for (doc_id, doc), group in grouped:
        counts = {"male": 0, "female": 0, "both": 0, "unknown": 0}
        for gender, mentions in zip(group["Gender"], group["Mentions"], strict=True):
            label = str(gender) if str(gender) in _GENDERS else "unknown"
            counts[label] += int(mentions)
        total = sum(counts.values())
        known = counts["male"] + counts["female"] + counts["both"]
        rows.append(
            {
                "Document": doc,
                "Document ID": doc_id,
                "Male": counts["male"],
                "Female": counts["female"],
                "Both": counts["both"],
                "Unknown": counts["unknown"],
                "Coverage": round(known / total, 4) if total else 0.0,
            }
        )
    return Result.success(pd.DataFrame(rows, columns=_SUMMARY_COLUMNS))
