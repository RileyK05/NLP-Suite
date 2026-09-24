"""Deverbal nominalization detection (FR-2.7).

Ports the method of legacy ``nominalization_util.py`` (WordNet
derivational morphology with a derivation-direction length constraint)
onto the shared parse table:

- nouns only (Penn ``NN*`` or Universal ``NOUN``/``PROPN``), first
  qualifying verb sense wins;
- a noun is a nominalization iff WordNet derives it from a verb that is
  strictly shorter (``destruction`` <- ``destroy``; ``table`` -> ``tabulate``
  and ``quality`` -> ``qualify`` are rejected, as in the legacy);
- the legacy ``check_ending`` suffix prefilter (ent/ing/ion/ance/ence)
  is a parameter (default on), with an optional curated wordlist whose
  members bypass the filter — e.g. the legacy
  ``nominalized-verbs-list.csv`` irregulars like ``attack``;
- deadjectival derivations (``happiness`` <- ``happy``) are not counted:
  the phenomenon here is deverbal only, as in the legacy.

Intentional differences from the legacy (all defect-grade, none silent):

- no Stanza re-parse per sentence and no Tkinter popups: the caller
  supplies the shared parse frame;
- per-surface-word memoization replaces the legacy ``true_word`` /
  ``false_word`` lists (same semantics, no quadratic rescan);
- the by-sentence frame lists only sentences containing at least one
  nominalization (the legacy's zero-rows carried no information);
- the nominalized-words column is honestly named (the legacy header
  ``Nouns/Nominalized Verbs`` held only the nominalizations).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import string
from typing import Any, Protocol

import pandas as pd

from core.analysis.postags import is_noun_tag
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "NOMINAL_SUFFIXES",
    "NltkNominalization",
    "NominalizationBackend",
    "default_wordnet_module",
    "detect_nominalizations",
    "load_curated_words",
    "sentence_frequency",
]

_NOM_COLUMNS: tuple[str, ...] = ("Word", "Base Verb", "Document ID", "Document", "Sentence ID")
_SENT_COLUMNS: tuple[str, ...] = (
    "Words in Sentence",
    "Nominalized Words",
    "Nominalizations in Sentence",
    "Nominalization %",
    "Sentence ID",
    "Sentence",
    "Document ID",
    "Document",
)

#: Legacy ``check_word_for_nominalization`` suffixes: word[-3:] in
#: {ent, ing, ion} or word[-4:] in {ance, ence}.
NOMINAL_SUFFIXES: tuple[str, ...] = ("ent", "ing", "ion", "ance", "ence")


def _is_noun(pos: str) -> bool:
    return is_noun_tag(pos)


def _passes_ending(word: str, curated: frozenset[str]) -> bool:
    """Legacy suffix prefilter: pass on a nominal suffix or curated membership."""
    if word in curated:
        return True
    return word[-3:] in ("ent", "ing", "ion") or word[-4:] in ("ance", "ence")


class NominalizationBackend(Protocol):
    """One question the traversal needs: which verb is this noun derived from?"""

    def base_verb(self, noun_lemma: str) -> str | None: ...


class _DerivationalForm(Protocol):
    def name(self) -> str: ...
    def synset_pos(self) -> str: ...


class NltkNominalization:
    """Production backend: NLTK WordNet derivational morphology.

    ``wordnet`` is the ``nltk.corpus.wordnet`` module (or a test double
    with ``NOUN`` and ``lemmas()``). Only verb forms strictly shorter
    than the noun qualify — the derivation-direction constraint.
    """

    def __init__(self, wordnet: Any) -> None:
        self._wn = wordnet

    def base_verb(self, noun_lemma: str) -> str | None:
        try:
            lemmas = self._wn.lemmas(noun_lemma, pos=self._wn.NOUN)
        except Exception:
            return None
        for lemma in lemmas:
            hit = _first_shorter_verb(lemma, noun_lemma)
            if hit is not None:
                return hit
        return None


def _form_name_and_pos(form: Any) -> tuple[str, str] | None:
    """(name, synset pos) for one derivationally related form, or None."""
    try:
        return str(form.name()), str(form.synset().pos())
    except Exception:
        return None


def _first_shorter_verb(lemma: Any, noun_lemma: str) -> str | None:
    """First verb form strictly shorter than the noun, or None."""
    try:
        forms = lemma.derivationally_related_forms()
    except Exception:
        return None
    for form in forms:
        read = _form_name_and_pos(form)
        if read is not None and read[1] == "v" and len(noun_lemma) > len(read[0]):
            return read[0]
    return None


def default_wordnet_module(data_dir: str | Path | None = None) -> Result[Any]:
    """Resolve the NLTK WordNet module, or fail with the install command."""
    try:
        import nltk
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "NOMINALIZATION_BACKEND_MISSING",
                "NLTK is not installed; nominalization detection needs the optional 'wordnet' extra",
                fix="python -m pip install nlp-suite-ng[wordnet]",
            )
        )
    try:
        if data_dir is not None:
            nltk.data.path.insert(0, str(data_dir))
        from nltk.corpus import wordnet as wn

        wn.synsets("dog", pos=wn.NOUN)
    except LookupError:
        target = str(Path(data_dir) / "corpora" / "wordnet") if data_dir is not None else "<NLTK_DATA>/corpora/wordnet"
        return Result.failure(
            Diagnostic.error(
                "NOMINALIZATION_DATA_MISSING",
                f"WordNet 3.0 corpus not found at {target}",
                fix="python -m nltk.downloader -d <NLTK_DATA> wordnet",
            )
        )
    return Result.success(wn)


def load_curated_words(path: str | Path) -> Result[frozenset[str]]:
    """Load a one-word-per-row curated list (first column) for the ending filter."""
    target = Path(path)
    if not target.is_file():
        return Result.failure(
            Diagnostic.error("NOMINALIZATION_LIST_NOT_FOUND", f"curated list not found at {target}", path=str(target))
        )
    try:
        frame = pd.read_csv(target, encoding="utf-8", on_bad_lines="skip")
    except (OSError, ValueError) as exc:
        return Result.failure(
            Diagnostic.error("NOMINALIZATION_LIST_UNREADABLE", f"cannot read curated list {target}: {exc}")
        )
    if frame.shape[1] < 1:
        return Result.failure(Diagnostic.error("NOMINALIZATION_LIST_EMPTY", f"curated list {target} has no columns"))
    words = frozenset(str(value).strip().lower() for value in frame.iloc[:, 0].tolist() if str(value).strip())
    return Result.success(words)


def _resolve(
    frame: pd.DataFrame, field: Col, backend: NominalizationBackend | None
) -> tuple[Col, NominalizationBackend | None, list[Diagnostic]]:
    if field not in (Col.FORM, Col.LEMMA):
        return (
            field,
            None,
            [Diagnostic.error("NOMINALIZATION_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}")],
        )
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return field, None, list(checked.diagnostics)
    for column in (field.value, Col.POS.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value):
        if column not in frame.columns:
            return (
                field,
                None,
                [Diagnostic.error("NOMINALIZATION_MISSING_COLUMN", f"parse table needs column {column!r}")],
            )
    if backend is not None:
        return field, backend, []
    resolved = default_wordnet_module()
    if resolved.value is None:
        return field, None, list(resolved.diagnostics)
    return field, NltkNominalization(resolved.unwrap()), list(resolved.diagnostics)


def _scan(
    frame: pd.DataFrame,
    field: Col,
    backend: NominalizationBackend,
    check_ending: bool,
    curated: frozenset[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """One pass: nominalization rows + per-sentence rows (sentences with hits only)."""
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    nom_rows: list[dict[str, object]] = []
    sent_rows: list[dict[str, object]] = []
    memo: dict[str, str | None] = {}
    for (doc_id, sent_id), group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        forms = [str(v) for v in group[Col.FORM.value].tolist()]
        sentence = " ".join(forms)
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        word_count = 0
        hits: list[tuple[str, str]] = []
        for _, row in group.iterrows():
            surface = str(row[Col.FORM.value])
            if not surface or surface in string.punctuation or surface[0] in ('"', "'", "`"):
                continue
            word_count += 1
            if not _is_noun(str(row[Col.POS.value])):
                continue
            word = surface.lower()
            lemma = str(row[field.value]).lower()
            # The legacy workflow uses the suffix list as a prefilter. Apply
            # it before the WordNet lookup so large corpora do not spend time
            # resolving derivations for nouns that cannot qualify. Curated
            # words still bypass this filter through _passes_ending().
            if check_ending and not _passes_ending(word, curated):
                continue
            if word not in memo:
                base = backend.base_verb(lemma) if lemma else None
                memo[word] = base
            base = memo[word]
            if base is not None:
                hits.append((word, base))
                nom_rows.append(
                    {
                        "Word": word,
                        "Base Verb": base,
                        "Document ID": doc_id,
                        "Document": doc,
                        "Sentence ID": sent_id,
                    }
                )
        if hits:
            words = "; ".join(word for word, _ in hits)
            sent_rows.append(
                {
                    "Words in Sentence": word_count,
                    "Nominalized Words": words,
                    "Nominalizations in Sentence": len(hits),
                    "Nominalization %": round(100.0 * len(hits) / word_count, 2) if word_count else 0.0,
                    "Sentence ID": sent_id,
                    "Sentence": sentence,
                    "Document ID": doc_id,
                    "Document": doc,
                }
            )
    return nom_rows, sent_rows


def detect_nominalizations(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    backend: NominalizationBackend | None = None,
    check_ending: bool = True,
    curated_words: Sequence[str] = (),
) -> Result[pd.DataFrame]:
    """One row per nominalized noun: Word, Base Verb, Document ID, Document, Sentence ID."""
    _, resolved, diags = _resolve(frame, field, backend)
    if resolved is None:
        return Result.failure(*diags)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=list(_NOM_COLUMNS)), *diags)
    curated = frozenset(w.strip().lower() for w in curated_words if str(w).strip())
    nom_rows, _ = _scan(frame, field, resolved, check_ending, curated)
    return Result.success(pd.DataFrame(nom_rows, columns=list(_NOM_COLUMNS)), *diags)


def sentence_frequency(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    backend: NominalizationBackend | None = None,
    check_ending: bool = True,
    curated_words: Sequence[str] = (),
) -> Result[pd.DataFrame]:
    """Per-sentence counts for sentences containing at least one nominalization."""
    _, resolved, diags = _resolve(frame, field, backend)
    if resolved is None:
        return Result.failure(*diags)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=list(_SENT_COLUMNS)), *diags)
    curated = frozenset(w.strip().lower() for w in curated_words if str(w).strip())
    _, sent_rows = _scan(frame, field, resolved, check_ending, curated)
    return Result.success(pd.DataFrame(sent_rows, columns=list(_SENT_COLUMNS)), *diags)
