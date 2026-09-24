"""Author-gender guesser from writing style (FR-2.10) — transparent reimplementation.

Classroom-native equivalent of the HackerFactor GenderGuesser method
(http://www.hackerfactor.com/GenderGuesser.php#About; the original JavaScript
can be run for evidence via ``scripts/run_gender_guesser.py``). That tool scores
WEAK WORDS — function words such as prepositions, articles and pronouns — on a
male/female writing-style axis after Koppel, Argamon & Shimoni (2002) and the
Gender Genie word categories. This module reimplements that METHOD with its own
small, documented word-category lexicon. It does NOT copy the HackerFactor or
Gender Genie dictionaries (those are proprietary and license-restricted), and
the labels here are PREDICTIONS ABOUT WRITING STYLE, not facts about people.
Collaborative and presidential texts are not writer ground truth: a speech is
often drafted by many hands, quotations carry their author's style, and a
copy-editor can shift every score.

Coding intuition per category (the axis used throughout this module):
the informal/involved register scores toward "male" and the formal register
scores toward "female" ("informal male, formal female" per the course's coding
of Koppel et al. 2002); pronouns and prepositions carry most of the signal,
which is why they are separate visible categories rather than one hidden bag.
Every category is its own ``CATEGORIES`` entry so a reader can audit or
re-weight the sums:

- ``prepositions`` — prepositions; the formal family (carry the signal).
- ``articles`` — a/an/the; the formal family.
- ``numbers`` — number words and dozen/hundred/...; the formal family.
- ``certainty`` — never/really/must/always/...; forceful, unhedged register,
  formal family.
- ``pronouns_i``, ``pronouns_we``, ``pronouns_you`` — first/second person;
  involved, informal family.
- ``pronouns_he_she`` — third-person gendered pronouns, kept SPLIT into
  ``pronouns_he`` and ``pronouns_she`` so the opposite folk signal of "he" and
  "she" stays visible; the scores read the combined ``pronouns_he_she`` count.
- ``weak`` — high-frequency function words (and/the/with/a/to/of/is/that/for/
  it/as/was/on/at/by/been/this) in the published weak-word flavor; informal
  family. These overlap articles and prepositions on purpose: the two scores
  read different families out of the same tokens.
- ``hedging`` — maybe/perhaps/believe/think/...; Koppel's private-verb hedges.
  Counted and reported but NOT in either published score sum (neither sum
  names it); kept as a documented category for readers who extend the formula.
- ``question_marks`` — "?" counted as text marks on tokens, not as words.

Two signed scores (per-thousand-word rates, explicit linear sums — see
:func:`score_from_counts` for the constants):

- ``Informal score`` = 0.004*(pronouns rate - 85) + 0.0015*(weak rate - 260)
  + 0.015*(question+exclamation rate - 6)  — positive leans male
  ("informal male"), so it is reported in the ``Masculine`` column.
- ``Formal score`` = 0.004*(prepositions rate - 110) + 0.004*(articles rate - 75)
  + 0.006*(numbers rate - 6) + 0.012*(certainty rate - 3)
  + 0.25*(average word length - 4.4)  — positive leans female
  ("formal female"), so it is reported in the ``Feminine`` column.

The reference constants are rough English-prose averages used only to centre
the scores near zero; they are not fitted to any gender-labelled corpus.
Label thresholds (one table, on the male-positive scale) are documented on
:func:`label_for`. Documents under 50 content tokens are "unknown" on both
axes — the original tool wants 300+ words, and 50 is this suite's hard floor.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "CATEGORIES",
    "GUESS_COLUMNS",
    "SUMMARY_COLUMNS",
    "guess_from_counts",
    "guess_gender",
    "label_for",
    "score_from_counts",
    "summarize_labels",
]

GUESS_COLUMNS = ["Document", "Document ID", "Words", "Masculine", "Feminine", "Formal label", "Informal label", "Label"]
SUMMARY_COLUMNS = ["Label", "Documents"]

MIN_WORDS = 50

# Reference rates per 1,000 content tokens (average word length in characters)
# used to centre the two scores. Calibrated so ordinary prose lands near 0.0
# and the documented thresholds split "mostly_" from strong bands.
_REF_PREPOSITIONS = 110.0
_REF_ARTICLES = 75.0
_REF_NUMBERS = 6.0
_REF_CERTAINTY = 3.0
_REF_PRONOUNS = 85.0
_REF_WEAK = 260.0
_REF_PUNCT = 6.0
_REF_AVG_LEN = 4.4

CATEGORIES: dict[str, frozenset[str]] = {
    "prepositions": frozenset(
        {
            "about",
            "above",
            "across",
            "after",
            "against",
            "along",
            "among",
            "around",
            "as",
            "at",
            "before",
            "behind",
            "below",
            "beneath",
            "beside",
            "between",
            "beyond",
            "by",
            "despite",
            "down",
            "during",
            "except",
            "for",
            "from",
            "in",
            "inside",
            "into",
            "like",
            "near",
            "of",
            "off",
            "on",
            "onto",
            "out",
            "outside",
            "over",
            "past",
            "since",
            "through",
            "throughout",
            "to",
            "toward",
            "towards",
            "under",
            "underneath",
            "unlike",
            "until",
            "up",
            "upon",
            "with",
            "within",
            "without",
        }
    ),
    "articles": frozenset({"a", "an", "the"}),
    "pronouns_i": frozenset({"i", "me", "my", "mine", "myself"}),
    "pronouns_we": frozenset({"we", "us", "our", "ours", "ourselves"}),
    "pronouns_you": frozenset({"you", "your", "yours", "yourself", "yourselves"}),
    "pronouns_he": frozenset({"he", "him", "his", "himself"}),
    "pronouns_she": frozenset({"she", "her", "hers", "herself"}),
    "pronouns_he_she": frozenset({"he", "him", "his", "himself", "she", "her", "hers", "herself"}),
    "numbers": frozenset(
        {
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
            "hundred",
            "thousand",
            "million",
            "billion",
            "dozen",
        }
    ),
    "weak": frozenset(
        {
            "and",
            "the",
            "with",
            "a",
            "to",
            "of",
            "is",
            "that",
            "for",
            "it",
            "as",
            "was",
            "on",
            "at",
            "by",
            "been",
            "this",
        }
    ),
    "hedging": frozenset(
        {
            "maybe",
            "perhaps",
            "believe",
            "think",
            "suppose",
            "guess",
            "feel",
            "seem",
            "apparently",
            "probably",
            "possibly",
            "likely",
            "somewhat",
            "rather",
            "quite",
            "tend",
        }
    ),
    "certainty": frozenset(
        {"never", "really", "must", "always", "certainly", "definitely", "absolutely", "clearly", "surely", "totally"}
    ),
}

_RANK = {"female": -2, "mostly_female": -1, "androgynous": 0, "mostly_male": 1, "male": 2}


def label_for(score: float) -> str:
    """Label for a male-positive score (> 0 leans male, < 0 leans female).

    Documented thresholds: > +0.5 "male", > +0.15 "mostly_male",
    >= -0.15 "androgynous", >= -0.5 "mostly_female", else "female".
    """
    if score > 0.5:
        return "male"
    if score > 0.15:
        return "mostly_male"
    if score >= -0.15:
        return "androgynous"
    if score >= -0.5:
        return "mostly_female"
    return "female"


def _rate(count: float, words: int) -> float:
    """Hits per 1,000 content tokens."""
    return 1000.0 * float(count) / float(words or 1)


def score_from_counts(counts: Mapping[str, float], words: int) -> tuple[float, float]:
    """(Informal score, Formal score) as the documented linear sums of rates.

    ``counts`` keys are the ``CATEGORIES`` names plus optional ``question_marks``,
    ``exclamation_marks`` (text marks, not words) and ``avg_word_length``
    (characters; defaults to the 4.4 reference, contributing nothing). When
    ``pronouns_he_she`` is absent the split ``pronouns_he`` + ``pronouns_she``
    counts feed the same combined pronoun rate.
    """
    he_she = float(counts.get("pronouns_he_she", counts.get("pronouns_he", 0.0) + counts.get("pronouns_she", 0.0)))
    pronouns = (
        float(counts.get("pronouns_i", 0.0))
        + float(counts.get("pronouns_we", 0.0))
        + float(counts.get("pronouns_you", 0.0))
        + he_she
    )
    punct = float(counts.get("question_marks", 0.0)) + float(counts.get("exclamation_marks", 0.0))
    avg_len = float(counts.get("avg_word_length", _REF_AVG_LEN))
    informal = (
        0.004 * (_rate(pronouns, words) - _REF_PRONOUNS)
        + 0.0015 * (_rate(counts.get("weak", 0.0), words) - _REF_WEAK)
        + 0.015 * (_rate(punct, words) - _REF_PUNCT)
    )
    formal = (
        0.004 * (_rate(counts.get("prepositions", 0.0), words) - _REF_PREPOSITIONS)
        + 0.004 * (_rate(counts.get("articles", 0.0), words) - _REF_ARTICLES)
        + 0.006 * (_rate(counts.get("numbers", 0.0), words) - _REF_NUMBERS)
        + 0.012 * (_rate(counts.get("certainty", 0.0), words) - _REF_CERTAINTY)
        + 0.25 * (avg_len - _REF_AVG_LEN)
    )
    return informal, formal


def _combined(formal_label: str, informal_label: str) -> str:
    """The more extreme of two same-direction labels; "androgynous" on disagreement."""
    if formal_label == "unknown" or informal_label == "unknown":
        return "unknown"
    f_rank = _RANK.get(formal_label, 0)
    i_rank = _RANK.get(informal_label, 0)
    if f_rank * i_rank < 0:
        return "androgynous"
    if abs(i_rank) > abs(f_rank):
        return informal_label
    if abs(f_rank) > abs(i_rank):
        return formal_label
    return formal_label


def guess_from_counts(counts: Mapping[str, float], words: int, mode: str = "both") -> tuple[str, str, str]:
    """(formal_label, informal_label, label) from per-category hit counts.

    Public seam so the documented thresholds are unit-testable without frames.
    Both labels come from one male-positive threshold table: the informal score
    ("informal male") is used as is, the formal score ("formal female") is
    negated first. Under 50 content tokens both labels are "unknown".
    ``mode`` selects which axis drives the combined label and leaves the other
    label empty; an unrecognised mode is treated as "both" (:func:`guess_gender`
    validates the flag first and fails loudly with ``GENDER_BAD_MODE``).
    """
    words = int(words)
    if words < MIN_WORDS:
        return "unknown", "unknown", "unknown"
    informal, formal = score_from_counts(counts, words)
    informal_label = label_for(informal)
    formal_label = label_for(-formal)
    if mode == "formal":
        return formal_label, "", formal_label
    if mode == "informal":
        return "", informal_label, informal_label
    return formal_label, informal_label, _combined(formal_label, informal_label)


def _count_document(forms: Iterable[object]) -> dict[str, float]:
    """Category hits, text marks and length stats for one document's forms."""
    counts: dict[str, float] = {key: 0.0 for key in CATEGORIES}
    counts["question_marks"] = 0.0
    counts["exclamation_marks"] = 0.0
    counts["sentence_count"] = 0.0
    words = 0
    total_len = 0
    for raw in forms:
        if raw is None:
            continue
        try:
            if bool(pd.isna(raw)):
                continue
        except (TypeError, ValueError):
            pass
        text = str(raw)
        counts["question_marks"] += text.count("?")
        counts["exclamation_marks"] += text.count("!")
        if text.strip().endswith((".", "!", "?")):
            counts["sentence_count"] += 1
        bare = text.strip().strip(".,;:!?\"'()[]{}").lower()
        if bare.isdigit():
            counts["numbers"] += 1
        if not bare.isalpha():
            continue
        words += 1
        total_len += len(bare)
        for key, members in CATEGORIES.items():
            if bare in members:
                counts[key] += 1
    counts["words"] = float(words)
    counts["avg_word_length"] = (total_len / words) if words else _REF_AVG_LEN
    return counts


def guess_gender(frame: pd.DataFrame, *, mode: str = "both") -> Result[pd.DataFrame]:
    """Per-document writing-style gender guesses from a canonical CoNLL frame.

    Scores the surface ``Form`` column (the weak words are word forms, not
    lemmas). Empty input yields an empty frame with the declared columns.
    """
    if mode not in ("formal", "informal", "both"):
        return Result.failure(
            Diagnostic.error("GENDER_BAD_MODE", f"mode must be formal, informal, or both, got {mode!r}", mode=mode)
        )
    if Col.FORM.value not in frame.columns:
        return Result.failure(Diagnostic.error("GENDER_MISSING_COLUMN", f"missing {Col.FORM.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=GUESS_COLUMNS))

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        counts = _count_document(group[Col.FORM.value].tolist())
        words = int(counts.pop("words"))
        informal, formal = score_from_counts(counts, words)
        formal_label, informal_label, label = guess_from_counts(counts, words, mode=mode)
        rows.append(
            {
                "Document": str(group[doc_col].iloc[0]) if doc_col is not None else "",
                "Document ID": str(doc_id),
                "Words": words,
                "Masculine": round(informal, 4),
                "Feminine": round(formal, 4),
                "Formal label": formal_label,
                "Informal label": informal_label,
                "Label": label,
            }
        )
    return Result.success(pd.DataFrame(rows, columns=GUESS_COLUMNS))


def summarize_labels(annotated: pd.DataFrame) -> Result[pd.DataFrame]:
    """Doc-level aggregate: how many documents landed on each combined Label."""
    if annotated.empty or "Label" not in annotated.columns:
        return Result.success(pd.DataFrame(columns=SUMMARY_COLUMNS))
    tally = annotated["Label"].astype(str).value_counts().sort_index()
    out = pd.DataFrame({"Label": tally.index.astype(str), "Documents": tally.values.astype(int)})
    return Result.success(out[SUMMARY_COLUMNS])
