"""Readability (FR-2.1) — five legacy formulas plus SMOG, on raw document text.

Parity basis: ``statistics_corpus_readability_util.py`` (legacy). Counts,
the five formulas, the FRE [0, 121] / grade [0, 30] clamps, 2dp rounding,
and the seven interpretation bands are ported verbatim.

Intentional differences from legacy:

* SMOG is new-from-formula (McLaughlin 1969): legacy delegated it to the
  ``textstat`` package, which is not installed here and was never pinned, so
  no valid golden exists. Policy (C6-8): the full 30-sentence formula is
  used, NOT textstat's reduced-sample variant; below 3 sentences the term
  is degenerate and scores 0.0 with a ``READABILITY_SMOG_SHORT_TEXT``
  warning. Unicode text is supported: the word regex is Unicode-aware
  (accented words, non-Latin scripts count as words), matching the
  lexical_diversity tokenizer's unicode behavior.
* Gunning Fog keeps the legacy simplification (every 3+ syllable word counts;
  proper nouns and jargon are NOT excluded the way textbook Fog does).
* Empty documents emit a zero row with ``Interpretation == "n/a"`` and a
  ``READABILITY_EMPTY_DOC`` warning. Legacy skipped them silently (single-file
  mode) or crashed on them downstream — neither survives fail-big.
* The syllable counter is the readability legacy's own heuristic. It differs
  second-order from the ``text_statistics`` variant (``agreed``: 2 here, 1
  there). Unifying syllable counting behind one core function is follow-up
  work, not this packet — changing ``text_statistics`` outputs here would
  bundle an unrelated behavior change.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from core.io.reader import Corpus, display_names
from core.result import Diagnostic, Result

__all__ = [
    "ReadabilityCounts",
    "ReadabilityResult",
    "analyze_text",
    "automated_readability_index",
    "clamp_fre",
    "clamp_grade",
    "coleman_liau",
    "count_syllables",
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "gunning_fog",
    "interpret_fre",
    "run",
    "smog_index",
]


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
# C6-8: unicode words; the curly apostrophe is intentional (RUF001 suppressed)
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*['’]?", re.UNICODE)  # noqa: RUF001
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
_SUFFIX_RE = re.compile(r"(?:es|ed|e)$")

_FRE_BANDS: tuple[tuple[float, str], ...] = (
    (90.0, "Very Easy (5th grade)"),
    (80.0, "Easy (6th grade)"),
    (70.0, "Fairly Easy (7th grade)"),
    (60.0, "Standard (8th-9th grade)"),
    (50.0, "Fairly Difficult (10th-12th grade)"),
    (30.0, "Difficult (College)"),
    (float("-inf"), "Very Difficult (Graduate)"),
)


@dataclass(frozen=True, slots=True)
class ReadabilityCounts:
    sentences: int
    words: int
    syllables: int
    polysyllabic: int
    characters: int


@dataclass(frozen=True, slots=True)
class ReadabilityResult:
    frame: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def count_syllables(word: str) -> int:
    """Legacy readability syllable heuristic, verbatim."""
    w = word.lower().strip()
    if len(w) <= 3:
        return 1
    w = _SUFFIX_RE.sub("", w) or w
    return max(1, len(_VOWEL_GROUP_RE.findall(w)))


def analyze_text(text: str) -> ReadabilityCounts:
    """Shared validated counts: sentences, words, syllables, polysyllabic, chars."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    words = _WORD_RE.findall(text)
    syllable_counts = [count_syllables(w) for w in words]
    return ReadabilityCounts(
        sentences=max(1, len(sentences)),
        words=max(1, len(words)),
        syllables=sum(syllable_counts),
        polysyllabic=sum(1 for c in syllable_counts if c >= 3),
        characters=sum(len(w) for w in words),
    )


def flesch_reading_ease(counts: ReadabilityCounts) -> float:
    return 206.835 - 1.015 * (counts.words / counts.sentences) - 84.6 * (counts.syllables / counts.words)


def flesch_kincaid_grade(counts: ReadabilityCounts) -> float:
    return 0.39 * (counts.words / counts.sentences) + 11.8 * (counts.syllables / counts.words) - 15.59


def gunning_fog(counts: ReadabilityCounts) -> float:
    return 0.4 * ((counts.words / counts.sentences) + 100 * (counts.polysyllabic / counts.words))


def coleman_liau(counts: ReadabilityCounts) -> float:
    score_l = (counts.characters / counts.words) * 100
    score_s = (counts.sentences / counts.words) * 100
    return 0.0588 * score_l - 0.296 * score_s - 15.8


def automated_readability_index(counts: ReadabilityCounts) -> float:
    return 4.71 * (counts.characters / counts.words) + 0.5 * (counts.words / counts.sentences) - 21.43


def smog_index(counts: ReadabilityCounts) -> float:
    """McLaughlin SMOG from formula; degenerate below 3 sentences."""
    if counts.sentences < 3:
        return 0.0
    return float(1.043 * (counts.polysyllabic * 30 / counts.sentences) ** 0.5 + 3.1291)


def clamp_fre(score: float) -> float:
    return max(0.0, min(121.0, score))


def clamp_grade(score: float) -> float:
    return max(0.0, min(30.0, score))


def interpret_fre(score: float) -> str:
    for threshold, label in _FRE_BANDS:
        if score >= threshold:
            return label
    return "Very Difficult (Graduate)"  # unreachable; keeps mypy total


_OUTPUT_COLUMNS = [
    "Document ID",
    "Document",
    "Sentences",
    "Words",
    "Syllables",
    "Polysyllabic Words",
    "Characters",
    "Flesch Reading Ease",
    "Flesch-Kincaid Grade",
    "Gunning Fog Index",
    "Coleman-Liau Index",
    "Automated Readability Index",
    "SMOG Index",
    "Interpretation",
]


def _score_row(counts: ReadabilityCounts) -> dict[str, float | str]:
    return {
        "Flesch Reading Ease": round(clamp_fre(flesch_reading_ease(counts)), 2),
        "Flesch-Kincaid Grade": round(clamp_grade(flesch_kincaid_grade(counts)), 2),
        "Gunning Fog Index": round(clamp_grade(gunning_fog(counts)), 2),
        "Coleman-Liau Index": round(clamp_grade(coleman_liau(counts)), 2),
        "Automated Readability Index": round(clamp_grade(automated_readability_index(counts)), 2),
        "SMOG Index": round(clamp_grade(smog_index(counts)), 2),
    }


def run(corpus: Corpus) -> Result[ReadabilityResult]:
    """Per-document readability scores over raw corpus texts (no parse needed)."""
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    short_docs: list[str] = []
    names = display_names(corpus.docs)
    for doc in corpus.docs:
        name = names[doc.doc_id]
        if not doc.text.strip():
            diags.append(
                Diagnostic.warning("READABILITY_EMPTY_DOC", f"{name} has no text", doc_id=doc.doc_id, path=name)
            )
            rows.append(
                {
                    "Document ID": str(doc.doc_id),
                    "Document": name,
                    "Sentences": 0,
                    "Words": 0,
                    "Syllables": 0,
                    "Polysyllabic Words": 0,
                    "Characters": 0,
                    "Flesch Reading Ease": 0.0,
                    "Flesch-Kincaid Grade": 0.0,
                    "Gunning Fog Index": 0.0,
                    "Coleman-Liau Index": 0.0,
                    "Automated Readability Index": 0.0,
                    "SMOG Index": 0.0,
                    "Interpretation": "n/a",
                }
            )
            continue
        counts = analyze_text(doc.text)
        scores = _score_row(counts)
        if counts.sentences < 3:
            short_docs.append(name)
        rows.append(
            {
                "Document ID": str(doc.doc_id),
                "Document": name,
                "Sentences": counts.sentences,
                "Words": counts.words,
                "Syllables": counts.syllables,
                "Polysyllabic Words": counts.polysyllabic,
                "Characters": counts.characters,
                **scores,
                "Interpretation": interpret_fre(float(scores["Flesch Reading Ease"])),
            }
        )
    if short_docs:
        shown = short_docs[:5]
        diags.append(
            Diagnostic.warning(
                "READABILITY_SMOG_SHORT_TEXT",
                f"{len(short_docs)} document(s) have <3 sentences; SMOG scores 0.0 there",
                count=len(short_docs),
                documents=shown,
            )
        )
    frame = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    return Result.success(ReadabilityResult(frame=frame), *diags)
