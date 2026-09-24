"""Spell checking (FR-3.2) — detection and optional correction, explicit wordlist.

No bundled dictionary and no downloads: the language wordlist is an explicit
input file (one word per line), so the dictionary in use is always knowable.
Suggestions reuse the tested Levenshtein ``best_match`` engine. Corrections
are written as new artifacts; inputs are never overwritten.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

import pandas as pd

from core.io.reader import Corpus, display_names
from core.result import Diagnostic, Result
from pathlib import Path

__all__ = [
    "FindingsResult",
    "apply_correction",
    "check_text",
    "correct_text",
    "load_wordlist",
    "run",
    "tokenize_words",
]

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*['’]?", re.UNICODE)  # noqa: RUF001 — curly apostrophe intentional


@dataclass(frozen=True, slots=True)
class FindingsResult:
    frame: pd.DataFrame  # Document, Word, Count, Suggestion, Score

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def load_wordlist(path: Path) -> Result[frozenset[str]]:
    """Load an explicit wordlist file (one word per line, casefolded)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return Result.failure(
            Diagnostic.error("SPELL_NO_WORDLIST", f"could not read wordlist {path}: {exc}", path=str(path))
        )
    words = frozenset(line.strip().casefold() for line in text.splitlines() if line.strip())
    if not words:
        return Result.failure(Diagnostic.error("SPELL_EMPTY_WORDLIST", f"wordlist {path} has no words"))
    return Result.success(words)


def tokenize_words(text: str) -> list[str]:
    """Unicode word tokens in surface order (C6-12: café/naïve/Cyrillic/CJK
    survive; digits and punctuation-only tokens skipped; apostrophes kept)."""
    return _WORD_RE.findall(text)


def validate_threshold(threshold: float) -> Diagnostic | None:
    """Finite value in 0..100, else a diagnostic (C6-12)."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return Diagnostic.error("SPELL_BAD_THRESHOLD", f"threshold must be a number, got {type(threshold).__name__}")
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 100:
        return Diagnostic.error("SPELL_BAD_THRESHOLD", f"threshold must be finite in 0..100, got {threshold}")
    return None


def check_text(text: str, vocab: frozenset[str], threshold: float) -> list[dict[str, object]]:
    """One row per distinct unknown word: Word, Count, Suggestion, Score.

    Performance invariant (review finding S2): the vocabulary is bucketed by
    length ONCE per run, and each lookup scores only candidates whose length
    can possibly reach *threshold*. A candidate's minimum possible weighted
    edit distance is the length difference |len(word) - len(cand)|; if that
    alone exceeds the similarity budget, no candidate of that length can
    qualify, so it is never scored. The filter is exact, so suggestions are
    identical to an unfiltered scan.
    """
    counts = Counter(w.casefold() for w in tokenize_words(text))
    buckets: dict[int, list[str]] = {}
    for entry in sorted(vocab):
        buckets.setdefault(len(entry), []).append(entry)
    rows: list[dict[str, object]] = []
    for word in sorted(counts):
        if word in vocab:
            continue
        candidates = _plausible_candidates(word, buckets, threshold)
        match = _best_match_budget(word, candidates, threshold)
        rows.append(
            {
                "Word": word,
                "Count": counts[word],
                "Suggestion": match[0] if match else "",
                "Score": match[1] if match else 0.0,
            }
        )
    return rows


def _plausible_candidates(
    word: str,
    buckets: dict[int, list[str]],
    threshold: float,
) -> list[str]:
    """Every bucket whose length can reach *threshold* against *word*.

    score = (total - dist)/total * 100 >= threshold  requires
    dist <= total * (1 - threshold/100). With dist >= |l1 - l2| as the
    lower bound, a bucket qualifies iff its length difference fits.
    """
    length = len(word)
    max_delta = int((length + length) * (1.0 - threshold / 100.0)) + 1
    candidates: list[str] = []
    for target in range(max(0, length - max_delta), length + max_delta + 1):
        total = length + target
        min_dist = abs(length - target)
        if min_dist > total * (1.0 - threshold / 100.0):
            continue
        candidates.extend(buckets.get(target, []))
    return candidates


def _bounded_weighted_distance(s1: str, s2: str, budget: float) -> float | None:
    """Weighted edit distance, abandoned as soon as it must exceed *budget*.

    Same substitution-weighted DP as the legacy engine. Each completed row
    certifies a lower bound for the final cell: the last cell of row *i*
    (``cur[n2]``, the cost of consuming s1's first i+1 characters against all
    of s2) can decrease by at most 1 per remaining row (a delete costs 1),
    so ``cur[n2] - (n1 - i - 1) > budget`` proves the final distance exceeds
    the budget and the scan stops without finishing. Verified exact against
    the full DP over randomized cases; None means "distance > budget".
    """
    if s1 == s2:
        return 0.0
    n1, n2 = len(s1), len(s2)
    previous = list(range(n2 + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + (2 if c1 != c2 else 0)))
        # Strongest valid early bound: the final distance D[n1][n2] can be at
        # most 1 below this row's smallest cell per remaining row (each future
        # row may reduce a cell by at most 1, the cost of one delete). A
        # cheap first check uses the row's last cell before the full min.
        floor = current[n2] - (n1 - (i + 1))
        if floor > budget:
            return None
        if min(current) - (n1 - (i + 1)) > budget:
            return None
        previous = current
    return previous[-1]


def _best_match_budget(
    word: str,
    candidates: list[str],
    threshold: float,
) -> tuple[str, float] | None:
    """Exact equivalent of ``best_match`` over *candidates*, with banded abort.

    For each candidate the threshold translates to a maximum weighted distance
    given its length; candidates whose DP floor exceeds that budget mid-row
    are abandoned without finishing. Returns ``(suggestion, score)``.
    """
    best: tuple[str, float] | None = None
    normalized = _normalize_word(word)
    for cand_word in candidates:
        if cand_word == word:
            continue
        norm_cand = _normalize_word(cand_word)
        score = _bounded_similarity(normalized, norm_cand, threshold)
        if score is None:
            continue
        if " " in normalized or " " in norm_cand:
            sorted_a = " ".join(sorted(normalized.split()))
            sorted_b = " ".join(sorted(norm_cand.split()))
            alt = _bounded_similarity(sorted_a, sorted_b, threshold)
            if alt is not None:
                score = max(score, alt)
        if score < threshold:
            continue
        if best is None or score > best[1]:
            best = (cand_word, score)
    return best


def _normalize_word(s: str) -> str:
    return " ".join(s.lower().replace(",", " ").split())


def _bounded_similarity(a: str, b: str, threshold: float) -> float | None:
    """Legacy ``levenshtein_ratio`` with a banded abort at the threshold budget."""
    total = len(a) + len(b)
    if total == 0:
        return 100.0
    dist = _bounded_weighted_distance(a, b, total * (1.0 - threshold / 100.0))
    if dist is None:
        return None
    return round((total - dist) / total * 100, 1)


def _match_case(surface: str, replacement: str) -> str:
    if surface.isupper():
        return replacement.upper()
    if surface[:1].isupper():
        return replacement.capitalize()
    return replacement


def apply_correction(text: str, word: str, suggestion: str) -> str:
    """Replace whole-word occurrences of *word*, preserving capitalisation."""
    if not suggestion:
        return text
    # ``\b`` is not sufficient for tokens containing apostrophes or hyphens:
    # it can match the interior of ``mother-in-law`` or leave a trailing
    # apostrophe behind. Use the same token alphabet as ``tokenize_words``.
    pattern = re.compile(
        r"(?<![\w'\u2019\-])" + re.escape(word) + r"(?![\w'\u2019\-])",
        flags=re.IGNORECASE,
    )

    def swap(match: re.Match[str]) -> str:
        return _match_case(match.group(0), suggestion)

    return pattern.sub(swap, text)


def correct_text(text: str, findings: pd.DataFrame) -> str:
    """Apply all suggested corrections from a findings frame to *text*.

    C6-12: the frame is validated first — a NaN/blank suggestion becomes no
    replacement, never the literal string "nan"; Word must be a real token.

    Review finding S8: applying findings sequentially lets one correction's
    suggestion be re-replaced by a later finding's word (a cascade). All
    replacements are decided against the ORIGINAL text in one pass: each
    token position is rewritten at most once, and the result does not depend
    on row order in the findings frame.
    """
    replacements: list[tuple[str, str]] = []
    for record in findings.to_dict(orient="records"):
        row = {str(k): v for k, v in dict(record).items()}
        suggestion_raw = row.get("Suggestion", "")
        word = row.get("Word", "")
        if pd.isna(suggestion_raw) or pd.isna(word):
            continue
        suggestion = str(suggestion_raw).strip()
        word = str(word).strip()
        if suggestion and word and suggestion.casefold() != "nan":
            replacements.append((word, suggestion))
    if not replacements:
        return text
    # One alternation over all words, longest first so overlapping words
    # (e.g. "run" vs "running") cannot shadow each other's matches.
    alternation = "|".join(re.escape(word) for word, _ in sorted(replacements, key=lambda ws: -len(ws[0])))
    suggestion_by_word = {word.casefold(): suggestion for word, suggestion in replacements}
    pattern = re.compile(
        r"(?<![\w'\u2019\-])(" + alternation + r")(?![\w'\u2019\-])",
        flags=re.IGNORECASE,
    )

    def swap(match: re.Match[str]) -> str:
        surface = match.group(0)
        suggestion = suggestion_by_word.get(surface.casefold(), "")
        return _match_case(surface, suggestion) if suggestion else surface

    return pattern.sub(swap, text)


def run(corpus: Corpus, vocab: frozenset[str], threshold: float = 80.0) -> Result[FindingsResult]:
    """Unknown-word findings per document (originals untouched, always).

    C6-12: threshold is validated; wordlist normalization matches tokens
    (both casefolded, NFKC-normalized) so café in a list matches café in
    text under either normalization.
    """
    problem = validate_threshold(threshold)
    if problem is not None:
        return Result.failure(problem)
    import unicodedata

    normalized_vocab = frozenset(unicodedata.normalize("NFKC", w.casefold()) for w in vocab)
    rows: list[dict[str, object]] = []
    names = display_names(corpus.docs)
    for doc in corpus.docs:
        text = unicodedata.normalize("NFKC", doc.text)
        for finding in check_text(text, normalized_vocab, threshold):
            rows.append({"Document": names[doc.doc_id], **finding})
    out = pd.DataFrame(rows, columns=["Document", "Word", "Count", "Suggestion", "Score"])
    return Result.success(FindingsResult(frame=out))
