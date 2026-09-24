"""String similarity (FR-2.9) — Levenshtein metrics, verbatim legacy port.

Basis: ``string_similarity_util.py`` (legacy, stdlib-only). Unit-cost edit
distance, substitution-weighted ratio on the 0-100 python-Levenshtein scale,
case/order-insensitive ``similarity``, and best-match-that-is-best (never
first-over-threshold, never the word itself).
"""

from __future__ import annotations

__all__ = [
    "best_match",
    "levenshtein_distance",
    "levenshtein_ratio",
    "similarity",
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Unit-cost Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + (c1 != c2)))
        previous = current
    return previous[-1]


def _weighted_distance(s1: str, s2: str) -> int:
    """Edit distance with substitutions weighted 2 (insert plus delete)."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + (2 if c1 != c2 else 0)))
        previous = current
    return previous[-1]


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Similarity on a 0-100 scale (100 = identical). Case- and order-sensitive."""
    total = len(s1) + len(s2)
    if total == 0:
        return 100.0
    return round((total - _weighted_distance(s1, s2)) / total * 100, 1)


def _normalize(s: object) -> str:
    return " ".join(str(s).lower().replace(",", " ").split())


def _sort_tokens(s: object) -> str:
    return " ".join(sorted(_normalize(s).split()))


def similarity(s1: str, s2: str) -> float:
    """Similarity on 0-100, ignoring case, comma spacing, and word order."""
    n1, n2 = _normalize(s1), _normalize(s2)
    score = levenshtein_ratio(n1, n2)
    if " " in n1 or " " in n2:
        score = max(score, levenshtein_ratio(_sort_tokens(s1), _sort_tokens(s2)))
    return score


def best_match(
    word: str,
    candidates: list[tuple[str, object] | str],
    threshold: float,
) -> tuple[str, object, float, int] | None:
    """Closest candidate at or above *threshold*, or None.

    ``candidates`` are ``(word, frequency)`` pairs or plain strings. The word
    itself (byte-identical) is skipped; case/order variants scoring 100 stay.
    Returns ``(candidate, frequency, score, edit_distance)``.
    """
    best: tuple[str, object, float, int] | None = None
    for candidate in candidates:
        if isinstance(candidate, (tuple, list)):
            cand_word = candidate[0]
            cand_freq: object = candidate[1] if len(candidate) > 1 else ""
        else:
            cand_word, cand_freq = candidate, ""
        if cand_word == word:
            continue
        score = similarity(word, cand_word)
        if score < threshold:
            continue
        if best is None or score > best[2]:
            best = (cand_word, cand_freq, score, levenshtein_distance(_normalize(word), _normalize(cand_word)))
    return best
