"""Text cleaning at read time: what is not the author's words, out of the analysis.

Transcripts carry stage directions -- ``(Applause.)``, ``(Laughter.)``,
``[inaudible]`` -- and audience reactions. They are not the speaker's words,
and every tool counts them: they inflate word counts, they move sentiment
(Applause is positive to VADER), and they reach keyness and topics as if they
were rhetoric (backlog 4d-5). NER even tags "Speaker" and "Boo" as people.

The clean is **at read time and never writes** (R3): the imported file is the
record of what was received, and the project's ``text_cleaning`` setting says
whether the analysis sees the text with or without the stage directions. A
run's envelope records the setting and the counts, so a result says it was
cleaned.

Only spans that are unambiguously stage directions are removed: a bracketed
span of one to four words whose every word is a known reaction word (or a
joiner like "and"). ``(see page 12)`` and ``[sic]`` are kept: a parenthetical
can carry meaning, and guessing is how a cleaner destroys evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

__all__ = ["KNOWN_STAGE_WORDS", "count_stage_directions", "strip_stage_directions"]

#: The reactions and annotations a transcript records in brackets. "sic" is
#: deliberately absent: it marks the author's words as written, not a sound.
KNOWN_STAGE_WORDS: frozenset[str] = frozenset(
    {
        "applause",
        "laughter",
        "laughs",
        "laughing",
        "cheers",
        "cheering",
        "clapping",
        "claps",
        "boos",
        "booing",
        "hisses",
        "hissing",
        "groans",
        "groaning",
        "shouts",
        "shouting",
        "murmurs",
        "murmuring",
        "coughs",
        "coughing",
        "chuckles",
        "sighs",
        "inaudible",
        "crosstalk",
        "cross-talk",
        "music",
        "silence",
        "pause",
        "commotion",
        "interruption",
        "protests",
        "protest",
    }
)

#: Words that can join reactions inside one span ("(applause and laughter)")
#: without making the span about something else.
_JOINERS: frozenset[str] = frozenset({"and", "or", "mixed", "sustained", "brief", "mild", "loud", "light", "briefly"})

_SPAN = re.compile(r"\([^()]{1,80}\)|\[[^[\]]{1,80}\]")
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)?")

#: A stage direction is short: "(sustained applause that continues for many
#: minutes)" is prose, and prose in brackets is evidence.
_MAX_SPAN_WORDS = 4
#: How many examples the Corpus page's preview shows.
_MAX_SAMPLES = 5


def _is_stage_direction(inner: str, vocabulary: frozenset[str]) -> bool:
    words = _WORD.findall(inner.casefold())
    if not 1 <= len(words) <= _MAX_SPAN_WORDS:
        return False
    known = False
    for word in words:
        if word in vocabulary:
            known = True
        elif word not in _JOINERS:
            return False
    return known


def strip_stage_directions(text: str, extra: Sequence[str] = ()) -> tuple[str, int]:
    """``(text with stage directions removed, how many were removed)``.

    *extra* is the project's own list of words that make a bracketed span a
    stage direction (``["booing", "table banging"]``'s words).
    """
    if not text:
        return text, 0
    vocabulary = KNOWN_STAGE_WORDS | frozenset(word for term in extra for word in _WORD.findall(term.casefold()))
    removed = 0

    def drop(match: re.Match[str]) -> str:
        nonlocal removed
        if _is_stage_direction(match.group(0)[1:-1], vocabulary):
            removed += 1
            return ""
        return match.group(0)

    return _SPAN.sub(drop, text), removed


def count_stage_directions(texts: Sequence[str], extra: Sequence[str] = ()) -> tuple[int, list[str]]:
    """How many stage directions the texts carry, and examples of them.

    The Corpus page's preview: "Found 2,314 bracketed stage directions like
    (Applause.), (Laughter.)" -- counted without changing anything.
    """
    found = 0
    samples: list[str] = []
    for text in texts:
        vocabulary = KNOWN_STAGE_WORDS | frozenset(word for term in extra for word in _WORD.findall(term.casefold()))
        for match in _SPAN.finditer(text):
            if _is_stage_direction(match.group(0)[1:-1], vocabulary):
                found += 1
                if len(samples) < _MAX_SAMPLES and match.group(0) not in samples:
                    samples.append(match.group(0))
    return found, samples
