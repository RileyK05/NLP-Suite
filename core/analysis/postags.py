"""The one noun-tag rule, in one place.

Parts of speech arrive as Penn Treebank tags from spaCy (``NN``, ``NNS``,
``NNP``) or as Universal Dependencies tags from Stanza (``NOUN``, ``PROPN``)
-- either backend, whichever was configured, and the cached parse carries
whatever the parse that produced it used. A check that accepted one tagset
only silently kept nothing on the other: ``tokens_from_frame(nouns_only=
True)`` returned an empty corpus, and the topic model then failed with "no
documents to model", pointing the reader at their corpus rather than at the
tag scheme. This module is that fix, written once.

The rule every noun check in this codebase applies: **Penn ``NN*`` or
Universal ``NOUN``/``PROPN``.** Unknown or blank tags are never nouns, and
the callers that must not drop a token on an unknown tag check
:func:`is_noun` only when a tag is present.
"""

from __future__ import annotations

__all__ = ["NOUN_TAGS", "WORD_CLASSES", "is_noun_tag", "word_class"]


#: The tag names of a noun under either tagset. One tuple, one place.
NOUN_TAGS: frozenset[str] = frozenset({"NOUN", "PROPN"})


def is_noun_tag(pos: object) -> bool:
    """True for a noun under either tagset: Penn ``NN*`` or Universal
    ``NOUN``/``PROPN``.

    Accepts any input (a cell may hold ``None``, a number, whitespace) and
    answers for the normalised string, so seven callers cannot drift by
    stripping or upper-casing differently. Penn ``NN*`` covers ``NN``,
    ``NNS``, ``NNP`` and ``NNS`` -- and, deliberately, no other prefix: a
    tag is checked as a word, not a guess.
    """
    text = str(pos).strip().upper()
    return text.startswith("NN") or text in NOUN_TAGS


#: The word classes a figure can be restricted to, named for a reader.
WORD_CLASSES: tuple[str, ...] = ("noun", "verb", "adjective", "adverb")
_UNIVERSAL = {"NOUN": "noun", "PROPN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}
_PENN = (("NN", "noun"), ("VB", "verb"), ("JJ", "adjective"), ("RB", "adverb"))


def word_class(pos: object) -> str:
    """``noun``, ``verb``, ``adjective``, ``adverb`` or ``""``, under either tagset.

    The same rule as :func:`is_noun_tag`, widened to the four open classes:
    Penn ``NN*``/``VB*``/``JJ*``/``RB*`` or Universal ``NOUN``/``PROPN``,
    ``VERB``, ``ADJ``, ``ADV``. Auxiliaries (``AUX``, and Penn's modal
    ``MD``) are not verbs of meaning and answer ``""`` with everything else.
    """
    text = str(pos).strip().upper()
    if text in _UNIVERSAL:
        return _UNIVERSAL[text]
    for prefix, name in _PENN:
        if text.startswith(prefix):
            return name
    return ""
