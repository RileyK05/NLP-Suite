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

__all__ = ["NOUN_TAGS", "is_noun_tag"]


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
