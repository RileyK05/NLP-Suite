"""Rejoin a parse's tokens into text as it was written.

A parser gives punctuation, contractions and currency signs tokens of their
own, so a sentence rebuilt by joining tokens with spaces reads "we 've
worked , for $ 30,000 ." -- wrong on screen, and slightly wrong for a
sentence model, which was trained on text as people write it. Only the
attachments a reader would never write apart are reversed.
"""

from __future__ import annotations

from collections.abc import Iterable
import re

__all__ = ["detokenize"]

# Typographic marks, spelled as escapes so the source stays plain ASCII.
_APOSTROPHE = "'" + chr(0x2019)
_LEFT_QUOTE, _RIGHT_QUOTE = chr(0x201C), chr(0x201D)
_DASHES = "-" + chr(0x2010) + chr(0x2013)

_CLITICS = r"(?:[" + _APOSTROPHE + r"](?:s|re|ve|ll|m|d)|n[" + _APOSTROPHE + r"]t)\b"
_CLOSING = re.compile(r"\s+([.,!?;:%)\]}>…]|" + _CLITICS + ")")
_OPENING = re.compile(r"([(\[{<$£€#])\s+")
_HYPHEN = re.compile(r"(\w)\s+([" + _DASHES + r"])\s+(\w)")
_QUOTED = re.compile(r'"\s+(.*?)\s+"')
_CURLY = re.compile(_LEFT_QUOTE + r"\s+(.*?)\s+" + _RIGHT_QUOTE)


def detokenize(tokens: Iterable[str] | str) -> str:
    """*tokens* (or a space-joined string of them) as readable text."""
    text = tokens if isinstance(tokens, str) else " ".join(str(token) for token in tokens if str(token).strip())
    text = " ".join(text.split())
    text = _CLOSING.sub(r"\1", text)
    text = _OPENING.sub(r"\1", text)
    text = _HYPHEN.sub(r"\1\2\3", text)
    text = _QUOTED.sub(r'"\1"', text)
    text = _CURLY.sub(_LEFT_QUOTE + r"\1" + _RIGHT_QUOTE, text)
    return text.strip()
