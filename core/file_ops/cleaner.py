"""File cleaner — normalizes whitespace, line endings and control characters."""

from __future__ import annotations

import re

from core.result import Result

__all__ = ["clean_text"]


_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")


def clean_text(
    text: str,
    *,
    normalize_whitespace: bool = True,
    remove_empty_lines: bool = False,
    remove_control_chars: bool = True,
    normalize_line_endings: bool = True,
) -> Result[str]:
    """Return a cleaned copy of *text*.

    Cleaning is pure and never touches disk. The caller decides where to write.
    """
    cleaned = text
    if normalize_line_endings:
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if remove_control_chars:
        cleaned = _CONTROL_RE.sub("", cleaned)
    if normalize_whitespace:
        cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
        cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    if remove_empty_lines:
        cleaned = "\n".join(line for line in cleaned.split("\n") if line.strip())
    cleaned = cleaned.strip()
    return Result.success(cleaned)
