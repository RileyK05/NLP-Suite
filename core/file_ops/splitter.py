"""Splitter — one parameterized engine for all splitting strategies."""

from __future__ import annotations

import re

from core.result import Diagnostic, Result

__all__ = ["SplitMethod", "split_text"]


SplitMethod = str


def split_text(
    text: str,
    method: SplitMethod = "by_length",
    chunk_size: int = 1000,
    overlap: int = 0,
    delimiter: str = "\n\n",
    keyword: str = "",
) -> Result[list[str]]:
    """Split *text* into chunks.

    Methods:
    - ``by_length`` — fixed character length
    - ``by_words`` — fixed word count
    - ``by_lines`` — fixed line count
    - ``by_delimiter`` — split on *delimiter*
    - ``by_keyword`` — split before each occurrence of *keyword*
    """
    if not text:
        return Result.failure(Diagnostic.error("SPLIT_EMPTY", "text is empty"))
    if chunk_size <= 0:
        return Result.failure(Diagnostic.error("SPLIT_BAD_SIZE", f"chunk_size must be positive, got {chunk_size}"))
    if overlap < 0:
        return Result.failure(Diagnostic.error("SPLIT_BAD_OVERLAP", f"overlap must be >= 0, got {overlap}"))
    if overlap >= chunk_size:
        # by_length would never advance (start = end - overlap <= start), and
        # by_words/by_lines would degrade to a step of 1. Refuse instead.
        return Result.failure(
            Diagnostic.error("SPLIT_BAD_OVERLAP", f"overlap must be < chunk_size, got {overlap} >= {chunk_size}")
        )
    if method == "by_length":
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            if overlap and end < len(text):
                start = end - overlap
            else:
                start = end
        return Result.success(chunks)
    if method == "by_words":
        words = text.split()
        chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), max(1, chunk_size - overlap))]
        return Result.success(chunks)
    if method == "by_lines":
        lines = text.splitlines()
        chunks = ["\n".join(lines[i : i + chunk_size]) for i in range(0, len(lines), max(1, chunk_size - overlap))]
        return Result.success(chunks)
    if method == "by_delimiter":
        if not delimiter:
            return Result.failure(Diagnostic.error("SPLIT_BAD_DELIM", "delimiter must be non-empty"))
        parts = text.split(delimiter)
        return Result.success(parts)
    if method == "by_keyword":
        if not keyword:
            return Result.failure(Diagnostic.error("SPLIT_BAD_KEYWORD", "keyword must be non-empty"))
        pattern = re.compile(re.escape(keyword))
        indices = [match.start() for match in pattern.finditer(text)]
        if not indices:
            return Result.success([text])
        # Split before *each* occurrence, per the docstring. A keyword at
        # position 0 would otherwise produce a leading empty chunk.
        chunks = []
        if indices[0] > 0:
            chunks.append(text[: indices[0]])
        for i, idx in enumerate(indices):
            end = indices[i + 1] if i + 1 < len(indices) else len(text)
            chunks.append(text[idx:end])
        return Result.success(chunks)
    return Result.failure(Diagnostic.error("SPLIT_UNKNOWN_METHOD", f"unknown split method {method!r}", method=method))
