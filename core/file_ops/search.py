"""Search, filename and sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re

import pandas as pd

from core.io.reader import Corpus, Document
from core.result import Diagnostic, Result

__all__ = ["CsvHitsResult", "filename_sanitize", "sample_corpus", "search_csv_frame", "search_in_text"]


def search_in_text(
    text: str, query: str, case_sensitive: bool = False, use_regex: bool = False
) -> Result[list[tuple[int, str]]]:
    """Return ``(line_number, line)`` for lines matching *query*."""
    if not query:
        return Result.failure(Diagnostic.error("SEARCH_EMPTY_QUERY", "query must be non-empty"))
    # Compile once, before the loop, so an invalid pattern fails even when
    # the text has no lines to iterate.
    pattern: re.Pattern[str] | None = None
    if use_regex:
        try:
            pattern = re.compile(query, flags=0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            return Result.failure(Diagnostic.error("SEARCH_BAD_REGEX", f"invalid regex: {exc}"))
    needle = query if case_sensitive else query.lower()
    hits: list[tuple[int, str]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if pattern is not None:
            if pattern.search(line):
                hits.append((idx, line))
        else:
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                hits.append((idx, line))
    return Result.success(hits)


def filename_sanitize(name: str, replacement: str = "_") -> str:
    """Make *name* safe for the filesystem."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', replacement, name)
    sanitized = sanitized.strip().strip(".")
    return sanitized or "untitled"


def sample_corpus(corpus: Corpus, k: int, seed: int = 0) -> Result[Corpus]:
    """Return a random sample of *k* documents from *corpus*."""
    if k <= 0:
        return Result.failure(Diagnostic.error("SAMPLE_BAD_K", f"k must be positive, got {k}"))
    if k > len(corpus.docs):
        return Result.failure(Diagnostic.error("SAMPLE_TOO_LARGE", f"k={k} > corpus size {len(corpus.docs)}"))
    rng = random.Random(seed)
    sampled = tuple(rng.sample(list(corpus.docs), k))
    new_docs = tuple(
        Document(doc_id=i + 1, path=doc.path, text=doc.text, date=doc.date, sha256=doc.sha256)
        for i, doc in enumerate(sampled)
    )
    fingerprint = corpus.sha256[:8] + f"-sample{k}"
    return Result.success(Corpus(docs=new_docs, sha256=fingerprint))


@dataclass(frozen=True, slots=True)
class CsvHitsResult:
    frame: pd.DataFrame  # Row (1-based data row), Column, Value

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def search_csv_frame(
    frame: pd.DataFrame,
    query: str,
    *,
    case_sensitive: bool = False,
    use_regex: bool = False,
) -> Result[CsvHitsResult]:
    """Search every cell of a plain CSV frame (no CoNLL schema required).

    Hits are row-major (row 1's columns in table order, then row 2). A missing
    cell is blank, not the text ``"nan"``/``"None"``.
    """
    if not query:
        return Result.failure(Diagnostic.error("SEARCH_EMPTY_QUERY", "query must be non-empty"))
    pattern: re.Pattern[str] | None = None
    if use_regex:
        try:
            pattern = re.compile(query, flags=0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            return Result.failure(Diagnostic.error("SEARCH_BAD_REGEX", f"invalid regex: {exc}"))
    needle = query if case_sensitive else query.lower()
    # One string conversion and one mask per column, then gather the matching
    # cells. The old loop materialized a pandas Series per row and lowered
    # every cell one at a time; a 100,000-row CSV paid Python overhead for
    # every cell even when the query matched a handful.
    pieces: list[pd.DataFrame] = []
    for column_index, column in enumerate(frame.columns):
        texts = frame[column].astype("string").fillna("").astype(str)
        if pattern is not None:
            mask = texts.str.contains(pattern, na=False)
        elif case_sensitive:
            mask = texts.str.contains(needle, regex=False, na=False)
        else:
            mask = texts.str.lower().str.contains(needle, regex=False, na=False)
        positions = mask.to_numpy().nonzero()[0]
        if not positions.size:
            continue
        pieces.append(
            pd.DataFrame(
                {
                    "Row": positions + 1,
                    "_column": column_index,
                    "Column": str(column),
                    "Value": texts.to_numpy()[positions],
                }
            )
        )
    if not pieces:
        return Result.success(CsvHitsResult(frame=pd.DataFrame(columns=["Row", "Column", "Value"])))
    out = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["Row", "_column"], kind="stable")
        .drop(columns="_column")
        .reset_index(drop=True)
    )
    return Result.success(CsvHitsResult(frame=out[["Row", "Column", "Value"]]))
