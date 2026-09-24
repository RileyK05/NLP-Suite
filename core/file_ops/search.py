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
    """Search every cell of a plain CSV frame (no CoNLL schema required)."""
    if not query:
        return Result.failure(Diagnostic.error("SEARCH_EMPTY_QUERY", "query must be non-empty"))
    pattern: re.Pattern[str] | None = None
    if use_regex:
        try:
            pattern = re.compile(query, flags=0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            return Result.failure(Diagnostic.error("SEARCH_BAD_REGEX", f"invalid regex: {exc}"))
    needle = query if case_sensitive else query.lower()
    rows: list[dict[str, object]] = []
    for row_idx, (_, record) in enumerate(frame.iterrows(), start=1):
        for column in frame.columns:
            raw = record[column]
            cell = "" if raw is None or bool(pd.isna(raw)) else str(raw)
            if pattern is not None:
                matched = pattern.search(cell) is not None
            else:
                haystack = cell if case_sensitive else cell.lower()
                matched = needle in haystack
            if matched:
                rows.append({"Row": row_idx, "Column": str(column), "Value": cell})
    out = pd.DataFrame(rows, columns=["Row", "Column", "Value"])
    return Result.success(CsvHitsResult(frame=out))
