"""N-gram frequency over time — the culturomics viewer's data (FR-1.6).

The question is: "Why can you only run the co-occurrence VIEWER and not the
N-grams [if your filenames do not embed a date]? ... Why not use Google Ngram
Viewer ... What does culturomics mean". This module is the answer in code: a
time series needs TIME, so when no document carries a date the failure says
exactly that (and why the undated co-occurrence viewer still runs). Culturomics
is quantitative study of culture through word frequencies across a dated
corpus; Google Ngram Viewer does this over Google Books, this tool does it over
YOUR corpus — the same chart, your documents, your queries, inspectable counts.

Dates come from the corpus reader (``date_from_filename`` parses the year out
of names like ``1934_address.txt``); this module never touches filenames or the
filesystem. Queries are 1-3 word phrases matched against each document's token
sequence (consecutive tokens, so "cold war" never matches across a gap — supply
tokens in row order per document and phrases cannot cross sentence boundaries
only if the caller keeps one sequence per document on purpose; with flat
document tokens a phrase could in principle straddle a sentence break, which
is why the CLI and adapters pass whole-document forms and accept that
documented limitation rather than guessing sentence splits here).

Output rows are one per (N-gram, Year) with dated corpus exposure, including
zero-hit years. Years without dated corpus documents have no row and remain
gaps. ``smooth`` is a centered calendar-year window over exposure years for
``Count`` and ``Per Million``; zero-hit exposure years count as zero,
no-corpus years are omitted, and the window truncates at the ends.
``Share of Documents`` always describes the unsmoothed year.
The result also retains raw counts/rates and yearly token/document exposure,
so changing the view or checking a peak never requires reconstructing what
the smoothing hid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from core.conll.schema import Col
from core.result import Diagnostic, Result

__all__ = ["SERIES_COLUMNS", "frame_tokens", "ngram_series"]

SERIES_COLUMNS = [
    "N-gram",
    "Year",
    "Count",
    "Per Million",
    "Share of Documents",
    "Raw Count",
    "Raw Per Million",
    "Corpus Tokens",
    "Corpus Documents",
    "Documents With Hit",
]

_MAX_QUERY_WORDS = 3

_NO_DATES_FIX = (
    "put the year in each filename (e.g. 1934_address.txt) — that is also the answer to the question "
    "question: the co-occurrence viewer can run undated because it does not plot over time, this one cannot"
)


def frame_tokens(frame: pd.DataFrame, *, field: Col = Col.FORM) -> dict[str, list[str]]:
    """Per-document token lists keyed by ``str(Document ID)`` in row order.

    Forms are kept as they are (no lower-casing, no stopword filtering): the
    n-gram viewer counts surface sequences and must see "Cold War" and "of the"
    exactly as the parse split them. Deliberately NOT
    ``core.analysis.lda.tokens_from_frame``, which lower-cases, drops
    non-alphabetic tokens and removes stopwords — three changes that would
    empty every phrase query.
    """
    tokens: dict[str, list[str]] = {}
    if frame.empty or field.value not in frame.columns or Col.DOCUMENT_ID.value not in frame.columns:
        return tokens
    for _, row in frame.iterrows():
        raw = row[field.value]
        if raw is None:
            continue
        try:
            if bool(pd.isna(raw)):
                continue
        except (TypeError, ValueError):
            pass
        text = str(raw).strip()
        if text:
            tokens.setdefault(str(row[Col.DOCUMENT_ID.value]), []).append(text)
    return tokens


def _year_of(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    year = getattr(value, "year", None)
    if isinstance(year, int):
        return year
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _count_phrase(tokens: Sequence[str], phrase: Sequence[str], case_sensitive: bool) -> int:
    width = len(phrase)
    hits = 0
    for index in range(len(tokens) - width + 1):
        window = tokens[index : index + width]
        if all(
            (w == p) if case_sensitive else (w.casefold() == p.casefold()) for w, p in zip(window, phrase, strict=True)
        ):
            hits += 1
    return hits


def _smooth(values: Sequence[float], years: Sequence[int], window: int) -> list[float]:
    """Centered moving average in calendar years; window truncated at the edges."""
    if window <= 1:
        return [float(v) for v in values]
    radius = (window - 1) // 2
    extra = (window - 1) % 2  # even windows take the extra neighbour forward
    smoothed: list[float] = []
    for year, _value in zip(years, values, strict=True):
        lo, hi = year - radius, year + radius + extra
        picked = [float(v) for y, v in zip(years, values, strict=True) if lo <= y <= hi]
        smoothed.append(sum(picked) / len(picked))
    return smoothed


def ngram_series(
    doc_tokens: Mapping[str, Sequence[str]],
    dates: Mapping[str, object],
    queries: Sequence[str],
    *,
    case_sensitive: bool = False,
    smooth: int = 1,
) -> Result[pd.DataFrame]:
    """Year-by-year frequency of 1-3 word phrases over dated documents.

    ``dates`` maps the same document-id keys as ``doc_tokens`` to a year
    (int/str, or a date whose ``.year`` is read) or None. Aggregates per
    (N-gram, Year): ``Count`` is raw matches, ``Per Million`` is count divided
    by that year's total tokens times 1e6 (2dp), ``Share of Documents`` is
    docs-with-a-hit over docs-that-year (4dp). ``smooth`` averages ``Count``
    and ``Per Million`` over exposure years in a centered calendar-year window
    (1 = raw). A year without dated documents is omitted; a year with
    documents and no hits is included as zero. The moving average considers
    only exposure years in the window. ``Share of Documents`` remains raw.
    ``Raw Count``, ``Raw Per Million`` and exposure denominators remain in
    the table even when the displayed Count and Per Million are smoothed.
    """
    if isinstance(smooth, bool) or not isinstance(smooth, int) or smooth < 1:
        return Result.failure(
            Diagnostic.error(
                "NG_VIEWER_BAD_SMOOTH",
                f"smooth must be an integer window of at least 1 year, got {smooth!r}",
                fix="pass --smooth 1 for the raw series",
                smooth=smooth,
            )
        )
    cleaned: list[tuple[str, list[str]]] = []
    for query in queries:
        phrase = str(query).split()
        if not phrase or len(phrase) > _MAX_QUERY_WORDS:
            return Result.failure(
                Diagnostic.error(
                    "NG_VIEWER_BAD_QUERY",
                    f"each query must be a 1-{_MAX_QUERY_WORDS} word phrase, got {query!r}",
                    fix="comma-separate short phrases, e.g. --queries 'war, cold war, united nations'",
                    query=str(query),
                )
            )
        cleaned.append((str(query), phrase))
    if not cleaned:
        return Result.failure(
            Diagnostic.error(
                "NG_VIEWER_BAD_QUERY",
                "no queries given; name the words or phrases to track",
                fix="pass --queries 'war, peace'",
            )
        )

    if not doc_tokens:
        return Result.success(pd.DataFrame(columns=SERIES_COLUMNS))

    dated: dict[str, tuple[int, Sequence[str]]] = {}
    undated = 0
    for doc_id, tokens in doc_tokens.items():
        year = _year_of(dates.get(doc_id))
        if year is None:
            undated += 1
            continue
        dated[doc_id] = (year, tokens)
    if not dated:
        return Result.failure(
            Diagnostic.error(
                "NG_VIEWER_NO_DATES",
                "the n-gram viewer plots frequency over time, so it needs dates, and no document carries one",
                fix=_NO_DATES_FIX,
            )
        )

    diags: list[Diagnostic] = []
    if undated:
        diags.append(
            Diagnostic.warning(
                "NG_VIEWER_UNDATED_DROPPED",
                f"dropped {undated} document(s) with no date from the time series; "
                "put the year in their filenames to include them",
                dropped=undated,
            )
        )

    tokens_per_year: dict[int, int] = {}
    docs_per_year: dict[int, int] = {}
    for year, tokens in dated.values():
        tokens_per_year[year] = tokens_per_year.get(year, 0) + len(tokens)
        docs_per_year[year] = docs_per_year.get(year, 0) + 1

    rows: list[dict[str, object]] = []
    misses: list[str] = []
    for query, phrase in cleaned:
        hits_by_year: dict[int, int] = {}
        docs_hit: dict[int, int] = {}
        total_hits = 0
        for year, tokens in dated.values():
            hits = _count_phrase(tokens, phrase, case_sensitive)
            if hits:
                hits_by_year[year] = hits_by_year.get(year, 0) + hits
                docs_hit[year] = docs_hit.get(year, 0) + 1
                total_hits += hits
        if not total_hits:
            misses.append(query)
        years = sorted(docs_per_year)
        raw_counts = [hits_by_year.get(year, 0) for year in years]
        raw_rates = [
            1000000.0 * hits_by_year.get(year, 0) / tokens_per_year[year] if tokens_per_year[year] else 0.0
            for year in years
        ]
        counts = [float(value) for value in raw_counts]
        per_million = raw_rates
        if smooth > 1:
            counts = [round(value, 4) for value in _smooth(counts, years, smooth)]
            per_million = [round(value, 2) for value in _smooth(per_million, years, smooth)]
        else:
            per_million = [round(value, 2) for value in per_million]
        for index, year in enumerate(years):
            rows.append(
                {
                    "N-gram": query,
                    "Year": year,
                    "Count": int(counts[index]) if smooth == 1 else counts[index],
                    "Per Million": per_million[index],
                    "Share of Documents": round(docs_hit.get(year, 0) / docs_per_year[year], 4),
                    "Raw Count": raw_counts[index],
                    "Raw Per Million": round(raw_rates[index], 2),
                    "Corpus Tokens": tokens_per_year[year],
                    "Corpus Documents": docs_per_year[year],
                    "Documents With Hit": docs_hit.get(year, 0),
                }
            )
    if misses:
        diags.append(
            Diagnostic.info(
                "NG_VIEWER_NO_HITS",
                f"no occurrences anywhere in the dated corpus for: {', '.join(misses)}",
                queries=misses,
            )
        )
    out = pd.DataFrame(rows, columns=SERIES_COLUMNS)
    return Result.success(out, *diags)
