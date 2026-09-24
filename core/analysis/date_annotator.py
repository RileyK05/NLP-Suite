"""Normalized date extraction (SUTime-style, regex) (FR-4.8).

CoreNLP's normalized-date annotator answers "are there comparable
dates/times?" -- so every date expression is extracted from raw document text
and normalized to ISO, including PARTIAL dates (a decade is still a decade).
No parse is needed: this is text-level, like the legacy SUTime call.

Recognized surfaces (minimum): ISO ``2020-03-05`` / ``2020-03``;
``March 5, 2020`` / ``March 5th, 2020`` / ``Mar 5 2020``; ``5 March 2020``;
``03/05/2020`` and ``3/5/2020``; bare years (``1934``); decades (``1930s``).
Month names are full English names plus three-letter abbreviations.

AMBIGUITY, documented rather than guessed: ``03/05/2020`` is read US
month/day order (March 5), matching the legacy's English-language default. A
corpus written day/month will be misread; check a sample before trusting
numeric dates. Day bounds are 1..31 only -- February 30 is normalized rather
than calendar-validated, because partial dates are kept partial.

Years outside ``min_year``..``max_year`` are not the corpus's dates (``1,500
soldiers`` must not become a year): they are skipped and counted, reported
as a ``DATE_YEAR_SKIPPED`` WARNING per document rather than dropped silently.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["annotate_corpus", "annotate_dates", "summarize_dates"]

_COLUMNS = ["Document", "Document ID", "Date ID", "Surface", "Normalized", "Type"]
_SUMMARY_COLUMNS = ["Document", "Document ID", "Dates", "Distinct", "Earliest", "Latest"]

_MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
# Longest first so "september" wins over "sep" at the same position.
_MONTHS_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ORDINAL = r"(?:st|nd|rd|th)?"
_SEP = r"(?:\s*,\s*|\s+)"

_DATE_RE = re.compile(
    rf"(?P<iso>(?<!\d)(?P<iso_y>\d{{4}})-(?P<iso_m>\d{{2}})-(?P<iso_d>\d{{2}})(?!\d))"
    rf"|(?P<isom>(?<!\d)(?P<isom_y>\d{{4}})-(?P<isom_m>\d{{2}})(?!\d))"
    rf"|(?P<mdy>(?<![A-Za-z])(?P<mdy_mon>{_MONTHS_ALT})(?![A-Za-z])\.?\s+"
    rf"(?P<mdy_d>\d{{1,2}}){_ORDINAL}\.?{_SEP}(?P<mdy_y>\d{{4}})(?!\d))"
    rf"|(?P<dmy>(?<!\d)(?P<dmy_d>\d{{1,2}}){_ORDINAL}\.?\s+"
    rf"(?P<dmy_mon>{_MONTHS_ALT})(?![A-Za-z])\.?{_SEP}(?P<dmy_y>\d{{4}})(?!\d))"
    rf"|(?P<my>(?<![A-Za-z])(?P<my_mon>{_MONTHS_ALT})(?![A-Za-z])\.?\s+(?P<my_y>\d{{4}})(?!\d))"
    rf"|(?P<us>(?<!\d)(?P<us_m>\d{{1,2}})/(?P<us_d>\d{{1,2}})/(?P<us_y>\d{{4}})(?!\d))"
    rf"|(?P<dec>(?<!\w)(?P<dec_y>\d{{4}})s(?!\w))"
    rf"|(?P<yr>(?<!\d)(?P<yr_y>\d{{4}})(?!\d))",
    re.IGNORECASE,
)


def _month_number(name: str) -> int:
    return _MONTHS[name.strip().lower()]


def _parse_match(match: re.Match[str]) -> tuple[str, str, int] | None:
    """(Normalized, Type, year) for one regex hit, or None when it is not a date.

    Overreach is refused here (month 13, day 45) rather than emitted and
    regretted: those spans are regex artifacts, not the corpus's dates.
    """
    groups = match.groupdict()
    if groups["iso"]:
        year, month, day = int(groups["iso_y"]), int(groups["iso_m"]), int(groups["iso_d"])
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        return (f"{year:04d}-{month:02d}-{day:02d}", "date", year)
    if groups["isom"]:
        year, month = int(groups["isom_y"]), int(groups["isom_m"])
        if not 1 <= month <= 12:
            return None
        return (f"{year:04d}-{month:02d}", "month_year", year)
    if groups["mdy"]:
        year, month, day = int(groups["mdy_y"]), _month_number(groups["mdy_mon"]), int(groups["mdy_d"])
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        return (f"{year:04d}-{month:02d}-{day:02d}", "date", year)
    if groups["dmy"]:
        year, month, day = int(groups["dmy_y"]), _month_number(groups["dmy_mon"]), int(groups["dmy_d"])
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        return (f"{year:04d}-{month:02d}-{day:02d}", "date", year)
    if groups["my"]:
        year, month = int(groups["my_y"]), _month_number(groups["my_mon"])
        return (f"{year:04d}-{month:02d}", "month_year", year)
    if groups["us"]:
        # US month/day order -- the documented ambiguity, not a guess per run.
        year, month, day = int(groups["us_y"]), int(groups["us_m"]), int(groups["us_d"])
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        return (f"{year:04d}-{month:02d}-{day:02d}", "date", year)
    if groups["dec"]:
        year = int(groups["dec_y"])
        return (str(year), "decade", year)
    if groups["yr"]:
        year = int(groups["yr_y"])
        return (str(year), "year", year)
    return None


def annotate_dates(
    text: str,
    *,
    document: str = "",
    document_id: str = "",
    min_year: int = 1000,
    max_year: int = 2100,
    date_id_start: int = 1,
) -> Result[pd.DataFrame]:
    """Date expressions in *text*, normalized to ISO.

    Rows: Document, Document ID, Date ID, Surface, Normalized, Type
    (date|month_year|year|decade). Out-of-range years are skipped and
    counted in a ``DATE_YEAR_SKIPPED`` WARNING; empty text yields the empty
    frame with its columns.
    """
    if min_year > max_year:
        return Result.failure(
            Diagnostic.error(
                "DATE_BAD_RANGE",
                f"min_year {min_year} is above max_year {max_year}",
                min_year=min_year,
                max_year=max_year,
            )
        )
    rows: list[dict[str, object]] = []
    skipped = 0
    next_id = date_id_start
    for match in _DATE_RE.finditer(text or ""):
        parsed = _parse_match(match)
        if parsed is None:
            continue
        normalized, kind, year = parsed
        if not min_year <= year <= max_year:
            skipped += 1
            continue
        rows.append(
            {
                "Document": document,
                "Document ID": document_id,
                "Date ID": next_id,
                "Surface": match.group(0),
                "Normalized": normalized,
                "Type": kind,
            }
        )
        next_id += 1
    diags: tuple[Diagnostic, ...] = ()
    if skipped:
        diags = (
            Diagnostic.warning(
                "DATE_YEAR_SKIPPED",
                f"skipped {skipped} date-like candidate(s) outside {min_year}..{max_year}",
                document=document,
                document_id=document_id,
                skipped=skipped,
                min_year=min_year,
                max_year=max_year,
            ),
        )
    return Result.success(pd.DataFrame(rows, columns=_COLUMNS), *diags)


def annotate_corpus(
    docs: Sequence[Sequence[object]],
    *,
    min_year: int = 1000,
    max_year: int = 2100,
    date_id_start: int = 1,
) -> Result[pd.DataFrame]:
    """Date rows for ``(document, document_id, text)`` tuples, IDs continuing.

    Date IDs run 1..n across the whole corpus (start at ``date_id_start``)
    so two documents can share one comparable identifier space.
    """
    if min_year > max_year:
        return Result.failure(
            Diagnostic.error(
                "DATE_BAD_RANGE",
                f"min_year {min_year} is above max_year {max_year}",
                min_year=min_year,
                max_year=max_year,
            )
        )
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    next_id = date_id_start
    for entry in docs:
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 3:
            return Result.failure(
                Diagnostic.error("DATE_BAD_DOCS", "each document must be a (document, document_id, text) tuple")
            )
        document, document_id, text = str(entry[0]), str(entry[1]), str(entry[2])
        result = annotate_dates(
            text,
            document=document,
            document_id=document_id,
            min_year=min_year,
            max_year=max_year,
            date_id_start=next_id,
        )
        if result.value is None:
            return Result.failure(*result.diagnostics)
        frame = result.unwrap()
        rows.extend(frame.to_dict("records"))
        diags.extend(result.diagnostics)
        next_id += len(frame)
    return Result.success(pd.DataFrame(rows, columns=_COLUMNS), *diags)


def summarize_dates(annotated: pd.DataFrame) -> Result[pd.DataFrame]:
    """Per-document spread of the extracted dates.

    Dates is the row count, Distinct the distinct Normalized values, and
    Earliest/Latest the min/max Normalized strings (mixed-precision ISO
    values order lexicographically, which is what partial dates allow).
    """
    for need in ("Document", "Document ID", "Normalized"):
        if need not in annotated.columns:
            return Result.failure(Diagnostic.error("DATE_MISSING_COLUMN", f"missing {need!r}", missing=need))
    if annotated.empty:
        return Result.success(pd.DataFrame(columns=_SUMMARY_COLUMNS))
    rows: list[dict[str, object]] = []
    for (doc_id, doc), group in annotated.groupby(["Document ID", "Document"], sort=False):
        values = [str(v) for v in group["Normalized"]]
        rows.append(
            {
                "Document": doc,
                "Document ID": doc_id,
                "Dates": len(values),
                "Distinct": len(set(values)),
                "Earliest": min(values),
                "Latest": max(values),
            }
        )
    return Result.success(pd.DataFrame(rows, columns=_SUMMARY_COLUMNS))
