"""What a document's details say about where it sits: its date and its order.

One function per question, used by every caller that has a project document's
details (the runner, selections, the server), so a date typed on the Corpus
page means the same thing to all of them. :func:`core.io.reader.date_from_filename`
stays the detector for names without details.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
import re

from core.io.reader import date_from_filename

__all__ = ["DETAIL_PREFIX", "FIELD_VALUE_PREFIX", "document_date", "document_order", "field_value"]

_ISO = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")

#: A detail whose name a tool's own result column already uses is shown under
#: this prefix (``Detail: Tokens``), so a detail never overwrites a measure.
#: One spelling, here, because both the executor that writes the column and
#: the figures that group by it must read the same one.
DETAIL_PREFIX = "Detail: "

#: A parameter value that names a document detail does so with this prefix
#: (``field:Party``): one spelling shared by the parameter checks, the tools
#: that read it, and the desktop menus that offer it.
FIELD_VALUE_PREFIX = "field:"


def field_value(fields: Mapping[str, str] | None, name: str) -> str:
    """A detail's value, whatever the case of its name; empty when absent."""
    if not fields:
        return ""
    wanted = name.casefold()
    for key, value in fields.items():
        if key.casefold() == wanted:
            return str(value).strip()
    return ""


def document_date(fields: Mapping[str, str] | None, name: str = "") -> date | None:
    """The document's date: its ``Date`` detail (``YYYY-MM-DD``, ``YYYY-MM`` or ``YYYY``), else its file name's.

    A year alone is read as the first of January and a month as its first day:
    enough to place a document on a time axis and in the right year.
    """
    text = field_value(fields, "Date")
    if text:
        match = _ISO.match(text)
        if match:
            year, month, day = match.groups()
            try:
                return date(int(year), int(month or 1), int(day or 1))
            except ValueError:
                return None
        return None
    return date_from_filename(Path(name)) if name else None


def document_order(fields: Mapping[str, str] | None) -> float | None:
    """The document's ``Order`` detail as a number (chapter 3 -> 3.0), or None."""
    text = field_value(fields, "Order")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
