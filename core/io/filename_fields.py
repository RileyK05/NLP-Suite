"""Document details read from file names (docs/internal/PLAN_0.5.0.md 1.4.1).

A folder of speeches named ``1934-01-03_franklin d roosevelt_sotu.txt``
already says when each was given, by whom, and what kind of speech it is.
Until now the suite read only the date, and the figure helpers parsed the
speaker again with a regex of their own. This module reads the whole pattern
once, for every caller.

The rule is to find a *common* pattern and never to guess. A template must fit
at least :data:`MIN_FIT` of the names, and must be one of a few shapes whose
parts can be named honestly. Anything else gives ``None``, and the names it
missed are listed, so the Corpus page can say which files did not fit.

Pure: names in, fields out. No I/O.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

from core.io.reader import date_from_filename

__all__ = [
    "MIN_FIT",
    "Template",
    "apply_template",
    "detect_template",
    "order_value",
    "speaker_name",
]

#: The share of names a template must fit before it is used for any of them.
MIN_FIT = 0.8
#: A part with at most this many distinct values (or one in five names) is a kind, not a name.
_KIND_MAX_VALUES = 6

_SEPARATORS = ("_", " - ")
_YEAR = re.compile(r"^(1[5-9]|20)\d{2}$")
_ORDER = re.compile(r"^(?:ch(?:apter)?|part|section|no|#)?\s*\.?\s*(\d{1,4})$", re.IGNORECASE)
_ROMAN = re.compile(r"^(?=[IVXLC])(C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_ORDER = re.compile(r"^(?:(?:chapter|part|book|volume)\s+)?([IVXLC]+)$", re.IGNORECASE)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _stem(name: str) -> str:
    text = str(name).strip()
    return text.rsplit(".", 1)[0] if re.search(r"\.[A-Za-z0-9]{1,5}$", text) else text


def _date_value(part: str) -> str | None:
    """An ISO date (or bare year) when the whole part is one, else None."""
    text = part.strip()
    if _YEAR.match(text):
        return text
    found = date_from_filename(Path(text))
    if found is None:
        return None
    # The whole part must be the date: "1934-01-03", not "speech 1934-01-03 draft".
    digits = re.sub(r"\D", "", text)
    if len(digits) not in (6, 8) or re.search(r"[A-Za-z]", text):
        return None
    return found.isoformat() if len(digits) == 8 else found.isoformat()[:7]  # noqa: PLR2004 - YYYYMMDD vs YYYYMM


def _roman(text: str) -> int | None:
    upper = text.upper()
    if not _ROMAN.match(upper):
        return None
    total = 0
    for index, letter in enumerate(upper):
        value = _ROMAN_VALUES[letter]
        following = _ROMAN_VALUES[upper[index + 1]] if index + 1 < len(upper) else 0
        total += -value if value < following else value
    return total or None


def order_value(part: str) -> str | None:
    """A position ("1", "01", "ch1", "Chapter 12", "IV") as a number, else None."""
    text = part.strip()
    match = _ORDER.match(text)
    if match:
        return str(int(match.group(1)))
    roman = _ROMAN_ORDER.match(text)
    if roman:
        value = _roman(roman.group(1))
        return None if value is None else str(value)
    return None


def speaker_name(part: str) -> str:
    """A speaker as file names spell them, as a reader writes it: ``Franklin D Roosevelt``."""
    return " ".join(word.capitalize() for word in part.split())


@dataclass(frozen=True, slots=True)
class Template:
    """How a set of file names is built: its parts, in order, and how well it fits."""

    parts: tuple[str, ...]
    separator: str
    fits: int
    total: int
    misses: tuple[str, ...] = ()

    def describe(self) -> str:
        """For the Corpus page: "Date, then Speaker, then Kind, separated by _ (fits 87 of 87)"."""
        joined = ", then ".join(self.parts)
        separator = f" separated by {self.separator.strip() or repr(self.separator)}" if len(self.parts) > 1 else ""
        return f"{joined}{separator} (fits {self.fits} of {self.total})"


def _split(stem: str, separator: str) -> list[str]:
    return [part.strip() for part in stem.split(separator)]


def _classify(values: list[str]) -> str:
    if all(_date_value(value) for value in values):
        return "date"
    if all(order_value(value) for value in values):
        return "order"
    return "text"


#: The shapes a reader would recognise, and what each part is called.
_SHAPES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("date", "text"): ("Date", "Speaker"),
    ("date", "text", "text"): ("Date", "Speaker", "Kind"),
    ("order", "text"): ("Order", "Title"),
    ("order",): ("Order",),
    ("text", "date"): ("Title", "Date"),
    ("text", "text", "date"): ("Author", "Title", "Date"),
}


def _name_parts(kinds: list[str], columns: list[list[str]]) -> tuple[str, ...] | None:
    """Name each part, for the shapes a reader would recognise; None for anything else."""

    def low_cardinality(values: list[str]) -> bool:
        distinct = len(set(value.casefold() for value in values))
        return distinct <= max(_KIND_MAX_VALUES, len(values) // 5) and distinct < len(values)

    shape = tuple(kinds)
    # Two shapes are only what they look like under a condition: a third part
    # that repeats (sotu, ina) is a kind, and a trailing date that is a bare
    # year is a publication year after author and title.
    if shape == ("date", "text", "text") and not low_cardinality(columns[2]):
        return None
    if shape == ("text", "text", "date") and not all(_YEAR.match(value) for value in columns[2]):
        return None
    return _SHAPES.get(shape)


def _value(part_name: str, raw: str) -> str:
    if part_name == "Date":
        return _date_value(raw) or ""
    if part_name == "Order":
        return order_value(raw) or ""
    if part_name == "Speaker":
        return speaker_name(raw)
    return raw.strip()


def detect_template(names: Sequence[str]) -> Template | None:
    """The pattern most of *names* share, or None when there is none worth naming.

    Needs at least two names: one name alone cannot show which parts vary.
    """
    stems = [_stem(name) for name in names]
    if len(stems) < 2:  # noqa: PLR2004 - a pattern is what several names share
        return None
    for separator in _SEPARATORS:
        split = [_split(stem, separator) for stem in stems]
        width, count = Counter(len(parts) for parts in split).most_common(1)[0]
        if count < MIN_FIT * len(stems):
            continue
        fitting = [parts for parts in split if len(parts) == width]
        columns = [[parts[i] for parts in fitting] for i in range(width)]
        kinds = [_classify(column) for column in columns]
        named = _name_parts(kinds, columns)
        if named is None:
            continue
        template = Template(named, separator, 0, len(stems))
        misses = tuple(name for name in names if not apply_template(name, template))
        fits = len(stems) - len(misses)
        if fits < MIN_FIT * len(stems):
            continue
        return Template(named, separator, fits, len(stems), misses)
    # One part per name: "Chapter 1", "Chapter 2", ...
    if all(order_value(stem) for stem in stems):
        return Template(("Order",), "", len(stems), len(stems))
    return None


def apply_template(name: str, template: Template) -> dict[str, str]:
    """*name*'s fields under *template*; empty when the name does not fit it."""
    stem = _stem(name)
    parts = [stem.strip()] if not template.separator else _split(stem, template.separator)
    if len(parts) != len(template.parts):
        return {}
    fields = {part_name: _value(part_name, raw) for part_name, raw in zip(template.parts, parts, strict=True)}
    if any(not value for value in fields.values()):
        return {}
    return fields
