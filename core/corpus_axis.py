"""How a corpus's documents line up: by date, by order, or not at all (docs/PLAN_0.5.0.md 1.6).

A corpus of speeches lines up in time; a novel split into chapters lines up
by chapter number; a folder of essays does not line up at all. Every figure
that used to say "over time" asks the axis instead, so the same trend line
reads "across the chapters" for a book and still reads "over time" for the
State of the Union, where nothing changes.

Pure: it reads each document's date and ``Order`` detail and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
import re
from typing import Literal

from core.io.document_fields import document_order
from core.io.reader import Corpus

__all__ = ["AXIS_KINDS", "DASH", "DEFAULT_NOUN", "Axis", "AxisKind", "along", "axis_of", "period_key", "plural"]

AxisKind = Literal["time", "order", "none"]
AXIS_KINDS: tuple[AxisKind, ...] = ("time", "order", "none")
#: What one step along an order axis is called until the project names it (Chapter, Session).
DEFAULT_NOUN = "Document"
#: A before and an after: fewer periods than this and there is no change to show.
PERIODS_NEEDED = 2
#: An axis needs at least this many placed documents to order anything.
_PLACED_NEEDED = 2
#: Order blocks: how many, and the fewest documents one may hold.
_BLOCKS = 4
_BLOCK_MIN = 2
#: Between the ends of a range of chapters ("Chapters 1" en dash "15").
DASH = chr(0x2013)


@dataclass(frozen=True, slots=True)
class Axis:
    """Where each document sits, and what one step along the way is called.

    ``positions`` are numbers a figure can put on x: a decimal year for time
    (1934.0 for 3 January 1934), the order value for order. ``labels`` are
    what a reader is shown for the same document ("1934-01-03", "Chapter 3").
    Documents with no date (time) or no order (order) are simply absent.
    """

    kind: AxisKind
    noun: str
    positions: Mapping[str, float] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)

    @property
    def placed(self) -> bool:
        """Whether the axis orders anything at all."""
        return self.kind != "none" and len(self.positions) >= _PLACED_NEEDED

    def periods(self, grouping: Mapping[str, str] | None = None, blocks: int = _BLOCKS) -> dict[str, str]:
        """Each placed document's period, for before-and-after comparisons.

        Time: its decade, or its year when every document shares one decade.
        Order: the value of *grouping* (a detail such as Volume) when given,
        else equal-count blocks labelled "Chapters 1-15". Empty when fewer
        than two periods would result: change needs a before and an after.
        """
        if not self.placed:
            return {}
        if self.kind == "time":
            decades = {doc: f"{int(at) // 10 * 10}s" for doc, at in self.positions.items()}
            if len(set(decades.values())) >= PERIODS_NEEDED:
                return decades
            years = {doc: str(int(at)) for doc, at in self.positions.items()}
            return years if len(set(years.values())) >= PERIODS_NEEDED else {}
        if grouping:
            grouped = {doc: grouping[doc] for doc in self.positions if grouping.get(doc)}
            if len(set(grouped.values())) >= PERIODS_NEEDED:
                return grouped
        return self._blocks(blocks)

    def _blocks(self, blocks: int) -> dict[str, str]:
        ordered = sorted(self.positions, key=lambda doc: (self.positions[doc], doc))
        count = min(blocks, len(ordered) // _BLOCK_MIN)
        if count < PERIODS_NEEDED:
            return {}
        size, extra = divmod(len(ordered), count)
        found: dict[str, str] = {}
        start = 0
        for index in range(count):
            members = ordered[start : start + size + (1 if index < extra else 0)]
            start += len(members)
            first, last = _number(self.positions[members[0]]), _number(self.positions[members[-1]])
            label = f"{self.noun} {first}" if first == last else f"{plural(self.noun)} {first}{DASH}{last}"
            found.update(dict.fromkeys(members, label))
        return found


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _decimal_year(when: date) -> float:
    start = date(when.year, 1, 1)
    days = (date(when.year + 1, 1, 1) - start).days
    return round(when.year + (when - start).days / days, 6)


_DIGITS = re.compile(r"(\d+)")


def period_key(label: object) -> tuple[object, ...]:
    """Sort key for period labels in reading order: "Chapters 1-3" before "Chapters 10-12".

    Sorting labels as text put chapter 10 before chapter 1, so "first period
    against last" compared the wrong chapters. Decades and years sort as they
    always did.
    """
    parts = _DIGITS.split(str(label))
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in parts)


def plural(noun: str) -> str:
    """ "Chapter" -> "Chapters", "Diary entry" -> "Diary entries"; enough for the nouns people give an order."""
    if not noun or noun.lower().endswith("s"):
        return noun
    if noun.lower().endswith("y") and noun[-2:-1].lower() not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def along(kind: str, noun: str = "") -> str:
    """How a title says "along the axis": "over time", "across the chapters", or nothing."""
    if kind == "time":
        return "over time"
    if kind == "order":
        return f"across the {plural(noun or DEFAULT_NOUN).lower()}"
    return ""


def axis_of(corpus: Corpus | Sequence[object] | None, kind: str | None = None, noun: str = "") -> Axis:
    """The corpus's axis. *kind* ``None`` or ``"auto"`` picks one from what the documents carry.

    The automatic rule: time when at least two documents are dated, else
    order when at least two have an Order detail, else none. An explicit
    *kind* is kept even when it places nothing, so a figure can say why it
    has nothing to draw rather than silently switching axes.
    """
    docs = list(corpus.docs) if isinstance(corpus, Corpus) else list(corpus or [])
    dated = {str(doc.doc_id): doc.date for doc in docs if getattr(doc, "date", None) is not None}  # type: ignore[attr-defined]
    ordered: dict[str, float] = {}
    for doc in docs:
        value = document_order(getattr(doc, "details", None))
        if value is not None:
            ordered[str(doc.doc_id)] = value  # type: ignore[attr-defined]
    chosen = kind if kind in AXIS_KINDS else None
    if chosen is None:
        if len(dated) >= _PLACED_NEEDED:
            chosen = "time"
        elif len(ordered) >= _PLACED_NEEDED:
            chosen = "order"
        else:
            chosen = "none"
    if chosen == "time":
        return Axis(
            "time",
            "Year",
            {doc: _decimal_year(when) for doc, when in dated.items()},
            {doc: when.isoformat() for doc, when in dated.items()},
        )
    if chosen == "order":
        # A step nobody has named is a document: "across the documents",
        # "Documents 1-3", not "across the orders".
        label = noun.strip()
        if not label or label.casefold() == "order":
            label = DEFAULT_NOUN
        return Axis(
            "order",
            label,
            dict(ordered),
            {doc: f"{label} {_number(value)}" for doc, value in ordered.items()},
        )
    return Axis("none", "")
