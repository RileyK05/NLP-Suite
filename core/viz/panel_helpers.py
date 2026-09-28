"""Shared rules every panel builder applies the same way.

Each rule here exists because two builders computing it separately would
disagree, and the reader would see two figures of one run that contradict
each other: a document labelled "1934 Roosevelt" in one and
"1934-01-03_franklin d roosevelt_sotu.txt" in the next, a decade that starts
in 1930 in one and 1934 in another, a network laid out differently every time
it is drawn. See ``docs/internal/FIGURE_RECIPES.md`` for the house rules these
implement.

Pure functions of their inputs: no I/O, no rendering imports, no randomness
(the network layout is seeded from its node order, never from a clock).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import math
import re

import numpy as np
import pandas as pd

from core.analysis.lda import STOPWORDS
from core.corpus_axis import along
from core.io.document_fields import DETAIL_PREFIX
from core.io.filename_fields import speaker_name
from core.result import Diagnostic

__all__ = [
    "FUNCTION_WORDS",
    "GROUPINGS",
    "POSITION",
    "POSITION_LABEL",
    "AxisInfo",
    "communities",
    "community_order",
    "dated",
    "decimal_year",
    "detail_columns",
    "document_labels",
    "force_layout",
    "group_of",
    "groupings",
    "is_function_word",
    "no_axis",
    "per_10k",
    "positioned",
    "rolling_median",
    "short_document_label",
    "speaker_of",
]

#: Words a term ranking can hide on request. The same list LDA removes, so a
#: word hidden here is a word the topic panels also never show.
FUNCTION_WORDS: frozenset[str] = STOPWORDS

#: The groupings every table offers, before its document details (:func:`groupings`).
#: ``speaker`` reads the ``Speaker`` detail when the table carries one and
#: falls back to the file-name rule for run tables made before details
#: existed.
GROUPINGS: tuple[str, ...] = ("none", "year", "decade", "speaker")

# date_speaker_kind.ext, with the date optional. The kind is the last
# underscore-separated part and is dropped from the speaker.
_NAMED = re.compile(r"^(?P<date>\d{4}(?:-\d{2}-\d{2})?)_(?P<speaker>.+?)(?:_(?P<kind>[^_]+))?$")


def is_function_word(word: str) -> bool:
    """True for a word every ranking of raw frequency is topped by ("of", "be")."""
    return str(word).strip().casefold() in FUNCTION_WORDS


def _stem(name: str) -> str:
    text = str(name).strip()
    return text.rsplit(".", 1)[0] if re.search(r"\.[A-Za-z0-9]{1,5}$", text) else text


def speaker_of(name: str) -> str:
    """The speaker parsed from a ``date_speaker_kind`` file name, title-cased.

    Empty when the name does not follow the pattern, never a guess: a
    grouping by a wrongly parsed "speaker" would sort documents into
    categories nobody named.
    """
    match = _NAMED.match(_stem(name))
    if not match or not match.group("kind"):
        return ""
    # One spelling rule for speakers, shared with the file-name detector that
    # gives documents a Speaker detail (core/io/filename_fields.py).
    return speaker_name(match.group("speaker"))


def short_document_label(name: str) -> str:
    """A document's name as an axis can show it: ``1934 Roosevelt``.

    Falls back to the file name without its extension. Use
    :func:`document_labels` for a whole axis, which keeps labels unique.
    """
    stem = _stem(name)
    match = _NAMED.match(stem)
    speaker = speaker_of(name)
    if match and speaker:
        return f"{match.group('date')[:4]} {speaker.split()[-1]}"
    return stem


def document_labels(names: Iterable[str]) -> dict[str, str]:
    """Short, *unique* labels for every document on one axis.

    Two speeches by one president in one year ("1946 Truman" twice) fall back
    to the full date, then to the file stem, so no two rows of a figure
    share a label.
    """
    ordered = list(dict.fromkeys(str(name) for name in names))
    labels = {name: short_document_label(name) for name in ordered}
    for fallback in (_dated_label, _stem):
        counts: dict[str, int] = {}
        for label in labels.values():
            counts[label] = counts.get(label, 0) + 1
        clashing = {label for label, count in counts.items() if count > 1}
        if not clashing:
            break
        for name in ordered:
            if labels[name] in clashing:
                labels[name] = fallback(name)
    return labels


def _dated_label(name: str) -> str:
    match = _NAMED.match(_stem(name))
    speaker = speaker_of(name)
    if match and speaker:
        return f"{match.group('date')} {speaker.split()[-1]}"
    return _stem(name)


def decimal_year(value: object) -> float | None:
    """A date as a position on a year axis: 1934-07-02 is about 1934.5.

    Accepts an ISO day, a :class:`datetime.date`, or a bare year. None for
    anything else, so an undated document is left off a time axis rather
    than placed at year zero.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, date):
        day = value
    else:
        text = str(value).strip()
        if re.fullmatch(r"\d{4}", text):
            return float(text)
        try:
            day = date.fromisoformat(text[:10])
        except ValueError:
            return None
    start = date(day.year, 1, 1)
    length = (date(day.year + 1, 1, 1) - start).days
    return day.year + (day - start).days / length


#: A run's per-document tables place each document on the corpus's axis with
#: these two columns (core/profiler/executor.py): a number for x, and what a
#: reader is shown ("1934-01-03", "Chapter 3").
DATE = "Date"
YEAR = "Year"
POSITION = "Position"
POSITION_LABEL = "Position label"
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class AxisInfo:
    """What a table's documents are placed along: ``time`` (x is a decimal year),
    ``order`` (x is a chapter or session number, called *noun*), or ``none``."""

    kind: str
    noun: str

    @property
    def along(self) -> str:
        """ "over time", "across the chapters", or "" (core/corpus_axis.py)."""
        return along(self.kind, self.noun)

    @property
    def x_label(self) -> str:
        return "Year" if self.kind == "time" else self.noun

    @property
    def x_axis(self) -> str:
        """For :class:`~core.viz.panelspec.PreparedPanel`: the kind, or "" when nothing places the documents."""
        return self.kind if self.kind in ("time", "order") else ""

    @property
    def x_noun(self) -> str:
        """For :class:`~core.viz.panelspec.PreparedPanel`: what one order step is called."""
        return self.noun if self.kind == "order" else ""


def positioned(frame: pd.DataFrame, date_column: str = "Date") -> tuple[pd.Series, AxisInfo]:
    """Each row's place on the corpus's axis, and what that axis is.

    Time reads the date exactly as figures always have (:func:`decimal_year`),
    so no dated figure changes. Order reads ``Position`` when the run lined
    the documents up by chapter or session; its noun comes from the label
    ("Chapter 3" -> "Chapter"). Neither: every place is missing and the kind
    is ``none``, for the builder to refuse with :func:`no_axis`.
    """
    labels = frame[POSITION_LABEL].dropna().astype(str) if POSITION_LABEL in frame.columns else pd.Series(dtype=str)
    ordered = (
        POSITION in frame.columns
        and frame[POSITION].notna().any()
        and not labels.empty
        and not labels.map(lambda text: bool(_ISO_DAY.match(text))).all()
    )
    if ordered:
        first = labels.iloc[0]
        noun = first.rsplit(" ", 1)[0] if " " in first else "Order"
        return pd.to_numeric(frame[POSITION], errors="coerce"), AxisInfo("order", noun)
    if date_column in frame.columns:
        years = frame[date_column].map(decimal_year)
        if years.notna().any():
            return years, AxisInfo("time", "Year")
    return pd.Series([None] * len(frame), index=frame.index, dtype=object), AxisInfo("none", "")


def no_axis(what: str) -> Diagnostic:
    """The refusal of a figure about change when nothing places the documents."""
    return Diagnostic.error(
        "PANEL_NO_AXIS",
        f"{what} needs the documents lined up: add a date or an order (chapter, session) to each document on the "
        "Corpus page, under Document details.",
    )


def detail_columns(frame: pd.DataFrame) -> list[str]:
    """The document details this table carries, named as a grouping names them ("Speaker", "Party").

    Details the executor added under :data:`~core.io.document_fields.DETAIL_PREFIX`
    (a name a tool column already had) keep their plain name here: the
    grouping is the detail, whichever column it landed in. The plain-named
    ones come from the frame's ``attrs["details"]``, written as the columns
    were added.
    """
    found: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name.startswith(DETAIL_PREFIX):
            found.append(name[len(DETAIL_PREFIX) :])
    for name in getattr(frame, "attrs", {}).get("details", ()) or ():
        if str(name) not in found and str(name) in {str(column) for column in frame.columns}:
            found.append(str(name))
    return sorted({name for name in found if name.casefold() != "speaker"}, key=str.casefold)


def groupings(frame: pd.DataFrame) -> tuple[str, ...]:
    """What this table can be grouped by: the axis's buckets, then its document details.

    On a table lined up by chapter, "year" and "decade" are answered with
    blocks of chapters by the figures that offer them, so there is no separate
    "period" choice. ``speaker`` covers the ``Speaker`` detail when there is
    one and the file-name rule when there is not (run tables from before
    details existed), so it is never listed beside a detail of that name.
    Everything else in the list names a detail column the table has.
    """
    found = ["none"]
    if YEAR in frame.columns or DATE in frame.columns or POSITION in frame.columns:
        found += ["year", "decade"]
    found.append("speaker")
    return tuple(dict.fromkeys(found + detail_columns(frame)))


def group_of(document: str, when: object, by: str, row: object | None = None) -> str:
    """Which group a document belongs to under :data:`GROUPINGS` -- or by its own detail.

    Decades start on the zero year (the 1930s are 1930 to 1939). An undated
    or unparsable document is named as such, so a reader can see how many
    fell outside the grouping. *by* may also name a document detail
    (``"Party"``, ``"Kind"``) or any column of *row*, which is how "group by
    Party" works without a second pass over the table.
    """
    if by not in GROUPINGS:
        value = _group_of_column(row, by)
        if value is None:
            raise ValueError(f"unknown grouping {by!r}; expected one of {GROUPINGS} or a detail column of the table")
        return value
    if by == "none":
        return "All documents"
    if by == "speaker":
        named = _group_of_column(row, "Speaker", empty="") or ""
        return named or speaker_of(document) or "(speaker not in file name)"
    year = decimal_year(when)
    if year is None:
        return "(undated)"
    if by == "year":
        return str(int(year))
    return f"{int(year) // 10 * 10}s"


def _group_of_column(row: object, by: str, empty: str | None = None) -> str | None:
    """A grouping read off one table row: its detail column, or None when the table has none.

    *empty* names a present-but-blank value (``"(no party)"``); ``""`` says
    the caller prefers another reading over a label, which is how an empty
    ``Speaker`` detail falls back to the file name.
    """
    if row is None:
        return None
    for column in (by, DETAIL_PREFIX + by):
        try:
            if column not in row.index:  # type: ignore[attr-defined]
                continue
            raw = row[column]  # type: ignore[index]
        except (AttributeError, KeyError, TypeError):
            return None
        text = "" if raw is None else str(raw)
        if text and text != "nan":
            return text
        return f"(no {by.lower()})" if empty is None else empty
    return None


def per_10k(count: float, tokens: float) -> float | None:
    """A count as a rate per 10,000 tokens; None when there are no tokens.

    A raw count pooled over documents of different lengths is mostly a
    measure of how long the documents were.
    """
    if not tokens or tokens <= 0 or math.isnan(tokens):
        return None
    return float(count) * 10_000.0 / float(tokens)


def rolling_median(xs: Sequence[float], ys: Sequence[float], window: int) -> list[tuple[float, float]]:
    """A rolling median over *window* consecutive points, one per point, in x order.

    Counted in points (documents), not in years, so a decade with two
    speeches is not smoothed as though it had twenty. The window is centred
    where it fits; within half a window of either end it is the first (or
    last) *window* points instead. So the line spans every document -- the
    earlier rule dropped the last four speeches of the corpus at a window of
    nine, which is where a reader looks first -- and every point is still a
    median of the same number of documents, as resistant to one outlier at
    the ends as in the middle. The price is a flat stretch at each end, which
    is honest: there is no more data there to bend it.

    Fewer points than the window gives an empty list; callers say why.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd number of points, got {window}")
    pairs = sorted(
        (float(x), float(y)) for x, y in zip(xs, ys, strict=True) if not (math.isnan(float(x)) or math.isnan(float(y)))
    )
    count = len(pairs)
    if count < window:
        return []
    half = window // 2
    out: list[tuple[float, float]] = []
    for index in range(count):
        start = min(max(0, index - half), count - window)
        values = [y for _, y in pairs[start : start + window]]
        out.append((pairs[index][0], float(np.median(values))))
    return out


def force_layout(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
    *,
    iterations: int = 300,
) -> dict[str, tuple[float, float]]:
    """Node positions in [0, 1] x [0, 1] from a Fruchterman-Reingold layout.

    Deterministic: nodes start evenly on a circle in the order given, so the
    same graph always draws the same way, in the app and in the export.
    Heavier edges pull harder. An isolated node is pushed to the edge of the
    picture rather than dropped.
    """
    count = len(nodes)
    if count == 0:
        return {}
    if count == 1:
        return {nodes[0]: (0.5, 0.5)}
    index = {name: position for position, name in enumerate(nodes)}
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])
    weights = np.zeros((count, count))
    for source, target, weight in edges:
        if source in index and target in index and source != target:
            a, b = index[source], index[target]
            weights[a, b] = weights[b, a] = max(weights[a, b], float(weight))
    if weights.max() > 0:
        weights = weights / weights.max()
    k = math.sqrt(4.0 / count)
    temperature = 0.2
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.maximum(np.linalg.norm(delta, axis=-1), 1e-6)
        force = (k * k / distance**2 - weights * distance / k)[:, :, None] * delta
        displacement = force.sum(axis=1)
        length = np.maximum(np.linalg.norm(displacement, axis=-1), 1e-9)
        pos += displacement / length[:, None] * np.minimum(length, temperature)[:, None]
        temperature *= 0.985
    low, high = pos.min(axis=0), pos.max(axis=0)
    span = np.where(high - low > 1e-9, high - low, 1.0)
    unit = (pos - low) / span
    return {name: (round(float(unit[i, 0]), 6), round(float(unit[i, 1]), 6)) for name, i in index.items()}


def dated(frame: pd.DataFrame, date_column: str = "Date", year_column: str = "Year") -> pd.Series:
    """Each row's position on a year axis, from ``Date`` or else ``Year``."""
    if date_column in frame.columns:
        return frame[date_column].map(decimal_year)
    if year_column in frame.columns:
        return frame[year_column].map(decimal_year)
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


#: At most this many communities get their own colour; the rest share one.
MAX_COMMUNITIES = 7
#: In parentheses: a remainder group, which every renderer draws grey.
OTHER_COMMUNITY = "(smaller clusters)"


def communities(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
    mass: Mapping[str, float],
) -> dict[str, str]:
    """Each node's community, named by its two heaviest words ("war · peace").

    Weighted label propagation, deterministic: nodes visit in order of mass
    (heaviest first, then name), each taking the label its neighbours carry
    the most edge weight for, keeping its own on a tie. It finds groups of
    words more linked to each other than to the rest -- the clusters a reader
    of a word network looks for, coloured instead of left to the eye.

    The largest :data:`MAX_COMMUNITIES` are named; smaller ones, and single
    words, share :data:`OTHER_COMMUNITY`.
    """
    order = sorted(nodes, key=lambda node: (-float(mass.get(node, 0.0)), node))
    neighbours: dict[str, dict[str, float]] = {node: {} for node in nodes}
    for source, target, weight in edges:
        if source in neighbours and target in neighbours and source != target:
            neighbours[source][target] = neighbours[source].get(target, 0.0) + float(weight)
            neighbours[target][source] = neighbours[target].get(source, 0.0) + float(weight)
    label = {node: node for node in nodes}
    for _ in range(50):
        changed = False
        for node in order:
            if not neighbours[node]:
                continue
            pull: dict[str, float] = {}
            for other, weight in neighbours[node].items():
                pull[label[other]] = pull.get(label[other], 0.0) + weight
            best = max(pull.values())
            if pull.get(label[node], -1.0) >= best:
                continue
            winner = min((candidate for candidate, value in pull.items() if value == best), key=order.index)
            label[node] = winner
            changed = True
        if not changed:
            break
    members: dict[str, list[str]] = {}
    for node in order:
        members.setdefault(label[node], []).append(node)
    ranked = sorted(
        (group for group in members.values() if len(group) > 1),
        key=lambda group: (-sum(float(mass.get(node, 0.0)) for node in group), group[0]),
    )
    names: dict[str, str] = {}
    for group in ranked[:MAX_COMMUNITIES]:
        title = " · ".join(group[:2])
        for node in group:
            names[node] = title
    for node in nodes:
        names.setdefault(node, OTHER_COMMUNITY)
    return names


def community_order(names: Mapping[str, str], mass: Mapping[str, float]) -> tuple[str, ...]:
    """Community names, heaviest first, :data:`OTHER_COMMUNITY` last."""
    totals: dict[str, float] = {}
    for node, name in names.items():
        totals[name] = totals.get(name, 0.0) + float(mass.get(node, 0.0))
    ordered = sorted((name for name in totals if name != OTHER_COMMUNITY), key=lambda name: (-totals[name], name))
    return (*ordered, *((OTHER_COMMUNITY,) if OTHER_COMMUNITY in totals else ()))
