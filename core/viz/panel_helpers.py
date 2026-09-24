"""Shared rules every panel builder applies the same way.

Each rule here exists because two builders computing it separately would
disagree, and the reader would see two figures of one run that contradict
each other: a document labelled "1934 Roosevelt" in one and
"1934-01-03_franklin d roosevelt_sotu.txt" in the next, a decade that starts
in 1930 in one and 1934 in another, a network laid out differently every time
it is drawn. See ``docs/FIGURE_RECIPES.md`` for the house rules these
implement.

Pure functions of their inputs: no I/O, no rendering imports, no randomness
(the network layout is seeded from its node order, never from a clock).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
import math
import re

import numpy as np
import pandas as pd

from core.analysis.lda import STOPWORDS

__all__ = [
    "FUNCTION_WORDS",
    "GROUPINGS",
    "communities",
    "community_order",
    "dated",
    "decimal_year",
    "document_labels",
    "force_layout",
    "group_of",
    "is_function_word",
    "per_10k",
    "rolling_median",
    "short_document_label",
    "speaker_of",
]

#: Words a term ranking can hide on request. The same list LDA removes, so a
#: word hidden here is a word the topic panels also never show.
FUNCTION_WORDS: frozenset[str] = STOPWORDS

#: How a per-document view may group its documents. ``speaker`` is read from
#: the file name (``1934-01-03_franklin d roosevelt_sotu.txt``) and is empty
#: for names that do not follow that pattern.
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
    return " ".join(part.capitalize() for part in match.group("speaker").split())


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


def group_of(document: str, when: object, by: str) -> str:
    """Which group a document belongs to under one of :data:`GROUPINGS`.

    Decades start on the zero year (the 1930s are 1930 to 1939). An undated
    or unparsable document is named as such, so a reader can see how many
    fell outside the grouping.
    """
    if by not in GROUPINGS:
        raise ValueError(f"unknown grouping {by!r}; expected one of {GROUPINGS}")
    if by == "none":
        return "All documents"
    if by == "speaker":
        return speaker_of(document) or "(speaker not in file name)"
    year = decimal_year(when)
    if year is None:
        return "(undated)"
    if by == "year":
        return str(int(year))
    return f"{int(year) // 10 * 10}s"


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
