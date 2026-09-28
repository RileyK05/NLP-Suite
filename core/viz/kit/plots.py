"""The drawing pieces: a stream, mixture bars, ranked bars, a time axis, events.

Each is one idea the reference figures repeat, generalised and given the kit's
rules (direct labels instead of legends, values where they are read, a
background group in grey, labels kept off the events). They draw onto an axis
the caller supplies and take plain data -- a DataFrame, a list of
``(x, label)`` -- so a script's own table and a finished run's table draw the
same way.

matplotlib is imported inside each function (R1: importing the library must
stay instant); only numpy and pandas load with the module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["bars", "dots", "events", "mixture_bars", "stream", "tiles", "time_axis", "word_grid"]

#: A segment narrower than this prints no inside value (the plan's panel B).
_VALUE_MIN_SHARE = 0.12
#: Event labels closer than this on the axis are nudged apart.
_EVENT_GAP = 4.5


def _percent(value: float, _: Any = None) -> str:
    return f"{value:.0%}"


def _style_axes(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def time_axis(
    ax: Any,
    documents: pd.DataFrame,
    *,
    x: str = "Year",
    band: str | None = None,
    label_bands: bool = True,
) -> None:
    """Year ticks, and optionally one detail (a speaker, a presidency) as bands.

    ``documents`` is a per-document table with an ``x`` column; ``band`` is a
    detail column whose contiguous runs are shaded so a reader can see which
    documents share a speaker or a term. Bands alternate a faint grey so
    adjacent runs separate without competing with the data.
    """
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
    if band is None or band not in documents.columns:
        _style_axes(ax)
        return
    ordered = documents[[x, band]].dropna().sort_values(x).reset_index(drop=True)
    if ordered.empty:
        _style_axes(ax)
        return
    runs: list[tuple[float, float, str]] = []
    start = float(ordered.loc[0, x])
    current = str(ordered.loc[0, band])
    for index in range(1, len(ordered)):
        value = str(ordered.loc[index, band])
        if value != current:
            runs.append((start, float(ordered.loc[index, x]), current))
            start = float(ordered.loc[index, x])
            current = value
    last = float(ordered.loc[len(ordered) - 1, x])
    runs.append((start, last, current))
    for index, (lo, hi, name) in enumerate(runs):
        shade = "#f0f2ee" if index % 2 == 0 else "#e4e8e2"
        ax.axvspan(lo - 0.5, hi - 0.5, color=shade, zorder=0, linewidth=0)
        if label_bands and hi - lo >= 3:
            ax.text(
                (lo + hi) / 2 - 0.5,
                0.985,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.8,
                color="#6b7268",
                zorder=1,
                clip_on=True,
            )
    _style_axes(ax)


def events(ax: Any, points: Sequence[tuple[float, str]], *, y: float = 1.012) -> None:
    """Vertical stoplines at dated events, with labels kept clear of each other.

    Turns a trend into a test the reader can check: did the line change at the
    event? Two events close together along the axis are nudged apart so their
    labels do not overprint.
    """
    placed: list[float] = []
    for position, label in sorted((float(pos), str(name)) for pos, name in points):
        ax.axvline(position, color="#3a3f38", linewidth=1.4, zorder=3)
        text_x = position
        for other in placed:
            if abs(text_x - other) < _EVENT_GAP:
                text_x = other + _EVENT_GAP if text_x >= other else other - _EVENT_GAP
        placed.append(text_x)
        ax.text(
            text_x,
            y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            zorder=4,
            clip_on=False,
        )


def _band_label_positions(
    years: np.ndarray, values: Sequence[np.ndarray], event_at: Sequence[float], minimum: float = 0.2
) -> list[tuple[float, float, int]]:
    """(x, y, band index) for each band's label, at its widest point.

    The reference figure's rule: label where the band is widest, nudge off the
    event lines, and skip a band too thin to hold text.
    """
    placed: list[tuple[float, float, int]] = []
    bottom = np.zeros(len(years))
    for index, value in enumerate(values):
        smooth = pd.Series(value).rolling(7, center=True, min_periods=1).mean().to_numpy()
        peak = int(np.argmax(smooth))
        x = float(np.clip(years[peak], years.min(), years.max()))
        for line in event_at:
            if abs(x - line) < _EVENT_GAP:
                x = line + _EVENT_GAP if x >= line else line - _EVENT_GAP
        near = np.flatnonzero(np.abs(years - x) <= 2)
        chosen = int(near[np.argmax(value[near])]) if len(near) else peak
        if float(value[chosen]) > minimum:
            placed.append((float(years[chosen]), float(bottom[chosen] + value[chosen] / 2), index))
        bottom += value
    return placed


def stream(
    ax: Any,
    shares: pd.DataFrame,
    *,
    labels: Mapping[Any, str] | None = None,
    colors: Mapping[Any, str] | None = None,
    background: Any | None = None,
    event_at: Sequence[float] = (),
    order: Sequence[Any] | None = None,
) -> dict[Any, str]:
    """Stacked shares over time with a direct label on each band.

    ``shares`` is indexed by the x position (a year) with one column per
    group; a group named by ``background`` is drawn grey and last, so events
    sit on the baseline. Labels are placed largest band first and a label that
    would overprint one already placed is dropped -- an overprinted label is
    unreadable, and the figure's caption can say how many were left out.
    Returns the colour actually used per group.
    """
    from core.viz.kit.palette import palette
    from core.viz.static.style import house_style

    keys = list(order) if order is not None else list(shares.columns)
    if background is not None and background in keys:
        keys = [key for key in keys if key != background] + [background]
    found = palette(shares.columns, background=background, colors=dict(colors or {}))
    years = shares.index.to_numpy(dtype=float)
    values = [shares[key].to_numpy(dtype=float) for key in keys]
    with house_style():
        ax.stackplot(years, values, colors=[found[key] for key in keys], edgecolor="white", linewidth=0.6)
        names = labels or {}
        for x, y, band in _ordered_labels(years, values, event_at):
            text = names.get(keys[band], str(keys[band]))
            if not text:
                continue
            color = "white" if found[keys[band]] != "#9aa39a" else "#3a3f38"
            ax.text(x, y, text, ha="center", va="center", fontsize=8.6, fontweight="bold", color=color, zorder=5)
    ax.set(ylim=(0, 1))
    ax.yaxis.set_major_formatter(_percent)
    _style_axes(ax)
    return found


def _ordered_labels(
    years: np.ndarray, values: Sequence[np.ndarray], event_at: Sequence[float]
) -> list[tuple[float, float, int]]:
    """Band labels, largest first, dropping any that would overprint another.

    The reference figure nudges a label off the event lines and drops a band
    too thin to label; with many bands two labels can also land on each other,
    so a label too close to one already placed is dropped rather than
    smudged.
    """
    candidates = _band_label_positions(years, values, event_at)
    # Largest band first, so the important topics keep their labels.
    candidates.sort(key=lambda placed: values[placed[2]][np.argmin(np.abs(years - placed[0]))], reverse=True)
    span_x = max(float(years.max() - years.min()), 1.0)
    placed: list[tuple[float, float, int]] = []
    for x, y, band in candidates:
        collides = any(abs(x - px) < span_x * 0.12 and abs(y - py) < 0.09 for px, py, _ in placed)
        if not collides:
            placed.append((x, y, band))
    return placed


def mixture_bars(
    ax: Any,
    shares: pd.DataFrame,
    *,
    row_labels: Sequence[str] | None = None,
    colors: Mapping[Any, str] | None = None,
    background: Any | None = None,
    order: Sequence[Any] | None = None,
    threshold: float = _VALUE_MIN_SHARE,
) -> dict[Any, str]:
    """One document per horizontal bar, split into its shares.

    The values are printed inside segments wide enough to hold them, so the
    reader does not measure a bar against an axis. Returns the colour per
    group.
    """
    from core.viz.kit.palette import palette
    from core.viz.static.style import house_style

    keys = list(order) if order is not None else list(shares.columns)
    if background is not None and background in keys:
        keys = [key for key in keys if key != background] + [background]
    found = palette(shares.columns, background=background, colors=dict(colors or {}))
    with house_style():
        for row in range(len(shares)):
            left = 0.0
            for key in keys:
                share = float(shares.iloc[row][key])
                ax.barh(row, share, left=left, color=found[key], height=0.66, edgecolor="white", linewidth=0.8)
                if share >= threshold:
                    ax.text(
                        left + share / 2,
                        row,
                        f"{share:.0%}",
                        ha="center",
                        va="center",
                        fontsize=8.4,
                        fontweight="bold",
                        color="white" if found[key] != "#9aa39a" else "#3a3f38",
                    )
                left += share
    names = list(row_labels) if row_labels is not None else [str(index) for index in shares.index]
    ax.set_yticks(range(len(names)), names, fontsize=9.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(_percent)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return found


def bars(
    ax: Any,
    values: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | Mapping[int, str] | str | None = None,
    fmt: str | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Horizontal bars with the value at the end and full-length labels.

    ``labels`` are the full row names (never cut); ``colors`` is a colour, one
    per row, or a mapping by row index.
    """
    from core.viz.static.style import MUTED, house_style
    from core.viz.static.text import format_value

    ys = np.arange(len(values))
    if isinstance(colors, Mapping):
        row_colors = [colors[index] for index in range(len(values))]
    elif isinstance(colors, str) or colors is None:
        row_colors = [colors or "#0072B2"] * len(values)
    else:
        row_colors = list(colors)
    with house_style():
        ax.barh(ys, list(values), color=row_colors, height=0.66)
    high = max((abs(float(v)) for v in values), default=1.0) or 1.0
    for y, value in zip(ys, values, strict=True):
        shown = fmt.format(value) if fmt else format_value(float(value))
        ax.text(float(value) + high * 0.03, y, shown, va="center", fontsize=8.3, color=MUTED)
    names = list(labels) if labels is not None else [str(index) for index in ys]
    ax.set_yticks(ys, names, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlim(*(xlim or (0.0, high * 1.35)))
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)


def dots(
    ax: Any,
    frame: pd.DataFrame,
    *,
    by: str,
    value: str,
    colors: Mapping[Any, str] | None = None,
    order: Sequence[Any] | None = None,
) -> None:
    """A strip of points per group, with a median line and a median label.

    "No averages without spread", drawn: the median sits on the strip that
    produced it, and its value is named so a thin group is not read as solid.
    """
    from core.viz.static.style import INK, MUTED, house_style
    from core.viz.static.text import format_value

    keys = list(order) if order is not None else sorted(frame[by].dropna().unique(), key=str)
    with house_style():
        for index, key in enumerate(keys):
            values = pd.to_numeric(frame.loc[frame[by] == key, value], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            jitter = (np.random.default_rng(index).random(len(values)) - 0.5) * 0.18
            color = (colors or {}).get(key, "#0072B2")
            ax.scatter(values, index + jitter, s=14, color=color, alpha=0.55, linewidths=0)
            median = float(np.median(values))
            ax.plot([median, median], [index - 0.32, index + 0.32], color=INK, linewidth=2.0, zorder=3)
            ax.text(median, index - 0.4, format_value(median), ha="center", va="bottom", fontsize=7.6, color=MUTED)
    ax.set_yticks(range(len(keys)), [str(key) for key in keys], fontsize=9.0)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)


def word_grid(
    ax: Any,
    words: Sequence[str],
    *,
    shade: Sequence[float] | None = None,
    columns: int = 4,
    fmt: str | None = None,
) -> None:
    """Ranked words in a grid, shaded by a measure (corpus frequency).

    The reference lambda figure's middle row: the words a topic learned, laid
    out so rank reads top-to-bottom and a light background carries a second
    measure. The shade stays a pale background, never a hue, so the words are
    the subject.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    from core.viz.static.style import INK, SEQUENTIAL, house_style
    from core.viz.static.text import format_value

    count = len(words)
    rows = max(1, (count + columns - 1) // columns)
    cmap = LinearSegmentedColormap.from_list("kit_word", list(SEQUENTIAL))
    fractions: list[float] | None = None
    if shade is not None and len(shade) == count:
        low, high = float(min(shade)), float(max(shade))
        span = (high - low) or 1.0
        fractions = [0.25 + 0.6 * (float(v) - low) / span for v in shade]
    with house_style():
        for index, word in enumerate(words):
            row, col = divmod(index, columns)
            face = cmap(fractions[index]) if fractions is not None else "#f3f5f1"
            ax.add_patch(
                Rectangle((col, rows - row - 0.9), 0.96, 0.9, facecolor=face, edgecolor="white", linewidth=1.0)
            )
            ax.text(col + 0.48, rows - row - 0.45, str(word), ha="center", va="center", fontsize=8.4, color=INK)
            if fractions is not None and shade is not None:
                label = fmt.format(shade[index]) if fmt else format_value(float(shade[index]))
                ax.text(col + 0.9, rows - row - 0.86, label, ha="right", va="bottom", fontsize=6.2, color="#6b7268")
    ax.set_xlim(0, columns)
    ax.set_ylim(0, rows)
    ax.axis("off")


def tiles(
    ax: Any,
    fits: Sequence[tuple[str, Sequence[int]]],
    *,
    references: Sequence[tuple[str, Sequence[int]]] = (),
    colors: Sequence[str] | None = None,
    stoplines: Sequence[tuple[float, str]] = (),
    tick_labels: Sequence[str] | None = None,
) -> None:
    """Rows of topic assignments, relabelled by first appearance, with stoplines.

    One row per fit, one tile per document (columns, in date order); topic
    numbers are arbitrary per fit, so each row is relabelled 0,1,2... in the
    order it first appears from the left. A fit that "saw" an event changes
    colour at the stopline, so the reader checks the claim by eye. Reference
    rows (a clean split, a word-count split) are framed.
    """
    from matplotlib.patches import Rectangle

    from core.viz.plotters import OKABE_ITO
    from core.viz.static.style import house_style

    rows: list[tuple[str, np.ndarray, bool]] = [(label, _by_first_appearance(row), False) for label, row in fits]
    rows += [(label, _by_first_appearance(row), True) for label, row in references]
    gap = 0.06
    y_positions: list[float] = []
    y = 0.0
    with house_style():
        for _label, assignment, is_reference in rows:
            row_colors = list(colors) if colors else list(OKABE_ITO)
            for col, topic in enumerate(assignment):
                face = row_colors[int(topic) % len(row_colors)]
                ax.add_patch(
                    Rectangle((col + gap, y + gap), 1 - 2 * gap, 1 - 2 * gap, facecolor=face, edgecolor="none")
                )
            if is_reference:
                ax.add_patch(Rectangle((0, y), len(assignment), 1, fill=False, edgecolor="#3d3c39", linewidth=1.1))
            y_positions.append(y + 0.5)
            y += 1
    width = len(rows[0][1]) if rows else 1
    for position, label in sorted((float(pos), str(name)) for pos, name in stoplines):
        ax.axvline(position, color="#3a3f38", linewidth=2.2, zorder=5)
        ax.text(position, -0.35, label, ha="center", va="bottom", fontsize=8.6, fontweight="bold", zorder=6)
    ax.set_xlim(0, width)
    ax.set_ylim(y, -1.4)
    ax.set_yticks(y_positions, [label for label, _, _ in rows], fontsize=8.2)
    if tick_labels is not None:
        ax.set_xticks([index + 0.5 for index in range(width)], list(tick_labels), fontsize=7.4, rotation=90)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def _by_first_appearance(topics: Sequence[int]) -> np.ndarray:
    order: dict[int, int] = {}
    out: list[int] = []
    for topic in topics:
        out.append(order.setdefault(int(topic), len(order)))
    return np.asarray(out, dtype=int)
