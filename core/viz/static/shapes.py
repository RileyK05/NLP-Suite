"""One drawing function per panel shape, for publication figures.

Each takes a :class:`PreparedPanel` -- the same object the app draws -- and
returns a matplotlib figure plus the notes its caption owes the reader
("rows reordered by clustering", "3 labels omitted"). What a figure *claims*
was decided by the builder; these functions decide how much more a static
figure can *show* than the interactive one:

* trends get the spread around them (the interquartile band over the same
  windows as the rolling median), a distribution strip of every document at
  the side, decade shading, and their extremes named;
* groups get violins under the box and points, with n and the median printed;
* a document-by-document matrix is reordered by clustering, with its tree and
  a decade bar, so blocks of similar speeches show as blocks;
* labels are placed against measured text, never over each other, and never
  cut -- long ones wrap, and the figure grows to hold them.

No pyplot: every figure is a bare :class:`~matplotlib.figure.Figure` on an
Agg canvas, so rendering is safe in the desktop server's worker threads and
leaves no global state behind.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

from core.viz.panelspec import PanelMark, PreparedPanel
from core.viz.static.labels import LabelRequest, freeze_layout, note_dropped, place_labels
from core.viz.static.stats import decade_of, rolling_band
from core.viz.static.style import (
    AXIS,
    DIVERGING,
    INK,
    MUTED,
    PALE,
    SEQUENTIAL,
    UNGROUPED,
    color_of,
    group_colors,
)
from core.viz.static.text import AxisFormatter, format_value, is_year_axis, tick_labels, wrap

__all__ = ["DRAWERS", "Drawn"]

#: A drawn figure and the sentences its caption must add.
Drawn = tuple[Figure, list[str]]

_WIDTH_IN = 10.0
_LABEL_WRAP = 42


# ------------------------------------------------------------------ helpers --


def _figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _height_for(prepared: PreparedPanel, default: float = 5.0) -> float:
    """The builder's height in inches (100 px each), at least ``default``."""
    return max(default, prepared.height / 100.0)


def _grouped(prepared: PreparedPanel) -> tuple[dict[str, list[PanelMark]], list[str]]:
    """Marks by group in declared order, then any undeclared (sorted)."""
    buckets: dict[str, list[PanelMark]] = {}
    for mark in prepared.marks:
        buckets.setdefault(mark.group or UNGROUPED, []).append(mark)
    declared = list(prepared.groups) or [UNGROUPED]
    ordered = [group for group in declared if group in buckets]
    ordered += sorted(group for group in buckets if group not in ordered)
    return buckets, ordered


def _formatters(ax: Axes, xs: Sequence[float] = (), ys: Sequence[float] = ()) -> None:
    ax.xaxis.set_major_formatter(AxisFormatter(plain=is_year_axis(xs)))
    ax.yaxis.set_major_formatter(AxisFormatter(plain=is_year_axis(ys)))


def _pad(low: float, high: float, fraction: float = 0.06) -> tuple[float, float]:
    span = high - low
    pad = span * fraction if span > 0 else (abs(high) * 0.1 or 1.0)
    return low - pad, high + pad


#: A non-negative series starts its axis at zero only when its lowest value
#: is within this share of its highest: MTLD between 55 and 120 drawn from
#: zero spent half the plot on empty space, a rate from 3 to 40 does not.
_ZERO_NEAR = 0.35


def _limits(values: Sequence[float], *, floor_zero: bool = False, top: float = 0.08) -> tuple[float, float]:
    low, high = min(values), max(values)
    if floor_zero and low >= 0 and low <= high * _ZERO_NEAR:
        span = high - low if high > low else (abs(high) or 1.0)
        return 0.0, high + span * top
    return _pad(low, high, top)


def _segments(marks: list[PanelMark], gap: float) -> list[list[PanelMark]]:
    """Marks in x order, split where consecutive x differ by more than ``gap``."""
    ordered = sorted(marks, key=lambda mark: mark.x)
    out: list[list[PanelMark]] = []
    for mark in ordered:
        if not out or mark.x - out[-1][-1].x > gap:
            out.append([])
        out[-1].append(mark)
    return out


def _decade_shading(ax: Axes, xs: Sequence[float]) -> None:
    """Every other decade on a pale ground, so a year is placed at a glance."""
    if not xs or not is_year_axis(xs):
        return
    start = int(math.floor(min(xs) / 10.0) * 10)
    for decade in range(start, int(max(xs)) + 1, 10):
        if (decade // 10) % 2 == 0:
            ax.axvspan(decade, decade + 10, color=PALE, zorder=0, linewidth=0)


def _legend(fig: Figure, ax: Axes, handles: list[Any], labels: list[str]) -> None:
    if len(labels) > 1:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)


def _reference_lines(ax: Axes, prepared: PreparedPanel) -> list[Any]:
    """Reference lines with their short label on the plot; returns the text artists.

    The reason for a line (its ``note``) goes in the caption: printed beside
    the line it ran four lines deep over the data.
    """
    texts = []
    for annotation in prepared.annotations:
        if annotation.kind == "note":
            continue
        label = annotation.label
        if annotation.kind == "vline":
            ax.axvline(annotation.value, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=2)
            texts.append(
                ax.text(
                    annotation.value,
                    1.0,
                    " " + wrap(label, 40),
                    transform=blended_transform_factory(ax.transData, ax.transAxes),
                    ha="left",
                    va="top",
                    fontsize=7,
                    color=MUTED,
                )
            )
        else:
            ax.axhline(annotation.value, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=2)
            texts.append(
                ax.text(
                    1.0,
                    annotation.value,
                    wrap(label, 40) + " ",
                    transform=blended_transform_factory(ax.transAxes, ax.transData),
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color=MUTED,
                )
            )
    return texts


def _radius_fn(marks: Sequence[PanelMark], *, small: float = 3.0, big: float = 10.0) -> Callable[[PanelMark], float]:
    """Marker radius in points, area proportional to ``size`` (the app's rule)."""
    largest = max((mark.size or 0.0) for mark in marks) if marks else 0.0

    def radius(mark: PanelMark) -> float:
        if mark.size is None or largest <= 0:
            return small + 0.8
        return max(small * 0.7, math.sqrt(mark.size / largest) * big)

    return radius


def log_tick_values(low: float, high: float) -> list[float]:
    """Round quantities (1, 2, 5 x 10^k) whose log10 lies in [low, high];
    only the powers of ten when the range spans more than two decades.
    The app's ``panelTicks.logTicks`` is the same rule."""
    values = []
    for power in range(math.floor(low) - 1, math.ceil(high) + 1):
        steps = (1,) if high - low > 2 else (1, 2, 5)
        for step in steps:
            value = step * 10.0**power
            if value > 0 and low <= math.log10(value) <= high:
                values.append(value)
    return values


def _log_ticks(ax: Axes, low: float, high: float) -> None:
    """Ticks at round quantities on an axis that holds their log10."""
    from matplotlib.ticker import FixedLocator

    values = log_tick_values(low, high)
    if len(values) < 2:
        return
    ax.xaxis.set_major_locator(FixedLocator([math.log10(v) for v in values]))
    labels = tick_labels(values, plain=False)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: labels[pos] if pos is not None and pos < len(labels) else format_value(10**v))
    )


def _colormap(prepared: PreparedPanel) -> LinearSegmentedColormap:
    return _colormap_named(prepared.color_scale or "sequential")


def _colormap_named(scale: str) -> LinearSegmentedColormap:
    """The app's heatmap stops as a matplotlib colour map."""
    stops = DIVERGING if scale == "diverging" else SEQUENTIAL
    return LinearSegmentedColormap.from_list(scale, list(stops))


# ------------------------------------------------------------------ scatter --


def draw_scatter(prepared: PreparedPanel) -> Drawn:
    fig = _figure(_WIDTH_IN, _height_for(prepared, 5.6))
    ax = fig.add_subplot()
    buckets, ordered = _grouped(prepared)
    colors = group_colors(prepared.groups)
    dense = len(prepared.marks) > 300
    # Fewer, bigger points when there are few; many points share the area.
    big = 10.0 if not dense else max(4.0, 10.0 * math.sqrt(300 / len(prepared.marks)))
    radius = _radius_fn(prepared.marks, small=2.2 if dense else 3.0, big=big)
    xs = [mark.x for mark in prepared.marks]
    ys = [mark.y for mark in prepared.marks]
    notes: list[str] = []
    if dense:
        import seaborn as sns

        # Where the points are thick, a density ground says so; single points
        # would only pile up into one blot.
        sns.kdeplot(x=xs, y=ys, ax=ax, fill=True, levels=8, thresh=0.03, cmap="Greys", alpha=0.35, warn_singular=False)
        notes.append("Grey shading is the density of all points (a kernel density estimate).")
    handles, names = [], []
    for group in ordered:
        marks = buckets[group]
        color = color_of(colors, group)
        artist = ax.scatter(
            [m.x for m in marks],
            [m.y for m in marks],
            s=[(2 * radius(m)) ** 2 for m in marks],
            color=color,
            alpha=0.45 if dense else 0.78,
            linewidths=0.5,
            edgecolors="white",
            zorder=3,
        )
        handles.append(artist)
        names.append(group)
    reference = _reference_lines(ax, prepared)
    notes.extend(f"Dashed line, {a.label}: {a.note}" for a in prepared.annotations if a.kind != "note" and a.note)
    extra_x = [a.value for a in prepared.annotations if a.kind == "vline"]
    extra_y = [a.value for a in prepared.annotations if a.kind == "hline"]
    ax.set_xlim(*_pad(min(xs + extra_x), max(xs + extra_x), 0.06))
    ax.set_ylim(*_pad(min(ys + extra_y), max(ys + extra_y), 0.08))
    _formatters(ax, xs, ys)
    if prepared.x_log10:
        _log_ticks(ax, *ax.get_xlim())
    ax.set_xlabel(prepared.x_label)
    ax.set_ylabel(prepared.y_label)
    if len(names) > 1:
        _legend(fig, ax, [Line2D([], [], marker="o", linestyle="", color=color_of(colors, g)) for g in names], names)
    renderer = freeze_layout(fig)
    labelled = sorted((m for m in prepared.marks if m.labelled), key=lambda m: -(m.size or 0.0))
    obstacles = [text.get_window_extent(renderer) for text in reference]
    dropped, _ = place_labels(
        ax,
        [LabelRequest(m.x, m.y, m.label, radius(m)) for m in labelled],
        renderer,
        obstacles=obstacles,
    )
    note_dropped(ax, dropped)
    return fig, notes


# -------------------------------------------------------------- line series --


def _summary_window(marks: Sequence[PanelMark]) -> int:
    counts = [mark.evidence.count for mark in marks if mark.evidence.count > 1]
    return int(counts[0]) if counts else 0


#: Independent annual series past this many are drawn one panel each.
_OVERLAID_SERIES = 2
#: Rolling median over this many years in a per-series panel.
_SERIES_WINDOW = 5


def _series_panels(prepared: PreparedPanel, buckets: dict[str, list[PanelMark]], ordered: list[str]) -> Drawn:
    """One panel per series on a shared scale: points, and a rolling median.

    Four phrases' per-year rates overlaid drew four jagged lines through one
    another; apart, on one y scale, each shape can be read and compared.
    """
    from core.viz.panel_helpers import rolling_median

    colors = group_colors(prepared.groups)
    columns = 2 if len(ordered) <= 4 else 3
    rows = math.ceil(len(ordered) / columns)
    fig = _figure(_WIDTH_IN, max(3.2, 2.4 * rows + 0.6))
    axes = fig.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    xs = [m.x for m in prepared.marks]
    ys = [m.y for m in prepared.marks]
    for index, group in enumerate(ordered):
        ax = axes[index // columns][index % columns]
        marks = sorted(buckets[group], key=lambda m: m.x)
        color = color_of(colors, group)
        _decade_shading(ax, xs)
        ax.vlines([m.x for m in marks], 0, [m.y for m in marks], color=color, linewidth=0.8, alpha=0.45)
        ax.scatter([m.x for m in marks], [m.y for m in marks], s=9, color=color, zorder=3, linewidths=0)
        if len(marks) >= _SERIES_WINDOW:
            # The smoothed line crosses a year or two without documents and
            # breaks only across a real hole; the stems still show each year.
            for segment in _segments(marks, max(prepared.line_gap, 3.0)):
                if len(segment) < _SERIES_WINDOW:
                    continue
                smooth = rolling_median([m.x for m in segment], [m.y for m in segment], _SERIES_WINDOW)
                ax.plot([x for x, _ in smooth], [y for _, y in smooth], color=INK, linewidth=1.6, zorder=4)
        ax.set_title(group, loc="left", fontsize=9, fontweight="bold", color=color)
        ax.set_xlim(*_pad(min(xs), max(xs), 0.02))
        ax.set_ylim(*_limits(ys, floor_zero=True))
        _formatters(ax, xs, ys)
        ax.grid(axis="x", visible=False)
    for index in range(len(ordered), rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    fig.supxlabel(prepared.x_label, fontsize=9, color=MUTED)
    fig.supylabel(prepared.y_label, fontsize=9, color=MUTED)
    freeze_layout(fig)
    return fig, [
        f"One panel per series on a shared scale: stems are single years, the dark line a {_SERIES_WINDOW}-year "
        "rolling median of the years present."
    ]


def draw_line_series(prepared: PreparedPanel) -> Drawn:
    buckets, ordered = _grouped(prepared)
    if not prepared.points_only and len(ordered) > _OVERLAID_SERIES:
        return _series_panels(prepared, buckets, ordered)
    colors = group_colors(prepared.groups)
    points_only = set(prepared.points_only)
    documents = [group for group in ordered if group in points_only]
    summaries = [group for group in ordered if group not in points_only]
    notes: list[str] = []
    fig = _figure(_WIDTH_IN, _height_for(prepared, 5.2))
    if documents:
        grid = fig.add_gridspec(1, 2, width_ratios=(12, 1.4), wspace=0.02)
        ax = fig.add_subplot(grid[0, 0])
        side = fig.add_subplot(grid[0, 1], sharey=ax)
    else:
        ax = fig.add_subplot()
        side = None
    xs = [mark.x for mark in prepared.marks]
    ys = [mark.y for mark in prepared.marks]
    _decade_shading(ax, xs)
    handles: list[Any] = []
    names: list[str] = []
    point_marks = [mark for group in documents for mark in buckets[group]]
    for group in documents:
        marks = buckets[group]
        color = color_of(colors, group)
        handles.append(
            ax.scatter([m.x for m in marks], [m.y for m in marks], s=15, color=color, alpha=0.6, linewidths=0, zorder=3)
        )
        names.append(group)
    for group in summaries:
        marks = buckets[group]
        color = color_of(colors, group) if not documents else INK
        window = _summary_window(marks) if documents else 0
        if documents and window:
            band = rolling_band([m.x for m in point_marks], [m.y for m in point_marks], window)
            if band:
                bx, lo, hi = zip(*band, strict=True)
                handles.append(
                    ax.fill_between(bx, lo, hi, color=color_of(colors, documents[0]), alpha=0.16, linewidth=0, zorder=2)
                )
                names.append(f"Middle half of the same {window} documents (IQR)")
                notes.append(
                    f"The shaded band is the interquartile range of the same {window}-document windows as the "
                    "rolling median: half of the documents in each window fall inside it."
                )
        zero_share = sum(1 for m in marks if m.y == 0) / len(marks)
        segments = _segments(marks, prepared.line_gap)
        if not documents and zero_share >= 0.5 and len(marks) > 10:
            # Mostly-zero years: a line through them reads as a seismograph.
            # Stems from the baseline show where the phrase occurs, and how
            # much; zero years stay on the axis as small ticks.
            nonzero = [m for m in marks if m.y != 0]
            ax.vlines([m.x for m in nonzero], 0, [m.y for m in nonzero], color=color, linewidth=1.4, zorder=3)
            handles.append(
                ax.scatter([m.x for m in nonzero], [m.y for m in nonzero], s=16, color=color, zorder=4, linewidths=0)
            )
            ax.scatter(
                [m.x for m in marks if m.y == 0],
                [0.0] * sum(1 for m in marks if m.y == 0),
                s=10,
                marker="|",
                color=color,
                alpha=0.5,
                zorder=3,
            )
            notes.append(
                f"{group}: {zero_share:.0%} of years have none; stems mark the years that do "
                "(ticks on the axis are years with zero)."
            )
        else:
            for segment in segments:
                (line,) = ax.plot(
                    [m.x for m in segment],
                    [m.y for m in segment],
                    color=color,
                    linewidth=2.3 if documents else 1.7,
                    marker="" if documents else "o",
                    markersize=0 if documents else (2.5 if len(marks) > 40 else 3.6),
                    zorder=4,
                    solid_joinstyle="round",
                )
            handles.append(line)
        names.append(group)
    ax.set_ylim(*_limits(ys, floor_zero=True))
    ax.set_xlim(*_pad(min(xs), max(xs), 0.015))
    _formatters(ax, xs, ys)
    ax.set_xlabel(prepared.x_label)
    ax.set_ylabel(prepared.y_label)
    ax.grid(axis="x", visible=False)
    if side is not None and point_marks:
        import seaborn as sns

        values = [m.y for m in point_marks]
        if len(set(values)) > 2:
            sns.kdeplot(y=values, ax=side, fill=True, color=color_of(colors, documents[0]), alpha=0.35, linewidth=0.8)
        side.scatter([0.0] * len(values), values, marker="_", s=30, color=MUTED, alpha=0.5, linewidths=0.8)
        side.set_xlabel("all docs", fontsize=7)
        side.set_xticks([])
        side.tick_params(axis="y", labelleft=False, length=0)
        side.grid(False)
        for spine in ("left", "bottom"):
            side.spines[spine].set_visible(False)
    if len(names) > 1:
        # Above the plot, not on it: inside, it sat on the highest points.
        ax.legend(
            handles,
            names,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            borderaxespad=0.0,
            fontsize=7.5,
            ncol=min(4, len(names)),
        )
    renderer = freeze_layout(fig)
    if point_marks:
        # The two highest and two lowest documents, named: the reader's first
        # question of a scatter of speeches is "which ones are those?".
        ranked = sorted(point_marks, key=lambda m: m.y)
        extremes = [*ranked[-2:][::-1], *ranked[:2]]
        legend = ax.get_legend()
        obstacles = [legend.get_window_extent(renderer)] if legend else []
        dropped, _ = place_labels(
            ax, [LabelRequest(m.x, m.y, m.label, 3.0) for m in extremes], renderer, fontsize=7, obstacles=obstacles
        )
        del dropped  # the extremes are a courtesy; a missing one is not a finding
    return fig, notes


# -------------------------------------------------------------- ranked bars --


def draw_ranked_bars(prepared: PreparedPanel) -> Drawn:
    buckets, ordered = _grouped(prepared)
    colors = group_colors(prepared.groups)
    rank_of: dict[str, float] = {}
    for mark in prepared.marks:
        rank_of.setdefault(mark.label, mark.y)
    rows = sorted(rank_of, key=lambda label: (rank_of[label], label))
    row_index = {label: index for index, label in enumerate(rows)}
    height = max(3.0, 1.2 + 0.3 * len(rows))
    fig = _figure(_WIDTH_IN, height)
    ax = fig.add_subplot()
    # Side by side only where one row carries several series (a topic's
    # share against the corpus); a row with one bar gets the full height.
    per_label: dict[str, int] = {}
    for mark in prepared.marks:
        per_label[mark.label] = per_label.get(mark.label, 0) + 1
    dodge = max(per_label.values()) > 1
    per_row = max(1, len(ordered)) if dodge else 1
    bar = 0.74 / per_row
    values = [mark.x for mark in prepared.marks]
    handles = []
    for offset, group in enumerate(ordered):
        marks = buckets[group]
        slot = offset if dodge else 0
        positions = [row_index[m.label] - 0.37 + bar * (slot + 0.5) for m in marks]
        handles.append(
            ax.barh(positions, [m.x for m in marks], height=bar * 0.92, color=color_of(colors, group), alpha=0.92)
        )
        for position, mark in zip(positions, marks, strict=True):
            ax.annotate(
                format_value(mark.x),
                (mark.x, position),
                xytext=(3 if mark.x >= 0 else -3, 0),
                textcoords="offset points",
                ha="left" if mark.x >= 0 else "right",
                va="center",
                fontsize=6.8,
                color=MUTED,
            )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([wrap(label, _LABEL_WRAP) for label in rows], fontsize=8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    low, high = min(0.0, *values), max(0.0, *values)
    span = high - low or 1.0
    ax.set_xlim(low - (span * 0.12 if low < 0 else 0), high + (span * 0.12 if high > 0 else 0))
    if low < 0:
        ax.axvline(0, color=AXIS, linewidth=0.9)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(AxisFormatter())
    ax.set_xlabel(prepared.x_label)
    if prepared.y_label:
        ax.set_ylabel(prepared.y_label)
    if len(ordered) > 1 and ordered != [UNGROUPED]:
        _legend(fig, ax, list(handles), ordered)
    freeze_layout(fig)
    return fig, []


# ------------------------------------------------------------- distribution --


def draw_distribution(prepared: PreparedPanel) -> Drawn:
    import seaborn as sns

    categories = list(prepared.y_categories)
    rows = [
        {"value": mark.x, "row": categories[int(mark.y)], "group": mark.group or UNGROUPED}
        for mark in prepared.marks
        if 0 <= int(mark.y) < len(categories)
    ]
    frame = pd.DataFrame(rows)
    height = max(3.2, 1.3 + 0.34 * len(categories))
    fig = _figure(_WIDTH_IN, height)
    ax = fig.add_subplot()
    counts = frame.groupby("row")["value"].size()
    wide = [row for row in categories if counts.get(row, 0) >= 3]
    if wide:
        # Violins only where there are enough points to have a shape.
        sns.violinplot(
            data=frame[frame["row"].isin(wide)],
            x="value",
            y="row",
            order=categories,
            orient="h",
            inner=None,
            cut=0,
            density_norm="width",
            color="#dfe6dc",
            linewidth=0,
            ax=ax,
        )
    sns.boxplot(
        data=frame,
        x="value",
        y="row",
        order=categories,
        orient="h",
        width=0.24,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": MUTED, "linewidth": 0.8},
        whiskerprops={"color": MUTED, "linewidth": 0.8},
        capprops={"color": MUTED, "linewidth": 0.8},
        medianprops={"color": INK, "linewidth": 1.8},
        ax=ax,
    )
    declared = [group for group in prepared.groups if group in set(frame["group"])]
    palette = group_colors(prepared.groups)
    sns.stripplot(
        data=frame,
        x="value",
        y="row",
        order=categories,
        orient="h",
        hue="group" if declared else None,
        hue_order=declared or None,
        palette={g: palette[g] for g in declared} if declared else None,
        color=None if declared else color_of(palette, UNGROUPED),
        jitter=0.2,
        size=3.3,
        alpha=0.75,
        linewidth=0,
        legend=False,
        ax=ax,
        zorder=4,
    )
    summary = frame.groupby("row")["value"].agg(["size", "median"])
    right = blended_transform_factory(ax.transAxes, ax.transData)
    for index, row in enumerate(categories):
        if row in summary.index:
            n, median = summary.loc[row, "size"], summary.loc[row, "median"]
            ax.text(
                1.01,
                index,
                f"n={int(n)} · median {format_value(float(median))}",
                transform=right,
                va="center",
                ha="left",
                fontsize=6.8,
                color=MUTED,
            )
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([wrap(label, _LABEL_WRAP) for label in categories])
    ax.set_ylabel("")
    ax.set_xlabel(prepared.x_label)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(AxisFormatter())
    freeze_layout(fig)
    return fig, [
        "Each violin is the shape of its group's values (drawn where a group has three or more); "
        "the box spans the middle half, the bar is the median, and every document is a point."
    ]


# ------------------------------------------------------------------ heatmap --


def _matrix(prepared: PreparedPanel) -> np.ndarray:
    matrix = np.full((len(prepared.y_categories), len(prepared.x_categories)), np.nan)
    for mark in prepared.marks:
        matrix[int(mark.y), int(mark.x)] = mark.value if mark.value is not None else np.nan
    return matrix


def _square_documents(prepared: PreparedPanel) -> bool:
    return prepared.x_categories == prepared.y_categories and len(prepared.x_categories) >= 4


def draw_heatmap(prepared: PreparedPanel) -> Drawn:
    if _square_documents(prepared):
        return _clustered(prepared)
    import seaborn as sns

    matrix = _matrix(prepared)
    rows, columns = matrix.shape
    width = min(18.0, max(6.5, 3.0 + columns * 0.3))
    height = min(26.0, max(3.6, 1.9 + rows * 0.27))
    fig = _figure(width, height)
    ax = fig.add_subplot()
    finite = matrix[np.isfinite(matrix)]
    diverging = prepared.color_scale == "diverging"
    bound = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    annotate = rows * columns <= 400 and max(rows, columns) <= 20
    step = float(np.nanmax(finite) - np.nanmin(finite)) / 10 if finite.size else 1.0
    digits = 0 if step >= 5 else 1 if step >= 0.5 else 2
    # One label per column while they fit side by side on the slant; past
    # that, every k-th, as the app thins them (85 year columns overprinted).
    column_room = (width - 2.5) * 72 / max(1, columns)
    every_column = max(1, math.ceil(8 * 1.25 / column_room))
    sns.heatmap(
        pd.DataFrame(matrix, index=list(prepared.y_categories), columns=list(prepared.x_categories)),
        ax=ax,
        cmap=_colormap(prepared),
        center=0.0 if diverging else None,
        vmin=-bound if diverging else None,
        vmax=bound if diverging else None,
        mask=~np.isfinite(matrix),
        annot=annotate,
        fmt=f".{digits}f",
        annot_kws={"fontsize": 7},
        linewidths=0.5 if max(rows, columns) <= 40 else 0,
        linecolor="white",
        cbar_kws={"label": prepared.value_label, "shrink": 0.6 if rows > 12 else 0.9},
        xticklabels=every_column if every_column > 1 else True,
        yticklabels=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7 if columns > 25 else 8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7 if rows > 25 else 8)
    ax.set_xlabel(prepared.x_label)
    ax.set_ylabel(prepared.y_label)
    ax.tick_params(length=0)
    freeze_layout(fig)
    notes = ["Blank cells have no value."] if (~np.isfinite(matrix)).any() else []
    return fig, notes


def _clustered(prepared: PreparedPanel) -> Drawn:
    """A document x document matrix, reordered so similar documents sit together.

    Built by hand (scipy's dendrogram, seaborn's heatmap, one Figure) rather
    than with seaborn's clustermap, which creates its figure through pyplot.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    import seaborn as sns

    matrix = _matrix(prepared)
    labels = list(prepared.x_categories)
    count = len(labels)
    finite = matrix[np.isfinite(matrix)]
    top = float(finite.max()) if finite.size else 1.0
    # Distance is how far a pair falls below the most similar pair; the
    # diagonal (a document with itself) is distance zero.
    distance = np.where(np.isfinite(matrix), top - matrix, top - (float(finite.min()) if finite.size else 0.0))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(np.clip(distance, 0.0, None), checks=False), method="average")
    size = min(18.0, max(7.0, 3.2 + count * 0.13))
    fig = _figure(size + 1.2, size)
    fig.set_layout_engine("none")
    left, bar, gap = 0.12, 0.018, 0.004
    heat_left = left + bar + gap
    bottom, height = 0.07, 0.84
    ax_tree = fig.add_axes((0.01, bottom, left - 0.012, height))
    order = dendrogram(
        tree, ax=ax_tree, orientation="left", no_labels=True, color_threshold=0, above_threshold_color=AXIS
    )["leaves"][::-1]
    ax_tree.invert_yaxis()
    ax_tree.set_axis_off()
    ordered = [labels[i] for i in order]
    reordered = matrix[np.ix_(order, order)]
    decades = [decade_of(label) for label in ordered]
    known = sorted({d for d in decades if d})
    notes = [
        "Rows and columns are reordered by average-linkage clustering (distance = how far a pair falls below "
        "the most similar pair); the tree at the left shows which documents join first."
    ]
    if known:
        import matplotlib as mpl

        ramp = mpl.colormaps["viridis"]
        tone = {d: ramp(i / max(1, len(known) - 1)) for i, d in enumerate(known)}
        ax_bar = fig.add_axes((left, bottom, bar, height))
        ax_bar.imshow([[tone.get(d, (1, 1, 1, 1))] for d in decades], aspect="auto", interpolation="nearest")
        ax_bar.set_axis_off()
        handles = [Patch(color=tone[d], label=d) for d in known]
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.995),
            ncol=min(5, len(known)),
            fontsize=7,
            title="Decade (side bar)",
            title_fontsize=7,
        )
        notes.append("The side bar colours each document by its decade, so an era that clusters shows as one colour.")
    # The matrix, its labels to the right; the colour bar above it, clear of
    # the labels it used to sit on.
    ax = fig.add_axes((heat_left, bottom, 0.84 - heat_left, height))
    cax = fig.add_axes((0.62, bottom + height + 0.035, 0.2, 0.012))
    font = 8 if count <= 30 else 6.5 if count <= 60 else 5.2
    sns.heatmap(
        pd.DataFrame(reordered, index=ordered, columns=ordered),
        ax=ax,
        cbar_ax=cax,
        cmap=_colormap(prepared),
        mask=~np.isfinite(reordered),
        linewidths=0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": prepared.value_label, "orientation": "horizontal"},
    )
    cax.xaxis.set_label_position("top")
    ax.yaxis.tick_right()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=font)
    ax.tick_params(length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    cax.tick_params(labelsize=7)
    fig.canvas.draw()
    return fig, notes


# --------------------------------------------------------- small multiples --


def draw_small_multiples(prepared: PreparedPanel) -> Drawn:
    names = list(prepared.facets)
    columns = max(1, min(3, len(names)))
    rows = math.ceil(len(names) / columns)
    fig = _figure(_WIDTH_IN, max(3.0, 2.5 * rows + 0.5))
    axes = fig.subplots(rows, columns, sharex=True, squeeze=False)
    colors = group_colors(prepared.groups)
    points_only = set(prepared.points_only)
    xs = [mark.x for mark in prepared.marks]
    for index, name in enumerate(names):
        ax = axes[index // columns][index % columns]
        facet = [mark for mark in prepared.marks if mark.facet == name]
        by_group: dict[str, list[PanelMark]] = {}
        for mark in facet:
            by_group.setdefault(mark.group or UNGROUPED, []).append(mark)
        dots = [m for g, ms in by_group.items() if g in points_only for m in ms]
        _decade_shading(ax, xs)
        for group, marks in by_group.items():
            if group in points_only:
                ax.scatter(
                    [m.x for m in marks],
                    [m.y for m in marks],
                    s=7,
                    color=color_of(colors, group),
                    alpha=0.55,
                    linewidths=0,
                    zorder=3,
                )
                continue
            window = _summary_window(marks)
            if dots and window:
                band = rolling_band([m.x for m in dots], [m.y for m in dots], window)
                if band:
                    bx, lo, hi = zip(*band, strict=True)
                    ax.fill_between(bx, lo, hi, color=color_of(colors, group), alpha=0.18, linewidth=0, zorder=2)
            for segment in _segments(marks, prepared.line_gap):
                ax.plot(
                    [m.x for m in segment],
                    [m.y for m in segment],
                    color=INK if dots else color_of(colors, group),
                    linewidth=1.6,
                    zorder=4,
                )
        ax.set_title(wrap(name, 34), loc="left", fontsize=8.5, fontweight="bold", color=INK)
        ax.set_xlim(*_pad(min(xs), max(xs), 0.02))
        values = [m.y for m in facet]
        if values:
            ax.set_ylim(*_limits(values, top=0.1))
        _formatters(ax, xs, values)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x", visible=False)
    for index in range(len(names), rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    fig.supxlabel(prepared.x_label, fontsize=9, color=MUTED)
    freeze_layout(fig)
    notes = []
    if points_only:
        notes.append(
            "In each panel, dots are documents, the line their rolling median, and the band the middle half "
            "of the same windows. Each panel has its own vertical scale."
        )
    return fig, notes


# ------------------------------------------------------------------ network --


def draw_network(prepared: PreparedPanel) -> Drawn:
    elbow = prepared.edge_style == "elbow"
    fig = _figure(_WIDTH_IN, _height_for(prepared, 7.0))
    ax = fig.add_subplot()
    colors = group_colors(prepared.groups)
    by_key = {mark.key: mark for mark in prepared.marks}
    heaviest = max((edge.weight for edge in prepared.edges), default=0.0) or 1.0
    segments, widths, shades = [], [], []
    for edge in prepared.edges:
        source, target = by_key.get(edge.source), by_key.get(edge.target)
        if source is None or target is None:
            continue
        share = edge.weight / heaviest
        if elbow:
            segments.append([(source.x, source.y), (source.x, target.y), (target.x, target.y)])
        else:
            segments.append([(source.x, source.y), (target.x, target.y)])
        widths.append(0.4 + 3.4 * share if not elbow else 1.0)
        shades.append(to_rgba(AXIS, 0.25 + 0.55 * share if not elbow else 0.9))
    ax.add_collection(LineCollection(segments, linewidths=widths, colors=shades, zorder=1, capstyle="round"))
    radius = _radius_fn(prepared.marks, small=3.0, big=11.0)
    buckets, ordered = _grouped(prepared)
    handles = []
    for group in ordered:
        marks = buckets[group]
        handles.append(
            ax.scatter(
                [m.x for m in marks],
                [m.y for m in marks],
                s=[(2 * radius(m)) ** 2 if not (elbow and m.size is None and not m.labelled) else 6 for m in marks],
                color=color_of(colors, group),
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
        )
    ax.set_xlim(-0.42 if elbow else -0.04, 1.04)
    ax.set_ylim(1.04, -0.04)
    ax.set_axis_off()
    # A dendrogram's leaves and merges are not categories a reader compares;
    # a legend for them only takes room.
    if len(ordered) > 1 and ordered != [UNGROUPED] and not elbow:
        _legend(
            fig, ax, [Line2D([], [], marker="o", linestyle="", color=color_of(colors, g)) for g in ordered], ordered
        )
    notes: list[str] = []
    if prepared.x_label:
        notes.append(prepared.x_label)
    renderer = freeze_layout(fig)
    if elbow:
        leaves = [m for m in prepared.marks if m.labelled]
        font = 8 if len(leaves) <= 30 else 6.5 if len(leaves) <= 60 else 5.4
        for leaf in leaves:
            ax.text(leaf.x - 0.012, leaf.y, leaf.label, ha="right", va="center", fontsize=font, color=INK)
    else:
        labelled = sorted((m for m in prepared.marks if m.labelled), key=lambda m: -(m.size or 0.0))
        dropped, _ = place_labels(
            ax, [LabelRequest(m.x, m.y, m.label, radius(m)) for m in labelled], renderer, fontsize=7.5
        )
        note_dropped(ax, dropped)
        notes.append("Line width and darkness show each link's weight; node size shows each word's frequency.")
    return fig, notes


# ------------------------------------------------------------------- ribbon --


def draw_ribbon(prepared: PreparedPanel) -> Drawn:
    bands: dict[int, list[PanelMark]] = {}
    for mark in prepared.marks:
        bands.setdefault(int(mark.y), []).append(mark)
    rows = sorted(bands)
    fig = _figure(_WIDTH_IN, max(3.2, 1.6 + 0.28 * len(rows)))
    ax = fig.add_subplot()
    colors = group_colors(prepared.groups)
    for index, row in enumerate(rows):
        segments = sorted(bands[row], key=lambda m: m.x)
        ax.broken_barh(
            [(m.x, m.size or 0.0) for m in segments],
            (index - 0.38, 0.76),
            facecolors=[color_of(colors, m.group) for m in segments],
            linewidth=0,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([wrap(bands[row][0].label, _LABEL_WRAP) for row in rows], fontsize=7.5 if len(rows) > 25 else 8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    end = max(m.x + (m.size or 0.0) for m in prepared.marks)
    ax.set_xlim(0, end)
    if end <= 1.0001:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(False)
    ax.set_xlabel(prepared.x_label)
    _, ordered = _grouped(prepared)
    handles = [Patch(color=color_of(colors, g), label=g) for g in ordered if g != UNGROUPED]
    if handles:
        # As many columns as fit the plot's width: one row of six named
        # topics was wider than the plot, and the layout squeezed the ribbons
        # into a sliver to make room for it.
        longest = max(len(str(h.get_label())) for h in handles)
        fit = int(_WIDTH_IN * 72 * 0.8 / (longest * 7.5 * 0.56 + 28))
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.005),
            borderaxespad=0.0,
            ncol=max(1, min(6, len(handles), fit)),
            fontsize=7.5,
        )
    freeze_layout(fig)
    filled = [sum(m.size or 0.0 for m in bands[row]) for row in rows]
    spans = [max(m.x + (m.size or 0.0) for m in bands[row]) - min(m.x for m in bands[row]) for row in rows]
    gaps = any(span - fill > 1e-6 * max(1.0, span) for fill, span in zip(filled, spans, strict=True))
    return fig, (["A gap in a band is a passage too short to score."] if gaps else [])


# ------------------------------------------------------------------- stream --


def draw_stream(prepared: PreparedPanel) -> Drawn:
    buckets, ordered = _grouped(prepared)
    colors = group_colors(prepared.groups)
    axis = sorted({mark.x for mark in prepared.marks})
    position = {value: index for index, value in enumerate(axis)}
    stack = np.zeros((len(ordered), len(axis)))
    for row, group in enumerate(ordered):
        for mark in buckets[group]:
            stack[row, position[mark.x]] = mark.y
    fig = _figure(_WIDTH_IN, _height_for(prepared, 5.2))
    ax = fig.add_subplot()
    ax.stackplot(
        axis,
        stack,
        colors=[color_of(colors, g) for g in ordered],
        labels=ordered,
        alpha=0.9,
        linewidth=0.4,
        edgecolor="white",
    )
    ax.set_xlim(axis[0], axis[-1] if len(axis) > 1 else axis[0] + 1)
    ax.set_ylim(0, float(stack.sum(axis=0).max()) * 1.04 or 1.0)
    _formatters(ax, axis, [])
    ax.set_xlabel(prepared.x_label)
    ax.set_ylabel(prepared.y_label)
    ax.grid(axis="x", visible=False)
    if len(ordered) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, reverse=True)
    freeze_layout(fig)
    return fig, []


# ---------------------------------------------------------------- positions --


def draw_positions(prepared: PreparedPanel) -> Drawn:
    categories = list(prepared.y_categories)
    fig = _figure(_WIDTH_IN, max(3.4, 2.0 + 0.26 * len(categories)))
    grid = fig.add_gridspec(2, 1, height_ratios=(1, max(4, len(categories) / 4)), hspace=0.02)
    top = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1], sharex=top)
    colors = group_colors(prepared.groups)
    buckets, ordered = _grouped(prepared)
    xs = [mark.x for mark in prepared.marks]
    for group in ordered:
        marks = buckets[group]
        ax.vlines(
            [m.x for m in marks],
            [m.y - 0.36 for m in marks],
            [m.y + 0.36 for m in marks],
            color=color_of(colors, group),
            linewidth=1.1,
            alpha=0.85,
        )
    top.hist(xs, bins=40, color=MUTED, alpha=0.55)
    top.set_ylabel("all", fontsize=7)
    top.tick_params(labelbottom=False, labelsize=6.5)
    top.grid(False)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([wrap(label, _LABEL_WRAP) for label in categories], fontsize=7 if len(categories) > 30 else 8)
    ax.set_ylim(len(categories) - 0.5, -0.5)
    low, high = min(0.0, *xs), max(1.0, *xs)
    ax.set_xlim(low, high)
    if high <= 1.0001:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    else:
        ax.xaxis.set_major_formatter(AxisFormatter())
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(prepared.x_label)
    if len(ordered) > 1 and ordered != [UNGROUPED]:
        _legend(fig, ax, [Line2D([], [], color=color_of(colors, g), linewidth=2) for g in ordered], ordered)
    freeze_layout(fig)
    return fig, ["The strip on top counts every hit across all documents at each position."]


DRAWERS: dict[str, Callable[[PreparedPanel], Drawn]] = {
    "scatter_labelled": draw_scatter,
    "line_series": draw_line_series,
    "ranked_bars": draw_ranked_bars,
    "distribution": draw_distribution,
    "heatmap": draw_heatmap,
    "small_multiples": draw_small_multiples,
    "network": draw_network,
    "ribbon": draw_ribbon,
    "stream": draw_stream,
    "positions": draw_positions,
}

# ``tick_labels`` is re-exported for tests that compare the two renderers.
__all__ += ["tick_labels"]
