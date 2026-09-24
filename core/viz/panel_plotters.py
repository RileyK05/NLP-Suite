"""Plotly renderers for prepared panels: HTML, and image bytes for export.

The panel counterpart to ``core/viz/plotters.py``, and deliberately the same
shape: take something already decided (:class:`~core.viz.panelspec.
PreparedPanel` rather than a prepared chart), return ``Result[str]`` HTML or
``Result[bytes]`` image, and touch no filesystem — the caller persists bytes
through ``OutputWriter.write_bytes`` (R3: the writer stays the only writer).

What is different, and why:

* **The caption is drawn into the figure, not around it.** A provenance line
  in surrounding HTML is lost the moment anyone exports a PNG and pastes it
  into a document, which is exactly when the reader most needs to know where
  the numbers came from. :meth:`Provenance.caption` becomes a paper
  annotation inside the figure, so it survives every format.
* **Notes travel with the panel.** What a figure cannot show is part of the
  figure's meaning, so :attr:`PreparedPanel.notes` renders beneath the HTML
  and, being prose, is left out of the image — where it would not be legible
  anyway — with the caption still carrying the provenance.
* **Rendering is a dispatch on shape, not on kind.** :data:`IMPLEMENTED_SHAPES`
  is the authority on what can be drawn; ``core/viz/panels.py`` refuses a
  registered panel whose shape is not in it, so a panel is never half-served.
  Adding a shape means adding a branch here and one entry to that tuple.
* **Marks keep their identity.** Each point's hover text is its evidence's
  own description, so what the figure says about a mark and what a click on
  it would retrieve are the same sentence, written once.

Degradation matches the chart path exactly: without plotly the HTML becomes
an honest table of the same marks in the same order with a WARNING naming
the degradation; an explicitly requested image export fails loudly rather
than quietly substituting HTML.
"""

from __future__ import annotations

import html as html_lib
import math
from typing import TYPE_CHECKING, Any

from core.result import Diagnostic, Result
from core.viz.panelspec import COLOR_SCALES, Annotation, PanelMark, PreparedPanel
from core.viz.plotters import OKABE_ITO

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

__all__ = [
    "IMPLEMENTED_SHAPES",
    "build_panel_figure",
    "panel_html",
    "panel_image_bytes",
]

#: Shapes a renderer in this module can actually draw. The vocabulary in
#: ``panelspec.PANEL_SHAPES`` is larger on purpose: it names the geometries
#: the roadmap needs, and this tuple names the ones that exist. A panel
#: declaring a shape missing here is rejected before it runs.
IMPLEMENTED_SHAPES: tuple[str, ...] = (
    "scatter_labelled",
    "ranked_bars",
    "stream",
    "line_series",
    "ribbon",
    "small_multiples",
    "heatmap",
    "distribution",
    "network",
    "positions",
)

_FALLBACK_FIX = 'pip install "nlp-suite-ng[plotly]"'
_IMAGE_FIX = 'pip install "nlp-suite-ng[plotly-image]"'
_MARKER_MAX_PX = 26
_MARKER_MIN_PX = 5
_UNGROUPED = "(all)"
_CAPTION_SIZE = 10
_LABEL_SIZE = 11


def _group_color_map(groups: tuple[str, ...]) -> dict[str, str]:
    """Colour per group in the panel's declared draw order.

    Declared order, not sorted order: the panel already decided what the
    legend should read like, and re-sorting here would let two panels over
    the same data disagree about which group is blue.
    """
    # One rule for all three renderers; a remainder group ("(smaller
    # clusters)") is grey and takes no palette colour.
    from core.viz.static.style import group_colors

    return group_colors(list(groups) or [_UNGROUPED])


def _size_reference(marks: tuple[PanelMark, ...]) -> float | None:
    """Plotly's ``sizeref`` for area-proportional markers, or None.

    Area proportional rather than radius proportional: a mark standing for
    twice as many tokens should look twice as big, and radius scaling makes
    it look four times as big. This is plotly's documented formula.
    """
    sizes = [mark.size for mark in marks if mark.size is not None and mark.size > 0]
    if not sizes:
        return None
    return 2.0 * max(sizes) / float(_MARKER_MAX_PX**2)


def _marker_sizes(marks: list[PanelMark]) -> list[float]:
    return [float(mark.size) if mark.size is not None and mark.size > 0 else 0.0 for mark in marks]


def _grouped(prepared: PreparedPanel) -> tuple[dict[str, list[PanelMark]], list[str]]:
    """Marks bucketed by group, plus the order to draw them in.

    Declared order first so the legend reads as the panel intended; any group
    the panel did not declare follows, sorted, rather than vanishing. Shared
    by every shape so three renderers cannot disagree about which band is
    which colour.
    """
    by_group: dict[str, list[PanelMark]] = {}
    for mark in prepared.marks:
        by_group.setdefault(mark.group or _UNGROUPED, []).append(mark)
    ordered = [group for group in (prepared.groups or (_UNGROUPED,)) if group in by_group]
    ordered += sorted(group for group in by_group if group not in ordered)
    return by_group, ordered


def _scatter_labelled(prepared: PreparedPanel) -> Figure:
    """Points in a plane, selectively labelled, with reference lines."""
    import plotly.graph_objects as go

    colors = _group_color_map(prepared.groups)
    sizeref = _size_reference(prepared.marks)
    fig = go.Figure()

    by_group, ordered = _grouped(prepared)

    for group in ordered:
        marks = by_group[group]
        marker: dict[str, Any] = {
            "color": colors.get(group, OKABE_ITO[0]),
            "line": {"width": 0.5, "color": "rgba(255,255,255,0.65)"},
            "opacity": 0.82,
        }
        if sizeref:
            marker |= {
                "size": _marker_sizes(marks),
                "sizemode": "area",
                "sizeref": sizeref,
                "sizemin": _MARKER_MIN_PX,
            }
        else:
            marker["size"] = 8
        fig.add_trace(
            go.Scatter(
                x=[mark.x for mark in marks],
                y=[mark.y for mark in marks],
                mode="markers+text",
                name=group,
                # Text only where the panel asked for it: labelling every
                # point in a 200-word volcano produces an unreadable smear.
                text=[mark.label if mark.labelled else "" for mark in marks],
                textposition="top center",
                textfont={"size": _LABEL_SIZE},
                hovertext=[mark.evidence.describe for mark in marks],
                hoverinfo="text",
                marker=marker,
                showlegend=group != _UNGROUPED,
            )
        )

    for annotation in prepared.annotations:
        _add_reference_line(fig, annotation)
    _style(fig, prepared)
    return fig


def _ranked_bars(prepared: PreparedPanel) -> Figure:
    """Horizontal bars in a computed order, with an optional second series.

    The order is the panel's, not the renderer's: ``PanelMark.y`` carries the
    rank, and the category axis is pinned to it. Plotly inserts categories in
    trace-traversal order, which for two overlaid series would put the second
    series' rows wherever they first appeared -- the same desynchronisation
    ``plotters.py`` pins ``categoryarray`` to avoid.

    Two series overlay rather than sitting side by side, because the question
    they answer is "how far apart are these two readings of the same term?"
    and adjacent half-width bars make that a comparison of lengths across a
    gap instead of a comparison at a shared baseline.
    """
    import plotly.graph_objects as go

    by_group, ordered = _grouped(prepared)
    colors = _group_color_map(prepared.groups)

    # Rank 0 belongs at the top. Plotly's category axis counts upward, so the
    # array is built in reverse rank order rather than reversing the axis
    # (which would also flip the value axis on a horizontal bar).
    ranked: dict[str, float] = {}
    for mark in prepared.marks:
        ranked.setdefault(mark.label, mark.y)
    categories = [label for label, _ in sorted(ranked.items(), key=lambda item: -item[1])]

    fig = go.Figure()
    for index, group in enumerate(ordered):
        marks = by_group[group]
        fig.add_trace(
            go.Bar(
                x=[mark.x for mark in marks],
                y=[mark.label for mark in marks],
                orientation="h",
                name=group,
                marker={"color": colors.get(group, OKABE_ITO[0])},
                # The first series reads as the foreground; anything behind it
                # stays visible through it rather than hiding underneath.
                opacity=1.0 if index == 0 else 0.45,
                hovertext=[mark.evidence.describe for mark in marks],
                hoverinfo="text",
                showlegend=group != _UNGROUPED,
            )
        )
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=categories)
    _style(fig, prepared)
    return fig


def _stream(prepared: PreparedPanel) -> Figure:
    """Stacked bands over an ordered axis.

    Every band is filled to the one below it (``stackgroup``), so the height
    of a band is its own value and the height of the stack is the total. The
    x values stay numeric: a decade axis sorted as text puts 1940 between
    1930 and 2020, and a reader has no way to see that it happened.

    A band with a gap in the middle would be drawn as a straight line across
    the missing period, which invents data. Missing (bucket, group) cells are
    filled with zero instead, which is what "this topic was absent here"
    actually means on a stack.
    """
    import plotly.graph_objects as go

    by_group, ordered = _grouped(prepared)
    colors = _group_color_map(prepared.groups)
    axis = sorted({mark.x for mark in prepared.marks})

    fig = go.Figure()
    for group in ordered:
        at = {mark.x: mark for mark in by_group[group]}
        fig.add_trace(
            go.Scatter(
                x=axis,
                y=[at[value].y if value in at else 0.0 for value in axis],
                name=group,
                mode="lines",
                stackgroup="one",
                line={"width": 0.5, "color": colors.get(group, OKABE_ITO[0])},
                fillcolor=colors.get(group, OKABE_ITO[0]),
                hovertext=[at[value].evidence.describe if value in at else "" for value in axis],
                hoverinfo="text",
                showlegend=group != _UNGROUPED,
            )
        )
    _style(fig, prepared)
    return fig


def _broken_line(marks: list[PanelMark], gap: float) -> tuple[list[float | None], list[float | None], list[str | None]]:
    """x, y and hover lists in x order, with a ``None`` wherever x jumps by
    more than *gap*.

    ``panelLayout.ts::lineSeries`` applies the same rule (``mark.x - previous
    > gap`` starts a new segment), so a line broken in the export is broken in
    the app.
    """
    xs: list[float | None] = []
    ys: list[float | None] = []
    hovers: list[str | None] = []
    previous: float | None = None
    for mark in sorted(marks, key=lambda item: item.x):
        if previous is not None and mark.x - previous > gap:
            xs.append(None)
            ys.append(None)
            hovers.append(None)
        xs.append(mark.x)
        ys.append(mark.y)
        hovers.append(mark.evidence.describe)
        previous = mark.x
    return xs, ys, hovers


def _series_trace(prepared: PreparedPanel, group: str, marks: list[PanelMark], color: str, **extra: Any) -> Any:
    """One group's trace for ``line_series`` and ``small_multiples``.

    A group in ``points_only`` is drawn as markers and never joined: those are
    single documents behind a smoothed line, and connecting them in date order
    draws a trend out of noise.
    """
    import plotly.graph_objects as go

    if group in prepared.points_only:
        ordered = sorted(marks, key=lambda item: item.x)
        return go.Scatter(
            x=[mark.x for mark in ordered],
            y=[mark.y for mark in ordered],
            name=group,
            mode="markers",
            marker={"size": 5, "color": color, "opacity": 0.7},
            hovertext=[mark.evidence.describe for mark in ordered],
            hoverinfo="text",
            **extra,
        )
    xs, ys, hovers = _broken_line(marks, prepared.line_gap)
    return go.Scatter(
        x=xs,
        y=ys,
        name=group,
        mode="lines+markers",
        line={"color": color, "width": 2},
        marker={"size": 6},
        hovertext=hovers,
        hoverinfo="text",
        connectgaps=False,
        **extra,
    )


def _line_series(prepared: PreparedPanel) -> Figure:
    """Independent series, with uncovered stretches left visibly unconnected.

    The builder emits explicit zero-valued marks for years where the corpus
    has documents but no hits. An absent year therefore means no dated corpus
    exposure, and connecting across it would invent an observation. "Absent"
    is a gap wider than ``PreparedPanel.line_gap``: one for an annual series,
    whatever the builder chose for documents at fractional dates.
    """
    import plotly.graph_objects as go

    by_group, ordered = _grouped(prepared)
    colors = _group_color_map(prepared.groups)
    fig = go.Figure()
    for group in ordered:
        fig.add_trace(
            _series_trace(
                prepared,
                group,
                by_group[group],
                colors.get(group, OKABE_ITO[0]),
                showlegend=group != _UNGROUPED,
            )
        )
    _style(fig, prepared)
    return fig


def _ribbon(prepared: PreparedPanel) -> Figure:
    """Coloured segments along one axis: each document as a band of segments.

    The ribbon geometry is the panel's, not plotly's: a mark's ``x`` is where
    its segment starts along the band, ``size`` is how wide it is (same
    units), and ``y`` is the band's row (0 at the top). Plotly's horizontal
    bars count categories upward, so the bands are drawn with an explicit
    reversed row axis -- same trick the ranked-bars renderer uses for rank
    order. A segment with no topic (too short to score) is a gap: it is not
    drawn, and the band simply shows its neighbours.
    """
    import plotly.graph_objects as go

    by_group, ordered = _grouped(prepared)
    colors = _group_color_map(prepared.groups)

    fig = go.Figure()
    for group in ordered:
        marks = by_group[group]
        fig.add_trace(
            go.Bar(
                # base = segment start, x = width: the left edge is where the
                # paragraph starts, not a bar's length from zero.
                base=[mark.x for mark in marks],
                x=[mark.size if mark.size is not None else 0.0 for mark in marks],
                y=[mark.label for mark in marks],
                orientation="h",
                name=group,
                marker={"color": colors.get(group, OKABE_ITO[0])},
                hovertext=[mark.evidence.describe for mark in marks],
                hoverinfo="text",
                showlegend=group != _UNGROUPED,
            )
        )
    # Rank 0 at the top: build the category array in reverse row order.
    label_of_row: dict[float, str] = {}
    for mark in prepared.marks:
        label_of_row.setdefault(mark.y, mark.label)
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[label_of_row[row] for row in sorted(label_of_row, reverse=True)],
    )
    fig.update_layout(
        barmode="stack",
        bargap=0.25,
        xaxis_title=prepared.x_label,
        yaxis_title=prepared.y_label,
    )
    _style(fig, prepared)
    return fig


# ------------------------------------------------- rules shared with the app --
#
# Each function below is one rule ``desktop/src/panelLayout.ts`` implements
# again in TypeScript, with the same arithmetic in the same order so the two
# produce the same doubles. ``tests/test_panel_parity.py`` checks that the
# TypeScript tests pin the same numbers the Python tests pin.

#: Longest category label drawn in full; longer ones end in an ellipsis. The
#: ribbon's row labels already use this length in the app.
_LABEL_CHARS = 18

#: The fractional part of the golden ratio. Multiples of it, mod 1, spread
#: evenly over [0, 1) whatever their count, and deterministically, so a
#: distribution's points sit in the same place in both renderers and on
#: every run.
_JITTER_STEP = 0.6180339887498949
_JITTER_SPREAD = 0.25

#: Edge stroke width in pixels, from the lightest link to the heaviest.
_EDGE_MIN_PX = 1.0
_EDGE_MAX_PX = 6.0
_EDGE_COLOR = "#9aa39a"
_BOX_COLOR = "#8a938a"


def _truncate(label: str, limit: int = _LABEL_CHARS) -> str:
    return label if len(label) <= limit else f"{label[: limit - 1]}…"


def _heat_domain(values: list[float], scale: str) -> tuple[float, float]:
    """The value range a heatmap's colours span.

    ``sequential`` runs from the lowest cell to the highest. ``diverging`` is
    symmetric around zero, [-m, m] with m the largest magnitude, so a
    correlation of +0.3 and one of -0.3 are equally far from the neutral
    centre whatever the rest of the matrix holds.
    """
    if scale == "diverging":
        largest = max((abs(value) for value in values), default=0.0)
        return -largest, largest
    return min(values, default=0.0), max(values, default=0.0)


def _heat_color(value: float, low: float, high: float, stops: tuple[str, ...]) -> str:
    """The colour of *value*: linear in RGB between evenly spaced *stops*.

    A domain of one value (every cell equal) maps to the middle of the scale
    rather than dividing by zero. Channels round half up, as
    ``Math.floor(c + 0.5)`` does in the app; Python's ``round`` rounds half to
    even and would disagree on exactly the halfway cells.
    """
    t = 0.5 if high == low else (value - low) / (high - low)
    t = min(1.0, max(0.0, t))
    scaled = t * (len(stops) - 1)
    index = min(math.floor(scaled), len(stops) - 2)
    fraction = scaled - index
    start, end = _rgb(stops[index]), _rgb(stops[index + 1])
    channels = [math.floor(a + (b - a) * fraction + 0.5) for a, b in zip(start, end, strict=True)]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)


def _quantile(ordered: list[float], p: float) -> float:
    """Linear interpolation between order statistics at position (n-1)*p.

    numpy's default ("type 7"). Stated here rather than delegated to numpy or
    plotly because the app computes the same box in TypeScript, and plotly's
    own box statistics use a different method by default.
    """
    position = (len(ordered) - 1) * p
    below = math.floor(position)
    above = math.ceil(position)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)


def _box_stats(values: list[float]) -> dict[str, float]:
    """Whiskers to the extremes, not 1.5 IQR: every point is drawn anyway, so
    nothing is hidden by calling it an outlier."""
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "q1": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "q3": _quantile(ordered, 0.75),
        "max": ordered[-1],
    }


def _jitter(index: int) -> float:
    """Vertical offset of the *index*-th point of a row, in row units.

    Deterministic and spread within ±0.25 of the row's centre. Not evenly
    stepped in index order: a builder that sorts its points by value would
    then draw every row as a diagonal staircase.
    """
    step = index * _JITTER_STEP
    return (step - math.floor(step) - 0.5) * 2 * _JITTER_SPREAD


def _jitter_offsets(marks: tuple[PanelMark, ...]) -> list[float]:
    """Each mark's jitter, numbered by its order among its row's marks."""
    seen: dict[float, int] = {}
    offsets: list[float] = []
    for mark in marks:
        index = seen.get(mark.y, 0)
        seen[mark.y] = index + 1
        offsets.append(_jitter(index))
    return offsets


def _edge_width(weight: float, heaviest: float) -> float:
    """1px for the lightest possible link up to 6px for the heaviest, linear
    on weight / max weight."""
    share = weight / heaviest if heaviest > 0 else 0.0
    return _EDGE_MIN_PX + (_EDGE_MAX_PX - _EDGE_MIN_PX) * share


def _edge_points(source: PanelMark, target: PanelMark, style: str) -> list[tuple[float, float]]:
    """The path an edge takes. ``elbow``: source → (source.x, target.y) →
    target, the right-angled link of a dendrogram."""
    if style == "elbow":
        return [(source.x, source.y), (source.x, target.y), (target.x, target.y)]
    return [(source.x, source.y), (target.x, target.y)]


def _facet_grid(count: int) -> tuple[int, int]:
    """(rows, columns) for *count* small multiples: at most three across."""
    columns = max(1, min(3, count))
    return math.ceil(count / columns), columns


def _row_axis(fig: Figure, categories: tuple[str, ...]) -> None:
    """Row indexes labelled with their categories, row 0 at the top.

    Numeric indexes rather than category strings, so two documents with the
    same title stay two rows. Plotly's y axis grows upward, so it is
    reversed -- the SVG renderer draws row order directly.
    """
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(categories))),
        ticktext=[_truncate(label) for label in categories],
        autorange="reversed",
        automargin=True,
        showgrid=False,
        zeroline=False,
    )


def _heatmap(prepared: PreparedPanel) -> Figure:
    """A matrix of cells, each coloured by its value on the declared scale.

    The z matrix is built from the marks, so a cell with no mark is a gap
    (``None``), not a zero. The colourscale is the same stops the app uses,
    over the same domain (:func:`_heat_domain`).
    """
    import plotly.graph_objects as go

    stops = COLOR_SCALES[prepared.color_scale]
    columns, rows = len(prepared.x_categories), len(prepared.y_categories)
    z: list[list[float | None]] = [[None] * columns for _ in range(rows)]
    hover: list[list[str]] = [[""] * columns for _ in range(rows)]
    values: list[float] = []
    for mark in prepared.marks:
        value = float(mark.value) if mark.value is not None else 0.0
        z[int(mark.y)][int(mark.x)] = value
        hover[int(mark.y)][int(mark.x)] = mark.evidence.describe
        values.append(value)
    low, high = _heat_domain(values, prepared.color_scale)
    if high == low:
        # One value everywhere: centre it, as _heat_color does.
        low, high = low - 1, high + 1
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(range(columns)),
            y=list(range(rows)),
            colorscale=[[index / (len(stops) - 1), stop] for index, stop in enumerate(stops)],
            zmin=low,
            zmax=high,
            hovertext=hover,
            hoverinfo="text",
            hoverongaps=False,
            xgap=1,
            ygap=1,
            colorbar={"thickness": 12, "len": 0.6},
        )
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(columns)),
        ticktext=[_truncate(label) for label in prepared.x_categories],
        tickangle=-45,
        automargin=True,
        showgrid=False,
    )
    _row_axis(fig, prepared.y_categories)
    _style(fig, prepared)
    return fig


def _distribution(prepared: PreparedPanel) -> Figure:
    """A box per row with every observation drawn over it.

    The box statistics are computed here (:func:`_box_stats`) and handed to
    plotly precomputed, so plotly's own quartile method -- which differs from
    the app's -- never decides where a box edge falls.
    """
    import plotly.graph_objects as go

    by_row: dict[int, list[float]] = {}
    for mark in prepared.marks:
        by_row.setdefault(int(mark.y), []).append(mark.x)
    rows = sorted(by_row)
    stats = [_box_stats(by_row[row]) for row in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=rows,
            q1=[s["q1"] for s in stats],
            median=[s["median"] for s in stats],
            q3=[s["q3"] for s in stats],
            lowerfence=[s["min"] for s in stats],
            upperfence=[s["max"] for s in stats],
            orientation="h",
            name="spread",
            width=0.6,
            line={"color": _BOX_COLOR, "width": 1},
            fillcolor="rgba(138,147,138,0.12)",
            boxpoints=False,
            showlegend=False,
        )
    )
    offsets = dict(zip((mark.key for mark in prepared.marks), _jitter_offsets(prepared.marks), strict=True))
    by_group, ordered = _grouped(prepared)
    colors = _group_color_map(prepared.groups)
    for group in ordered:
        marks = by_group[group]
        fig.add_trace(
            go.Scatter(
                x=[mark.x for mark in marks],
                y=[mark.y + offsets[mark.key] for mark in marks],
                mode="markers",
                name=group,
                marker={"size": 6, "color": colors.get(group, OKABE_ITO[0]), "opacity": 0.8},
                hovertext=[mark.evidence.describe for mark in marks],
                hoverinfo="text",
                showlegend=group != _UNGROUPED,
            )
        )
    _row_axis(fig, prepared.y_categories)
    _style(fig, prepared)
    return fig


def _positions(prepared: PreparedPanel) -> Figure:
    """A tick per mark, at its position along its row's document."""
    import plotly.graph_objects as go

    by_group, ordered = _grouped(prepared)
    colors = _group_color_map(prepared.groups)
    fig = go.Figure()
    for group in ordered:
        marks = by_group[group]
        color = colors.get(group, OKABE_ITO[0])
        fig.add_trace(
            go.Scatter(
                x=[mark.x for mark in marks],
                y=[mark.y for mark in marks],
                mode="markers",
                name=group,
                marker={"symbol": "line-ns-open", "size": 14, "color": color, "line": {"width": 1.5, "color": color}},
                hovertext=[mark.evidence.describe for mark in marks],
                hoverinfo="text",
                showlegend=group != _UNGROUPED,
            )
        )
    xs = [mark.x for mark in prepared.marks]
    # The whole document, 0 to 1, even when every hit falls in its first half:
    # where the ticks are not is part of the reading.
    fig.update_xaxes(range=[min(0.0, *xs), max(1.0, *xs)])
    _row_axis(fig, prepared.y_categories)
    _style(fig, prepared)
    return fig


def _network(prepared: PreparedPanel) -> Figure:
    """Nodes where the builder placed them, edges beneath them.

    The builder's layout is in [0, 1] x [0, 1] with y = 0 at the top, so the
    y axis is reversed and neither axis shows ticks: the coordinates are a
    layout, not a measurement. Each edge is its own line trace (its width is
    its weight); plotly cannot hover a line segment, so a transparent marker
    at the middle of each path carries the edge's evidence.
    """
    import plotly.graph_objects as go

    nodes = {mark.key: mark for mark in prepared.marks}
    heaviest = max((edge.weight for edge in prepared.edges), default=0.0)
    fig = go.Figure()
    middles: list[tuple[float, float, str]] = []
    for edge in prepared.edges:
        points = _edge_points(nodes[edge.source], nodes[edge.target], prepared.edge_style)
        fig.add_trace(
            go.Scatter(
                x=[x for x, _ in points],
                y=[y for _, y in points],
                mode="lines",
                line={"color": _EDGE_COLOR, "width": _edge_width(edge.weight, heaviest)},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        middle = points[len(points) // 2] if len(points) % 2 else _midpoint(points[0], points[1])
        middles.append((middle[0], middle[1], edge.evidence.describe))
    if middles:
        fig.add_trace(
            go.Scatter(
                x=[x for x, _, _ in middles],
                y=[y for _, y, _ in middles],
                mode="markers",
                marker={"size": 10, "color": "rgba(0,0,0,0)"},
                hovertext=[text for _, _, text in middles],
                hoverinfo="text",
                name="links",
                showlegend=False,
            )
        )
    colors = _group_color_map(prepared.groups)
    sizeref = _size_reference(prepared.marks)
    by_group, ordered = _grouped(prepared)
    for group in ordered:
        marks = by_group[group]
        marker: dict[str, Any] = {
            "color": colors.get(group, OKABE_ITO[0]),
            "line": {"width": 0.5, "color": "rgba(255,255,255,0.8)"},
        }
        if sizeref:
            marker |= {"size": _marker_sizes(marks), "sizemode": "area", "sizeref": sizeref, "sizemin": _MARKER_MIN_PX}
        else:
            marker["size"] = 10
        fig.add_trace(
            go.Scatter(
                x=[mark.x for mark in marks],
                y=[mark.y for mark in marks],
                mode="markers+text",
                name=group,
                text=[mark.label if mark.labelled else "" for mark in marks],
                # A dendrogram's leaves stand in one column a row apart: a
                # label above each would sit on the leaf above it, so they
                # are written to the left instead (the app does the same).
                textposition="middle left" if prepared.edge_style == "elbow" else "top center",
                textfont={"size": _LABEL_SIZE},
                hovertext=[mark.evidence.describe for mark in marks],
                hoverinfo="text",
                marker=marker,
                showlegend=group != _UNGROUPED,
            )
        )
    hidden = {"showticklabels": False, "showgrid": False, "zeroline": False}
    fig.update_xaxes(range=[-0.35 if prepared.edge_style == "elbow" else -0.05, 1.05], **hidden)
    fig.update_yaxes(range=[1.05, -0.05], **hidden)
    _style(fig, prepared)
    return fig


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def _small_multiples(prepared: PreparedPanel) -> Figure:
    """One small plot per facet, sharing x, each with its own y scale.

    The independent y axes are the point: a speech's length in words and its
    average sentence length do not share units, and one scale would flatten
    one of them into a line along the floor.
    """
    from plotly.subplots import make_subplots

    rows, columns = _facet_grid(len(prepared.facets))
    fig = make_subplots(
        rows=rows,
        cols=columns,
        shared_xaxes="all",
        shared_yaxes=False,
        subplot_titles=list(prepared.facets),
        vertical_spacing=0.12 if rows > 1 else 0.0,
        horizontal_spacing=0.08,
    )
    colors = _group_color_map(prepared.groups)
    in_legend: set[str] = set()
    for index, facet in enumerate(prepared.facets):
        row, column = index // columns + 1, index % columns + 1
        facet_marks = [mark for mark in prepared.marks if mark.facet == facet]
        by_group: dict[str, list[PanelMark]] = {}
        for mark in facet_marks:
            by_group.setdefault(mark.group or _UNGROUPED, []).append(mark)
        ordered = [group for group in (prepared.groups or (_UNGROUPED,)) if group in by_group]
        ordered += sorted(group for group in by_group if group not in ordered)
        for group in ordered:
            trace = _series_trace(
                prepared,
                group,
                by_group[group],
                colors.get(group, OKABE_ITO[0]),
                legendgroup=group,
                showlegend=group != _UNGROUPED and group not in in_legend,
            )
            in_legend.add(group)
            fig.add_trace(trace, row=row, col=column)
    _style(fig, prepared)
    # _style titles the first axes; in a grid the x title belongs under the
    # bottom plot of each column, and the facet titles already name each y.
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")
    for column in range(1, columns + 1):
        last = max(index for index in range(len(prepared.facets)) if index % columns == column - 1)
        fig.update_xaxes(title_text=prepared.x_label, row=last // columns + 1, col=column)
    return fig


def _add_reference_line(fig: Figure, annotation: Annotation) -> None:
    """A reference line labelled with what it means, not just where it is."""
    if annotation.kind == "note":
        return
    shared: dict[str, Any] = {
        "line_dash": "dot",
        "line_color": "#8a938a",
        "line_width": 1,
        "annotation_text": annotation.label,
        "annotation_font": {"size": _CAPTION_SIZE, "color": "#5d6a5c"},
    }
    if annotation.kind == "vline":
        fig.add_vline(x=annotation.value, annotation_position="top", **shared)
    else:
        fig.add_hline(y=annotation.value, annotation_position="right", **shared)


def _style(fig: Figure, prepared: PreparedPanel) -> None:
    """Title, subtitle, axes and the provenance caption inside the figure."""
    title = html_lib.escape(prepared.title)
    if prepared.subtitle:
        title += f"<br><sup>{html_lib.escape(prepared.subtitle)}</sup>"
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.02, "xanchor": "left"},
        xaxis_title=prepared.x_label,
        yaxis_title=prepared.y_label,
        width=prepared.width,
        height=prepared.height,
        # Room at the top for the subtitle and at the bottom for the caption.
        margin={"t": 90, "b": 96, "l": 72, "r": 32},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
    )
    fig.add_annotation(
        text=html_lib.escape(prepared.caption),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.16,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        font={"size": _CAPTION_SIZE, "color": "#6b756a"},
    )


def build_panel_figure(prepared: PreparedPanel) -> Result[Figure]:
    """Build the plotly figure for a prepared panel (no I/O, no writes)."""
    if prepared.shape not in IMPLEMENTED_SHAPES:
        return Result.failure(
            Diagnostic.error(
                "PANEL_SHAPE_UNIMPLEMENTED",
                f"no renderer draws shape {prepared.shape!r} (implemented: {', '.join(IMPLEMENTED_SHAPES)})",
                shape=prepared.shape,
            )
        )
    if not prepared.marks:
        return Result.failure(
            Diagnostic.error("PANEL_NO_MARKS", f"{prepared.panel} prepared no marks, so there is nothing to draw")
        )
    renderers = {
        "scatter_labelled": _scatter_labelled,
        "ranked_bars": _ranked_bars,
        "stream": _stream,
        "line_series": _line_series,
        "ribbon": _ribbon,
        "heatmap": _heatmap,
        "distribution": _distribution,
        "positions": _positions,
        "network": _network,
        "small_multiples": _small_multiples,
    }
    return Result.success(renderers[prepared.shape](prepared))


def panel_html(prepared: PreparedPanel, *, offline: bool = True) -> Result[str]:
    """Render to self-contained HTML: the figure, its caption, and its limits."""
    try:
        built = build_panel_figure(prepared)
        if built.value is None:
            return Result[str](None, built.diagnostics)
        include = "inline" if offline else "cdn"
        body: str = built.unwrap().to_html(full_html=False, include_plotlyjs=include)
        return Result.success(_wrap(body, prepared))
    except ImportError:
        return Result.success(
            _wrap(_fallback_table(prepared), prepared),
            Diagnostic.warning(
                "PANEL_PLOTLY_UNAVAILABLE",
                "plotly is not installed; rendered a plain HTML table of the same marks in the same order "
                f"instead. Fix: {_FALLBACK_FIX}",
                fix=_FALLBACK_FIX,
            ),
        )
    except Exception as exc:  # plotly can raise broadly; one clear diagnostic
        return Result[str].failure(Diagnostic.error("PANEL_RENDER_FAILED", f"plotly render failed: {exc}"))


def _wrap(body: str, prepared: PreparedPanel) -> str:
    """The figure inside a ``<figure>``, with caption and notes beneath it."""
    notes = ""
    if prepared.notes:
        items = "".join(f"<li>{html_lib.escape(note)}</li>" for note in prepared.notes)
        notes = (
            '<details class="panel-notes" open><summary>What this can and cannot show</summary>'
            f"<ul>{items}</ul></details>"
        )
    caption = html_lib.escape(prepared.caption)
    return (
        '<figure class="panel" style="margin:0">'
        f"{body}"
        f'<figcaption style="font-size:11px;color:#6b756a;margin-top:6px">{caption}</figcaption>'
        f"{notes}"
        "</figure>"
    )


def _fallback_table(prepared: PreparedPanel) -> str:
    """Honest table for the no-plotly fallback: same marks, same order."""
    header = "".join(
        f"<th>{html_lib.escape(name)}</th>"
        for name in ("Mark", "Group", prepared.x_label, prepared.y_label, "Stands for")
    )
    rows = ""
    for mark in prepared.marks:
        cells = (
            html_lib.escape(mark.label),
            html_lib.escape(mark.group or _UNGROUPED),
            _number(mark.x),
            _number(mark.y),
            html_lib.escape(mark.evidence.describe),
        )
        rows += "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
    lines = "".join(
        f"<li>{html_lib.escape(a.label)} at {_number(a.value)} — {html_lib.escape(a.note)}</li>"
        for a in prepared.annotations
        if a.note
    )
    reference = f"<p>Reference lines:</p><ul>{lines}</ul>" if lines else ""
    return (
        f"<h3>{html_lib.escape(prepared.title)}</h3>"
        f"<p><em>{html_lib.escape(prepared.subtitle)}</em></p>"
        f"<p>Panel shape: {html_lib.escape(prepared.shape)} (data table fallback — plotly not installed)</p>"
        "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
        f"<tr>{header}</tr>{rows}</table>{reference}"
    )


def _number(value: float) -> str:
    return f"{value:.6g}" if math.isfinite(value) else "—"


def panel_image_bytes(
    prepared: PreparedPanel,
    fmt: str = "png",
    *,
    browser_path: str | None = None,
) -> Result[bytes]:
    """Export the panel as in-memory image bytes (PNG/SVG/PDF).

    Bytes only — the caller writes them through ``OutputWriter.write_bytes``
    and registers the artifact (R3). The provenance caption is inside the
    figure, so it is present in every format. A missing kaleido or browser
    is a precise failure: an explicitly requested export never silently
    degrades to HTML.
    """
    if fmt not in ("png", "svg", "pdf"):
        return Result[bytes].failure(
            Diagnostic.error("PANEL_BAD_FORMAT", f"format must be png, svg, or pdf, got {fmt!r}")
        )
    try:
        built = build_panel_figure(prepared)
    except ImportError:
        return Result[bytes].failure(
            Diagnostic.error(
                "PANEL_IMAGE_PLOTLY_MISSING",
                "image export needs plotly, which is not installed",
                fix=_FALLBACK_FIX,
            )
        )
    if built.value is None:
        return Result[bytes](None, built.diagnostics)
    try:
        return Result.success(_kaleido_bytes(built.unwrap(), prepared, fmt, browser_path))
    except ImportError as exc:
        return Result[bytes].failure(
            Diagnostic.error(
                "PANEL_IMAGE_KALEIDO_MISSING",
                f"image export needs kaleido: {exc}. Install the export extra: {_IMAGE_FIX}",
                fix=_IMAGE_FIX,
            )
        )
    except Exception as exc:
        return Result[bytes].failure(_image_diagnostic(exc))


def _kaleido_bytes(fig: Any, prepared: PreparedPanel, fmt: str, browser_path: str | None) -> bytes:
    """One kaleido v1 call, with an optional per-call browser path.

    The same call shape as ``plotters._kaleido_bytes``: ``calc_fig_sync``
    with ``kopts={"path": ...}``, which raises ``ChromeNotFoundError`` for a
    nonexistent executable rather than silently falling back. No global
    configuration is touched.
    """
    import kaleido  # type: ignore[import-not-found]  # optional [plotly-image] extra

    kopts: dict[str, Any] = {}
    if browser_path:
        kopts["path"] = browser_path
    img_bytes: bytes = kaleido.calc_fig_sync(
        fig.to_plotly_json(),
        opts={"format": fmt, "width": prepared.width, "height": prepared.height},
        kopts=kopts,
    )
    return img_bytes


def _image_diagnostic(exc: Exception) -> Diagnostic:
    """kaleido/Chrome failures get the actionable fix; others stay generic."""
    message = str(exc)
    lowered = message.lower()
    if type(exc).__name__ == "ChromeNotFoundError" or any(
        token in lowered for token in ("chrome", "chromium", "kaleido")
    ):
        return Diagnostic.error(
            "PANEL_IMAGE_BROWSER_NOT_FOUND",
            f"image export failed: {message}. Kaleido v1 needs Chrome/Chromium; point at an installed "
            "browser with the BROWSER_PATH environment variable or pass --browser-path. "
            f"Install the export extra with: {_IMAGE_FIX}",
            fix=_IMAGE_FIX,
        )
    return Diagnostic.error("PANEL_IMAGE_FAILED", f"image export failed: {message}")
