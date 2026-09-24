"""Collision-free point labels, placed against measured text.

The app's ``placeLabels`` estimates text width from a table of character
widths; here the renderer is at hand, so every candidate is *measured*. The
rule is otherwise the same: labels in priority order, each trying positions
around its point, taking the first that stays inside the axes and meets no
label already placed (and preferring one that covers no other point). A label
with no free position is dropped -- an overprinted label is unreadable, and
the published figure's caption says how many were left out.

The figure's layout must be final before this runs: positions are measured
in display space, and a layout engine that moved the axes afterwards would
move the points out from under their labels. :func:`freeze_layout` does that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from core.viz.static.style import INK, MUTED

__all__ = ["LabelRequest", "freeze_layout", "place_labels"]


@dataclass(frozen=True, slots=True)
class LabelRequest:
    """One point to label: data coordinates, marker radius in points, text."""

    x: float
    y: float
    text: str
    radius: float = 4.0


# Offsets as multiples of (radius + gap), with alignment: above, below,
# right, left, then the four diagonals -- the app's order.
_CANDIDATES: tuple[tuple[float, float, str, str], ...] = (
    (0.0, 1.0, "center", "bottom"),
    (0.0, -1.0, "center", "top"),
    (1.0, 0.0, "left", "center"),
    (-1.0, 0.0, "right", "center"),
    (0.75, 0.75, "left", "bottom"),
    (-0.75, 0.75, "right", "bottom"),
    (0.75, -0.75, "left", "top"),
    (-0.75, -0.75, "right", "top"),
)


def freeze_layout(fig: Figure) -> Any:
    """Run the layout once and switch it off; return the renderer."""
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig.canvas.get_renderer()  # type: ignore[attr-defined]


def place_labels(  # noqa: PLR0913 - keyword-only drawing options
    ax: Axes,
    requests: Sequence[LabelRequest],
    renderer: Any,
    *,
    fontsize: float = 7.5,
    obstacles: Sequence[Bbox] = (),
    color: str = INK,
) -> tuple[int, list[Bbox]]:
    """Place what fits; return (how many were dropped, the placed boxes)."""
    fig = ax.figure
    dpi_scale = fig.dpi / 72.0
    area = ax.get_window_extent(renderer)
    placed: list[Bbox] = list(obstacles)
    kept: list[Bbox] = []
    point_boxes = []
    for request in requests:
        px, py = ax.transData.transform((request.x, request.y))
        r = request.radius * dpi_scale
        point_boxes.append(Bbox.from_extents(px - r, py - r, px + r, py + r))
    dropped = 0
    for index, request in enumerate(requests):
        gap = request.radius + 2.5
        chosen = None
        fallback = None
        for dx, dy, ha, va in _CANDIDATES:
            text = ax.annotate(
                request.text,
                (request.x, request.y),
                xytext=(dx * gap, dy * gap),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=fontsize,
                color=color,
                annotation_clip=False,
                zorder=6,
            )
            box = text.get_window_extent(renderer).expanded(1.04, 1.1)
            inside = box.x0 >= area.x0 and box.x1 <= area.x1 and box.y0 >= area.y0 and box.y1 <= area.y1
            clear = inside and not any(box.overlaps(other) for other in placed)
            if not clear:
                text.remove()
                continue
            covers_point = any(box.overlaps(point) for other, point in enumerate(point_boxes) if other != index)
            if not covers_point:
                if fallback is not None:
                    fallback[0].remove()
                chosen = (text, box)
                break
            if fallback is None:
                fallback = (text, box)
            else:
                text.remove()
        if chosen is None:
            chosen = fallback
        if chosen is None:
            dropped += 1
            continue
        placed.append(chosen[1])
        kept.append(chosen[1])
    return dropped, kept


def note_dropped(ax: Axes, dropped: int) -> None:
    """Say, inside the plot's corner, how many labels did not fit."""
    if dropped:
        ax.text(
            0.995,
            0.005,
            f"{dropped} label(s) omitted where they would overlap",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
        )
