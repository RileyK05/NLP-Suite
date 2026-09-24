"""House style for publication figures: one look, set in one place.

The palette is the app's (Okabe-Ito, colour-blind safe, in the same declared
order), so a group is the same colour in the interactive figure and in the
published one. The font is DejaVu Sans because it ships with matplotlib: the
same figure renders the same way on every machine, including the frozen
desktop engine, with no system font lookup.

Everything here is applied through :func:`house_style`, a context manager, so
rendering a figure never leaves matplotlib's global state changed for the
next caller (a notebook, a test, another renderer).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import contextlib
from typing import Any

from core.viz.plotters import OKABE_ITO

__all__ = [
    "AXIS",
    "DIVERGING",
    "GRID",
    "INK",
    "MUTED",
    "OKABE_ITO",
    "PALE",
    "SEQUENTIAL",
    "group_colors",
    "house_style",
]

#: Text and structure colours, matching the app's (desktop/src/PanelCanvas.tsx).
INK = "#2b3329"
MUTED = "#5d6a5c"
AXIS = "#9aa39a"
GRID = "#e8ece5"
PALE = "#f3f5f1"

#: Heatmap scales, low to high: the stops of ``panelspec.COLOR_SCALES``, so
#: the published heatmap and the app's use the same colours for a value.
SEQUENTIAL = ("#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b")
DIVERGING = ("#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac")

UNGROUPED = "(all)"

_RC: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.formatter.useoffset": False,
    "axes.formatter.limits": (-6, 9),
    "axes.unicode_minus": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "nlp-suite",
    "pdf.fonttype": 42,
    "text.color": INK,
    "image.cmap": "Blues",
}


@contextlib.contextmanager
def house_style() -> Iterator[None]:
    """Matplotlib's settings for one figure, restored afterwards."""
    import matplotlib as mpl

    with mpl.rc_context(_RC):
        yield


#: A remainder group -- named in parentheses, like "(smaller clusters)" -- is
#: drawn in this grey, so what is left over does not take a palette colour.
REMAINDER = "#9aa39a"


def is_remainder(group: str) -> bool:
    return group.startswith("(") and group.endswith(")") and group != UNGROUPED


def group_colors(groups: Sequence[str]) -> dict[str, str]:
    """Colour per group by *declared* order, the app's rule exactly
    (``panelLayout.groupColors``): no groups colours the implicit one blue,
    and a remainder group is grey without using up a palette colour."""
    ordered = list(groups) or [UNGROUPED]
    colors: dict[str, str] = {}
    index = 0
    for group in ordered:
        if is_remainder(group):
            colors[group] = REMAINDER
            continue
        colors[group] = OKABE_ITO[index % len(OKABE_ITO)]
        index += 1
    return colors


def color_of(colors: dict[str, str], group: str) -> str:
    return colors.get(group or UNGROUPED, OKABE_ITO[0])
