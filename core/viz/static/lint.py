"""What a drawn figure got wrong, measured from the figure itself.

``core/viz/figure_lint.py`` checks a prepared panel's *content*; this checks
the *drawing*: text printed over text, and tick labels that say "-0". Both
are found with the renderer's own extents, so the check agrees with what the
saved image shows. Tests run it over every panel; ``scripts/audit_figures.py``
runs it on real data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from matplotlib.figure import Figure
from matplotlib.text import Text

__all__ = ["DrawnProblem", "lint_figure"]


@dataclass(frozen=True, slots=True)
class DrawnProblem:
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def _undrawn_tick_labels(fig: Figure) -> set[int]:
    """Tick labels matplotlib keeps but does not draw: ticks outside the
    axis's view limits are skipped at draw time, yet stay "visible"."""
    hidden: set[int] = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            span = (high - low) or 1.0
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                location = tick.get_loc()
                if location is None or not (low - span * 1e-9 <= location <= high + span * 1e-9):
                    hidden.update((id(tick.label1), id(tick.label2)))
    return hidden


def _visible_texts(fig: Figure) -> list[Text]:
    hidden = _undrawn_tick_labels(fig)
    texts = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip() or id(artist) in hidden:
            continue
        texts.append(artist)
    return texts


def _really_overlap(a: Text, b: Text, box_a: Any, box_b: Any, dpi: float) -> bool:
    """Whether two texts whose *boxes* meet actually print over each other.

    Upright text fills its box, so boxes meeting is the answer. Two labels
    slanted at the same angle (a heatmap's column names) have boxes that
    always meet while the glyphs sit side by side; they overlap only if the
    gap between their baselines, across the slant, is less than a line.
    """
    angle_a, angle_b = a.get_rotation() % 180, b.get_rotation() % 180
    if angle_a == 0 or abs(angle_a - angle_b) > 1e-6:
        return True
    theta = math.radians(angle_a)
    ax_, ay = (box_a.x0 + box_a.x1) / 2, (box_a.y0 + box_a.y1) / 2
    bx, by = (box_b.x0 + box_b.x1) / 2, (box_b.y0 + box_b.y1) / 2
    across = abs(-(bx - ax_) * math.sin(theta) + (by - ay) * math.cos(theta))
    line = max(float(a.get_fontsize()), float(b.get_fontsize())) * dpi / 72.0 * 0.9
    return bool(across < line)


def lint_figure(fig: Figure, renderer: Any | None = None) -> list[DrawnProblem]:
    """Overlapping text and negative-zero ticks in a drawn figure."""
    renderer = renderer or fig.canvas.get_renderer()  # type: ignore[attr-defined]
    problems: list[DrawnProblem] = []
    texts = _visible_texts(fig)
    boxes = []
    for text in texts:
        try:
            box = text.get_window_extent(renderer)
        except (RuntimeError, ValueError):
            continue
        if box.width <= 0 or box.height <= 0:
            continue
        # A hair smaller than drawn: glyph boxes include side bearings, and
        # two labels that merely touch are readable.
        boxes.append((text, box.shrunk(0.94, 0.8)))
    overlaps = []
    for index, (a, box_a) in enumerate(boxes):
        for b, box_b in boxes[index + 1 :]:
            if box_a.overlaps(box_b) and _really_overlap(a, b, box_a, box_b, fig.dpi):
                overlaps.append((a.get_text(), b.get_text()))
    if overlaps:
        sample = "; ".join(f"{a[:30]!r} over {b[:30]!r}" for a, b in overlaps[:4])
        problems.append(DrawnProblem("TEXT_OVERLAP", f"{len(overlaps)} overlapping text pair(s): {sample}"))
    negative_zero = sorted(
        {t.get_text() for t in texts if t.get_text().strip().replace(chr(0x2212), "-") in ("-0", "-0.0", "-0.00")}
    )
    if negative_zero:
        problems.append(DrawnProblem("NEGATIVE_ZERO", f"tick label(s) {negative_zero}"))
    return problems
