"""Numbers and words as a publication figure prints them.

The same rules as the app's ``desktop/src/panelTicks.ts`` -- no "-0", one
precision per axis, one magnitude suffix per axis, years ungrouped -- so an
axis reads the same in both renderers. Wrapping lives here too: a published
figure never cuts a label, it wraps it or makes room.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
import math
import textwrap

from matplotlib.ticker import Formatter

__all__ = ["AxisFormatter", "decimals_of", "format_value", "tick_labels", "wrap"]


def decimals_of(value: float) -> int:
    """The fewest decimals that write ``value`` exactly (to 1e-6 of a unit)."""
    for digits in range(9):
        scaled = abs(value) * 10**digits
        if abs(scaled - round(scaled)) < 1e-6 * max(1.0, scaled):
            return digits
    return 8


def tick_labels(ticks: Sequence[float], *, plain: bool = False) -> list[str]:
    """Labels for one axis's ticks at one precision (``panelTicks.tickLabels``)."""
    values = [float(tick) for tick in ticks]
    if not values:
        return []
    steps = [abs(b - a) for a, b in pairwise(values) if abs(b - a) > 0]
    step = min(steps) if steps else (abs(values[0]) or 1.0)
    if plain:
        digits = decimals_of(step)
        return [f"{_zero(v):.{digits}f}" for v in values]
    largest = max(abs(v) for v in values)
    divisor, suffix = (
        (1e9, "B") if largest >= 1e9 else (1e6, "M") if largest >= 1e6 else (1e3, "k") if largest >= 1e4 else (1.0, "")
    )
    digits = min(6, decimals_of(step / divisor))
    out = []
    for v in values:
        scaled = _zero(v / divisor)
        if scaled == 0:
            out.append("0")
        elif divisor == 1 and digits == 0:
            out.append(f"{scaled:,.0f}")
        else:
            out.append(f"{scaled:,.{digits}f}{suffix}")
    return out


def _zero(value: float) -> float:
    return 0.0 if abs(value) < 1e-12 else value


def format_value(value: float) -> str:
    """A single number for a label or an annotation: three significant
    figures, thousands grouped, never "-0"."""
    if not math.isfinite(value):
        return "—"
    value = _zero(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    whole = float(value).is_integer()
    if whole and magnitude < 1e4:
        # A count prints as a count: "85", never "85.0" beside "9".
        return f"{value:,.0f}"
    if magnitude >= 1e6:
        return f"{value / 1e6:,.2f}M"
    if magnitude >= 1e4:
        return f"{value / 1e3:,.1f}k"
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:.3g}" if magnitude < 10 else f"{value:.1f}"
    return f"{value:.2g}"


class AxisFormatter(Formatter):
    """Matplotlib tick formatter applying :func:`tick_labels` to a whole axis."""

    def __init__(self, *, plain: bool = False) -> None:
        self.plain = plain

    def __call__(self, x: float, pos: int | None = None) -> str:
        return tick_labels([x], plain=self.plain)[0]

    def format_ticks(self, values: Sequence[float]) -> list[str]:
        return tick_labels(values, plain=self.plain)


def wrap(text: str, width: int) -> str:
    """``text`` wrapped at ``width`` characters, words kept whole."""
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)) or text


def is_year_axis(values: Sequence[float]) -> bool:
    """Whole-number-ish values in a calendar range print without grouping."""
    finite = [v for v in values if math.isfinite(v)]
    return bool(finite) and min(finite) >= 1000 and max(finite) <= 2500
