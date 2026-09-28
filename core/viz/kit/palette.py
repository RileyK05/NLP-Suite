"""Stable colours per key: the same topic is the same colour everywhere.

The kit's rule is the app's: keys are coloured by *declared order* with the
Okabe-Ito palette (colour-blind safe, in the same order), a remainder group is
grey without spending a palette colour, and one key may be declared the
``background`` -- drawn grey so the eye goes to the rest (the reference MALLET
figure greys the topic with the largest alpha for exactly this reason).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from core.viz.plotters import OKABE_ITO

__all__ = ["BACKGROUND", "palette"]

#: The grey a ``background`` key is drawn in (the app's remainder grey).
BACKGROUND = "#9aa39a"


def palette(
    keys: Iterable[Any],
    *,
    background: Any | None = None,
    colors: dict[Any, str] | None = None,
) -> dict[Any, str]:
    """One colour per key, by declared order.

    ``background`` is drawn :data:`BACKGROUND` and does not consume a palette
    slot; the rest take the Okabe-Ito palette in order. ``colors`` overrides
    individual keys (a caller pinning a colour for a concept).
    """
    ordered = list(keys)
    found: dict[Any, str] = {}
    slot = 0
    for key in ordered:
        if key == background:
            found[key] = BACKGROUND
            continue
        found[key] = OKABE_ITO[slot % len(OKABE_ITO)]
        slot += 1
    if colors:
        found.update(colors)
    return found
