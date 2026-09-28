"""The figure kit: the drawing pieces a script (and a showcase figure) builds on.

``docs/internal/SHOWCASE_FIGURES_PLAN.md`` section 3 names the pieces. They sit on
:mod:`core.viz.static` (house style, measured labels, the overlap linter,
number formatting), which already draws the app's publication panels, so a
kit figure and a panel figure are the same look with the same colours.

Every piece takes plain data -- a DataFrame, a list of ``(x, label)`` -- and
draws onto an axis the caller supplies, so the same call works on a script's
own tables and on a finished run's. Nothing here reads a file or a session;
:mod:`core.script.viz` is the thin ``nlp.viz`` door onto it.

The kit is deliberately the *public* way to draw a showcase: a showcase built
with it is also a worked example a script can copy, and the kit is exercised
by real figures before anyone relies on it.
"""

from __future__ import annotations

from core.viz.kit.canvas import Canvas, Row, canvas
from core.viz.kit.lint import LintProblem, lint
from core.viz.kit.palette import BACKGROUND, palette
from core.viz.kit.plots import bars, dots, events, mixture_bars, stream, tiles, time_axis, word_grid

__all__ = [
    "BACKGROUND",
    "Canvas",
    "LintProblem",
    "Row",
    "bars",
    "canvas",
    "dots",
    "events",
    "lint",
    "mixture_bars",
    "palette",
    "stream",
    "tiles",
    "time_axis",
    "word_grid",
]
