"""The overlap linter, returned as readable problems.

``docs/internal/SHOWCASE_FIGURES_PLAN.md`` section 1.8: the reference figure's first
render had a clipped label, a label on a stopline and a title running into an
axis; the linter catches some of it and a look at the image catches the rest.
This is the linter as a script calls it -- ``viz.lint(fig)`` -- over the
drawing itself (``core.viz.static.lint``), with the messages a reader can act
on ("label 'war, free, men' overlaps a line").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["LintProblem", "lint"]


@dataclass(frozen=True, slots=True)
class LintProblem:
    """One problem found in a drawn figure."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def lint(fig: Any) -> list[LintProblem]:
    """Overlapping text and negative-zero ticks in a drawn figure.

    Draw or save the figure first: the check measures the renderer's own
    extents, so it agrees with the saved image.
    """
    from core.viz.static.lint import lint_figure

    fig.canvas.draw()
    return [LintProblem(problem.code, problem.message) for problem in lint_figure(fig)]
