"""A figure with the house chrome: left title, gray subtitle, provenance footer.

``canvas(...)`` opens a figure; ``.rows([...])`` and ``row.cols([...])`` place
axes in it; ``.done()`` draws the title, subtitle and footer around whatever
was drawn and returns the :class:`~matplotlib.figure.Figure`, ready for
``nlp.figure(fig)``. The chrome is the app's (``core.viz.static.add_chrome``),
so a kit figure and a published panel share one look.

Provenance comes from a :class:`~core.viz.panelspec.Source`: the artifact a
run's table came from and its hash, the settings that produced it. A script
passes one built from a run; a showcase passes the run's own. Without a
source, the footer says so rather than staying silent (the same rule the
panels keep).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.viz.panelspec import Provenance, Source

__all__ = ["Canvas", "Row", "canvas"]

#: Figure size when the caller does not say (inches). Between the app's
#: panels (a page wide) and the reference figures (a poster wide).
DEFAULT_SIZE: tuple[float, float] = (15.0, 12.0)


@dataclass
class Row:
    """One horizontal band of a canvas, split into columns on request."""

    figure: Any
    gridspec: Any
    axes: list[Any] = field(default_factory=list)

    def cols(self, ratios: Sequence[float], *, wspace: float = 0.24) -> Row:
        """Split this row into columns of ``ratios`` widths; return the row."""
        span = self.gridspec.subgridspec(1, len(ratios), width_ratios=list(ratios), wspace=wspace)
        self.axes = [self.figure.add_subplot(span[0, index]) for index in range(len(ratios))]
        return self

    def axis(self, index: int = 0) -> Any:
        """The axis for column *index* (call :meth:`cols` first)."""
        if not self.axes:
            self.cols([1.0])
        return self.axes[index]


@dataclass
class Canvas:
    """A figure being built: layout now, chrome at :meth:`done`."""

    title: str
    subtitle: str = ""
    source: Source | None = None
    footer: str = ""
    size: tuple[float, float] = DEFAULT_SIZE
    library: str = ""
    _rows: list[Row] = field(default_factory=list)
    _figure: Any = None

    def __post_init__(self) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from core.viz.static.style import house_style

        with house_style():
            self._figure = Figure(figsize=self.size, layout="constrained")
            FigureCanvasAgg(self._figure)

    @property
    def figure(self) -> Any:
        """The matplotlib figure (for pieces that need it directly)."""
        return self._figure

    def rows(self, ratios: Sequence[float], *, hspace: float = 0.3) -> Canvas:
        """Lay the canvas out as horizontal bands of ``ratios`` heights."""
        self._rows = [
            Row(
                self._figure, self._figure.add_gridspec(len(ratios), 1, height_ratios=list(ratios), hspace=hspace)[i, 0]
            )
            for i in range(len(ratios))
        ]
        return self

    def row(self, index: int) -> Row:
        """The :class:`Row` at *index* (call :meth:`rows` first)."""
        if not self._rows:
            self.rows([1.0])
        return self._rows[index]

    def axis(self, row: int = 0, col: int = 0) -> Any:
        """The axis at (*row*, *col*), creating a single column if none was set."""
        target = self.row(row)
        if not target.axes:
            target.cols([1.0])
        return target.axes[col]

    def _caption(self) -> str:
        if self.footer:
            return self.footer
        if self.source is None:
            return "source not recorded"
        provenance = Provenance(
            tool="",
            panel=self.title,
            source=self.source.path,
            source_sha256=self.source.sha256,
            settings=dict(self.source.settings),
            library=self.library or self.source.library,
        )
        return " · ".join(part for part in provenance.caption().split(" · ") if part and part != " · ")

    def done(self) -> Any:
        """Draw the chrome and return the finished figure."""
        from core.viz.static import add_chrome
        from core.viz.static.style import house_style

        with house_style():
            self._figure.canvas.draw()
            add_chrome(self._figure, self.title, self.subtitle, [], self._caption())
        return self._figure


def canvas(
    title: str,
    subtitle: str = "",
    *,
    source: Source | None = None,
    footer: str = "",
    size: tuple[float, float] = DEFAULT_SIZE,
    library: str = "",
) -> Canvas:
    """A figure with the house chrome and the provenance footer.

    ``title`` is the finding or the question, not the tool's name.
    ``source`` is the artifact a run's numbers came from; ``footer`` is a
    literal provenance line for a script's own table (which has no envelope).
    """
    return Canvas(title=title, subtitle=subtitle, source=source, footer=footer, size=size, library=library)
