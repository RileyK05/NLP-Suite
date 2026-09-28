"""``nlp.viz``: the figure kit a script draws with, built on :mod:`core.viz.kit`.

The pieces are the plan's (``docs/internal/SHOWCASE_FIGURES_PLAN.md`` section 3). They
draw onto an axis the caller made, take plain tables, and keep the suite's
figure rules (the title is the finding, direct labels over legends, values
where they are read, a background group in grey). A script uses them for the
figures the suite does not ship; the suite's own showcase figures are built
with the same pieces, so each showcase is also a worked example.

Two helpers take a suite object rather than a raw table, so a script does not
re-derive what the library already knows:

* :func:`time_axis` takes an ``nlp.Corpus`` and shades a document detail.
* :func:`events` takes a list of ``(year, label)`` the project defines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

__all__ = [
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


def canvas(
    title: str,
    subtitle: str = "",
    *,
    source: str = "",
    sha256: str = "",
    settings: Mapping[str, Any] | None = None,
    footer: str = "",
    size: tuple[float, float] | None = None,
) -> Any:
    """A figure with the house chrome: left title, gray subtitle, provenance footer.

    Lay it out with ``.rows([...])`` and ``row.cols([...])``, draw into the
    axes, then ``.done()`` and hand the figure to :func:`nlpsuite.figure`.
    ``source``/``sha256``/``settings`` fill the provenance footer the way a
    run's envelope does; ``footer`` is a literal line for a script's own table.
    """
    from core.viz.kit import canvas as _canvas
    from core.viz.panelspec import Source

    record = Source(path=source, sha256=sha256, settings=dict(settings or {})) if source else None
    kwargs: dict[str, Any] = {"source": record, "footer": footer}
    if size is not None:
        kwargs["size"] = size
    return _canvas(title, subtitle, **kwargs)


def palette(keys: Any, *, background: Any | None = None, colors: Mapping[Any, str] | None = None) -> dict[Any, str]:
    """One stable colour per key, by declared order; ``background`` is drawn grey."""
    from core.viz.kit import palette as _palette

    return _palette(keys, background=background, colors=dict(colors or {}))


def time_axis(
    ax: Any,
    corpus: Any,
    *,
    x: str = "Year",
    band: str | None = None,
    label_bands: bool = True,
) -> None:
    """Year ticks, and optionally a document detail (Speaker, Party) as bands.

    ``corpus`` is an ``nlp.Corpus``; ``band`` names one of its detail columns,
    whose contiguous runs are shaded so a reader sees which documents share a
    speaker or a term.
    """
    from core.viz.kit import time_axis as _time_axis

    documents: pd.DataFrame = corpus.documents
    _time_axis(ax, documents, x=x, band=band, label_bands=label_bands)


def events(ax: Any, points: Sequence[tuple[float, str]], *, y: float = 1.012) -> None:
    """Vertical stoplines at dated events, labels kept clear of each other."""
    from core.viz.kit import events as _events

    _events(ax, points, y=y)


def stream(
    ax: Any,
    shares: pd.DataFrame,
    *,
    labels: Mapping[Any, str] | None = None,
    colors: Mapping[Any, str] | None = None,
    background: Any | None = None,
    event_at: Sequence[float] = (),
    order: Sequence[Any] | None = None,
) -> dict[Any, str]:
    """Stacked shares over time with a direct label on each band."""
    from core.viz.kit import stream as _stream

    return _stream(
        ax,
        shares,
        labels=dict(labels or {}),
        colors=dict(colors or {}),
        background=background,
        event_at=event_at,
        order=order,
    )


def mixture_bars(
    ax: Any,
    shares: pd.DataFrame,
    *,
    row_labels: Sequence[str] | None = None,
    colors: Mapping[Any, str] | None = None,
    background: Any | None = None,
    order: Sequence[Any] | None = None,
    threshold: float = 0.12,
) -> dict[Any, str]:
    """One document per bar, split into its shares, values printed inside."""
    from core.viz.kit import mixture_bars as _mixture_bars

    return _mixture_bars(
        ax,
        shares,
        row_labels=row_labels,
        colors=dict(colors or {}),
        background=background,
        order=order,
        threshold=threshold,
    )


def bars(
    ax: Any,
    values: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    colors: Any = None,
    fmt: str | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Horizontal bars with the value at the end and full-length labels."""
    from core.viz.kit import bars as _bars

    _bars(ax, values, labels=labels, colors=colors, fmt=fmt, xlim=xlim)


def dots(
    ax: Any,
    frame: pd.DataFrame,
    *,
    by: str,
    value: str,
    colors: Mapping[Any, str] | None = None,
    order: Sequence[Any] | None = None,
) -> None:
    """A strip of points per group, with a median line and its value labelled."""
    from core.viz.kit import dots as _dots

    _dots(ax, frame, by=by, value=value, colors=dict(colors or {}), order=order)


def word_grid(
    ax: Any,
    words: Sequence[str],
    *,
    shade: Sequence[float] | None = None,
    columns: int = 4,
    fmt: str | None = None,
) -> None:
    """Ranked words in a grid, shaded by a measure (corpus frequency)."""
    from core.viz.kit import word_grid as _word_grid

    _word_grid(ax, words, shade=shade, columns=columns, fmt=fmt)


def tiles(
    ax: Any,
    fits: Sequence[tuple[str, Sequence[int]]],
    *,
    references: Sequence[tuple[str, Sequence[int]]] = (),
    colors: Sequence[str] | None = None,
    stoplines: Sequence[tuple[float, str]] = (),
    tick_labels: Sequence[str] | None = None,
) -> None:
    """Rows of topic assignments, relabelled by first appearance, with stoplines."""
    from core.viz.kit import tiles as _tiles

    _tiles(ax, fits, references=references, colors=colors, stoplines=stoplines, tick_labels=tick_labels)


def lint(fig: Any) -> list[Any]:
    """Overlapping text and negative-zero ticks in a drawn figure, as messages."""
    from core.viz.kit import lint as _lint

    return _lint(fig)
