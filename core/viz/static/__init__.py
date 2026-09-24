"""Publication figures: a prepared panel drawn with matplotlib and seaborn.

The second renderer for :class:`~core.viz.panelspec.PreparedPanel`, beside
the app's interactive SVG (``desktop/src/PanelCanvas.tsx``) and the Plotly
export (``core/viz/panel_plotters.py``). Same input, so what a figure claims
-- its marks, evidence, caption and limits -- is decided once by the builder;
this renderer's job is to show more of it than an interactive chart can
(spread bands, violins, clustered matrices, named extremes) and to be fit to
publish: nothing overprinted, nothing cut, provenance under every figure.

    render_static(prepared, "png") -> Result[bytes]

Pure: bytes out, nothing written (the OutputWriter is the only writer, R3).
Imports of matplotlib and seaborn are deferred to the first call, so the
live bench's cold start does not pay for them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import io
from typing import Any, Literal
import warnings

from core.result import Diagnostic, Result
from core.viz.panelspec import PreparedPanel

__all__ = ["LIBRARY", "STATIC_FORMATS", "StaticFormat", "add_chrome", "draw_static", "render_static"]

StaticFormat = Literal["png", "svg", "pdf"]
STATIC_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")
#: The library line a published caption carries (Provenance.library).
LIBRARY = "NLP Suite publication renderer (matplotlib + seaborn)"

_TITLE_PT = 14.0
_SUBTITLE_PT = 9.5
_CAPTION_PT = 6.8


def draw_static(prepared: PreparedPanel, *, notes: bool = False, dpi: int = 100) -> tuple[Any, list[str], list[Any]]:
    """The drawn figure, its caption lines and its drawn problems (for tests).

    The figure has a frozen layout and its chrome; callers that only need
    bytes use :func:`render_static`.
    """
    from core.viz.static.lint import lint_figure
    from core.viz.static.shapes import DRAWERS
    from core.viz.static.style import house_style

    drawer = DRAWERS.get(prepared.shape)
    if drawer is None:
        raise ValueError(f"the publication renderer cannot draw shape {prepared.shape!r}")
    prepared = _printable(prepared)
    with house_style(), warnings.catch_warnings():
        # seaborn 0.13's boxplot still passes ``vert``, which matplotlib 3.10
        # deprecates; the figure is unaffected, and the message is not ours.
        warnings.filterwarnings("ignore", message=r"vert: bool will be deprecated", category=PendingDeprecationWarning)
        fig, extra = drawer(prepared)
        # Measured and saved at one resolution: a tight save at a dpi other
        # than the drawing's shifted every figure-level text (the title
        # landed inside the plot at 110 dpi and was right at 100).
        fig.set_dpi(dpi)
        fig.canvas.draw()
        lines = _chrome(fig, prepared, extra, include_notes=notes)
        problems = lint_figure(fig)
    return fig, lines, problems


def render_static(
    prepared: PreparedPanel,
    fmt: str = "png",
    *,
    dpi: int = 200,
    notes: bool = False,
) -> Result[bytes]:
    """PNG, SVG or PDF bytes of a publication figure, or diagnostics.

    ``notes`` adds the panel's "what this can and cannot show" list under the
    caption. Overlapping text is reported as a warning, not a failure: the
    figure is still worth having, and the audit script lists every one.
    """
    if fmt not in STATIC_FORMATS:
        return Result.failure(
            Diagnostic.error("STATIC_BAD_FORMAT", f"format must be one of {STATIC_FORMATS}, got {fmt!r}")
        )
    try:
        fig, _, problems = draw_static(prepared, notes=notes, dpi=dpi)
    except ImportError as exc:
        return Result.failure(
            Diagnostic.error(
                "STATIC_UNAVAILABLE",
                f"publication figures need matplotlib and seaborn: {exc}. "
                'Install them with: pip install "nlp-suite-ng[figures]"',
            )
        )
    except Exception as exc:  # a drawing bug must not take the caller down (R4)
        return Result.failure(
            Diagnostic.error(
                "STATIC_RENDER_FAILED",
                f"{prepared.panel}: the publication renderer failed: {type(exc).__name__}: {exc}",
                panel=prepared.panel,
            )
        )
    from core.viz.static.style import house_style

    buffer = io.BytesIO()
    metadata: dict[str, Any] = {"svg": {"Date": None}, "pdf": {"CreationDate": None, "ModDate": None}}.get(fmt, {})
    with house_style():
        fig.savefig(buffer, format=fmt, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.3, metadata=metadata)
    notices = [
        Diagnostic.warning(f"STATIC_{problem.code}", f"{prepared.panel}: {problem.message}", panel=prepared.panel)
        for problem in problems
    ]
    return Result.success(buffer.getvalue(), *notices)


def _clean(text: str) -> str:
    """Text as a font can print it: a C1 control character (U+0080-U+009F)
    in a label is a cp1252 byte decoded as Latin-1 -- the real corpus had
    "\\x97" for an em dash -- so it is read back as cp1252; anything else
    unprintable is dropped."""
    out = []
    for char in text:
        if "\x80" <= char <= "\x9f":
            try:
                out.append(bytes([ord(char)]).decode("cp1252"))
            except UnicodeDecodeError:
                continue
        elif char.isprintable() or char == "\n":
            out.append(char)
    return "".join(out)


def _printable(prepared: PreparedPanel) -> PreparedPanel:
    """The panel with every label and category run through :func:`_clean`."""
    return replace(
        prepared,
        title=_clean(prepared.title),
        subtitle=_clean(prepared.subtitle),
        marks=tuple(replace(mark, label=_clean(mark.label)) for mark in prepared.marks),
        x_categories=tuple(_clean(c) for c in prepared.x_categories),
        y_categories=tuple(_clean(c) for c in prepared.y_categories),
        facets=prepared.facets,
    )


def _chrome(fig: Any, prepared: PreparedPanel, extra: Sequence[str], *, include_notes: bool) -> list[str]:
    """A panel's title, subtitle, caption lines and provenance, around its drawing."""
    lines = [*extra]
    if include_notes:
        lines.extend(prepared.notes)
    # The provenance line, naming this renderer: the same figure drawn by
    # the app says "drawn by NLP Suite desktop (SVG)".
    caption = replace(prepared.provenance, library=LIBRARY).caption()
    add_chrome(fig, prepared.title, prepared.subtitle, lines, caption)
    return lines


def add_chrome(fig: Any, title: str, subtitle: str, lines: Sequence[str], caption: str) -> None:
    """Title and subtitle above the drawing, caption lines and provenance below.

    Placed from the drawing's own measured extent, so they never sit on a
    tick label or a legend however the drawing was laid out; the tight
    bounding box at save time takes them in.
    """
    from core.viz.static.style import INK, MUTED
    from core.viz.static.text import wrap

    renderer = fig.canvas.get_renderer()
    box = fig.get_tightbbox(renderer)  # inches
    width_in, height_in = fig.get_size_inches()
    # Figure-fraction coordinates, not inches: a tight save shifts the
    # figure's own transform to crop it, and text placed in inches stays
    # behind -- the title landed inside the plot.
    frame = fig.transFigure

    def fx(inches: float) -> float:
        return float(inches / width_in)

    def fy(inches: float) -> float:
        return float(inches / height_in)

    left = fx(box.x0)
    chars = max(60, int(box.width * 72 / (_CAPTION_PT * 0.55)))
    title_chars = max(40, int(box.width * 72 / (_TITLE_PT * 0.58)))
    top = box.y1 + 0.12
    if subtitle:
        wrapped = wrap(_clean(subtitle), max(60, int(box.width * 72 / (_SUBTITLE_PT * 0.55))))
        text = fig.text(
            left, fy(top), wrapped, transform=frame, ha="left", va="bottom", fontsize=_SUBTITLE_PT, color=MUTED
        )
        top = text.get_window_extent(renderer).y1 / fig.dpi + 0.06
    fig.text(
        left,
        fy(top),
        wrap(_clean(title), title_chars),
        transform=frame,
        ha="left",
        va="bottom",
        fontsize=_TITLE_PT,
        fontweight="bold",
        color=INK,
    )
    body = [wrap(_clean(line), chars) for line in lines]
    body.append(wrap(_clean(caption), chars))
    fig.text(
        left,
        fy(box.y0 - 0.14),
        "\n".join(body),
        transform=frame,
        ha="left",
        va="top",
        fontsize=_CAPTION_PT,
        color=MUTED,
        linespacing=1.35,
    )
