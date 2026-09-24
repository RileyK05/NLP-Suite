"""Lexical dispersion plot — where each word actually falls in the corpus.

The dispersion table answers "how evenly is this word spread?" with a number.
This answers the question the number cannot: *where*. Each occurrence is one
tick on a horizontal line running from the start of the corpus to the end, one
line per word, with document boundaries marked. Two words with identical DP
scores can have visibly different shapes — clustered at the opening, absent
from the middle, tailing off — and that shape is often the finding.

Classic reference: NLTK's ``dispersion_plot``. This version adds document
boundaries, a hover readout naming the document and sentence for each tick,
and per-word totals, because a tick you cannot trace back to a document is
not evidence.

Knowable degradation (the suite's viz rule): with plotly installed you get an
interactive chart; without it you get an inline SVG drawn from the same
offsets. The fallback is a real plot rather than a table of numbers, because
a dispersion plot is simple enough geometry to draw exactly, and a table of
token offsets communicates nothing at a glance. Either way the HTML is
self-contained apart from the plotly CDN.
"""

from __future__ import annotations

from dataclasses import dataclass
import html as html_lib

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["MAX_TERMS", "MAX_TICKS_PER_TERM", "dispersion_plot", "occurrence_offsets"]

MAX_TERMS = 40  # Beyond this the rows are too thin to read.
# Ticks actually drawn per word. A common word on a modest corpus has tens of
# thousands of occurrences, and one SVG element each produced a 17 MB file for
# a single lane at 200k tokens -- too large for the desktop to render. The
# plot is about 750 pixels wide, so beyond a couple of thousand ticks the
# extra elements land on pixels that are already inked and add nothing.
MAX_TICKS_PER_TERM = 2000

_SVG_WIDTH = 900
_SVG_ROW_HEIGHT = 26
_SVG_LEFT_GUTTER = 130
_SVG_TOP = 28
_SVG_BOTTOM = 34


def occurrence_offsets(
    frame: pd.DataFrame,
    terms: list[str],
    *,
    field: Col = Col.LEMMA,
) -> Result[pd.DataFrame]:
    """One row per occurrence: Term, Offset, Document, Sentence ID.

    ``Offset`` is the token's zero-based position in the corpus read in row
    order, which is the x-axis of the plot. Matching is casefolded, matching
    the dispersion table that this plot accompanies.
    """
    if not terms:
        return Result.failure(Diagnostic.error("DISPPLOT_NO_TERMS", "no terms given to plot"))
    if len(terms) > MAX_TERMS:
        return Result.failure(
            Diagnostic.error("DISPPLOT_TOO_MANY_TERMS", f"at most {MAX_TERMS} terms, got {len(terms)}")
        )
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(
            Diagnostic.error("DISPPLOT_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}")
        )
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)

    wanted = {term.casefold() for term in terms}
    columns = ["Term", "Offset", "Document", "Sentence ID"]
    if frame.empty:
        return Result.success(pd.DataFrame(columns=columns))

    has_document = Col.DOCUMENT.value in frame.columns
    rows: list[dict[str, object]] = []
    for offset, (_index, row) in enumerate(frame.iterrows()):
        token = str(row[field.value]).casefold()
        if token not in wanted:
            continue
        rows.append(
            {
                "Term": token,
                "Offset": offset,
                "Document": str(row[Col.DOCUMENT.value]) if has_document else "",
                "Sentence ID": row[Col.SENTENCE_ID.value],
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return Result.success(
            out,
            Diagnostic.warning("DISPPLOT_NO_MATCHES", "none of the requested terms occur in this corpus"),
        )
    return Result.success(out)


@dataclass(frozen=True, slots=True)
class _Plot:
    """Everything both renderers need, resolved once so they agree.

    ``drawn`` holds the offsets actually plotted (possibly thinned) while
    ``totals`` holds the true occurrence counts shown in the lane labels.
    Keeping them apart is what stops a thinned plot from also under-reporting.
    """

    lanes: list[str]
    drawn: dict[str, list[int]]
    totals: dict[str, int]
    total_tokens: int
    boundaries: list[tuple[int, str]]
    heading: str


def _thin(offsets: list[int]) -> list[int]:
    """Uniform subsample to at most MAX_TICKS_PER_TERM, order preserved.

    Every k-th occurrence, so the *shape* of the distribution along the corpus
    survives: a word clustered in the opening still looks clustered. The lane
    label keeps the true total, and a diagnostic names the thinned terms, so
    the picture is never silently a different picture.
    """
    if len(offsets) <= MAX_TICKS_PER_TERM:
        return offsets
    step = len(offsets) / MAX_TICKS_PER_TERM
    return [offsets[int(i * step)] for i in range(MAX_TICKS_PER_TERM)]


def _document_boundaries(frame: pd.DataFrame) -> list[tuple[int, str]]:
    """(offset, document name) for the first token of each document."""
    if Col.DOCUMENT_ID.value not in frame.columns:
        return []
    has_name = Col.DOCUMENT.value in frame.columns
    boundaries: list[tuple[int, str]] = []
    previous: object = object()
    for offset, (_index, row) in enumerate(frame.iterrows()):
        current = row[Col.DOCUMENT_ID.value]
        if current != previous:
            name = str(row[Col.DOCUMENT.value]) if has_name else str(current)
            boundaries.append((offset, name))
            previous = current
    return boundaries


def dispersion_plot(
    frame: pd.DataFrame,
    terms: list[str],
    *,
    field: Col = Col.LEMMA,
    title: str = "",
) -> Result[str]:
    """Self-contained HTML dispersion plot for *terms* over *frame*."""
    offsets = occurrence_offsets(frame, terms, field=field)
    if offsets.value is None:
        return Result[str](None, offsets.diagnostics)
    points = offsets.unwrap()
    total_tokens = len(frame)
    # Plot every requested term, including ones with no occurrences: an empty
    # row is the informative answer to "where does this word appear?".
    lanes: list[str] = []
    for term in terms:
        folded = term.casefold()
        if folded not in lanes:
            lanes.append(folded)
    boundaries = _document_boundaries(frame)
    heading = title or "Lexical dispersion"

    # Thin once, so both renderers draw exactly the same picture.
    drawn: dict[str, list[int]] = {}
    totals: dict[str, int] = {}
    thinned: list[str] = []
    for term in lanes:
        all_offsets = [int(value) for value in points.loc[points["Term"] == term, "Offset"]]
        totals[term] = len(all_offsets)
        drawn[term] = _thin(all_offsets)
        if len(drawn[term]) < len(all_offsets):
            thinned.append(term)
    extra: list[Diagnostic] = []
    if thinned:
        extra.append(
            Diagnostic.info(
                "DISPPLOT_THINNED",
                f"drew at most {MAX_TICKS_PER_TERM} evenly-spaced ticks for {', '.join(thinned)}; "
                "lane labels show the true totals",
                terms=thinned,
            )
        )

    plot = _Plot(
        lanes=lanes,
        drawn=drawn,
        totals=totals,
        total_tokens=total_tokens,
        boundaries=boundaries,
        heading=heading,
    )
    rendered = _plotly_html(points, plot)
    if rendered is not None:
        return Result.success(rendered, *offsets.diagnostics, *extra)
    return Result.success(
        _svg_html(plot),
        *offsets.diagnostics,
        *extra,
        Diagnostic.info(
            "DISPPLOT_SVG_FALLBACK",
            "plotly is not installed; rendered a static SVG plot from the same offsets",
        ),
    )


def _plotly_html(points: pd.DataFrame, plot: _Plot) -> str | None:
    """Interactive plot, or None when plotly is unavailable or fails."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    try:
        figure = go.Figure()
        for term in plot.lanes:
            keep = set(plot.drawn[term])
            subset = points[(points["Term"] == term) & (points["Offset"].isin(keep))]
            figure.add_trace(
                go.Scatter(
                    x=subset["Offset"],
                    y=[term] * len(subset),
                    mode="markers",
                    marker={"symbol": "line-ns-open", "size": 12, "line": {"width": 1.4}},
                    name=f"{term} ({plot.totals[term]})",
                    customdata=subset[["Document", "Sentence ID"]].to_numpy(),
                    hovertemplate=(
                        "%{y}<br>token %{x}<br>%{customdata[0]}<br>sentence %{customdata[1]}<extra></extra>"
                    ),
                )
            )
        for offset, _name in plot.boundaries[1:]:  # The first boundary is x = 0.
            figure.add_vline(x=offset, line_width=1, line_dash="dot", line_color="#bbb")
        figure.update_layout(
            title_text=plot.heading,
            template="plotly_white",
            xaxis_title=f"token position in corpus (0 to {plot.total_tokens})",
            yaxis_title="",
            showlegend=True,
            font_size=11,
            height=max(260, _SVG_TOP + _SVG_ROW_HEIGHT * len(plot.lanes) + 120),
        )
        figure.update_yaxes(categoryorder="array", categoryarray=list(reversed(plot.lanes)))
        figure.update_xaxes(range=[0, max(plot.total_tokens, 1)])
        html_str: str = figure.to_html(full_html=False, include_plotlyjs="cdn")
        return html_str
    except Exception:
        return None


def _svg_html(plot: _Plot) -> str:
    """Static SVG with the same geometry as the interactive chart."""
    span = max(plot.total_tokens, 1)
    plot_width = _SVG_WIDTH - _SVG_LEFT_GUTTER - 20
    height = _SVG_TOP + _SVG_ROW_HEIGHT * len(plot.lanes) + _SVG_BOTTOM

    def x_for(offset: float) -> float:
        return _SVG_LEFT_GUTTER + plot_width * offset / span

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_WIDTH}" height="{height}" '
        f'viewBox="0 0 {_SVG_WIDTH} {height}" role="img" '
        f'aria-label="{html_lib.escape(plot.heading)}">',
        f'<text x="12" y="18" font-family="sans-serif" font-size="13" font-weight="600">'
        f"{html_lib.escape(plot.heading)}</text>",
    ]
    for offset, name in plot.boundaries[1:]:
        x = x_for(offset)
        parts.append(
            f'<line x1="{x:.1f}" y1="{_SVG_TOP - 6}" x2="{x:.1f}" y2="{height - _SVG_BOTTOM + 6}" '
            f'stroke="#ccc" stroke-width="1" stroke-dasharray="2,3"><title>{html_lib.escape(name)}</title></line>'
        )
    for index, term in enumerate(plot.lanes):
        y = _SVG_TOP + _SVG_ROW_HEIGHT * index + _SVG_ROW_HEIGHT / 2
        parts.append(
            f'<text x="{_SVG_LEFT_GUTTER - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="11">{html_lib.escape(term)} ({plot.totals[term]})</text>'
        )
        parts.append(
            f'<line x1="{_SVG_LEFT_GUTTER}" y1="{y:.1f}" x2="{_SVG_LEFT_GUTTER + plot_width}" '
            f'y2="{y:.1f}" stroke="#eee" stroke-width="1"/>'
        )
        for offset in plot.drawn[term]:
            x = x_for(float(offset))
            parts.append(
                f'<line x1="{x:.1f}" y1="{y - 8:.1f}" x2="{x:.1f}" y2="{y + 8:.1f}" '
                f'stroke="#2b6cb0" stroke-width="1.4"/>'
            )
    parts.append(
        f'<text x="{_SVG_LEFT_GUTTER}" y="{height - 10}" font-family="sans-serif" font-size="10" '
        f'fill="#666">token position in corpus (0 to {plot.total_tokens})</text>'
    )
    parts.append("</svg>")
    return "".join(parts)
