"""Plotly renderers for prepared charts (FR-6.5): six kinds, HTML + image bytes.

Every renderer takes a :class:`~core.viz.chartspec.PreparedChart` (tidy data
from :func:`core.viz.chartspec.prepare_chart_data`) plus a
:class:`~core.viz.chartspec.ChartSpec`, and returns ``Result[str]`` HTML.
:func:`chart_image_bytes` returns ``Result[bytes]`` (PNG/SVG/PDF via kaleido)
for the caller to persist through ``OutputWriter.write_bytes`` — no renderer
touches the filesystem (R3: the writer stays the only writer).

Axis handling (dtype-honoring):

* **Numeric/datetime x** stays on a linear/date axis with raw values —
  never converted to strings, never collapsed; ticks remain numeric.
* **Categorical x** renders through ``tickvals``/``ticktext`` with wrapped
  labels AND an explicitly synchronized ``categoryorder``/``categoryarray``:
  Plotly inserts categories in trace-traversal order (so a sparse grouped bar
  can land B under the tick labelled C), which would desynchronize the tick
  mapping from the drawn positions. Pinning ``categoryarray`` to the tick
  order makes the rendered category axis identical across every trace —
  verified against the actual figure, not just ticktext. The raw values in
  the prepared frame are NOT mutated — wrapping/ordering is a rendering
  concern only.
* **Effective labels win.** Renderers always label axes with
  ``PreparedChart.y_label``/``x_label`` (which preparation already set to
  ``Percent``/``Share``/rate labels — and, for histograms/heatmaps, to the
  kind-correct axis names), not the raw column name — overrides in the spec
  win over everything.
* **Horizontal bars swap the axes**: the categorical axis is y (receives
  categorical   ticktext), the numeric measure axis is x (numeric ticks, measure
  label). Only the categorical axis gets ticktext substitution. Category
  ticks are upright (angle 0) by default; an explicitly supplied
  ``x_tick_angle`` — including -30 — is honored verbatim (``None`` means
  "unspecified", not -30).
* **Histogram**: x axis is the numeric value (bin edges, no tick
  substitution, labeled with the value column), y is Count/Percent/Share
  from preparation.
* **Heatmap**: rendered with ``go.Heatmap`` (NOT imshow) so irregular
  numeric/datetime coordinates keep their true values on the axes —
  ``px.imshow`` assumes regular integer grid coordinates and would corrupt
  them. x on the column axis, group on the row axis, y as the numeric cell
  value; missing (x, group) combinations stay missing (blank cells), never
  zero-filled. Ordering is dtype-aware: numeric/datetime columns sort by
  value (years 1, 2, 10 never become 1, 10, 2), categorical by text; group
  rows sort by text — and the axis coordinates carry the actual values, so
  the matrix can never be resampled onto unrelated positions.
* **Explicit tick angle is honored for any number of categories** (no
  hidden threshold override; the CLI default is -30, spec-controlled).

Comparison layout:

* **Grouped bars default to side-by-side** (``barmode="group"``) — stacking
  means is a known chart crime; ``ChartSpec.bar_mode`` offers the explicit
  documented ``"stack"`` (and ``"relative"``) opt-in.
* **Grouped histograms draw with partial opacity** in overlay mode so both
  distributions stay readable.
* **Group colors are deterministic** within a chart: Okabe-Ito assigned by
  sorted group name. Across DIFFERENT charts the palette assignment depends
  on the groups present in that chart — colors are NOT claimed stable across
  different category subsets.

Styling: ``plotly_white`` template, title + subtitle (smaller ``<sup>`` line,
extra top margin), explicit axis titles from effective labels, 900x500
default, **offline self-contained HTML by default** (plotly.js inlined;
``--cdn`` opts into the CDN-script variant).

Missing plotly degrades to an honest HTML data table (same numbers, same
order) with a WARNING diagnostic naming the degradation — the knowable-
fallback contract of ``bar_chart_html``.

Image export (kaleido v1): a per-call browser override goes through
``kaleido.calc_fig_sync(..., kopts={"path": ...})``, which raises
``ChromeNotFoundError`` for a nonexistent executable — the invalid-override
case fails loudly (kaleido v0's module-attribute ``chrome`` assignment was a
silent no-op and is NOT used). kaleido v1 requires plotly >= 6.1.1 (it warns
below 6.1.1 and image export breaks); the ``plotly-image`` extra pins
``kaleido>=1.0, plotly>=6.1.1`` so the supported pair installs together.
Without an override kaleido uses ``BROWSER_PATH``/its auto-detection; that
runtime configuration is documented in ``docs/viz-charts.md``. Rendering is
pure: tests call :func:`build_figure` without filesystem or network.
"""

from __future__ import annotations

import html as html_lib
import math
from typing import TYPE_CHECKING, Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.chartspec import ChartSpec, PreparedChart, wrap_label

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

__all__ = ["OKABE_ITO", "chart_html", "chart_image_bytes", "render_chart"]

# Largest bubble diameter in pixels; past this they overlap into a blob.
_BUBBLE_SIZE_MAX = 40

OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#000000",  # black
)

_FALLBACK_FIX = 'pip install "nlp-suite-ng[plotly]"'
_IMAGE_FIX = 'pip install "nlp-suite-ng[plotly-image]"'

# Vertical-axis rotation applied when x_tick_angle is unspecified (None);
# horizontal axes render upright by default. Explicit angles always win.
_DEFAULT_TICK_ANGLE = -30


def _group_color_map(groups: list[str]) -> dict[str, str]:
    """Deterministic Okabe-Ito assignment within a chart: groups sorted, palette cycled."""
    ordered = sorted(set(groups))
    return {g: OKABE_ITO[i % len(OKABE_ITO)] for i, g in enumerate(ordered)}


def _is_temporal_or_numeric(series: pd.Series) -> bool:
    """Whether an axis series must keep its raw numeric/datetime axis."""
    return bool(pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series))


def _categorical_tick_mapping(frame: pd.DataFrame, column: str, spec: ChartSpec) -> tuple[list[Any], list[str]] | None:
    """Tick positions + wrapped labels for a categorical axis, else None.

    Positions are first-appearance order (the order preparation produced, which
    renderers draw in) so group subsets and interleaving stay aligned across
    traces. Numeric/datetime axes return None (raw axis, numeric ticks).
    """
    if column not in frame.columns:
        return None
    series = frame[column]
    if _is_temporal_or_numeric(series):
        return None
    ordered = list(dict.fromkeys(str(v) for v in series))
    if not ordered:
        return None
    return [i for i in range(len(ordered))], [wrap_label(v, spec.wrap_labels) for v in ordered]


def _pin_category_order(fig: Figure, axis: str, ticks: tuple[list[Any], list[str]] | None) -> None:
    """Pin a categorical axis to its tick order (blocker 2).

    Plotly places categories by TRACE TRAVERSAL: a sparse grouped bar whose
    second trace starts at 'B' inserts B after A/C/E, so positional ticktext
    A..F mislabels every later bar. Setting ``categoryorder="array"`` +
    ``categoryarray`` to the tick order synchronizes the axis with the tick
    mapping; numeric ``tickvals`` on a linear axis keep positions 0..n-1
    fixed, so only the traversal order needs pinning. No-op when the axis
    carries no categorical ticks.
    """
    if ticks is None:
        return
    _vals, ticktext = ticks
    categories = [str(t).replace("<br>", "") for t in ticktext]
    fig.update_layout(**{f"{axis}axis_categoryorder": "array", f"{axis}axis_categoryarray": categories})


def _axis_label(prepared_label: str, override: str) -> str:
    """Explicit override wins; otherwise the effective prepared label."""
    return override or prepared_label


def _title_text(spec: ChartSpec) -> str:
    """Title (defaulting to '<y> by <x>') with an optional smaller subtitle."""
    base = spec.title or f"{spec.y} by {spec.x}"
    if spec.subtitle:
        return f"{base}<br><sup>{spec.subtitle}</sup>"
    return base


def _apply_ticks(
    fig: Figure,
    axis: str,
    ticks: tuple[list[Any], list[str]] | None,
    angle: int,
    title: str,
) -> None:
    """Set tickmode/tickvals/ticktext/angle/title on one axis (array mode)."""
    if ticks is None:
        fig.update_layout(**{f"{axis}axis_title": title})
        return
    tickvals, ticktext = ticks
    fig.update_layout(
        **{
            f"{axis}axis": {
                "tickmode": "array",
                "tickvals": tickvals,
                "ticktext": ticktext,
                "tickangle": angle,
                "title": title,
            }
        }
    )


def _histogram_opacity(fig: Figure, frame: pd.DataFrame, spec: ChartSpec) -> None:
    """Partial opacity + explicit bin widths so grouped histograms stay readable."""
    widths = frame["bin_width"].astype(float)
    max_width = float(widths.max()) if len(widths) else 1.0
    fig.update_traces(width=max_width * 0.95)
    if "group" in frame.columns and frame["group"].nunique() > 1:
        fig.update_traces(opacity=0.65)


def _heatmap_matrix(frame: pd.DataFrame, spec: ChartSpec) -> tuple[list[Any], list[str], list[list[float | None]]]:
    """Dtype-aware heatmap pivot for ``go.Heatmap`` (blocker 3).

    Returns ``(x_coords, group_rows, z)``: numeric/datetime columns keep their
    raw values sorted by value (years 1, 2, 10 never collapse to text order
    1, 10, 2 — string sorting the coordinates corrupts the axis); group rows
    sort by text. Missing (x, group) cells stay ``None`` (rendered gaps,
    never zero-filled). No imshow resampling: coordinates go straight onto
    the rendered axes.
    """
    x_is_special = _is_temporal_or_numeric(frame["x"])
    x_coords = sorted(frame["x"].unique().tolist()) if x_is_special else sorted({str(v) for v in frame["x"]})
    groups = sorted(frame["group"].astype(str).unique().tolist())
    cell: dict[tuple[Any, str], float] = {}
    for row in frame.itertuples(index=False):
        key = (row.x, str(row.group))
        cell[key] = float(row.y)
    z: list[list[float | None]] = []
    for g in groups:
        z.append([cell.get((x, g)) for x in x_coords])
    return x_coords, groups, z


def _heatmap_figure(frame: pd.DataFrame, spec: ChartSpec) -> Any:
    """Render the prepared matrix with go.Heatmap (dtype-true coordinates)."""
    from plotly.graph_objects import Heatmap

    x_coords, groups, z = _heatmap_matrix(frame, spec)
    fig = _new_figure()
    fig.add_trace(
        Heatmap(
            x=x_coords,
            y=groups,
            z=z,
            colorbar={"title": {"text": spec.y}},
            hovertemplate="x: %{x}<br>group: %{y}<br>value: %{z}<extra></extra>",
        )
    )
    fig.update_layout(yaxis_title=str(spec.group or "(all)"))
    return fig


def _waffle_figure(frame: pd.DataFrame, spec: ChartSpec) -> Any:
    """100 squares (10x10) shaded by category share, deterministic order.

    Category order is the prepared frame's x order (text-sorted). Squares
    fill in reading order; remainder squares go to the largest shares first.
    """
    import numpy as np

    total = float(frame["y"].sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("waffle needs a positive finite total")
    shares = frame["y"].astype(float) / total * 100.0
    counts = np.floor(shares.to_numpy()).astype(int)
    remainder = 100 - int(counts.sum())
    order = np.argsort(-shares.to_numpy(), kind="stable")
    for index in order:
        if remainder <= 0:
            break
        counts[index] += 1
        remainder -= 1
    squares: list[tuple[int, int, str, float]] = []
    cursor = 0
    for label, count, share in zip(frame["x"].astype(str).tolist(), counts.tolist(), shares.tolist(), strict=True):
        for _ in range(int(count)):
            row, col = divmod(cursor, 10)
            squares.append((row, col, label, share))
            cursor += 1
    fig = _new_figure()
    palette = list(OKABE_ITO)
    for index, label in enumerate(frame["x"].astype(str).tolist()):
        cells = [(r, c) for r, c, name, _share in squares if name == label]
        if not cells:
            continue
        share = float(shares[frame["x"].astype(str) == label].iloc[0])
        fig.add_trace(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [c for c, _ in cells],
                "y": [r for r, _ in cells],
                "name": label,
                "text": [f"{label}: {share:.1f}% ({int(count)} squares)"] * len(cells),
                "marker": {"symbol": "square", "size": 26, "color": palette[index % len(palette)]},
                "hoverinfo": "text",
            }
        )
    fig.update_layout(
        xaxis={"visible": False, "range": (-0.5, 9.5)},
        yaxis={"visible": False, "range": (9.5, -0.5), "scaleanchor": "x", "scaleratio": 1},
        showlegend=True,
        legend_title_text=str(spec.x),
    )
    return fig


def _calendar_figure(frame: pd.DataFrame, spec: ChartSpec) -> Any:
    """One row per month, seven weekday columns; cells sum the daily values."""
    from plotly.graph_objects import Heatmap

    dates = pd.Series(pd.to_datetime(frame["x"]), name="date")
    y_values = frame["y"].astype(float).to_numpy()
    z: list[list[float | None]] = []
    y_labels: list[str] = []
    months = sorted({(d.year, d.month) for d in dates})
    for year, month in months:
        mask = (dates.dt.year == year) & (dates.dt.month == month)
        row: list[float | None] = [None] * 7
        for day_value, value in zip(dates[mask], y_values[mask.to_numpy()], strict=True):
            weekday = int(day_value.weekday())
            row[weekday] = (row[weekday] or 0.0) + float(value)
        z.append(row)
        y_labels.append(f"{year}-{month:02d}")
    fig = _new_figure()
    fig.add_trace(
        Heatmap(
            x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            y=y_labels,
            z=z,
            hovertemplate="week: %{y}<br>day: %{x}<br>value: %{z}<extra></extra>",
            colorbar={"title": {"text": str(spec.y)}},
        )
    )
    fig.update_layout(yaxis_title="Month")
    return fig


def _new_figure() -> Any:
    from plotly.graph_objects import Figure as Fig

    return Fig()


def _dtype_ordered_values(series: pd.Series) -> list[Any]:
    """Numeric/datetime ascending, else text-sorted (same rule as chartspec)."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return list(series.drop_duplicates().sort_values())
    return list(series.astype(str).drop_duplicates().sort_values())


def build_figure(prepared: PreparedChart, spec: ChartSpec) -> Result[Figure]:
    """Construct the plotly Figure for the prepared chart (no HTML, no disk).

    The pure construction half of rendering — tests call this directly to
    assert on traces/axes without filesystem or network. ImportError from a
    missing plotly propagates to the caller.
    """
    import plotly.express as px
    from plotly.graph_objects import Figure as Fig

    frame = prepared.data
    fig: Fig
    has_groups = frame["group"].nunique() > 1 if "group" in frame.columns else False
    color_arg = "group" if has_groups else None

    if spec.kind == "bar":
        # Grouped comparisons default to side-by-side; stacking is the
        # explicit, documented opt-in (stacking means something different).
        barmode = spec.bar_mode if spec.bar_mode in ("group", "stack", "relative") else "group"
        if spec.horizontal:
            fig = px.bar(
                frame,
                x="y",
                y="x",
                color=color_arg,
                orientation="h",
                barmode=barmode,
                color_discrete_sequence=list(OKABE_ITO),
            )
        else:
            fig = px.bar(
                frame,
                x="x",
                y="y",
                color=color_arg,
                barmode=barmode,
                color_discrete_sequence=list(OKABE_ITO),
            )
    elif spec.kind == "line":
        fig = px.line(frame, x="x", y="y", color=color_arg, markers=True, color_discrete_sequence=list(OKABE_ITO))
    elif spec.kind == "scatter":
        fig = px.scatter(frame, x="x", y="y", color=color_arg, color_discrete_sequence=list(OKABE_ITO))
    elif spec.kind == "bubble":
        # The legacy Excel bubble chart sizes each point by its value, with no
        # third column; the HTML form matches it so the same spec draws the
        # same picture whichever format is asked for.
        fig = px.scatter(
            frame,
            x="x",
            y="y",
            size=frame["y"].abs(),
            size_max=_BUBBLE_SIZE_MAX,
            color=color_arg,
            color_discrete_sequence=list(OKABE_ITO),
        )
    elif spec.kind == "box":
        fig = px.box(frame, x="x", y="y", color=color_arg, color_discrete_sequence=list(OKABE_ITO))
    elif spec.kind == "violin":
        fig = px.violin(
            frame, x="x", y="y", color=color_arg, box=True, points="all", color_discrete_sequence=list(OKABE_ITO)
        )
    elif spec.kind == "pie":
        fig = px.pie(frame, names="x", values="y", color_discrete_sequence=list(OKABE_ITO))
    elif spec.kind == "sunburst":
        fig = px.sunburst(frame, path=["x", "group"], values="y", color_discrete_sequence=list(OKABE_ITO))
    elif spec.kind == "treemap":
        fig = px.treemap(frame, path=["x", "group"], values="y", color_discrete_sequence=list(OKABE_ITO))
    elif spec.kind == "radar":
        fig = px.line_polar(
            frame, r="y", theta="x", color=color_arg, line_close=True, color_discrete_sequence=list(OKABE_ITO)
        )
    elif spec.kind == "waffle":
        fig = _waffle_figure(frame, spec)
    elif spec.kind == "calendar":
        fig = _calendar_figure(frame, spec)
    elif spec.kind == "histogram":
        fig = px.bar(
            frame,
            x="bin_center",
            y="count",
            color=color_arg,
            barmode="overlay",
            color_discrete_sequence=list(OKABE_ITO),
        )
    else:  # heatmap: x on columns, group on rows, y as the numeric cell
        fig = _heatmap_figure(frame, spec)

    # Effective labels: x axis label; y axis label reflects normalization/rate.
    x_title = _axis_label(prepared.x_label, spec.x_label)
    y_title = _axis_label(prepared.y_label, spec.y_label)

    if spec.kind == "histogram":
        _histogram_opacity(fig, frame, spec)

    if spec.kind in ("pie", "sunburst", "treemap", "radar", "waffle", "calendar"):
        # Non-Cartesian kinds: no axis titles/ticks to pin. Polar gets a
        # radial label; calendar/waffle label their own layouts.
        if spec.subtitle:
            fig.update_layout(title_text=f"{spec.title or _default_title(spec)}<br><sup>{spec.subtitle}</sup>")
        if has_groups and spec.kind in ("radar", "violin"):
            fig.update_layout(legend_title_text=spec.group or "group")
        return Result.success(fig)

    if spec.horizontal and spec.kind == "bar":
        # Swapped axes: categorical ticks ONLY on the y axis; the numeric
        # measure axis (x) keeps numeric ticks and the measure label.
        # ``None`` (unspecified) renders upright (0) on the horizontal axis;
        # ANY explicit angle — including -30 — is honored verbatim.
        cat_ticks = _categorical_tick_mapping(frame, "x", spec)
        _apply_ticks(fig, "y", cat_ticks, spec.x_tick_angle if spec.x_tick_angle is not None else 0, x_title)
        _pin_category_order(fig, "y", cat_ticks)
        fig.update_layout(xaxis_title=y_title)
    elif spec.kind == "heatmap":
        # go.Heatmap carries its own axis titles/coordinates (dtype-true);
        # effective-label defaults (prepared.x/y_label) apply, and explicit
        # overrides still win over the prepared names.
        fig.update_layout(xaxis_title=spec.x_label or x_title)
        fig.update_layout(yaxis_title=spec.y_label or y_title)
    else:
        cat_ticks = None if spec.kind == "histogram" else _categorical_tick_mapping(frame, "x", spec)
        _apply_ticks(
            fig, "x", cat_ticks, spec.x_tick_angle if spec.x_tick_angle is not None else _DEFAULT_TICK_ANGLE, x_title
        )
        _pin_category_order(fig, "x", cat_ticks)
        fig.update_layout(yaxis_title=y_title)

    fig.update_layout(
        template="plotly_white",
        width=spec.width,
        height=spec.height,
        title={"text": spec.title or _default_title(spec), "x": 0.5},
        margin_t=80 if spec.subtitle else 60,
        margin_l=70,
        margin_r=30,
        margin_b=80,
        font_family="Helvetica, Arial, sans-serif",
    )
    if spec.subtitle:
        fig.update_layout(title_text=f"{spec.title or _default_title(spec)}<br><sup>{spec.subtitle}</sup>")
    if has_groups:
        fig.update_layout(legend_title_text=spec.group or "group")
    return Result.success(fig)


def _default_title(spec: ChartSpec) -> str:
    if spec.kind in ("pie", "sunburst", "treemap", "waffle"):
        return f"{spec.y} by {spec.x}" if spec.kind == "pie" else f"{spec.y} across {spec.x} / {spec.group}"
    if spec.kind == "calendar":
        return f"{spec.y} per day"
    return f"{spec.y} by {spec.x}"


def render_chart(
    prepared: PreparedChart,
    spec: ChartSpec,
) -> Result[str]:
    """Render the prepared data to HTML (plotly), with a knowable fallback."""
    try:
        result = build_figure(prepared, spec)
        if result.value is None:
            return Result[str](None, result.diagnostics)
        fig = result.unwrap()
        include = "inline" if spec.offline else "cdn"
        html_str: str = fig.to_html(full_html=False, include_plotlyjs=include)
        return Result.success(html_str)
    except ImportError:
        return Result.success(
            _fallback_table(prepared, spec),
            Diagnostic.warning(
                "CHART_PLOTLY_UNAVAILABLE",
                "plotly is not installed; rendered a plain HTML table with the same data instead. "
                f"Fix: {_FALLBACK_FIX}",
                fix=_FALLBACK_FIX,
            ),
        )
    except Exception as exc:  # plotly can raise broadly; one clear diagnostic
        return Result.failure(Diagnostic.error("CHART_RENDER_FAILED", f"plotly render failed: {exc}"))


def chart_html(prepared: PreparedChart, spec: ChartSpec) -> Result[str]:
    """The renderer entry point (historical name kept for callers)."""
    return render_chart(prepared, spec)


def _fallback_table(prepared: PreparedChart, spec: ChartSpec) -> str:
    """Honest data table for the no-plotly fallback (same numbers, same order)."""
    frame = prepared.data
    cols = list(frame.columns)
    header = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in cols)
    rows_html = ""
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.6g}" if math.isfinite(v) else "—")
            else:
                cells.append(html_lib.escape(str(v)))
        rows_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
    subtitle_html = f"<p><em>{html_lib.escape(spec.subtitle)}</em></p>" if spec.subtitle else ""
    note = f"<p>Chart kind: {html_lib.escape(spec.kind)} (data table fallback — plotly not installed)</p>"
    return (
        f"<h3>{html_lib.escape(spec.title or f'{spec.y} by {spec.x}')}</h3>"
        f"{subtitle_html}{note}"
        "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
        f"<tr>{header}</tr>{rows_html}</table>"
    )


def _kaleido_bytes(fig: Any, spec: ChartSpec, fmt: str, browser_path: str | None) -> bytes:
    """One kaleido call (v1 API) with an optional per-call browser path.

    kaleido v1 has no ``pio.kaleido.chrome`` attribute (v0's assignment was a
    silent no-op); the supported override is ``kopts={"path": ...}`` through
    ``calc_fig_sync``, which raises ``ChromeNotFoundError`` for a nonexistent
    executable. No global configuration is touched.
    """
    import kaleido  # type: ignore[import-not-found]  # optional [plotly-image] extra

    kopts: dict[str, Any] = {}
    if browser_path:
        kopts["path"] = browser_path
    img_bytes: bytes = kaleido.calc_fig_sync(
        fig.to_plotly_json(),
        opts={
            "format": fmt,
            "width": spec.width,
            "height": spec.height,
        },
        kopts=kopts,
    )
    return img_bytes


def chart_image_bytes(
    prepared: PreparedChart,
    spec: ChartSpec,
    fmt: str = "png",
    *,
    browser_path: str | None = None,
) -> Result[bytes]:
    """Export the prepared chart as in-memory image bytes (PNG/SVG/PDF).

    Uses kaleido v1 directly (``calc_fig_sync``); returns bytes only — the
    CLI writes them through ``OutputWriter.write_bytes`` and registers the
    artifact (R3). An invalid explicit browser path FAILS with
    ``ChromeNotFoundError`` (a precise diagnostic, never a silent default).
    A missing kaleido or Chrome is a precise failure diagnostic; an
    explicitly requested export failing here fails the run — HTML is never
    substituted.
    """
    if fmt not in ("png", "svg", "pdf"):
        return Result.failure(Diagnostic.error("CHART_BAD_FORMAT", f"format must be png, svg, or pdf, got {fmt!r}"))
    built = _build_or_fail(prepared, spec)
    if built.value is None:
        return Result[bytes](None, built.diagnostics)
    fig = built.unwrap()
    try:
        img_bytes: bytes = _kaleido_bytes(fig, spec, fmt, browser_path)
        return Result.success(img_bytes)
    except ImportError as exc:
        return Result.failure(
            Diagnostic.error(
                "CHART_IMAGE_KALEIDO_MISSING",
                f"image export needs kaleido: {exc}. Install the export extra: {_IMAGE_FIX}",
                fix=_IMAGE_FIX,
            )
        )
    except Exception as exc:
        return Result.failure(_image_diagnostic(exc))


def _build_or_fail(prepared: PreparedChart, spec: ChartSpec) -> Result[Figure]:
    """Figure construction with plotly-missing translated to a diagnostic."""
    try:
        return build_figure(prepared, spec)
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "CHART_IMAGE_PLOTLY_MISSING",
                "image export needs plotly, which is not installed",
                fix=_FALLBACK_FIX,
            )
        )


def _image_diagnostic(exc: Exception) -> Diagnostic:
    """kaleido/Chrome failures get the actionable fix, others stay generic."""
    message = str(exc)
    lowered = message.lower()
    name = type(exc).__name__
    if name == "ChromeNotFoundError" or "chrome" in lowered or "chromium" in lowered or "kaleido" in lowered:
        return Diagnostic.error(
            "CHART_IMAGE_BROWSER_NOT_FOUND",
            f"image export failed: {message}. Kaleido v1 needs Chrome/Chromium; point at an "
            "installed browser with the BROWSER_PATH environment variable or pass --browser-path. "
            f"Install the export extra with: {_IMAGE_FIX}",
            fix=_IMAGE_FIX,
        )
    return Diagnostic.error("CHART_IMAGE_FAILED", f"image export failed: {message}")
