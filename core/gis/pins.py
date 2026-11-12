"""Pin maps — offline equirectangular scatter HTML (HW4 GIS pin maps).

A pin map marks each geocoded place once; a heatmap (``core.gis.mapping``)
aggregates the same points into intensity. The pin radius here is driven by an
optional weight column, so a place carrying more SVO rows reads louder than a
stopover (the syllabus's "how can you improve the radius size of a heatmap"
has the same answer: let a data column drive the size).

Plotly draws the interactive world scatter when it is installed; without it
the same points render as plain SVG and the envelope carries
``PINS_PLOTLY_UNAVAILABLE`` — knowable degradation, the house viz rule.
Rows outside WGS-84 bounds are skipped with a warning count, through the same
``_coords`` validation the KML writers use so every map agrees on what a
coordinate is.
"""

from __future__ import annotations

import html as html_lib
import math

import pandas as pd

from core.gis.mapping import _coords, _weight
from core.result import Diagnostic, Result
from core.viz.panel_maps import PIN, add_plotly_land, svg_world

__all__ = ["pinmap_html"]

_WIDTH = 800
_HEIGHT = 400
_MIN_RADIUS = 3.0
_MAX_RADIUS = 10.0


def _radius(weight: float, max_weight: float) -> float:
    """Pin radius from a weight: heavier place, bigger pin (bounded)."""
    if max_weight <= 0:
        return _MIN_RADIUS
    return max(_MIN_RADIUS, _MAX_RADIUS * math.sqrt(weight / max_weight))


def pinmap_html(  # noqa: PLR0913, PLR0911 -- column names plus weight/title, and one named return per failure mode
    frame: pd.DataFrame,
    *,
    lat_col: str = "Lat",
    lon_col: str = "Lon",
    name_col: str = "Place",
    weight_col: str | None = None,
    title: str = "",
) -> Result[str]:
    """Pin-map HTML for a geocoded frame (plotly world scatter, SVG fallback)."""
    if frame.empty:
        return Result.failure(Diagnostic.error("PINS_EMPTY", "frame is empty"))
    if lat_col not in frame.columns or lon_col not in frame.columns:
        return Result.failure(
            Diagnostic.error("PINS_MISSING_COLUMN", f"columns {lat_col!r}, {lon_col!r} must be in frame")
        )
    points: list[tuple[float, float, str, float]] = []
    skipped = 0
    for _, row in frame.iterrows():
        pair = _coords(row, lat_col, lon_col)
        if pair is None:
            skipped += 1
            continue
        lat, lon = pair
        name = str(row[name_col]) if name_col in frame.columns else f"{lat},{lon}"
        weight = _weight(row.get(weight_col, 1) if weight_col else 1)
        points.append((lat, lon, name, weight))
    if not points:
        return Result.failure(Diagnostic.error("PINS_NO_COORDS", "no valid coordinates"))

    diags: list[Diagnostic] = []
    if skipped:
        # A quietly shorter map is exactly the failure mode this suite exists
        # to avoid: say what was dropped.
        diags.append(
            Diagnostic.warning(
                "PINS_SKIPPED_ROWS",
                f"{skipped} row(s) were unresolved or had invalid coordinates and were skipped",
                count=skipped,
            )
        )

    try:
        import plotly.graph_objects as go  # noqa: PLC0415 -- plotly is an optional viz backend

        max_weight = max((weight for *_, weight in points), default=1.0) or 1.0
        figure = go.Figure()
        add_plotly_land(figure)
        figure.add_trace(
            go.Scatter(
                y=[lat for lat, _, _, _ in points],
                x=[lon for _, lon, _, _ in points],
                text=[html_lib.escape(name) for _, _, name, _ in points],
                hovertemplate="%{text}<br>%{y:.2f}°, %{x:.2f}°<extra></extra>",
                mode="markers",
                marker={
                    "size": [_radius(weight, max_weight) * 2.2 for *_, weight in points],
                    "color": PIN,
                    "opacity": 0.83,
                    "line": {"color": "white", "width": 1},
                },
                showlegend=False,
            )
        )
        figure.update_layout(
            template="plotly_white",
            title_text=html_lib.escape(title or "Pin map"),
            font_size=11,
            height=540,
            margin={"l": 38, "r": 30, "t": 55, "b": 50},
        )
        return Result.success(
            figure.to_html(full_html=False, include_plotlyjs=True) + "<p>Land outlines: Natural Earth.</p>",
            *diags,
        )
    except ImportError:
        pass
    except Exception as exc:
        return Result.failure(Diagnostic.error("PINS_PLOT_FAILED", f"plotly failed: {exc}"))

    try:
        html_str = _svg_html(points, title)
    except Exception as exc:
        return Result.failure(Diagnostic.error("PINS_PLOT_FAILED", f"fallback failed: {exc}"))
    diags.append(
        Diagnostic.warning(
            "PINS_PLOTLY_UNAVAILABLE",
            "plotly is not installed; rendered a plain SVG pin map instead of the interactive "
            "world scatter. Fix: pip install plotly",
            fix="pip install plotly",
        )
    )
    return Result.success(html_str, *diags)


def _svg_html(points: list[tuple[float, float, str, float]], title: str) -> str:
    """Equirectangular SVG scatter — same points the chart would show."""
    safe_title = html_lib.escape(title or "Pin map")
    max_weight = max((weight for *_, weight in points), default=1.0) or 1.0
    circles: list[str] = []
    for lat, lon, name, weight in points:
        x = (lon + 180.0) / 360.0 * _WIDTH
        y = (90.0 - lat) / 180.0 * _HEIGHT
        radius = _radius(weight, max_weight)
        circles.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' fill='{PIN}' fill-opacity='0.83' tabindex='0'>"
            f"<title>{html_lib.escape(name)}</title></circle>"
        )
    marks = f"<g transform='translate(-180 -90) scale({360 / _WIDTH})'>{''.join(circles)}</g>"
    return f"<h3>{safe_title}</h3>{svg_world(marks, safe_title)}<p>Land outlines: Natural Earth.</p>"
