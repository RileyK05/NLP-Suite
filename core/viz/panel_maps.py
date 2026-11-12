"""Offline map geometry shared by interactive and publication renderers."""

from __future__ import annotations

from functools import cache
from math import floor, sqrt
from typing import TYPE_CHECKING

from core.viz.world_land import LAND

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

WATER = "#e9f2f6"
LAND_FILL = "#fbfcf9"
COAST = "#a8bcbf"
GRID = "#cadbe2"
PIN = "#006b8f"
CELL_LOW = "#a9e4e0"
CELL_HIGH = "#004f70"


@cache
def land_polygons() -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(tuple((float(x), float(y)) for x, y in (point.split(",") for point in ring.split())) for ring in LAND)


def cell_center(lon: float, lat: float, degrees: int) -> tuple[float, float]:
    """Index half-open bins, with +180° identified with -180°."""
    lon = -180 if lon == 180 else lon
    x = floor((lon + 180) / degrees)
    y = min(floor((lat + 90) / degrees), 180 // degrees - 1)
    return (x + 0.5) * degrees - 180, (y + 0.5) * degrees - 90


def pin_radius(weight: float, maximum: float) -> float:
    """Area tracks the weight, with a small floor for legibility."""
    return max(3.5, 14 * sqrt(weight / maximum)) if maximum > 0 else 5


def cell_color(value: float, maximum: float) -> str:
    share = sqrt(value / maximum)
    channels = [
        round(int(CELL_LOW[i : i + 2], 16) * (1 - share) + int(CELL_HIGH[i : i + 2], 16) * share) for i in (1, 3, 5)
    ]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def add_plotly_land(fig: Figure) -> None:
    import plotly.graph_objects as go

    land_x: list[float | None] = []
    land_y: list[float | None] = []
    for ring in land_polygons():
        xs, ys = zip(*ring, strict=True)
        land_x.extend((*xs, None))
        land_y.extend((*ys, None))
    fig.add_trace(
        go.Scatter(
            x=land_x,
            y=land_y,
            mode="lines",
            fill="toself",
            fillcolor=LAND_FILL,
            line={"color": COAST, "width": 0.7},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(plot_bgcolor=WATER, dragmode="zoom")
    fig.update_xaxes(range=[-180, 180], constrain="domain", dtick=60, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(range=[-90, 90], scaleanchor="x", constrain="domain", dtick=30, gridcolor=GRID, zeroline=False)


def svg_world(marks: str, title: str) -> str:
    """SVG in geographic degrees, with upright text and a fixed 2:1 aspect."""
    coast = "".join("M" + "L".join(f"{lon:g},{-lat:g}" for lon, lat in polygon) + "Z" for polygon in land_polygons())
    return (
        '<svg width="100%" viewBox="-180 -90 360 180" xmlns="http://www.w3.org/2000/svg" role="img">'
        f"<title>{title}</title>"
        f'<rect x="-180" y="-90" width="360" height="180" fill="{WATER}"/>'
        f'<path d="{coast}" fill="{LAND_FILL}" stroke="{COAST}" stroke-width="0.28"/>'
        f"{marks}</svg>"
    )
