"""Geocoding — offline baked coordinates and haversine."""

from __future__ import annotations

import math

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["distance", "distance_places", "geocode", "geocode_many"]

# Baked coordinates for tests and demo (no API key, no network).
_COORDS: dict[str, tuple[float, float]] = {
    "paris": (48.8566, 2.3522),
    "london": (51.5074, -0.1278),
    "new york": (40.7128, -74.0060),
    "berlin": (52.5200, 13.4050),
    "tokyo": (35.6895, 139.692),
    "rome": (41.9028, 12.4964),
}


def geocode(place: str) -> Result[tuple[float, float]]:
    """Lookup lat/lon for *place* (case-insensitive)."""
    if not place or not place.strip():
        return Result.failure(Diagnostic.error("GEOCODE_EMPTY", "place must be non-empty"))
    key = place.strip().lower()
    coords = _COORDS.get(key)
    if coords is None:
        return Result.failure(Diagnostic.error("GEOCODE_UNKNOWN", f"{place!r} not in baked KB", place=place))
    return Result.success(coords)


def geocode_many(places: list[str]) -> Result[pd.DataFrame]:
    """Batch geocode; unknown places become INFO diagnostics."""
    if not places:
        return Result.failure(Diagnostic.error("GEOCODE_EMPTY_LIST", "no places to geocode"))
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    for place in places:
        res = geocode(place)
        if res.ok:
            lat, lon = res.unwrap()
            rows.append({"Place": place, "Lat": lat, "Lon": lon, "Status": "OK"})
        else:
            rows.append({"Place": place, "Lat": 0.0, "Lon": 0.0, "Status": "UNKNOWN"})
            diags.append(Diagnostic.info("GEOCODE_UNKNOWN", f"{place!r} not in baked KB", place=place))
    df = pd.DataFrame(rows, columns=["Place", "Lat", "Lon", "Status"])
    if diags:
        return Result.success(df, *diags)
    return Result.success(df)


def distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Haversine distance in kilometres."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(r * c, 3)


def distance_places(a: str, b: str) -> Result[float]:
    """Distance between two named places via baked coords."""
    ra = geocode(a)
    if not ra.ok:
        return Result[float](None, ra.diagnostics)
    rb = geocode(b)
    if not rb.ok:
        return Result[float](None, rb.diagnostics)
    lat1, lon1 = ra.unwrap()
    lat2, lon2 = rb.unwrap()
    return Result.success(distance(lat1, lon1, lat2, lon2))
