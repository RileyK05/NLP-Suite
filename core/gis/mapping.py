"""Mapping — KML, Earth tour KML, and heatmap HTML."""

from __future__ import annotations

import html as html_lib
import math

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["heatmap_html", "kml", "tour_kml"]

# WGS-84 bounds. A row outside these is not a place on Earth: KML consumers
# either reject the document or silently clamp the point somewhere wrong, so
# both writers skip such rows rather than emit them.
_MAX_LAT = 90.0
_MAX_LON = 180.0


def _finite(value: object) -> float | None:
    """Coerce to a finite float, or None. ``float(nan)`` does not raise."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def _coords(row: pd.Series, lat_col: str, lon_col: str) -> tuple[float, float] | None:
    """A finite, in-range ``(lat, lon)`` pair, or None.

    Shared by both KML writers so a coordinate one accepts the other accepts
    too. ``kml`` previously checked only for NaN, so an out-of-range latitude
    reached the document while ``tour_kml`` dropped it.
    """
    lat = _finite(row[lat_col])
    lon = _finite(row[lon_col])
    if lat is None or lon is None or abs(lat) > _MAX_LAT or abs(lon) > _MAX_LON:
        return None
    return lat, lon


def kml(frame: pd.DataFrame, lat_col: str = "Lat", lon_col: str = "Lon", name_col: str = "Place") -> Result[str]:
    """KML placemarks for a geocoded frame."""
    if frame.empty:
        return Result.failure(Diagnostic.error("KML_EMPTY", "frame is empty"))
    for need in [lat_col, lon_col]:
        if need not in frame.columns:
            return Result.failure(Diagnostic.error("KML_MISSING_COLUMN", f"missing {need!r}"))
    placemarks: list[str] = []
    skipped = 0
    for _, row in frame.iterrows():
        pair = _coords(row, lat_col, lon_col)
        if pair is None:
            skipped += 1
            continue
        lat, lon = pair
        name = html_lib.escape(str(row[name_col])) if name_col in frame.columns else f"{lat},{lon}"
        placemarks.append(
            f"  <Placemark><name>{name}</name><Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>"
        )
    if not placemarks:
        return Result.failure(Diagnostic.error("KML_NO_COORDS", "no valid coordinates"))
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n' + "\n".join(placemarks) + "\n</Document></kml>\n"
    )
    diags: tuple[Diagnostic, ...] = ()
    if skipped:
        # A quietly shorter map is exactly the failure mode this suite exists
        # to avoid: say what was dropped.
        diags = (
            Diagnostic.warning(
                "KML_SKIPPED_ROWS",
                f"{skipped} row(s) had missing or out-of-range coordinates and were skipped",
                count=skipped,
            ),
        )
    return Result.success(doc, *diags)


def tour_kml(  # noqa: PLR0913 -- column names plus grouping/title, all keyword-only
    frame: pd.DataFrame,
    *,
    lat_col: str = "Lat",
    lon_col: str = "Lon",
    name_col: str = "Place",
    group_col: str | None = None,
    title: str = "Tour",
) -> Result[str]:
    """Google Earth tour KML: one gx:Track per group, ordered placemarks.

    CAP-GIS-07 (legacy ``GIS_Google_Earth_main`` shipped tours through a
    desktop Google Earth install). Each group's rows become one ``gx:Track``
    whose ``when`` stamps are synthetic (1 s apart) and whose ``coord``
    sequence is the row order — Google Earth plays the route as an animated
    fly-through. Rows with invalid coordinates are skipped; a group with no
    valid coordinates is dropped with a warning. The KML uses the gx
    extension namespace so tracks open in Google Earth Pro (free desktop)
    and respect ``<open>1</open>`` for immediate playback.
    """
    if frame.empty:
        return Result.failure(Diagnostic.error("TOUR_EMPTY", "frame is empty"))
    if lat_col not in frame.columns or lon_col not in frame.columns:
        return Result.failure(
            Diagnostic.error("TOUR_MISSING_COLUMN", f"columns {lat_col!r}, {lon_col!r} must be in frame")
        )
    if group_col is not None and group_col not in frame.columns:
        return Result.failure(Diagnostic.error("TOUR_BAD_GROUP", f"group column {group_col!r} not in frame"))

    folders: list[str] = []
    skipped = 0
    groups: list[tuple[str, pd.DataFrame]] = (
        [(title, frame)] if group_col is None else [(str(k), g) for k, g in frame.groupby(group_col, sort=False)]
    )
    for group_name, group in groups:
        whens: list[str] = []
        coords: list[str] = []
        placemarks: list[str] = []
        for idx, (_, row) in enumerate(group.iterrows()):
            pair = _coords(row, lat_col, lon_col)
            if pair is None:
                skipped += 1
                continue
            lat, lon = pair
            whens.append(f"          <when>2026-01-01T00:{idx // 60:02d}:{idx % 60:02d}Z</when>")
            coords.append(f"          {lon},{lat},0")
            name = html_lib.escape(str(row[name_col])) if name_col in frame.columns else f"{lat},{lon}"
            placemarks.append(
                f"      <Placemark><name>{name}</name><Point><coordinates>{lon},{lat},0</coordinates></Point></Placemark>"
            )
        if not coords:
            continue
        when_xml = "\n".join(whens)
        coord_xml = "\n".join(coords)
        mark_xml = "\n".join(placemarks)
        folders.append(
            f"  <Folder>\n"
            f"    <name>{html_lib.escape(group_name)}</name>\n"
            f"{mark_xml}\n"
            f"    <Placemark>\n"
            f"      <name>{html_lib.escape(group_name)} tour path</name>\n"
            f"      <Style><LineStyle><width>4</width></LineStyle></Style>\n"
            f"      <gx:Track>\n"
            f"        <altitudeMode>clampToGround</altitudeMode>\n"
            f"{when_xml}\n"
            f"{coord_xml}\n"
            f"      </gx:Track>\n"
            f"    </Placemark>\n"
            f"  </Folder>"
        )
    if not folders:
        return Result.failure(Diagnostic.error("TOUR_NO_COORDS", "no valid coordinates in frame"))
    doc = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2" '
        'xmlns:gx="http://www.google.com/kml/ext/2.2"><Document>\n'
        f"  <name>{html_lib.escape(title)}</name>\n" + "\n".join(folders) + "\n</Document></kml>\n"
    )
    diags: tuple[Diagnostic, ...] = ()
    if skipped:
        diags = (
            Diagnostic.warning(
                "TOUR_SKIPPED_ROWS",
                f"{skipped} row(s) had missing or out-of-range coordinates and were skipped",
            ),
        )
    return Result.success(doc, *diags)


def heatmap_html(
    frame: pd.DataFrame,
    lat_col: str = "Lat",
    lon_col: str = "Lon",
    weight_col: str | None = None,
    title: str = "",
) -> Result[str]:
    """Simple heatmap as an HTML table with intensity bars.

    If leaflet is desired, the caller can enhance the HTML; this stub is
    dependency-free and offline.
    """
    if frame.empty:
        return Result.failure(Diagnostic.error("HEATMAP_EMPTY", "frame is empty"))
    if lat_col not in frame.columns or lon_col not in frame.columns:
        return Result.failure(
            Diagnostic.error("HEATMAP_MISSING_COLUMN", f"columns {lat_col!r}, {lon_col!r} must be in frame")
        )
    title_html = f"<h3>{html_lib.escape(title)}</h3>" if title else ""
    # Weight for intensity
    if weight_col and weight_col in frame.columns:
        vals = pd.to_numeric(frame[weight_col], errors="coerce").fillna(1).tolist()
    else:
        vals = [1.0] * len(frame)
    max_v = max(vals) if vals else 1
    rows_html = ""
    for (_, row), w in zip(frame.iterrows(), vals, strict=True):
        lat = html_lib.escape(str(row[lat_col]))
        lon = html_lib.escape(str(row[lon_col]))
        pct = (float(w) / max_v * 100) if max_v else 0
        rows_html += (
            f"<tr><td>{lat}</td><td>{lon}</td>"
            f"<td><div style='background:#FF4500;height:12px;width:{pct:.1f}%'></div></td>"
            f"<td>{w}</td></tr>"
        )
    html_str = (
        f"{title_html}<table border='1' cellpadding='4' style='border-collapse:collapse;width:100%'>"
        f"<tr><th>Lat</th><th>Lon</th><th>Heat</th><th>Weight</th></tr>"
        f"{rows_html}</table>"
    )
    return Result.success(html_str)
