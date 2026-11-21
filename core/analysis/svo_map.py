"""SVO-to-GIS handoff — where narrative events happen (HW4 SVO maps).

``map_svo`` joins subject-verb-object triples to geocoded places through the
document provenance both tables carry: an SVO row is mapped when its document
names at least one geocoded place. When a document names several places the
result carries one row per (SVO triple, Place) pair — HW4 asks what the Google
Earth Pro of SVOs tells you and how it differs from the map GIS_main produces,
and the difference is exactly this: the SVO map is a map of narrative EVENTS
(every triple pinned to every place its document is about), not a map of where
places happen to be mentioned. Keeping every (triple, place) pair is what makes
the verbs land on geography instead of the documents landing on their first
place.

Triples whose document has no geography are dropped with a warning count: a
narrative triple with no geography is not a map point.
"""

from __future__ import annotations

import math

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["map_svo", "svo_geo_summary"]

_MAPPED_COLUMNS = ["Subject", "Verb", "Object", "Place", "Lat", "Lon", "Document", "Document ID", "Sentence ID"]
_SUMMARY_COLUMNS = ["Place", "Lat", "Lon", "SVO rows", "Documents"]


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _resolved_places(locations: pd.DataFrame, geocoded: pd.DataFrame) -> dict[str, list[tuple[str, float, float]]]:
    """Document ID -> [(place, lat, lon)] for geocoded places only.

    A Status column decides: only "OK" rows carry real coordinates (unknown
    rows hold 0.0 placeholders in the shared geocode contract, and 0.0/0.0 is
    Null Island — a real point every unresolved place would otherwise claim).
    Without a Status column, finite coordinates decide.
    """
    coords: dict[str, tuple[str, float, float]] = {}
    statuses = geocoded["Status"].astype(str).tolist() if "Status" in geocoded.columns else None
    lats = geocoded["Lat"].tolist()
    lons = geocoded["Lon"].tolist()
    geo_places = geocoded["Place"].astype(str).tolist()
    for idx, (place, raw_lat, raw_lon) in enumerate(zip(geo_places, lats, lons, strict=True)):
        if statuses is not None and statuses[idx] != "OK":
            continue
        lat = _finite(raw_lat)
        lon = _finite(raw_lon)
        if lat is None or lon is None:
            continue
        coords[place.strip().lower()] = (place, lat, lon)

    by_document: dict[str, list[tuple[str, float, float]]] = {}
    loc_places = locations["Place"].astype(str).tolist()
    loc_dids = locations["Document ID"].astype(str).tolist()
    for raw_place, key in zip(loc_places, loc_dids, strict=True):
        hit = coords.get(raw_place.strip().lower())
        if hit is None:
            continue
        bucket = by_document.setdefault(key, [])
        if hit not in bucket:
            bucket.append(hit)
    return by_document


def map_svo(
    svo_rows: pd.DataFrame,
    locations: pd.DataFrame,
    *,
    geocoded: pd.DataFrame,
) -> Result[pd.DataFrame]:
    """Pin every SVO triple to every geocoded place its document names.

    One row per (triple, place) pair — a document about three places maps each
    of its triples three times, which is what makes this a map of narrative
    events rather than a map of documents. Unmatched triples (no geocoded
    place in their document) are dropped with a warning count.
    """
    svo_needed = ["Subject", "Verb", "Object", "Sentence ID", "Document ID", "Document"]
    loc_needed = ["Document ID", "Place"]
    geo_needed = ["Place", "Lat", "Lon"]
    for label, frame, needed in (
        ("svo_rows", svo_rows, svo_needed),
        ("locations", locations, loc_needed),
        ("geocoded", geocoded, geo_needed),
    ):
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            return Result.failure(
                Diagnostic.error("SVO_MAP_MISSING_COLUMN", f"{label} is missing column(s): {missing}", missing=missing)
            )
    if svo_rows.empty:
        return Result.success(pd.DataFrame(columns=_MAPPED_COLUMNS))

    by_document = _resolved_places(locations, geocoded)
    rows: list[dict[str, object]] = []
    unmatched = 0
    subjs = svo_rows["Subject"].tolist()
    verbs = svo_rows["Verb"].tolist()
    objs = svo_rows["Object"].tolist()
    docs = svo_rows["Document"].tolist()
    dids = svo_rows["Document ID"].astype(str).tolist()
    sids = svo_rows["Sentence ID"].tolist()
    for subj, verb, obj, doc, did, sid in zip(subjs, verbs, objs, docs, dids, sids, strict=True):
        hits = by_document.get(did)
        if not hits:
            unmatched += 1
            continue
        for place, lat, lon in hits:
            rows.append(
                {
                    "Subject": subj,
                    "Verb": verb,
                    "Object": obj,
                    "Place": place,
                    "Lat": lat,
                    "Lon": lon,
                    "Document": doc,
                    "Document ID": did,
                    "Sentence ID": sid,
                }
            )
    out = pd.DataFrame(rows, columns=_MAPPED_COLUMNS)
    diags: list[Diagnostic] = []
    if unmatched:
        diags.append(
            Diagnostic.warning(
                "SVO_MAP_UNMATCHED",
                f"{unmatched} SVO row(s) named no geocoded place in their document and were dropped",
                count=unmatched,
            )
        )
    return Result.success(out, *diags)


def svo_geo_summary(mapped: pd.DataFrame) -> Result[pd.DataFrame]:
    """Per-place roll-up of a mapped frame (Place, Lat, Lon, SVO rows, Documents).

    The pin/heatmap feed: "SVO rows" is how many narrative events land on the
    place (pin weight), "Documents" how many documents put them there.
    """
    needed = ["Place", "Lat", "Lon", "Document ID"]
    missing = [column for column in needed if column not in mapped.columns]
    if missing:
        return Result.failure(
            Diagnostic.error("SVO_MAP_MISSING_COLUMN", f"mapped frame is missing column(s): {missing}", missing=missing)
        )
    if mapped.empty:
        return Result.success(pd.DataFrame(columns=_SUMMARY_COLUMNS))

    rows: list[dict[str, object]] = []
    for place, group in mapped.groupby("Place", sort=False):
        rows.append(
            {
                "Place": place,
                "Lat": float(group["Lat"].iloc[0]),
                "Lon": float(group["Lon"].iloc[0]),
                "SVO rows": len(group),
                "Documents": int(group["Document ID"].nunique()),
            }
        )
    out = pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)
    out = out.sort_values("SVO rows", ascending=False, kind="stable").reset_index(drop=True)
    return Result.success(out)
