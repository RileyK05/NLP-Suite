"""gis_map — thin CLI: a geocoded CSV to pin map, heatmap and KML (HW4).

``python -m tools.gis_map geocoded.csv out/ --title "SOTU places"`` reads any
table with latitude/longitude columns (the ``geocoded.csv`` from
``tools.geocode`` is the usual input) and writes four views of the same
points: ``pin_map.html``, ``heatmap.html``, ``pins.kml`` for Google Earth Pro,
and — with ``--group-col`` — ``pins_tour.kml``, an Earth tour that flies the
placemarks grouped and in order.

A pin map marks each place once (radius driven by ``--weight-col``); a heatmap
aggregates the same points into intensity. CSV-in tool: no corpus, no parse.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from core.result import Diagnostic, Result

__all__ = ["main"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pin map, heatmap, KML and Earth tour from a geocoded CSV")
    p.add_argument("input", type=Path, help="input CSV of geocoded places")
    p.add_argument("output", type=Path, help="output root directory (a run dir is created inside)")
    p.add_argument("--lat-col", default="Lat", help="latitude column")
    p.add_argument("--lon-col", default="Lon", help="longitude column")
    p.add_argument("--name-col", default="Place", help="place-name column")
    p.add_argument("--weight-col", default="", help="column driving pin radius / heat intensity")
    p.add_argument("--group-col", default="", help="column grouping placemarks into a Google Earth tour")
    p.add_argument("--title", default="", help="map title")
    return p.parse_args(argv)


def _read_input(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        print(f"gis_map: cannot read {path}: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    frame = _read_input(args.input)
    if frame is None:
        return 1
    missing = [column for column in (args.lat_col, args.lon_col) if column not in frame.columns]
    if missing:
        diag = Diagnostic.error(
            "GIS_MAP_MISSING_COLUMN",
            f"missing column(s): {missing}",
            missing=missing,
            fix=f"expected columns {args.lat_col!r} and {args.lon_col!r}; pass --lat-col/--lon-col or fix the CSV header",
        )
        print(diag, file=sys.stderr)
        print(f"fix: {diag.context['fix']}", file=sys.stderr)
        return 1

    from core.gis.mapping import heatmap_html, kml, tour_kml
    from core.gis.pins import pinmap_html

    weight = args.weight_col or None
    tour = None
    if args.group_col:
        tour = tour_kml(
            frame,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            name_col=args.name_col,
            group_col=args.group_col,
            title=args.title or "Tour",
        )
    planned: list[tuple[str, Result[str], str, str]] = [
        (
            "pin_map.html",
            pinmap_html(
                frame,
                lat_col=args.lat_col,
                lon_col=args.lon_col,
                name_col=args.name_col,
                weight_col=weight,
                title=args.title,
            ),
            "pin map",
            "chart",
        ),
        (
            "heatmap.html",
            heatmap_html(frame, args.lat_col, args.lon_col, weight_col=weight, title=args.title),
            "heatmap",
            "chart",
        ),
        ("pins.kml", kml(frame, args.lat_col, args.lon_col, args.name_col), "KML placemarks", "map"),
    ]
    if tour is not None:
        planned.append(("pins_tour.kml", tour, "Google Earth tour", "map"))

    all_diags: list[Diagnostic] = []
    for _, result, _, _ in planned:
        all_diags.extend(result.diagnostics)
    if any(result.value is None for _, result, _, _ in planned):
        for diag in all_diags:
            print(diag, file=sys.stderr)
        return 1

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(args.output, tool="gis_map", params=vars(args), inputs=[args.input], corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1
    for name, result, description, kind in planned:
        written = writer.write_text(result.unwrap(), name, kind=kind, description=description)
        if not written.ok:
            writer.abandon()
            for diag in written.diagnostics:
                print(diag, file=sys.stderr)
            return 1
    writer.add_diagnostics(*all_diags)
    env = writer.finalize()
    if not env.ok:
        for diag in env.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    print(f"Wrote {len(planned)} artifact(s) to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
