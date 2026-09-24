"""earth — thin CLI for Google Earth tour KML from a geocoded CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Google Earth tour (gx:Track KML over a geocoded CSV)")
    p.add_argument("input", type=Path, help="geocoded CSV (Place,Lat,Lon[,group column])")
    p.add_argument("output", type=Path, help="output root")
    p.add_argument("--title", default="Tour")
    p.add_argument("--group-col", default=None, help="column whose values become one tour folder each")
    # Named columns so the suite's own geocoded tables feed straight in: the
    # movement tracks from `ner --movement --geocode` label places "Location",
    # not "Place", and renaming a column by hand to use the next tool is the
    # kind of seam this rebuild exists to remove.
    p.add_argument("--name-col", default="Place", help="column holding the place label (default: Place)")
    p.add_argument("--lat-col", default="Lat", help="latitude column (default: Lat)")
    p.add_argument("--lon-col", default="Lon", help="longitude column (default: Lon)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.input.is_file():
        print(f"{args.input} is not a file", file=sys.stderr)
        return 1
    try:
        df = pd.read_csv(args.input, encoding="utf-8-sig")
    except Exception as exc:
        print(f"could not read {args.input}: {exc}", file=sys.stderr)
        return 1

    from core.gis.mapping import tour_kml
    from core.io.writer import OutputWriter

    from tools._cli import write_or_die

    try:
        writer = OutputWriter(args.output, tool="earth", params=vars(args), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    tour = tour_kml(
        df,
        title=args.title,
        group_col=args.group_col,
        name_col=args.name_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
    )
    if not tour.ok:
        # Discard the staged run: a failed tour has no artifact to publish.
        writer.abandon()
        for d in tour.diagnostics:
            print(d, file=sys.stderr)
        return 1
    write_or_die(writer, writer.write_text(tour.unwrap(), "tour.kml", kind="map", description="Google Earth tour"))
    writer.add_diagnostics(*tour.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote tour.kml to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
