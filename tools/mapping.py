"""mapping — thin CLI for KML and heatmap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mapping (KML + heatmap)")
    p.add_argument("input", type=Path, help="geocoded CSV (Place,Lat,Lon)")
    p.add_argument("output", type=Path, help="output root")
    p.add_argument("--title", default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.input.is_file():
        print(f"{args.input} is not a file", file=sys.stderr)
        return 1
    try:
        df = pd.read_csv(args.input, encoding="utf-8")
    except Exception as exc:
        print(f"could not read {args.input}: {exc}", file=sys.stderr)
        return 1

    from core.gis.mapping import heatmap_html, kml
    from core.io.writer import OutputWriter

    from tools._cli import write_or_die

    try:
        writer = OutputWriter(args.output, tool="mapping", params=vars(args), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    k = kml(df, lat_col="Lat", lon_col="Lon", name_col="Place")
    if k.ok:
        # KML is a map artifact; "table" made the viewer try to parse it as CSV.
        write_or_die(writer, writer.write_text(k.unwrap(), "map.kml", kind="map", description="KML"))
        writer.add_diagnostics(*k.diagnostics)
    else:
        for d in k.diagnostics:
            print(d, file=sys.stderr)

    h = heatmap_html(df, lat_col="Lat", lon_col="Lon", title=args.title)
    if h.ok:
        write_or_die(writer, writer.write_html(h.unwrap(), "heatmap.html", description="heatmap"))
        writer.add_diagnostics(*h.diagnostics)

    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
