"""geocode — thin CLI: the places named in a corpus to lat/lon (HW4 geocoding).

Pipeline: the shared parse -> ``extract_locations`` (whole multi-word NER
place spans) -> the most frequent ``--limit`` places -> one of three
providers:

* ``offline`` — the baked KB in ``core.gis.geocode`` (no network, no key)
* ``google`` — ``core.gis.online.GoogleGeocoder``; needs ``GOOGLE_MAPS_KEY``
  in the environment (never on the command line)
* ``nominatim`` — ``core.gis.nominatim``; no key at all, which is why it is
  the provider to prefer for coursework

Outputs ``locations.csv`` (the place extract) and ``geocoded.csv``
(``Place, Lat, Lon, Status``).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from core.result import Diagnostic, Result
from tools._cli import (
    build_common_parser,
    get_pipeline,
    handle_result_states,
    make_writer,
    resolve_config,
)

__all__ = ["geocode_places", "main"]

_GEOCODE_COLUMNS = ["Place", "Lat", "Lon", "Status"]
_NO_KEY_FIX = "set the GOOGLE_MAPS_KEY environment variable or use --provider nominatim"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Geocode the places named in a corpus (offline KB, Google, or Nominatim)")
    p.add_argument(
        "--provider",
        choices=["offline", "google", "nominatim"],
        default="offline",
        help="geocoder: offline baked KB, google, or nominatim",
    )
    p.add_argument("--limit", type=int, default=200, help="most frequent places to geocode")
    p.add_argument("--min-count", type=int, default=1, help="fewest mentions for a place to count")
    return p.parse_args(argv)


def geocode_places(provider: str, places: list[str]) -> Result[pd.DataFrame]:
    """Geocode *places* with *provider* to the shared (Place, Lat, Lon, Status) contract.

    Kept public so ``tools.svo_map`` geocodes through exactly one code path;
    an empty place list is an empty table, not a failure.
    """
    if not places:
        return Result.success(
            pd.DataFrame(columns=_GEOCODE_COLUMNS),
            Diagnostic.info("GEOCODE_EMPTY_LIST", "no places to geocode"),
        )
    if provider == "offline":
        from core.gis.geocode import geocode_many

        return geocode_many(places)
    if provider == "nominatim":
        from core.gis.nominatim import NominatimGeocoder

        return NominatimGeocoder().geocode_many(places)
    if provider != "google":
        return Result.failure(
            Diagnostic.error(
                "GEOCODE_BAD_PROVIDER", f"unknown provider {provider!r}", fix="use offline, google, or nominatim"
            )
        )
    from core.gis.online import GoogleGeocoder, api_key_from_env

    key = api_key_from_env()
    if key is None:
        return Result.failure(Diagnostic.error("GEOCODE_NO_KEY", "GOOGLE_MAPS_KEY is not set", fix=_NO_KEY_FIX))
    client = GoogleGeocoder(key)
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    for place in places:
        resolved = client.geocode(place)
        if resolved.ok:
            lat, lon = resolved.unwrap()
            rows.append({"Place": place, "Lat": lat, "Lon": lon, "Status": "OK"})
        elif resolved.diagnostics and resolved.diagnostics[0].code == "GEOCODE_UNKNOWN":
            rows.append({"Place": place, "Lat": 0.0, "Lon": 0.0, "Status": "UNKNOWN"})
            diags.append(Diagnostic.info("GEOCODE_UNKNOWN", resolved.diagnostics[0].message, place=place))
        else:
            rows.append({"Place": place, "Lat": 0.0, "Lon": 0.0, "Status": "UNAVAILABLE"})
            diags.extend(resolved.diagnostics)
    return Result.success(pd.DataFrame(rows, columns=_GEOCODE_COLUMNS), *diags)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.limit < 1 or args.min_count < 1:
        print("geocode: --limit and --min-count must both be >= 1", file=sys.stderr)
        return 2
    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    from core.io.reader import read_corpus

    corpus_result = read_corpus(args.corpus)
    # A bad document is a diagnostic, not an aborted run: partial corpora flow
    # into the four-state writer policy (tools/_cli.handle_result_states).
    if corpus_result.value is None:
        for diag in corpus_result.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    for diag in corpus_result.diagnostics:
        print(diag, file=sys.stderr)
    corpus = corpus_result.unwrap()

    from core.pipelines.cache import PipelineCache

    pipeline = get_pipeline(PipelineCache(), config, args)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()

    from core.analysis.location_extract import extract_locations

    found = extract_locations(table, min_count=args.min_count)
    if found.value is None:
        for diag in found.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    locations = found.unwrap().head(args.limit).reset_index(drop=True)
    places = [str(place) for place in locations["Place"].tolist()]

    geo_result = geocode_places(args.provider, places)
    if geo_result.value is None:
        for diag in geo_result.diagnostics:
            print(diag, file=sys.stderr)
            fix = diag.context.get("fix") if isinstance(diag.context, dict) else None
            if fix:
                print(f"fix: {fix}", file=sys.stderr)
        return 1
    geocoded = geo_result.unwrap()

    writer = make_writer(args, corpus, tool="geocode", params=vars(args))
    for frame, name, description in (
        (locations, "locations.csv", "places named in the corpus"),
        (geocoded, "geocoded.csv", "geocoded places"),
    ):
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for diag in written.diagnostics:
                print(diag, file=sys.stderr)
            return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *found.diagnostics,
        *geo_result.diagnostics,
    )
    env = writer.finalize()
    if not env.ok:
        for diag in env.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    print(f"Wrote {len(geocoded)} place(s) to {writer.run_dir / 'geocoded.csv'}")
    return 1 if parse_state == "partial" or geo_result.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
