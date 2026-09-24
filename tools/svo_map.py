"""svo_map — thin CLI: subject-verb-object events on a map (HW4 SVO maps).

Pipeline: the shared parse -> ``clause_svo.extract_svo`` (who does what to
whom) + ``locations_by_document`` (where) -> geocode the most frequent places
-> ``svo_map.map_svo``, which pins every triple to every geocoded place its
document names -> a pin map weighted by SVO rows per place.

What the Google Earth Pro of SVOs tells you is WHERE NARRATIVE EVENTS HAPPEN
— verbs on the places their documents are about — which is a different map
from the one GIS_main draws from place mentions alone.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from core.result import Diagnostic
from tools._cli import (
    build_common_parser,
    get_pipeline,
    handle_result_states,
    make_writer,
    resolve_config,
)
from tools.geocode import geocode_places

__all__ = ["main"]

_SVO_COLUMNS = ["Subject", "Verb", "Object", "Sentence ID", "Document ID", "Document"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Map SVO triples to the places their documents name (offline, Google, or Nominatim)")
    p.add_argument(
        "--provider",
        choices=["offline", "google", "nominatim"],
        default="offline",
        help="geocoder: offline baked KB, google, or nominatim",
    )
    p.add_argument("--limit", type=int, default=200, help="most frequent places to geocode")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.limit < 1:
        print("svo_map: --limit must be >= 1", file=sys.stderr)
        return 2
    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    from core.io.reader import read_corpus

    corpus_result = read_corpus(args.corpus)
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

    from core.analysis.clause_svo import extract_svo
    from core.analysis.location_extract import locations_by_document
    from core.analysis.svo_map import map_svo, svo_geo_summary

    svos = extract_svo(table)
    if svos.value is None:
        for diag in svos.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    svo_rows = pd.DataFrame(
        [
            {
                "Subject": row.subject,
                "Verb": row.verb,
                "Object": row.obj,
                "Sentence ID": row.sentence_id,
                "Document ID": str(row.document_id),
                "Document": row.document,
            }
            for row in svos.unwrap()
        ],
        columns=_SVO_COLUMNS,
    )
    locs = locations_by_document(table)
    if locs.value is None:
        for diag in locs.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    locations = locs.unwrap()

    totals = locations.groupby("Place", sort=False)["Mentions"].sum().sort_values(ascending=False)
    places = [str(place) for place in totals.head(args.limit).index.tolist()]
    geo_result = geocode_places(args.provider, places)
    if geo_result.value is None:
        for diag in geo_result.diagnostics:
            print(diag, file=sys.stderr)
            fix = diag.context.get("fix") if isinstance(diag.context, dict) else None
            if fix:
                print(f"fix: {fix}", file=sys.stderr)
        return 1
    geocoded = geo_result.unwrap()

    mapped = map_svo(svo_rows, locations, geocoded=geocoded)
    if mapped.value is None:
        for diag in mapped.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    summary = svo_geo_summary(mapped.unwrap())
    if summary.value is None:
        for diag in summary.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    summary_frame = summary.unwrap()

    from core.gis.mapping import kml
    from core.gis.pins import pinmap_html

    artifacts: list[tuple[str, object, str, str]] = [
        (locations, "svo_locations.csv", "places per document", "table"),
        (summary_frame, "svo_geocoded.csv", "SVO rows per geocoded place", "table"),
    ]
    # No geocoded place means no map exists to draw; the tables still report
    # that finding, and the envelope says why the map artifacts are absent.
    extra_diags: list[Diagnostic] = []
    if summary_frame.empty:
        extra_diags.append(
            Diagnostic.warning(
                "SVO_MAP_NO_PLACES",
                "no geocoded place carries SVO rows; the map artifacts were not produced",
            )
        )
    else:
        pin = pinmap_html(summary_frame, weight_col="SVO rows", title="SVO map")
        marks = kml(summary_frame, "Lat", "Lon", "Place")
        extra_diags.extend(pin.diagnostics)
        extra_diags.extend(marks.diagnostics)
        if pin.value is None or marks.value is None:
            for diag in [*pin.diagnostics, *marks.diagnostics]:
                print(diag, file=sys.stderr)
            return 1
        artifacts.append((pin.unwrap(), "svo_map.html", "SVO pin map", "chart"))
        artifacts.append((marks.unwrap(), "svo_map.kml", "SVO KML placemarks", "map"))

    writer = make_writer(args, corpus, tool="svo_map", params=vars(args))
    for payload, name, description, kind in artifacts:
        if kind == "table":
            written = writer.write_table(payload, name, kind=kind, description=description)  # type: ignore[arg-type]
        else:
            written = writer.write_text(payload, name, kind=kind, description=description)  # type: ignore[arg-type]
        if not written.ok:
            for diag in written.diagnostics:
                print(diag, file=sys.stderr)
            return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *svos.diagnostics,
        *locs.diagnostics,
        *geo_result.diagnostics,
        *mapped.diagnostics,
        *summary.diagnostics,
        *extra_diags,
    )
    env = writer.finalize()
    if not env.ok:
        for diag in env.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    print(f"Wrote {len(summary_frame)} mapped place(s) to {writer.run_dir / 'svo_geocoded.csv'}")
    return 1 if parse_state == "partial" or geo_result.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
