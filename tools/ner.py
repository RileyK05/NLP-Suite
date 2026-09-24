"""ner — thin CLI for NER timeline + location tracking + movement tracks."""

from __future__ import annotations

import argparse
import sys

from core.analysis.movement import movement_summary, movement_tracks
from core.analysis.ner import entity_timeline, location_tracking
from core.io.reader import read_corpus
from core.result import Diagnostic
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("NER — entity timeline + location tracking + movement")
    p.add_argument("--movement", action="store_true", help="also write person-location movement tracks")
    p.add_argument("--geocode", action="store_true", help="add Lat/Lon to movement tracks (offline KB)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    corpus_result = read_corpus(args.corpus)
    # A bad document is a diagnostic, not an aborted run: the reader returns
    # the good documents with ERROR diagnostics; only a total failure (value
    # is None) stops the tool. Partial corpora flow into the four-state
    # writer policy, matching tools/_cli.load_corpus.
    if corpus_result.value is None:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    if corpus_result.diagnostics:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
    corpus = corpus_result.unwrap()

    from core.pipelines.cache import PipelineCache

    cache = PipelineCache()
    pipeline = get_pipeline(cache, config, args)
    table_result = pipeline.parse(corpus)
    parse_state = handle_result_states(table_result)
    table = table_result.unwrap()

    tl = entity_timeline(table)
    if not tl.ok:
        for d in tl.diagnostics:
            print(d, file=sys.stderr)
        return 1
    loc = location_tracking(table)
    if not loc.ok:
        for d in loc.diagnostics:
            print(d, file=sys.stderr)
        return 1

    writer = make_writer(args, corpus, tool="ner", params=vars(args))
    tl_frame = tl.unwrap()
    loc_frame = loc.unwrap()

    w1 = writer.write_table(tl_frame, "entity_timeline.csv", kind="table", description="NER timeline")
    if not w1.ok:
        for d in w1.diagnostics:
            print(d, file=sys.stderr)
        return 1
    w2 = writer.write_table(loc_frame, "locations.csv", kind="table", description="location tracking")
    if not w2.ok:
        for d in w2.diagnostics:
            print(d, file=sys.stderr)
        return 1

    movement_diags: tuple[Diagnostic, ...] = ()
    if args.movement:
        tracks = movement_tracks(table, geocode=args.geocode)
        writer.add_diagnostics(*tracks.diagnostics)
        if tracks.value is not None:
            pairs = tracks.unwrap()
            w3 = writer.write_table(pairs, "movement_tracks.csv", kind="table", description="person-location pairs")
            if not w3.ok:
                for d in w3.diagnostics:
                    print(d, file=sys.stderr)
                return 1
            w4 = writer.write_table(
                movement_summary(pairs), "movement_summary.csv", kind="table", description="entity-location counts"
            )
            if not w4.ok:
                for d in w4.diagnostics:
                    print(d, file=sys.stderr)
                return 1
            movement_diags = tracks.diagnostics
        else:
            for d in tracks.diagnostics:
                print(d, file=sys.stderr)
            return 1

    writer.add_diagnostics(
        *corpus_result.diagnostics, *table_result.diagnostics, *tl.diagnostics, *loc.diagnostics, *movement_diags
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    printed = f"Wrote {len(tl_frame)} timeline + {len(loc_frame)} location rows to {writer.run_dir}"
    if args.movement:
        printed += " (+ movement tracks)"
    print(printed)
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
