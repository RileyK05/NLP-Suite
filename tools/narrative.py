"""narrative — thin CLI for emotion and length arcs."""

from __future__ import annotations

import argparse
import sys

from core.conll.schema import Col
from core.io.reader import read_corpus
from core.narrative.arcs import emotion_arc, length_arc
from core.narrative.characters import character_arcs, character_mentions
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Narrative arcs")
    p.add_argument("--field", choices=["form", "lemma"], default="form")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    field = Col.FORM if args.field == "form" else Col.LEMMA
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
    e_res = emotion_arc(table, field=field)
    if not e_res.ok:
        for d in e_res.diagnostics:
            print(d, file=sys.stderr)
        return 1
    l_res = length_arc(table)
    if not l_res.ok:
        for d in l_res.diagnostics:
            print(d, file=sys.stderr)
        return 1
    m_res = character_mentions(table)
    if not m_res.ok:
        for d in m_res.diagnostics:
            print(d, file=sys.stderr)
        return 1
    c_res = character_arcs(table, field=field)
    if not c_res.ok:
        for d in c_res.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer = make_writer(args, corpus, tool="narrative", params=vars(args))
    e_frame = e_res.unwrap()
    l_frame = l_res.unwrap()
    for frame, name, description in (
        (e_frame, "emotion_arc.csv", "emotion arc per sentence"),
        (l_frame, "length_arc.csv", "length arc per sentence"),
        (m_res.unwrap(), "character_mentions.csv", "character mentions"),
        (c_res.unwrap(), "character_arcs.csv", "character emotion arcs"),
    ):
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *table_result.diagnostics,
        *e_res.diagnostics,
        *l_res.diagnostics,
        *m_res.diagnostics,
        *c_res.diagnostics,
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(e_frame)} emotion + {len(l_frame)} length rows to {writer.run_dir}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
