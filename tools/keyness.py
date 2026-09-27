"""keyness — thin CLI for G2 log-likelihood keyness between document groups."""

from __future__ import annotations

import argparse
import sys

from core.analysis.keyness import detail_groups, keyness
from core.conll.schema import Col
from core.io.filename_fields import apply_template, detect_template
from core.io.reader import read_corpus
from tools._cli import handle_result_states, build_common_parser, get_pipeline, make_writer, resolve_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Keyness (G2 log-likelihood) between two document groups")
    p.add_argument(
        "--group-field",
        default="",
        help="a detail read from the file names (Kind, Speaker) that names the groups; used instead of --group-pattern",
    )
    p.add_argument("--group-a", default="", help="the detail's value(s) for group A, comma-separated")
    p.add_argument("--group-b", default="", help="the detail's value(s) for group B; empty = every other document")
    p.add_argument(
        "--group-pattern",
        default="",
        help="regex over document names; matching docs form group A, the rest group B",
    )
    p.add_argument("--field", choices=["form", "lemma"], default="lemma")
    p.add_argument("--smoothing", type=float, default=0.5, help="Log Ratio smoothing (Hardie 0.5)")
    p.add_argument("--top-n", type=int, default=200, help="rows kept (0 = all)")
    args = p.parse_args(argv)
    # A usage error, before any parsing: naming neither kind of group is a
    # mistake in the command, not something a corpus can answer.
    if args.group_field and not args.group_a:
        p.error("--group-field needs --group-a (the detail's value for group A)")
    if not args.group_field and not args.group_pattern:
        p.error("say which documents are group A: --group-field with --group-a, or --group-pattern")
    return args


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
    groups, labels = None, ("Group A", "Group B")
    if args.group_field:
        # A folder has no stored details; its file names are read as the app reads them.
        names = [doc.path.name for doc in corpus.docs]
        template = detect_template(names)
        details = {name: apply_template(name, template) if template else {} for name in names}
        chosen = detail_groups(details, args.group_field, args.group_a, args.group_b)
        if chosen.value is None:
            for d in chosen.diagnostics:
                print(d, file=sys.stderr)
            return 1
        groups, labels = chosen.unwrap()
    result = keyness(
        table,
        args.group_pattern,
        groups=groups,
        labels=labels,
        field=field,
        smoothing=args.smoothing,
        top_n=args.top_n,
    )
    if not result.ok:
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = result.unwrap()
    writer = make_writer(args, corpus, tool="keyness", params=vars(args))
    written = writer.write_table(frame, "keyness.csv", kind="table", description="G2 keyness")
    if not written.ok:
        for d in written.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *result.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} keyness row(s) to {writer.run_dir / 'keyness.csv'}")
    return 1 if parse_state == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
