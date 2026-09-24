"""corpus_validation — thin CLI for corpus checker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corpus validation")
    p.add_argument("corpus", type=Path, help="corpus dir")
    p.add_argument("output", type=Path, help="output root")
    p.add_argument("--min-tokens", type=int, default=1)
    p.add_argument("--ext", default=".txt")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from core.data.validation import validate
    from core.io.writer import OutputWriter

    res = validate(args.corpus, min_tokens=args.min_tokens, required_ext=args.ext)
    if not res.ok:
        for d in res.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = res.unwrap()
    from core.io.reader import read_corpus

    # Hash the same files the validator looked at: the corpus reader's
    # default pattern is *.txt, but --ext may select any suffix. The
    # auxiliary read is a provenance step, not the product: when it merely
    # finds no matching files the validator's own VALIDATE_EMPTY report is
    # the truthful record, and an ERROR diagnostic must not ride a
    # green-exit envelope.
    pattern = f"*{args.ext}" if args.ext else "*"
    corpus_res = read_corpus(args.corpus, pattern=pattern)
    corpus = corpus_res.unwrap() if corpus_res.ok else None
    corpus_unreadable = not corpus_res.ok and any(d.code != "CORPUS_EMPTY" for d in corpus_res.errors)
    if corpus is None and not corpus_unreadable:
        corpus_res = read_corpus(args.corpus)
        corpus = corpus_res.unwrap() if corpus_res.ok else None
        corpus_unreadable = not corpus_res.ok and any(d.code != "CORPUS_EMPTY" for d in corpus_res.errors)
    try:
        # OutputWriter refuses output roots inside the corpus directory
        # (ARCHITECTURE.md R3). That is a hard stop, not something to retry
        # with provenance stripped off — an envelope with no inputs proves
        # nothing about what was validated.
        writer = OutputWriter(
            args.output,
            tool="corpus_validation",
            params=vars(args),
            inputs=() if corpus is not None else (args.corpus,),
            corpus=corpus,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1
    write_or_die(writer, writer.write_table(frame, "validation.csv", kind="table", description="corpus validation"))
    if corpus is not None:
        writer.add_diagnostics(*corpus_res.diagnostics, *res.diagnostics)
    else:
        writer.add_diagnostics(*res.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} file(s) validation to {writer.run_dir / 'validation.csv'}")
    return 1 if corpus is None and corpus_unreadable else 0


if __name__ == "__main__":
    raise SystemExit(main())
