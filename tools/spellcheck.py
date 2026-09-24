"""spellcheck — thin CLI for wordlist-explicit spell checking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.analysis.spellcheck import correct_text, load_wordlist, run, validate_threshold
from core.io.reader import read_corpus
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spell-check a corpus against an explicit wordlist")
    p.add_argument("corpus", type=Path, help="corpus directory of .txt files")
    p.add_argument("output", type=Path, help="output root directory")
    p.add_argument("--wordlist", type=Path, required=True, help="wordlist file, one word per line")
    p.add_argument("--threshold", type=float, default=80.0, help="suggestion threshold 0-100")
    p.add_argument("--correct", action="store_true", help="also write corrected copies (inputs untouched)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    threshold_problem = validate_threshold(args.threshold)
    if threshold_problem is not None:
        print(threshold_problem, file=sys.stderr)
        return 2
    vocab_result = load_wordlist(args.wordlist)
    if vocab_result.value is None:
        for d in vocab_result.diagnostics:
            print(d, file=sys.stderr)
        return 2
    vocab = vocab_result.unwrap()

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

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(
            args.output,
            tool="spellcheck",
            params={"threshold": args.threshold, "correct": args.correct},
            # The wordlist is a named resource file, not corpus content:
            # recorded for provenance, but its folder stays writable.
            inputs=(args.wordlist,),
            corpus=corpus,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    result = run(corpus, vocab, args.threshold)
    if result.value is None:
        writer.abandon()
        for d in result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    writer.add_diagnostics(*corpus_result.diagnostics, *result.diagnostics)
    findings = result.unwrap().to_frame()
    write_or_die(
        writer,
        writer.write_table(findings, "spellcheck.csv", kind="table", description="Unknown words with suggestions"),
    )
    if args.correct:
        used: set[str] = set()
        for doc in corpus.docs:
            findings_for_doc = findings[findings["Document"] == doc.path.as_posix()]
            corrected = correct_text(doc.text, findings_for_doc)
            # C6-12: collision-safe artifact names (duplicate basenames).
            candidate = f"corrected_{doc.path.name}"
            counter = 2
            while candidate in used:
                candidate = f"corrected_{doc.path.stem}_{counter}{doc.path.suffix}"
                counter += 1
            used.add(candidate)
            write_or_die(
                writer,
                writer.write_text(corrected, candidate, kind="text", description=f"Corrected copy of {doc.path}"),
            )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(findings)} finding(s) to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
