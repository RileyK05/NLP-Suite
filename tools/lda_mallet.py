"""lda_mallet — thin CLI for MALLET LDA topics and document-topic composition.

The other half of the syllabus's LDA comparison (with ``lda_gensim``). MALLET
is a Java binary outside this suite; a missing binary fails loudly with the
install pointer rather than substituting another backend, because "compare
MALLET with Gensim" must not become "compare Gensim with itself".

Staging happens here, in the CLI, and never in ``core/``: the write-custody
gate (R3) says analysis modules write nothing, so the run directory receives
only parsed tables through the OutputWriter — MALLET's own scratch files live
in a temporary directory that dies with the process.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from core.analysis.mallet import parse_doc_topics, train_topics
from core.io.reader import read_corpus
from core.result import Diagnostic
from tools._cli import build_common_parser, make_writer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Topic models (MALLET LDA, needs the MALLET binary)")
    p.add_argument("--topics", type=int, default=10)
    p.add_argument("--seed", type=int, default=42, help="MALLET random seed (recorded in the envelope)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus_result = read_corpus(args.corpus)
    if corpus_result.value is None:
        for d in corpus_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    for d in corpus_result.diagnostics:
        print(d, file=sys.stderr)
    corpus = corpus_result.unwrap()
    writer = make_writer(args, corpus, tool="lda_mallet", params=vars(args))
    # MALLET reads one directory of .txt files. The documents already have
    # their text in hand; stage copies in a temporary directory so nothing is
    # written beside the samples and nothing lands in the run directory but
    # the parsed tables (R3).
    with tempfile.TemporaryDirectory(prefix="mallet-in-") as staged:
        stage = Path(staged)
        for position, doc in enumerate(corpus.docs, 1):
            (stage / f"{position:04d}_{doc.label or doc.path.name}.txt").write_text(doc.text, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="mallet-out-") as out:
            keys = train_topics(stage, out, n_topics=args.topics, seed=args.seed)
            if not keys.ok:
                writer.add_diagnostics(*keys.diagnostics)
                writer.finalize()
                for d in keys.diagnostics:
                    print(d, file=sys.stderr)
                    fix = d.context.get("fix") if isinstance(d.context, dict) else None
                    if fix:
                        print(f"fix: {fix}", file=sys.stderr)
                return 1
            dominant = parse_doc_topics(Path(out) / "mallet_doc_topics.txt")
            written = writer.write_table(keys.unwrap(), "topics.csv", kind="table", description="MALLET topic keys")
            if not written.ok:
                for d in written.diagnostics:
                    print(d, file=sys.stderr)
                return 1
            if dominant.ok:
                written = writer.write_table(
                    dominant.unwrap(),
                    "topics_dominant.csv",
                    kind="table",
                    description="MALLET document-topic composition",
                )
                if not written.ok:
                    for d in written.diagnostics:
                        print(d, file=sys.stderr)
                    return 1
            else:
                # Keys alone are still a result; the composition is the part
                # that could not be read, and it says so in the envelope.
                writer.add_diagnostics(*dominant.diagnostics)
            writer.add_diagnostics(
                *keys.diagnostics,
                *dominant.diagnostics,
                Diagnostic.info("MALLET_SEED", f"MALLET random seed {args.seed}", seed=args.seed),
            )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote MALLET topic keys to {writer.run_dir / 'topics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
