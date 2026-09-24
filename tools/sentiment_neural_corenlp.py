"""sentiment_neural_corenlp — thin CLI for neural sentiment via the CoreNLP server.

The CoreNLP sentiment annotator (the RNTN) runs in the Java server and does
its own tokenize/split, so this tool needs no local parse — mirroring
``tools/lda_mallet.py``, which is the other tool that talks to a Stanford
binary rather than to the shared parse. A missing server fails loudly with
the start command: "compare four annotators" must never become "compare one
annotator with itself".
"""

from __future__ import annotations

import argparse
import sys

from core.analysis.sentiment_neural import corenlp_sentences, summarize_neural
from core.io.reader import read_corpus
from core.result import Diagnostic
from tools._cli import build_common_parser, make_writer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_common_parser("Neural sentiment (Stanford CoreNLP sentiment annotator, needs the Java server)")
    p.add_argument("--server", default="http://localhost:9000")
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
    docs = [(str(doc.doc_id), doc.label or doc.path.name, doc.text) for doc in corpus.docs]
    result = corenlp_sentences(docs, server_url=args.server)
    writer = make_writer(args, corpus, tool="sentiment_neural_corenlp", params=vars(args))
    if result.value is None:
        writer.add_diagnostics(*result.diagnostics)
        writer.finalize()
        for d in result.diagnostics:
            print(d, file=sys.stderr)
            fix = d.context.get("fix") if isinstance(d.context, dict) else None
            if fix:
                print(f"fix: {fix}", file=sys.stderr)
        return 1
    sentences = result.unwrap()
    summary = summarize_neural(sentences)
    for frame, name, description in (
        (sentences, "sentiment_sentences.csv", "neural sentiment per sentence"),
        (summary.unwrap(), "sentiment_documents.csv", "neural sentiment per document"),
    ):
        written = writer.write_table(frame, name, kind="table", description=description)
        if not written.ok:
            for d in written.diagnostics:
                print(d, file=sys.stderr)
            return 1
    writer.add_diagnostics(
        *corpus_result.diagnostics,
        *result.diagnostics,
        *summary.diagnostics,
        Diagnostic.info(
            "SENTIMENT_NN_BACKEND",
            f"backend corenlp (RNTN) at {args.server}",
            backend="corenlp",
            server=args.server,
        ),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(sentences)} sentence score(s) to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
