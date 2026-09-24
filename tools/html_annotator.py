"""html_annotator — thin CLI for extract + dictionary annotate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core.analysis.html_annotator import annotate, extract_text
from core.result import Diagnostic, Severity
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HTML annotator (extract + dictionary)")
    p.add_argument("input", type=Path, help="input: HTML file or directory of .html")
    p.add_argument("output", type=Path, help="output root directory")
    p.add_argument("--dict", dest="dict_path", type=Path, default=None, help="JSON dictionary {term: tag}")
    p.add_argument("--case-sensitive", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = args.input
    output_root = args.output

    # Collect HTML files
    if input_path.is_file():
        html_files = [input_path]
    elif input_path.is_dir():
        html_files = sorted(input_path.rglob("*.html")) + sorted(input_path.rglob("*.htm"))
        if not html_files:
            print(f"no HTML files under {input_path}", file=sys.stderr)
            return 1
    else:
        print(f"{input_path} is not a file or directory", file=sys.stderr)
        return 1

    dictionary: dict[str, str] = {}
    if args.dict_path is not None:
        try:
            dictionary = json.loads(args.dict_path.read_text(encoding="utf-8"))
            if not isinstance(dictionary, dict):
                raise ValueError("dictionary JSON must be an object")
        except Exception as exc:
            print(f"could not read dictionary {args.dict_path}: {exc}", file=sys.stderr)
            return 1

    from core.io.reader import Corpus, Document, corpus_fingerprint, hash_text, read_text
    from core.io.writer import OutputWriter

    docs: list[Document] = []
    input_diagnostics: list[Diagnostic] = []
    for idx, p in enumerate(html_files, start=1):
        text_result = read_text(p)
        input_diagnostics.extend(text_result.diagnostics)
        if text_result.value is None:
            for d in text_result.diagnostics:
                print(d, file=sys.stderr)
            continue
        raw = text_result.unwrap()
        docs.append(Document(doc_id=idx, path=p, text=raw, sha256=hash_text(raw)))

    if not docs:
        print("no documents to process", file=sys.stderr)
        return 1

    corpus = Corpus(docs=tuple(docs), sha256=corpus_fingerprint(tuple(docs)))

    try:
        writer = OutputWriter(output_root, tool="html_annotator", params=vars(args), corpus=corpus)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    for doc in docs:
        ext = extract_text(doc.text)
        input_diagnostics.extend(ext.diagnostics)
        if ext.value is None:
            for d in ext.diagnostics:
                print(d, file=sys.stderr)
            continue
        plain = ext.unwrap()
        if dictionary:
            ann = annotate(plain, dictionary, case_sensitive=args.case_sensitive)
            input_diagnostics.extend(ann.diagnostics)
            if ann.ok:
                plain = ann.unwrap()
            else:
                for d in ann.diagnostics:
                    print(d, file=sys.stderr)
        # Extracted plain text is a report, not a table — a "table" kind made
        # the viewer try to pd.read_csv a prose file.
        write_or_die(
            writer,
            writer.write_text(plain, f"{doc.path.stem}.txt", kind="report", description=f"extracted {doc.path.name}"),
        )

    # Also emit a summary CSV
    summary = pd.DataFrame(
        [{"File": d.path.name, "Chars": len(d.text)} for d in docs],
        columns=["File", "Chars"],
    )
    write_or_die(writer, writer.write_table(summary, "summary.csv", kind="table", description="HTML files summary"))

    writer.add_diagnostics(*input_diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(docs)} file(s) to {writer.run_dir}")
    return 1 if any(d.severity is Severity.ERROR for d in input_diagnostics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
