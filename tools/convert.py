"""convert — thin CLI converting a directory of documents to intake text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core.file_ops.converter import convert_document_to_text
from core.result import Diagnostic
from tools._cli import write_or_die


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert documents (txt/csv/tsv/html) to intake text")
    p.add_argument("input", type=Path, help="input directory of documents")
    p.add_argument("output", type=Path, help="output root directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.input.is_dir():
        print(f"not a directory: {args.input}", file=sys.stderr)
        return 2
    if not any(args.input.iterdir()):
        print(f"input directory is empty: {args.input}", file=sys.stderr)
        return 2

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(args.output, tool="convert", params={}, inputs=(args.input,), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    used_names: set[str] = set()
    source_n = 0
    for path in sorted(p for p in args.input.iterdir() if p.is_file()):
        source_n += 1
        result = convert_document_to_text(path)
        if result.value is None:
            for d in result.diagnostics:
                # Batch conversion is best-effort: malformed optional files
                # are warnings while valid files still produce a usable run.
                warning = Diagnostic.warning(d.code, d.message, **dict(d.context))
                print(warning, file=sys.stderr)
                writer.add_diagnostics(warning)
            rows.append(
                {"File": path.name, "Status": "FAILED", "Detail": "; ".join(d.code for d in result.diagnostics)}
            )
            continue
        # C6-10: collision-safe output names keep the source suffix in the
        # stem so a.txt + a.html never collapse onto one output.
        candidate = f"{path.stem}_{path.suffix.lstrip('.') or 'txt'}.txt"
        counter = 2
        while candidate in used_names:
            candidate = f"{path.stem}_{path.suffix.lstrip('.') or 'txt'}_{counter}.txt"
            counter += 1
        used_names.add(candidate)
        write_or_die(
            writer,
            writer.write_text(result.unwrap(), candidate, kind="text", description=f"Converted {path.name}"),
        )
        rows.append({"File": path.name, "Status": "OK", "Detail": candidate})
    write_or_die(
        writer,
        writer.write_table(
            pd.DataFrame(rows, columns=["File", "Status", "Detail"]),
            "conversion_report.csv",
            kind="table",
            description="Per-file conversion report",
        ),
    )
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    ok_n = sum(1 for row in rows if row["Status"] == "OK")
    print(f"Converted {ok_n}/{source_n} file(s) to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
