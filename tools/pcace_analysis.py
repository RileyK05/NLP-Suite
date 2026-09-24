"""pcace_analysis — thin CLI for PC-ACE analysis/validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PC-ACE analysis")
    p.add_argument("output", type=Path, help="output root")
    p.add_argument("--db", type=Path, default=None, help="pcace db to analyze")
    p.add_argument("--codes", nargs="+", default=None, help="codes to validate")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from core.io.writer import OutputWriter
    from core.pcace.analysis import analyze, validate_codes

    from tools._cli import write_or_die

    try:
        writer = OutputWriter(args.output, tool="pcace_analysis", params=vars(args), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    wrote = False
    if args.codes:
        res = validate_codes(args.codes)
        if not res.ok:
            for d in res.diagnostics:
                print(d, file=sys.stderr)
            return 1
        write_or_die(
            writer, writer.write_table(res.unwrap(), "validation.csv", kind="table", description="code validation")
        )
        writer.add_diagnostics(*res.diagnostics)
        wrote = True

    if args.db is not None:
        res = analyze(args.db)
        if not res.ok:
            for d in res.diagnostics:
                print(d, file=sys.stderr)
            return 1
        write_or_die(
            writer, writer.write_table(res.unwrap(), "analysis.csv", kind="table", description="per-category analysis")
        )
        writer.add_diagnostics(*res.diagnostics)
        wrote = True

    if not wrote:
        # Default: write grammar
        from core.pcace.core import grammar_info

        write_or_die(writer, writer.write_table(grammar_info(), "grammar.csv", kind="table", description="grammar"))

    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
