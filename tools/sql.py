"""sql — thin CLI for safe sqlite queries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SQL safe layer")
    p.add_argument("db", type=Path, help="sqlite db path")
    p.add_argument("output", type=Path, help="output root")
    p.add_argument("--query", required=True, help="SQL (use ? placeholders)")
    p.add_argument("--params", nargs="*", default=[], help="params for ? placeholders")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from core.data.sql import query
    from core.io.writer import OutputWriter

    from tools._cli import write_or_die

    res = query(args.db, args.query, tuple(args.params) if args.params else None)
    if not res.ok:
        for d in res.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = res.unwrap()
    try:
        writer = OutputWriter(args.output, tool="sql", params=vars(args), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1
    write_or_die(writer, writer.write_table(frame, "result.csv", kind="table", description="query result"))
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote {len(frame)} row(s) to {writer.run_dir / 'result.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
