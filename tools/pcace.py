"""pcace — thin CLI for PC-ACE core."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PC-ACE core")
    p.add_argument("output", type=Path, help="output root (db will be output/pcace.db)")
    p.add_argument("--codes", nargs="+", default=None, help="PC-ACE codes to insert")
    p.add_argument("--db", type=Path, default=None, help="existing db to read")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from core.io.writer import OutputWriter
    from core.pcace.core import create_db, grammar_info, read_db

    from tools._cli import write_or_die

    try:
        writer = OutputWriter(args.output, tool="pcace", params=vars(args), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    if args.db is not None:
        res = read_db(args.db)
        if not res.ok:
            for d in res.diagnostics:
                print(d, file=sys.stderr)
            return 1
        write_or_die(writer, writer.write_table(res.unwrap(), "pcace.csv", kind="table", description="pcace from db"))
    elif args.codes:
        db_path = writer.run_dir / "pcace.db"
        res = create_db(db_path, args.codes)
        if not res.ok:
            for d in res.diagnostics:
                print(d, file=sys.stderr)
            return 1
        # Also write the grammar
        write_or_die(writer, writer.write_table(grammar_info(), "grammar.csv", kind="table", description="grammar"))
        r2 = read_db(db_path)
        if r2.ok:
            write_or_die(writer, writer.write_table(r2.unwrap(), "pcace.csv", kind="table", description="pcace codes"))
        write_or_die(writer, writer.write_text(str(db_path), "db_path.txt", kind="report", description="db location"))
    else:
        # No codes nor db -> just write grammar
        write_or_die(writer, writer.write_table(grammar_info(), "grammar.csv", kind="table", description="grammar"))

    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote PC-ACE to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
