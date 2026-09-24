"""filenames — thin CLI for preview-first filename standardization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.file_ops.filenames import apply_renames, plan_renames


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview (default) or apply standardized renames")
    p.add_argument("directory", type=Path, help="directory of files to standardize")
    p.add_argument("--apply", action="store_true", help="actually rename (default is preview only)")
    p.add_argument("--dry-run", action="store_true", help="explicit preview (the default)")
    p.add_argument("--suffix", default=".txt", help="file suffix to consider")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 2
    files = sorted(p for p in args.directory.iterdir() if p.is_file() and p.suffix == args.suffix)
    plan_result = plan_renames(files)
    if not plan_result.ok:
        for d in plan_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    plan = plan_result.unwrap()
    print(plan.to_frame().to_string(index=False))
    if not args.apply:
        print(f"preview only ({len(plan.to_frame())} file(s)); pass --apply to rename")
        return 0
    applied_result = apply_renames(plan, dry_run=False)
    if not applied_result.ok:
        for d in applied_result.diagnostics:
            print(d, file=sys.stderr)
        return 1
    for diag in applied_result.diagnostics:
        print(diag, file=sys.stderr)
    for old, new in applied_result.unwrap():
        print(f"renamed: {old} -> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
