"""assets — thin CLI reporting registry status (seed for FR-9.3 management)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.assets.registry import default_registry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report data-asset status")
    p.add_argument("--root", type=Path, default=None, help="asset root (default: repo assets/)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    registry = default_registry(args.root)
    table = registry.status_table().unwrap()
    print(table.to_string(index=False))
    bad_required = table[(table["Required"]) & (table["Status"] != "OK")]
    if not bad_required.empty:
        print(f"{len(bad_required)} required asset(s) not usable", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
