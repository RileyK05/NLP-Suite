"""freeze_legacy — print a frozen record of the legacy oracle directory.

The record goes to stdout; the operator redirects it into
docs/LEGACY_ENVIRONMENT.md and commits. Exit 0 on success, 2 when the
oracle directory itself is unreadable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.legacy_freeze import freeze_oracle

_DEFAULT_ORACLE = Path(__file__).resolve().parent.parent.parent / "NLP-Suite-1.6.38"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze the legacy oracle into a reproducible record")
    p.add_argument("--oracle", type=Path, default=_DEFAULT_ORACLE, help="legacy oracle directory")
    p.add_argument(
        "--working-tree",
        choices=("clean", "dirty"),
        default=None,
        help="operator attestation from running git status by hand",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = freeze_oracle(args.oracle, working_tree=args.working_tree or "")
    if result.value is None:
        for diag in result.diagnostics:
            print(diag)
        return 2
    print(result.unwrap().to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
