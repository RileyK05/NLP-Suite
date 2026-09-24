"""install — thin CLI for environment check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Install check")
    p.add_argument("output", type=Path, help="output root")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from core.install.check import check_environment
    from core.io.writer import OutputWriter

    from tools._cli import write_or_die

    res = check_environment()
    if not res.ok:
        for d in res.diagnostics:
            print(d, file=sys.stderr)
        return 1
    frame = res.unwrap()
    try:
        writer = OutputWriter(args.output, tool="install", params=vars(args), corpus=None)
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1
    write_or_die(writer, writer.write_table(frame, "environment.csv", kind="table", description="environment check"))
    # Also write a markdown summary
    lines = ["# Install check", "", "| Extra | Module | Installed | Python |", "|---|---|---|---|"]
    for _, row in frame.iterrows():
        lines.append(f"| {row['Extra']} | {row['Module']} | {row['Installed']} | {row['Python']} |")
    write_or_die(writer, writer.write_text("\n".join(lines) + "\n", "INSTALL.md", description="install markdown"))
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote to {writer.run_dir}")
    for _, row in frame.iterrows():
        status = "OK" if row["Installed"] else "MISSING"
        print(f"  {row['Extra']}/{row['Module']}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
