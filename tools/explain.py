"""explain — read a result table back, and say which charts are worth drawing.

Point it at any CSV this suite produced (or any CSV at all) and it prints a
plain-language reading plus a short list of recommended charts. With
``--check X Y`` it answers the narrower question "is this pairing worth
plotting?", which is the one that stops a corpus-size-against-corpus-size
chart being drawn in the first place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from core.insight.readout import readout
from core.insight.recommend import degenerate_reasons, recommend_charts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a result table and recommend charts")
    parser.add_argument("table", type=Path, help="CSV file to read")
    parser.add_argument("--tool", default="", help="tool that produced it (sharpens the reading)")
    parser.add_argument("--limit", type=int, default=5, help="how many chart recommendations to list")
    parser.add_argument("--check", nargs=2, metavar=("X", "Y"), help="ask whether charting X against Y is worthwhile")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--encoding", default="utf-8-sig", help="input encoding")
    parser.add_argument("--delimiter", default=",", help="input delimiter")
    return parser.parse_args(argv)


def _infer_tool(path: Path) -> str:
    """A run directory is named ``<tool>__<timestamp>``; the file is <tool>.csv."""
    stem = path.stem
    parent = path.parent.name
    if "__" in parent:
        return parent.split("__", 1)[0]
    return stem


def main(argv: list[str] | None = None) -> int:
    from tools._cli import make_console_encoding_tolerant

    make_console_encoding_tolerant()
    args = _parse_args(argv)
    try:
        frame = pd.read_csv(args.table, encoding=args.encoding, sep=args.delimiter)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"explain: could not read {args.table}: {exc}", file=sys.stderr)
        return 1

    tool = args.tool or _infer_tool(args.table)

    if args.check:
        x, y = args.check
        reasons = degenerate_reasons(frame, x, y)
        if args.json:
            print(json.dumps({"x": x, "y": y, "worthwhile": not reasons, "reasons": reasons}, ensure_ascii=False))
            return 0 if not reasons else 3
        if reasons:
            print(f"Charting {x!r} against {y!r} is not worth drawing:")
            for reason in reasons:
                print(f"  - {reason}")
            return 3
        print(f"Charting {x!r} against {y!r} is reasonable: the two columns vary independently.")
        return 0

    reading = readout(frame, tool=tool)
    recommendations = recommend_charts(frame, tool=tool, limit=args.limit)
    if args.json:
        print(
            json.dumps(
                {
                    "tool": tool,
                    "headline": reading.headline,
                    "observations": list(reading.observations),
                    "cautions": list(reading.cautions),
                    "recommended_charts": [
                        {
                            "kind": r.spec.kind,
                            "x": r.spec.x,
                            "y": r.spec.y,
                            "question": r.question,
                            "why": r.why,
                        }
                        for r in recommendations
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0

    print(reading.to_text())
    if recommendations:
        print()
        print("Charts worth drawing:")
        for recommendation in recommendations:
            command = (
                f"nlp-suite charts {args.table} OUT --kind {recommendation.spec.kind} "
                f"--x {recommendation.spec.x!r} --y {recommendation.spec.y!r}"
            )
            print(f"  - {recommendation.question}")
            print(f"    {recommendation.why}")
            print(f"    {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
