"""visualizations — what this suite can draw, and which are legacy.

The original NLP Suite's Excel workbooks and standalone HTML charts are still
what a lot of coursework and review expects, so they are carried over rather
than replaced. This lists every visualization with its provenance, so
"give me the legacy set" and "what do we add on top" are both one command.

    nlp-suite visualizations                    # everything
    nlp-suite visualizations --origin legacy    # only what the old suite drew
    nlp-suite visualizations --origin new       # only what it could not
    nlp-suite visualizations --name wordcloud   # full detail on one
"""

from __future__ import annotations

import argparse
import json
import sys

from core.viz.catalog import (
    LEGACY_VISUALIZATION_MODULES,
    VISUALIZATIONS,
    Origin,
    Visualization,
    get_visualization,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List the visualizations this suite can produce")
    parser.add_argument(
        "--origin",
        choices=["legacy", "legacy-port", "legacy-extended", "new"],
        default="",
        help="filter by provenance ('legacy' covers both ported and extended)",
    )
    parser.add_argument("--name", default="", help="show full detail for one visualization")
    parser.add_argument("--tool", default="", help="only visualizations produced by this tool")
    parser.add_argument("--format", dest="fmt", default="", help="only visualizations producing this format")
    parser.add_argument("--legacy-map", action="store_true", help="show the legacy module coverage map")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser.parse_args(argv)


def _selected(args: argparse.Namespace) -> list[Visualization]:
    chosen = list(VISUALIZATIONS)
    if args.origin == "legacy":
        chosen = [v for v in chosen if v.origin.is_legacy]
    elif args.origin:
        chosen = [v for v in chosen if str(v.origin) == args.origin]
    if args.tool:
        chosen = [v for v in chosen if v.tool == args.tool]
    if args.fmt:
        chosen = [v for v in chosen if args.fmt in v.produces]
    return chosen


def _as_dict(visualization: Visualization) -> dict[str, object]:
    return {
        "name": visualization.name,
        "title": visualization.title,
        "origin": str(visualization.origin),
        "is_legacy": visualization.origin.is_legacy,
        "tool": visualization.tool,
        "module": visualization.module,
        "produces": list(visualization.produces),
        "summary": visualization.summary,
        "legacy_module": visualization.legacy_module,
        "legacy_note": visualization.legacy_note,
        "advantage": visualization.advantage,
    }


def _print_detail(visualization: Visualization) -> None:
    print(f"{visualization.title} ({visualization.name})")
    print(f"  origin    : {visualization.origin}")
    print(f"  produces  : {', '.join(visualization.produces)}")
    print(f"  tool      : nlp-suite {visualization.tool}")
    print(f"  module    : {visualization.module}")
    print(f"  summary   : {visualization.summary}")
    if visualization.legacy_module:
        print(f"  legacy    : {visualization.legacy_module}")
        print(f"              {visualization.legacy_note}")
    if visualization.advantage:
        print(f"  adds      : {visualization.advantage}")


def main(argv: list[str] | None = None) -> int:
    from tools._cli import make_console_encoding_tolerant

    make_console_encoding_tolerant()
    args = _parse_args(argv)

    if args.legacy_map:
        carried = {k: v for k, v in LEGACY_VISUALIZATION_MODULES.items() if not v.startswith("dropped:")}
        dropped = {k: v for k, v in LEGACY_VISUALIZATION_MODULES.items() if v.startswith("dropped:")}
        if args.json:
            print(json.dumps({"carried": carried, "dropped": dropped}, ensure_ascii=False))
            return 0
        print(f"Legacy visualization modules carried over ({len(carried)}):")
        for module, target in sorted(carried.items()):
            print(f"  {module:<44} -> {target}")
        print(f"\nNot carried ({len(dropped)}), each for a stated reason:")
        for module, reason in sorted(dropped.items()):
            print(f"  {module:<44} {reason.removeprefix('dropped: ')}")
        return 0

    if args.name:
        visualization = get_visualization(args.name)
        if visualization is None:
            print(f"visualizations: unknown visualization {args.name!r}", file=sys.stderr)
            print("run 'nlp-suite visualizations' to see them all", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(_as_dict(visualization), ensure_ascii=False))
            return 0
        _print_detail(visualization)
        return 0

    chosen = _selected(args)
    if args.json:
        print(json.dumps([_as_dict(v) for v in chosen], ensure_ascii=False))
        return 0
    if not chosen:
        print("No visualization matches that filter.")
        return 0
    legacy = sum(1 for v in chosen if v.origin.is_legacy)
    print(f"{len(chosen)} visualization(s): {legacy} from the original suite, {len(chosen) - legacy} new.\n")
    for visualization in chosen:
        print(f"  {visualization.describe()}")
    print("\nRun 'nlp-suite visualizations --name NAME' for provenance and what each adds.")
    if any(v.origin is Origin.LEGACY_EXTENDED for v in chosen):
        print("'legacy-extended' draws the original picture and adds to it; the addition is named there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
