"""wordcloud_gephi — thin CLI for wordcloud and GEXF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from core.viz.wordcloud_gephi import gephi_gexf, wordcloud_html, wordcloud_image


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wordcloud and Gephi GEXF")
    p.add_argument("input", type=Path, help="input CSV")
    p.add_argument("output", type=Path, help="output root")
    p.add_argument("--word-col", required=True)
    p.add_argument("--weight-col", required=True)
    p.add_argument("--source-col", default=None, help="for GEXF source")
    p.add_argument("--target-col", default=None, help="for GEXF target")
    p.add_argument("--title", default="")
    p.add_argument(
        "--image",
        action="store_true",
        help="also render a raster PNG wordcloud (needs the [wordcloud] extra)",
    )
    p.add_argument("--mask", type=Path, default=None, help="image mask: words render in non-white regions")
    p.add_argument(
        "--shape",
        default=None,
        help="procedural mask shape instead of --mask (available: butterfly)",
    )
    p.add_argument("--group-col", default=None, help="color words by this column (with --colors)")
    p.add_argument(
        "--colors",
        default="",
        help="group=hexcolor pairs for --group-col, e.g. 'Subject=#ff0000,Verb=#0000ff'",
    )
    p.add_argument("--max-words", type=int, default=200, help="maximum words in the cloud (image mode)")
    p.add_argument("--width", type=int, default=800, help="image width in px")
    p.add_argument("--height", type=int, default=800, help="image height in px")
    p.add_argument("--colormap", default="viridis", help="matplotlib colormap (image mode, no --group-col)")
    p.add_argument("--background", default="white", help="background color (image mode)")
    p.add_argument("--seed", type=int, default=42, help="layout RNG seed (image mode)")
    return p.parse_args(argv)


def _parse_colors(spec: str) -> dict[str, str]:
    """'Subject=#ff0000,Verb=#0000ff' -> {'Subject': '#ff0000', ...}."""
    colors: dict[str, str] = {}
    for pair in spec.split(","):
        if "=" in pair:
            name, _, value = pair.partition("=")
            if name.strip() and value.strip():
                colors[name.strip()] = value.strip()
    return colors


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.input.is_file():
        print(f"{args.input} is not a file", file=sys.stderr)
        return 1
    try:
        df = pd.read_csv(args.input, encoding="utf-8")
    except Exception as exc:
        print(f"could not read {args.input}: {exc}", file=sys.stderr)
        return 1

    from core.io.writer import OutputWriter

    from tools._cli import write_or_die

    try:
        writer = OutputWriter(args.output, tool="wordcloud_gephi", params=vars(args), inputs=[args.input])
    except (ValueError, FileExistsError) as exc:
        print(f"writer error: {exc}", file=sys.stderr)
        return 1

    wc = wordcloud_html(df, word_col=args.word_col, weight_col=args.weight_col, title=args.title)
    if not wc.ok:
        for d in wc.diagnostics:
            print(d, file=sys.stderr)
        return 1
    write_or_die(writer, writer.write_html(wc.unwrap(), "wordcloud.html", description="wordcloud (HTML)"))

    if args.image:
        img = wordcloud_image(
            df,
            word_col=args.word_col,
            weight_col=args.weight_col,
            output_path=writer.run_dir / "wordcloud.png",
            width=args.width,
            height=args.height,
            max_words=args.max_words,
            background_color=args.background,
            colormap=args.colormap,
            mask_path=args.mask,
            shape=args.shape,
            group_col=args.group_col,
            group_colors=_parse_colors(args.colors) if args.group_col else None,
            seed=args.seed,
        )
        if not img.ok:
            writer.abandon()
            from tools._cli import exit_with_diagnostics

            exit_with_diagnostics(img.diagnostics, code=1)
        # The wordcloud package writes the PNG itself (R3: register, not re-write).
        write_or_die(writer, writer.register_artifact("wordcloud.png", kind="image", description="wordcloud (PNG)"))

    if args.source_col and args.target_col:
        gex = gephi_gexf(df, source_col=args.source_col, target_col=args.target_col, weight_col=args.weight_col)
        if gex.ok:
            # GEXF is a graph file, not a table — a "table" kind made the
            # viewer try to parse XML as CSV.
            write_or_die(writer, writer.write_text(gex.unwrap(), "graph.gexf", kind="report", description="GEXF"))
        else:
            for d in gex.diagnostics:
                print(d, file=sys.stderr)
            return 1

    writer.add_diagnostics(*wc.diagnostics)
    env = writer.finalize()
    if not env.ok:
        for d in env.diagnostics:
            print(d, file=sys.stderr)
        return 1
    print(f"Wrote wordcloud to {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
