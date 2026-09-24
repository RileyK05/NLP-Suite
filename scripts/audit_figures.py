"""Audit every figure over real results: lint, render, and a contact sheet.

    python scripts/audit_figures.py RUNS_OR_FIXTURES [--out out/figure-audit] [--only NAME ...]

RUNS_OR_FIXTURES is either a folder of finished runs (any folder containing
run directories with a ``result.json``, e.g. a project's ``runs/``) or the
test fixtures (``tests/fixtures/figures``, files named ``tool__table.parquet``).

For every registered panel with a table to draw from, and every choice of
every parameter, it builds the figure and runs the content checks
(``core/viz/figure_lint``); for the defaults it renders the publication
figure and runs the drawing checks (``core/viz/static/lint``). It writes
``index.html`` -- one thumbnail per figure with its findings beside it -- so
the figures can be looked at, not just counted. Lint finds geometry; only a
person finds "this chart tells you nothing".

Writes only under --out. Needs matplotlib and seaborn.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
#: A figure slower than this to draw is worth a look.
SLOW_SECONDS = 5
sys.path.insert(0, str(ROOT))

from core.viz.figure_lint import lint_definition, lint_prepared  # noqa: E402
from core.viz.panels import PANELS, best_table, fit_to_rows  # noqa: E402
from core.viz.panelspec import PanelDefinition, Provenance  # noqa: E402
from core.viz.static import render_static  # noqa: E402


def load_tables(source: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Tables by tool, from run directories or from ``tool__table.parquet``."""
    by_tool: dict[str, dict[str, pd.DataFrame]] = {}
    for parquet in sorted(source.glob("*.parquet")):
        tool, _, table = parquet.stem.partition("__")
        by_tool.setdefault(tool, {})[f"{table}.csv"] = pd.read_parquet(parquet)
    for envelope in sorted(source.rglob("result.json")):
        try:
            tool = json.loads(envelope.read_text(encoding="utf-8"))["tool"]
        except (OSError, ValueError, KeyError):
            continue
        for csv in sorted(envelope.parent.glob("*.csv")):
            if csv.name == "input_files.csv":
                continue
            try:
                by_tool.setdefault(tool, {}).setdefault(csv.name, pd.read_csv(csv, encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
    return by_tool


def variants(definition: PanelDefinition) -> list[dict[str, object]]:
    out: list[dict[str, object]] = [{}]
    for param in definition.params:
        if param.type == "choice":
            out += [{param.name: choice} for choice in param.choices if choice != param.default]
        elif param.type == "bool":
            out.append({param.name: not param.default})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "figure-audit")
    parser.add_argument("--only", nargs="*", default=[], help="panel names, tools or shapes")
    args = parser.parse_args()
    tables = load_tables(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    total = 0
    for definition in PANELS:
        if args.only and not {definition.name, definition.tool, definition.shape} & set(args.only):
            continue
        frames = tables.get(definition.tool, {})
        chosen = best_table(
            definition, [(name, list(frame.columns)) for name, frame in frames.items() if not frame.empty]
        )
        if chosen is None:
            continue
        frame = frames[chosen]
        findings = [str(f) for f in lint_definition(definition)]
        for variant in variants(definition):
            params = definition.defaults() | variant
            built = definition.build(
                frame, params, Provenance(tool=definition.tool, panel=definition.name, source=chosen)
            )
            if built.value is None:
                findings.append(
                    f"{variant or 'defaults'}: refused: {'; '.join(d.message for d in built.diagnostics)[:200]}"
                )
                continue
            findings += [
                f"{variant or 'defaults'}: {f}"
                for f in lint_prepared(fit_to_rows(built.unwrap()), definition, params, diagnostics=built.diagnostics)
            ]
        image_name = ""
        built = definition.build(
            frame, definition.defaults(), Provenance(tool=definition.tool, panel=definition.name, source=chosen)
        )
        if built.value is not None:
            start = time.perf_counter()
            image = render_static(fit_to_rows(built.unwrap()), "png", dpi=90)
            seconds = time.perf_counter() - start
            if image.value is not None:
                image_name = f"{definition.name}.png"
                (args.out / image_name).write_bytes(image.unwrap())
            findings += [f"drawing: {d.message}" for d in image.diagnostics if d.code.startswith("STATIC_")]
            if seconds > SLOW_SECONDS:
                findings.append(f"drawing: slow ({seconds:.1f}s)")
        total += len(findings)
        rows.append((definition, chosen, image_name, findings))
        print(f"{'OK ' if not findings else 'FIX'} {definition.name} ({len(findings)} finding(s))")
    cards = []
    for definition, chosen, image_name, findings in rows:
        items = "".join(f"<li>{html.escape(f)}</li>" for f in findings) or "<li class='ok'>no findings</li>"
        picture = (
            f"<a href='{image_name}'><img src='{image_name}' loading='lazy'></a>" if image_name else "<p>not drawn</p>"
        )
        cards.append(
            f"<section><h2>{html.escape(definition.title)}</h2><p class='meta'>{definition.name} · "
            f"{definition.shape} · from {html.escape(chosen)}</p><p>{html.escape(definition.question)}</p>"
            f"{picture}<ul>{items}</ul></section>"
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Figure audit</title><style>"
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f6f7f4;color:#2b3329}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}"
        "section{background:#fff;border:1px solid #e3e7df;border-radius:8px;padding:12px}"
        "img{width:100%;border:1px solid #e3e7df}h2{font-size:15px;margin:0}.meta{color:#616d5b;font-size:12px}"
        "li{font-size:12px;color:#8a2b1d}li.ok{color:#2e6b3a}</style>"
        f"<h1>Figure audit</h1><p>{len(rows)} figures, {total} finding(s), from {html.escape(str(args.source))}</p>"
        f"<main>{''.join(cards)}</main>"
    )
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"\n{len(rows)} figures, {total} finding(s). Contact sheet: {args.out / 'index.html'}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
