"""Synthetic chart gallery + review-manifest generator (visual review fixtures).

Builds one small synthetic dataset (no corpus data) and produces a run
directory per chart case with the same artifacts the ``charts`` CLI writes
(chart.html + chart_data.csv), through the real preparation and rendering
path. With ``--png`` (and plotly+kaleido importable) it also exports real
PNG images per case and writes ``manifest.json``.

Offline by default; deterministic by construction (seed 42). Run directly:

    python scripts/make_chart_gallery.py out/visual_review/gallery
    python scripts/make_chart_gallery.py out/visual_review/review --png
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.io.writer import OutputWriter  # noqa: E402
from core.result import Result  # noqa: E402
from core.viz.chartspec import ChartSpec, prepare_chart_data  # noqa: E402
from core.viz.plotters import chart_image_bytes, render_chart  # noqa: E402

__all__ = ["main", "make_gallery", "review_specs"]


def _synthetic_frame() -> pd.DataFrame:
    """A deterministic synthetic dataset covering all chart kinds."""
    rng = np.random.default_rng(42)
    categories = [f"Category {chr(65 + i)}" for i in range(6)]
    groups = ["Group 1", "Group 2", "Group 3"]
    rows: list[dict[str, object]] = []
    for i, category in enumerate(categories):
        for g, group in enumerate(groups):
            base = 10.0 + i * 2.0 + g * 1.5
            for _rep in range(3):
                rows.append(
                    {
                        "category": category,
                        "group": group,
                        "value": round(base + float(rng.normal(0.0, 1.0)), 4),
                        "year": 2000 + (i * len(groups) + g) % 25,
                        "tokens": 1000 + i * 100,
                    }
                )
    return pd.DataFrame(rows)


def _gallery_specs() -> list[tuple[str, ChartSpec]]:
    """The six kinds over the synthetic frame (semantics per chartspec)."""
    return [
        (
            "bar",
            ChartSpec(
                kind="bar",
                x="category",
                y="value",
                group="group",
                agg="mean",
                title="Mean value by category",
                subtitle="synthetic gallery data (seed 42)",
            ),
        ),
        (
            "line",
            ChartSpec(
                kind="line",
                x="year",
                y="value",
                group="group",
                agg="mean",
                title="Value trend by year",
                subtitle="numeric axis stays chronological",
            ),
        ),
        (
            "scatter",
            ChartSpec(
                kind="scatter", x="year", y="value", title="Observations", subtitle="one point per row, numeric axis"
            ),
        ),
        (
            "histogram",
            ChartSpec(
                kind="histogram",
                x="category",
                y="value",
                title="Value distribution",
                subtitle="common bin edges across the whole dataset",
            ),
        ),
        (
            "box",
            ChartSpec(
                kind="box",
                x="category",
                y="value",
                title="Value spread by category",
                subtitle="all observations, unaggregated",
            ),
        ),
        (
            "heatmap",
            ChartSpec(
                kind="heatmap",
                x="category",
                y="value",
                group="group",
                agg="mean",
                title="Category by group heatmap",
                subtitle="missing cells stay missing",
            ),
        ),
    ]


def review_specs() -> list[tuple[str, pd.DataFrame, ChartSpec]]:
    """Senior-review cases: axis correctness + comparison layout + labels."""
    rng = np.random.default_rng(7)
    # horizontal: categories A/B, counts 20/80 (review §1 numeric ticks)
    horizontal = pd.DataFrame({"cat": ["A", "B"], "n": [20.0, 80.0]})
    # grouped bars side-by-side default + explicit stack
    grouped = _synthetic_frame()
    # percent normalization (effective label must be Percent)
    percent = pd.DataFrame({"cat": ["A", "B", "C"], "n": [5.0, 3.0, 8.0]})
    # rate: mentions per 10k tokens
    rate = pd.DataFrame(
        {
            "cat": ["A", "A", "B", "B"],
            "grp": ["S1", "S2", "S1", "S2"],
            "mentions": [3.0, 2.0, 5.0, 1.0],
            "tokens": [100.0, 100.0, 300.0, 200.0],
        }
    )
    # long labels (wrapping)
    long_labels = pd.DataFrame(
        {
            "term": [f"Very long category label number {i:02d}" for i in range(1, 5)],
            "n": [10.0, 30.0, 20.0, 40.0],
        }
    )
    # sparse/interleaved groups (review §1 tick mapping)
    sparse = pd.DataFrame(
        {
            "cat": ["A", "C", "E", "A", "C", "E"],
            "grp": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "n": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    # grouped histograms (opacity overlay)
    hist_groups = pd.DataFrame(
        {
            "grp": ["A"] * 20 + ["B"] * 20,
            "val": [float(rng.normal(5.0, 1.0)) for _ in range(20)] + [float(rng.normal(8.0, 1.5)) for _ in range(20)],
        }
    )
    return [
        (
            "horizontal_bar",
            horizontal,
            ChartSpec(
                kind="bar",
                x="cat",
                y="n",
                horizontal=True,
                title="Horizontal bar: numeric ticks on x",
                subtitle="A/B ticks on the y axis only",
            ),
        ),
        (
            "grouped_bar_sidebyside",
            grouped,
            ChartSpec(
                kind="bar",
                x="category",
                y="value",
                group="group",
                agg="mean",
                title="Grouped bars (side-by-side default)",
                subtitle="stacking means is never implied",
            ),
        ),
        (
            "grouped_bar_stack",
            grouped,
            ChartSpec(
                kind="bar",
                x="category",
                y="value",
                group="group",
                agg="mean",
                bar_mode="stack",
                title="Grouped bars (explicit stack opt-in)",
                subtitle="stack means only when that is the analysis",
            ),
        ),
        (
            "percent_bar",
            percent,
            ChartSpec(
                kind="bar",
                x="cat",
                y="n",
                normalize="percent",
                title="Percent normalization",
                subtitle="y label must say Percent, not the column name",
            ),
        ),
        (
            "rate_bar",
            rate,
            ChartSpec(
                kind="bar",
                x="cat",
                y="mentions",
                group="grp",
                agg="sum",
                rate_per=10000.0,
                denominator_column="tokens",
                title="Mentions per 10k tokens",
                subtitle="rate label on the measure axis",
            ),
        ),
        (
            "long_labels",
            long_labels,
            ChartSpec(
                kind="bar",
                x="term",
                y="n",
                title="Wrapped long labels",
                subtitle="rendering wraps; data identity unchanged",
            ),
        ),
        (
            "sparse_groups",
            sparse,
            ChartSpec(
                kind="bar",
                x="cat",
                y="n",
                group="grp",
                agg="mean",
                title="Sparse interleaved groups",
                subtitle="tick mapping aligned across traces",
            ),
        ),
        (
            "grouped_histogram",
            hist_groups,
            ChartSpec(
                kind="histogram",
                x="grp",
                y="val",
                group="grp",
                title="Grouped histogram",
                subtitle="common edges, overlay opacity",
            ),
        ),
    ]


def _write_case(out_root: Path, name: str, frame: pd.DataFrame, spec: ChartSpec) -> Result[str]:
    """One run per review case: chart.html + chart_data.csv (+ chart.png)."""
    prepared = prepare_chart_data(frame, spec)
    if prepared.value is None:
        return Result[str](None, prepared.diagnostics)
    chart = prepared.unwrap()
    rendered = render_chart(chart, spec)
    if rendered.value is None:
        return Result[str](None, rendered.diagnostics)
    params: dict[str, Any] = {"kind": spec.kind, "x": spec.x, "y": spec.y, "_case": name}
    writer = OutputWriter(out_root, tool=f"review_{name}", params=params, inputs=[])
    html_written = writer.write_html(rendered.unwrap(), "chart.html", description=f"{name} review chart")
    if html_written.value is None:
        writer.abandon()
        return Result[str](None, html_written.diagnostics)
    csv_written = writer.write_table(chart.data, "chart_data.csv", description=f"{name} prepared data")
    if csv_written.value is None:
        writer.abandon()
        return Result[str](None, csv_written.diagnostics)
    env = writer.finalize()
    if env.value is None:
        return Result[str](None, env.diagnostics)
    return Result.success(str(writer.run_dir))


def make_gallery(out_root: Path) -> Result[Path]:
    """Write one run per chart kind; returns the out root on success."""
    out_root = Path(out_root)
    frame = _synthetic_frame()
    for name, spec in _gallery_specs():
        result = _write_case(out_root, name, frame, spec)
        if result.value is None:
            return Result[Path](None, result.diagnostics)
    return Result.success(out_root)


def make_review_images(out_root: Path) -> Result[Path]:
    """Render + export the senior-review cases (PNG via kaleido) + manifest.

    Requires kaleido on the import path (the isolated out/visual_review/deps
    per the review); the HTML/CSV artifacts are written regardless, and the
    manifest records per-case export status honestly.
    """
    out_root = Path(out_root)
    manifest: list[dict[str, Any]] = []
    try:
        import kaleido  # type: ignore[import-not-found]  # optional [plotly-image] extra

        export_available = True
    except ImportError:
        export_available = False

    for name, frame, spec in review_specs():
        result = _write_case(out_root, name, frame, spec)
        if result.value is None:
            return Result[Path](None, result.diagnostics)
        run_dir = result.unwrap()
        entry: dict[str, Any] = {"case": name, "run_dir": run_dir}
        if export_available:
            prepared = prepare_chart_data(frame, spec)
            if prepared.value is None:
                return Result[Path](None, prepared.diagnostics)
            chart = prepared.unwrap()
            image = chart_image_bytes(chart, spec, "png", browser_path=None)
            entry["png_export"] = "ok" if image.ok else f"failed: {image.errors[0].code}"
            if image.ok:
                # Write the PNG through the writer and FINALIZE so the file
                # survives in its published run dir (never a staging dir).
                from core.io.writer import OutputWriter

                img_writer = OutputWriter(out_root, tool=f"review_{name}_png", params={}, inputs=[])
                written = img_writer.write_bytes(image.unwrap(), f"{name}.png", kind="image", description=name)
                if written.value is None:
                    img_writer.abandon()
                    entry["png_export"] = f"write failed: {written.errors[0].code}"
                    entry["png_bytes"] = 0
                else:
                    env = img_writer.finalize()
                    if env.value is None:
                        img_writer.abandon()
                        entry["png_export"] = f"finalize failed: {env.errors[0].code}"
                        entry["png_bytes"] = 0
                    else:
                        entry["png_path"] = str((img_writer.run_dir / f"{name}.png").resolve())
                        entry["png_bytes"] = len(image.unwrap())
        else:
            entry["png_export"] = "kaleido unavailable"
            entry["png_bytes"] = 0
        manifest.append(entry)
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return Result.success(manifest_path)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    with_png = "--png" in args
    args = [a for a in args if a != "--png"]
    target = Path(args[0]) if args else Path("out") / "gallery"
    result = make_review_images(target) if with_png else make_gallery(target)
    if not result.ok:
        for diag in result.diagnostics:
            print(diag, file=sys.stderr)
        return 1
    print(f"Gallery written to {result.unwrap()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
