"""Generate final blocker evidence: real PNGs through the real charts CLI.

Run with the isolated kaleido on the path:

    $env:PYTHONPATH = "<abs>/out/visual_review/deps;$env:PYTHONPATH"
    python scripts/make_final_blocker_evidence.py

Writes runs (HTML + CSV + PNG, one per case) under
out/visual_review/final_blockers/ plus manifest.json recording the ACTUAL
figure metadata (axis titles, categoryarray, heatmap coordinates) read back
from the rendered figures — evidence, not claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.io.writer import OutputWriter  # noqa: E402
from core.result import Result  # noqa: E402
from core.viz.chartspec import ChartSpec, prepare_chart_data  # noqa: E402
from core.viz.plotters import build_figure, chart_image_bytes, render_chart  # noqa: E402

OUT_ROOT = ROOT / "out" / "visual_review" / "final_blockers"


def _sparse() -> pd.DataFrame:
    """The genuinely sparse reproducer from the second review."""
    return pd.DataFrame(
        {"c": ["A", "C", "E", "B", "D", "F"], "g": ["one"] * 3 + ["two"] * 3, "n": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]}
    )


def _numeric_heat() -> pd.DataFrame:
    return pd.DataFrame(
        {"c": [1, 2, 10, 1, 2, 10], "g": ["one"] * 3 + ["two"] * 3, "n": [1.0, 2.0, 10.0, 11.0, 12.0, 20.0]}
    )


def _datetime_heat() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": pd.to_datetime(["2024-03-01", "2024-01-01", "2024-03-01", "2024-01-01"]),
            "role": ["R1", "R1", "R2", "R2"],
            "hits": [2.0, 1.0, 4.0, 3.0],
        }
    )


def _gappy_heat() -> pd.DataFrame:
    return pd.DataFrame({"doc": ["d1", "d1", "d2"], "role": ["R1", "R2", "R1"], "hits": [1.0, 2.0, 3.0]})


def _hist_values() -> pd.DataFrame:
    return pd.DataFrame({"grp": ["A", "A", "A", "A", "B", "B"], "val": [1.0, 2.0, 3.0, 4.0, 9.0, 10.0]})


def cases() -> list[tuple[str, pd.DataFrame, ChartSpec]]:
    return [
        (
            "sparse_bar_vertical",
            _sparse(),
            ChartSpec(
                kind="bar",
                x="c",
                y="n",
                group="g",
                title="Genuinely sparse groups: every tick must carry its own value",
                subtitle="ticktext A..F synchronized with categoryarray (value 3 sits under C)",
            ),
        ),
        (
            "sparse_bar_horizontal",
            _sparse(),
            ChartSpec(
                kind="bar",
                x="c",
                y="n",
                group="g",
                horizontal=True,
                title="Sparse groups, horizontal: upright category labels",
                subtitle="numeric measure axis keeps numeric ticks + measure label",
            ),
        ),
        (
            "histogram_counts",
            _hist_values(),
            ChartSpec(
                kind="histogram",
                x="grp",
                y="val",
                group="grp",
                title="Histogram: x is the VALUE column, y is Count",
                subtitle="the category selector grp never labels an axis",
            ),
        ),
        (
            "histogram_percent",
            _hist_values(),
            ChartSpec(
                kind="histogram",
                x="grp",
                y="val",
                group="grp",
                normalize="percent",
                title="Histogram, percent normalization",
                subtitle="y must read Percent (was the raw column name)",
            ),
        ),
        (
            "heatmap_categorical_gaps",
            _gappy_heat(),
            ChartSpec(
                kind="heatmap",
                x="doc",
                y="hits",
                group="role",
                title="Categorical heatmap: y names the ROW axis, colorbar names hits",
                subtitle="missing (d2, R2) stays a gap",
            ),
        ),
        (
            "heatmap_numeric",
            _numeric_heat(),
            ChartSpec(
                kind="heatmap",
                x="c",
                y="n",
                group="g",
                agg="sum",
                title="Numeric heatmap: x coordinates 1, 2, 10 (dtype-true)",
                subtitle="go.Heatmap; string sorting previously collapsed 6 values into a broken axis",
            ),
        ),
        (
            "heatmap_datetime",
            _datetime_heat(),
            ChartSpec(
                kind="heatmap",
                x="day",
                y="hits",
                group="role",
                title="Datetime heatmap: chronological x coordinates",
                subtitle="no imshow resampling; datetime ticks stay dates",
            ),
        ),
    ]


def _figure_metadata(prepared: Any, spec: ChartSpec) -> dict[str, Any]:
    """The ACTUAL rendered metadata read back from the built figure."""
    fig = build_figure(prepared, spec).unwrap()
    info: dict[str, Any] = {"trace_types": [t.type for t in fig.data]}
    if spec.kind == "heatmap":
        trace = fig.data[0]
        info["x_coordinates"] = [str(v) for v in trace.x]
        info["y_coordinates"] = [str(v) for v in trace.y]
        info["z"] = [[None if v is None or pd.isna(v) else float(v) for v in row] for row in trace.z]
        info["xaxis_title"] = fig.layout.xaxis.title.text
        info["yaxis_title"] = fig.layout.yaxis.title.text
        info["colorbar_title"] = trace.colorbar.title.text
    elif spec.kind == "histogram":
        info["xaxis_title"] = fig.layout.xaxis.title.text
        info["yaxis_title"] = fig.layout.yaxis.title.text
    else:
        axis = fig.layout.yaxis if spec.horizontal else fig.layout.xaxis
        info["xaxis_title"] = fig.layout.xaxis.title.text if not spec.horizontal else axis.title.text
        info["yaxis_title"] = fig.layout.yaxis.title.text
        info["ticktext"] = list(axis.ticktext or [])
        info["categoryorder"] = axis.categoryorder
        info["categoryarray"] = list(axis.categoryarray or ())
        info["tickangle"] = axis.tickangle
        # rendered mapping: each trace's drawn category -> tick index
        position = {str(c): i for i, c in enumerate(axis.categoryarray or ())}
        mapping: dict[str, list[dict[str, Any]]] = {}
        for trace in fig.data:
            entries = []
            xs = trace.x if not spec.horizontal else trace.y
            ys = trace.y if not spec.horizontal else trace.x
            for category, value in zip(xs, ys, strict=True):
                entries.append({"tick_index": position.get(str(category)), "value": float(value)})
            mapping[str(trace.name)] = entries
        info["rendered_mapping"] = mapping
    return info


def main() -> int:
    import shutil

    # Previous evidence runs are preserved (senior inspection history); each
    # regeneration writes NEW timestamped run dirs and rewrites the manifest
    # from those fresh runs only. A unique staging subdir keeps the temp
    # manifest work isolated from any older content.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    try:
        import kaleido  # type: ignore[import-not-found]  # optional extra

        export_available = True
    except ImportError:
        export_available = False

    for name, frame, spec in cases():
        prepared_result = prepare_chart_data(frame, spec)
        if not prepared_result.ok:
            print(f"PREPARE FAILED {name}: {prepared_result.errors}", file=sys.stderr)
            return 1
        chart = prepared_result.unwrap()
        rendered = render_chart(chart, spec)
        if not rendered.ok:
            print(f"RENDER FAILED {name}: {rendered.errors}", file=sys.stderr)
            return 1
        params = {"kind": spec.kind, "x": spec.x, "y": spec.y, "group": spec.group, "_case": name}
        writer = OutputWriter(OUT_ROOT, tool=f"final_{name}", params=params, inputs=[])
        html_ok = writer.write_html(rendered.unwrap(), "chart.html", description=name)
        csv_ok = writer.write_table(chart.data, "chart_data.csv", description=f"{name} prepared data")
        if not (html_ok.ok and csv_ok.ok):
            writer.abandon()
            print(f"WRITE FAILED {name}", file=sys.stderr)
            return 1
        entry: dict[str, Any] = {
            "case": name,
            "effective_labels": {"x": chart.x_label, "y": chart.y_label},
            "figure": _figure_metadata(chart, spec),
        }
        env = writer.finalize()
        if not env.ok:
            # Finalize failed: the staging dir is abandoned (gone or stale) —
            # record the failure honestly, never a path that will not exist.
            writer.abandon()
            entry["run_dir"] = None
            entry["finalize"] = f"failed: {env.errors[0].code}"
            manifest.append(entry)
            print(f"FINALIZE FAILED {name}: {env.errors}", file=sys.stderr)
            return 1
        # Only a PUBLISHED (renamed) run dir goes into the manifest, absolute.
        entry["run_dir"] = str(writer.run_dir.resolve())
        if export_available:
            image = chart_image_bytes(chart, spec, "png")
            if image.ok:
                img_writer = OutputWriter(OUT_ROOT, tool=f"final_{name}_png", params={}, inputs=[])
                written = img_writer.write_bytes(image.unwrap(), f"{name}.png", kind="image", description=name)
                env2 = img_writer.finalize() if written.ok else None
                if written.ok and env2 is not None and env2.ok:
                    entry["png_path"] = str((img_writer.run_dir / f"{name}.png").resolve())
                    entry["png_bytes"] = len(image.unwrap())
                else:
                    img_writer.abandon()
                    entry["png_export"] = "write/finalize failed"
            else:
                entry["png_export"] = f"failed: {image.errors[0].code}"
        else:
            entry["png_export"] = "kaleido unavailable"
        manifest.append(entry)
    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps({"export_available": export_available, "cases": manifest}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # The manifest must be self-verifying: every recorded run_dir exists and
    # carries its envelope, every png_path exists. A stale path here is a
    # bug in this script, not a consumer's problem.
    problems: list[str] = []
    for entry in manifest:
        run_dir = entry.get("run_dir")
        if run_dir is None:
            problems.append(f"{entry['case']}: run_dir never published")
            continue
        run_path = Path(run_dir)
        if not run_path.is_dir():
            problems.append(f"{entry['case']}: run_dir missing on disk: {run_dir}")
        elif not (run_path / "result.json").is_file():
            problems.append(f"{entry['case']}: no result.json in {run_dir}")
        png_path = entry.get("png_path")
        if png_path is not None and not Path(png_path).is_file():
            problems.append(f"{entry['case']}: png_path missing on disk: {png_path}")
    if problems:
        for problem in problems:
            print(f"MANIFEST INVALID: {problem}", file=sys.stderr)
        return 1
    print(f"Wrote {manifest_path} ({len(manifest)} cases, all paths verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
