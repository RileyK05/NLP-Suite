"""charts — thin CLI for publication-ready charts over an analyst CSV.

One chart per run: ``python -m tools.charts input.csv out/ --kind bar --x cat
--y n``. Data preparation semantics (aggregation, top-n, normalization, rates)
live in :mod:`core.viz.chartspec`; rendering in :mod:`core.viz.plotters`.

Every run writes, through the OutputWriter only:

* ``chart.html`` — the offline self-contained chart (CDN with ``--cdn``)
* ``chart_data.csv`` — the prepared (aggregated/scaled) data, reproducibility
* ``chart.png``/``.svg``/``.pdf`` — only when explicitly requested via
  ``--format``; a failed export fails the run but keeps HTML + CSV
* ``result.json`` — the envelope: input CSV sha256, effective parameters,
  artifacts, diagnostics

``--json`` prints exactly one JSON object (stdout) on success AND runtime
failure — ``ok``, ``run_dir``, absolute artifact paths, diagnostics — so
callers can consume runs machine-readably. Human-readable output on stderr
accompanies failures; a successful run prints the run directory (stdout,
before the JSON object only in legacy plain mode... actually: plain mode
prints the run-dir line; ``--json`` mode prints ONLY the JSON object).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.chartspec import CHART_KINDS, ChartSpec, PreparedChart

__all__ = ["main"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="charts",
        description=(
            "Publication-ready charts over a CSV: bar/line/scatter/histogram/box/heatmap plus "
            "pie/sunburst/treemap/violin/radar/waffle/calendar"
        ),
    )
    p.add_argument("input", type=Path, help="input CSV/TSV file")
    p.add_argument("output", type=Path, help="output root directory (a run dir is created inside)")
    p.add_argument(
        "--kind",
        # Derived, never re-listed: a hand-copied second list is how
        # "bubble" came to be drawable by the Excel exporter but not
        # reachable from any command line.
        choices=list(CHART_KINDS),
        default="bar",
    )
    p.add_argument("--x", required=True, help="x column (calendar: date column; heatmap: column axis)")
    p.add_argument("--y", required=True, help="y column (numeric measure; histogram: values to bin)")
    p.add_argument(
        "--group",
        default=None,
        help="group column (color/series; heatmap row axis; sunburst/treemap inner ring)",
    )
    p.add_argument(
        "--agg",
        choices=["sum", "mean", "median", "count"],
        default=None,
        help="how to combine duplicate (x, group) observations — required when duplicates exist, never a silent default",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="keep the n categories with the largest totals across groups (bar/line/pie/sunburst/treemap/waffle)",
    )
    p.add_argument("--normalize", choices=["none", "percent", "share"], default="none")
    p.add_argument(
        "--scale-by",
        choices=["none", "total", "group", "category"],
        default="total",
        help="normalization denominator level (bar/line only)",
    )
    p.add_argument(
        "--rate-per",
        type=float,
        default=None,
        help="rate multiplier (e.g. 10000 = per 10k); needs --denominator-column and --agg sum",
    )
    p.add_argument(
        "--denominator-column", default=None, help="numeric column whose per-cell total divides the numerator"
    )
    p.add_argument(
        "--bins",
        type=int,
        default=None,
        help="histogram bin count (documented control; edges stay common across groups)",
    )
    p.add_argument("--horizontal", action="store_true", help="horizontal bars (bar only)")
    p.add_argument(
        "--bar-mode",
        choices=["group", "stack", "relative"],
        default="group",
        help="grouped bar layout: group (side-by-side, default), stack or relative (explicit opt-in)",
    )
    p.add_argument("--title", default="")
    p.add_argument("--subtitle", default="")
    p.add_argument("--x-label", default="")
    p.add_argument("--y-label", default="")
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=500)
    p.add_argument("--wrap-labels", type=int, default=24, help="wrap x tick labels at this many characters")
    p.add_argument(
        "--x-tick-angle",
        type=int,
        default=None,
        help="explicit category tick rotation in degrees (e.g. -30); default: per-orientation "
        "(vertical -30, horizontal upright 0) — an explicitly supplied -30 is honored as -30",
    )
    p.add_argument("--delimiter", default=",", help="input field delimiter: , (default), ;, tab, |")
    p.add_argument("--encoding", default="utf-8-sig", help="input text encoding (default utf-8-sig)")
    p.add_argument(
        "--format",
        choices=["html", "png", "svg", "pdf", "xlsx"],
        default="html",
        help="output format; png/svg/pdf need the [plotly-image] extra (kaleido + Chrome), "
        "xlsx is a native Excel workbook ([excel] extra; bar/line/pie/scatter/radar/bubble only)",
    )
    p.add_argument(
        "--browser-path",
        default=None,
        help="per-call kaleido browser executable override (invalid path fails the export); "
        "default: BROWSER_PATH env, then kaleido auto-detection",
    )
    p.add_argument(
        "--cdn",
        action="store_true",
        help="reference plotly.js from the CDN instead of inlining it (smaller files, needs network)",
    )
    p.add_argument(
        "--json", action="store_true", help="print exactly one JSON result object on stdout (success and failure)"
    )
    return p.parse_args(argv)


def _resolve_delimiter(raw: str) -> Result[str]:
    """Named delimiters (tab) or the literal character; validated."""
    named = {"tab": "\t", "comma": ",", "semicolon": ";", "pipe": "|", "space": " "}
    value = named.get(raw.lower(), raw)
    if len(value) != 1:
        return Result.failure(
            Diagnostic.error("CHART_BAD_DELIMITER", f"--delimiter must be one character, got {raw!r}")
        )
    return Result.success(value)


def _read_input(path: Path, delimiter: str, encoding: str) -> Result[pd.DataFrame]:
    """Read the analyst CSV/TSV; every row parses or the run fails (no skipping)."""
    try:
        frame = pd.read_csv(path, sep=delimiter, encoding=encoding)
    except UnicodeDecodeError as exc:
        return Result.failure(
            Diagnostic.error(
                "CHART_READ_FAILED",
                f"could not decode {path} as {encoding}: {exc}. Try --encoding (e.g. cp1252, latin-1).",
                path=str(path),
            )
        )
    except LookupError as exc:
        # pandas passes unknown encodings to codecs.lookup -> LookupError
        return Result.failure(
            Diagnostic.error(
                "CHART_BAD_ENCODING",
                f"unknown --encoding {encoding!r}: {exc}. Use a standard codec name (utf-8, cp1252, latin-1).",
                encoding=encoding,
            )
        )
    except (OSError, ValueError) as exc:
        return Result.failure(
            Diagnostic.error(
                "CHART_READ_FAILED",
                f"could not read {path}: {exc} (ragged rows fail, never skip)",
                path=str(path),
            )
        )
    return Result.success(frame)


def _build_spec(args: argparse.Namespace) -> Result[Any]:
    """ChartSpec from CLI args; ValueError becomes a diagnostic."""

    try:
        spec = ChartSpec(
            kind=args.kind,
            x=args.x,
            y=args.y,
            group=args.group,
            agg=args.agg,
            top_n=args.top_n,
            normalize=args.normalize,
            scale_by="none" if args.scale_by == "none" else args.scale_by,
            rate_per=args.rate_per,
            denominator_column=args.denominator_column,
            horizontal=args.horizontal,
            bar_mode=args.bar_mode,
            bins=args.bins,
            title=args.title,
            subtitle=args.subtitle,
            x_label=args.x_label,
            y_label=args.y_label,
            width=args.width,
            height=args.height,
            wrap_labels=args.wrap_labels,
            x_tick_angle=args.x_tick_angle,
            offline=not args.cdn,
        )
    except ValueError as exc:
        return Result.failure(Diagnostic.error("CHART_BAD_SPEC", str(exc)))
    return Result.success(spec)


def _json_result(
    ok: bool,
    run_dir: Path | None,
    artifacts: list[dict[str, str]],
    diagnostics: list[dict[str, Any]],
) -> str:
    """Exactly one JSON object for stdout (machine-readable contract)."""
    payload = {
        "ok": ok,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "artifacts": artifacts,
        "diagnostics": diagnostics,
    }
    return json.dumps(payload, ensure_ascii=False)


def _emit_json(
    ok: bool,
    run_dir: Path | None,
    artifacts: list[dict[str, str]],
    diagnostics: list[dict[str, Any]],
) -> None:
    print(_json_result(ok, run_dir, artifacts, diagnostics))


def _print_plain(artifacts: list[dict[str, str]], run_dir: Path | None) -> None:
    for artifact in artifacts:
        print(f"Wrote {artifact['path']}")
    if run_dir is not None:
        print(f"Run directory: {run_dir}")


def _emit(
    args: argparse.Namespace,
    ok: bool,
    run_dir: Path | None,
    artifacts: list[dict[str, str]],
    diagnostics: list[dict[str, Any]],
) -> None:
    """One JSON object with --json; human lines otherwise."""
    if args.json:
        _emit_json(ok, run_dir, artifacts, diagnostics)
    else:
        for diag in diagnostics:
            print(f"[{diag.get('severity', '')}] {diag.get('code', '')}: {diag.get('message', '')}", file=sys.stderr)
        _print_plain(artifacts, run_dir)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.input.is_file():
        diagnostics = [Diagnostic.error("CHART_INPUT_MISSING", f"{args.input} is not a file", path=str(args.input))]
        _emit(args, False, None, [], _to_json_diags(diagnostics))
        return 1
    delimiter = _resolve_delimiter(args.delimiter)
    if delimiter.value is None:
        _emit(args, False, None, [], _to_json_diags(delimiter.diagnostics))
        return 1
    frame_result = _read_input(args.input, delimiter.unwrap(), args.encoding)
    if frame_result.value is None:
        _emit(args, False, None, [], _to_json_diags(frame_result.diagnostics))
        return 1
    frame = frame_result.unwrap()

    spec_result = _build_spec(args)
    if spec_result.value is None:
        _emit(args, False, None, [], _to_json_diags(spec_result.diagnostics))
        return 1
    spec = spec_result.unwrap()

    from core.viz.chartspec import coerce_date_axis, prepare_chart_data
    from core.viz.plotters import chart_image_bytes, render_chart

    # A CSV has no dtypes: date columns arrive as text. Kinds that need a real
    # date axis get one here, where the file is read.
    dated = coerce_date_axis(frame, spec)
    if dated.value is None:
        _emit(args, False, None, [], _to_json_diags(list(dated.diagnostics)))
        return 1
    prepared = prepare_chart_data(dated.unwrap(), spec)
    if prepared.value is None:
        _emit(args, False, None, [], _to_json_diags(list(prepared.diagnostics)))
        return 1
    chart = prepared.unwrap()

    rendered = render_chart(chart, spec)
    if rendered.value is None:
        _emit(args, False, None, [], _to_json_diags(list(rendered.diagnostics)))
        return 1
    html_str = rendered.unwrap()
    # Renderer warnings (fallback degradation) travel with the run everywhere.
    all_diagnostics = list(rendered.diagnostics)

    from core.io.writer import OutputWriter

    params = _effective_params(args, chart)
    try:
        writer = OutputWriter(
            args.output,
            tool="charts",
            params=params,
            inputs=[args.input],
        )
    except (ValueError, OSError) as exc:
        _emit(args, False, None, [], _to_json_diags([Diagnostic.error("CHART_WRITER_FAILED", f"writer error: {exc}")]))
        return 1

    artifacts: list[dict[str, str]] = []

    # HTML (always; survives as the successful artifact of a failed image export)
    written_html = writer.write_html(html_str, "chart.html", description=f"{spec.kind} chart {spec.y} by {spec.x}")
    if written_html.value is None:
        writer.abandon()
        _emit(args, False, None, [], _to_json_diags(list(written_html.diagnostics) + all_diagnostics))
        return 1
    artifacts.append({"kind": "chart", "path": "chart.html"})

    # Prepared data as CSV (reproducibility artifact)
    written_csv = writer.write_table(chart.data, "chart_data.csv", description="prepared chart data")
    if written_csv.value is None:
        writer.abandon()
        _emit(args, False, None, [], _to_json_diags(list(written_csv.diagnostics) + all_diagnostics))
        return 1
    artifacts.append({"kind": "table", "path": "chart_data.csv"})

    # Image export: explicitly requested formats must succeed or FAIL the run,
    # keeping the successful HTML + CSV in a failure-marked envelope.
    if args.format in ("png", "svg", "pdf"):
        image = chart_image_bytes(chart, spec, args.format, browser_path=_browser_path(args))
        if image.value is None:
            writer.add_diagnostics(*all_diagnostics)
            published = writer.publish_failure(*image.diagnostics, keep_artifacts=True)
            if published.ok:
                _emit(
                    args,
                    False,
                    writer.run_dir.resolve(),
                    _absolute_artifacts(writer, artifacts),
                    _to_json_diags(list(image.diagnostics) + all_diagnostics),
                )
            else:
                # Publication failed: abandon the staging dir so no partial run
                # lingers, and surface the publication error too. Exactly ONE
                # object: paths to abandoned files are never advertised.
                writer.abandon()
                _emit(
                    args,
                    False,
                    None,
                    [],
                    _to_json_diags(list(published.diagnostics) + list(image.diagnostics) + all_diagnostics),
                )
            return 1
        written_img = writer.write_bytes(
            image.unwrap(), f"chart.{args.format}", kind="image", description=f"{spec.kind} chart ({args.format})"
        )
        if written_img.value is None:
            writer.add_diagnostics(*all_diagnostics)
            published = writer.publish_failure(*written_img.diagnostics, keep_artifacts=True)
            if not published.ok:
                writer.abandon()
                _emit(
                    args,
                    False,
                    None,
                    [],
                    _to_json_diags(list(published.diagnostics) + list(written_img.diagnostics) + all_diagnostics),
                )
            else:
                _emit(
                    args,
                    False,
                    writer.run_dir.resolve(),
                    _absolute_artifacts(writer, artifacts),
                    _to_json_diags(list(written_img.diagnostics) + list(published.diagnostics) + all_diagnostics),
                )
            return 1
        artifacts.append({"kind": "image", "path": f"chart.{args.format}"})

    # Excel export: a native openpyxl workbook (legacy-parity visuals).
    # Explicitly requested, so failure marks the run but keeps HTML + CSV.
    if args.format == "xlsx":
        from core.viz.charts_excel import excel_chart_bytes

        workbook = excel_chart_bytes(chart, spec)
        if workbook.value is None:
            writer.add_diagnostics(*all_diagnostics)
            published = writer.publish_failure(*workbook.diagnostics, keep_artifacts=True)
            if published.ok:
                _emit(
                    args,
                    False,
                    writer.run_dir.resolve(),
                    _absolute_artifacts(writer, artifacts),
                    _to_json_diags(list(workbook.diagnostics) + all_diagnostics),
                )
            else:
                writer.abandon()
                _emit(
                    args,
                    False,
                    None,
                    [],
                    _to_json_diags(list(published.diagnostics) + list(workbook.diagnostics) + all_diagnostics),
                )
            return 1
        written_xlsx = writer.write_bytes(
            workbook.unwrap(), "chart.xlsx", kind="table", description=f"{spec.kind} Excel chart (native workbook)"
        )
        if written_xlsx.value is None:
            writer.add_diagnostics(*all_diagnostics)
            published = writer.publish_failure(*written_xlsx.diagnostics, keep_artifacts=True)
            if not published.ok:
                writer.abandon()
                _emit(
                    args,
                    False,
                    None,
                    [],
                    _to_json_diags(list(published.diagnostics) + list(written_xlsx.diagnostics) + all_diagnostics),
                )
            else:
                _emit(
                    args,
                    False,
                    writer.run_dir.resolve(),
                    _absolute_artifacts(writer, artifacts),
                    _to_json_diags(list(written_xlsx.diagnostics) + list(published.diagnostics) + all_diagnostics),
                )
            return 1
        artifacts.append({"kind": "table", "path": "chart.xlsx"})

    writer.add_diagnostics(*all_diagnostics)
    env = writer.finalize()
    if env.value is None:
        # Finalization failed: clean the staging dir immediately (the caller
        # cannot consume it) and emit exactly one object with both the
        # finalization and run diagnostics — never paths to abandoned files.
        writer.abandon()
        _emit(args, False, None, [], _to_json_diags(list(env.diagnostics) + all_diagnostics))
        return 1
    # Absolute paths are emitted only AFTER finalize (staging has been renamed).
    absolute = _absolute_artifacts(writer, artifacts)
    _emit(args, True, writer.run_dir.resolve(), absolute, _to_json_diags(all_diagnostics))
    return 0


def _absolute_artifacts(writer: Any, artifacts: list[dict[str, str]]) -> list[dict[str, str]]:
    """Every artifact path resolved against the (final) run dir, absolute."""
    run_dir = Path(writer.run_dir).resolve()
    return [{"kind": a["kind"], "path": str((run_dir / a["path"]).resolve())} for a in artifacts]


def _browser_path(args: argparse.Namespace) -> str | None:
    """Per-call kaleido browser override: --browser-path wins, then env."""
    import os

    return args.browser_path or os.environ.get("BROWSER_PATH") or os.environ.get("KALEIDO_CHROME_PATH")


def _effective_params(args: argparse.Namespace, chart: PreparedChart) -> dict[str, Any]:
    """The effective parameters + prepared metadata recorded in the envelope."""
    spec = chart.spec  # typed ChartSpec (PreparedChart always carries it)
    return {
        "kind": spec.kind,
        "x": spec.x,
        "y": spec.y,
        "group": spec.group,
        "agg": spec.agg,
        "top_n": spec.top_n,
        "normalize": spec.normalize,
        "scale_by": spec.scale_by,
        "rate_per": spec.rate_per,
        "denominator_column": spec.denominator_column,
        "horizontal": spec.horizontal,
        "bar_mode": spec.bar_mode,
        "bins": args.bins,
        "delimiter": args.delimiter,
        "encoding": args.encoding,
        "format": args.format,
        "cdn": args.cdn,
        "title": args.title,
        "subtitle": args.subtitle,
        "x_label": args.x_label,
        "y_label": args.y_label,
        "width": args.width,
        "height": args.height,
        "wrap_labels": args.wrap_labels,
        "x_tick_angle": args.x_tick_angle,
        "offline": not args.cdn,
        "browser_path": _browser_path(args),
        "prepared": dict(chart.prepared_by),
        "effective_labels": {"x": chart.x_label, "y": chart.y_label},
    }


def _to_json_diags(diagnostics: list[Diagnostic] | tuple[Diagnostic, ...]) -> list[dict[str, Any]]:
    return [d.to_dict() for d in diagnostics]


if __name__ == "__main__":
    raise SystemExit(main())
