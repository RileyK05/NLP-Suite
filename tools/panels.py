"""panels — draw a tool-specific figure over an analysis CSV.

``python -m tools.panels keyness.csv out/ --panel keyness_volcano``

One CLI for every panel, present and future. It takes no per-panel flags:
each panel declares its parameters (``core/viz/panelspec.PanelParam``) and
this command builds its interface from those declarations, so adding a panel
adds no argparse code and cannot drift from what the panel actually accepts.
Settings are passed generically::

    --set label-top=40 --set label-by=effect --set significance=0.01

``--list`` prints the registered panels; ``--list --json`` prints their full
declarations, which is what a front end reads to build its controls.

Every run writes, through the OutputWriter only (R3):

* ``panel.html`` — the self-contained figure, its caption and its limits
* ``panel_data.csv`` — the rows the figure was drawn from
* ``panel.png``/``.svg``/``.pdf`` — only when ``--format`` asks for one; a
  failed export fails the run but keeps the HTML and CSV
* ``result.json`` — the envelope: input sha256, effective parameters,
  artifacts, diagnostics

The exported figure carries its provenance caption *inside* the image, so a
PNG pasted into a document still says which artifact it came from. The input
file's real sha256 goes into that caption, not just into the envelope: a
figure that cannot be traced to its data is the one nobody can check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panels import PANELS, get_panel, panel_names

__all__ = ["main"]

_FORMATS = ("png", "svg", "pdf")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="panels",
        description="Tool-specific figures over an analysis CSV (one panel per run)",
        epilog="Run with --list to see the registered panels and what each one takes.",
    )
    p.add_argument("input", type=Path, nargs="?", help="analysis CSV/TSV produced by the panel's tool")
    p.add_argument("output", type=Path, nargs="?", help="output root directory (a run dir is created inside)")
    p.add_argument("--panel", default="", help=f"which panel to draw: {', '.join(panel_names())}")
    p.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a panel parameter, repeatable (see --list for each panel's parameters)",
    )
    p.add_argument("--format", choices=[*_FORMATS, "none"], default="none", help="also export a static image")
    p.add_argument("--browser-path", default="", help="Chrome/Chromium for image export (kaleido)")
    p.add_argument("--cdn", action="store_true", help="link plotly from a CDN instead of inlining it")
    p.add_argument("--delimiter", default=",", help="input field delimiter: , (default), ;, tab, |")
    p.add_argument("--encoding", default="utf-8-sig", help="input text encoding (default utf-8-sig)")
    p.add_argument("--list", action="store_true", help="list the registered panels and exit")
    p.add_argument("--json", action="store_true", help="print exactly one JSON object to stdout")
    return p.parse_args(argv)


def _list_panels(as_json: bool) -> int:
    if as_json:
        print(json.dumps({"panels": [definition.to_dict() for definition in PANELS]}, indent=2))
        return 0
    for definition in PANELS:
        print(definition.describe())
        for param in definition.params:
            allowed = f" ({'|'.join(param.choices)})" if param.choices else ""
            print(f"    --set {param.name}={param.default}{allowed}  {param.help}")
    return 0


def _coerce(raw: str, declared_type: str) -> Result[Any]:
    """One ``--set`` value from text into what the panel declared.

    argparse gives strings; the registry's validation wants real types and
    refuses a string where a number belongs. Coercing here keeps that
    strictness intact rather than loosening it for every caller.
    """
    if declared_type == "int":
        try:
            return Result.success(int(raw))
        except ValueError:
            return Result[Any].failure(Diagnostic.error("PANEL_CLI_BAD_VALUE", f"{raw!r} is not a whole number"))
    if declared_type == "float":
        try:
            return Result.success(float(raw))
        except ValueError:
            return Result[Any].failure(Diagnostic.error("PANEL_CLI_BAD_VALUE", f"{raw!r} is not a number"))
    if declared_type == "bool":
        lowered = raw.strip().lower()
        if lowered in ("true", "yes", "1", "on"):
            return Result.success(True)
        if lowered in ("false", "no", "0", "off"):
            return Result.success(False)
        return Result[Any].failure(Diagnostic.error("PANEL_CLI_BAD_VALUE", f"{raw!r} is not true or false"))
    return Result.success(raw)


def _collect_settings(panel: str, pairs: list[str]) -> Result[dict[str, Any]]:
    """``NAME=VALUE`` strings into typed parameters the registry will check."""
    definition = get_panel(panel)
    if definition is None:
        return Result[dict[str, Any]].failure(
            Diagnostic.error("PANEL_UNKNOWN", f"no panel named {panel!r}; known panels: {', '.join(panel_names())}")
        )
    declared = {param.name: param for param in definition.params}
    values: dict[str, Any] = {}
    for pair in pairs:
        name, separator, raw = pair.partition("=")
        if not separator:
            return Result[dict[str, Any]].failure(
                Diagnostic.error("PANEL_CLI_BAD_SETTING", f"--set expects NAME=VALUE, got {pair!r}")
            )
        param = declared.get(name.strip())
        if param is None:
            known = ", ".join(sorted(declared)) or "(none)"
            return Result[dict[str, Any]].failure(
                Diagnostic.error("PANEL_UNKNOWN_PARAM", f"{panel} has no parameter {name.strip()!r}; it takes: {known}")
            )
        coerced = _coerce(raw, param.type)
        if coerced.value is None:
            return Result[dict[str, Any]](None, coerced.diagnostics)
        values[param.name] = coerced.unwrap()
    return Result.success(values)


def _resolve_delimiter(raw: str) -> str:
    return {"tab": "\t", "\\t": "\t"}.get(raw, raw)


def _read_input(path: Path, delimiter: str, encoding: str) -> Result[pd.DataFrame]:
    try:
        frame = pd.read_csv(path, delimiter=delimiter, encoding=encoding)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return Result[pd.DataFrame].failure(
            Diagnostic.error("PANEL_INPUT_UNREADABLE", f"could not read {path}: {exc}", path=str(path))
        )
    return Result.success(frame)


def _diags(diagnostics: list[Diagnostic]) -> list[dict[str, Any]]:
    return [d.to_dict() for d in diagnostics]


def _emit(
    as_json: bool,
    ok: bool,
    run_dir: Path | None,
    artifacts: list[dict[str, str]],
    diagnostics: list[dict[str, Any]],
) -> None:
    """Exactly one JSON object in --json mode, on success and on failure."""
    if as_json:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "run_dir": str(run_dir) if run_dir else None,
                    "artifacts": artifacts,
                    "diagnostics": diagnostics,
                },
                indent=2,
            )
        )
        return
    for diagnostic in diagnostics:
        print(f"[{diagnostic['severity']}] {diagnostic['code']}: {diagnostic['message']}", file=sys.stderr)
    if ok and run_dir:
        print(run_dir)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        return _list_panels(args.json)
    if not args.panel:
        _emit(
            args.json,
            False,
            None,
            [],
            _diags(
                [
                    Diagnostic.error(
                        "PANEL_CLI_NO_PANEL", f"--panel is required; choose one of: {', '.join(panel_names())}"
                    )
                ]
            ),
        )
        return 1
    if args.input is None or args.output is None:
        _emit(
            args.json,
            False,
            None,
            [],
            _diags([Diagnostic.error("PANEL_CLI_NO_INPUT", "an input CSV and an output directory are required")]),
        )
        return 1
    if not args.input.is_file():
        _emit(
            args.json,
            False,
            None,
            [],
            _diags([Diagnostic.error("PANEL_INPUT_MISSING", f"{args.input} is not a file", path=str(args.input))]),
        )
        return 1

    settings = _collect_settings(args.panel, args.settings)
    if settings.value is None:
        _emit(args.json, False, None, [], _diags(list(settings.diagnostics)))
        return 1
    frame_result = _read_input(args.input, _resolve_delimiter(args.delimiter), args.encoding)
    if frame_result.value is None:
        _emit(args.json, False, None, [], _diags(list(frame_result.diagnostics)))
        return 1

    from core.io.reader import hash_file
    from core.viz.panel_plotters import panel_html, panel_image_bytes
    from core.viz.panels import prepare_panel
    from core.viz.panelspec import Source

    # The caption names the file AND its hash: provenance with an empty
    # sha256 is an advertisement, not evidence.
    origin = Source(
        path=args.input.name,
        sha256=hash_file(args.input),
        library=_renderer_version(),
    )
    prepared = prepare_panel(args.panel, frame_result.unwrap(), settings.unwrap(), source=origin)
    if prepared.value is None:
        _emit(args.json, False, None, [], _diags(list(prepared.diagnostics)))
        return 1
    panel = prepared.unwrap()
    notices = list(prepared.diagnostics)

    rendered = panel_html(panel, offline=not args.cdn)
    if rendered.value is None:
        _emit(args.json, False, None, [], _diags(notices + list(rendered.diagnostics)))
        return 1
    notices += list(rendered.diagnostics)

    from core.io.writer import OutputWriter

    try:
        writer = OutputWriter(
            args.output,
            tool="panels",
            params={"panel": args.panel, **panel.provenance.params},
            inputs=[args.input],
        )
    except (ValueError, OSError) as exc:
        _emit(args.json, False, None, [], _diags([*notices, Diagnostic.error("PANEL_WRITER_FAILED", str(exc))]))
        return 1

    artifacts: list[dict[str, str]] = []
    written = writer.write_html(rendered.unwrap(), "panel.html", description=f"{panel.panel}: {panel.title}")
    if written.value is None:
        writer.abandon()
        _emit(args.json, False, None, [], _diags(notices + list(written.diagnostics)))
        return 1
    artifacts.append({"kind": "chart", "path": "panel.html"})

    table = writer.write_table(panel.data, "panel_data.csv", description="the rows the figure was drawn from")
    if table.value is None:
        writer.abandon()
        _emit(args.json, False, None, [], _diags(notices + list(table.diagnostics)))
        return 1
    artifacts.append({"kind": "table", "path": "panel_data.csv"})

    if args.format in _FORMATS:
        failure = _export_image(writer, panel, args, artifacts, notices)
        if failure is not None:
            return failure

    writer.add_diagnostics(*notices)
    envelope = writer.finalize()
    if envelope.value is None:
        writer.abandon()
        _emit(args.json, False, None, [], _diags(notices + list(envelope.diagnostics)))
        return 1
    _emit(args.json, True, writer.run_dir.resolve(), _absolute(writer, artifacts), _diags(notices))
    return 0


def _export_image(
    writer: Any,
    panel: Any,
    args: argparse.Namespace,
    artifacts: list[dict[str, str]],
    notices: list[Diagnostic],
) -> int | None:
    """Write the requested image, or fail the run keeping HTML and CSV.

    An explicitly requested export that cannot happen is a failed run: the
    alternative is handing back a directory that silently lacks the figure
    someone asked to publish. The successful artifacts stay, marked.
    """
    from core.viz.panel_plotters import panel_image_bytes

    image = panel_image_bytes(panel, args.format, browser_path=args.browser_path or None)
    problems: list[Diagnostic] = []
    if image.value is None:
        problems = list(image.diagnostics)
    else:
        written = writer.write_bytes(
            image.unwrap(),
            f"panel.{args.format}",
            kind="image",
            description=f"{panel.panel} ({args.format})",
        )
        if written.value is None:
            problems = list(written.diagnostics)
        else:
            artifacts.append({"kind": "image", "path": f"panel.{args.format}"})
            return None

    writer.add_diagnostics(*notices)
    published = writer.publish_failure(*problems, keep_artifacts=True)
    if published.ok:
        _emit(args.json, False, writer.run_dir.resolve(), _absolute(writer, artifacts), _diags(notices + problems))
    else:
        # Publication failed too: clean the staging directory rather than
        # advertising paths to files the caller cannot consume.
        writer.abandon()
        _emit(args.json, False, None, [], _diags(notices + problems + list(published.diagnostics)))
    return 1


def _absolute(writer: Any, artifacts: list[dict[str, str]]) -> list[dict[str, str]]:
    """Artifact paths resolved against the final run dir, after finalize."""
    return [{**artifact, "path": str((writer.run_dir / artifact["path"]).resolve())} for artifact in artifacts]


def _renderer_version() -> str:
    """What drew the figure, for the caption. Absent plotly is not an error
    here — the renderer degrades, and the caption should say so honestly."""
    try:
        import plotly

        return f"plotly {plotly.__version__}"
    except ImportError:
        return "html table (plotly not installed)"


if __name__ == "__main__":
    raise SystemExit(main())
