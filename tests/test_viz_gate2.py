"""Gate-2 focused tests: writer bytes, charts CLI, unified discovery.

Covers the reviewed OutputWriter.write_bytes addition (mirroring write_text
validation), the charts CLI JSON contract (one object on success and
failure, absolute artifact paths, envelope provenance), image-export
failure semantics, and unified --list --json discovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from core.artifacts.envelope import Envelope
from core.io.writer import OutputWriter
from core.result import Diagnostic
from core.viz.chartspec import ChartSpec, PreparedChart, prepare_chart_data
from core.viz.plotters import build_figure, chart_image_bytes, render_chart


def _write_csv(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.write_text(text, encoding=encoding)


def _simple_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "data.csv"
    _write_csv(csv, "cat,n\nA,5\nB,3\nC,8\n")
    return csv


# ---------------------------------------------------------------------------
# OutputWriter.write_bytes regression tests (the reviewed shared extension)
# ---------------------------------------------------------------------------


class TestWriterBytes:
    def test_write_bytes_roundtrip(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        writer = OutputWriter(out, tool="t", params={}, corpus=None)
        written = writer.write_bytes(b"\x89PNG\r\n\x1a\n-fake", "chart.png", description="d")
        assert written.ok
        target = written.unwrap()
        assert target.read_bytes() == b"\x89PNG\r\n\x1a\n-fake"
        assert writer.artifacts[-1].kind == "image"

    def test_write_bytes_duplicate_rejected(self, tmp_path: Path) -> None:
        writer = OutputWriter(tmp_path / "out", tool="t", params={}, corpus=None)
        first = writer.write_bytes(b"a", "chart.png")
        assert first.ok
        again = writer.write_bytes(b"b", "chart.png")
        assert not again.ok
        assert again.errors[0].code == "WRITER_EXISTS"

    def test_write_bytes_reserved_name_rejected(self, tmp_path: Path) -> None:
        writer = OutputWriter(tmp_path / "out", tool="t", params={}, corpus=None)
        reserved = writer.write_bytes(b"{}", "result.json")
        assert not reserved.ok
        assert reserved.errors[0].code == "WRITER_RESERVED"

    def test_write_bytes_escape_rejected(self, tmp_path: Path) -> None:
        writer = OutputWriter(tmp_path / "out", tool="t", params={}, corpus=None)
        escaped = writer.write_bytes(b"x", "../evil.png")
        assert not escaped.ok
        assert escaped.errors[0].code == "WRITER_ESCAPE"

    def test_write_bytes_after_finalize_rejected(self, tmp_path: Path) -> None:
        writer = OutputWriter(tmp_path / "out", tool="t", params={}, corpus=None)
        writer.finalize()
        late = writer.write_bytes(b"x", "late.png")
        assert not late.ok
        assert late.errors[0].code == "WRITER_TERMINAL"

    def test_write_bytes_registered_in_envelope(self, tmp_path: Path) -> None:
        writer = OutputWriter(tmp_path / "out", tool="t", params={}, corpus=None)
        writer.write_bytes(b"\x00\x01", "chart.png", kind="image", description="d")
        env = writer.finalize().unwrap()
        assert any(a.path == "chart.png" and a.kind == "image" for a in env.artifacts)


class TestPublishFailureKeepArtifacts:
    def test_default_strips_artifacts(self, tmp_path: Path) -> None:

        writer = OutputWriter(tmp_path / "out", tool="t", params={}, corpus=None)
        writer.write_text("<html></html>", "chart.html")
        published = writer.publish_failure(Diagnostic.error("X_FAILED", "boom"))
        assert published.ok
        failed_dir = published.unwrap()
        # staged files removed; only the envelope remains
        assert sorted(p.name for p in failed_dir.iterdir()) == ["result.json"]
        env = Envelope.read(failed_dir).unwrap()
        assert env.artifacts == ()

    def test_keep_artifacts_retains_files_and_declares_them(self, tmp_path: Path) -> None:

        writer = OutputWriter(tmp_path / "out", tool="t", params={}, corpus=None)
        writer.write_text("<html></html>", "chart.html", kind="chart")
        writer.write_table(pd.DataFrame({"x": [1]}), "chart_data.csv")
        published = writer.publish_failure(Diagnostic.error("CHART_IMAGE_FAILED", "export failed"), keep_artifacts=True)
        assert published.ok
        failed_dir = published.unwrap()
        assert failed_dir.name.endswith("-failed")
        # files retained on disk
        assert (failed_dir / "chart.html").is_file()
        assert (failed_dir / "chart_data.csv").is_file()
        # envelope declares them
        env = Envelope.read(failed_dir).unwrap()
        assert {a.path for a in env.artifacts} == {"chart.html", "chart_data.csv"}
        assert any(d.code == "CHART_IMAGE_FAILED" for d in env.diagnostics)


# ---------------------------------------------------------------------------
# Renderers over PreparedChart (construction testable without filesystem)
# ---------------------------------------------------------------------------


def _prepared(spec: ChartSpec, frame: pd.DataFrame) -> PreparedChart:
    """Prepare or fail the test immediately (data for the renderer)."""
    result = prepare_chart_data(frame, spec)
    assert result.ok
    return result.unwrap()


class TestRenderers:
    def test_bar_offline_html_self_contained(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="cat", y="n")
        rendered = render_chart(_prepared(spec, frame), spec)
        assert rendered.ok
        html = rendered.unwrap()
        assert len(html) > 100_000  # plotly.js inlined (offline default)
        assert "plotly" in html.lower()

    def test_cdn_optin_is_small(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="cat", y="n", offline=False)
        rendered = render_chart(_prepared(spec, frame), spec)
        assert rendered.ok
        assert len(rendered.unwrap()) < 20_000  # CDN script, no inlined js

    def test_numeric_axis_stays_numeric(self) -> None:
        frame = pd.DataFrame({"year": [10, 1, 2], "n": [8.0, 5.0, 3.0]})
        spec = ChartSpec(kind="line", x="year", y="n")
        fig = build_figure(_prepared(spec, frame), spec).unwrap()
        x_values = list(fig.data[0].x)
        # numbers stay numbers (numpy ints, not strings); axis is linear
        assert all(pd.api.types.is_number(v) for v in x_values)
        assert sorted(int(v) for v in x_values) == [1, 2, 10]
        assert not any(isinstance(v, str) for v in x_values)

    def test_categorical_uses_ticktext_without_mutating_values(self) -> None:
        frame = pd.DataFrame({"cat": ["verylonglabel", "b"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="cat", y="n", wrap_labels=5)
        prepared = _prepared(spec, frame)
        fig = build_figure(prepared, spec).unwrap()
        # raw values untouched in prepared frame (sorted, not wrapped/mutated)
        assert sorted(prepared.data["x"]) == ["b", "verylonglabel"]
        assert all("<br>" not in v for v in prepared.data["x"])
        # ticktext carries wrapped labels
        xaxis = fig.layout.xaxis
        assert xaxis.ticktext is not None
        assert any("<br>" in str(t) for t in xaxis.ticktext)

    def test_heatmap_missing_cells_stay_missing(self) -> None:
        frame = pd.DataFrame({"doc": ["d1", "d1", "d2"], "role": ["R1", "R2", "R1"], "hits": [1.0, 2.0, 3.0]})
        spec = ChartSpec(kind="heatmap", x="doc", y="hits", group="role")
        fig = build_figure(_prepared(spec, frame), spec).unwrap()
        z = fig.data[0].z
        flat = [v for row in z for v in row]
        assert any(pd.isna(v) for v in flat)  # missing (d2, R2) is blank
        assert sum(v for v in flat if not pd.isna(v)) == pytest.approx(6.0)

    def test_histogram_uses_centers_and_widths(self) -> None:
        frame = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 9.0, 10.0]})
        spec = ChartSpec(kind="histogram", x="val", y="val")
        fig = build_figure(_prepared(spec, frame), spec).unwrap()
        trace = fig.data[0]
        assert len(trace.x) == 10  # default 10 common bins
        # every bar carries an explicit width (0.95 of bin width)
        assert trace.width is not None

    def test_horizontal_bar_swaps_axes(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="cat", y="n", horizontal=True)
        fig = build_figure(_prepared(spec, frame), spec).unwrap()
        assert fig.data[0].orientation == "h"

    def test_title_subtitle_margins(self) -> None:
        frame = pd.DataFrame({"cat": ["A"], "n": [1.0]})
        plain = ChartSpec(kind="bar", x="cat", y="n", title="T")
        subbed = ChartSpec(kind="bar", x="cat", y="n", title="T", subtitle="S")
        fig_plain = build_figure(_prepared(plain, frame), plain).unwrap()
        fig_sub = build_figure(_prepared(subbed, frame), subbed).unwrap()
        assert fig_sub.layout.margin.t > fig_plain.layout.margin.t
        assert "<sup>S</sup>" in fig_sub.layout.title.text

    def test_group_colors_deterministic_okabe_ito(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "A", "B", "B"], "grp": ["S1", "S2", "S1", "S2"], "n": [1.0, 2.0, 3.0, 4.0]})
        spec = ChartSpec(kind="bar", x="cat", y="n", group="grp")
        fig1 = build_figure(_prepared(spec, frame), spec).unwrap()
        fig2 = build_figure(_prepared(spec, frame), spec).unwrap()
        colors1 = [t.marker.color for t in fig1.data]
        colors2 = [t.marker.color for t in fig2.data]
        assert colors1 == colors2  # deterministic across runs
        assert "#0072B2" in colors1 or "#E69F00" in colors1  # Okabe-Ito family

    def test_axis_labels_reflect_normalization(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "n": [5.0, 5.0]})
        spec = ChartSpec(kind="bar", x="cat", y="n", normalize="percent")
        prepared = _prepared(spec, frame)
        assert prepared.y_label == "Percent"

    def test_fallback_table_when_plotly_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins
        from collections.abc import Callable

        real_import: Callable[..., object] = builtins.__import__

        def no_plotly(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("plotly"):
                raise ImportError("plotly disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_plotly)
        frame = pd.DataFrame({"cat": ["A", "B"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="cat", y="n")
        rendered = render_chart(_prepared(spec, frame), spec)
        assert rendered.ok
        assert "<table" in rendered.unwrap()
        assert any(d.code == "CHART_PLOTLY_UNAVAILABLE" for d in rendered.warnings)

    def test_image_format_validated(self) -> None:
        frame = pd.DataFrame({"cat": ["A"], "n": [1.0]})
        spec = ChartSpec(kind="bar", x="cat", y="n")
        result = chart_image_bytes(_prepared(spec, frame), spec, "gif")
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_FORMAT"


class TestImageExportMissingKaleido:
    """kaleido is absent on this machine: the failure must be a precise
    diagnostic, never a silent HTML substitute. (This documents the real
    missing-dependency path; a successful export is not claimed here.)"""

    def test_image_export_fails_without_kaleido(self) -> None:
        pytest.importorskip("plotly")
        try:
            import kaleido  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("kaleido installed; missing-dependency path not reachable")
        frame = pd.DataFrame({"cat": ["A"], "n": [1.0]})
        spec = ChartSpec(kind="bar", x="cat", y="n")
        result = chart_image_bytes(_prepared(spec, frame), spec, "png")
        assert not result.ok
        error = result.errors[0]
        assert error.code in ("CHART_IMAGE_KALEIDO_MISSING", "CHART_IMAGE_UNAVAILABLE")
        assert error.context.get("fix") is not None or "kaleido" in error.message.lower()


# ---------------------------------------------------------------------------
# charts CLI: JSON contract, provenance, image-failure semantics
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, Any]:
    from tools.charts import main

    code = main(argv)
    out = capsys.readouterr().out
    payload: Any = json.loads(out)  # exactly one JSON object
    return code, payload


class TestChartsCLI:
    def test_success_json_contract(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = _simple_csv(inp)
        code, payload = _run_cli([str(csv), str(out), "--kind", "bar", "--x", "cat", "--y", "n", "--json"], capsys)
        assert code == 0
        assert payload["ok"] is True
        run_dir = Path(payload["run_dir"])
        assert run_dir.is_dir()
        # absolute artifact paths that exist on disk
        for artifact in payload["artifacts"]:
            assert Path(artifact["path"]).is_absolute()
            assert Path(artifact["path"]).is_file()
        kinds = {a["kind"] for a in payload["artifacts"]}
        assert kinds == {"chart", "table"}
        assert payload["diagnostics"] == []

    def test_envelope_records_input_hash_and_params(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = _simple_csv(inp)
        code, payload = _run_cli(
            [
                str(csv),
                str(out),
                "--kind",
                "bar",
                "--x",
                "cat",
                "--y",
                "n",
                "--agg",
                "sum",
                "--normalize",
                "percent",
                "--json",
            ],
            capsys,
        )
        assert code == 0
        envelope = json.loads((Path(payload["run_dir"]) / "result.json").read_text(encoding="utf-8"))
        assert len(envelope["inputs"]) == 1
        assert len(envelope["inputs"][0]["sha256"]) == 64  # real hash, not ""
        assert envelope["inputs"][0]["path"].endswith("data.csv")
        assert envelope["params"]["kind"] == "bar"
        assert envelope["params"]["agg"] == "sum"
        assert envelope["params"]["normalize"] == "percent"
        # prepared CSV artifact exists and is the aggregated data
        csv_artifact = next(a for a in payload["artifacts"] if a["kind"] == "table")
        prepared = pd.read_csv(csv_artifact["path"])
        assert prepared["y"].sum() == pytest.approx(100.0)  # percent shares

    def test_ambiguous_without_agg_fails_with_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = inp / "d.csv"
        _write_csv(csv, "cat,n\nA,2\nA,3\n")
        code, payload = _run_cli([str(csv), str(out), "--x", "cat", "--y", "n", "--json"], capsys)
        assert code == 1
        assert payload["ok"] is False
        assert payload["diagnostics"][0]["code"] == "CHART_AMBIGUOUS"
        assert payload["run_dir"] is None  # nothing published on failure

    def test_runtime_failure_json_single_object(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = inp / "bad.csv"
        _write_csv(csv, "cat,n\nA,1\nB,2,3,4\n")  # ragged row must fail
        code, payload = _run_cli([str(csv), str(out), "--x", "cat", "--y", "n", "--json"], capsys)
        assert code == 1
        assert payload["ok"] is False
        assert payload["diagnostics"][0]["code"] == "CHART_READ_FAILED"

    def test_image_export_failure_keeps_html_csv_and_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Explicit image request failing must be an ERROR + exit 1, with the
        successful HTML/CSV preserved in a failure-marked run directory."""
        pytest.importorskip("plotly")
        try:
            import kaleido  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("kaleido installed; failure path not reachable")
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = _simple_csv(inp)
        code, payload = _run_cli([str(csv), str(out), "--x", "cat", "--y", "n", "--format", "png", "--json"], capsys)
        assert code == 1
        assert payload["ok"] is False
        codes = [d["code"] for d in payload["diagnostics"]]
        assert any(c.startswith("CHART_IMAGE_") for c in codes)
        # run_dir published (failure-marked) and HTML survived on disk
        run_dir = Path(payload["run_dir"])
        assert run_dir.name.endswith("-failed")
        assert run_dir.is_dir()
        assert (run_dir / "chart.html").is_file()
        assert (run_dir / "result.json").is_file()

    def test_tsv_delimiter(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        tsv = inp / "d.tsv"
        _write_csv(tsv, "cat\tn\nA\t5\nB\t3\n")
        code, payload = _run_cli([str(tsv), str(out), "--x", "cat", "--y", "n", "--delimiter", "tab", "--json"], capsys)
        assert code == 0
        assert payload["ok"] is True

    def test_bad_encoding_diagnosed(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = inp / "d.csv"
        csv.write_bytes(b"caf\xe9,1\n")  # latin-1 bytes, invalid utf-8
        code, payload = _run_cli(
            [str(csv), str(out), "--x", "cat", "--y", "1", "--encoding", "ascii", "--json"], capsys
        )
        assert code == 1
        assert payload["diagnostics"][0]["code"] == "CHART_READ_FAILED"

    def test_bad_delimiter_rejected(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        csv = _simple_csv(inp)
        code, payload = _run_cli(
            [str(csv), str(tmp_path / "out"), "--x", "cat", "--y", "n", "--delimiter", "ab", "--json"], capsys
        )
        assert code == 1
        assert payload["diagnostics"][0]["code"] == "CHART_BAD_DELIMITER"

    def test_incompatible_flags_rejected(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = _simple_csv(inp)
        code, payload = _run_cli(
            [str(csv), str(out), "--x", "cat", "--y", "n", "--kind", "scatter", "--agg", "sum", "--json"], capsys
        )
        assert code == 1
        assert payload["diagnostics"][0]["code"] == "CHART_UNSUPPORTED"

    def test_rate_flags_flow_through(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = inp / "d.csv"
        _write_csv(csv, "cat,mentions,tokens\nA,3,100\nA,2,100\n")
        code, payload = _run_cli(
            [
                str(csv),
                str(out),
                "--kind",
                "bar",
                "--x",
                "cat",
                "--y",
                "mentions",
                "--agg",
                "sum",
                "--rate-per",
                "10000",
                "--denominator-column",
                "tokens",
                "--json",
            ],
            capsys,
        )
        assert code == 0
        csv_artifact = next(a for a in payload["artifacts"] if a["kind"] == "table")
        prepared = pd.read_csv(csv_artifact["path"])
        # (3+2)/200 * 10000 = 250 — hand-computed
        assert prepared["y"].iloc[0] == pytest.approx(250.0)

    def test_plain_output_without_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = _simple_csv(inp)
        from tools.charts import main

        code = main([str(csv), str(out), "--x", "cat", "--y", "n"])
        captured = capsys.readouterr()
        assert code == 0
        assert "Run directory:" in captured.out
        # plain mode must NOT print a JSON object
        with pytest.raises(json.JSONDecodeError):
            json.loads(captured.out)

    def test_heatmap_requires_group_flag(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        inp = tmp_path / "in"
        inp.mkdir()
        out = tmp_path / "out"
        csv = inp / "d.csv"
        _write_csv(csv, "doc,hits\nd1,1\n")
        code, payload = _run_cli(
            [str(csv), str(out), "--kind", "heatmap", "--x", "doc", "--y", "hits", "--json"], capsys
        )
        assert code == 1
        assert payload["diagnostics"][0]["code"] == "CHART_MISSING_GROUP"


# ---------------------------------------------------------------------------
# unified --list --json discovery
# ---------------------------------------------------------------------------


class TestUnifiedDiscovery:
    def test_list_json_shape(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.unified import main

        assert main(["--list", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "tools" in payload and "utility_clis" in payload
        names = {t["name"] for t in payload["tools"]}
        assert "charts" in names
        for tool in payload["tools"]:
            assert set(tool) == {"name", "description", "source", "optional_package"}
            assert tool["source"] == "registry"
        for utility in payload["utility_clis"]:
            assert utility["source"] == "utility"

    def test_plain_list_preserved(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.unified import main

        assert main(["--list"]) == 0
        out = capsys.readouterr().out
        assert "charts" in out
        assert "utility CLIs" in out
        # plain list is not JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(out)

    def test_unknown_tool_vs_import_failure(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools import unified

        # unknown tool: exit 2 with the list hint
        assert unified.main(["definitely-not-a-tool"]) == 2
        err = capsys.readouterr().err
        assert "unknown tool" in err

        # import failure: exit 3, explicitly NOT "unknown"
        import importlib as importlib_module

        monkeypatch.setattr(
            importlib_module,
            "import_module",
            lambda name: (_ for _ in ()).throw(ImportError("No module named 'plotly'")),
        )
        code = unified.main(["charts"])
        err = capsys.readouterr().err
        assert code == 3
        assert "failed to import" in err
        assert "missing dependency" in err

    def test_routing_still_works(self) -> None:
        from tools import unified

        with pytest.raises(SystemExit) as exc_info:
            unified.main(["corpus_validation", "--help"])
        assert exc_info.value.code == 0


class TestUnifiedSubprocess:
    """python -m tools.unified runs as a real CLI (the __main__ guard)."""

    def test_module_cli_list_json_emits_object(self) -> None:
        import os
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "tools.unified", "--list", "--json"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=root,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert "tools" in payload and "utility_clis" in payload
        assert any(t["name"] == "charts" for t in payload["tools"])

    def test_module_cli_plain_list(self) -> None:
        import os
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "tools.unified", "--list"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=60,
        )
        assert result.returncode == 0
        assert "charts" in result.stdout
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stdout)

    def test_module_cli_unknown_tool_exit_2(self) -> None:
        import os
        import subprocess
        import sys

        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "tools.unified", "not-a-real-tool"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=60,
        )
        assert result.returncode == 2
        assert "unknown tool" in result.stderr

    def test_unknown_vs_dependency_failure_in_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ModuleNotFoundError naming tools.<name> = unknown (2); any other
        module = missing dependency (3). Both are verified in-process."""
        import importlib as importlib_module

        from tools import unified

        # unknown: tools.<name> itself missing
        assert unified._route("no-such-module-xyz", []) == 2

        # dependency: tools.<name> exists but its import raises ModuleNotFoundError
        # for a DIFFERENT module name
        def fake_import(name: object, *args: object, **kwargs: object) -> object:
            raise ModuleNotFoundError("No module named 'some_optional_package'")

        monkeypatch.setattr(importlib_module, "import_module", fake_import)
        assert unified._route("charts", []) == 3

        # other ImportError flavors are also dependency-shaped
        def fake_import2(name: object, *args: object, **kwargs: object) -> object:
            raise ImportError("plotly is broken")

        monkeypatch.setattr(importlib_module, "import_module", fake_import2)
        assert unified._route("charts", []) == 3


# ---------------------------------------------------------------------------
# synthetic gallery generator (visual review fixture)
# ---------------------------------------------------------------------------


class TestGallery:
    def test_gallery_generates_all_six_kinds(self, tmp_path: Path) -> None:
        """One synthetic dataset drives a gallery of all six kinds (offline)."""
        from scripts.make_chart_gallery import make_gallery

        out = tmp_path / "gallery"
        result = make_gallery(out)
        assert result.ok
        htmls = sorted(p.name for p in out.rglob("chart.html"))
        assert len(htmls) == 6  # bar, line, scatter, histogram, box, heatmap
        data_csvs = list(out.rglob("chart_data.csv"))
        assert len(data_csvs) == 6

    def test_gallery_deterministic(self, tmp_path: Path) -> None:
        from scripts.make_chart_gallery import make_gallery

        first = tmp_path / "g1"
        second = tmp_path / "g2"
        make_gallery(first)
        make_gallery(second)
        for a, b in zip(sorted(first.rglob("chart_data.csv")), sorted(second.rglob("chart_data.csv")), strict=True):
            assert a.read_bytes() == b.read_bytes()
