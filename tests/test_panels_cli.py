"""The panels CLI: one command that serves every panel from its declaration.

The point of the generic ``--set NAME=VALUE`` interface is that adding a
panel adds no argparse code, so nothing can drift between what a panel
accepts and what the command offers. These tests hold that: the listing is
built from the registry, unknown names and bad values are refused with the
panel's own vocabulary, and the run writes through the OutputWriter only.

The provenance assertions matter most. A caption naming an artifact without
its hash is an advertisement rather than evidence, so
``test_the_caption_carries_the_inputs_real_hash`` checks the digest in the
figure is the digest of the file on disk, and that it agrees with the
envelope.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from core.viz.panels import panel_names
from tools.panels import main

_CSV = "\n".join(
    [
        "Word,Freq Group A (pattern docs),Freq Group B (other docs),G2 (log-likelihood),p-value,Log Ratio,Overrepresented in",
        "freedom,120,12,88.4,0.0,2.31,Group A",
        "the,9000,8700,41.2,0.0,0.04,Group A",
        "tariff,1,19,2.1,0.14,-3.20,Group B",
        "war,60,140,22.8,0.0,-1.20,Group B",
    ]
)


@pytest.fixture
def keyness_csv(tmp_path: Path) -> Path:
    path = tmp_path / "keyness.csv"
    path.write_text(_CSV, encoding="utf-8")
    return path


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    """Run the CLI in --json mode and return (exit code, the one object)."""
    code = main([*argv, "--json"])
    out = capsys.readouterr().out.strip()
    assert out, "--json mode must print exactly one object on success and on failure"
    return code, json.loads(out)


class TestListing:
    def test_the_listing_comes_from_the_registry(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--list"]) == 0
        printed = capsys.readouterr().out
        for name in panel_names():
            assert name in printed

    def test_json_listing_carries_each_panels_declaration(self, capsys: pytest.CaptureFixture[str]) -> None:
        """What a front end reads to build controls without hard-coding them."""
        assert main(["--list", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        volcano = next(p for p in payload["panels"] if p["name"] == "keyness_volcano")
        assert volcano["tool"] == "keyness"
        assert volcano["shape"] == "scatter_labelled"
        assert volcano["notes"], "a panel must ship what it cannot show"
        names = {param["name"] for param in volcano["params"]}
        assert {"label-top", "label-by", "significance", "min-frequency"} <= names
        label_by = next(p for p in volcano["params"] if p["name"] == "label-by")
        assert label_by["choices"] == ["evidence", "effect"]
        assert label_by["help"]


class TestRun:
    def test_a_run_writes_figure_table_and_envelope(
        self, keyness_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pytest.importorskip("plotly")
        code, payload = run([str(keyness_csv), str(tmp_path / "out"), "--panel", "keyness_volcano"], capsys)
        assert code == 0, payload
        assert payload["ok"]
        kinds = {artifact["kind"] for artifact in payload["artifacts"]}
        assert {"chart", "table"} <= kinds
        for artifact in payload["artifacts"]:
            assert Path(artifact["path"]).is_file(), f"{artifact['path']} was advertised but not written"
        run_dir = Path(payload["run_dir"])
        assert (run_dir / "result.json").is_file(), "every run records its envelope"

    def test_the_envelope_records_the_effective_parameters(
        self, keyness_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pytest.importorskip("plotly")
        code, payload = run(
            [str(keyness_csv), str(tmp_path / "out"), "--panel", "keyness_volcano", "--set", "label-top=3"],
            capsys,
        )
        assert code == 0
        envelope = json.loads((Path(payload["run_dir"]) / "result.json").read_text(encoding="utf-8"))
        assert envelope["tool"] == "panels"
        assert envelope["params"]["panel"] == "keyness_volcano"
        assert envelope["params"]["label-top"] == 3
        # Defaults are recorded too: a run is reproducible only if the values
        # that were not typed are written down as well.
        assert envelope["params"]["significance"] == "0.05"

    def test_the_caption_carries_the_inputs_real_hash(
        self, keyness_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Provenance with an empty or invented digest is an advertisement."""
        pytest.importorskip("plotly")
        code, payload = run([str(keyness_csv), str(tmp_path / "out"), "--panel", "keyness_volcano"], capsys)
        assert code == 0
        digest = hashlib.sha256(keyness_csv.read_bytes()).hexdigest()
        html = (Path(payload["run_dir"]) / "panel.html").read_text(encoding="utf-8")
        assert digest[:12] in html, "the figure's caption does not carry the input's hash"
        envelope = json.loads((Path(payload["run_dir"]) / "result.json").read_text(encoding="utf-8"))
        assert envelope["inputs"][0]["sha256"] == digest, "caption and envelope must agree on the input"

    def test_the_drawn_rows_are_the_published_table(
        self, keyness_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pytest.importorskip("plotly")
        code, payload = run(
            [str(keyness_csv), str(tmp_path / "out"), "--panel", "keyness_volcano", "--set", "min-frequency=100"],
            capsys,
        )
        assert code == 0
        table = (Path(payload["run_dir"]) / "panel_data.csv").read_text(encoding="utf-8")
        assert "freedom" in table
        assert "tariff" not in table, "a row left out of the figure is left out of the figure's table"


class TestRefusals:
    def test_an_unknown_panel_lists_the_ones_that_exist(
        self, keyness_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload = run([str(keyness_csv), str(tmp_path / "out"), "--panel", "volcano"], capsys)
        assert code == 1
        assert not payload["ok"]
        assert payload["diagnostics"][0]["code"] == "PANEL_UNKNOWN"
        assert "keyness_volcano" in payload["diagnostics"][0]["message"]

    def test_a_misspelled_setting_names_what_the_panel_takes(
        self, keyness_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload = run(
            [str(keyness_csv), str(tmp_path / "out"), "--panel", "keyness_volcano", "--set", "label_top=5"], capsys
        )
        assert code == 1
        assert payload["diagnostics"][0]["code"] == "PANEL_UNKNOWN_PARAM"
        assert "label-top" in payload["diagnostics"][0]["message"]

    @pytest.mark.parametrize(
        ("setting", "code"),
        [
            ("label-top=lots", "PANEL_CLI_BAD_VALUE"),
            ("label-top", "PANEL_CLI_BAD_SETTING"),
            ("significance=0.5", "PANEL_BAD_PARAM"),
            ("label-top=-4", "PANEL_BAD_PARAM"),
        ],
    )
    def test_bad_settings_are_refused_before_anything_is_written(
        self, keyness_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], setting: str, code: str
    ) -> None:
        out = tmp_path / "out"
        status, payload = run([str(keyness_csv), str(out), "--panel", "keyness_volcano", "--set", setting], capsys)
        assert status == 1
        assert payload["diagnostics"][0]["code"] == code
        assert not out.exists(), "a refused run must not leave a directory behind"

    def test_a_missing_input_is_named(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code, payload = run([str(tmp_path / "nope.csv"), str(tmp_path / "out"), "--panel", "keyness_volcano"], capsys)
        assert code == 1
        assert payload["diagnostics"][0]["code"] == "PANEL_INPUT_MISSING"

    def test_a_table_from_the_wrong_tool_names_the_missing_column(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        wrong = tmp_path / "ngrams.csv"
        wrong.write_text("Ngram,Frequency\nof the,120\n", encoding="utf-8")
        code, payload = run([str(wrong), str(tmp_path / "out"), "--panel", "keyness_volcano"], capsys)
        assert code == 1
        assert payload["diagnostics"][0]["code"] == "PANEL_MISSING_COLUMN"
        assert "Log Ratio" in payload["diagnostics"][0]["message"]

    def test_no_panel_named_is_refused_with_the_choices(
        self, keyness_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, payload = run([str(keyness_csv), str(tmp_path / "out")], capsys)
        assert code == 1
        assert payload["diagnostics"][0]["code"] == "PANEL_CLI_NO_PANEL"
        assert "keyness_volcano" in payload["diagnostics"][0]["message"]


class TestImageExport:
    def test_a_failed_export_fails_the_run_but_keeps_the_figure(
        self, keyness_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An explicitly requested PNG that cannot be made is a failed run --
        never a directory that quietly lacks the thing someone asked for."""
        pytest.importorskip("plotly")
        try:
            import kaleido  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("kaleido installed; the export-failure path is not reachable here")
        code, payload = run(
            [str(keyness_csv), str(tmp_path / "out"), "--panel", "keyness_volcano", "--format", "png"], capsys
        )
        assert code == 1
        assert not payload["ok"]
        assert any(d["code"].startswith("PANEL_IMAGE_") for d in payload["diagnostics"])
        # The HTML and CSV that did succeed are still there, in a run marked failed.
        kept = {artifact["kind"] for artifact in payload["artifacts"]}
        assert {"chart", "table"} <= kept
        for artifact in payload["artifacts"]:
            assert Path(artifact["path"]).is_file()

    def test_an_unsupported_format_is_refused_by_the_interface(self, keyness_csv: Path, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main([str(keyness_csv), str(tmp_path / "out"), "--panel", "keyness_volcano", "--format", "gif"])
