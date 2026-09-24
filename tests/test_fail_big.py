"""Fail-big contract: no silent degradation, every fallback knowable.

These tests pin the idiot-proofing contract:

* A missing parser model is a hard failure with an actionable fix command,
  never a blank-pipeline fallback (a blank pipeline strips POS/lemma/deps
  from every analysis while the run still exits green).
* Reads decode through the encoding chain or fail — never ``errors="ignore"``.
* CSV reads parse every row or fail — never ``on_bad_lines="skip"``.
* Rendering fallbacks (chart table, sankey table) attach a WARNING diagnostic
  so the envelope says plotly was not used.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pandas as pd
import pytest

from conftest import has_spacy_model
from core.pipelines.spacy_backend import build_spacy_pipeline, spacy_model_name


def _block_plotly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any ``import plotly...`` raise ImportError inside the test."""
    real_import = builtins.__import__

    def no_plotly(name: str, *args: object, **kwargs: object):
        if name.startswith("plotly"):
            raise ImportError("plotly disabled for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_plotly)


class TestMissingModelFailsLoudly:
    def test_missing_spacy_model_fails_with_fix(self) -> None:
        """Without a trained model the build fails and names the fix.

        If a model IS installed on this machine the contract is vacuous here;
        the doctor probes cover the installed state directly.
        """
        if has_spacy_model():
            pytest.skip("model installed; hard-failure path not reachable")
        result = build_spacy_pipeline("en", frozenset())
        assert not result.ok
        error = result.errors[0]
        assert error.code == "PIPELINE_MODEL_MISSING"
        assert error.context["fix"] == f"python -m spacy download {spacy_model_name('en')}"
        assert "silently" in error.message  # says WHY it refuses to degrade

    def test_cli_failure_prints_the_fix(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The CLI failure path prints the fix on its own line, exit 1."""
        if has_spacy_model():
            pytest.skip("model installed; hard-failure path not reachable")
        from tools.conll_wordlist import main as wordlist_main

        with pytest.raises(SystemExit) as exc_info:
            wordlist_main(["tests/fixtures/mini-corpus", "out"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "PIPELINE_MODEL_MISSING" in captured.err
        assert f"python -m spacy download {spacy_model_name('en')}" in captured.err


class TestReadsNeverMangleSilently:
    def test_html_annotator_reads_with_encoding_chain(self, tmp_path: Path) -> None:
        from tools.html_annotator import main as annotator_main

        html_file = tmp_path / "page.html"
        # cp1252-encoded bytes: decoded via the fallback chain (with a
        # warning), not silently mangled by errors="ignore".
        html_file.write_bytes(b"<html><body>caf\xe9 latte</body></html>")
        # Output must not live under the input directory (R3); use a sibling.
        out = tmp_path.parent / (tmp_path.name + "-out")
        code = annotator_main([str(html_file), str(out)])
        assert code == 0
        extracted = next(out.rglob("page.txt"))
        assert "caf" in extracted.read_text(encoding="utf-8")

    def test_bad_csv_fails_the_tool(self, tmp_path: Path) -> None:
        """A ragged CSV must fail the run, not skip rows silently."""
        from tools.charts import main as charts_main

        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("x,y\n1,2\n3,4,5,6\n", encoding="utf-8")
        code = charts_main([str(bad_csv), str(tmp_path / "out"), "--x", "x", "--y", "y"])
        assert code == 1


class TestChartFallbackIsKnowable:
    def test_chart_fallback_emits_warning_without_plotly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.viz.charts import bar_chart_html

        _block_plotly(monkeypatch)
        frame = pd.DataFrame({"x": ["a", "b"], "y": [1.0, 2.0]})
        result = bar_chart_html(frame, "x", "y")
        assert result.ok
        assert any(d.code == "CHART_PLOTLY_UNAVAILABLE" for d in result.warnings)
        assert result.warnings[0].context["fix"] == "pip install plotly"

    def test_sankey_fallback_emits_warning_without_plotly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from core.viz.shapes import sankey_html

        _block_plotly(monkeypatch)
        frame = pd.DataFrame({"Source": ["a"], "Target": ["b"]})
        result = sankey_html(frame, "Source", "Target")
        assert result.ok
        assert any(d.code == "SANKEY_PLOTLY_UNAVAILABLE" for d in result.warnings)
