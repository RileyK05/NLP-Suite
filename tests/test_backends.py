"""Backends — Stanza, CoreNLP, SRL, MALLET."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.analysis.mallet import mallet_binary, train_topics
from core.io.reader import Corpus, Document, hash_text
from core.pipelines.corenlp_backend import build_corenlp_pipeline, probe_server
from core.pipelines.srl_backend import build_srl_pipeline, probe_worker
from core.pipelines.stanza_backend import build_stanza_pipeline


def _corpus() -> Corpus:
    return Corpus(
        docs=(Document(doc_id=1, path=Path("a.txt"), text="Hello world.", sha256=hash_text("Hello world.")),),
        sha256="x",
    )


class TestStanza:
    def test_rejects_unsupported_language(self) -> None:
        result = build_stanza_pipeline("xx", frozenset())
        assert not result.ok
        assert result.errors[0].code == "PIPELINE_UNSUPPORTED"

    def test_build_without_model_reports_download(self) -> None:
        result = build_stanza_pipeline("en", frozenset())
        # Either succeeds (model present) or fails with a helpful message
        if result.ok:
            assert result.unwrap().backend == "stanza"
            assert result.unwrap().supports("en")
        elif result.errors[0].code == "PIPELINE_MISSING_DEP":
            assert "stanza" in result.errors[0].message
        else:
            assert result.errors[0].code in {
                "PIPELINE_MODEL_MISSING",
                "PIPELINE_MODEL_UNREADABLE",
                "PIPELINE_MODEL_CORRUPT",
            }
            assert "stanza.download" in result.errors[0].message.lower()
            assert result.errors[0].context.get("fix", "")


class TestCoreNLP:
    def test_build_succeeds_but_parse_needs_server(self) -> None:
        pipeline = build_corenlp_pipeline("en", frozenset()).unwrap()
        result = pipeline.parse(_corpus())
        assert not result.ok
        assert result.errors[0].code == "CORENLP_UNAVAILABLE"

    def test_rejects_unsupported_language(self) -> None:
        result = build_corenlp_pipeline("xx", frozenset())
        assert not result.ok

    def test_probe_closed_port_fails_loudly(self) -> None:
        res = probe_server("http://127.0.0.1:9", timeout=2.0)
        assert not res.ok
        assert res.diagnostics[0].code == "CORENLP_UNAVAILABLE"
        assert "fix" in res.diagnostics[0].context

    @pytest.mark.model_integration
    def test_probe_live_server(self) -> None:
        url = os.environ.get("NLP_SUITE_CORENLP_URL", "")
        if not url:
            pytest.skip("no live CoreNLP server (set NLP_SUITE_CORENLP_URL)")
        assert probe_server(url).ok


class TestSRL:
    def test_supports_only_english(self) -> None:
        assert build_srl_pipeline("en", frozenset()).ok
        assert not build_srl_pipeline("de", frozenset()).ok

    def test_parse_without_worker_names_the_env_var(self) -> None:
        pipeline = build_srl_pipeline("en", frozenset()).unwrap()
        result = pipeline.parse(_corpus())
        assert not result.ok
        assert result.errors[0].code == "SRL_WORKER_MISSING"
        assert "NLP_SUITE_SRL_PYTHON" in result.errors[0].message

    def test_bogus_worker_path_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        missing = tmp_path / "no-python-here"
        monkeypatch.setenv("NLP_SUITE_SRL_PYTHON", str(missing))
        res = probe_worker()
        assert not res.ok
        assert res.diagnostics[0].code == "SRL_WORKER_MISSING"

    def test_worker_without_package_fails(self) -> None:
        import sys

        res = probe_worker(sys.executable)
        if res.ok:
            pytest.skip("this interpreter has transformer_srl — worker is real")
        assert res.diagnostics[0].code == "SRL_WORKER_BROKEN"


class TestMallet:
    def test_missing_binary_fails_loudly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", str(tmp_path))  # empty PATH: no mallet anywhere
        assert mallet_binary() is None
        res = train_topics(tmp_path, tmp_path / "out")
        assert not res.ok
        assert res.diagnostics[0].code == "MALLET_MISSING"

    def test_broken_binary_maps_to_failed(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        res = train_topics(tmp_path, out, binary=str(tmp_path / "no-mallet"))
        assert not res.ok
        assert res.diagnostics[0].code == "MALLET_FAILED"

    def test_missing_output_dir_fails(self, tmp_path: Path) -> None:
        res = train_topics(tmp_path, tmp_path / "no-dir", binary=str(tmp_path / "no-mallet"))
        assert not res.ok
        assert res.diagnostics[0].code == "MALLET_NO_OUTPUT_DIR"

    def test_bad_topics_rejected(self, tmp_path: Path) -> None:
        res = train_topics(tmp_path, tmp_path / "out", n_topics=0)
        assert not res.ok
        assert res.diagnostics[0].code == "MALLET_BAD_TOPICS"
