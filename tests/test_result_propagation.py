"""C6-3 Result propagation — four-state policy for parse results.

Policy (tools/_cli.PARTIAL_PARSE_POLICY): a parse with a value AND ERROR
diagnostics is a partial run — artifacts + failure envelope are still
written and the CLI exits 1. A value with only warnings is a clean
success (exit 0). A value-less parse exits 1 with diagnostics. No state
may silently pass as exit 0 while carrying an ERROR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import has_spacy_model
from core.result import Diagnostic, Result
from tools._cli import PARTIAL_PARSE_POLICY, handle_result_states


class TestResultStates:
    def test_is_partial_property(self) -> None:
        assert not Result.success([1, 2]).is_partial
        assert Result.success([1], Diagnostic.error("PARSE_FAILED", "doc 2 blew up")).is_partial
        assert not Result[int].failure(Diagnostic.error("BOOM", "no value")).is_partial
        assert not Result.success([1], Diagnostic.warning("EMPTY_DOC", "doc 2 empty")).is_partial

    def test_handle_result_states_classifies(self, capsys: object) -> None:
        assert handle_result_states(Result.success([1])) == "ok"
        assert handle_result_states(Result.success([1], Diagnostic.warning("W", "w"))) == "ok"
        out = capsys.readouterr()  # type: ignore[attr-defined]
        assert "W" in out.err
        assert (
            handle_result_states(Result.success([1], Diagnostic.error("PARSE_FAILED", "doc 2 failed", path="b.txt")))
            == "partial"
        )
        out = capsys.readouterr()  # type: ignore[attr-defined]
        assert "partial run" in out.err and "PARSE_FAILED" in out.err
        with pytest.raises(SystemExit) as exc:
            handle_result_states(Result[int].failure(Diagnostic.error("BOOM", "no value")))
        assert exc.value.code == 1


class TestPartialParsePolicy:
    def test_policy_text_documented(self) -> None:
        assert "never silently" in PARTIAL_PARSE_POLICY
        assert "exits 1" in PARTIAL_PARSE_POLICY
        assert "envelope" in PARTIAL_PARSE_POLICY.lower()


class TestParseCliIntegration:
    @pytest.mark.model_integration
    @pytest.mark.skipif(not has_spacy_model(), reason="integration; spaCy model required")
    def test_one_doc_parses_sibling_reports_not_dropped(self) -> None:
        """Good doc + empty sibling: the sibling is reported, never dropped."""
        from core.io.reader import Corpus, Document, hash_text
        from core.pipelines.spacy_backend import build_spacy_pipeline

        corpus_mixed = Corpus(
            docs=(
                Document(doc_id=1, path=Path("good.txt"), text="The cat sat on the mat.", sha256=hash_text("x")),
                Document(doc_id=2, path=Path("empty.txt"), text="   ", sha256=hash_text("")),
            ),
            sha256="x",
        )
        pipeline = build_spacy_pipeline("en", frozenset()).unwrap()
        result = pipeline.parse(corpus_mixed)
        # The empty doc must be reported, never dropped silently.
        assert any(d.code == "EMPTY_DOC" for d in result.diagnostics)
        assert any("empty.txt" in str(d.context.get("path", d.message)) for d in result.diagnostics)
        # The good document's rows still exist (partial value).
        assert result.value is not None and len(result.value) > 0
        # EMPTY_DOC is a WARNING: this state is clean success per policy.
        assert handle_result_states(result) == "ok"

    def test_partial_envelope_exit_contract(self, tmp_path: Path) -> None:
        """A finalized partial run must record ERROR diagnostics in result.json."""
        from core.io.writer import OutputWriter

        writer = OutputWriter(tmp_path, tool="probe", params={}, corpus=None)
        writer.add_diagnostics(Diagnostic.error("PARSE_FAILED", "bad.txt failed", doc_id=2, path="bad.txt"))
        env_result = writer.finalize()
        assert env_result.ok
        run_dir = writer.run_dir
        data = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        codes = [d["code"] for d in data["diagnostics"]]
        assert "PARSE_FAILED" in codes
        # The failed document coordinates travel with the diagnostic.
        ctx = next(d["context"] for d in data["diagnostics"] if d["code"] == "PARSE_FAILED")
        assert ctx["doc_id"] == 2 and "bad.txt" in str(ctx["path"])
