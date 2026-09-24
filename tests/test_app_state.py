"""FR-8.1/FR-8.2 — shared app state contract tests (pure, no Streamlit)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from app.state import DEFAULT_STATE, AppState, resolve_corpus, setup_readiness, summarize_validation

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini-corpus"


class TestDefaults:
    def test_default_state_values(self) -> None:
        assert (
            AppState(corpus_dir="tests/fixtures/mini-corpus", output_root="out", parser="spacy", language="en")
            == DEFAULT_STATE
        )

    def test_state_is_immutable(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_STATE.parser = "stanza"  # type: ignore[misc]


class TestResolveCorpus:
    def test_existing_dir_resolves(self) -> None:
        res = resolve_corpus(str(FIXTURE))
        assert res.ok, res.diagnostics
        assert res.unwrap() == FIXTURE

    def test_blank_fails(self) -> None:
        res = resolve_corpus("   ")
        assert not res.ok
        assert res.diagnostics[0].code == "APP_NO_CORPUS"

    def test_missing_dir_fails(self, tmp_path: Path) -> None:
        res = resolve_corpus(str(tmp_path / "nope"))
        assert not res.ok
        assert res.diagnostics[0].code == "APP_CORPUS_MISSING"

    def test_file_not_dir_fails(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_text("hi", encoding="utf-8")
        res = resolve_corpus(str(target))
        assert not res.ok
        assert res.diagnostics[0].code == "APP_CORPUS_NOT_A_DIR"


class TestSummarize:
    def test_rollup_counts(self) -> None:
        from core.data.validation import validate

        frame = validate(FIXTURE).unwrap()
        summary = summarize_validation(frame)
        assert summary["files"] == len(frame)
        assert summary["ok"] + summary["issues"] == len(frame)
        assert summary["tokens"] >= 0

    def test_empty_frame(self) -> None:
        assert summarize_validation(pd.DataFrame()) == {"files": 0, "ok": 0, "issues": 0, "tokens": 0}


class TestReadiness:
    def test_fixture_corpus_is_ready(self) -> None:
        items = setup_readiness(str(FIXTURE))
        by_label = {item.label: item for item in items}
        assert by_label["Python 3.12+"].ok
        assert by_label["core dependency: pandas"].ok
        assert by_label["parser backend package (spacy)"].ok
        corpus_rows = [item for item in items if item.label.startswith("corpus readable")]
        assert len(corpus_rows) == 1 and corpus_rows[0].ok

    def test_missing_corpus_fails_with_fix(self, tmp_path: Path) -> None:
        items = setup_readiness(str(tmp_path / "nope"))
        corpus_rows = [item for item in items if item.label == "corpus directory"]
        assert len(corpus_rows) == 1 and not corpus_rows[0].ok
        assert corpus_rows[0].fix

    def test_every_failure_names_a_fix(self, tmp_path: Path) -> None:
        for item in setup_readiness(str(tmp_path / "nope"), parser="no-such-backend"):
            if not item.ok:
                assert item.fix, item.label
