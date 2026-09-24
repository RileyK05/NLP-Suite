"""FR-7 child 3 (C2) — executor contract tests.

Per-tool isolation, timings, input declaration, and unknown-adapter
handling. Offline except the marked integration replay. Fails until
core/profiler/executor.py lands (C3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.io.reader import Corpus, Document, read_corpus
from core.profiler import executor as exec_mod
from core.profiler.plan import Plan, PlannedTool, build_plan

MINI_CORPUS = Path(__file__).resolve().parent / "fixtures" / "mini-corpus"


def _plan(*names: str, params: dict | None = None) -> Plan:
    res = build_plan(list(names), params)
    assert res.ok, res.diagnostics
    return res.unwrap()


def _boom(_ctx: object, _params: dict) -> object:
    raise RuntimeError("boom")


def _ok(_ctx: object, _params: dict) -> object:
    from core.result import Result

    return Result.success({"ok.csv": pd.DataFrame([{"a": 1}])})


class TestIsolation:
    def test_failure_isolates_and_batch_continues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(exec_mod, "ADAPTERS", {"boom": _boom, "ok_tool": _ok})
        monkeypatch.setattr(exec_mod, "ADAPTER_NEEDS", {})
        plan = Plan(
            tools=(
                PlannedTool(name="boom", params={}, requires_parse=False, phase=2),
                PlannedTool(name="ok_tool", params={}, requires_parse=False, phase=2),
            ),
            params={},
        )
        batch = exec_mod.execute(plan)
        assert [o.name for o in batch.outcomes] == ["boom", "ok_tool"]
        failed, ok = batch.outcomes
        assert failed.ok is False
        assert any(d.code == "PROFILER_TOOL_CRASH" for d in failed.diagnostics)
        assert failed.seconds >= 0
        assert ok.ok is True and list(ok.frames) == ["ok.csv"]

    def test_unknown_adapter_fails(self) -> None:
        plan = Plan(
            tools=(PlannedTool(name="nope", params={}, requires_parse=False, phase=2),),
            params={},
        )
        (outcome,) = exec_mod.execute(plan).outcomes
        assert outcome.ok is False
        assert any(d.code == "PROFILER_NO_ADAPTER" for d in outcome.diagnostics)

    def test_missing_table_fails_without_calling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []

        def spy(_ctx: object, _params: dict) -> object:
            called.append("ran")
            return _ok(_ctx, _params)

        monkeypatch.setattr(exec_mod, "ADAPTERS", {"sentence_complexity": spy})
        monkeypatch.setattr(exec_mod, "ADAPTER_NEEDS", {"sentence_complexity": frozenset({"table"})})
        plan = _plan("sentence_complexity")
        (outcome,) = exec_mod.execute(plan, table=None).outcomes
        assert outcome.ok is False
        assert called == []
        assert any(d.code == "PROFILER_NO_TABLE" for d in outcome.diagnostics)

    def test_empty_plan_succeeds_empty(self) -> None:
        batch = exec_mod.execute(Plan(tools=(), params={}))
        assert batch.outcomes == ()


def test_saved_phrase_question_publishes_complete_comparable_tables(tmp_path: Path) -> None:
    corpus = Corpus(
        (
            Document(
                doc_id=1,
                path=tmp_path / "2020-speech.txt",
                text="public health and private health public health",
                sha256="content-hash",
                source_id="document-1",
                label="2020 speech.txt",
            ),
        ),
        "corpus-hash",
    )
    table = pd.DataFrame(
        {
            "ID": range(1, 8),
            "Form": ["public", "health", "and", "private", "health", "public", "health"],
            "Sentence ID": [1] * 7,
            "Document ID": [1] * 7,
        }
    )
    params = {
        "phrase_distribution": {
            "phrase": "public health",
            "comparison": "private health",
            "question-name": "Health language",
            "question-id": "question-1",
            "snapshot-id": "a" * 64,
        }
    }
    plan = Plan(
        (PlannedTool("phrase_distribution", params["phrase_distribution"], True, 2),),
        params,
    )
    (outcome,) = exec_mod.execute(plan, corpus=corpus, table=table).outcomes

    assert outcome.ok, outcome.diagnostics
    assert set(outcome.frames) == {
        "phrase_question.csv",
        "phrase_summary.csv",
        "phrase_time.csv",
        "phrase_positions.csv",
        "phrase_documents.csv",
        "phrase_occurrences.csv",
    }
    summary = outcome.frames["phrase_summary.csv"].set_index("subject")
    assert summary.loc["public health", "occurrences"] == 2
    assert summary.loc["private health", "occurrences"] == 1
    occurrences = outcome.frames["phrase_occurrences.csv"]
    assert len(occurrences) == 3
    assert set(occurrences["document_id"]) == {"document-1"}


@pytest.mark.model_integration
class TestReplay:
    def test_offline_tools_run_on_mini_corpus(self) -> None:
        corpus = read_corpus(MINI_CORPUS).unwrap()
        plan = _plan("readability", "lexical_diversity", "doc_duplicates", "doc_similarity")
        batch = exec_mod.execute(plan, corpus=corpus)
        assert len(batch.outcomes) == 4
        for outcome in batch.outcomes:
            assert outcome.ok, (outcome.name, outcome.diagnostics)
            assert outcome.frames, outcome.name

    def test_search_conll_mode_needs_table(self) -> None:
        plan = _plan("search", params={"search": {"mode": "conll", "query": "the"}})
        (outcome,) = exec_mod.execute(plan, table=None).outcomes
        assert outcome.ok is False
        assert any(d.code == "PROFILER_NO_TABLE" for d in outcome.diagnostics)

    def test_search_csv_mode_unavailable_in_batch(self) -> None:
        corpus = read_corpus(MINI_CORPUS).unwrap()
        plan = _plan("search", params={"search": {"mode": "csv", "query": "the"}})
        (outcome,) = exec_mod.execute(plan, corpus=corpus).outcomes
        assert outcome.ok is False
        assert any(d.code == "PROFILER_MODE_UNAVAILABLE" for d in outcome.diagnostics)


def _kwic_table() -> pd.DataFrame:
    rows = []
    rec = 1
    for did, sentences in {"1": [["Alice", "loved", "Paris"], ["Bob", "loved", "Rome"]]}.items():
        for sid, sent in enumerate(sentences, start=1):
            for i, tok in enumerate(sent, start=1):
                rec += 1
                rows.append(
                    {
                        "ID": i,
                        "Form": tok,
                        "Lemma": tok.lower(),
                        "POS": "NN",
                        "NER": "GPE" if tok in ("Paris", "Rome") else "PERSON" if tok in ("Alice", "Bob") else "O",
                        "Head": 0,
                        "DepRel": "root",
                        "Sentence ID": sid,
                        "Document ID": did,
                        "Document": "a.txt",
                        "Deps": "",
                        "Record ID": rec,
                        "Clause Tag": "",
                    }
                )
    return pd.DataFrame(rows)


class TestExpansionAdapters:
    def test_kwic_keyness_ner_run(self) -> None:
        table = _kwic_table()
        plan = _plan(
            "kwic",
            "keyness",
            "ner",
            params={
                "kwic": {"query": "Paris"},
                "keyness": {"group-pattern": "a"},
            },
        )
        batch = exec_mod.execute(plan, table=table)
        by_name = {o.name: o for o in batch.outcomes}
        assert by_name["kwic"].ok, by_name["kwic"].diagnostics
        kwic_frame = by_name["kwic"].frames["kwic.csv"]
        assert len(kwic_frame) == 1 and kwic_frame.iloc[0]["Hit"] == "Paris"
        # one document table: keyness refuses with a clear tool-level error
        assert by_name["keyness"].ok is False
        assert any(d.code == "KEYNESS_ONE_GROUP" for d in by_name["keyness"].diagnostics)
        assert by_name["ner"].ok
        assert "movement_tracks.csv" in by_name["ner"].frames
        assert "movement_summary.csv" in by_name["ner"].frames

    def test_keyness_two_documents_scores(self) -> None:
        plan = _plan("keyness", params={"keyness": {"group-pattern": "alpha"}})
        table = _kwic_table()
        table["Document"] = "alpha.txt"
        extra = table.iloc[:3].copy()
        extra["Document"] = "beta.txt"
        extra["Form"] = "zeta"
        extra["Lemma"] = "zeta"
        table = pd.concat([table, extra], ignore_index=True)
        (outcome,) = exec_mod.execute(plan, table=table).outcomes
        assert outcome.ok, outcome.diagnostics
        frame = outcome.frames["keyness.csv"]
        zeta = frame[frame["Word"] == "zeta"]
        assert len(zeta) == 1

    def test_kwic_bad_query_fails_tool_only(self) -> None:
        plan = _plan("kwic", params={"kwic": {"query": "([", "regex": True}})
        batch = exec_mod.execute(plan, table=_kwic_table())
        (outcome,) = batch.outcomes
        assert outcome.ok is False
        assert any(d.code == "KWIC_BAD_REGEX" for d in outcome.diagnostics)


class TestNewToolAdapters:
    """The three tools added in the desktop-expansion batch, end to end."""

    def test_collocations_tfidf_dispersion_run(self) -> None:
        table = _kwic_table()
        plan = _plan("collocations", "tfidf", "dispersion")
        batch = exec_mod.execute(plan, table=table)
        by_name = {o.name: o for o in batch.outcomes}
        for name in ("collocations", "tfidf", "dispersion"):
            assert by_name[name].ok, (name, by_name[name].diagnostics)
        assert "collocations.csv" in by_name["collocations"].frames
        assert "tfidf.csv" in by_name["tfidf"].frames
        # two one-sentence documents: dispersion needs >= 2 non-empty parts
        assert "dispersion.csv" in by_name["dispersion"].frames

    def test_dispersion_plot_param_writes_html(self) -> None:
        table = _kwic_table()
        extra = table.copy()
        extra["Document ID"] = "2"
        table = pd.concat([table, extra], ignore_index=True)
        plan = _plan("dispersion", params={"dispersion": {"plot": True, "min-count": 1}})
        (outcome,) = exec_mod.execute(plan, table=table).outcomes
        assert outcome.ok, outcome.diagnostics
        # the registry promises the plot alongside the table when --plot is set
        assert "dispersion_plot.html" in outcome.frames
        html = outcome.frames["dispersion_plot.html"].iloc[0]["html"]
        assert "plotly" in html or "<html" in html.lower()

    def test_dispersion_plot_false_writes_table_only(self) -> None:
        table = _kwic_table()
        extra = table.copy()
        extra["Document ID"] = "2"
        table = pd.concat([table, extra], ignore_index=True)
        plan = _plan("dispersion", params={"dispersion": {"min-count": 1}})
        (outcome,) = exec_mod.execute(plan, table=table).outcomes
        assert outcome.ok, outcome.diagnostics
        assert "dispersion.csv" in outcome.frames
        assert "dispersion_plot.html" not in outcome.frames
