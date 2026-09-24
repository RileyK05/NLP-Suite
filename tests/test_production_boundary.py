"""End-user boundaries and regressions missed by successful-run smoke tests."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from core.analysis.ner import entity_timeline
from core.analysis.ngrams import ngrams
from desktop_backend.catalog import INTERNAL_TOOLS, desktop_spec
from desktop_backend.runner import DESKTOP_TOOLS, Runner
from desktop_backend.store import Workspace
from desktop_backend.tables import TABLE_TOOLS
from scripts.build_desktop_backend import build_arguments


@pytest.mark.parametrize("tool", sorted(INTERNAL_TOOLS))
def test_internal_tools_cannot_be_submitted(tmp_path: Path, tool: str) -> None:
    workspace = Workspace(tmp_path)
    project = workspace.create("Research")
    runner = Runner(workspace)
    try:
        with pytest.raises(ValueError, match="not available"):
            runner.submit(project["id"], tool, {}, "spacy")
        assert not workspace.jobs(project["id"])
    finally:
        runner.close()


def test_cancel_during_startup_does_not_send_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path)
    project = workspace.create("Startup cancellation")
    workspace.import_document(project["id"], "text.txt", b"A readable document.")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    job = runner.submit(project["id"], "readability", {}, "spacy")
    incoming = io.StringIO()

    def spawn() -> SimpleNamespace:
        runner.cancel(project["id"], job["id"])
        return SimpleNamespace(stdin=incoming)

    monkeypatch.setattr(runner, "_spawn_worker", spawn)
    try:
        runner._dispatch(job["id"])
        assert incoming.getvalue() == ""
        assert workspace.jobs(project["id"])[0]["state"] == "CANCELLED"
    finally:
        runner.close()


def test_unresponsive_worker_startup_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import time

    monkeypatch.setattr("desktop_backend.runner.WORKER_START_SECONDS", 0.1)
    monkeypatch.setattr(
        "desktop_backend.runner.worker_command", lambda root: [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    runner = Runner(Workspace(tmp_path))
    started = time.monotonic()
    try:
        assert runner._spawn_worker() is None
        assert time.monotonic() - started < 5
        assert runner._worker is None
    finally:
        runner.close()


def test_archived_projects_are_read_only(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    project = workspace.create("Archived")
    workspace.import_document(project["id"], "text.txt", b"Keep this document.")
    workspace.archive(project["id"], True)
    runner = Runner(workspace)
    try:
        with pytest.raises(ValueError, match="Restore this project"):
            runner.submit(project["id"], "readability", {}, "spacy")
        with pytest.raises(ValueError, match="Restore this project"):
            workspace.import_document(project["id"], "new.txt", b"New content.")
        assert len(workspace.documents(project["id"])) == 1
    finally:
        runner.close()


def test_release_does_not_collect_developer_entrypoints(tmp_path: Path) -> None:
    args = build_arguments(tmp_path, with_parser=True, system="win32", python="python", version="0.3.0")
    excluded = {args[i + 1] for i, arg in enumerate(args) if arg == "--exclude-module"}
    assert {"tools", "app", "scripts", "tests"} <= excluded
    assert "plotly" not in excluded
    resources = {args[i + 1] for i, arg in enumerate(args) if arg == "--add-data"}
    assert len(resources) == 2  # release notices/manifest and lexical data only


def test_packaged_chart_formats_need_no_external_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    spec = desktop_spec(TABLE_TOOLS["table_charts"])
    formats = next(param.choices for param in spec.params if param.name == "format")
    assert formats == ("html", "xlsx")
    assert "png" in next(param.choices for param in TABLE_TOOLS["table_charts"].params if param.name == "format")


def test_shipped_learning_material_omits_internal_workflows() -> None:
    root = Path(__file__).resolve().parents[1]
    guides = json.loads((root / "desktop/src/toolGuides.json").read_text(encoding="utf-8"))
    assert set(guides) == set(DESKTOP_TOOLS)
    source = (root / "desktop/src/App.tsx").read_text(encoding="utf-8")
    assert "pip install" not in source
    assert "Developer installation commands" not in source
    assert "CLI or Streamlit" not in source


def test_sample_corpus_is_fictional_bundled_content() -> None:
    from desktop_backend.server import sample_dir

    directory = sample_dir()
    assert directory.parent.name == "assets"
    assert {path.name for path in directory.iterdir()} == {"community.txt", "library.txt", "transport.txt"}


def test_numeric_document_ids_keep_ngram_filenames(conll_frame: pd.DataFrame) -> None:
    frame = conll_frame.copy()
    frame["Document ID"] = frame["Document ID"].astype(int)
    result = ngrams(frame, n=1).unwrap()
    assert set(result["Document"]) == {"doc_a.txt", "doc_b.txt"}


def test_bioes_end_boundaries_prevent_adjacent_mentions_merging(conll_frame: pd.DataFrame) -> None:
    frame = conll_frame.iloc[:3].copy()
    frame["Form"] = ["Paris", "London", "today"]
    frame["NER"] = ["S-GPE", "GPE", "O"]
    result = entity_timeline(frame).unwrap()
    assert set(result["Entity"]) == {"Paris", "London"}


def test_vader_desktop_default_does_not_require_anew(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace(tmp_path)
    project = workspace.create("Sentiment")
    workspace.import_document(project["id"], "text.txt", b"Today is a wonderful day.")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    monkeypatch.setattr("importlib.util.find_spec", lambda *args: object())
    try:
        job = runner.submit(project["id"], "sentiment_vader_anew", {}, "spacy")
        assert json.loads(job["request"])["params"]["sentiment_vader_anew"]["analysis"] == "vader"
        with pytest.raises(ValueError, match="Choose an ANEW"):
            runner.submit(project["id"], "sentiment_vader_anew", {"analysis": "both"}, "spacy")
    finally:
        runner.close()


def test_vader_only_does_not_call_resource_loader(conll_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.profiler import executor
    from core.profiler.plan import build_plan
    from core.result import Result

    monkeypatch.setattr(executor, "load_vader_analyzer", lambda *args: Result.success(object()))
    monkeypatch.setattr(executor, "vader", lambda *args, **kwargs: Result.success(pd.DataFrame({"Compound": [0.5]})))
    monkeypatch.setattr(
        executor, "vader_sentences", lambda *args, **kwargs: Result.success(pd.DataFrame({"Compound": [0.5]}))
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("VADER-only execution must not load ANEW")

    monkeypatch.setattr(executor, "anew", forbidden)
    plan = build_plan(["sentiment_vader_anew"], {"sentiment_vader_anew": {"analysis": "vader"}}).unwrap()
    outcome = executor.execute(plan, table=conll_frame).outcomes[0]
    assert outcome.ok
    assert set(outcome.frames) == {"vader.csv", "vader_sentences.csv"}


def test_requested_wordcloud_image_failure_fails_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.result import Diagnostic, Result
    from desktop_backend.runner import run_job

    source = tmp_path / "words.csv"
    source.write_text("word,count\napple,3\norange,2\n", encoding="utf-8")
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Wordcloud failure")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    monkeypatch.setattr(
        "core.viz.wordcloud_gephi.wordcloud_image",
        lambda *args, **kwargs: Result.failure(Diagnostic.error("IMAGE_FAILED", "Unable to render the image.")),
    )
    try:
        job = runner.submit(
            project["id"],
            "table_wordcloud_gephi",
            {"input": str(source), "mode": "wordcloud", "word-col": "word", "weight-col": "count", "image": True},
            "spacy",
        )
        assert run_job(workspace.root, job["id"]) == 1
        saved = workspace.jobs(project["id"])[0]
        assert saved["state"] == "FAILED"
        assert saved["run_dir"] is None
        assert saved["diagnostics"][0]["code"] == "IMAGE_FAILED"
    finally:
        runner.close()


def test_desktop_chart_respects_top_n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from desktop_backend.runner import run_job

    pytest.importorskip("plotly")
    source = tmp_path / "words.csv"
    source.write_text("word,count\napple,3\norange,2\npear,8\n", encoding="utf-8")
    workspace = Workspace(tmp_path / "workspace")
    project = workspace.create("Chart ranking")
    runner = Runner(workspace)
    monkeypatch.setattr(runner.pool, "submit", lambda *args: None)
    try:
        job = runner.submit(
            project["id"],
            "table_charts",
            {"input": str(source), "kind": "bar", "x": "word", "y": "count", "top-n": 1},
            "spacy",
        )
        assert run_job(workspace.root, job["id"]) == 0
        saved = workspace.jobs(project["id"])[0]
        chart_data = pd.read_csv(workspace.project_dir(project["id"]) / saved["run_dir"] / "chart_data.csv")
        assert len(chart_data) == 1
        assert chart_data.iloc[0]["x"] == "pear"
    finally:
        runner.close()
