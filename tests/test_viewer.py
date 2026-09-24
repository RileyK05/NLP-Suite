"""Scanner — finding runs on disk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.scanner import artifact_view, find_runs
from core.io.reader import Corpus, Document, hash_text
from core.io.writer import OutputWriter


def test_find_runs_discovers_envelopes(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    corpus = Corpus(
        docs=(Document(doc_id=1, path=inputs / "a.txt", text="hi", sha256=hash_text("hi")),),
        sha256="sha",
    )
    writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
    writer.write_text("hello", "out.txt")
    writer.finalize()

    runs = find_runs(tmp_path / "out")
    assert len(runs) == 1
    assert runs[0].envelope.tool == "demo"
    assert runs[0].run_dir == writer.run_dir


def test_find_runs_returns_empty_for_missing_root(tmp_path: Path) -> None:
    assert find_runs(tmp_path / "nope") == []


def test_find_runs_sorts_by_created(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    corpus = Corpus(
        docs=(Document(doc_id=1, path=inputs / "a.txt", text="hi", sha256=hash_text("hi")),),
        sha256="sha",
    )
    writer1 = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus, created="2026-01-01T00:00:00+00:00")
    writer1.write_text("a", "a.txt")
    writer1.finalize()

    writer2 = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus, created="2026-01-02T00:00:00+00:00")
    writer2.write_text("b", "b.txt")
    writer2.finalize()

    runs = find_runs(tmp_path / "out")
    assert runs[0].envelope.created == "2026-01-02T00:00:00+00:00"
    assert runs[1].envelope.created == "2026-01-01T00:00:00+00:00"


def test_find_runs_ignores_non_run_dirs(tmp_path: Path) -> None:
    (tmp_path / "out" / "not_a_run").mkdir(parents=True)
    (tmp_path / "out" / "not_a_run" / "something.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert find_runs(tmp_path / "out") == []


def test_viewer_renders_table_via_scanner(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    corpus = Corpus(
        docs=(Document(doc_id=1, path=inputs / "a.txt", text="hi", sha256=hash_text("hi")),),
        sha256="sha",
    )
    writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
    frame = pd.DataFrame({"Word": ["hello"], "Count": [1]})
    writer.write_table(frame, "table.csv")
    envelope = writer.finalize().unwrap()

    runs = find_runs(tmp_path / "out")
    artifact = runs[0].envelope.artifacts[0]
    assert artifact.kind == "table"
    assert artifact.path == "table.csv"
    assert envelope.artifacts[0].path == "table.csv"
    # The CSV itself is readable
    assert (runs[0].run_dir / "table.csv").is_file()


def test_envelope_kinds_are_free_strings(tmp_path: Path) -> None:
    corpus = Corpus(docs=(), sha256="sha")
    writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
    writer.write_text("custom", "custom.dat", kind="custom_kind")
    envelope = writer.finalize().unwrap()
    assert envelope.artifacts[0].kind == "custom_kind"


def test_artifact_view_mapping() -> None:
    assert artifact_view("table", "a.csv") == "table"
    assert artifact_view("report", "report.md") == "markdown"
    assert artifact_view("chart", "chart.html") == "html"
    assert artifact_view("chart", "wordcloud.html") == "html"
    assert artifact_view("manifest", "batch.json") == "download"
    assert artifact_view("report", "graph.gexf") == "download"
    assert artifact_view("custom_kind", "custom.dat") == "download"
