"""Custody layer: envelope and writer (ARCHITECTURE.md R3, R8).

Covers: envelope round-trip, float rounding, writer as the only writer,
writer refusing to write inside inputs, artifact kinds, and run-directory
layout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.artifacts.envelope import Artifact, Envelope, InputRef
from core.io.reader import Corpus, Document, hash_text
from core.io.writer import OutputWriter, is_inside
from core.result import Diagnostic, Result


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    docs = (
        Document(doc_id=1, path=inputs / "a.txt", text="hello", sha256=hash_text("hello")),
        Document(doc_id=2, path=inputs / "b.txt", text="world", sha256=hash_text("world")),
    )
    return Corpus(docs=docs, sha256="corpus-sha")


class TestInputRef:
    def test_round_trip(self) -> None:
        ref = InputRef(path="a.txt", sha256="abc")
        assert InputRef.from_dict(ref.to_dict()) == ref

    def test_rejects_non_strings(self) -> None:
        with pytest.raises(TypeError):
            InputRef.from_dict({"path": 1, "sha256": "abc"})


class TestArtifact:
    def test_round_trip(self) -> None:
        artifact = Artifact(kind="table", path="out.csv", description="the table")
        assert Artifact.from_dict(artifact.to_dict()) == artifact

    def test_without_description(self) -> None:
        artifact = Artifact(kind="table", path="out.csv")
        assert "description" not in artifact.to_dict()

    def test_rejects_empty_kind(self) -> None:
        with pytest.raises(ValueError):
            Artifact.from_dict({"kind": "", "path": "out.csv"})


class TestEnvelope:
    def test_round_trip_with_diagnostics(self, tmp_path: Path) -> None:
        envelope = Envelope(
            tool="demo",
            params={"alpha": 0.123456789},
            inputs=(InputRef(path="a.txt", sha256="abc"),),
            artifacts=(Artifact(kind="table", path="out.csv"),),
            diagnostics=(Diagnostic.warning("EMPTY_DOC", "empty", doc_id=1),),
            corpus_sha256="corpus-sha",
        )
        written = envelope.write(tmp_path / "run")
        assert written.ok
        loaded = Envelope.read(tmp_path / "run")
        assert loaded.ok
        assert loaded.unwrap().tool == "demo"
        assert loaded.unwrap().diagnostics[0].code == "EMPTY_DOC"

    def test_floats_are_rounded_to_6dp(self, tmp_path: Path) -> None:
        envelope = Envelope(tool="demo", params={"value": 1.23456789})
        envelope.write(tmp_path / "run")
        loaded = Envelope.read(tmp_path / "run")
        assert loaded.unwrap().params["value"] == 1.234568

    def test_missing_envelope_is_an_error(self, tmp_path: Path) -> None:
        result = Envelope.read(tmp_path / "missing")
        assert not result.ok
        assert result.errors[0].code == "ENVELOPE_MISSING"

    def test_malformed_envelope_is_an_error(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        run.mkdir()
        (run / "result.json").write_text("{not json", encoding="utf-8")
        result = Envelope.read(run)
        assert not result.ok
        assert result.errors[0].code == "ENVELOPE_UNREADABLE"

    def test_empty_tool_rejected(self) -> None:
        with pytest.raises(ValueError):
            Envelope(tool="")


class TestOutputWriter:
    def test_creates_run_dir_and_writes_table(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        df = pd.DataFrame({"a": [1, 2]})
        result = writer.write_table(df, "table.csv")
        assert result.ok
        assert (writer.run_dir / "table.csv").is_file()
        finalized = writer.finalize()
        assert finalized.ok
        assert (writer.run_dir / "result.json").is_file()
        assert finalized.unwrap().artifacts[0].path == "table.csv"

    def test_refuses_to_write_inside_input_dir(self, tmp_path: Path) -> None:
        (tmp_path / "corpus").mkdir()
        (tmp_path / "corpus" / "a.txt").write_text("hi", encoding="utf-8")
        corpus = Corpus(
            docs=(Document(doc_id=1, path=tmp_path / "corpus" / "a.txt", text="hi", sha256="abc"),),
            sha256="sha",
        )
        with pytest.raises(ValueError):
            OutputWriter(tmp_path / "corpus" / "out", tool="demo", params={}, corpus=corpus)

    def test_refuses_escape_path(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        result = writer.write_text("hi", "../escape.txt")
        assert not result.ok
        assert result.errors[0].code == "WRITER_ESCAPE"

    def test_refuses_duplicate_filename(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        assert writer.write_text("a", "file.txt").ok
        second = writer.write_text("b", "file.txt")
        assert not second.ok
        assert second.errors[0].code == "WRITER_EXISTS"

    def test_records_diagnostics_in_envelope(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        writer.add_diagnostics(Diagnostic.warning("EMPTY_DOC", "empty", doc_id=1))
        writer.write_text("hi", "out.txt")
        envelope = writer.finalize().unwrap()
        assert envelope.diagnostics[0].code == "EMPTY_DOC"

    def test_run_dir_is_unique(self, tmp_path: Path, corpus: Corpus) -> None:
        writer1 = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        writer2 = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        assert writer1.run_dir != writer2.run_dir

    def test_is_inside(self, tmp_path: Path) -> None:
        assert is_inside(tmp_path / "a" / "b", tmp_path / "a")
        assert not is_inside(tmp_path / "a", tmp_path / "b")
        assert is_inside(tmp_path, tmp_path)

    def test_write_html(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        assert writer.write_html("<b>hi</b>", "chart.html", kind="chart").ok
        assert writer.artifacts[0].kind == "chart"

    def test_write_json_rounds_floats(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        writer.write_json({"value": 1.23456789}, "data.json")
        text = (writer.run_dir / "data.json").read_text(encoding="utf-8")
        assert "1.234568" in text

    def test_corpus_sha_in_envelope(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        envelope = writer.finalize().unwrap()
        assert envelope.corpus_sha256 == corpus.sha256
        assert len(envelope.inputs) == 2


class TestStagingAndProvenance:
    """C6-4: atomic publication, real input hashes, failure envelopes."""

    def test_path_input_gets_real_sha(self, tmp_path: Path, corpus: Corpus) -> None:
        inputs_dir = tmp_path / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        raw = inputs_dir / "data.csv"
        raw.write_text("a,b\n1,2\n", encoding="utf-8")
        writer = OutputWriter(tmp_path / "out", tool="probe", params={}, inputs=(raw,), corpus=None)
        assert len(writer._inputs) == 1
        assert writer._inputs[0].sha256 != ""
        assert len(writer._inputs[0].sha256) == 64  # real SHA-256, not the old ""

    def test_visible_run_dir_always_has_envelope(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="probe", params={}, corpus=corpus)
        assert writer.write_table(pd.DataFrame({"x": [1]}), "t.csv").ok
        result = writer.finalize()
        assert result.ok
        assert (writer.run_dir / "result.json").is_file()
        assert writer.run_dir.name.startswith("probe__")
        assert not writer.run_dir.name.startswith(".staging")

    def test_abandon_leaves_nothing_behind(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="probe", params={}, corpus=corpus)
        writer.write_table(pd.DataFrame({"x": [1]}), "x.csv")
        staging = writer._staging
        assert staging.is_dir()
        writer.abandon()
        assert not staging.exists()
        assert list((tmp_path / "out").iterdir()) == []

    def test_failure_envelope_never_advertises_artifacts(self, tmp_path: Path, corpus: Corpus) -> None:
        writer = OutputWriter(tmp_path / "out", tool="probe", params={}, corpus=corpus)
        writer.write_table(pd.DataFrame({"x": [1]}), "x.csv")
        published = writer.publish_failure(Diagnostic.error("ANALYSIS_FAILED", "second stage blew up"))
        assert published.ok
        envelope = Envelope.read(writer.run_dir).unwrap()
        assert envelope.artifacts == ()  # failure never advertises success
        assert any(d.code == "ANALYSIS_FAILED" for d in envelope.diagnostics)

    def test_finalize_failure_does_not_publish(self, tmp_path: Path, corpus: Corpus) -> None:
        from unittest import mock

        writer = OutputWriter(tmp_path / "out", tool="probe", params={}, corpus=corpus)
        writer.write_table(pd.DataFrame({"x": [1]}), "x.csv")
        with mock.patch.object(
            Envelope, "write", return_value=Result.failure(Diagnostic.error("ENVELOPE_WRITE_FAILED", "disk gone"))
        ):
            result = writer.finalize()
        assert result.value is None
        # Nothing visible was published; staging remains for abandon/decision.
        visible = [p for p in (tmp_path / "out").iterdir() if not p.name.startswith(".staging")]
        assert visible == []
        writer.abandon()

    def test_late_validation_failure_does_not_leave_visible_run(self, tmp_path: Path, corpus: Corpus) -> None:
        """Invalid threshold (>100) after earlier calc: publish failure, not a lying run."""
        writer = OutputWriter(tmp_path / "out", tool="probe", params={"threshold": 101.0}, corpus=corpus)
        writer.write_table(pd.DataFrame({"x": [1]}), "x.csv")
        # Simulate the late validation error the reviewer describes:
        if writer.params["threshold"] > 100.0:
            published = writer.publish_failure(Diagnostic.error("DUPS_BAD_THRESHOLD", "threshold must be 0-100"))
            assert published.ok
        visible = [p for p in (tmp_path / "out").iterdir() if not p.name.startswith(".staging")]
        assert len(visible) == 1 and visible[0].name.endswith("-failed")
        envelope = Envelope.read(visible[0]).unwrap()
        assert envelope.artifacts == ()
