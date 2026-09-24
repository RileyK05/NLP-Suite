"""Regression coverage for the second replacement review pass."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.analysis.sentence_complexity import run as complexity_run
from core.analysis.stats_categorical import log_likelihood
from core.analysis.stats_groups import mann_whitney
from core.compare import CompareConfig, compare_frames
from core.file_ops.filenames import RenamePlan, RenameRow, apply_renames, plan_renames
from core.io.reader import Corpus, hash_file
from core.io.writer import OutputWriter


def _empty_corpus() -> Corpus:
    return Corpus(docs=(), sha256="empty")


def test_writer_rejects_artifacts_after_finalize(tmp_path: Path) -> None:
    writer = OutputWriter(tmp_path / "out", tool="terminal", params={}, corpus=_empty_corpus())
    writer.finalize().unwrap()
    late = writer.write_text("late", "late.txt")
    assert late.value is None
    assert late.errors[0].code == "WRITER_TERMINAL"
    assert not (writer.run_dir / "late.txt").exists()


def test_directory_input_fingerprint_includes_content(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    member = source / "same-size.txt"
    member.write_text("AAAA", encoding="utf-8")
    first = OutputWriter(tmp_path / "one", tool="hash", params={}, inputs=(source,))
    first_hash = first._inputs[0].sha256
    first.abandon()
    member.write_text("BBBB", encoding="utf-8")
    second = OutputWriter(tmp_path / "two", tool="hash", params={}, inputs=(source,))
    assert first_hash != second._inputs[0].sha256
    second.abandon()


def test_unordered_compare_uses_numeric_tolerance() -> None:
    expected = pd.DataFrame({"value": [2.0, 10.0]})
    actual = pd.DataFrame({"value": [2.0000001, 9.9999999]})
    assert compare_frames(expected, actual, CompareConfig(rtol=1e-5, atol=1e-5)).ok


def test_duplicate_threshold_uses_unrounded_similarity(monkeypatch: object) -> None:
    from core.analysis import doc_similarity
    from core.analysis.doc_similarity import _Entry

    entries = [_Entry(1, "a.txt"), _Entry(2, "b.txt")]
    matrix = [[100.0, 79.96], [79.96, 100.0]]
    monkeypatch.setattr(doc_similarity, "_matrix", lambda _corpus, _stopwords: (entries, matrix, []))
    result = doc_similarity.find_duplicates(_empty_corpus(), threshold=80.0)
    assert result.ok
    assert result.unwrap().frame.empty


def test_infinity_is_not_missing_or_equal() -> None:
    frame = pd.DataFrame({"value": [float("inf")]})
    result = compare_frames(frame, frame)
    assert not result.ok
    assert any(d.code == "COMPARE_VALUE_MISMATCH" for d in result.diagnostics)


def test_keyness_pivots_numeric_frequency_values() -> None:
    frame = pd.DataFrame({"word": ["a", "b", "a", "b"], "freq": ["2", "3", "4", "1"], "corpus": ["x", "x", "y", "y"]})
    result = log_likelihood(frame, "word", "freq", corpus_col="corpus")
    assert result.ok
    out = result.unwrap().frame
    assert set(out["Freq x"]) == {3, 2}
    assert set(out["Freq y"]) == {1, 4}


def test_negative_smoothing_is_a_diagnostic() -> None:
    frame = pd.DataFrame({"word": ["a", "b"], "x": [1, 1], "y": [1, 1]})
    result = log_likelihood(frame, "word", "x", "y", smoothing=-0.5)
    assert result.value is None
    assert result.errors[0].code == "STATS_BAD_SMOOTHING"


def test_group_statistics_reject_infinity() -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0, float("inf"), 4.0, 5.0, 6.0], "group": ["a"] * 3 + ["b"] * 3})
    result = mann_whitney(frame, "value", "group")
    assert result.value is None
    assert result.errors[0].code == "STATS_NONFINITE_VALUE"


def test_non_dense_dependency_ids_are_not_rebased() -> None:
    frame = pd.DataFrame(
        [
            [1, "a", "a", "NN", "O", 0, "root", 1, 1, "1", "d"],
            [7, "b", "b", "NN", "O", 1, "dep", 2, 1, "1", "d"],
        ],
        columns=[
            "ID",
            "Form",
            "Lemma",
            "POS",
            "NER",
            "Head",
            "DepRel",
            "Record ID",
            "Sentence ID",
            "Document ID",
            "Document",
        ],
    )
    result = complexity_run(frame)
    assert result.ok
    assert result.unwrap().frame.iloc[0]["Mean Dependency Distance"] == 6.0


def test_filename_cycle_preserves_source_identity(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    plan = RenamePlan(
        rows=(RenameRow("a.txt", "b.txt", "", "rename"), RenameRow("b.txt", "a.txt", "", "rename")),
        directory=tmp_path,
    )
    result = apply_renames(plan, dry_run=False)
    assert result.ok
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "B"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "A"


def test_filename_plan_detects_source_change(tmp_path: Path) -> None:
    source = tmp_path / "a b.txt"
    source.write_text("before", encoding="utf-8")
    plan = plan_renames([source]).unwrap()
    source.write_text("after", encoding="utf-8")
    result = apply_renames(plan, dry_run=False)
    assert result.ok and result.unwrap() == []
    assert any(d.code == "FILENAMES_SOURCE_CHANGED" for d in result.diagnostics)
    assert source.exists()


def test_changed_cycle_source_cannot_be_overwritten(tmp_path: Path) -> None:
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    source_a.write_text("A", encoding="utf-8")
    source_b.write_text("B", encoding="utf-8")
    plan = RenamePlan(
        rows=(
            RenameRow("a.txt", "b.txt", "", "rename", hash_file(source_a), source_a.stat().st_size),
            RenameRow("b.txt", "a.txt", "", "rename", hash_file(source_b), source_b.stat().st_size),
        ),
        directory=tmp_path,
    )
    source_a.write_text("changed", encoding="utf-8")
    result = apply_renames(plan, dry_run=False)
    assert result.ok and result.unwrap() == []
    assert source_a.read_text(encoding="utf-8") == "changed"
    assert source_b.read_text(encoding="utf-8") == "B"
    assert any(d.code == "FILENAMES_SKIPPED" for d in result.diagnostics)


def test_corpus_validation_non_txt_ext_exits_zero(tmp_path: Path) -> None:
    """--ext selects the validated suffix; the corpus reader must follow it.

    A .csv corpus validated with --ext .csv is a success (exit 0), not a
    failure triggered by the reader's default *.txt pattern.
    """
    from tools.corpus_validation import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    out = tmp_path / "out"
    assert main([str(corpus), str(out), "--ext", ".csv"]) == 0
    run_dir = next(out.iterdir())
    assert (run_dir / "validation.csv").is_file()


def test_corpus_validation_txt_corpus_exits_zero(tmp_path: Path) -> None:
    from tools.corpus_validation import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world", encoding="utf-8")
    out = tmp_path / "out"
    assert main([str(corpus), str(out)]) == 0


def test_corpus_validation_empty_dir_is_reported_not_failed(tmp_path: Path) -> None:
    """An empty corpus yields an empty validation frame; the run is fine."""
    from tools.corpus_validation import main

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    out = tmp_path / "out"
    assert main([str(corpus), str(out)]) == 0
