"""End-to-end: corpus -> parse -> wordlist -> envelope -> scan."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore[assignment]

from app.scanner import find_runs
from core.analysis.conll_wordlist import run as wordlist_run
from core.conll.schema import Col
from core.io.reader import read_corpus
from core.io.writer import OutputWriter
from core.pipelines.cache import PipelineCache
from core.pipelines.spacy_backend import build_spacy_pipeline, spacy_model_name
from tools.conll_wordlist import main as wordlist_main

FIXTURE = Path(__file__).parent / "fixtures" / "mini-corpus"


def _has_spacy_model() -> bool:
    try:
        spacy.load(spacy_model_name("en"))
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.model_integration,
    pytest.mark.skipif(
        not _has_spacy_model(),
        reason=(
            "spaCy model not installed — the suite hard-fails on missing models "
            f"(run: python -m spacy download {spacy_model_name('en')})"
        ),
    ),
]


def test_end_to_end_mini_corpus(tmp_path: Path) -> None:
    corpus_result = read_corpus(FIXTURE)
    assert corpus_result.ok, corpus_result.diagnostics
    corpus = corpus_result.unwrap()
    assert len(corpus) == 3

    cache = PipelineCache()
    cache.register("spacy", build_spacy_pipeline)  # type: ignore[arg-type]
    pipeline = cache.get("spacy", "en").unwrap()
    table_result = pipeline.parse(corpus)
    assert table_result.ok, table_result.diagnostics
    table = table_result.unwrap()
    assert len(table) > 0
    assert Col.FORM.value in table.columns
    assert Col.POS.value in table.columns

    word_result = wordlist_run(table, field=Col.FORM, category="all", top_n=10)
    assert word_result.ok, word_result.diagnostics
    frame = word_result.unwrap().to_frame()
    assert len(frame) > 0
    assert "Word" in frame.columns

    writer = OutputWriter(tmp_path / "out", tool="test_e2e", params={"test": True}, corpus=corpus)
    writer.write_table(frame, "wordlist.csv", kind="table")
    writer.add_diagnostics(*corpus_result.diagnostics, *table_result.diagnostics, *word_result.diagnostics)
    envelope_result = writer.finalize()
    assert envelope_result.ok, envelope_result.diagnostics
    envelope = envelope_result.unwrap()
    assert (writer.run_dir / "wordlist.csv").is_file()
    assert (writer.run_dir / "result.json").is_file()
    assert envelope.tool == "test_e2e"
    # The table, plus the plain-language readout the writer now records beside
    # every result. The count alone said 1 here long after it became 2.
    assert [artifact.path for artifact in envelope.artifacts] == ["wordlist.csv", "readout.md"]

    runs = find_runs(tmp_path / "out")
    assert len(runs) == 1
    assert runs[0].envelope.tool == "test_e2e"


def test_end_to_end_via_cli(tmp_path: Path) -> None:
    """The CLI produces the same shape as the direct API."""
    corpus_dir = FIXTURE
    out_dir = tmp_path / "out2"
    code = wordlist_main(
        [str(corpus_dir), str(out_dir), "--parser", "spacy", "--field", "lemma", "--category", "noun", "--top-n", "5"]
    )
    assert code == 0
    runs = find_runs(out_dir)
    assert len(runs) == 1
    assert (runs[0].run_dir / "wordlist.csv").is_file()
