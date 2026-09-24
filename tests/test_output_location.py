"""Where a run is allowed to write, and why the two cases differ.

R3 exists so a run cannot write into a directory it reads *as a whole*: output
landing inside a corpus folder is picked up as corpus content by the next run
over that folder, which silently corrupts every later result.

The rule was being applied to single named files as well, by treating each
input file's parent directory as off-limits. That made ordinary things
impossible:

* ``nlp-suite charts data.csv out/`` was refused whenever ``out/`` sat beside
  ``data.csv`` -- the most natural place to put it.
* a wordlist in your home directory made your home directory unwritable for
  ``spellcheck``.

A named file is one file, not a folder being scanned. The run directory is
freshly named and cannot collide with anything already there, so protecting the
file itself is enough.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.io.reader import Corpus, Document, hash_text
from core.io.writer import OutputWriter


def _corpus(root: Path) -> Corpus:
    root.mkdir(parents=True, exist_ok=True)
    docs = []
    for index, name in enumerate(["a.txt", "b.txt"], start=1):
        path = root / name
        path.write_text("some words here", encoding="utf-8")
        docs.append(Document(doc_id=index, path=path, text="some words here", sha256=hash_text(name)))
    return Corpus(docs=tuple(docs), sha256="corpus")


class TestACorpusDirectoryStaysProtected:
    def test_output_inside_the_corpus_is_refused(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path / "corpus")
        with pytest.raises(ValueError, match="inside input directory"):
            OutputWriter(tmp_path / "corpus" / "out", tool="demo", params={}, corpus=corpus)

    def test_output_deeper_inside_the_corpus_is_refused(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path / "corpus")
        with pytest.raises(ValueError, match="inside input directory"):
            OutputWriter(tmp_path / "corpus" / "nested" / "deep", tool="demo", params={}, corpus=corpus)

    def test_a_sibling_of_the_corpus_is_fine(self, tmp_path: Path) -> None:
        corpus = _corpus(tmp_path / "corpus")
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=corpus)
        assert writer.write_table(pd.DataFrame({"a": [1]}), "t.csv").ok

    def test_a_directory_named_as_an_input_stays_protected(self, tmp_path: Path) -> None:
        """Refused; which of the guards catches it first is not the contract."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "a.txt").write_text("hi", encoding="utf-8")
        with pytest.raises(ValueError) as refusal:
            OutputWriter(tmp_path / "source" / "out", tool="demo", params={}, inputs=(source,))
        assert "source" in str(refusal.value)


class TestANamedFileDoesNotLockItsFolder:
    def test_output_may_sit_beside_the_csv_it_analyses(self, tmp_path: Path) -> None:
        """The chart case: data.csv and out/ in one project folder."""
        data = tmp_path / "project" / "data.csv"
        data.parent.mkdir(parents=True)
        data.write_text("Word,Count\nthe,81\n", encoding="utf-8")
        writer = OutputWriter(tmp_path / "project" / "out", tool="charts", params={}, inputs=(data,))
        assert writer.write_table(pd.DataFrame({"a": [1]}), "t.csv").ok

    def test_a_resource_file_does_not_lock_the_output_tree(self, tmp_path: Path) -> None:
        """The spellcheck case: a wordlist beside where you want results."""
        wordlist = tmp_path / "work" / "words.txt"
        wordlist.parent.mkdir(parents=True)
        wordlist.write_text("the\nharvest\n", encoding="utf-8")
        corpus = _corpus(tmp_path / "corpus")
        writer = OutputWriter(
            tmp_path / "work" / "out", tool="spellcheck", params={}, inputs=(wordlist,), corpus=corpus
        )
        assert writer.write_table(pd.DataFrame({"a": [1]}), "t.csv").ok

    def test_the_file_itself_is_still_never_the_output_root(self, tmp_path: Path) -> None:
        data = tmp_path / "data.csv"
        data.write_text("Word,Count\nthe,81\n", encoding="utf-8")
        with pytest.raises(ValueError) as refusal:
            OutputWriter(data, tool="charts", params={}, inputs=(data,))
        assert "data.csv" in str(refusal.value)

    def test_the_run_directory_may_not_contain_the_input(self, tmp_path: Path) -> None:
        nested = tmp_path / "out" / "data.csv"
        nested.parent.mkdir(parents=True)
        nested.write_text("Word,Count\nthe,81\n", encoding="utf-8")
        with pytest.raises(ValueError):
            OutputWriter(tmp_path / "out" / "data.csv" / "sub", tool="charts", params={}, inputs=(nested,))


class TestProvenanceIsUnaffected:
    def test_a_resource_and_the_corpus_are_both_recorded(self, tmp_path: Path) -> None:
        """Relaxing the guard must not quietly drop an input from the record (R8)."""
        wordlist = tmp_path / "work" / "words.txt"
        wordlist.parent.mkdir(parents=True)
        wordlist.write_text("the\n", encoding="utf-8")
        corpus = _corpus(tmp_path / "corpus")
        writer = OutputWriter(
            tmp_path / "work" / "out", tool="spellcheck", params={}, inputs=(wordlist,), corpus=corpus
        )
        writer.write_table(pd.DataFrame({"a": [1]}), "t.csv")
        envelope = writer.finalize().unwrap()
        recorded = {Path(ref.path).name for ref in envelope.inputs}
        assert recorded == {"a.txt", "b.txt", "words.txt"}
        assert all(ref.sha256 for ref in envelope.inputs), "every input needs a real hash"
