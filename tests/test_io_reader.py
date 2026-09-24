"""Corpus intake contracts (ARCHITECTURE.md R3, R6, R8, section 2.1).

Anti-regression targets, all named legacy defects:

* `doc_id` derived by slicing ID strings (`str(id)[:-2]`), which breaks past
  nine documents. Here ids are assigned dense 1..N.
* The reader abandoning the rest of the corpus on the first empty file.
* The file checker crashing on a None encoding from chardet, then opening
  files with errors='ignore' — which defeats the check.
* The date classifier passing separator and format the wrong way round.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from core.io.reader import (
    ENCODING_CHAIN,
    Corpus,
    Document,
    corpus_fingerprint,
    date_from_filename,
    hash_file,
    hash_text,
    read_corpus,
    read_text,
)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Five documents, one of them empty, one with a date in its name."""
    (tmp_path / "01.txt").write_text("The president went to Italy.", encoding="utf-8")
    (tmp_path / "02.txt").write_text("He stayed for a while.", encoding="utf-8")
    (tmp_path / "03.txt").write_text("", encoding="utf-8")
    (tmp_path / "report_2024-01-15.txt").write_text("Dated document.", encoding="utf-8")
    (tmp_path / "04.txt").write_text("Ladies and gentlemen.", encoding="utf-8")
    return tmp_path


class TestHashing:
    def test_hash_text_is_stable(self) -> None:
        assert hash_text("abc") == hash_text("abc")

    def test_hash_text_differs_on_content(self) -> None:
        assert hash_text("abc") != hash_text("abd")

    def test_hash_file_matches_hash_of_contents(self, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_text("hello", encoding="utf-8")
        assert hash_file(target) == hash_text("hello")

    def test_hash_file_reads_in_chunks(self, tmp_path: Path) -> None:
        target = tmp_path / "big.txt"
        target.write_text("x" * 500_000, encoding="utf-8")
        assert hash_file(target) == hash_text("x" * 500_000)


class TestReadText:
    def test_reads_utf8(self, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_text("hello", encoding="utf-8")
        result = read_text(target)
        assert result.ok
        assert result.unwrap() == "hello"
        assert not result.diagnostics

    def test_missing_file_is_an_error_not_an_exception(self, tmp_path: Path) -> None:
        result = read_text(tmp_path / "nope.txt")
        assert not result.ok
        assert result.errors[0].code == "FILE_NOT_FOUND"

    def test_fallback_is_announced(self, tmp_path: Path) -> None:
        """latin-1 content that is not valid utf-8."""
        target = tmp_path / "a.txt"
        target.write_bytes(b"caf\xe9")
        result = read_text(target)
        assert result.ok
        assert any(d.code == "ENCODING_FALLBACK" for d in result.warnings)

    def test_explicit_encoding_is_honoured(self, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_bytes(b"caf\xe9")
        result = read_text(target, encoding="latin-1")
        assert result.unwrap() == "café"
        assert not any(d.code == "ENCODING_FALLBACK" for d in result.diagnostics)

    def test_failed_explicit_encoding_is_reported(self, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_bytes(b"caf\xe9")
        result = read_text(target, encoding="utf-8")
        assert not result.ok
        assert result.errors[0].code == "FILE_UNDECODABLE"

    def test_encoding_chain_terminates_in_latin1(self) -> None:
        assert ENCODING_CHAIN[-1] == "latin-1"


class TestDateFromFilename:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("2024-01-15_report.txt", date(2024, 1, 15)),
            ("2024_01_15_report.txt", date(2024, 1, 15)),
            ("20240115_report.txt", date(2024, 1, 15)),
            ("01-15-2024_report.txt", date(2024, 1, 15)),
            ("2024-01_report.txt", date(2024, 1, 1)),
            ("no_date_here.txt", None),
            ("12345.txt", None),
        ],
    )
    def test_extraction(self, name: str, expected: date | None) -> None:
        assert date_from_filename(Path(name)) == expected

    def test_unparseable_match_yields_none_rather_than_a_wrong_date(self) -> None:
        """9999-99-99 looks like a date but is not one."""
        assert date_from_filename(Path("9999-99-99.txt")) is None


class TestReadCorpus:
    def test_reads_every_document(self, corpus_dir: Path) -> None:
        result = read_corpus(corpus_dir)
        assert result.ok
        assert len(result.unwrap()) == 5

    def test_doc_ids_are_dense_and_assigned(self, corpus_dir: Path) -> None:
        """Legacy derived ids by slicing strings, breaking past nine documents."""
        result = read_corpus(corpus_dir)
        assert result.unwrap().doc_ids == (1, 2, 3, 4, 5)

    def test_documents_are_sorted_by_path(self, corpus_dir: Path) -> None:
        result = read_corpus(corpus_dir)
        names = [doc.name for doc in result.unwrap()]
        assert names == ["01.txt", "02.txt", "03.txt", "04.txt", "report_2024-01-15.txt"]

    def test_empty_document_warns_but_does_not_abort(self, corpus_dir: Path) -> None:
        """Legacy broke out of the loop here, dropping every later document."""
        result = read_corpus(corpus_dir)
        assert result.ok
        empty = [d for d in result.warnings if d.code == "EMPTY_DOC"]
        assert len(empty) == 1
        assert len(result.unwrap()) == 5

    def test_date_is_extracted_when_present(self, corpus_dir: Path) -> None:
        result = read_corpus(corpus_dir)
        dated = [doc for doc in result.unwrap() if doc.date is not None]
        assert len(dated) == 1
        assert dated[0].date == date(2024, 1, 15)

    def test_missing_directory_is_an_error(self, tmp_path: Path) -> None:
        result = read_corpus(tmp_path / "nope")
        assert not result.ok
        assert result.errors[0].code == "CORPUS_DIR_MISSING"

    def test_empty_directory_is_an_error(self, tmp_path: Path) -> None:
        result = read_corpus(tmp_path)
        assert not result.ok
        assert result.errors[0].code == "CORPUS_EMPTY"

    def test_pattern_filters_files(self, corpus_dir: Path) -> None:
        (corpus_dir / "notes.md").write_text("# not corpus", encoding="utf-8")
        result = read_corpus(corpus_dir, pattern="*.txt")
        assert len(result.unwrap()) == 5

    def test_unreadable_file_is_skipped_not_fatal(self, corpus_dir: Path) -> None:
        """A file that cannot be decoded is recorded; the corpus continues."""
        (corpus_dir / "bad.txt").write_bytes(b"\xff\xfe\x00\x81\x81")
        result = read_corpus(corpus_dir, encoding="utf-8")
        assert result.value is not None
        assert len(result.unwrap()) == 5
        assert any(d.code == "FILE_UNDECODABLE" for d in result.errors)
        assert not result.ok

    def test_recursive_by_default(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "deep.txt").write_text("deep", encoding="utf-8")
        assert len(read_corpus(tmp_path).unwrap()) == 1
        assert not read_corpus(tmp_path, recursive=False).ok

    def test_fingerprint_is_order_independent(self, corpus_dir: Path) -> None:
        first = read_corpus(corpus_dir).unwrap()
        rebuilt = Corpus(docs=tuple(reversed(first.docs)), sha256=first.sha256)
        assert corpus_fingerprint(rebuilt.docs) == first.sha256

    def test_fingerprint_changes_when_content_changes(self, corpus_dir: Path) -> None:
        first = read_corpus(corpus_dir).unwrap().sha256
        (corpus_dir / "01.txt").write_text("different", encoding="utf-8")
        assert read_corpus(corpus_dir).unwrap().sha256 != first


class TestDocumentShape:
    def test_is_frozen(self, corpus_dir: Path) -> None:
        doc = read_corpus(corpus_dir).unwrap().docs[0]
        with pytest.raises(FrozenInstanceError):
            doc.text = "mutated"  # type: ignore[misc]

    def test_is_empty(self, corpus_dir: Path) -> None:
        docs = {doc.name: doc for doc in read_corpus(corpus_dir).unwrap()}
        assert docs["03.txt"].is_empty
        assert not docs["01.txt"].is_empty

    def test_corpus_exposes_texts_and_paths(self, corpus_dir: Path) -> None:
        corpus = read_corpus(corpus_dir).unwrap()
        assert len(corpus.texts) == len(corpus)
        assert len(corpus.paths) == len(corpus)

    def test_document_carries_its_hash(self, corpus_dir: Path) -> None:
        doc = read_corpus(corpus_dir).unwrap().docs[0]
        assert doc.sha256 == hash_text(doc.text)

    def test_constructed_directly(self) -> None:
        doc = Document(doc_id=1, path=Path("a.txt"), text="hi")
        assert doc.date is None
        assert doc.name == "a.txt"
