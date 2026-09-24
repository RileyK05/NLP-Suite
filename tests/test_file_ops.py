"""File ops — checker, cleaner, converter, splitter, merger, classifier, search."""

from __future__ import annotations

from pathlib import Path

from core.file_ops.checker import check_file
from core.file_ops.classifier import classify_by_date, classify_by_ner_pattern
from core.file_ops.cleaner import clean_text
from core.file_ops.converter import convert_table_to_text, convert_text_to_table
from core.file_ops.merger import match_files, merge_files, merge_texts
from core.file_ops.search import filename_sanitize, sample_corpus, search_in_text
from core.file_ops.splitter import split_text
from core.io.reader import Corpus, Document, hash_text


def test_checker_ok(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    result = check_file(target)
    assert result.ok
    assert result.unwrap().ok
    assert not result.unwrap().is_empty


def test_checker_missing() -> None:
    result = check_file(Path("/nonexistent/file.txt"))
    assert not result.ok
    assert result.errors[0].code == "FILE_NOT_FOUND"


def test_cleaner_normalizes_whitespace() -> None:
    result = clean_text("hello   world \n\n\n next", normalize_whitespace=True)
    assert result.ok
    assert "   " not in result.unwrap()
    assert result.unwrap().strip() == result.unwrap()


def test_cleaner_empty() -> None:
    assert clean_text("").ok


def test_converter_round_trip() -> None:
    text = "a,b\n1,2\n3,4\n"
    table = convert_text_to_table(text).unwrap()
    assert list(table.columns) == ["a", "b"]
    back = convert_table_to_text(table).unwrap()
    assert "a,b" in back


def test_splitter_by_length() -> None:
    result = split_text("abcdefgh", method="by_length", chunk_size=3)
    assert result.unwrap() == ["abc", "def", "gh"]


def test_splitter_by_words() -> None:
    result = split_text("one two three four five", method="by_words", chunk_size=2)
    assert len(result.unwrap()) == 3


def test_splitter_bad_method() -> None:
    result = split_text("hi", method="unknown")
    assert not result.ok


def test_match_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    assert len(match_files(tmp_path, "*.txt")) == 2


def test_merge_texts() -> None:
    result = merge_texts(["a", "b"], separator="\n")
    assert result.unwrap() == "a\nb"


def test_merge_files(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    out = tmp_path / "out.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("world", encoding="utf-8")
    result = merge_files([a, b], out, separator=" ")
    assert result.ok
    assert out.read_text(encoding="utf-8") == "hello world"


def test_classify_by_date(tmp_path: Path) -> None:
    paths = [tmp_path / "report_2024-01-15.txt", tmp_path / "notes.txt"]
    result = classify_by_date(paths)
    assert result.ok
    frame = result.unwrap()
    assert frame.iloc[0]["date"] == "2024-01-15"
    assert frame.iloc[1]["date"] == ""


def test_classify_by_ner_pattern() -> None:
    result = classify_by_ner_pattern("Alice went to Paris", pattern=r"[A-Z][a-z]+")
    assert result.ok
    assert len(result.unwrap()) >= 2


def test_search_in_text() -> None:
    result = search_in_text("hello world\nhello again", query="hello")
    assert len(result.unwrap()) == 2
    assert result.unwrap()[0][0] == 1


def test_filename_sanitize() -> None:
    assert filename_sanitize("a/b:c") == "a_b_c"
    assert filename_sanitize("") == "untitled"


def test_sample_corpus() -> None:
    docs = tuple(
        Document(doc_id=i + 1, path=Path(f"{i}.txt"), text=f"doc {i}", sha256=hash_text(f"doc {i}")) for i in range(5)
    )
    corpus = Corpus(docs=docs, sha256="x")
    result = sample_corpus(corpus, k=3, seed=0)
    assert result.ok
    assert len(result.unwrap()) == 3
    assert result.unwrap().doc_ids == (1, 2, 3)

    bad = sample_corpus(corpus, k=10)
    assert not bad.ok
