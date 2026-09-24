"""Regression tests for the systematic review fixes.

Each test pins a concrete defect that shipped in the reviewed build:
infinite loops, swallowed decode errors, NaN leaking into generated XML,
lexicographic sorts that scramble 10+ document corpora, envelope artifacts
advertised but never written, and safety guards that rejected legitimate
inputs.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from core.analysis.html_annotator import annotate
from core.analysis.ner import entity_timeline
from core.artifacts.envelope import InputRef
from core.conll.schema import Col
from core.data.sql import _check_sql, query as sql_query
from core.file_ops.classifier import classify_by_ner_pattern
from core.file_ops.cleaner import clean_text
from core.file_ops.converter import convert_file
from core.file_ops.merger import merge_files
from core.file_ops.search import search_in_text
from core.file_ops.splitter import split_text
from core.gis.mapping import kml
from core.io.reader import Corpus, Document, hash_text
from core.io.writer import OutputWriter
from core.narrative.arcs import emotion_arc
from core.viz.shapes import sankey_html, story_shape
from core.viz.wordcloud_gephi import gephi_gexf

# ---------------------------------------------------------------------------
# splitter
# ---------------------------------------------------------------------------


class TestSplitter:
    def test_overlap_larger_than_chunk_is_rejected_not_an_endless_loop(self) -> None:
        # Before: start = end - overlap <= start, so by_length never advanced
        # and the CLI hung forever.
        result = split_text("abcdefgh", method="by_length", chunk_size=3, overlap=3)
        assert not result.ok
        assert result.errors[0].code == "SPLIT_BAD_OVERLAP"

    def test_negative_overlap_is_rejected(self) -> None:
        result = split_text("abcdefgh", method="by_length", chunk_size=3, overlap=-1)
        assert not result.ok
        assert result.errors[0].code == "SPLIT_BAD_OVERLAP"

    def test_valid_overlap_still_splits_with_overlap(self) -> None:
        result = split_text("abcdef", method="by_length", chunk_size=4, overlap=2)
        assert result.ok
        assert result.unwrap() == ["abcd", "cdef"]

    def test_by_keyword_splits_before_every_occurrence(self) -> None:
        # Before: indices[1:] merged the first keyword into the leading chunk.
        result = split_text("intro KEY one KEY two", method="by_keyword", keyword="KEY")
        assert result.ok
        assert result.unwrap() == ["intro ", "KEY one ", "KEY two"]

    def test_by_keyword_at_position_zero_makes_no_empty_chunk(self) -> None:
        result = split_text("KEY one KEY two", method="by_keyword", keyword="KEY")
        assert result.ok
        assert result.unwrap() == ["KEY one ", "KEY two"]


# ---------------------------------------------------------------------------
# merger / converter — decode errors must be Results, not exceptions
# ---------------------------------------------------------------------------


class TestDecodeErrorsAreResults:
    def test_merge_files_reports_undecodable_input(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.txt"
        bad.write_bytes(b"caf\xe9 latte")  # cp1252 bytes, invalid UTF-8
        good = tmp_path / "good.txt"
        good.write_text("ok", encoding="utf-8")
        result = merge_files([good, bad], tmp_path / "out.txt")
        assert not result.ok
        assert result.errors[0].code == "MERGE_READ_FAILED"

    def test_convert_file_reports_undecodable_input(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"a,b\ncaf\xe9,1\n")
        result = convert_file(bad, tmp_path / "out.csv")
        assert not result.ok
        assert result.errors[0].code == "CONVERT_READ_FAILED"


# ---------------------------------------------------------------------------
# search — invalid regex must fail even against empty text
# ---------------------------------------------------------------------------


class TestSearchRegex:
    def test_bad_regex_fails_on_empty_text(self) -> None:
        # Before: compilation sat inside the per-line loop, so empty text
        # meant the pattern was never compiled and the bad regex "succeeded".
        result = search_in_text("", query="(", use_regex=True)
        assert not result.ok
        assert result.errors[0].code == "SEARCH_BAD_REGEX"

    def test_good_regex_still_matches(self) -> None:
        result = search_in_text("alpha\nbeta\nalphabet", query="alpha", use_regex=True)
        assert result.ok
        assert result.unwrap() == [(1, "alpha"), (3, "alphabet")]


# ---------------------------------------------------------------------------
# classifier — empty results keep their schema
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_no_matches_frame_has_columns(self) -> None:
        result = classify_by_ner_pattern("no capitals here")
        assert result.ok
        frame = result.unwrap()
        assert list(frame.columns) == ["entity", "start"]
        assert frame.empty


# ---------------------------------------------------------------------------
# cleaner — dead multi-newline collapse removed
# ---------------------------------------------------------------------------


class TestCleaner:
    def test_remove_empty_lines_yields_single_newlines(self) -> None:
        result = clean_text("a\n\n\nb", remove_empty_lines=True)
        assert result.ok
        assert result.unwrap() == "a\nb"


# ---------------------------------------------------------------------------
# sql — the multi-statement guard
# ---------------------------------------------------------------------------


class TestSqlGuard:
    def test_two_statements_with_single_semicolon_are_rejected(self) -> None:
        diag = _check_sql("SELECT 1; DELETE FROM t")
        assert diag is not None
        assert diag.code == "SQL_MULTI_STATEMENT"

    def test_single_statement_with_trailing_semicolon_is_fine(self) -> None:
        assert _check_sql("SELECT 1;") is None

    def test_query_rejects_two_selects(self, tmp_path: Path) -> None:
        db = tmp_path / "db.sqlite"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
        result = sql_query(db, "SELECT * FROM t; SELECT * FROM t")
        assert not result.ok
        assert result.errors[0].code == "SQL_MULTI_STATEMENT"


# ---------------------------------------------------------------------------
# ner — sentence ids must be ordered numerically
# ---------------------------------------------------------------------------


def _timeline_frame(sentence_count: int) -> pd.DataFrame:
    # The same entity mentioned in every sentence, so its timeline row
    # aggregates mentions across the whole document.
    rows = [
        {
            "ID": i + 1,
            "Form": "Paris",
            "Lemma": "paris",
            "POS": "NNP",
            "NER": "B-GPE",
            "Head": 0,
            "DepRel": "root",
            "Sentence ID": i + 1,
            "Document ID": "1",
            "Document": "doc.txt",
        }
        for i in range(sentence_count)
    ]
    return pd.DataFrame(rows)


class TestEntityTimeline:
    def test_first_last_sentence_with_twelve_sentences(self) -> None:
        # Before: min/max used key=str, so "First Sentence" was 1 and
        # "Last Sentence" was 9 for a 12-sentence document.
        result = entity_timeline(_timeline_frame(12))
        assert result.ok
        frame = result.unwrap()
        assert int(frame["First Sentence"].iloc[0]) == 1
        assert int(frame["Last Sentence"].iloc[0]) == 12


# ---------------------------------------------------------------------------
# html_annotator — single-pass annotation
# ---------------------------------------------------------------------------


class TestAnnotate:
    def test_later_term_cannot_match_inside_injected_markup(self) -> None:
        # Sequential re.sub let "tag" match inside the attribute value of the
        # markup injected for an earlier term, producing broken output.
        result = annotate("say tag now", {"tag": "T"})
        assert result.ok
        assert result.unwrap() == 'say <mark data-tag="T">tag</mark> now'

    def test_longest_term_wins(self) -> None:
        result = annotate("brown fox", {"brown": "B", "brown fox": "ANIMAL"})
        assert result.ok
        assert result.unwrap() == '<mark data-tag="ANIMAL">brown fox</mark>'

    def test_nested_dictionary_terms_stay_wellformed(self) -> None:
        result = annotate("he her", {"he": "M", "her": "F"})
        assert result.ok
        text = result.unwrap()
        assert text == '<mark data-tag="M">he</mark> <mark data-tag="F">her</mark>'


# ---------------------------------------------------------------------------
# writer — R3 guard accepts multi-directory inputs, still refuses corpora
# ---------------------------------------------------------------------------


class TestWriterGuard:
    def test_inputs_from_sibling_directories_are_accepted(self, tmp_path: Path) -> None:
        # Before: commonpath of sibling dirs is their common parent, which
        # contains every conceivable output, so the writer rejected them all.
        dir_a = tmp_path / "dirA"
        dir_b = tmp_path / "dirB"
        dir_a.mkdir()
        dir_b.mkdir()
        docs = (
            Document(doc_id=1, path=dir_a / "a.txt", text="x", sha256=hash_text("x")),
            Document(doc_id=2, path=dir_b / "b.txt", text="y", sha256=hash_text("y")),
        )
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, corpus=Corpus(docs=docs, sha256="s"))
        assert writer.run_dir.is_dir()
        assert len(writer.finalize().unwrap().inputs) == 2

    def test_output_inside_input_directory_is_still_refused(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "a.txt").write_text("hi", encoding="utf-8")
        docs = (Document(doc_id=1, path=corpus_dir / "a.txt", text="hi", sha256="abc"),)
        with pytest.raises(ValueError):
            OutputWriter(corpus_dir / "out", tool="demo", params={}, corpus=Corpus(docs=docs, sha256="s"))

    def test_input_refs_accepted_for_scattered_files(self, tmp_path: Path) -> None:
        # Same guarantee for the raw InputRef path used by tools without a corpus.
        refs = (
            InputRef(path=str(tmp_path / "one" / "a.txt"), sha256="a"),
            InputRef(path=str(tmp_path / "two" / "b.txt"), sha256="b"),
        )
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, inputs=refs)
        assert writer.run_dir.is_dir()


# ---------------------------------------------------------------------------
# viz/gis — NaN is not an exception
# ---------------------------------------------------------------------------


class TestNanHandling:
    def test_gexf_nan_weight_falls_back_to_one(self) -> None:
        frame = pd.DataFrame({"Source": ["a", "a"], "Target": ["b", "b"], "Weight": [2.0, float("nan")]})
        result = gephi_gexf(frame, "Source", "Target", weight_col="Weight")
        assert result.ok
        assert 'weight="nan"' not in result.unwrap()

    def test_kml_skips_nan_coordinates(self) -> None:
        frame = pd.DataFrame({"Place": ["Paris", "Nowhere"], "Lat": [48.85, float("nan")], "Lon": [2.35, float("nan")]})
        result = kml(frame)
        assert result.ok
        assert "nan" not in result.unwrap()

    def test_sankey_missing_value_column_fails(self) -> None:
        frame = pd.DataFrame({"Source": ["a"], "Target": ["b"]})
        result = sankey_html(frame, "Source", "Target", value="Missing")
        assert not result.ok
        assert result.errors[0].code == "SANKEY_BAD_COLUMN"


# ---------------------------------------------------------------------------
# narrative — vader failures propagate; document order is preserved
# ---------------------------------------------------------------------------


class TestEmotionArc:
    def test_bad_field_fails_instead_of_silent_zero_arc(self) -> None:
        frame = pd.DataFrame(
            {
                "ID": [1],
                "Form": ["good"],
                "Lemma": ["good"],
                "POS": ["JJ"],
                "NER": ["O"],
                "Head": [0],
                "DepRel": ["root"],
                "Sentence ID": [1],
                "Document ID": ["1"],
                "Document": ["d.txt"],
            }
        )
        result = emotion_arc(frame, field=Col.POS)
        assert not result.ok
        assert any(d.code == "VADER_BAD_FIELD" for d in result.errors)


# ---------------------------------------------------------------------------
# shapes — document order survives 10+ documents
# ---------------------------------------------------------------------------


class TestStoryShapeOrder:
    def test_rows_stay_in_document_order_past_nine_documents(self) -> None:

        rows: list[dict[str, object]] = []
        token_index = 0
        for doc in range(1, 13):
            for sent in (1, 2):
                for _ in range(2):
                    token_index += 1
                    rows.append(
                        {
                            "ID": token_index,
                            "Form": "word",
                            "Lemma": "word",
                            "POS": "NN",
                            "NER": "O",
                            "Head": 0,
                            "DepRel": "root",
                            "Sentence ID": sent,
                            "Document ID": str(doc),
                            "Document": f"doc{doc}.txt",
                        }
                    )
        frame = pd.DataFrame(rows)
        result = story_shape(frame)
        assert result.ok
        shape = result.unwrap()
        assert list(shape["Document ID"].astype(int)) == sorted(int(d) for d in shape["Document ID"])
        # First appearance order: doc 1, 2, ..., 12 — not "1", "10", "11", "12", "2".
        assert shape["Document ID"].iloc[0] == "1"
        assert shape["Document ID"].iloc[-1] == "12"


# ---------------------------------------------------------------------------
# envelope — write_json rounds nested floats like the envelope does
# ---------------------------------------------------------------------------


class TestWriteJsonNestedRounding:
    def test_nested_floats_are_rounded(self, tmp_path: Path) -> None:
        writer = OutputWriter(tmp_path / "out", tool="demo", params={}, inputs=())
        result = writer.write_json({"outer": {"value": 1.23456789}}, "data.json")
        assert result.ok
        assert "1.234568" in (writer.run_dir / "data.json").read_text(encoding="utf-8")
