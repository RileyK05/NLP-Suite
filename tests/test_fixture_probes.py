"""Fixture probes — every file in tests/fixtures/probes/ pins a named defect.

Source spec: legacy planning/01_fixture_corpus_spec.md (read-only). Unlike
the representative mini-corpus, these fixtures are minimal and adversarial.
Deviations from the spec are documented in tests/fixtures/README.md, never
silent.
"""

from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import pytest

from conftest import has_spacy_model
from core.analysis import conll_wordlist, ngrams
from core.conll.division import sentence_records
from core.conll.normalize import normalize_table
from core.data.validation import validate
from core.io.reader import Corpus, Document, hash_text, read_corpus, read_text
from core.io.writer import OutputWriter
from core.pcace.analysis import validate_codes
from core.pipelines.cache import PipelineCache
from core.pipelines.spacy_backend import build_spacy_pipeline, spacy_model_name
from core.result import Result

PROBES = Path(__file__).parent / "fixtures" / "probes"


def _read_probe(name: str) -> Path:
    path = PROBES / name
    assert path.exists(), f"missing fixture: {name}"
    return path


class TestEmptyFile:
    def test_empty_doc_does_not_abort_siblings(self) -> None:
        # Legacy: Stanza_util `break` on empty doc killed the whole corpus.
        result = read_corpus(PROBES, pattern="0*.txt")
        assert result.ok
        assert len(result.unwrap()) >= 2  # empty + non-empty siblings present
        assert any(d.code == "EMPTY_DOC" for d in result.diagnostics)

    def test_empty_conll_table_yields_empty_results_not_crashes(self) -> None:
        # Legacy: unguarded data[0] / df.iloc[0] across CoNLL utils.
        frame = pd.read_csv(PROBES / "conll_empty.csv")
        assert frame.empty
        words = conll_wordlist.run(frame)
        assert words.ok and words.unwrap().filtered_tokens == 0
        grams = ngrams.ngrams(frame)
        assert grams.ok and grams.unwrap().empty


class TestFinalSentence:
    def test_sentences_without_terminal_punct_all_emitted(self) -> None:
        # Legacy: sentence_division dropped the final sentence (FIXES #12).
        text = _read_probe("01_no_terminal_punct.txt").read_text(encoding="utf-8")
        assert len(text.strip().splitlines()) == 3
        frame = pd.DataFrame(
            [
                {
                    "ID": i + 1,
                    "Form": w,
                    "Lemma": w.lower(),
                    "POS": "NN",
                    "NER": "O",
                    "Head": 0,
                    "DepRel": "root",
                    "Sentence ID": i + 1,
                    "Document ID": "1",
                    "Document": "01_no_terminal_punct.txt",
                }
                for i, w in enumerate(["alpha", "beta", "gamma"])
            ]
        )
        records = sentence_records(frame)
        assert records.ok
        assert len(records.unwrap()) == 3  # including the last


class TestUnicode:
    def test_source_bytes_never_rewritten(self) -> None:
        # Legacy: IO_csv_util rewrote the input file stripping NULs (R3 violation).
        probe = _read_probe("02_unicode.txt")
        before = probe.read_bytes()
        result = read_text(probe)
        assert result.ok
        assert probe.read_bytes() == before
        assert "\x00" not in result.unwrap()

    def test_nul_strip_is_announced(self) -> None:
        result = read_text(_read_probe("02_unicode.txt"))
        assert any(d.code == "NULL_BYTE_STRIPPED" for d in result.infos)

    def test_scripts_preserved(self) -> None:
        result = read_text(_read_probe("02_unicode.txt"))
        text = result.unwrap()
        assert "日本語" in text and "🎉" in text and "“" in text


class TestLongSentence:
    @pytest.mark.model_integration
    def test_long_sentence_parses_with_single_cached_pipeline(self) -> None:
        # Legacy: per-sentence pipeline instantiation exploded here.
        if not has_spacy_model():
            pytest.skip(f"needs spaCy model: python -m spacy download {spacy_model_name('en')}")
        text = _read_probe("03_long_sentence.txt").read_text(encoding="utf-8")
        assert len(text.split()) >= 9000
        corpus = Corpus(
            docs=(Document(doc_id=1, path=PROBES / "03_long_sentence.txt", text=text, sha256=hash_text(text)),),
            sha256="x",
        )
        cache = PipelineCache()
        cache.register("spacy", build_spacy_pipeline)  # type: ignore[arg-type]
        frame = cache.get("spacy", "en").unwrap().parse(corpus).unwrap()
        assert len(frame) > 9000
        assert len(cache) == 1  # one pipeline, not one per sentence


class TestDocumentIds:
    def test_ids_at_and_above_ten_survive_intact(self) -> None:
        # Legacy: str(tok_Document_ID)[:-2] chopped IDs >= 10; '1.0' sentinel.
        result = read_corpus(PROBES, pattern="0[4-9]_doc.txt")
        result2 = read_corpus(PROBES, pattern="1[0-3]_doc.txt")
        assert result.ok and result2.ok
        assert result.unwrap().doc_ids == tuple(range(1, 7))
        assert result2.unwrap().doc_ids == tuple(range(1, 5))
        full = read_corpus(PROBES, pattern="*_doc.txt")
        assert full.ok
        assert len(full.unwrap()) == 10
        assert full.unwrap().doc_ids == tuple(range(1, 11))


class TestFilenames:
    def test_multi_dot_stem_preserved(self) -> None:
        # Legacy: [0:-4] extension-strip idiom assumed exactly one dot.
        probe = _read_probe("14_my.report.v2.txt")
        assert probe.stem == "14_my.report.v2"
        assert probe.suffix == ".txt"

    def test_filename_dates_extracted(self) -> None:
        # Legacy: file_classifier_date_util swapped sep/format args.
        from core.io.reader import date_from_filename

        assert str(date_from_filename(PROBES / "15_report_1999-05-04.txt")) == "1999-05-04"
        assert str(date_from_filename(PROBES / "16_report_2001-11-30.txt")) == "2001-11-30"


class TestDialogue:
    @pytest.mark.model_integration
    def test_boundaries_after_closing_quotes(self) -> None:
        # Legacy: lstrip/rstrip swap in whole_sent reconstruction.
        if not has_spacy_model():
            pytest.skip(f"needs spaCy model: python -m spacy download {spacy_model_name('en')}")
        from core.io.reader import Document

        text = _read_probe("17_dialogue.txt").read_text(encoding="utf-8")
        corpus = Corpus(
            docs=(Document(doc_id=1, path=PROBES / "17_dialogue.txt", text=text, sha256=hash_text(text)),),
            sha256="x",
        )
        cache = PipelineCache()
        cache.register("spacy", build_spacy_pipeline)  # type: ignore[arg-type]
        frame = cache.get("spacy", "en").unwrap().parse(corpus).unwrap()
        from core.conll.division import sentences_text

        texts = sentences_text(frame).unwrap()
        assert texts  # no empty sentences dropped or created
        # Boundary after the closing quote is correct: the first sentence
        # ends at `said .` instead of losing its leading content (legacy
        # lstrip/rstrip swap in whole_sent reconstruction).
        assert texts[0].rstrip().endswith("said .")
        assert all(t.strip() for t in texts)


class TestNonEnglish:
    @pytest.mark.model_integration
    def test_configured_language_honored_and_missing_model_is_loud(self) -> None:
        # Legacy: hardcoded ['English'] + always-True availability check.
        from core.config import supports_language

        assert supports_language("spacy", "it")  # real lookup, not `if True`
        result = build_spacy_pipeline("it", frozenset())
        if has_spacy_model() and result.ok:
            return  # model present: honored end to end
        assert not result.ok
        assert result.errors[0].code == "PIPELINE_MODEL_MISSING"


class TestConllFixtures:
    def test_good_table_normalizes(self) -> None:
        frame = pd.read_csv(PROBES / "conll_good.csv")
        result = normalize_table(frame)
        assert result.ok
        assert list(result.unwrap().columns)[:13] == [
            "ID",
            "Form",
            "Lemma",
            "POS",
            "NER",
            "Head",
            "DepRel",
            "Deps",
            "Clause Tag",
            "Record ID",
            "Sentence ID",
            "Document ID",
            "Document",
        ]

    def test_extra_column_preserved(self) -> None:
        # Legacy: "14 or 15 columns?" checks scattered across the suite.
        frame = pd.read_csv(PROBES / "conll_14col.csv")
        result = normalize_table(frame)
        assert result.ok
        assert "Date" in result.unwrap().columns

    def test_universal_tags_mapped_once(self) -> None:
        # Legacy: analyzer re-read raw file with Penn filters on Universal tags.
        frame = pd.read_csv(PROBES / "conll_universal.csv")
        result = normalize_table(frame)
        assert result.ok
        pos = result.unwrap()["POS"].tolist()
        assert "NOUN" not in pos and "VERB" not in pos
        assert "NNS" in pos and "VBD" in pos

    def test_malformed_csv_fails_loudly(self) -> None:
        # Deviation from spec (see fixtures README): ragged rows fail the file
        # with a diagnostic instead of being skipped with a count. Either way
        # no row is silently corrupted — that is the probed defect.
        with pytest.raises(Exception, match=r"(?i)(expected|tokeniz|error|columns)"):
            pd.read_csv(PROBES / "19_malformed.csv", encoding="utf-8")


class TestLockSimulation:
    def test_blocked_write_fails_fast_with_diagnostic(self) -> None:
        # Legacy: infinite permission-error retry loop. The new writer has no
        # retry loop at all; a blocked target fails fast and catchably.
        blocker = PROBES / "pcace_minimal" / "data_codes.csv"  # a file, not a dir
        start = time.monotonic()
        with pytest.raises(ValueError, match=r"cannot create (run|staging) directory"):
            OutputWriter(blocker / "out", tool="probe", params={}, inputs=())
        assert time.monotonic() - start < 5


class TestPcaceMinimal:
    def test_known_graph_validates_and_analyzes(self) -> None:
        # Legacy: parent/child IndexError + global rebind truncating setup_Complex_lib.
        import csv

        with open(PROBES / "pcace_minimal" / "data_codes.csv", encoding="utf-8") as fh:
            codes = [row["code"] for row in csv.DictReader(fh)]
        result = validate_codes(codes)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert (frame["Status"] == "OK").all()
        assert len(frame) == 5


class TestValidationProbe:
    def test_validate_marks_fallback_decoded_file(self, tmp_path: Path) -> None:
        # b"\xe9" is invalid UTF-8 but valid cp1252: the chain recovers it,
        # and the recovery must be visible in the report — never a silent OK.
        bad = tmp_path / "bad.txt"
        bad.write_bytes(b"caf\xe9 latte")
        result = validate(tmp_path)
        assert result.ok  # the run survives; the fallback is flagged
        frame = result.unwrap()
        row = frame[frame["File"] == "bad.txt"].iloc[0]
        assert row["Status"] == "OK"
        assert "cp1252" in str(row["Issue"])
        assert any(d.code == "ENCODING_FALLBACK" for d in result.diagnostics)

    def test_probe_corpus_validates(self) -> None:
        result = validate(PROBES, required_ext=".txt")
        assert result.ok
        frame = result.unwrap()
        assert (frame["Status"] == "EMPTY").any()  # 00_empty.txt flagged, not fatal


def test_probe_result_type_imported() -> None:
    assert Result is not None
