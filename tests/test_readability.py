"""FR-2.1 readability — counts, formulas, clamps, short-text rules, CLI.

Parity oracle: the legacy pure functions in
``statistics_corpus_readability_util.py``, loaded with stubbed GUI/IO imports
(the module only needs them for its installer guard and driver). The oracle
is legacy code, not a copy of this implementation. Skipped where the pinned
oracle checkout is absent (see docs/LEGACY_ENVIRONMENT.md).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import types

import pandas as pd
import pytest

from core.analysis import readability as R
from core.io.reader import Corpus, Document, hash_text

ORACLE = (
    Path(__file__).resolve().parent.parent.parent / "NLP-Suite-1.6.38" / "src" / "statistics_corpus_readability_util.py"
)
needs_oracle = pytest.mark.skipif(not ORACLE.is_file(), reason="legacy oracle checkout absent")


def load_legacy_oracle():  # type: ignore[no-untyped-def]
    """Import the legacy readability module with its GUI/IO deps stubbed."""
    import importlib.util

    for name in [
        "IO_libraries_util",
        "IO_csv_util",
        "IO_files_util",
        "IO_user_interface_util",
        "statistics_statistical_tests_util",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["IO_libraries_util"].install_all_Python_packages = lambda *a, **k: True
    gui = sys.modules.setdefault("GUI_util", types.ModuleType("GUI_util"))
    gui.window = None
    spec = importlib.util.spec_from_file_location("legacy_readability", ORACLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus(texts: list[str]) -> Corpus:
    docs = tuple(
        Document(doc_id=i + 1, path=Path(f"doc{i + 1}.txt"), text=text, sha256=hash_text(text))
        for i, text in enumerate(texts)
    )
    return Corpus(docs=docs, sha256="x")


TEXTS = [
    "The cat sat. Dogs barked loudly!",
    "Reading is to the mind what exercise is to the body. It keeps the intellect sharp and the imagination alive.",
    "Don't stop believin'. Hold on to that feelin'!",
]


class TestSyllableParity:
    @needs_oracle
    def test_matches_legacy_on_word_list(self) -> None:
        legacy = load_legacy_oracle()
        words = [
            "I",
            "a",
            "the",
            "agreed",
            "horses",
            "every",
            "queue",
            "don't",
            "rhythm",
            "extraordinary",
            "hippopotamus",
            "created",
            "baked",
            "people",
            "fire",
            "strengths",
        ]
        for word in words:
            assert R.count_syllables(word) == legacy._count_syllables(word), word


class TestCountsParity:
    @needs_oracle
    def test_matches_legacy_analyze_text(self) -> None:
        legacy = load_legacy_oracle()
        for text in TEXTS:
            assert R.analyze_text(text) == R.ReadabilityCounts(*legacy._analyze_text(text))


class TestFormulaParity:
    @needs_oracle
    def test_five_formulas_match_legacy_pre_clamp(self) -> None:
        legacy = load_legacy_oracle()
        cases = [(2, 6, 7, 0, 25), (1, 1, 1, 0, 1), (30, 600, 900, 120, 2700)]
        for n_sent, n_words, n_syll, n_poly, n_chars in cases:
            counts = R.ReadabilityCounts(n_sent, n_words, n_syll, n_poly, n_chars)
            assert R.flesch_reading_ease(counts) == pytest.approx(legacy._flesch_reading_ease(n_sent, n_words, n_syll))
            assert R.flesch_kincaid_grade(counts) == pytest.approx(
                legacy._flesch_kincaid_grade(n_sent, n_words, n_syll)
            )
            assert R.gunning_fog(counts) == pytest.approx(legacy._gunning_fog(n_sent, n_words, n_poly))
            assert R.coleman_liau(counts) == pytest.approx(legacy._coleman_liau(n_sent, n_words, n_chars))
            assert R.automated_readability_index(counts) == pytest.approx(legacy._ari(n_sent, n_words, n_chars))


class TestClampsAndBands:
    def test_fre_clamp_bounds(self) -> None:
        assert R.clamp_fre(200.0) == 121.0
        assert R.clamp_fre(-5.0) == 0.0
        assert R.clamp_fre(60.5) == 60.5

    def test_grade_clamp_bounds(self) -> None:
        assert R.clamp_grade(99.0) == 30.0
        assert R.clamp_grade(-1.0) == 0.0

    def test_interpretation_bands(self) -> None:
        assert R.interpret_fre(95.0) == "Very Easy (5th grade)"
        assert R.interpret_fre(85.0) == "Easy (6th grade)"
        assert R.interpret_fre(75.0) == "Fairly Easy (7th grade)"
        assert R.interpret_fre(65.0) == "Standard (8th-9th grade)"
        assert R.interpret_fre(55.0) == "Fairly Difficult (10th-12th grade)"
        assert R.interpret_fre(40.0) == "Difficult (College)"
        assert R.interpret_fre(10.0) == "Very Difficult (Graduate)"

    @needs_oracle
    def test_bands_match_legacy(self) -> None:
        legacy = load_legacy_oracle()
        for score in [0.0, 29.9, 30.0, 49.9, 50.0, 59.9, 60.0, 69.9, 70.0, 79.9, 80.0, 89.9, 90.0, 121.0]:
            assert R.interpret_fre(score) == legacy._interpret_fre(score), score


class TestSmog:
    def test_zero_polysyllables_gives_base_constant(self) -> None:
        text = " ".join(["The cat sat."] * 30)
        counts = R.analyze_text(text)
        assert counts.sentences == 30 and counts.polysyllabic == 0
        assert R.smog_index(counts) == pytest.approx(3.1291)

    def test_definitional_formula(self) -> None:
        text = " ".join(["The extraordinary hippopotamus."] * 30)
        counts = R.analyze_text(text)
        assert counts.sentences == 30
        expected = 1.043 * math.sqrt(counts.polysyllabic * 30 / counts.sentences) + 3.1291
        assert counts.polysyllabic == 60
        assert R.smog_index(counts) == pytest.approx(expected)

    def test_short_text_scores_zero_with_warning(self) -> None:
        counts = R.analyze_text("One sentence only here.")
        assert counts.sentences == 1
        assert R.smog_index(counts) == 0.0
        result = R.run(_corpus(["One sentence only here.", " ".join(["The cat sat."] * 30)]))
        assert result.ok
        assert any(d.code == "READABILITY_SMOG_SHORT_TEXT" for d in result.diagnostics)


class TestRun:
    def test_output_columns(self) -> None:
        frame = R.run(_corpus(TEXTS)).unwrap().to_frame()
        assert list(frame.columns) == [
            "Document ID",
            "Document",
            "Sentences",
            "Words",
            "Syllables",
            "Polysyllabic Words",
            "Characters",
            "Flesch Reading Ease",
            "Flesch-Kincaid Grade",
            "Gunning Fog Index",
            "Coleman-Liau Index",
            "Automated Readability Index",
            "SMOG Index",
            "Interpretation",
        ]
        assert len(frame) == len(TEXTS)

    def test_empty_document_emits_zero_row_with_warning(self) -> None:
        result = R.run(_corpus(["", "The cat sat."]))
        assert result.ok
        frame = result.unwrap().to_frame()
        assert len(frame) == 2
        empty = frame.iloc[0]
        assert empty["Words"] == 0 and empty["Flesch Reading Ease"] == 0.0
        assert empty["Interpretation"] == "n/a"
        assert any(d.code == "READABILITY_EMPTY_DOC" for d in result.diagnostics)

    def test_deterministic(self) -> None:
        first = R.run(_corpus(TEXTS)).unwrap().to_frame()
        second = R.run(_corpus(TEXTS)).unwrap().to_frame()
        pd.testing.assert_frame_equal(first, second)

    def test_scores_rounded_to_two_decimals(self) -> None:
        frame = R.run(_corpus(TEXTS)).unwrap().to_frame()
        for col in ["Flesch Reading Ease", "Flesch-Kincaid Grade", "SMOG Index"]:
            for value in frame[col].tolist():
                assert value == round(float(value), 2)


class TestCli:
    def test_cli_writes_table_and_envelope(self, tmp_path: Path) -> None:
        from tools.readability import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("The cat sat. Dogs barked loudly!", encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(corpus), str(out)]) == 0
        run_dir = next(out.iterdir())
        assert (run_dir / "readability.csv").is_file()
        assert (run_dir / "result.json").is_file()

    def test_cli_missing_corpus_fails(self, tmp_path: Path) -> None:
        from tools.readability import main

        assert main([str(tmp_path / "ghost"), str(tmp_path / "out")]) == 1

    def test_cli_partial_corpus_still_writes_good_documents(self, tmp_path: Path, monkeypatch) -> None:
        """Review finding S3: one bad file must not discard the good documents.

        read_corpus returns readable documents with ERROR diagnostics (its
        documented "partial" state); tools must process them (matching
        tools/_cli.load_corpus) instead of reproducing the legacy
        first-bad-file abort. The bad file here is a genuine reader failure —
        an explicit undecodable encoding — which is exactly the Result shape
        ``encoding="utf-8"`` produces for bytes no codec in the chain accepts.
        """
        from tools import readability

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("The cat sat. Dogs barked loudly!", encoding="utf-8")
        (corpus / "b.txt").write_bytes(b"\xff\xfe\x00\x00\x00broken")
        (corpus / "c.txt").write_text("Second document here. Words are good!", encoding="utf-8")

        from core.result import Result

        original = readability.read_corpus

        def partial_read_corpus(root, **kwargs):
            result = original(root, **kwargs)
            if result.value is None:
                return result
            # Drop b.txt and attach the ERROR diagnostic the reader emits
            # when a file is genuinely unreadable (explicit-encoding case).

            kept = [d for d in result.unwrap().docs if d.path.name != "b.txt"]
            from core.result import Diagnostic

            diag = Diagnostic.error(
                "FILE_UNDECODABLE", "could not decode b.txt with any of ['utf-8-sig']", path=str(corpus / "b.txt")
            )
            corpus_value = (
                readability.Corpus(docs=tuple(kept), sha256=result.unwrap().sha256)
                if hasattr(readability, "Corpus")
                else None
            )
            if corpus_value is None:
                from core.io.reader import Corpus

                corpus_value = Corpus(docs=tuple(kept), sha256=result.unwrap().sha256)
            return Result.success(corpus_value, diag)

        monkeypatch.setattr(readability, "read_corpus", partial_read_corpus)
        out = tmp_path / "out"
        code = readability.main([str(corpus), str(out)])
        assert code in (0, 1)  # ok or partial — never a thrown traceback
        run_dirs = list(out.iterdir())
        assert run_dirs, "the readable documents must still produce a run"
        import json

        envelope = json.loads((run_dirs[0] / "result.json").read_text(encoding="utf-8"))
        names = [a["path"] for a in envelope["artifacts"]]
        assert "readability.csv" in names
        import pandas as pd

        frame = pd.read_csv(run_dirs[0] / "readability.csv")
        assert set(frame["Document"]) == {"a.txt", "c.txt"}
        assert any("UNDECODABLE" in json.dumps(diag).upper() for diag in envelope["diagnostics"])


class TestC68Corrections:
    """C6-8: unicode words, SMOG policy, formula inventory."""

    def test_unicode_words_counted(self) -> None:
        counts = R.analyze_text("Le café est naïve. Привет мир.")
        assert counts.words == 6  # accented + Cyrillic words all count

    def test_smog_full_formula_documented(self) -> None:
        # 30 sentences x 2 polysyllables = 60; full-formula SMOG uses
        # sqrt(polysyllables * 30 / sentences) — the reduced-sample variant
        # (textstat's default) would sample differently. Definition check:
        text = " ".join(["The extraordinary hippopotamus."] * 30)
        counts = R.analyze_text(text)
        assert counts.sentences == 30 and counts.polysyllabic == 60
        import math

        assert R.smog_index(counts) == pytest.approx(1.043 * math.sqrt(60 * 30 / 30) + 3.1291)

    def test_formula_inventory_is_five_plus_smog(self) -> None:
        # The pinned legacy module contains exactly 5 formulas; SMOG is new.
        names = [f for f in dir(R) if not f.startswith("_")]
        formulas = [
            n
            for n in names
            if n
            in (
                "flesch_reading_ease",
                "flesch_kincaid_grade",
                "gunning_fog",
                "coleman_liau",
                "automated_readability_index",
                "smog_index",
            )
        ]
        assert len(formulas) == 6  # 5 legacy + SMOG
