"""FR-2.2 lexical diversity — Guiraud, MTLD, vocd-D with short-text rules.

Parity oracle: the legacy pure functions in
``statistics_corpus_lexical_diversity_util.py`` loaded with stubbed GUI/IO
imports. TTR-family and MTLD are deterministic, so parity is exact. vocd is
sampling-based (legacy used an unseeded global RNG), so its tests pin
determinism-under-seed, bounds, short-text rules, and diversity ordering —
never exact equality with one legacy draw.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pandas as pd
import pytest

from core.analysis import lexical_diversity as L
from core.io.reader import Corpus, Document, hash_text

ORACLE = (
    Path(__file__).resolve().parent.parent.parent
    / "NLP-Suite-1.6.38"
    / "src"
    / "statistics_corpus_lexical_diversity_util.py"
)
needs_oracle = pytest.mark.skipif(not ORACLE.is_file(), reason="legacy oracle checkout absent")


def load_legacy_oracle():  # type: ignore[no-untyped-def]
    """Import the legacy lexical-diversity module with its GUI/IO deps stubbed."""
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
    spec = importlib.util.spec_from_file_location("legacy_lexdiv", ORACLE)
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


def _long_text() -> str:
    base = (
        "The president addressed Congress on the state of the union with resolve and clarity "
        "while citizens listened across the nation hoping for progress and prosperity"
    )
    return " ".join([base] * 8)  # ~200 tokens, 35+ types


def _diverse_text() -> str:
    vocab = [
        "amber",
        "birch",
        "cable",
        "drift",
        "ember",
        "frost",
        "grove",
        "harbor",
        "ivy",
        "jewel",
        "kettle",
        "lemon",
        "meadow",
        "nerve",
        "ocean",
        "pearl",
        "quilt",
        "river",
        "stone",
        "trail",
        "umber",
        "vivid",
        "whale",
        "xenon",
        "yacht",
        "zephyr",
        "anchor",
        "blade",
        "crest",
        "dune",
        "elm",
        "flame",
        "glacier",
        "heath",
        "iris",
        "jade",
        "karma",
        "lotus",
        "mango",
        "nomad",
        "oasis",
        "panda",
        "query",
        "ridge",
        "solar",
        "tiger",
        "unity",
        "vapor",
        "willow",
        "xylem",
        "youth",
        "zonal",
        "argue",
        "brisk",
        "charm",
        "delta",
        "eager",
        "faint",
        "giant",
        "humble",
        "inert",
        "jolly",
        "keen",
        "lucid",
        "merry",
        "nimble",
        "overt",
        "proud",
        "quiet",
        "rapid",
        "sober",
        "tender",
        "upset",
        "woven",
        "yearn",
        "zippy",
        "acorn",
        "beard",
        "cloud",
        "dusty",
        "early",
        "forum",
        "globe",
        "honey",
        "insect",
        "joint",
        "kneel",
        "light",
        "mouse",
        "night",
        "outer",
        "paste",
    ]
    assert len(set(vocab)) == len(vocab)  # testdata invariant: all types unique
    return " ".join(vocab * 2)  # ~184 tokens, 92 types


class TestTokenizeParity:
    @needs_oracle
    def test_matches_legacy_tokenizer_on_clean_text(self) -> None:
        """C6-8: parity on punctuation-free text (recorded divergence:
        punctuation-attached words are RETAINED now, legacy dropped them)."""
        legacy = load_legacy_oracle()
        texts = [
            "the cat sat on mats and dogs bark loud",
            "  Mixed   whitespace\nand\ttabs  ",
        ]
        for text in texts:
            assert L.tokenize(text) == legacy._tokenize(text), text

    def test_punctuation_attached_words_retained(self) -> None:
        assert L.tokenize("Don't stop believin' go!") == ["don't", "stop", "believin'", "go"]

    def test_unicode_words_not_discarded(self) -> None:
        tokens = L.tokenize("Le café est naïve. Привет мир. 東京 タワー")
        assert "café" in tokens and "naïve" in tokens
        assert "привет" in tokens and "мир" in tokens
        assert "東京" in tokens and "タワー" in tokens


class TestTtrFamilyParity:
    @needs_oracle
    def test_exact_match_with_legacy(self) -> None:
        legacy = load_legacy_oracle()
        token_sets = [
            [],
            ["hello"],
            ["hello", "hello", "world"],
            L.tokenize(_long_text()),
        ]
        for tokens in token_sets:
            assert L.ttr(tokens) == legacy._ttr(tokens)
            assert L.root_ttr(tokens) == legacy._root_ttr(tokens)
            assert L.log_ttr(tokens) == legacy._log_ttr(tokens)

    def test_empty_and_single_token_edges(self) -> None:
        assert L.ttr([]) == 0.0
        assert L.root_ttr([]) == 0.0
        assert L.log_ttr([]) == 0.0
        assert L.log_ttr(["only"]) == 0.0
        assert L.ttr(["only"]) == 1.0


class TestMtldParity:
    @needs_oracle
    def test_exact_match_with_legacy(self) -> None:
        legacy = load_legacy_oracle()
        for text in [_long_text(), " ".join(["The cat sat."] * 20), _diverse_text()]:
            tokens = L.tokenize(text)
            assert len(tokens) >= 10
            assert L.mtld(tokens) == pytest.approx(legacy._mtld(tokens))

    def test_short_text_scores_zero(self) -> None:
        assert L.mtld([]) == 0.0
        assert L.mtld(["only", "nine", "tokens", "here", "today", "yes", "indeed", "quite", "so"]) == 0.0


class TestVocd:
    def test_deterministic_under_seed(self) -> None:
        tokens = L.tokenize(_long_text())
        assert L.vocd(tokens, seed=42) == L.vocd(tokens, seed=42)

    def test_fit_stays_on_the_grid(self) -> None:
        assert 10.0 <= L.vocd(L.tokenize(_long_text()), seed=7) <= 200.0

    def test_repetitive_text_scores_below_diverse_text(self) -> None:
        dull = L.tokenize(" ".join(["The cat sat."] * 40))
        rich = L.tokenize(_diverse_text())
        assert L.vocd(dull, seed=42) < L.vocd(rich, seed=42)

    def test_short_text_scores_zero(self) -> None:
        assert L.vocd([], seed=1) == 0.0
        assert L.vocd(L.tokenize("The cat sat. Dogs barked."), seed=1) == 0.0

    def test_expected_curve_matches_definition(self) -> None:
        # (D/ss) * (sqrt(1 + 2*ss/D) - 1), Malvern & Richards via legacy.
        assert L.expected_ttr(50.0, 35) == pytest.approx((50.0 / 35) * ((1 + 2 * 35 / 50.0) ** 0.5 - 1))


class TestRun:
    def test_output_columns(self) -> None:
        frame = L.run(_corpus([_long_text()])).unwrap().to_frame()
        assert list(frame.columns) == [
            "Document ID",
            "Document",
            "Total Tokens",
            "Unique Types",
            "TTR",
            "Root TTR (Guiraud)",
            "Log TTR (Herdan)",
            "MTLD",
            "vocd-D",
        ]

    def test_empty_and_short_docs_get_defined_measures_and_na(self) -> None:
        result = L.run(_corpus(["", "Too short.", _long_text()]))
        assert result.ok
        frame = result.unwrap().to_frame()
        assert len(frame) == 3
        # C6-8: defined measures continue; MTLD/vocd-D are NA, never 0.0.
        assert bool(pd.isna(frame.iloc[0]["TTR"]))  # empty doc: nothing defined
        assert frame.iloc[1]["TTR"] == 1.0  # 4 tokens, all distinct: defined
        assert bool(pd.isna(frame.iloc[1]["MTLD"])) and bool(pd.isna(frame.iloc[1]["vocd-D"]))
        assert frame.iloc[2]["MTLD"] > 0.0
        codes = [d.code for d in result.diagnostics]
        assert "LEXDIV_EMPTY_DOC" in codes and "LEXDIV_SHORT_TEXT" in codes

    def test_vocd_grid_includes_200_endpoint(self) -> None:
        # A vocabulary huge enough to push the fit to the grid ceiling.
        vocab = " ".join(f"w{i:03d}" for i in range(300))
        tokens = L.tokenize(" ".join([vocab] * 3))
        assert L.vocd(tokens, seed=1) <= 200.0
        # Degenerate TTR curve: expected_ttr(D, ss) at D=200 must be on-grid.
        assert L.expected_ttr(200.0, 35) == pytest.approx((200.0 / 35) * ((1 + 2 * 35 / 200.0) ** 0.5 - 1))

    def test_seeded_result_stable_across_corpus_order(self) -> None:
        # Seed stability: adding unrelated documents must not change a doc's score.
        tokens_a = L.tokenize(_long_text())
        score_alone = L.vocd(tokens_a, seed=42)
        corpus_mixed = _corpus([_long_text(), _diverse_text(), _long_text()])
        frame = L.run(corpus_mixed, seed=42).unwrap().to_frame()
        doc1 = frame[frame["Document"] == "doc1.txt"].iloc[0]
        assert float(doc1["vocd-D"]) == score_alone

    def test_deterministic_for_fixed_seed(self) -> None:
        first = L.run(_corpus([_long_text()]), seed=11).unwrap().to_frame()
        second = L.run(_corpus([_long_text()]), seed=11).unwrap().to_frame()
        pd.testing.assert_frame_equal(first, second)

    def test_rounding_matches_legacy_convention(self) -> None:
        frame = L.run(_corpus([_long_text()])).unwrap().to_frame()
        assert frame.iloc[0]["TTR"] == round(float(frame.iloc[0]["TTR"]), 4)
        assert frame.iloc[0]["MTLD"] == round(float(frame.iloc[0]["MTLD"]), 2)


class TestCli:
    def test_cli_writes_table_and_records_seed(self, tmp_path: Path) -> None:
        import json

        from tools.lexical_diversity import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text(_long_text(), encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(corpus), str(out), "--seed", "7"]) == 0
        run_dir = next(out.iterdir())
        assert (run_dir / "lexical_diversity.csv").is_file()
        envelope = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        assert envelope["params"]["seed"] == 7

    def test_cli_missing_corpus_fails(self, tmp_path: Path) -> None:
        from tools.lexical_diversity import main

        assert main([str(tmp_path / "ghost"), str(tmp_path / "out")]) == 1


class TestVocdShortText:
    """Review finding S4: 10..34-token docs must publish NA + a warning, not 0.0."""

    @staticmethod
    def _run(tokens_per_doc: dict[str, int]):
        from core.io.reader import Corpus, Document, hash_text

        docs = []
        for name, n in tokens_per_doc.items():
            text = " ".join(f"tok{i % 37}" for i in range(n))
            docs.append(Document(doc_id=len(docs) + 1, path=Path(f"{name}.txt"), text=text, sha256=hash_text(text)))
        return L.run(Corpus(docs=tuple(docs), sha256="x"))

    def test_boundary_9_10_tokens_mtld(self) -> None:
        result = self._run({"nine": 9, "ten": 10})
        frame = result.unwrap().frame
        assert pd.isna(frame[frame["Document"] == "nine.txt"]["MTLD"].iloc[0])
        assert not pd.isna(frame[frame["Document"] == "ten.txt"]["MTLD"].iloc[0])

    def test_boundary_34_35_tokens_vocd(self) -> None:
        result = self._run({"thirtyfour": 34, "thirtyfive": 35})
        frame = result.unwrap().frame
        assert pd.isna(frame[frame["Document"] == "thirtyfour.txt"]["vocd-D"].iloc[0])
        assert not pd.isna(frame[frame["Document"] == "thirtyfive.txt"]["vocd-D"].iloc[0])
        # MTLD stays computed at 34 tokens (its floor is 10)
        assert not pd.isna(frame[frame["Document"] == "thirtyfour.txt"]["MTLD"].iloc[0])

    def test_vocd_short_text_diagnostic(self) -> None:
        result = self._run({"mid": 20})
        assert any(d.code == "LEXDIV_VOCD_SHORT_TEXT" for d in result.diagnostics)
        # never a fabricated 0.0
        frame = result.unwrap().frame
        assert pd.isna(frame.iloc[0]["vocd-D"])
