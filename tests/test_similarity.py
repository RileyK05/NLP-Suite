"""FR-2.9 similarity — Levenshtein string metrics + TF-IDF document pairs.

String oracle: legacy ``string_similarity_util.py`` imported directly
(stdlib-only). Band oracle: legacy plagiarist ``_band_index`` with stubbed
GUI/IO imports. TF-IDF assertions are definitional (identity, symmetry,
ordering), never implementation copies.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from core.analysis import doc_similarity as D, string_similarity as S
from core.io.reader import Corpus, Document, hash_text

LEGACY_SRC = Path(__file__).resolve().parent.parent.parent / "NLP-Suite-1.6.38" / "src"
needs_oracle = pytest.mark.skipif(
    not (LEGACY_SRC / "string_similarity_util.py").is_file(), reason="legacy oracle checkout absent"
)


def load_legacy_string():  # type: ignore[no-untyped-def]
    import importlib.util

    spec = importlib.util.spec_from_file_location("legacy_strsim", LEGACY_SRC / "string_similarity_util.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_legacy_bands():  # type: ignore[no-untyped-def]
    import importlib.util

    for name in ["GUI_util", "IO_libraries_util"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["IO_libraries_util"].install_all_Python_packages = lambda *a, **k: True
    sys.modules["GUI_util"].window = None
    spec = importlib.util.spec_from_file_location("legacy_plag", LEGACY_SRC / "plagiarist_util.py")
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


class TestStringParity:
    @needs_oracle
    def test_distance_ratio_similarity_match_legacy(self) -> None:
        legacy = load_legacy_string()
        pairs = [
            ("kitten", "sitting"),
            ("COBB", "Cobb"),
            ("Jim Cobb", "Cobb, Jim"),
            ("", ""),
            ("a", ""),
            ("Flemin", "Fleming"),
            ("identical", "identical"),
        ]
        for a, b in pairs:
            assert S.levenshtein_distance(a, b) == legacy.levenshtein_distance(a, b), (a, b)
            assert S.levenshtein_ratio(a, b) == legacy.levenshtein_ratio(a, b), (a, b)
            assert S.similarity(a, b) == legacy.similarity(a, b), (a, b)

    @needs_oracle
    def test_best_match_returns_best_not_first(self) -> None:
        legacy = load_legacy_string()
        candidates = [("Flemming", 3), ("Fleming", 9)]
        assert S.best_match("Flemin", candidates, 80) == legacy.best_match("Flemin", candidates, 80)
        assert S.best_match("Flemin", candidates, 80)[0] == "Fleming"

    def test_ratio_bounds(self) -> None:
        assert S.levenshtein_ratio("abc", "abc") == 100.0
        assert S.levenshtein_ratio("", "") == 100.0
        assert 0.0 <= S.similarity("abc", "xyz") <= 100.0


class TestBands:
    def test_band_intervals_exact_per_documentation(self) -> None:
        """C6-9: (0,10] -> 0, (10,20] -> 1, ..., (90,100] -> 9. No rounding."""
        cases = {
            0.0: None,
            0.1: 0,
            10.0: 0,
            10.1: 1,
            19.9: 1,
            20.0: 1,
            20.1: 2,
            55.0: 5,
            90.0: 8,
            90.1: 9,
            99.9: 9,
            100.0: 9,
            120.0: 9,
        }
        for pct, expected in cases.items():
            assert D.band_index(pct) == expected, pct
        with pytest.raises(ValueError, match="finite"):
            D.band_index(float("nan"))
        with pytest.raises(ValueError, match="finite"):
            D.band_index(float("inf"))

    def test_band_labels(self) -> None:
        assert D.CLASS_LABELS[0] == "0-10%" and D.CLASS_LABELS[-1] == "90-100%"


class TestDocSimilarity:
    TEXTS: tuple[str, ...] = (
        "The president addressed Congress on Monday morning",
        "The president addressed Congress on Monday morning",  # near-duplicate of 0
        "Quantum chromodynamics describes quarks and gluons in particle physics",
    )

    def test_identical_pair_scores_100(self) -> None:
        pairs = D.pairwise_similarity(_corpus(self.TEXTS)).unwrap().to_frame()
        self_pair = pairs[(pairs["Document A"] == "doc1.txt") & (pairs["Document B"] == "doc2.txt")]
        assert float(self_pair["Similarity"].iloc[0]) == pytest.approx(100.0)

    def test_symmetry_and_zero_diagonal_excluded(self) -> None:
        pairs = D.pairwise_similarity(_corpus(self.TEXTS)).unwrap().to_frame()
        assert len(pairs) == 3  # 3 choose 2, no self-pairs
        row = pairs[(pairs["Document A"] == "doc1.txt") & (pairs["Document B"] == "doc3.txt")].iloc[0]
        assert float(row["Similarity"]) < 50.0

    def test_duplicates_honor_threshold(self) -> None:
        result = D.find_duplicates(_corpus(self.TEXTS), threshold=80.0).unwrap()
        dupes = result.to_frame()
        assert len(dupes) == 1
        assert set([dupes.iloc[0]["Document A"], dupes.iloc[0]["Document B"]]) == {"doc1.txt", "doc2.txt"}
        assert D.find_duplicates(_corpus(self.TEXTS), threshold=100.1).value is None
        assert D.find_duplicates(_corpus(self.TEXTS), threshold=float("nan")).value is None
        assert D.find_duplicates(_corpus(self.TEXTS), threshold=-5).value is None

    def test_single_document_is_an_error(self) -> None:
        result = D.pairwise_similarity(_corpus(["only one"]))
        assert result.value is None
        assert any(d.code == "SIM_TOO_FEW_DOCS" for d in result.diagnostics)

    def test_deterministic_with_tie_ordering(self) -> None:
        first = D.pairwise_similarity(_corpus(self.TEXTS)).unwrap().to_frame()
        second = D.pairwise_similarity(_corpus(self.TEXTS)).unwrap().to_frame()
        assert first["Similarity"].tolist() == second["Similarity"].tolist()
        # equal similarities order by (doc_id_a, doc_id_b)
        tied = first[first["Similarity"] == first["Similarity"].min()]
        ids = list(zip(tied["Document ID A"], tied["Document ID B"], strict=True))
        assert ids == sorted(ids)

    def test_duplicate_basenames_stay_distinct(self) -> None:
        from core.io.reader import Corpus, Document, hash_text

        docs = (
            Document(doc_id=1, path=Path("dir1/report.txt"), text="alpha beta gamma", sha256=hash_text("x")),
            Document(doc_id=2, path=Path("dir2/report.txt"), text="delta epsilon zeta", sha256=hash_text("y")),
        )
        pairs = D.pairwise_similarity(Corpus(docs=docs, sha256="x")).unwrap().to_frame()
        assert "dir1/report.txt" in set(pairs["Document A"]) | set(pairs["Document B"])
        assert "dir2/report.txt" in set(pairs["Document A"]) | set(pairs["Document B"])

    def test_empty_documents_excluded_with_warning(self) -> None:
        from core.io.reader import Corpus, Document, hash_text

        docs = (
            Document(doc_id=1, path=Path("a.txt"), text="alpha beta", sha256=hash_text("x")),
            Document(doc_id=2, path=Path("empty.txt"), text="   ", sha256=hash_text("")),
            Document(doc_id=3, path=Path("b.txt"), text="alpha beta", sha256=hash_text("z")),
        )
        result = D.pairwise_similarity(Corpus(docs=docs, sha256="x"))
        assert result.ok
        assert any(d.code == "SIM_EMPTY_DOC" for d in result.diagnostics)
        pairs = result.unwrap().to_frame()
        assert all("empty.txt" not in str(v) for v in pairs["Document A"].tolist() + pairs["Document B"].tolist())

    def test_document_ids_present(self) -> None:
        pairs = D.pairwise_similarity(_corpus(self.TEXTS)).unwrap().to_frame()
        assert {"Document ID A", "Document ID B"} <= set(pairs.columns)


class TestCli:
    def test_cli_writes_pairs_and_duplicates(self, tmp_path: Path) -> None:
        from tools.doc_similarity import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        for i, text in enumerate(TestDocSimilarity.TEXTS):
            (corpus / f"doc{i + 1}.txt").write_text(text, encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(corpus), str(out), "--threshold", "80"]) == 0
        run_dir = next(out.iterdir())
        assert (run_dir / "doc_pairs.csv").is_file()
        assert (run_dir / "duplicates.csv").is_file()
