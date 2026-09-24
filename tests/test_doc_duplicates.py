"""FR-3.3 matching and duplicates — exact, normalized, fuzzy (TF-IDF threshold)."""

from __future__ import annotations

from pathlib import Path

from core.analysis import doc_duplicates as D
from core.io.reader import Corpus, Document, hash_text


def _corpus() -> Corpus:
    texts = [
        "The cat sat on the mat.",  # doc1
        "The cat sat on the mat.",  # doc2: byte duplicate
        "  THE CAT SAT on the mat. ",  # doc3: normalized duplicate
        "The cat sat on the mat peacefully purring in the warm afternoon sun.",  # doc4: fuzzy kin
        "Quantum chromodynamics describes quarks and gluons.",  # doc5: unrelated
    ]
    docs = tuple(
        Document(doc_id=i + 1, path=Path(f"doc{i + 1}.txt"), text=text, sha256=hash_text(text))
        for i, text in enumerate(texts)
    )
    return Corpus(docs=docs, sha256="x")


class TestExactAndNormalized:
    def test_exact_groups(self) -> None:
        groups = D.exact_groups(_corpus()).unwrap().to_frame()
        assert len(groups) == 1
        assert groups.iloc[0]["Members"] == "doc1.txt, doc2.txt"

    def test_normalized_groups_catch_case_and_space(self) -> None:
        groups = D.normalized_groups(_corpus()).unwrap().to_frame()
        assert len(groups) == 1
        assert groups.iloc[0]["Members"] == "doc1.txt, doc2.txt, doc3.txt"

    def test_no_duplicates_gives_empty(self) -> None:
        docs = (
            Document(doc_id=1, path=Path("a.txt"), text="alpha beta", sha256=hash_text("alpha beta")),
            Document(doc_id=2, path=Path("b.txt"), text="gamma delta", sha256=hash_text("gamma delta")),
        )
        assert D.exact_groups(Corpus(docs=docs, sha256="x")).unwrap().to_frame().empty


class TestFuzzy:
    def test_threshold_finds_kin_not_stranger(self) -> None:
        pairs = D.fuzzy_pairs(_corpus(), threshold=30.0).unwrap().to_frame()
        names = {(r["Document A"], r["Document B"]) for _, r in pairs.iterrows()}
        assert ("doc1.txt", "doc4.txt") in names or ("doc2.txt", "doc4.txt") in names
        assert not any("doc5.txt" in pair for pair in names)

    def test_reproducible_thresholds(self) -> None:
        first = D.fuzzy_pairs(_corpus(), threshold=30.0).unwrap().to_frame()
        second = D.fuzzy_pairs(_corpus(), threshold=30.0).unwrap().to_frame()
        assert first["Similarity"].tolist() == second["Similarity"].tolist()

    def test_bad_threshold_rejected(self) -> None:
        result = D.fuzzy_pairs(_corpus(), threshold=101.0)
        assert result.value is None
        assert any(d.code == "DUPS_BAD_THRESHOLD" for d in result.diagnostics)


class TestCli:
    def test_cli_writes_three_tables(self, tmp_path: Path) -> None:
        from tools.doc_duplicates import main

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("The cat sat on the mat.", encoding="utf-8")
        (corpus / "b.txt").write_text("The cat sat on the mat.", encoding="utf-8")
        (corpus / "c.txt").write_text("Totally different words here zebra.", encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(corpus), str(out), "--threshold", "30"]) == 0
        run_dir = next(out.iterdir())
        for name in ("exact_groups.csv", "normalized_groups.csv", "fuzzy_pairs.csv"):
            assert (run_dir / name).is_file(), name
