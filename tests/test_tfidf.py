"""CAP-STATS-12 — TF-IDF term weighting.

Weights are checked against hand-computed values and against sklearn, which is
an independent implementation of the same smoothed conventions.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.analysis.tfidf import tfidf
from core.conll.schema import Col


def _table(documents: list[str]) -> pd.DataFrame:
    """One row per whitespace token; each document is a single sentence."""
    rows: list[dict[str, object]] = []
    for document_id, text in enumerate(documents, start=1):
        for token_id, token in enumerate(text.split(), start=1):
            rows.append(
                {
                    Col.ID.value: token_id,
                    Col.FORM.value: token,
                    Col.LEMMA.value: token.lower(),
                    Col.POS.value: "NOUN",
                    Col.NER.value: "",
                    Col.HEAD.value: 0,
                    Col.DEPREL.value: "root",
                    Col.DEPS.value: "",
                    Col.CLAUSE_TAG.value: "",
                    Col.RECORD_ID.value: len(rows) + 1,
                    Col.SENTENCE_ID.value: 1,
                    Col.DOCUMENT_ID.value: document_id,
                    Col.DOCUMENT.value: f"d{document_id}.txt",
                }
            )
    return pd.DataFrame(rows)


DOCUMENTS = [
    "alpha bravo charlie alpha",
    "bravo delta echo echo echo",
    "alpha alpha foxtrot golf",
]


class TestWeights:
    def test_idf_is_the_smoothed_definition(self) -> None:
        frame = tfidf(_table(DOCUMENTS), top_n=50, normalize=False).unwrap()
        total = 3
        for term, df in (("alpha", 2), ("bravo", 2), ("delta", 1), ("echo", 1)):
            expected = math.log((1 + total) / (1 + df)) + 1
            got = frame.loc[frame["Term"] == term, "IDF"].iloc[0]
            assert got == pytest.approx(round(expected, 6))

    def test_matches_sklearn(self) -> None:
        """Independent implementation of the same conventions."""
        sklearn_text = pytest.importorskip("sklearn.feature_extraction.text")
        vectorizer = sklearn_text.TfidfVectorizer(
            smooth_idf=True,
            sublinear_tf=False,
            norm="l2",
            # S106 reads "token_pattern" as a credential; it is sklearn's
            # tokenizer regex, widened here to match our whitespace split.
            token_pattern=r"(?u)\b\w+\b",  # noqa: S106
        )
        matrix = vectorizer.fit_transform(DOCUMENTS).toarray()
        terms = list(vectorizer.get_feature_names_out())
        frame = tfidf(_table(DOCUMENTS), top_n=50).unwrap()
        assert not frame.empty
        for _, row in frame.iterrows():
            document = int(row["Document ID"]) - 1
            assert matrix[document][terms.index(row["Term"])] == pytest.approx(row["TF-IDF"], abs=1e-6)

    def test_sublinear_tf_dampens_repeated_terms(self) -> None:
        raw = tfidf(_table(DOCUMENTS), top_n=50, normalize=False).unwrap()
        damped = tfidf(_table(DOCUMENTS), top_n=50, normalize=False, sublinear_tf=True).unwrap()
        # "echo" occurs three times in document 2.
        raw_tf = raw[(raw["Document ID"] == "2") & (raw["Term"] == "echo")]["TF"].iloc[0]
        damped_tf = damped[(damped["Document ID"] == "2") & (damped["Term"] == "echo")]["TF"].iloc[0]
        assert raw_tf == pytest.approx(3.0)
        assert damped_tf == pytest.approx(round(1 + math.log(3), 6))
        assert damped_tf < raw_tf

    def test_normalization_makes_documents_comparable(self) -> None:
        frame = tfidf(_table(DOCUMENTS), top_n=50).unwrap()
        for _document, group in frame.groupby("Document ID"):
            norm = math.sqrt(sum(value**2 for value in group["TF-IDF"]))
            assert norm == pytest.approx(1.0, abs=1e-5)

    def test_a_term_in_every_document_is_downweighted_not_deleted(self) -> None:
        """Smooth idf gives a universal term idf 1, not 0."""
        frame = tfidf(_table(["alpha bravo", "alpha charlie"]), top_n=50, normalize=False).unwrap()
        alpha = frame.loc[frame["Term"] == "alpha", "IDF"].iloc[0]
        bravo = frame.loc[frame["Term"] == "bravo", "IDF"].iloc[0]
        assert alpha == pytest.approx(1.0)
        assert bravo > alpha


class TestFiltering:
    def test_min_df_drops_rare_terms(self) -> None:
        frame = tfidf(_table(DOCUMENTS), top_n=50, min_df=2).unwrap()
        assert set(frame["Term"]) <= {"alpha", "bravo"}

    def test_max_df_ratio_drops_ubiquitous_terms(self) -> None:
        documents = ["common alpha", "common bravo", "common charlie"]
        frame = tfidf(_table(documents), top_n=50, max_df_ratio=0.9).unwrap()
        assert "common" not in set(frame["Term"])

    def test_min_length_drops_short_tokens(self) -> None:
        frame = tfidf(_table(["a alpha bravo"]), top_n=50, min_length=2).unwrap()
        assert "a" not in set(frame["Term"])

    def test_non_alphabetic_tokens_are_skipped(self) -> None:
        frame = tfidf(_table(["alpha 123 bravo"]), top_n=50).unwrap()
        assert "123" not in set(frame["Term"])

    def test_top_n_limits_per_document_not_overall(self) -> None:
        frame = tfidf(_table(DOCUMENTS), top_n=2).unwrap()
        assert frame.groupby("Document ID").size().max() == 2
        assert frame["Document ID"].nunique() == 3

    def test_rank_restarts_at_one_per_document(self) -> None:
        frame = tfidf(_table(DOCUMENTS), top_n=3).unwrap()
        for _document, group in frame.groupby("Document ID"):
            assert group["Rank"].tolist() == list(range(1, len(group) + 1))


class TestDeterminism:
    def test_repeated_runs_are_byte_identical(self) -> None:
        frame = _table(DOCUMENTS)
        first = tfidf(frame, top_n=50).unwrap()
        for _ in range(5):
            assert tfidf(frame, top_n=50).unwrap().equals(first)

    def test_ties_break_alphabetically(self) -> None:
        """Equal weights must not depend on dict iteration order."""
        frame = tfidf(_table(["zulu yankee xray"]), top_n=50).unwrap()
        assert frame["Term"].tolist() == sorted(frame["Term"].tolist())


class TestContracts:
    @pytest.mark.parametrize(
        ("kwargs", "code"),
        [
            ({"top_n": 0}, "TFIDF_BAD_TOP_N"),
            ({"min_df": 0}, "TFIDF_BAD_MIN_DF"),
            ({"max_df_ratio": 0.0}, "TFIDF_BAD_MAX_DF"),
            ({"max_df_ratio": 1.5}, "TFIDF_BAD_MAX_DF"),
            ({"min_length": 0}, "TFIDF_BAD_MIN_LENGTH"),
            ({"field": Col.POS}, "TFIDF_BAD_FIELD"),
        ],
    )
    def test_bad_arguments_fail_with_a_named_diagnostic(self, kwargs: dict[str, object], code: str) -> None:
        result = tfidf(_table(DOCUMENTS), **kwargs)  # type: ignore[arg-type]
        assert not result.ok
        assert [d.code for d in result.errors] == [code]

    def test_single_document_warns_that_idf_is_constant(self) -> None:
        result = tfidf(_table(["alpha bravo"]))
        assert result.ok
        assert any(d.code == "TFIDF_SINGLE_DOCUMENT" for d in result.diagnostics)

    def test_empty_vocabulary_warns_rather_than_failing(self) -> None:
        result = tfidf(_table(DOCUMENTS), min_df=99)
        assert result.ok
        assert result.unwrap().empty
        assert any(d.code == "TFIDF_EMPTY_VOCABULARY" for d in result.diagnostics)

    def test_empty_frame_returns_the_full_schema(self) -> None:
        result = tfidf(_table(["alpha"]).iloc[0:0])
        assert result.ok
        assert "TF-IDF" in result.unwrap().columns


class TestRegistryWiring:
    def test_registered_outputs_match_the_cli(self) -> None:
        from core.profiler.registry import get_tool

        spec = get_tool("tfidf")
        assert spec is not None
        assert spec.outputs == ("tfidf.csv",)
        assert spec.capability_ids == ("CAP-STATS-12",)

    def test_adapter_is_wired_for_the_profiler(self) -> None:
        from core.profiler.executor import ADAPTERS

        assert "tfidf" in ADAPTERS
