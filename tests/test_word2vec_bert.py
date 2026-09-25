"""word2vec_bert — BERT type vectors with the word2vec_gensim table contract.

The heavy backend is never needed here: ``EmbeddingBackend`` is the seam, the
same protocol ``core.analysis.contextual`` defines, so these tests fix the
table contracts (HW2 compares this tool against word2vec_gensim column by
column) without torch or transformers.
"""

from __future__ import annotations

import pandas as pd

from core.analysis.word2vec_bert import distances, project_tsne, train_bert, vectors
from core.result import Diagnostic, Result


class FakeBackend:
    """Deterministic four-dimensional stand-in for a transformer."""

    def dimension(self) -> int:
        return 4

    def embed(self, words, contexts):
        rows = []
        for word, context in zip(words, contexts, strict=True):
            rows.append(
                [
                    float((ord(word[0]) % 7) + 1),
                    float(len(word)),
                    float(len(context.split())),
                    1.0,
                ]
            )
        return rows


def conll(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Full canonical frame: (doc_id, sent_id, form, lemma)."""
    columns = [
        "ID",
        "Form",
        "Lemma",
        "POS",
        "NER",
        "Head",
        "DepRel",
        "Sentence ID",
        "Document ID",
        "Document",
        "Deps",
        "Record ID",
        "Clause Tag",
    ]
    body = []
    for record, (did, sid, form, lemma) in enumerate(rows, start=1):
        body.append(
            {
                "ID": record,
                "Form": form,
                "Lemma": lemma,
                "POS": "NN",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": int(sid),
                "Document ID": did,
                "Document": f"doc-{did}.txt",
                "Deps": "",
                "Record ID": record,
                "Clause Tag": "",
            }
        )
    return pd.DataFrame(body, columns=columns)


RICH = conll(
    [
        ("1", 1, "Cats", "cat"),
        ("1", 1, "chase", "chase"),
        ("1", 1, "mice", "mouse"),
        ("1", 2, "Dogs", "dog"),
        ("1", 2, "chase", "chase"),
        ("1", 2, "cats", "cat"),
        ("2", 1, "Mice", "mouse"),
        ("2", 1, "flee", "flee"),
        ("2", 1, "dogs", "dog"),
    ]
)


class TestTrain:
    def test_type_vectors_share_the_word2vec_columns(self) -> None:
        trained = train_bert(RICH, min_count=1, backend=FakeBackend())
        assert trained.ok, trained.diagnostics
        model = trained.unwrap()
        frame = vectors(model).unwrap()
        assert list(frame.columns) == ["Word", "Count", "Vector"]
        # Types are lowercased alphabetic tokens of the chosen field (lemma).
        assert set(frame["Word"]) == {"cat", "chase", "mouse", "dog", "flee"}
        row = frame.loc[frame["Word"] == "cat"].iloc[0]
        assert int(row["Count"]) == 2
        assert len(str(row["Vector"]).split(",")) == 4

    def test_same_input_and_backend_repeat_exactly(self) -> None:
        first = vectors(train_bert(RICH, min_count=1, backend=FakeBackend()).unwrap()).unwrap()
        second = vectors(train_bert(RICH, min_count=1, backend=FakeBackend()).unwrap()).unwrap()
        pd.testing.assert_frame_equal(first, second)

    def test_empty_frame_is_an_empty_result_not_an_error(self) -> None:
        trained = train_bert(conll([]), min_count=1, backend=FakeBackend())
        assert trained.ok, trained.diagnostics
        assert vectors(trained.unwrap()).unwrap().empty

    def test_no_usable_tokens_says_so(self) -> None:
        # Punctuation and numbers are dropped by the alphabetic filter.
        trained = train_bert(conll([("1", 1, "...", "3.14")]), min_count=1, backend=FakeBackend())
        assert not trained.ok
        assert trained.diagnostics[0].code == "W2V_BERT_NO_TOKENS"

    def test_min_count_can_empty_the_vocabulary_loudly(self) -> None:
        trained = train_bert(RICH, min_count=99, backend=FakeBackend())
        assert not trained.ok
        assert trained.diagnostics[0].code == "W2V_BERT_EMPTY_VOCAB"

    def test_a_missing_model_fails_naming_the_models_page(self) -> None:
        # The real backend is out of reach in tests; force its failure path.
        import core.analysis.word2vec_bert as subject

        def broken(model: str) -> Result[object]:
            return Result.failure(Diagnostic.error("W2V_BERT_UNAVAILABLE", "no transformers"))

        original = subject.default_backend
        subject.default_backend = broken  # type: ignore[assignment]
        try:
            trained = train_bert(RICH, min_count=1)
        finally:
            subject.default_backend = original  # type: ignore[assignment]
        assert not trained.ok
        code = trained.diagnostics[0]
        assert code.code == "W2V_BERT_UNAVAILABLE"
        assert "Open Models" in str(code.context.get("fix", ""))


class TestDistances:
    def test_neighbours_match_the_word2vec_columns(self) -> None:
        found = distances(RICH, "cat", min_count=1, backend=FakeBackend())
        assert found.ok, found.diagnostics
        frame = found.unwrap()
        assert list(frame.columns) == ["Word", "Neighbor", "Cosine"]
        assert frame["Word"].nunique() == 1
        assert 1 <= len(frame) <= 5

    def test_an_unknown_word_names_itself(self) -> None:
        found = distances(RICH, "unobtainium", min_count=1, backend=FakeBackend())
        assert not found.ok
        assert found.diagnostics[0].code == "W2V_BERT_UNKNOWN_WORD"

    def test_a_form_query_on_a_lemma_vocabulary_gets_a_fix(self) -> None:
        # "Cats" is a FORM; the default vocabulary holds lemmas, so the miss
        # is morphological and the fix names the lemma.
        found = distances(RICH, "Cats", min_count=1, backend=FakeBackend())
        assert not found.ok
        assert "cat" in str(found.diagnostics[0].context.get("fix", ""))

    def test_a_capitalization_miss_still_finds_the_word(self) -> None:
        # Lookup is case-insensitive (as in word2vec_gensim), so "Cat" works.
        found = distances(RICH, "Cat", min_count=1, backend=FakeBackend())
        assert found.ok, found.diagnostics

    def test_empty_query_is_refused(self) -> None:
        assert distances(RICH, "  ", min_count=1, backend=FakeBackend()).diagnostics[0].code == "W2V_BERT_BAD_QUERY"

    def test_empty_frame_yields_an_empty_neighbour_table(self) -> None:
        found = distances(conll([]), "cat", min_count=1, backend=FakeBackend())
        assert found.ok
        assert list(found.unwrap().columns) == ["Word", "Neighbor", "Cosine"]
        assert found.unwrap().empty


class TestProjection:
    def test_too_few_words_skips_tsne_with_a_warning(self) -> None:
        trained = train_bert(
            conll([("1", 1, "Cats", "cat"), ("1", 1, "sleep", "sleep")]), min_count=1, backend=FakeBackend()
        )
        projected = project_tsne(trained.unwrap(), seed=7)
        assert projected.ok
        assert projected.unwrap().empty
        assert any(d.code == "W2V_TSNE_SKIPPED" for d in projected.diagnostics)

    def test_coordinates_are_reproducible_at_a_fixed_seed(self) -> None:
        model = train_bert(RICH, min_count=1, backend=FakeBackend()).unwrap()
        first = project_tsne(model, seed=7).unwrap()
        second = project_tsne(model, seed=7).unwrap()
        pd.testing.assert_frame_equal(first, second)
        assert list(first.columns) == ["Word", "X", "Y"]
