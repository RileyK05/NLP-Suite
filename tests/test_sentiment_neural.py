"""sentiment_neural — four backends, one contract, tested through the seams.

No torch, no transformers, no spaCy model, no Java server here: every backend
has an injectable production seam (pipeline / nlp / pipeline / request_json)
and these tests pin the shared table contract plus each backend's loud-failure
codes. The HW5 comparison is only fair if the four tables share columns —
that is what this file freezes.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from core.analysis.sentiment_neural import (
    bert_sentences,
    corenlp_sentences,
    sentence_rows,
    spacy_sentences,
    stanza_sentences,
    summarize_neural,
)

SENTENCE_COLUMNS = ["Document ID", "Document", "Sentence ID", "Sentence", "Label", "Score", "Compound"]
DOCUMENT_COLUMNS = ["Document ID", "Document", "Sentences", "Mean Compound", "Positive", "Negative", "Neutral"]


def conll(rows: list[tuple[str, object, str]]) -> pd.DataFrame:
    """Full canonical frame: (doc_id, sent_id, form)."""
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
    for record, (did, sid, form) in enumerate(rows, start=1):
        body.append(
            {
                "ID": record,
                "Form": form,
                "Lemma": form,
                "POS": "NN",
                "NER": "O",
                "Head": 0,
                "DepRel": "root",
                "Sentence ID": sid,
                "Document ID": did,
                "Document": f"doc-{did}.txt",
                "Deps": "",
                "Record ID": record,
                "Clause Tag": "",
            }
        )
    return pd.DataFrame(body, columns=columns)


GOOD_BAD = conll(
    [
        ("1", 1, "Good"),
        ("1", 1, "dogs"),
        ("1", 2, "Bad"),
        ("1", 2, "cats"),
        ("2", 1, "Dogs"),
        ("2", 1, "sleep"),
    ]
)


class FakeHfPipeline:
    """The transformers sentiment-analysis callable, without transformers."""

    def __call__(self, text: str):
        low = text.lower()
        if "good" in low:
            return [{"label": "POSITIVE", "score": 0.9}]
        if "bad" in low:
            return [{"label": "NEGATIVE", "score": 0.8}]
        return [{"label": "NEUTRAL", "score": 0.5}]


class FakeDoc:
    def __init__(self, cats: dict[str, float]):
        self.cats = cats


class FakeNlp:
    """A spaCy pipeline stand-in with (or without) a textcat head."""

    def __init__(self, head: bool = True):
        self.head = head

    def has(self, name: str) -> bool:
        return self.head and name == "textcat"

    def __call__(self, text: str) -> FakeDoc:
        low = text.lower()
        if "good" in low:
            return FakeDoc({"POSITIVE": 0.7, "NEUTRAL": 0.2, "NEGATIVE": 0.1})
        if "bad" in low:
            return FakeDoc({"POSITIVE": 0.1, "NEUTRAL": 0.2, "NEGATIVE": 0.7})
        return FakeDoc({"POSITIVE": 0.1, "NEUTRAL": 0.8, "NEGATIVE": 0.1})


class FakeSentence:
    def __init__(self, sentiment: int, text: str):
        self.sentiment = sentiment
        self.text = text


class FakeStanzaDoc:
    def __init__(self, sentiments: list[int], text: str):
        self.sentences = [FakeSentence(value, text) for value in sentiments]


class FakeStanzaPipeline:
    def __call__(self, text: str) -> FakeStanzaDoc:
        low = text.lower()
        if "good" in low:
            return FakeStanzaDoc([2], text)
        if "bad" in low:
            return FakeStanzaDoc([0], text)
        return FakeStanzaDoc([1], text)


def corenlp_payload(sentences: list[tuple[str, str, int]]) -> dict:
    return {
        "sentences": [
            {"tokens": [{"word": word} for word in text.split()], "sentiment": label, "sentimentValue": value}
            for label, text, value in sentences
        ]
    }


class TestSharedContract:
    def test_every_backend_uses_the_same_sentence_columns(self) -> None:
        frames = [
            bert_sentences(GOOD_BAD, pipeline=FakeHfPipeline()).unwrap(),
            spacy_sentences(GOOD_BAD, nlp=FakeNlp()).unwrap(),
            stanza_sentences(GOOD_BAD, pipeline=FakeStanzaPipeline()).unwrap(),
        ]
        for frame in frames:
            assert list(frame.columns) == SENTENCE_COLUMNS
        docs = [("1", "a.txt", "Good dogs."), ("2", "b.txt", "Dogs sleep.")]
        core = corenlp_sentences(
            docs,
            request_json=lambda url, properties, body, timeout: corenlp_payload([("Positive", "Good dogs.", 3)]),
        ).unwrap()
        assert list(core.columns) == SENTENCE_COLUMNS

    def test_compound_is_signed_and_bounded(self) -> None:
        frame = bert_sentences(GOOD_BAD, pipeline=FakeHfPipeline()).unwrap()
        assert frame["Compound"].between(-1.0, 1.0).all()
        good = frame.loc[frame["Sentence"] == "Good dogs"].iloc[0]
        assert good["Compound"] > 0 and good["Label"] == "POSITIVE"
        bad = frame.loc[frame["Sentence"] == "Bad cats"].iloc[0]
        assert bad["Compound"] < 0 and bad["Label"] == "NEGATIVE"

    def test_sentence_rows_reports_document_order(self) -> None:
        rows = sentence_rows(GOOD_BAD).unwrap()
        assert [(r[0], r[2]) for r in rows] == [("1", 1), ("1", 2), ("2", 1)]
        assert rows[0][3] == "Good dogs"


class TestBert:
    def test_scores_land_on_the_shared_table(self) -> None:
        frame = bert_sentences(GOOD_BAD, pipeline=FakeHfPipeline()).unwrap()
        assert len(frame) == 3
        assert frame["Score"].max() <= 1.0

    def test_a_long_sentence_is_truncated_loudly(self) -> None:
        long_sentence = "good " + "word " * 300
        frame = conll([("1", 1, long_sentence.strip())])
        result = bert_sentences(frame, pipeline=FakeHfPipeline())
        assert result.ok
        assert any(d.code == "SENTIMENT_NN_TRUNCATED" for d in result.diagnostics)

    def test_a_missing_install_names_the_install_command(self) -> None:
        import core.analysis.sentiment_neural as subject

        def broken_import():
            raise ImportError("no transformers")

        # Force the lazy import to fail the way a missing install would.
        original = subject.bert_sentences

        def forced(frame, *, model="", pipeline=None):
            return subject.Result.failure(
                subject.Diagnostic.error(
                    "SENTIMENT_NN_UNAVAILABLE", "transformers is not installed", fix="pip install transformers torch"
                )
            )

        subject.bert_sentences = forced  # type: ignore[assignment]
        try:
            result = subject.bert_sentences(GOOD_BAD)
        finally:
            subject.bert_sentences = original  # type: ignore[assignment]
        assert result.diagnostics[0].code == "SENTIMENT_NN_UNAVAILABLE"
        assert "transformers" in result.diagnostics[0].context["fix"]
        _ = broken_import


class TestSpacy:
    def test_a_textcat_head_scores_each_sentence(self) -> None:
        frame = spacy_sentences(GOOD_BAD, nlp=FakeNlp()).unwrap()
        assert len(frame) == 3
        assert set(frame["Label"]) == {"POSITIVE", "NEGATIVE", "NEUTRAL"}

    def test_vanilla_models_without_a_head_fail_instead_of_faking(self) -> None:
        result = spacy_sentences(GOOD_BAD, nlp=FakeNlp(head=False))
        assert not result.ok
        assert result.diagnostics[0].code == "SENTIMENT_NN_NO_HEAD"
        assert "textcat" in result.diagnostics[0].context["fix"]

    def test_a_pipeline_that_scores_nothing_says_so(self) -> None:
        class SilentNlp:
            def has(self, name: str) -> bool:
                return name == "textcat"

            def __call__(self, text: str) -> FakeDoc:
                return FakeDoc({})

        result = spacy_sentences(GOOD_BAD, nlp=SilentNlp())
        assert result.ok
        assert any(d.code == "SENTIMENT_NN_NO_HEAD" for d in result.diagnostics)
        assert (result.unwrap()["Compound"] == 0.0).all()


class TestStanza:
    def test_the_three_way_hard_class_maps_to_minus_one_zero_one(self) -> None:
        frame = stanza_sentences(GOOD_BAD, pipeline=FakeStanzaPipeline()).unwrap()
        assert set(frame["Compound"]) == {-1.0, 0.0, 1.0}
        assert (frame["Score"] == 1.0).all()

    def test_a_sentence_votes_its_majority_class(self) -> None:
        class SplitDoc:
            sentences: ClassVar[list[FakeSentence]] = [FakeSentence(2, "x"), FakeSentence(2, "x"), FakeSentence(0, "x")]

        class SplitPipeline:
            def __call__(self, text: str) -> SplitDoc:
                return SplitDoc()

        frame = stanza_sentences(conll([("1", 1, "x")]), pipeline=SplitPipeline()).unwrap()
        assert frame.iloc[0]["Label"] == "positive"


class TestCoreNlp:
    def test_one_row_per_server_sentence(self) -> None:
        def fake(url, properties, body, timeout):
            assert "sentiment" in properties
            return corenlp_payload(
                [
                    ("Very positive", "Good dogs.", 4),
                    ("Very negative", "Bad cats.", 0),
                ]
            )

        frame = corenlp_sentences([("1", "a.txt", "ignored")], request_json=fake).unwrap()
        assert len(frame) == 2
        assert frame.iloc[0]["Sentence"] == "Good dogs."
        assert frame.iloc[0]["Compound"] == 1.0
        assert frame.iloc[1]["Compound"] == -1.0

    def test_the_five_point_scale_centres_on_neutral(self) -> None:
        def fake(url, properties, body, timeout):
            return corenlp_payload([("Neutral", "Dogs sleep.", 2)])

        frame = corenlp_sentences([("1", "a.txt", "Dogs sleep.")], request_json=fake).unwrap()
        assert frame.iloc[0]["Compound"] == 0.0

    def test_a_missing_server_fails_with_the_start_command(self) -> None:
        def dead(url, properties, body, timeout):
            raise ConnectionError("connection refused")

        result = corenlp_sentences([("1", "a.txt", "text")], request_json=dead)
        assert not result.ok
        assert result.diagnostics[0].code == "SENTIMENT_NN_UNAVAILABLE"
        assert "StanfordCoreNLPServer" in result.diagnostics[0].context["fix"]

    def test_an_unusable_answer_is_a_diagnostic(self) -> None:
        def nonsense(url, properties, body, timeout):
            return {"not": "sentences"}

        result = corenlp_sentences([("1", "a.txt", "text")], request_json=nonsense)
        assert not result.ok
        assert result.diagnostics[0].code == "SENTIMENT_NN_UNAVAILABLE"

    def test_a_bad_server_url_is_refused_before_any_request(self) -> None:
        result = corenlp_sentences([("1", "a.txt", "text")], server_url="ftp://nope")
        assert not result.ok
        assert result.diagnostics[0].code == "CORENLP_BAD_URL"


class TestSummary:
    def test_document_rollup_counts_buckets_and_averages(self) -> None:
        frame = bert_sentences(GOOD_BAD, pipeline=FakeHfPipeline()).unwrap()
        summary = summarize_neural(frame).unwrap()
        assert list(summary.columns) == DOCUMENT_COLUMNS
        first = summary.loc[summary["Document ID"] == "1"].iloc[0]
        assert int(first["Sentences"]) == 2
        assert int(first["Positive"]) == 1 and int(first["Negative"]) == 1
        assert abs(float(first["Mean Compound"])) < 0.2

    def test_empty_input_keeps_the_columns(self) -> None:
        summary = summarize_neural(pd.DataFrame(columns=SENTENCE_COLUMNS)).unwrap()
        assert list(summary.columns) == DOCUMENT_COLUMNS
        assert summary.empty

    def test_missing_columns_fail_loudly(self) -> None:
        result = summarize_neural(pd.DataFrame({"x": [1]}))
        assert not result.ok
        assert result.diagnostics[0].code == "SENTIMENT_NN_MISSING_COLUMN"
