"""C6-13 Stanza fake-backed unit tests — mapping logic needs NO downloaded model.

The production mapping in ``StanzaPipeline.parse`` is exercised through a
fake pipeline object shaped exactly like stanza's (tokens/words/NER), so
tokenization/POS/lemma/dependencies/NER/multi-word-token behavior is
pinned offline. Model-backed integration stays marker-separated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.io.reader import Corpus, Document, hash_text
from core.pipelines.stanza_backend import StanzaPipeline, _canonical_ner, build_stanza_pipeline


@dataclass
class FakeWord:
    id: int | tuple[int, int]
    text: str
    lemma: str
    upos: str
    xpos: str
    head: int
    deprel: str


@dataclass
class FakeToken:
    text: str
    ner: str
    words: list[FakeWord]


@dataclass
class FakeSentence:
    tokens: list[FakeToken]

    @property
    def words(self) -> list[FakeWord]:
        return [w for t in self.tokens for w in t.words]


@dataclass
class FakeDoc:
    sentences: list[FakeSentence]


@dataclass
class FakePipeline:
    """Raises on demand so per-document failure paths can be tested."""

    fail_on: str | None = None

    def __call__(self, text: str) -> FakeDoc:
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError(f"simulated backend failure on {self.fail_on!r}")
        return FakeDoc(
            sentences=[
                FakeSentence(
                    tokens=[
                        FakeToken(
                            text="Barack",
                            ner="B-PERSON",
                            words=[FakeWord(1, "Barack", "Barack", "PROPN", "NNP", 2, "nsubj")],
                        ),
                        FakeToken(
                            text="Obama",
                            ner="E-PERSON",
                            words=[FakeWord(2, "Obama", "Obama", "PROPN", "NNP", 3, "obj")],
                        ),
                        FakeToken(
                            text="visited",
                            ner="O",
                            words=[FakeWord(3, "visited", "visit", "VERB", "VBD", 0, "root")],
                        ),
                        FakeToken(
                            text="don't",
                            ner="O",
                            words=[
                                FakeWord((4, 1), "do", "do", "AUX", "VBP", 3, "aux"),
                                FakeWord((4, 2), "n't", "not", "PART", "RB", 3, "advmod"),
                            ],
                        ),
                    ]
                )
            ]
        )


def _pipeline(fake: FakePipeline) -> StanzaPipeline:
    return StanzaPipeline(_nlp=fake, language="en", tasks=frozenset(), model_name="stanza:en-fake")


def _corpus(*texts: str) -> Corpus:
    docs = tuple(
        Document(doc_id=i + 1, path=Path(f"d{i + 1}.txt"), text=text, sha256=hash_text(text))
        for i, text in enumerate(texts)
    )
    return Corpus(docs=docs, sha256="x")


class TestFakeBackedMapping:
    def test_tokenization_pos_lemma_deps(self) -> None:
        frame = _pipeline(FakePipeline()).parse(_corpus("Barack Obama visited")).unwrap()
        assert frame["Form"].tolist() == ["Barack", "Obama", "visited", "do", "n't"]
        assert frame["Lemma"].tolist() == ["Barack", "Obama", "visit", "do", "not"]
        assert frame["POS"].tolist() == ["NNP", "NNP", "VBD", "VBP", "RB"][:5]
        assert int(frame.iloc[0]["Head"]) == 2 and int(frame.iloc[2]["Head"]) == 0

    def test_mwt_tuple_ids_become_record_position(self) -> None:
        frame = _pipeline(FakePipeline()).parse(_corpus("don't")).unwrap()
        # tuple ids (4,1)/(4,1) are not ints: ID falls back to sequential
        assert all(isinstance(v, int) for v in frame["ID"].tolist())

    def test_empty_document_warns_sibling_parses(self) -> None:
        result = _pipeline(FakePipeline()).parse(_corpus("Barack Obama visited", "   "))
        codes = [d.code for d in result.diagnostics]
        assert "EMPTY_DOC" in codes
        assert len(result.unwrap()) == 5  # sibling parsed fully

    def test_parser_exception_is_PARSE_FAILED(self) -> None:
        result = _pipeline(FakePipeline(fail_on="boom")).parse(_corpus("ok text", "boom goes this one"))
        codes = [d.code for d in result.diagnostics]
        assert "PARSE_FAILED" in codes
        # the failed document is named in the diagnostic
        failed = next(d for d in result.diagnostics if d.code == "PARSE_FAILED")
        assert "d2.txt" in str(failed.context.get("path", failed.message))
        # the good sibling's rows survive (partial value, per C6-3 policy)
        assert len(result.unwrap()) == 5

    def test_canonical_ner(self) -> None:
        assert _canonical_ner("B-PERSON") == "PERSON"
        assert _canonical_ner("E-PERSON") == "PERSON"
        assert _canonical_ner("S-GPE") == "GPE"
        assert _canonical_ner("I-MISC") == "MISC"
        assert _canonical_ner("O") == "O"
        assert _canonical_ner("") == "O"

    def test_unsupported_language_needs_no_stanza(self) -> None:
        # Language gate runs BEFORE the import: 'xx' fails even though the
        # fake environment has no stanza at all here.
        result = build_stanza_pipeline("xx", frozenset())
        assert result.value is None
        assert result.errors[0].code == "PIPELINE_UNSUPPORTED"


class TestEnvelopeProvenance:
    def test_model_identity_recorded(self) -> None:
        pipeline = _pipeline(FakePipeline())
        assert pipeline.model_name == "stanza:en-fake"
        assert pipeline.backend == "stanza"
