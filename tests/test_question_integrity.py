"""A question must be asked of the corpus it names, in that corpus's own words.

Two failures motivate this file, and neither one announces itself in the answer.

*The query and the corpus disagreed about what a word is.* Queries were split by
a regular expression that approximated one parser's English rules, whatever
parser had actually produced the table. A corpus containing ``U.S.`` answered
"no occurrences" for ``U.S.`` -- not an error, not a warning, a confident zero.

*An answer could cite one corpus and be computed from another.* Loading set the
parser and the selection before parsing and the parse afterwards, so when two
loads overlapped the slower one's table ended up behind the faster one's labels.
Every number was real; the description of where they came from was not.

Both are invisible downstream, so they are pinned here rather than left to a
reading of the code.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import threading
from typing import Any

import pandas as pd
import pytest

from core.io.reader import Corpus, Document
from core.research.phrase import (
    MATCHING_PROFILE_VERSION,
    PhraseQuery,
    _approximate_tokens,
    track_phrase,
)
from desktop_backend.live import Session, StaleSnapshot, Warm

# --------------------------------------------------------------- tokenizing --

# What spaCy's English tokenizer actually does with each of these. Checked
# against the installed model by the model_integration test below, so this
# table cannot drift into wishful thinking.
PARSER_TOKENS: dict[str, tuple[str, ...]] = {
    "U.S.": ("U.S.",),
    "COVID-19": ("COVID-19",),
    "3.5": ("3.5",),
    "AT&T": ("AT&T",),
    "state_of_the_art": ("state_of_the_art",),
    "public health": ("public", "health"),
}


class TestTheQueryIsSplitLikeTheCorpus:
    @pytest.mark.parametrize("text", sorted(PARSER_TOKENS))
    def test_the_snapshot_decides_what_a_word_is(self, text: str) -> None:
        expected = PARSER_TOKENS[text]
        assert PhraseQuery(text).tokens(lambda value: PARSER_TOKENS[value]) == expected

    @pytest.mark.parametrize("text", ["U.S.", "COVID-19", "3.5", "AT&T", "state_of_the_art"])
    def test_guessing_gets_these_wrong_which_is_why_the_snapshot_is_asked(self, text: str) -> None:
        # Not a test of the fallback's quality -- a record of why a fallback
        # must never be used when a real tokenizer is available.
        assert tuple(_approximate_tokens(text)) != PARSER_TOKENS[text]

    def test_a_phrase_in_the_text_is_found_when_the_rules_agree(self) -> None:
        table, corpus = _table([("a.txt", 1, ["The", "U.S.", "economy", "grew"])])
        answer = track_phrase(
            table,
            corpus,
            PhraseQuery("U.S."),
            snapshot_id="s",
            tokenize=lambda text: PARSER_TOKENS[text],
            tokenizer_name="spacy/en_core_web_sm",
        )
        assert answer["summary"]["occurrences"] == 1
        assert answer["evidence"]["rows"][0]["match"] == "U.S."

    def test_the_same_phrase_is_lost_when_they_disagree(self) -> None:
        # The reported bug, kept as a test so the reason the parameter exists
        # cannot be optimised away: guessing returns a confident zero.
        table, corpus = _table([("a.txt", 1, ["The", "U.S.", "economy", "grew"])])
        answer = track_phrase(table, corpus, PhraseQuery("U.S."), snapshot_id="s")
        assert answer["summary"]["occurrences"] == 0

    def test_the_answer_records_which_rules_produced_it(self) -> None:
        # A saved question reopened under different rules is a different
        # question, and only a recorded profile can tell the two apart.
        table, corpus = _table([("a.txt", 1, ["The", "U.S.", "economy"])])
        answer = track_phrase(
            table,
            corpus,
            PhraseQuery("U.S."),
            snapshot_id="s",
            tokenize=lambda text: PARSER_TOKENS[text],
            tokenizer_name="spacy/en_core_web_sm",
        )
        profile = answer["question"]["matching_profile"]
        assert profile == {
            "version": MATCHING_PROFILE_VERSION,
            "tokenizer": "spacy/en_core_web_sm",
            "resolved_by_snapshot": True,
            "source": "snapshot",
            # Recorded even when off: "this answer did not count inflections"
            # is a fact about it, and an absent key is not that claim.
            "normalize": False,
            "match_lemma": False,
            "match_nominalization": False,
            "unavailable": [],
        }

    def test_an_answer_resolved_without_a_snapshot_admits_it(self) -> None:
        table, corpus = _table([("a.txt", 1, ["public", "health"])])
        answer = track_phrase(table, corpus, PhraseQuery("public health"), snapshot_id="s")
        assert answer["question"]["matching_profile"]["resolved_by_snapshot"] is False


@pytest.mark.model_integration
class TestTheRealParserAgrees:
    """The table above is only useful if it matches the installed tokenizer."""

    @staticmethod
    def _pipeline() -> Any:
        from core.config import NLPConfig
        from core.pipelines.cache import PipelineCache
        from core.pipelines.resolve import resolve_pipeline

        return resolve_pipeline(PipelineCache(), NLPConfig(parser="spacy", language="en")).unwrap()

    @pytest.mark.parametrize("text", sorted(PARSER_TOKENS))
    def test_the_expected_tokens_are_the_parsers_tokens(self, text: str) -> None:
        assert self._pipeline().tokenize(text) == PARSER_TOKENS[text]

    @pytest.mark.parametrize(
        "text",
        [
            "U.S.",  # abbreviation
            "COVID-19",  # hyphenated term
            "3.5",  # decimal
            "AT&T",  # symbol
            "state_of_the_art",  # underscores
            "can't",  # contraction
            "naïve résumé",  # non-ASCII
            "don’t stop",  # noqa: RUF001 - a curly apostrophe is exactly what gets typed
        ],
    )
    def test_a_query_tokenizes_to_what_a_parse_of_the_same_text_contains(self, text: str) -> None:
        # The join that matters: query tokens have to be findable in the Form
        # column a parse of the same characters produces.
        pipeline = self._pipeline()
        corpus = Corpus(
            docs=(Document(doc_id=1, path=Path("a.txt"), text=text, sha256="x"),),
            sha256="c",
        )
        forms = [str(value) for value in pipeline.parse(corpus).unwrap()["Form"].tolist()]
        assert list(pipeline.tokenize(text)) == forms


# ------------------------------------------------------------- snapshot own --


def _table(sentences: list[tuple[str, int, list[str]]]) -> tuple[pd.DataFrame, Corpus]:
    rows = []
    texts: dict[str, list[str]] = {}
    for document, sentence_id, tokens in sentences:
        texts.setdefault(document, []).extend(tokens)
        for position, token in enumerate(tokens, start=1):
            rows.append(
                {
                    "ID": position,
                    "Form": token,
                    "Sentence ID": sentence_id,
                    "Document ID": 1 + sorted(texts).index(document),
                    "Document": document,
                }
            )
    docs = tuple(
        Document(
            doc_id=index + 1,
            path=Path(name),
            text=" ".join(texts[name]),
            sha256=f"sha-{name}",
            date=date(2000 + index, 1, 1),
        )
        for index, name in enumerate(sorted(texts))
    )
    return pd.DataFrame(rows), Corpus(docs=docs, sha256="corpus")


class _SlowBench:
    """A bench whose parse finishes exactly when the test says so."""

    def __init__(self) -> None:
        self.gates: dict[str, threading.Event] = {}
        self.released = 0
        self.holding: list[str] = []
        self.forgotten: list[str] = []

    def gate(self, name: str) -> threading.Event:
        return self.gates.setdefault(name, threading.Event())

    def warm(self, corpus: Corpus, parser: str, *, stage: Any = None) -> Warm:
        name = corpus.sha256
        self.gate(name).wait(timeout=5)
        # A real bench holds the parse before it returns it, which is what
        # makes a superseded load able to leave one behind.
        self.holding.append(f"snapshot-{name}")
        return Warm(corpus=corpus, key=f"snapshot-{name}", identity={}, table=None)

    def release(self) -> None:
        self.released += 1
        self.holding.clear()

    def forget(self, key: str) -> None:
        self.forgotten.append(key)
        self.holding = [held for held in self.holding if held != key]


def _corpus(name: str) -> Corpus:
    return Corpus(
        docs=(Document(doc_id=1, path=Path(f"{name}.txt"), text=name, sha256=name),),
        sha256=name,
    )


class TestOverlappingLoadsCannotMixState:
    def test_a_late_load_does_not_overwrite_a_newer_one(self) -> None:
        # The reproduction from the review: start A, finish B, then release A.
        # A's table must not end up behind B's parser and selection.
        bench = _SlowBench()
        session = Session(bench)  # type: ignore[arg-type]
        first = threading.Thread(
            target=session.warm, args=(_corpus("A"), "spacy"), kwargs={"selection": {"ids": ["A"]}}
        )
        first.start()
        second = threading.Thread(
            target=session.warm, args=(_corpus("B"), "stanza"), kwargs={"selection": {"ids": ["B"]}}
        )
        second.start()
        bench.gate("B").set()
        second.join(timeout=5)
        bench.gate("A").set()
        first.join(timeout=5)

        state = session.state()
        assert state["snapshot_id"] == "snapshot-B"
        assert state["selection"] == {"ids": ["B"]}
        assert state["parser"] == "stanza"
        assert state["state"] == "ready"
        # B is what the session answers from, so A's table is dropped and B's
        # is not: releasing everything would cost the current corpus its parse.
        assert bench.forgotten == ["snapshot-A"]
        assert bench.holding == ["snapshot-B"]

    def test_the_three_always_describe_one_load(self) -> None:
        bench = _SlowBench()
        session = Session(bench)  # type: ignore[arg-type]
        for name, parser in (("A", "spacy"), ("B", "stanza")):
            bench.gate(name).set()
            session.warm(_corpus(name), parser, selection={"ids": [name]})
            state = session.state()
            assert (state["snapshot_id"], state["parser"], state["selection"]) == (
                f"snapshot-{name}",
                parser,
                {"ids": [name]},
            )

    def test_cooling_during_a_load_is_not_undone_by_it(self) -> None:
        # Otherwise "release these documents" frees the memory and then a parse
        # still in flight hands them back, with nothing on screen having asked.
        bench = _SlowBench()
        session = Session(bench)  # type: ignore[arg-type]
        loading = threading.Thread(target=session.warm, args=(_corpus("A"), "spacy"))
        loading.start()
        session.cool()
        bench.gate("A").set()
        loading.join(timeout=5)

        assert session.state()["state"] == "cold"
        assert session.state()["snapshot_id"] == ""
        # And the parse that arrived too late is not still sitting in memory
        # behind a session that reports having released it.
        assert bench.forgotten == ["snapshot-A"]
        assert bench.holding == []

    def test_a_superseded_failure_does_not_blame_the_current_load(self) -> None:
        class _Failing(_SlowBench):
            def warm(self, corpus: Corpus, parser: str, *, stage: Any = None) -> Warm:
                self.gate(corpus.sha256).wait(timeout=5)
                if corpus.sha256 == "A":
                    raise ValueError("A could not be parsed")
                return Warm(corpus=corpus, key=f"snapshot-{corpus.sha256}", identity={}, table=None)

        bench = _Failing()
        session = Session(bench)  # type: ignore[arg-type]
        first = threading.Thread(target=session.warm, args=(_corpus("A"), "spacy"))
        first.start()
        second = threading.Thread(target=session.warm, args=(_corpus("B"), "spacy"))
        second.start()
        bench.gate("B").set()
        second.join(timeout=5)
        bench.gate("A").set()
        first.join(timeout=5)

        state = session.state()
        assert state["state"] == "ready", state
        assert state["error"] == ""


class TestAQuestionNamesItsEvidence:
    @staticmethod
    def _ready() -> Session:
        bench = _SlowBench()
        session = Session(bench)  # type: ignore[arg-type]
        bench.gate("A").set()
        session.warm(_corpus("A"), "spacy", selection={"ids": ["A"]})
        return session

    def test_citing_another_snapshot_is_refused_not_answered(self) -> None:
        # Answering it from whatever is loaded now produces something that
        # looks exactly like the answer that was asked for.
        session = self._ready()
        with pytest.raises(StaleSnapshot):
            session.resolved("snapshot-somewhere-else")

    def test_citing_the_loaded_snapshot_is_allowed(self) -> None:
        session = self._ready()
        assert session.resolved("snapshot-A").id == "snapshot-A"

    def test_citing_nothing_still_works_for_a_first_question(self) -> None:
        assert self._ready().resolved().id == "snapshot-A"

    def test_a_stale_reference_is_its_own_kind_of_refusal(self) -> None:
        # The interface has to offer to load the documents again; it cannot do
        # that with a message it has to read.
        session = self._ready()
        with pytest.raises(StaleSnapshot):
            session.track_phrase({"text": "anything", "snapshot_id": "snapshot-gone"})

    def test_asking_before_loading_is_not_a_stale_reference(self) -> None:
        # Nothing to carry on from; the reader has simply not started.
        bench = _SlowBench()
        session = Session(bench)  # type: ignore[arg-type]
        with pytest.raises(ValueError) as raised:
            session.resolved()
        assert not isinstance(raised.value, StaleSnapshot)
