"""The same question, asked of real speeches through every door it can enter by.

Everything here runs against a folder of real documents -- by default the 87
State of the Union addresses the desktop is developed against -- because each
defect it pins was invisible to a hand-built fixture.

*A fixture contains the words its author typed.* Nobody writes ``U.S.``,
``COVID-19`` or ``9/11`` into a five-token frame, so nothing caught that a
phrase question tokenized by one set of rules and searched a table produced by
another. In this corpus ``U.S.`` occurs 107 times and used to answer zero.

*A fixture has one sentence per document.* A word position counted from the
start of its sentence is indistinguishable from one counted from the start of
its document until a document has several sentences. In this corpus every one
of the 282 occurrences of ``the American people`` reported itself in the first
tenth of its speech.

*A fixture is asked once.* The live workspace and the publisher call different
code to answer the same question, and agreeing about a hand-built frame is not
evidence they agree about 600,000 real tokens.

The parse is shared across the session; see ``real_snapshot`` in the root
conftest. Marked ``model_integration`` because it needs a parser model and a
corpus on disk.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from conftest import real_corpus_dir
from core.profiler.executor import BatchContext, _adapt_phrase_distribution
from core.research.phrase import (
    EvidenceRequest,
    PhraseQuery,
    Tokenization,
    _approximate_tokens,
    track_phrase,
)

pytestmark = [
    pytest.mark.model_integration,
    pytest.mark.skipif(real_corpus_dir() is None, reason="no real corpus on this machine"),
]

#: Phrases whose tokenization is the whole question. Each is checked against the
#: corpus text first, so a corpus that happens not to contain one skips it
#: rather than asserting against zero.
AWKWARD = ("U.S.", "COVID-19", "9/11", "public health", "war on terror")


def _publish(warm: Any, corpus: Any, phrase: str, comparison: str = "", **params: Any) -> dict[str, pd.DataFrame]:
    """The publisher's own answer to the question, as it writes it to CSV."""
    request: dict[str, object] = {
        "phrase": phrase,
        "comparison": comparison,
        "case-sensitive": False,
        "position-bins": 10,
        "snapshot-id": warm.key,
        "evidence-year": 0,
        "evidence-document": "",
        "question-name": "test",
        "question-id": "test",
        "question-revision": 1,
        "resolved-tokens": "",
        "comparison-tokens": "",
        **params,
    }
    context = BatchContext(corpus=corpus, table=warm.table, tokenizer=warm.tokenizer)
    return _adapt_phrase_distribution(context, request).unwrap()


def _live(warm: Any, corpus: Any, phrase: str, **kwargs: Any) -> dict[str, Any]:
    return track_phrase(
        warm.table,
        corpus,
        PhraseQuery(text=phrase, position_bins=10),
        snapshot_id=warm.key,
        evidence=EvidenceRequest(all_rows=True),
        tokenize=warm.tokenize,
        tokenizer_name=warm.tokenizer_name,
        **kwargs,
    )


class TestAPhraseInTheTextIsFoundInTheTable:
    """The oracle is the documents themselves: if the characters are there, the
    question has to find them."""

    @pytest.mark.parametrize("phrase", AWKWARD)
    def test_a_phrase_the_speeches_contain_has_occurrences(
        self, phrase: str, real_corpus: Any, real_snapshot: Any
    ) -> None:
        containing = [doc for doc in real_corpus.docs if phrase in doc.text]
        if not containing:
            pytest.skip(f"this corpus does not contain {phrase!r}")
        answer = _live(real_snapshot, real_corpus, phrase)
        assert answer["summary"]["occurrences"] > 0, (
            f"{phrase!r} appears in {len(containing)} documents of this corpus and the question found none"
        )
        # Every document whose text contains the characters should be reachable
        # through the answer, give or take the ones where the characters fall
        # across a token boundary the parser drew elsewhere.
        found = {row["document_id"] for row in answer["evidence"]["rows"]}
        assert found, f"{phrase!r} has occurrences but no evidence rows"

    def test_the_regular_expression_alone_would_have_missed_them(self, real_corpus: Any, real_snapshot: Any) -> None:
        """Why the snapshot's tokenizer is asked, stated against real text."""
        missed = []
        for phrase in AWKWARD:
            if not any(phrase in doc.text for doc in real_corpus.docs):
                continue
            guessed = _approximate_tokens(phrase)
            parsed = list(real_snapshot.tokenize(phrase))
            if guessed != parsed:
                missed.append((phrase, guessed, parsed))
        assert missed, "expected at least one phrase the guess and the parser disagree about"


class TestTheLiveAnswerAndThePublishedAnswerAreOneAnswer:
    """The workspace shows an answer and the publisher writes one. They are the
    same question or the workspace is a preview of nothing."""

    @pytest.mark.parametrize("phrase", AWKWARD)
    def test_counts_and_occurrences_match(self, phrase: str, real_corpus: Any, real_snapshot: Any) -> None:
        live = _live(real_snapshot, real_corpus, phrase)
        published = _publish(real_snapshot, real_corpus, phrase)

        occurrences = published["phrase_occurrences.csv"]
        assert len(occurrences) == live["summary"]["occurrences"]
        summary = published["phrase_summary.csv"].iloc[0]
        assert summary["occurrences"] == live["summary"]["occurrences"]
        assert summary["matching_documents"] == live["summary"]["matching_documents"]
        assert summary["word_tokens"] == live["summary"]["word_tokens"]

        # Not just the totals: every occurrence, in order, by identity and by
        # both coordinate systems.
        if live["summary"]["occurrences"]:
            rows = live["evidence"]["rows"]
            assert list(occurrences["id"]) == [row["id"] for row in rows]
            assert list(occurrences["token_start"]) == [row["token_start"] for row in rows]
            assert list(occurrences["word_start"]) == [row["word_start"] for row in rows]
            assert list(occurrences["position_bin"]) == [row["position_bin"] for row in rows]
            assert list(occurrences["character_start"]) == [row["character_start"] for row in rows]

    def test_a_published_comparison_is_the_comparison_that_was_asked(
        self, real_corpus: Any, real_snapshot: Any
    ) -> None:
        published = _publish(real_snapshot, real_corpus, "the American people", comparison="the Congress")
        occurrences = published["phrase_occurrences.csv"]
        for subject in ("the American people", "the Congress"):
            live = _live(real_snapshot, real_corpus, subject)
            mine = occurrences[occurrences["subject"] == subject]
            assert len(mine) == live["summary"]["occurrences"], subject
        question = published["phrase_question.csv"].iloc[0]
        assert json.loads(question["resolved_tokens"]) == ["the", "American", "people"]
        assert json.loads(question["comparison_tokens"]) == ["the", "Congress"]

    def test_without_the_tokenizer_the_published_answer_diverges(self, real_corpus: Any, real_snapshot: Any) -> None:
        """The defect this contract exists to prevent, reproduced on purpose.

        A publisher given no tokenizer falls back to the guess, and the guess
        cannot find ``U.S.`` in text that is full of it. The published tables
        would have said zero under the same question name as a workspace
        showing a hundred.
        """
        if not any("U.S." in doc.text for doc in real_corpus.docs):
            pytest.skip("this corpus does not contain 'U.S.'")
        blind = _adapt_phrase_distribution(
            BatchContext(corpus=real_corpus, table=real_snapshot.table),
            {
                "phrase": "U.S.",
                "comparison": "",
                "case-sensitive": False,
                "position-bins": 10,
                "snapshot-id": real_snapshot.key,
                "evidence-year": 0,
                "evidence-document": "",
                "question-name": "test",
                "question-id": "test",
                "question-revision": 1,
                "resolved-tokens": "",
                "comparison-tokens": "",
            },
        ).unwrap()
        with_tokenizer = _publish(real_snapshot, real_corpus, "U.S.")
        assert len(blind["phrase_occurrences.csv"]) == 0
        assert len(with_tokenizer["phrase_occurrences.csv"]) > 0
        assert blind["phrase_question.csv"].iloc[0]["matching_source"] == "approximate"
        assert with_tokenizer["phrase_question.csv"].iloc[0]["matching_source"] == "snapshot"


class TestPublishingASavedQuestionAsksTheSavedQuestion:
    def test_recorded_tokens_survive_a_tokenizer_that_would_disagree(
        self, real_corpus: Any, real_snapshot: Any
    ) -> None:
        """A saved question carries its tokens, so a parser change is visible.

        The stand-in for "the parser changed" is the rule this workspace
        actually used to use: the regular expression, which splits ``U.S.``
        into ``U``, ``.``, ``S``, ``.`` and finds nothing.
        """
        if not any("U.S." in doc.text for doc in real_corpus.docs):
            pytest.skip("this corpus does not contain 'U.S.'")
        live = _live(real_snapshot, real_corpus, "U.S.")
        saved = live["question"]["subject"]["tokens"]

        different = BatchContext(
            corpus=real_corpus,
            table=real_snapshot.table,
            tokenizer=Tokenization(_approximate_tokens, "regular-expression"),
        )
        base: dict[str, object] = {
            "phrase": "U.S.",
            "comparison": "",
            "case-sensitive": False,
            "position-bins": 10,
            "snapshot-id": real_snapshot.key,
            "evidence-year": 0,
            "evidence-document": "",
            "question-name": "test",
            "question-id": "test",
            "question-revision": 1,
            "comparison-tokens": "",
        }
        carried = _adapt_phrase_distribution(different, {**base, "resolved-tokens": json.dumps(saved)}).unwrap()
        forgotten = _adapt_phrase_distribution(different, {**base, "resolved-tokens": ""}).unwrap()

        assert len(carried["phrase_occurrences.csv"]) == live["summary"]["occurrences"]
        assert len(forgotten["phrase_occurrences.csv"]) == 0
        assert carried["phrase_question.csv"].iloc[0]["matching_source"] == "saved"
        assert forgotten["phrase_question.csv"].iloc[0]["matching_source"] == "snapshot"


class TestWhereInTheSpeechIsADocumentCoordinate:
    """A position is a fraction of the document, not of the sentence it is in."""

    PHRASE = "the American people"

    def _rows(self, warm: Any, corpus: Any) -> list[dict[str, Any]]:
        answer = track_phrase(
            warm.table,
            corpus,
            PhraseQuery(text=self.PHRASE, position_bins=10),
            snapshot_id=warm.key,
            evidence=EvidenceRequest(all_rows=True),
            tokenize=warm.tokenize,
            tokenizer_name=warm.tokenizer_name,
        )
        if not answer["evidence"]["rows"]:
            pytest.skip(f"this corpus does not contain {self.PHRASE!r}")
        return answer["evidence"]["rows"]

    def test_the_word_position_counts_from_the_start_of_the_document(
        self, real_corpus: Any, real_snapshot: Any
    ) -> None:
        """Checked against a count taken a different way: words before the
        occurrence's first token, over the document's whole token list."""
        from core.research.phrase import _is_word

        ids = real_snapshot.table["Document ID"].astype(str)
        by_document = {str(key): group for key, group in real_snapshot.table.groupby(ids, sort=False)}
        for row in self._rows(real_snapshot, real_corpus)[:40]:
            doc = next(d for d in real_corpus.docs if str(d.source_id) == row["document_id"])
            forms = [str(v) for v in by_document[str(doc.doc_id)]["Form"].fillna("").tolist()]
            independent = sum(_is_word(form) for form in forms[: row["token_start"]])
            assert row["word_start"] == independent, f"{doc.name} at token {row['token_start']}"

    def test_occurrences_land_across_the_document_not_all_at_its_start(
        self, real_corpus: Any, real_snapshot: Any
    ) -> None:
        rows = self._rows(real_snapshot, real_corpus)
        bins = {row["position_bin"] for row in rows}
        assert len(bins) > 1, "every occurrence in one tenth of its document is the sentence-local bug"
        assert max(row["relative_position"] or 0 for row in rows) > 0.5, "nothing past the halfway point of any speech"

    def test_a_phrase_late_in_a_speech_reports_a_late_position(self, real_corpus: Any, real_snapshot: Any) -> None:
        rows = self._rows(real_snapshot, real_corpus)
        late = [row for row in rows if (row["relative_position"] or 0) > 0.9]
        assert late, "no occurrence in the last tenth of any speech"
        for row in late:
            assert row["position_bin"] == 10


class TestTheEvidenceOnScreenIsLabelledByWhatIsOnScreen:
    def _answer(self, warm: Any, corpus: Any, **evidence: Any) -> dict[str, Any]:
        return track_phrase(
            warm.table,
            corpus,
            PhraseQuery(text="the American people", position_bins=10),
            snapshot_id=warm.key,
            evidence=EvidenceRequest(**evidence),
            tokenize=warm.tokenize,
            tokenizer_name=warm.tokenizer_name,
        )

    def test_every_speech_can_be_pointed_at_its_own_words(self, real_corpus: Any, real_snapshot: Any) -> None:
        """Alignment over the whole corpus, which used to fail everywhere.

        spaCy emits newlines as tokens and every one of these files begins
        with one, so each document failed on its first token and lost all its
        spans. This asserts the corpus-wide figure precisely because the
        failure was corpus-wide and looked like a property of the text.
        """
        answer = self._answer(real_snapshot, real_corpus, all_rows=True)
        offsets = answer["evidence"]["source_offsets"]
        assert offsets["total"] == answer["summary"]["occurrences"]
        assert offsets["status"] == "verified", (
            f"only {offsets['verified']} of {offsets['total']} occurrences could be located in the source"
        )

    def test_a_drill_down_reports_the_coverage_of_what_it_selected(self, real_corpus: Any, real_snapshot: Any) -> None:
        whole = self._answer(real_snapshot, real_corpus, all_rows=True)
        busiest = max(whole["documents"], key=lambda row: row["occurrences"])
        if not busiest["occurrences"]:
            pytest.skip("this corpus does not contain the phrase")
        filtered = self._answer(real_snapshot, real_corpus, document_id=busiest["document_id"], all_rows=True)
        offsets = filtered["evidence"]["source_offsets"]
        # The filtered figures describe the selection, the plain ones the corpus.
        assert offsets["filtered_total"] == busiest["occurrences"]
        assert offsets["filtered_verified"] == sum(
            bool(row["exact_source_highlight"]) for row in filtered["evidence"]["rows"]
        )
        assert offsets["total"] == whole["summary"]["occurrences"]
        assert offsets["filtered_total"] < offsets["total"]

    def test_highlights_point_at_the_phrase_in_the_imported_text(self, real_corpus: Any, real_snapshot: Any) -> None:
        """The offsets are read back out of the documents, which is the only
        check that they mean what they say."""
        answer = track_phrase(
            real_snapshot.table,
            real_corpus,
            PhraseQuery(text="the American people", position_bins=10),
            snapshot_id=real_snapshot.key,
            evidence=EvidenceRequest(offset=0, limit=60),
            tokenize=real_snapshot.tokenize,
            tokenizer_name=real_snapshot.tokenizer_name,
        )
        verified = [row for row in answer["evidence"]["rows"] if row["exact_source_highlight"]]
        assert verified, "no occurrence could be pointed at in the source text"
        for row in verified:
            doc = next(d for d in real_corpus.docs if str(d.source_id) == row["document_id"])
            quoted = doc.text[row["character_start"] : row["character_end"]]
            assert quoted.casefold() == row["match_source"].casefold()
            assert "american people" in quoted.casefold()
