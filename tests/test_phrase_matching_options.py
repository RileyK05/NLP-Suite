"""What counts as a match, when the reader widens the question.

Three options, each off by default and each *adding* forms rather than
replacing them: fold the typography, count every inflection, count the noun a
verb turns into. They exist because the literal question answers a narrower
thing than people mean. "Did they mention COVID-19?" is not a question about
which hyphen was typed, and "how often do they promise?" is not a question
about the present tense.

The oracle here comes from the corpus wherever it can. A fixture asked whether
"promise" matches "promised" would answer from whatever lemma its author typed
into the frame; the real parse answers from what spaCy actually assigned to
596,875 tokens of real speech, which is the thing the feature depends on.

Nominalization is stubbed the way ``test_nominalization.py`` stubs it, because
the WordNet corpus is an optional download and a test that skips is not a
check. The real-data half of it is marked and skips when the data is absent.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pytest

from conftest import real_corpus_dir
from core.research.phrase import (
    MATCHING_PROFILE_VERSION,
    EvidenceRequest,
    PhraseQuery,
    _accepted,
    _fold,
    track_phrase,
)

real_only = pytest.mark.skipif(real_corpus_dir() is None, reason="no real corpus on this machine")


def ask(snapshot: Any, corpus: Any, text: str, **options: bool) -> dict[str, Any]:
    """One question, with every occurrence returned rather than a page of them.

    The default page is 100 rows. Comparing two answers by their pages
    compares the first 100 of each, which differ as soon as one of them has
    more occurrences -- so a widening would look like a replacement.
    """
    return track_phrase(
        snapshot.table,
        corpus,
        PhraseQuery(text=text, **options),
        snapshot_id=snapshot.key,
        evidence=EvidenceRequest(all_rows=True),
        tokenize=snapshot.tokenize,
        tokenizer_name=snapshot.tokenizer_name,
    )


def where(answer: dict[str, Any]) -> set[tuple[str, int, int]]:
    """Every occurrence as a place in the corpus.

    Not by occurrence id: an id hashes the phrase as typed, so the same words
    found by ``COVID-19`` and by the same text typed with a non-breaking
    hyphen are deliberately different
    ids -- they are occurrences of two different questions that happen to land
    on one span of text. The span is what "the same occurrence" means here.
    """
    return {(row["document_id"], row["token_start"], row["token_end"]) for row in answer["evidence"]["rows"]}


def counted(answer: dict[str, Any]) -> list[list[str]]:
    return answer["question"]["subject"]["counted_forms"]


class TestFoldingTypography:
    """Normalization, which is about how a character was typed."""

    def test_the_several_dashes_are_one_dash(self) -> None:
        assert _fold("COVID\u201119") == _fold("COVID-19") == "COVID-19"
        assert _fold("well\u2014known") == "well-known"

    def test_curly_and_straight_quotes_are_one_quote(self) -> None:
        assert _fold("don\u2019t") == _fold("don't") == "don't"

    def test_accents_fold_to_their_letters(self) -> None:
        assert _fold("caf\u00e9") == "cafe"
        # Precomposed and decomposed spellings of the same word are one word.
        assert _fold("\u00e9lan") == _fold("e\u0301lan")

    def test_folding_never_changes_which_word_it_is(self) -> None:
        """It unifies spellings, and is not a stemmer. A fold that shortened a
        word would make two different words match and call it typography."""
        for word in ("promised", "government", "U.S.", "9/11", "AT&T"):
            assert _fold(word) == word


@pytest.mark.model_integration
@real_only
class TestNormalizingFindsWhatTheHyphenHid:
    def test_a_non_breaking_hyphen_finds_nothing_on_its_own(self, real_snapshot: Any, real_corpus: Any) -> None:
        """The trap, on real text: the query and the corpus disagree about one
        invisible character and the answer is a confident zero."""
        typed = ask(real_snapshot, real_corpus, "COVID\u201119")
        assert typed["summary"]["occurrences"] == 0

    def test_normalizing_finds_exactly_what_the_plain_hyphen_finds(self, real_snapshot: Any, real_corpus: Any) -> None:
        """Not "more than zero" -- the same occurrences, occurrence for
        occurrence, as asking with the hyphen the documents actually use."""
        plain = ask(real_snapshot, real_corpus, "COVID-19")
        folded = ask(real_snapshot, real_corpus, "COVID\u201119", normalize=True)
        assert plain["summary"]["occurrences"] > 0
        assert folded["summary"]["occurrences"] == plain["summary"]["occurrences"]
        assert where(folded) == where(plain)


@pytest.mark.model_integration
@real_only
class TestCountingEveryInflection:
    def test_the_inflections_counted_are_the_ones_the_corpus_shares_a_lemma_with(
        self, real_snapshot: Any, real_corpus: Any
    ) -> None:
        """Checked against the parse rather than against a list of endings.

        The claim is "every form this corpus lemmatises to the same word", so
        the parse decides what those are. A hand-written list would pass on a
        corpus where the parser disagrees with it.
        """
        answer = ask(real_snapshot, real_corpus, "promise", match_lemma=True)
        table = real_snapshot.table
        lemma = table.loc[table["Form"].str.casefold() == "promise", "Lemma"].str.casefold()
        expected = set(table.loc[table["Lemma"].str.casefold().isin(set(lemma)), "Form"].str.casefold())
        assert set(counted(answer)[0]) == expected
        assert len(expected) > 1, "expected the real corpus to inflect this word"

    def test_it_only_ever_adds(self, real_snapshot: Any, real_corpus: Any) -> None:
        """Every literal occurrence is still an occurrence. An option that
        widened by substituting would be a different question wearing this
        one's name."""
        literal = ask(real_snapshot, real_corpus, "promise")
        widened = ask(real_snapshot, real_corpus, "promise", match_lemma=True)
        assert widened["summary"]["occurrences"] > literal["summary"]["occurrences"]
        assert where(literal) <= where(widened)

    def test_the_evidence_says_which_word_was_actually_found(self, real_snapshot: Any, real_corpus: Any) -> None:
        """A row echoing the query back would hide that the hit was
        "promised"; the reader would see 174 rows all reading "promise"."""
        answer = ask(real_snapshot, real_corpus, "promise", match_lemma=True)
        found = {" ".join(row["matched_forms"]).casefold() for row in answer["evidence"]["rows"]}
        assert len(found) > 1, found
        assert found <= set(counted(answer)[0])

    def test_each_token_of_a_phrase_widens_on_its_own(self, real_snapshot: Any, real_corpus: Any) -> None:
        answer = ask(real_snapshot, real_corpus, "the American people", match_lemma=True)
        assert len(counted(answer)) == 3
        assert "americans" in counted(answer)[1]
        assert "peoples" in counted(answer)[2]


class FakeSynset:
    def pos(self) -> str:
        return "v"


class FakeForm:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def synset(self) -> FakeSynset:
        return FakeSynset()


class FakeLemma:
    def __init__(self, verb: str) -> None:
        self._verb = verb

    def derivationally_related_forms(self) -> list[FakeForm]:
        return [FakeForm(self._verb)]


class StubWordNet:
    """Only the two questions the backend asks, as the real module answers them."""

    NOUN = "n"
    MAP: ClassVar[dict[str, str]] = {"government": "govern", "governance": "govern"}

    def lemmas(self, word: str, pos: str) -> list[FakeLemma]:
        assert pos == "n"
        return [FakeLemma(self.MAP[word])] if word in self.MAP else []


@pytest.fixture
def wordnet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the derivation lookup answer, without the optional data.

    Patched at ``default_wordnet_module`` rather than at this module's own
    helper on purpose: unwrapping that Result is the step that was wrong, and
    a test that patched past it would have agreed with the defect.
    """
    import core.analysis.nominalization as nom
    from core.result import Result

    monkeypatch.setattr(nom, "default_wordnet_module", lambda *a, **k: Result.success(StubWordNet()))


def parsed(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """A minimal table: one sentence of (Form, Lemma, POS)."""
    return pd.DataFrame(
        [
            {
                "ID": index,
                "Form": form,
                "Lemma": lemma,
                "POS": pos,
                "Sentence ID": "s1",
                "Document ID": "1",
            }
            for index, (form, lemma, pos) in enumerate(rows, 1)
        ]
    )


CABINET = parsed(
    [
        ("The", "the", "DT"),
        ("government", "government", "NN"),
        ("governs", "govern", "VBZ"),
        ("governments", "government", "NNS"),
        ("governance", "governance", "NN"),
        ("govern", "govern", "VB"),
        ("governed", "govern", "VBN"),
    ]
)


class TestCountingTheNounAVerbTurnsInto:
    def test_a_verb_reaches_the_noun_derived_from_it(self, wordnet: None) -> None:
        accepted = _accepted(CABINET, ["govern"], PhraseQuery(text="govern", match_nominalization=True))
        assert set(accepted.forms[0]) >= {"govern", "government", "governments", "governance"}

    def test_a_noun_reaches_back_to_its_verb_and_its_siblings(self, wordnet: None) -> None:
        accepted = _accepted(CABINET, ["government"], PhraseQuery(text="government", match_nominalization=True))
        assert set(accepted.forms[0]) >= {"government", "govern", "governance"}

    def test_it_does_not_quietly_do_inflections_as_well(self, wordnet: None) -> None:
        """The defect this was found with. WordNet was returning nothing at
        all -- the Result was being passed where the module belonged -- and
        the option looked like it worked because it was adding the typed
        word's own inflections, which is the other option's job.
        """
        accepted = _accepted(CABINET, ["govern"], PhraseQuery(text="govern", match_nominalization=True))
        assert "governed" not in accepted.forms[0]
        assert "governs" not in accepted.forms[0]

    def test_with_inflections_ticked_as_well_it_does_both(self, wordnet: None) -> None:
        accepted = _accepted(
            CABINET,
            ["govern"],
            PhraseQuery(text="govern", match_lemma=True, match_nominalization=True),
        )
        assert set(accepted.forms[0]) >= {"govern", "governed", "governs", "government", "governance"}

    def test_an_unavailable_lookup_is_reported_rather_than_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Silence here is the whole bug: an option that cannot work and does
        not say so is indistinguishable from one that is switched off."""
        import core.analysis.nominalization as nom
        from core.result import Diagnostic, Result

        monkeypatch.setattr(
            nom,
            "default_wordnet_module",
            lambda *a, **k: Result.failure(
                Diagnostic.error("NOMINALIZATION_DATA_MISSING", "no WordNet here", fix="download it")
            ),
        )
        accepted = _accepted(CABINET, ["govern"], PhraseQuery(text="govern", match_nominalization=True))
        assert accepted.unavailable
        assert "no WordNet here" in accepted.unavailable[0]
        assert "download it" in accepted.unavailable[0]
        assert accepted.forms[0] == ("govern",)

    def test_a_parse_without_a_pos_column_says_so(self, wordnet: None) -> None:
        accepted = _accepted(
            CABINET.drop(columns=["POS"]),
            ["govern"],
            PhraseQuery(text="govern", match_nominalization=True),
        )
        assert any("POS" in reason for reason in accepted.unavailable)


class TestTheLiteralQuestionIsUnchanged:
    def test_nothing_ticked_accepts_only_the_token(self) -> None:
        accepted = _accepted(CABINET, ["govern"], PhraseQuery(text="govern"))
        assert accepted.keys == (frozenset({"govern"}),)
        assert accepted.unavailable == ()

    def test_the_version_moved_but_the_default_question_did_not(self) -> None:
        """Version 3 is version 2 plus three options that default to off, so a
        question saved under 2 and re-asked here is the same question."""
        assert MATCHING_PROFILE_VERSION == 3
        query = PhraseQuery(text="govern")
        assert not query.widened
        assert _accepted(CABINET, ["govern"], query).forms == (("govern",),)
