"""Occurrence evidence and aggregates for question-driven phrase research."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from core.io.reader import Corpus, Document, corpus_fingerprint
from core.research.phrase import EvidenceRequest, PhraseQuery, source_passage, track_phrase


def _corpus(tmp_path: Path) -> Corpus:
    return Corpus(
        (
            Document(
                1,
                tmp_path / "stored-a.txt",
                "alpha beta alpha beta.",
                date(2020, 5, 2),
                "hash-a",
                "stable-a",
                "2020-05-02 First.txt",
            ),
            Document(
                2,
                tmp_path / "stored-b.txt",
                "alpha, beta gamma gamma gamma gamma.",
                date(2022, 1, 1),
                "hash-b",
                "stable-b",
                "2022 Second.txt",
            ),
            Document(3, tmp_path / "stored-c.txt", "quiet text", None, "hash-c", "stable-c", "Undated.txt"),
        ),
        "corpus-fingerprint",
    )


def _table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def sentence(document: int, sentence_id: int, forms: list[str]) -> None:
        for token_id, form in enumerate(forms, 1):
            rows.append(
                {
                    "ID": token_id,
                    "Form": form,
                    "Lemma": form.casefold(),
                    "Sentence ID": sentence_id,
                    "Document ID": str(document),
                }
            )

    sentence(1, 1, ["alpha", "beta", "alpha", "beta", "."])
    sentence(2, 1, ["alpha", ",", "beta", "gamma", "gamma", "gamma", "gamma", "."])
    sentence(3, 1, ["quiet", "text"])
    return pd.DataFrame(rows)


def _answer(tmp_path: Path, text: str = "alpha beta", **kwargs: Any) -> dict[str, Any]:
    return track_phrase(
        _table(),
        _corpus(tmp_path),
        PhraseQuery(text, position_bins=5),
        snapshot_id="snapshot-1",
        evidence=EvidenceRequest(**kwargs),
    )


def test_literal_phrase_keeps_occurrences_and_full_denominators(tmp_path: Path) -> None:
    answer = _answer(tmp_path)
    assert answer["summary"] == {
        "occurrences": 2,
        "word_tokens": 12,
        "occurrences_per_10000": pytest.approx(1666.6666667),
        "eligible_documents": 3,
        "matching_documents": 1,
        "document_prevalence_percent": pytest.approx(100 / 3),
        "dated_documents": 2,
        "undated_documents": 1,
    }
    assert [row["occurrences"] for row in answer["documents"]] == [2, 0, 0]
    assert answer["documents"][0]["document_id"] == "stable-a"
    assert answer["documents"][0]["document"] == "2020-05-02 First.txt"

    evidence = answer["evidence"]["rows"]
    assert [(row["token_start"], row["token_end"]) for row in evidence] == [(0, 2), (2, 4)]
    assert all(row["content_sha256"] == "hash-a" for row in evidence)
    assert [(row["character_start"], row["character_end"]) for row in evidence] == [(0, 10), (11, 21)]
    assert all(row["exact_source_highlight"] is True for row in evidence)
    assert answer["evidence"]["source_offsets"]["status"] == "verified"
    assert len({row["id"] for row in evidence}) == 2


def test_sentence_and_punctuation_boundaries_cannot_invent_a_match(tmp_path: Path) -> None:
    # alpha and beta are separated by a real comma in document 2. The exact
    # phrase does not skip it, while an explicitly punctuated phrase does.
    plain = _answer(tmp_path, "alpha beta")
    punctuated = _answer(tmp_path, "alpha, beta")
    assert plain["summary"]["occurrences"] == 2
    assert punctuated["summary"]["occurrences"] == 1
    assert punctuated["evidence"]["rows"][0]["match"] == "alpha, beta"


def test_overlapping_matches_count(tmp_path: Path) -> None:
    corpus = Corpus((Document(1, tmp_path / "x", "go go go", source_id="x"),), "f")
    table = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Form": ["go", "go", "go"],
            "Sentence ID": [1, 1, 1],
            "Document ID": [1, 1, 1],
        }
    )
    answer = track_phrase(table, corpus, PhraseQuery("go go"), snapshot_id="s")
    assert answer["summary"]["occurrences"] == 2
    assert [(row["token_start"], row["token_end"]) for row in answer["evidence"]["rows"]] == [(0, 2), (1, 3)]


def test_time_series_distinguishes_observed_zero_from_no_text(tmp_path: Path) -> None:
    answer = _answer(tmp_path)
    rows = {row["year"]: row for row in answer["time"]}
    assert rows[2020]["occurrences"] == 2
    assert rows[2021]["observed"] is False
    assert rows[2021]["occurrences"] is None
    assert rows[2021]["occurrences_per_10000"] is None
    assert rows[2022]["observed"] is True
    assert rows[2022]["occurrences"] == 0
    assert rows[2022]["occurrences_per_10000"] == 0


def test_position_profile_uses_each_documents_own_length(tmp_path: Path) -> None:
    answer = _answer(tmp_path, "gamma")
    assert sum(row["occurrences"] for row in answer["position"]) == 4
    assert sum(row["word_tokens"] for row in answer["position"]) == 12
    # Four gamma tokens occupy the latter half of the six-word second document,
    # rather than a position in one concatenated corpus.
    relative = [row["relative_position"] for row in answer["evidence"]["rows"]]
    assert relative == pytest.approx([2 / 6, 3 / 6, 4 / 6, 5 / 6])


def test_every_document_has_a_track_including_documents_without_matches(tmp_path: Path) -> None:
    """The track is the denominator's picture, not a list of hits."""
    answer = _answer(tmp_path, "alpha beta")
    tracks = answer["tracks"]
    assert [row["document_id"] for row in tracks] == ["stable-a", "stable-b", "stable-c"]
    assert [row["occurrences"] for row in tracks] == [2, 0, 0]
    assert [len(row["positions"]) for row in tracks] == [2, 0, 0]
    first = tracks[0]["positions"]
    # Four words, five bins: word 0 falls in bin 1, word 2 in bin 3.
    assert [(p["word_start"], p["position_bin"]) for p in first] == [(0, 1), (2, 3)]
    # Positions reuse the occurrence ids, so a reader can follow one from the
    # track into the evidence without a second join.
    assert [p["occurrence_id"] for p in first] == [row["id"] for row in answer["evidence"]["rows"]]


def test_position_filter_selects_by_start_position(tmp_path: Path) -> None:
    """Brushing bins 3-5 selects what those bins count, no more and no less."""
    answer = _answer(tmp_path, "gamma", position_start=0.4, position_end=0.8)
    # gamma sits at 2/6..5/6 of document 2; the range [0.4, 0.8) holds the
    # occurrences starting at 3/6 and 4/6, and excludes 2/6 and 5/6.
    assert answer["evidence"]["filtered_total"] == 2
    assert [row["word_start"] for row in answer["evidence"]["rows"]] == [3, 4]
    # The aggregates keep describing the whole corpus.
    assert answer["summary"]["occurrences"] == 4
    assert answer["evidence"]["filter"]["position_start"] == 0.4
    assert answer["evidence"]["filter"]["position_end"] == 0.8


def test_position_filter_and_bin_boundary_agree(tmp_path: Path) -> None:
    """An occurrence is counted once, in the bin its start chose; a brush over
    that bin selects exactly that occurrence -- including one whose phrase
    crosses into the range from before it."""
    words = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ]
    text = " ".join(words) + "."
    corpus = Corpus((Document(1, tmp_path / "x", text, source_id="x"),), "f")
    table = pd.DataFrame(
        {
            "ID": list(range(1, len(words) + 2)),
            "Form": [*words, "."],
            "Sentence ID": [1] * (len(words) + 1),
            "Document ID": [1] * (len(words) + 1),
        }
    )
    answer = track_phrase(
        table,
        corpus,
        PhraseQuery("nine ten", position_bins=5),
        snapshot_id="s",
        evidence=EvidenceRequest(position_start=0.4, position_end=0.6),
    )
    # "nine ten" starts at word 8 of 20: 0.4, the first word of bin 3. The
    # bin and the brush agree it belongs here, and only here.
    assert answer["summary"]["occurrences"] == 1
    assert answer["evidence"]["filtered_total"] == 1
    assert answer["evidence"]["rows"][0]["position_bin"] == 3
    pooled = track_phrase(table, corpus, PhraseQuery("nine ten", position_bins=5), snapshot_id="s")
    assert pooled["position"][2]["occurrences"] == 1
    assert pooled["position"][1]["occurrences"] == 0


def test_a_phrase_crossing_a_bin_edge_counts_once_in_its_start_bin(tmp_path: Path) -> None:
    words = [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ]
    text = " ".join(words) + "."
    corpus = Corpus((Document(1, tmp_path / "x", text, source_id="x"),), "f")
    table = pd.DataFrame(
        {
            "ID": list(range(1, len(words) + 2)),
            "Form": [*words, "."],
            "Sentence ID": [1] * (len(words) + 1),
            "Document ID": [1] * (len(words) + 1),
        }
    )
    # "seven eight nine" spans 0.30-0.45: it crosses bin 3's lower edge but
    # starts in bin 2, so it counts in bin 2 and brushing bin 3 misses it.
    answer = track_phrase(table, corpus, PhraseQuery("seven eight nine", position_bins=5), snapshot_id="s")
    row = answer["evidence"]["rows"][0]
    assert row["position_bin"] == 2
    assert answer["position"][1]["occurrences"] == 1
    assert answer["position"][2]["occurrences"] == 0
    brushed = track_phrase(
        table,
        corpus,
        PhraseQuery("seven eight nine", position_bins=5),
        snapshot_id="s",
        evidence=EvidenceRequest(position_start=0.4, position_end=0.6),
    )
    assert brushed["evidence"]["filtered_total"] == 0


def test_an_empty_position_range_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="end after its start"):
        _answer(tmp_path, position_start=0.7, position_end=0.2)
    with pytest.raises(ValueError, match=r"0\.0 to 1\.0"):
        _answer(tmp_path, position_start=1.4)


def test_source_passage_returns_the_exact_window(tmp_path: Path) -> None:
    corpus = Corpus(
        (
            Document(
                1,
                tmp_path / "w.txt",
                "The opening words. " + "filler " * 60 + "TARGET phrase here. tail",
                source_id="w",
            ),
        ),
        "f",
    )
    start = corpus.docs[0].text.index("TARGET")
    read = source_passage(corpus, "w", start, start + 6, radius=40)
    assert read["text"].endswith("TARGET phrase here. tail")
    assert "TARGET" in read["text"]
    assert read["text"].endswith("here. tail")
    assert read["document_id"] == "w"
    assert read["character_offset_unit"] == "unicode_code_point"
    # The window bounds bracket the marked span, and the marked span is marked
    # against them: reading it back out must reproduce the phrase itself.
    marked = corpus.docs[0].text[start : start + 6]
    assert marked in read["text"]
    assert read["character_start"] < start
    assert read["character_end"] > start + 6


def test_source_passage_refuses_unknown_documents_and_bad_ranges(tmp_path: Path) -> None:
    corpus = Corpus((Document(1, tmp_path / "w.txt", "some text", source_id="w"),), "f")
    with pytest.raises(ValueError, match="not part of this answer"):
        source_passage(corpus, "elsewhere", 0)
    with pytest.raises(ValueError, match="cannot end before it starts"):
        source_passage(corpus, "w", 5, 2)
    with pytest.raises(ValueError, match="up to"):
        source_passage(corpus, "w", 0, radius=-1)
    with pytest.raises(ValueError, match="not part of this answer"):
        source_passage(corpus, "w", 10_000)


def test_evidence_pages_and_filters_do_not_change_global_aggregates(tmp_path: Path) -> None:
    page = _answer(tmp_path, offset=1, limit=1, year=2020)
    assert page["summary"]["occurrences"] == 2
    assert page["evidence"]["total"] == 2
    assert page["evidence"]["filtered_total"] == 2
    assert page["evidence"]["offset"] == 1
    assert len(page["evidence"]["rows"]) == 1

    absent = _answer(tmp_path, document_id="stable-b")
    assert absent["summary"]["occurrences"] == 2
    assert absent["evidence"]["filtered_total"] == 0


def test_case_policy_and_query_interpretation_are_explicit(tmp_path: Path) -> None:
    folded = track_phrase(
        _table(),
        _corpus(tmp_path),
        PhraseQuery("ALPHA BETA", position_bins=5),
        snapshot_id="s",
    )
    exact = track_phrase(
        _table(),
        _corpus(tmp_path),
        PhraseQuery("ALPHA BETA", case_sensitive=True, position_bins=5),
        snapshot_id="s",
    )
    assert folded["summary"]["occurrences"] == 2
    assert exact["summary"]["occurrences"] == 0
    assert folded["question"]["subject"]["tokens"] == ["ALPHA", "BETA"]
    assert folded["question"]["subject"]["boundary"] == "sentence"


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_questions_are_refused(text: str) -> None:
    with pytest.raises(ValueError, match="Enter a phrase"):
        PhraseQuery(text).validate()


def test_invalid_evidence_paging_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 1 and 250"):
        _answer(tmp_path, limit=251)


def test_browser_offsets_account_for_non_bmp_unicode(tmp_path: Path) -> None:
    text = "😀 alpha beta"
    corpus = Corpus((Document(1, tmp_path / "emoji.txt", text, source_id="emoji"),), "unicode")
    table = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Form": ["😀", "alpha", "beta"],
            "Sentence ID": [1, 1, 1],
            "Document ID": [1, 1, 1],
        }
    )
    answer = track_phrase(table, corpus, PhraseQuery("alpha beta"), snapshot_id="unicode")
    occurrence = answer["evidence"]["rows"][0]
    assert occurrence["character_start"] == 2
    assert occurrence["character_end"] == 12
    assert occurrence["browser_character_start"] == 3
    assert occurrence["browser_character_end"] == 13
    assert occurrence["match_source"] == "alpha beta"


def test_a_failed_alignment_never_becomes_an_approximate_exact_span(tmp_path: Path) -> None:
    corpus = Corpus((Document(1, tmp_path / "quotes.txt", "“alpha beta”", source_id="quotes"),), "quotes")
    table = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            # Simulate a parser normalising curly quotes to straight quotes.
            "Form": ['"', "alpha", "beta", '"'],
            "Sentence ID": [1, 1, 1, 1],
            "Document ID": [1, 1, 1, 1],
        }
    )
    answer = track_phrase(table, corpus, PhraseQuery("alpha beta"), snapshot_id="quotes")
    occurrence = answer["evidence"]["rows"][0]
    assert occurrence["exact_source_highlight"] is False
    assert occurrence["character_start"] is None
    assert occurrence["match"] == "alpha beta"
    assert answer["evidence"]["source_offsets"]["status"] == "unavailable"


def test_snapshot_identity_survives_moving_an_unchanged_corpus(tmp_path: Path) -> None:
    first = (
        Document(1, tmp_path / "original" / "a.txt", "alpha", sha256="hash-a"),
        Document(2, tmp_path / "original" / "nested" / "b.txt", "beta", sha256="hash-b"),
    )
    moved = (
        Document(1, tmp_path / "restored" / "a.txt", "alpha", sha256="hash-a"),
        Document(2, tmp_path / "restored" / "nested" / "b.txt", "beta", sha256="hash-b"),
    )
    assert corpus_fingerprint(first) == corpus_fingerprint(moved)


def test_source_coverage_is_reported_for_the_evidence_on_screen(tmp_path: Path) -> None:
    """A warning about highlights has to be about the highlights being shown.

    The status covered every occurrence in the corpus, so filtering down to a
    document whose spans are all verified still reported "partial" -- read as
    a doubt about the passages on screen, when the doubt was about others.
    """
    corpus = _corpus(tmp_path)
    table = _table()
    answer = track_phrase(
        table,
        corpus,
        PhraseQuery("alpha beta"),
        snapshot_id="coverage",
        evidence=EvidenceRequest(document_id="stable-a"),
    )
    offsets = answer["evidence"]["source_offsets"]
    assert offsets["filtered_total"] == answer["evidence"]["filtered_total"]
    assert offsets["filtered_verified"] <= offsets["verified"]
    assert offsets["filtered_status"] in {"verified", "partial", "unavailable"}
    # And the corpus-wide figures are still there, unchanged in meaning.
    assert offsets["total"] == answer["summary"]["occurrences"]
