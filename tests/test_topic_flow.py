"""Cutting speeches into segments for topic flow: which rule, and proved positions.

Two properties matter most. **The rule that cut a document is reported**,
because the figure changes with it -- the State of the Union files have no
blank lines, so a blank-line splitter alone would call every speech one
paragraph and draw a flat bar. **A token is placed in a paragraph only when
its position in the text is proved**; a document whose tokens cannot be
aligned falls back to sentence windows and is named, rather than being
mis-paragraphed by a guess.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.analysis.topic_flow import SEGMENT_RULES, paragraph_spans, plan_segments
from core.io.reader import Corpus, Document


def corpus_of(*docs: tuple[int, str]) -> Corpus:
    return Corpus(docs=tuple(Document(doc_id=i, path=Path(f"d{i}.txt"), text=t) for i, t in docs), sha256="test")


def rows_for(doc_id: int, name: str, sentences: list[list[str]]) -> pd.DataFrame:
    """Parser-shaped rows: one per token, sentences numbered from 1."""
    records = [
        {"Form": form, "Lemma": form.lower(), "Sentence ID": number, "Document ID": doc_id, "Document": name}
        for number, sentence in enumerate(sentences, start=1)
        for form in sentence
    ]
    return pd.DataFrame(records)


class TestParagraphSpans:
    def test_blank_lines_win_when_present(self) -> None:
        rule, spans = paragraph_spans("One.\nstill one.\n\nTwo.")
        assert rule == "blank-line"
        assert len(spans) == 2, "a single newline inside a blank-line paragraph does not split it"

    def test_single_lines_are_paragraphs_when_there_are_no_blank_lines(self) -> None:
        """The State of the Union files: one paragraph per line, no blank lines."""
        rule, spans = paragraph_spans("First.\nSecond.\nThird.")
        assert rule == "line"
        assert len(spans) == 3

    def test_a_single_block_falls_back_to_sentences(self) -> None:
        assert paragraph_spans("No breaks at all in this one.") == ("sentences", [])

    def test_spans_cover_content_not_the_whitespace_around_it(self) -> None:
        text = "  First.  \n\n\n   Second.\n\n"
        _, spans = paragraph_spans(text)
        assert [text[a:b] for a, b in spans] == ["First.", "Second."]

    def test_empty_paragraphs_are_not_segments(self) -> None:
        _, spans = paragraph_spans("A.\n\n   \n\nB.")
        assert len(spans) == 2

    def test_the_rules_are_the_declared_vocabulary(self) -> None:
        for text in ("a\n\nb", "a\nb", "a"):
            assert paragraph_spans(text)[0] in SEGMENT_RULES


class TestPlanSegments:
    def test_each_row_gets_the_paragraph_its_token_sits_in(self) -> None:
        text = "War came .\nPeace followed ."
        frame = rows_for(1, "a.txt", [["War", "came", "."], ["Peace", "followed", "."]])
        plan = plan_segments(frame, corpus_of((1, text)))
        assert plan.rules == {"a.txt": "line"}
        assert plan.labels["a.txt"].tolist() == [1, 1, 1, 2, 2, 2]
        assert plan.unaligned == ()

    def test_spans_read_back_as_the_paragraph_text(self) -> None:
        text = "War came .\nPeace followed ."
        frame = rows_for(1, "a.txt", [["War", "came", "."], ["Peace", "followed", "."]])
        plan = plan_segments(frame, corpus_of((1, text)))
        assert [text[a:b] for a, b in plan.spans["a.txt"]] == ["War came .", "Peace followed ."]

    def test_a_whitespace_token_belongs_to_the_paragraph_before_it(self) -> None:
        """spaCy emits the newline itself as a token; it must not open the
        next paragraph early or land in paragraph zero."""
        text = "War came\nPeace followed"
        frame = rows_for(1, "a.txt", [["War", "came", "\n", "Peace", "followed"]])
        plan = plan_segments(frame, corpus_of((1, text)))
        assert plan.labels["a.txt"].tolist() == [1, 1, 1, 2, 2]

    def test_an_unalignable_document_falls_back_and_is_named(self) -> None:
        """Tokens that do not match the text are not placed by guesswork."""
        text = "Something else entirely .\nAnother line ."
        frame = rows_for(1, "a.txt", [["War", "came"], ["Peace", "followed"]])
        plan = plan_segments(frame, corpus_of((1, text)), window=1)
        assert plan.unaligned == ("a.txt",)
        assert plan.rules["a.txt"] == "sentences"
        assert plan.labels["a.txt"].tolist() == [1, 1, 2, 2]
        assert plan.spans["a.txt"] == [None, None], "an unproved span is absent, not invented"

    def test_sentence_windows_group_consecutive_sentences(self) -> None:
        text = "A b . C d . E f . G h ."
        frame = rows_for(1, "a.txt", [["A", "b", "."], ["C", "d", "."], ["E", "f", "."], ["G", "h", "."]])
        plan = plan_segments(frame, corpus_of((1, text)), window=2)
        assert plan.rules["a.txt"] == "sentences"
        assert plan.labels["a.txt"].tolist() == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        first, second = plan.spans["a.txt"]
        assert first is not None and second is not None
        assert text[first[0] : first[1]] == "A b . C d ."
        assert text[second[0] : second[1]] == "E f . G h ."

    def test_labels_are_indexed_like_the_frame(self) -> None:
        """A caller joins labels back onto kept tokens by row index."""
        frame = rows_for(1, "a.txt", [["War", "came", "."], ["Peace", "followed", "."]])
        frame.index = frame.index + 100
        plan = plan_segments(frame, corpus_of((1, "War came .\nPeace followed .")))
        assert list(plan.labels["a.txt"].index) == list(frame.index)

    def test_a_document_the_table_never_parsed_is_skipped(self) -> None:
        frame = rows_for(1, "a.txt", [["War", "came"]])
        plan = plan_segments(frame, corpus_of((1, "War came"), (2, "Never parsed")))
        assert set(plan.rules) == {"a.txt"}

    def test_several_documents_are_cut_independently(self) -> None:
        frame = pd.concat(
            [
                rows_for(1, "a.txt", [["War", "came", "."], ["Peace", "followed", "."]]),
                rows_for(2, "b.txt", [["Only", "one", "block", "."]]),
            ],
            ignore_index=True,
        )
        plan = plan_segments(frame, corpus_of((1, "War came .\nPeace followed ."), (2, "Only one block .")))
        assert plan.rules == {"a.txt": "line", "b.txt": "sentences"}

    def test_a_window_must_hold_a_sentence(self) -> None:
        with pytest.raises(ValueError, match="window"):
            plan_segments(rows_for(1, "a.txt", [["A"]]), corpus_of((1, "A")), window=0)
