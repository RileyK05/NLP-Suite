"""A long text's own sections (docs/internal/PLAN_0.5.0.md 2.3, 2.9).

The fixtures in ``tests/fixtures/books/`` are the real opening of *Alice's
Adventures in Wonderland* with its Project Gutenberg header, contents and
footer, and three small synthetic books: chapters grouped in books, roman
numerals alone on their lines, and no headings at all. Every split must tile
the kept text exactly -- nothing lost, nothing repeated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.file_ops.sections import SectionPlan, detect_sections, front_and_back_matter, roman_value

BOOKS = Path(__file__).parent / "fixtures" / "books"


def _read(name: str) -> str:
    return (BOOKS / name).read_text(encoding="utf-8")


def _tiles(text: str, plan: SectionPlan) -> bool:
    """Joined, the sections are exactly the text from the first one's start to the body's end."""
    for before, after in zip(plan.sections, plan.sections[1:], strict=False):
        if before.end != after.start:
            return False
    joined = "".join(text[section.start : section.end] for section in plan.sections)
    return joined == text[plan.sections[0].start : plan.body_end]


@pytest.mark.parametrize("name", sorted(path.name for path in BOOKS.glob("*.txt")))
def test_every_fixture_tiles_its_text(name: str) -> None:
    text = _read(name)
    plan = detect_sections(text).unwrap()
    assert _tiles(text, plan)
    assert all(section.words == len(text[section.start : section.end].split()) for section in plan.sections)


class TestAlice:
    """The real book: licence, contents, then CHAPTER I. with its title on the next line."""

    def test_three_chapters_with_their_titles(self) -> None:
        plan = detect_sections(_read("alice_first3.txt")).unwrap()
        assert plan.rule == "headings"
        assert [section.title for section in plan.sections] == [
            "Chapter I: Down the Rabbit-Hole",
            "Chapter II: The Pool of Tears",
            "Chapter III: A Caucus-Race and a Long Tale",
        ]
        assert [section.number for section in plan.sections] == [1, 2, 3]
        assert plan.sections[1].name == "The Pool of Tears"

    def test_the_licence_and_the_contents_are_left_out_and_said(self) -> None:
        text = _read("alice_first3.txt")
        plan = detect_sections(text).unwrap()
        left = dict(plan.left_out)
        assert left["Project Gutenberg header"] > 0 and left["Project Gutenberg licence at the end"] > 0
        # The contents list every chapter; left in, chapter I would be a two-word section.
        assert left["table of contents lines"] == 12
        assert text[plan.sections[0].start :].startswith("CHAPTER I.\nDown the Rabbit-Hole")
        assert "PROJECT GUTENBERG" not in text[plan.sections[0].start : plan.body_end]

    def test_the_licence_can_be_kept(self) -> None:
        text = _read("alice_first3.txt")
        plan = detect_sections(text, keep_front_matter=True).unwrap()
        assert (plan.body_start, plan.body_end) == (0, len(text))
        assert "PROJECT GUTENBERG" in text[plan.sections[-1].start : plan.sections[-1].end]


def test_books_group_the_chapters_after_them() -> None:
    plan = detect_sections(_read("two_books.txt")).unwrap()
    assert [section.parents for section in plan.sections] == [(("Book", "I"),)] * 4 + [(("Book", "II"),)] * 2
    # A book's own heading opens its first chapter rather than being lost.
    text = _read("two_books.txt")
    assert text[plan.sections[4].start :].lstrip().startswith("BOOK II")
    assert plan.sections[0].title == "Chapter 1: The First Letter"


def test_roman_numerals_alone_on_a_line_are_chapters() -> None:
    plan = detect_sections(_read("roman_only.txt")).unwrap()
    assert [section.number for section in plan.sections] == [1, 2, 3, 4, 5, 6]


def test_no_headings_falls_back_to_blocks_that_say_what_they_are() -> None:
    result = detect_sections(_read("no_headings.txt"))
    plan = result.unwrap()
    assert plan.rule == "words"
    assert plan.sections[0].title == f"Words 1{chr(0x2013)}2,000"
    assert "SECTIONS_NOT_CHAPTERS" in [d.code for d in result.diagnostics]


def test_a_heading_inside_a_sentence_is_not_a_heading() -> None:
    body = "\n\n".join(
        f"CHAPTER {n}\n\n" + ("She thought of chapter 3 of her life and of part two of the plan. " * 60)
        for n in (1, 2, 3)
    )
    plan = detect_sections(body).unwrap()
    assert len(plan.sections) == 3


def test_a_pattern_that_matches_nothing_falls_back_and_says_so() -> None:
    result = detect_sections(_read("roman_only.txt"), rule="pattern", pattern="Canto")
    assert result.unwrap().rule == "words"
    assert "SECTIONS_NO_HEADINGS" in [d.code for d in result.diagnostics]


def test_a_literal_pattern_and_a_regex() -> None:
    text = "".join(f"Letter {n}\n\n" + ("word " * 400) + "\n\n" for n in range(1, 5))
    assert len(detect_sections(text, rule="pattern", pattern="letter").unwrap().sections) == 4
    assert len(detect_sections(text, rule="pattern", pattern=r"Letter \d+$", regex=True).unwrap().sections) == 4
    assert detect_sections(text, rule="pattern", pattern="(", regex=True).diagnostics[0].code == "SECTIONS_BAD_REGEX"


def test_blank_gaps_cut_when_asked() -> None:
    text = "\n\n\n\n\n".join("word " * 400 for _ in range(4))
    plan = detect_sections(text, rule="blank-gap").unwrap()
    assert plan.rule == "blank-gap" and len(plan.sections) == 4
    assert _tiles(text, plan)


def test_roman_numerals_are_read_strictly() -> None:
    assert [roman_value(v) for v in ("I", "iv", "XLIX", "LXI", "MCMXCIV")] == [1, 4, 49, 61, 1994]
    assert roman_value("IIII") is None and roman_value("mix") == 1009 and roman_value("hello") is None


def test_no_licence_means_the_whole_text_is_the_body() -> None:
    assert front_and_back_matter("plain text") == (0, len("plain text"))


def test_bad_input_is_refused_plainly() -> None:
    assert detect_sections("   ").diagnostics[0].code == "SECTIONS_EMPTY"
    assert detect_sections("text", rule="chapters").diagnostics[0].code == "SECTIONS_BAD_RULE"
    assert detect_sections("text", rule="pattern").diagnostics[0].code == "SECTIONS_NO_PATTERN"
