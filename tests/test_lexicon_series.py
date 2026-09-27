"""Dictionary-based content analysis: word groups counted along an axis.

The method these tests pin is the one the research scripts in ``scripts/``
build by hand every time — name a category, list its words, count them per
document, divide by how much was said, and look at it over time. The details
that make it trustworthy are the ones worth testing: a phrase has to match as
a phrase, a rate has to divide by the right denominator, and a year where a
category never appears has to be a zero rather than a missing row.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from core.analysis.lexicon_series import BY_CHOICES, facet_labels, lexicon_series, parse_lexicon
from core.conll.schema import Col


def frame_of(sentences: list[tuple[str, int, list[str]]]) -> pd.DataFrame:
    """A minimal canonical CoNLL frame from (document, sentence id, tokens)."""
    rows = []
    for document, sentence_id, tokens in sentences:
        for position, token in enumerate(tokens, start=1):
            rows.append(
                {
                    Col.ID.value: position,
                    Col.FORM.value: token,
                    Col.LEMMA.value: token,
                    Col.POS.value: "X",
                    Col.NER.value: "",
                    Col.HEAD.value: 0,
                    Col.DEPREL.value: "",
                    Col.DEPS.value: "",
                    Col.CLAUSE_TAG.value: "",
                    Col.RECORD_ID.value: 0,
                    Col.SENTENCE_ID.value: sentence_id,
                    Col.DOCUMENT_ID.value: 1,
                    Col.DOCUMENT.value: document,
                }
            )
    return pd.DataFrame(rows)


def rate(table: pd.DataFrame, facet: str, category: str, column: str = "Per 1000") -> float:
    row = table[(table["Facet"] == facet) & (table["Category"] == category)]
    assert len(row) == 1, f"expected one row for {facet}/{category}, got {len(row)}"
    return float(row.iloc[0][column])


# ------------------------------------------------------------- the lexicon --


class TestReadingWordGroups:
    def test_names_its_groups(self) -> None:
        got = parse_lexicon("Iraq: iraq, saddam; Vietnam: vietnam, hanoi").unwrap()
        assert got == {"Iraq": ("iraq", "saddam"), "Vietnam": ("vietnam", "hanoi")}

    def test_keeps_phrases_whole(self) -> None:
        # "civil rights" is one term. Splitting it on the space would count
        # every mention of "rights" as a mention of civil rights.
        assert parse_lexicon("Civil rights: civil rights, voting rights").unwrap() == {
            "Civil rights": ("civil rights", "voting rights")
        }

    def test_accepts_newlines_as_well_as_semicolons(self) -> None:
        # Semicolons so it fits one line in a form; newlines so a list pasted
        # in from a file does not have to be reformatted first.
        assert parse_lexicon("A: one\nB: two").unwrap() == {"A": ("one",), "B": ("two",)}

    def test_lowers_once_here_rather_than_at_every_comparison(self) -> None:
        assert parse_lexicon("A: Iraq, SADDAM").unwrap() == {"A": ("iraq", "saddam")}

    def test_drops_a_word_repeated_in_one_group(self) -> None:
        # Counting it twice would silently double that group's rate.
        assert parse_lexicon("A: iraq, iraq").unwrap() == {"A": ("iraq",)}

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            ("iraq, saddam", "LEXICON_NO_CATEGORY"),
            (": iraq", "LEXICON_EMPTY_NAME"),
            ("Iraq:", "LEXICON_EMPTY_GROUP"),
            ("Iraq: a; Iraq: b", "LEXICON_DUPLICATE"),
            ("   ", "LEXICON_EMPTY"),
        ],
    )
    def test_refuses_what_it_cannot_read(self, text: str, code: str) -> None:
        result = parse_lexicon(text)
        assert result.value is None
        assert [d.code for d in result.diagnostics] == [code]

    def test_the_error_shows_how_to_write_it(self) -> None:
        # Someone typing a bare word list is one colon away from correct, and
        # the message should say so rather than name a rule.
        message = parse_lexicon("iraq, saddam").diagnostics[0].message
        assert "Name: word, word" in message


# --------------------------------------------------------------- the facet --


class TestPlacingDocumentsOnAnAxis:
    def test_by_year(self) -> None:
        got = facet_labels([("a.txt", date(1934, 1, 3)), ("b.txt", date(2003, 1, 28))], "year").unwrap()
        assert got == {"a.txt": "1934", "b.txt": "2003"}

    def test_by_decade(self) -> None:
        got = facet_labels([("a.txt", date(1934, 1, 3)), ("b.txt", date(1939, 1, 4))], "decade").unwrap()
        assert got == {"a.txt": "1930s", "b.txt": "1930s"}

    def test_by_document(self) -> None:
        assert facet_labels([("a.txt", None)], "document").unwrap() == {"a.txt": "a.txt"}

    def test_by_pattern_names_the_axis_from_its_first_group(self) -> None:
        # How a speaker becomes an axis without the engine knowing anything
        # about how this corpus happens to name its files.
        got = facet_labels(
            [("1934-01-03_franklin d roosevelt_sotu.txt", None), ("2003-01-28_george w bush_sotu.txt", None)],
            "pattern",
            group_pattern=r"_([a-z. ]+)_sotu",
        ).unwrap()
        assert got == {
            "1934-01-03_franklin d roosevelt_sotu.txt": "franklin d roosevelt",
            "2003-01-28_george w bush_sotu.txt": "george w bush",
        }

    def test_by_pattern_falls_back_to_the_whole_match(self) -> None:
        got = facet_labels([("report_2019.txt", None)], "pattern", group_pattern=r"\d{4}").unwrap()
        assert got == {"report_2019.txt": "2019"}

    def test_a_document_with_no_date_is_left_out_and_said_so(self) -> None:
        # Never a fallback label. A speech filed under "unknown" sits in the
        # chart looking like evidence.
        result = facet_labels([("a.txt", date(1934, 1, 3)), ("b.txt", None)], "year")
        assert result.unwrap() == {"a.txt": "1934"}
        assert [d.code for d in result.diagnostics] == ["LEXICON_UNDATED"]
        assert "b.txt" in result.diagnostics[0].message

    def test_a_name_the_pattern_misses_is_left_out_and_said_so(self) -> None:
        result = facet_labels([("a.txt", None), ("b.txt", None)], "pattern", group_pattern="^a")
        assert result.unwrap() == {"a.txt": "a"}
        assert [d.code for d in result.diagnostics] == ["LEXICON_UNMATCHED"]

    def test_placing_nothing_is_a_failure_not_an_empty_answer(self) -> None:
        result = facet_labels([("a.txt", None)], "year")
        assert result.value is None
        assert "LEXICON_NO_FACET" in [d.code for d in result.diagnostics]

    @pytest.mark.parametrize(
        ("by", "pattern", "code"),
        [
            ("speaker", "", "LEXICON_BAD_BY"),
            ("pattern", "", "LEXICON_NO_PATTERN"),
            ("pattern", "([", "LEXICON_BAD_REGEX"),
        ],
    )
    def test_refuses_an_axis_it_cannot_build(self, by: str, pattern: str, code: str) -> None:
        result = facet_labels([("a.txt", date(1934, 1, 3))], by, group_pattern=pattern)
        assert result.value is None
        assert code in [d.code for d in result.diagnostics]

    def test_every_choice_the_registry_offers_is_one_this_understands(self) -> None:
        # The registry spells the choices out rather than importing them, so
        # that it stays data and imports no tool module. This is the join.
        from core.profiler.registry import get_tool

        spec = get_tool("lexicon_series")
        offered = next(p for p in spec.params if p.name == "by").choices
        assert tuple(offered) == BY_CHOICES

    def test_by_order_is_each_chapter(self) -> None:
        got = facet_labels(
            [("ch1.txt", None), ("ch2.txt", None)],
            "order",
            positions={"ch1.txt": 1.0, "ch2.txt": 2.0},
            noun="Chapter",
        ).unwrap()
        assert got == {"ch1.txt": "Chapter 1", "ch2.txt": "Chapter 2"}

    def test_by_period_is_blocks_of_the_axis(self) -> None:
        from core.corpus_axis import DASH

        positions = {f"ch{i}.txt": float(i) for i in range(1, 9)}
        got = facet_labels([(name, None) for name in positions], "period", positions=positions, noun="Chapter").unwrap()
        assert set(got.values()) == {
            f"Chapters 1{DASH}2",
            f"Chapters 3{DASH}4",
            f"Chapters 5{DASH}6",
            f"Chapters 7{DASH}8",
        }

    def test_by_a_detail_groups_by_its_values(self) -> None:
        got = facet_labels(
            [("a.txt", None), ("b.txt", None), ("c.txt", None)],
            "field:Party",
            details={"a.txt": {"Party": "Democratic"}, "b.txt": {"Party": "Republican"}, "c.txt": {}},
        ).unwrap()
        assert got == {"a.txt": "Democratic", "b.txt": "Republican"}

    def test_an_order_axis_without_orders_is_refused(self) -> None:
        result = facet_labels([("a.txt", None)], "order")
        assert result.value is None
        assert "LEXICON_NO_ORDER" in [d.code for d in result.diagnostics]

    def test_a_detail_axis_needs_details(self) -> None:
        result = facet_labels([("a.txt", None)], "field:Party")
        assert result.value is None
        assert "LEXICON_NO_DETAILS" in [d.code for d in result.diagnostics]

    def test_the_desktop_offers_the_same_prefix_for_a_detail_value(self) -> None:
        """The value spelling "field:Party" crosses the language boundary."""
        from pathlib import Path

        from core.io.document_fields import FIELD_VALUE_PREFIX

        source = (Path(__file__).resolve().parents[1] / "desktop" / "src" / "ParamFields.tsx").read_text(
            encoding="utf-8"
        )
        assert f'FIELD_VALUE_PREFIX = "{FIELD_VALUE_PREFIX}"' in source


# --------------------------------------------------------------- the count --

SPEECHES = [
    ("1934.txt", 1, ["we", "must", "defend", "civil", "rights"]),
    ("1934.txt", 2, ["poverty", "is", "the", "enemy"]),
    ("2003.txt", 1, ["iraq", "and", "saddam", "threaten", "us"]),
    ("2003.txt", 2, ["iraq", "again"]),
    ("2003.txt", 3, ["nothing", "relevant", "here"]),
]
LABELS = {"1934.txt": "1934", "2003.txt": "2003"}


class TestCountingGroupsAlongAnAxis:
    def test_counts_every_occurrence_not_every_sentence(self) -> None:
        table = lexicon_series(frame_of(SPEECHES), parse_lexicon("Iraq: iraq, saddam").unwrap(), LABELS).unwrap()
        # iraq, saddam, iraq = three occurrences across two sentences.
        assert rate(table, "2003", "Iraq", "Occurrences") == 3
        assert rate(table, "2003", "Iraq", "Matching Sentences") == 2

    def test_a_phrase_matches_as_a_phrase(self) -> None:
        table = lexicon_series(
            frame_of(SPEECHES), parse_lexicon("Civil rights: civil rights").unwrap(), LABELS
        ).unwrap()
        assert rate(table, "1934", "Civil rights", "Occurrences") == 1

    def test_a_phrase_does_not_match_its_words_apart(self) -> None:
        apart = [("a.txt", 1, ["civil", "war", "and", "human", "rights"])]
        table = lexicon_series(
            frame_of(apart), parse_lexicon("Civil rights: civil rights").unwrap(), {"a.txt": "a"}
        ).unwrap()
        assert rate(table, "a", "Civil rights", "Occurrences") == 0

    def test_a_phrase_survives_the_hyphen_its_corpus_spells_it_with(self) -> None:
        # Kennedy-era speeches write "Viet-Nam", which the parser splits into
        # three tokens. Without this the tool reported near-silence on the
        # subject of the decade, and the silence looked like a finding.
        hyphenated = [("a.txt", 1, ["we", "leave", "viet", "-", "nam", "now"])]
        table = lexicon_series(
            frame_of(hyphenated), parse_lexicon("Vietnam: viet nam").unwrap(), {"a.txt": "a"}
        ).unwrap()
        assert rate(table, "a", "Vietnam", "Occurrences") == 1

    def test_a_phrase_does_not_leap_over_a_word(self) -> None:
        # Only separators are skipped. Skipping any token would let "civil"
        # and "rights" three words apart count as "civil rights".
        apart = [("a.txt", 1, ["civil", "and", "human", "rights"])]
        table = lexicon_series(
            frame_of(apart), parse_lexicon("Civil rights: civil rights").unwrap(), {"a.txt": "a"}
        ).unwrap()
        assert rate(table, "a", "Civil rights", "Occurrences") == 0

    def test_a_phrase_does_not_run_across_a_sentence_boundary(self) -> None:
        # "civil" ending one sentence and "rights" opening the next is two
        # subjects, not one phrase.
        split = [("a.txt", 1, ["all", "civil"]), ("a.txt", 2, ["rights", "matter"])]
        table = lexicon_series(
            frame_of(split), parse_lexicon("Civil rights: civil rights").unwrap(), {"a.txt": "a"}
        ).unwrap()
        assert rate(table, "a", "Civil rights", "Occurrences") == 0

    def test_the_rate_divides_by_the_tokens_it_counted(self) -> None:
        table = lexicon_series(frame_of(SPEECHES), parse_lexicon("Iraq: iraq, saddam").unwrap(), LABELS).unwrap()
        # 2003 has 5 + 2 + 3 = 10 tokens and 3 occurrences.
        assert rate(table, "2003", "Iraq", "Tokens") == 10
        assert rate(table, "2003", "Iraq") == pytest.approx(300.0)

    def test_the_share_of_sentences_is_of_sentences_not_tokens(self) -> None:
        table = lexicon_series(frame_of(SPEECHES), parse_lexicon("Iraq: iraq, saddam").unwrap(), LABELS).unwrap()
        # Two of the three 2003 sentences mention it.
        assert rate(table, "2003", "Iraq", "Sentence Percent") == pytest.approx(200 / 3)

    def test_a_group_absent_from_a_year_is_a_zero_not_a_missing_row(self) -> None:
        # A gap in a series has to read as "they stopped saying it", not as
        # "we have no data for that year".
        table = lexicon_series(
            frame_of(SPEECHES), parse_lexicon("Iraq: iraq; Poverty: poverty").unwrap(), LABELS
        ).unwrap()
        assert len(table) == 4
        assert rate(table, "1934", "Iraq", "Occurrences") == 0
        assert rate(table, "2003", "Poverty", "Occurrences") == 0

    def test_the_same_question_always_writes_the_same_rows(self) -> None:
        groups = parse_lexicon("B: iraq; A: poverty").unwrap()
        first = lexicon_series(frame_of(SPEECHES), groups, LABELS).unwrap()
        shuffled = frame_of(list(reversed(SPEECHES)))
        second = lexicon_series(shuffled, groups, LABELS).unwrap()
        pd.testing.assert_frame_equal(first, second)

    def test_says_when_a_group_never_appears(self) -> None:
        # Almost always a field mistake or a typo, and silence looks like a
        # finding: "this president never mentioned it."
        result = lexicon_series(frame_of(SPEECHES), parse_lexicon("Space: apollo").unwrap(), LABELS)
        assert "LEXICON_NEVER_FOUND" in [d.code for d in result.diagnostics]
        assert "lemma" in result.diagnostics[0].message

    def test_counts_the_field_it_was_asked_for(self) -> None:
        rows = frame_of([("a.txt", 1, ["x"])])
        rows.loc[0, Col.FORM.value] = "immigrants"
        rows.loc[0, Col.LEMMA.value] = "immigrant"
        groups = parse_lexicon("Immigration: immigrants").unwrap()
        by_form = lexicon_series(rows, groups, {"a.txt": "a"}, field=Col.FORM).unwrap()
        by_lemma = lexicon_series(rows, groups, {"a.txt": "a"}, field=Col.LEMMA).unwrap()
        assert rate(by_form, "a", "Immigration", "Occurrences") == 1
        assert rate(by_lemma, "a", "Immigration", "Occurrences") == 0

    def test_a_document_off_the_axis_is_not_counted(self) -> None:
        table = lexicon_series(frame_of(SPEECHES), parse_lexicon("Iraq: iraq").unwrap(), {"2003.txt": "2003"}).unwrap()
        assert set(table["Facet"]) == {"2003"}
        assert rate(table, "2003", "Iraq", "Tokens") == 10

    @pytest.mark.parametrize(
        ("kwargs", "code"),
        [
            ({"field": Col.POS}, "LEXICON_BAD_FIELD"),
        ],
    )
    def test_refuses_a_field_it_cannot_count(self, kwargs: dict[str, object], code: str) -> None:
        result = lexicon_series(frame_of(SPEECHES), parse_lexicon("A: iraq").unwrap(), LABELS, **kwargs)  # type: ignore[arg-type]
        assert result.value is None
        assert code in [d.code for d in result.diagnostics]

    def test_an_empty_frame_is_a_diagnostic(self) -> None:
        empty = frame_of([]).reindex(columns=frame_of([("a.txt", 1, ["x"])]).columns)
        result = lexicon_series(empty, parse_lexicon("A: iraq").unwrap(), LABELS)
        assert result.value is None


CONTEXT = [
    ("2003.txt", 1, ["iraq", "has", "nuclear", "weapons"]),
    ("2003.txt", 2, ["iraq", "needs", "democracy"]),
    ("2003.txt", 3, ["the", "economy", "has", "weapons", "of", "growth"]),
]


class TestTheConditionalQuestion:
    """``within`` asks what vocabulary travelled with a topic."""

    def test_counts_only_inside_the_sentences_that_match(self) -> None:
        # "weapons" appears twice, but only once in a sentence about Iraq.
        table = lexicon_series(
            frame_of(CONTEXT),
            parse_lexicon("Weapons: weapons").unwrap(),
            {"2003.txt": "2003"},
            within=parse_lexicon("Iraq: iraq").unwrap(),
        ).unwrap()
        assert rate(table, "2003", "Weapons", "Occurrences") == 1

    def test_the_denominator_is_the_matching_sentences_too(self) -> None:
        # Otherwise the rate would be "per 1,000 words of the whole speech",
        # which is not a rate within the topic at all.
        table = lexicon_series(
            frame_of(CONTEXT),
            parse_lexicon("Weapons: weapons").unwrap(),
            {"2003.txt": "2003"},
            within=parse_lexicon("Iraq: iraq").unwrap(),
        ).unwrap()
        assert rate(table, "2003", "Weapons", "Sentences") == 2
        assert rate(table, "2003", "Weapons", "Tokens") == 7

    def test_says_when_nothing_matched_the_anchor(self) -> None:
        result = lexicon_series(
            frame_of(CONTEXT),
            parse_lexicon("Weapons: weapons").unwrap(),
            {"2003.txt": "2003"},
            within=parse_lexicon("Space: apollo").unwrap(),
        )
        assert "LEXICON_NO_ANCHOR" in [d.code for d in result.diagnostics]


# ------------------------------------------------------------- the figures --


def _facets(labels: list[str]) -> pd.DataFrame:
    """A lexicon_series table: one category, one row per facet."""
    return pd.DataFrame(
        {
            "Facet": labels,
            "Category": ["Queen"] * len(labels),
            "Per 1000": [float(i) for i in range(len(labels))],
            "Occurrences": [i for i in range(len(labels))],
            "Documents": [1] * len(labels),
            "Tokens": [1000] * len(labels),
        }
    )


def _drawn(panel: str, frame: pd.DataFrame, by: str):  # type: ignore[no-untyped-def]
    from core.viz.panels import prepare_panel
    from core.viz.panelspec import Source

    return prepare_panel(panel, frame, {}, source=Source(path="lexicon_series.csv", settings={"by": by}))


class TestFiguresAlongAnyAxis:
    """A book counted chapter by chapter draws along its chapters, in reading order."""

    CHAPTERS = tuple(f"Chapter {i}" for i in (10, 2, 1, 11, 3))

    def test_the_rate_line_runs_across_the_chapters(self) -> None:
        panel = _drawn("lexicon_series_rate", _facets(list(self.CHAPTERS)), "order").unwrap()
        assert [mark.x for mark in panel.marks] == [1.0, 2.0, 3.0, 10.0, 11.0]
        assert (panel.x_axis, panel.x_noun, panel.x_label) == ("order", "Chapter", "Chapter")
        assert "across the chapters" in panel.title

    def test_the_published_figure_says_chapters_not_years(self) -> None:
        pytest.importorskip("seaborn")
        import matplotlib.pyplot as plt

        from core.viz.static.shapes import draw_line_series

        three = pd.concat([_facets(list(self.CHAPTERS)).assign(Category=name) for name in ("A", "B", "C")])
        panel = _drawn("lexicon_series_rate", three, "order").unwrap()
        figure, notes = draw_line_series(panel)
        ticks = [label.get_text() for axes in figure.axes for label in axes.get_xticklabels()]
        plt.close(figure)
        assert any("single chapters" in note for note in notes), notes
        assert not any("year" in note for note in notes), notes
        assert any(tick.startswith("Ch. ") for tick in ticks), ticks

    def test_a_dated_run_is_still_a_line_over_time(self) -> None:
        panel = _drawn("lexicon_series_rate", _facets(["1990", "1980"]), "year").unwrap()
        assert [mark.x for mark in panel.marks] == [1980.0, 1990.0]
        assert (panel.x_axis, panel.x_label) == ("time", "Year")

    def test_a_detail_run_is_sent_to_the_heatmap_by_its_name(self) -> None:
        result = _drawn("lexicon_series_rate", _facets(["sotu", "ina"]), "field:Kind")
        assert result.value is None
        message = " ".join(d.message for d in result.diagnostics)
        assert "'Kind'" in message and "heatmap" in message

    def test_the_heatmap_reads_chapters_in_order_and_names_the_detail(self) -> None:
        panel = _drawn("lexicon_series_heatmap", _facets(list(self.CHAPTERS)), "order").unwrap()
        assert list(panel.x_categories) == [f"Chapter {i}" for i in (1, 2, 3, 10, 11)]
        by_kind = _drawn("lexicon_series_heatmap", _facets(["sotu", "ina"]), "field:Kind").unwrap()
        assert by_kind.x_label == "Kind"
