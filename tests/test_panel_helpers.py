"""The shared panel rules: labels, dates, groupings, rates, smoothing, layout.

File names are the real State of the Union names, spaces included, because
those are what the helpers are for.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.viz.panel_helpers import (
    decimal_year,
    document_labels,
    force_layout,
    group_of,
    is_function_word,
    per_10k,
    rolling_median,
    short_document_label,
    speaker_of,
)

FDR = "1934-01-03_franklin d roosevelt_sotu.txt"
TRUMAN_JAN = "1946-01-21_harry s truman_sotu.txt"
TRUMAN_OTHER = "1946-01-14_harry s truman_sotu.txt"


class TestNames:
    def test_speaker_is_parsed_from_the_middle_of_the_name(self) -> None:
        assert speaker_of(FDR) == "Franklin D Roosevelt"

    def test_a_name_without_the_pattern_has_no_speaker(self) -> None:
        assert speaker_of("notes.txt") == ""
        assert speaker_of("1934_only-a-date.txt") == "", "no kind part: not the pattern, not a guess"

    def test_short_label_is_year_and_surname(self) -> None:
        assert short_document_label(FDR) == "1934 Roosevelt"
        assert short_document_label("notes.txt") == "notes"

    def test_labels_on_one_axis_are_unique(self) -> None:
        labels = document_labels([FDR, TRUMAN_JAN, TRUMAN_OTHER])
        assert labels[FDR] == "1934 Roosevelt"
        assert labels[TRUMAN_JAN] == "1946-01-21 Truman"
        assert labels[TRUMAN_OTHER] == "1946-01-14 Truman"
        assert len(set(labels.values())) == 3


class TestDates:
    def test_decimal_year_places_a_day_within_its_year(self) -> None:
        assert decimal_year("1934-01-01") == 1934.0
        assert decimal_year("1934-07-02") == pytest.approx(1934.5, abs=0.01)
        assert decimal_year(date(2024, 3, 7)) == pytest.approx(2024.18, abs=0.01)
        assert decimal_year(1950) == 1950.0

    def test_undated_is_none_not_year_zero(self) -> None:
        assert decimal_year(None) is None
        assert decimal_year(float("nan")) is None
        assert decimal_year("someday") is None

    def test_groupings(self) -> None:
        assert group_of(FDR, "1934-01-03", "decade") == "1930s"
        assert group_of(FDR, "1934-01-03", "year") == "1934"
        assert group_of(FDR, "1934-01-03", "speaker") == "Franklin D Roosevelt"
        assert group_of(FDR, None, "decade") == "(undated)"
        assert group_of("notes.txt", None, "speaker") == "(speaker not in file name)"
        with pytest.raises(ValueError, match="grouping"):
            group_of(FDR, None, "century")


class TestRates:
    def test_per_10k(self) -> None:
        assert per_10k(5, 50_000) == 1.0
        assert per_10k(5, 0) is None

    def test_function_words(self) -> None:
        assert is_function_word("of") and is_function_word(" The ")
        assert not is_function_word("congress")


class TestRollingMedian:
    def test_centred_and_ordered_by_x(self) -> None:
        # Out of order on purpose; the median follows x, not input order.
        out = rolling_median([3, 1, 2, 5, 4], [30, 10, 20, 50, 40], window=3)
        assert out == [(1.0, 20.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0), (5.0, 40.0)]

    def test_the_line_reaches_both_ends_with_a_full_window(self) -> None:
        """Regression: a centred-only window dropped the last (window-1)/2
        speeches -- 2021-2024 on the real corpus at a window of nine."""
        xs = list(range(20))
        out = rolling_median(xs, [float(x) for x in xs], window=9)
        assert [x for x, _ in out] == [float(x) for x in xs]
        assert out[0][1] == 4.0, "the first point is the median of the first nine"
        assert out[-1][1] == 15.0, "the last point is the median of the last nine"

    def test_too_few_points_for_the_window_draws_nothing(self) -> None:
        assert rolling_median([1, 2], [1, 2], window=3) == []

    def test_resists_one_outlier(self) -> None:
        out = rolling_median([1, 2, 3], [1, 100, 2], window=3)
        assert [y for _, y in out] == [2.0, 2.0, 2.0]

    def test_even_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            rolling_median([1], [1], window=2)


class TestForceLayout:
    def test_deterministic_and_inside_the_unit_square(self) -> None:
        nodes = ["war", "peace", "tax", "budget", "school"]
        edges = [("war", "peace", 5.0), ("tax", "budget", 4.0), ("budget", "school", 1.0)]
        first = force_layout(nodes, edges)
        assert first == force_layout(nodes, edges)
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in first.values())

    def test_linked_nodes_sit_closer_than_unlinked(self) -> None:
        nodes = ["a", "b", "c", "d"]
        pos = force_layout(nodes, [("a", "b", 10.0), ("c", "d", 10.0)])

        def gap(p: str, q: str) -> float:
            return ((pos[p][0] - pos[q][0]) ** 2 + (pos[p][1] - pos[q][1]) ** 2) ** 0.5

        assert gap("a", "b") < gap("a", "c")
        assert gap("c", "d") < gap("b", "d")

    def test_trivial_graphs(self) -> None:
        assert force_layout([], []) == {}
        assert force_layout(["only"], []) == {"only": (0.5, 0.5)}
