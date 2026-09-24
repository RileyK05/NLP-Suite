"""CAP-ANNO-11 — normalized date extraction (SUTime-style, regex).

Hand-built expectations for every recognized surface, the documented US
month/day reading of numeric dates, out-of-range year rejection (counted,
not silent), and the empty-input contract. No corpus and no parse: this
annotator is pure text.
"""

from __future__ import annotations

import pandas as pd

from core.analysis.date_annotator import annotate_corpus, annotate_dates, summarize_dates


class TestSurfaces:
    def _one(self, text: str) -> dict[str, object]:
        result = annotate_dates(text, document="a.txt", document_id="1")
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert len(frame) == 1, frame
        return frame.iloc[0].to_dict()

    def test_iso_full_date(self) -> None:
        row = self._one("Due on 2020-03-05 .")
        assert row["Surface"] == "2020-03-05"
        assert row["Normalized"] == "2020-03-05"
        assert row["Type"] == "date"

    def test_iso_month(self) -> None:
        row = self._one("Since 2020-03 things changed.")
        assert row["Normalized"] == "2020-03"
        assert row["Type"] == "month_year"

    def test_month_name_day_year(self) -> None:
        row = self._one("On March 5, 2020 it began.")
        assert row["Normalized"] == "2020-03-05"
        assert row["Type"] == "date"

    def test_month_name_ordinal_day_year(self) -> None:
        row = self._one("On March 5th, 2020 it began.")
        assert row["Normalized"] == "2020-03-05"

    def test_abbreviated_month_no_comma(self) -> None:
        row = self._one("By Mar 5 2020 it was over.")
        assert row["Normalized"] == "2020-03-05"
        assert row["Surface"] == "Mar 5 2020"

    def test_day_month_year(self) -> None:
        row = self._one("Signed 5 March 2020 .")
        assert row["Normalized"] == "2020-03-05"
        assert row["Surface"] == "5 March 2020"

    def test_us_slash_dates_are_month_day(self) -> None:
        # The documented ambiguity: 03/05 is March 5, never 3 May.
        for surface in ("03/05/2020", "3/5/2020"):
            row = self._one(f"It happened {surface} .")
            assert row["Normalized"] == "2020-03-05", surface
            assert row["Type"] == "date"

    def test_bare_year(self) -> None:
        row = self._one("In 1934 the law passed.")
        assert row["Normalized"] == "1934"
        assert row["Type"] == "year"

    def test_decade_normalizes_to_its_start(self) -> None:
        row = self._one("The 1930s were hard.")
        assert row["Surface"] == "1930s"
        assert row["Normalized"] == "1930"
        assert row["Type"] == "decade"

    def test_month_year_by_name(self) -> None:
        row = self._one("By March 2020 it was over.")
        assert row["Normalized"] == "2020-03"
        assert row["Type"] == "month_year"

    def test_year_range_kept_by_its_own_regex(self) -> None:
        # "1934-1938" is two years, not one malformed ISO month.
        result = annotate_dates("from 1934-1938 .")
        frame = result.unwrap()
        assert list(frame["Type"]) == ["year", "year"]
        assert list(frame["Normalized"]) == ["1934", "1938"]

    def test_dates_get_sequential_ids(self) -> None:
        result = annotate_dates("1934 then 1936 .", document="a.txt", document_id="1", date_id_start=5)
        frame = result.unwrap()
        assert list(frame["Date ID"]) == [5, 6]

    def test_year_inside_a_longer_number_is_not_a_date(self) -> None:
        result = annotate_dates("12345 items and 12020-03 things.")
        assert result.unwrap().empty


class TestRejection:
    def test_out_of_range_year_is_skipped_and_counted(self) -> None:
        result = annotate_dates("founded in 1500 .", min_year=1800, max_year=2100)
        assert result.ok
        assert result.unwrap().empty
        diag = next(d for d in result.diagnostics if d.code == "DATE_YEAR_SKIPPED")
        assert diag.severity.value == "WARNING"
        assert diag.context["skipped"] == 1

    def test_comma_number_is_not_a_year(self) -> None:
        result = annotate_dates("1,500 soldiers advanced.")
        assert result.ok
        assert result.unwrap().empty
        assert not any(d.code == "DATE_YEAR_SKIPPED" for d in result.diagnostics)

    def test_range_applies_to_every_date_type(self) -> None:
        result = annotate_dates("March 5, 1492 and the 1490s .", min_year=1800, max_year=2100)
        assert result.unwrap().empty
        assert next(d for d in result.diagnostics if d.code == "DATE_YEAR_SKIPPED").context["skipped"] == 2

    def test_accepted_dates_are_not_counted_as_skipped(self) -> None:
        result = annotate_dates("1934 and 1500 .", min_year=1900, max_year=2100)
        frame = result.unwrap()
        assert list(frame["Normalized"]) == ["1934"]
        assert next(d for d in result.diagnostics if d.code == "DATE_YEAR_SKIPPED").context["skipped"] == 1


class TestEmptyContract:
    def test_empty_text_yields_the_declared_columns(self) -> None:
        result = annotate_dates("", document="a.txt", document_id="1")
        assert result.ok
        frame = result.unwrap()
        assert list(frame.columns) == ["Document", "Document ID", "Date ID", "Surface", "Normalized", "Type"]
        assert len(frame) == 0

    def test_empty_corpus_yields_the_declared_columns(self) -> None:
        result = annotate_corpus([])
        assert result.ok
        assert len(result.unwrap()) == 0

    def test_empty_summary_yields_the_declared_columns(self) -> None:
        empty = pd.DataFrame(columns=["Document", "Document ID", "Date ID", "Surface", "Normalized", "Type"])
        result = summarize_dates(empty)
        assert result.ok
        frame = result.unwrap()
        assert list(frame.columns) == ["Document", "Document ID", "Dates", "Distinct", "Earliest", "Latest"]
        assert len(frame) == 0


class TestLoudFailures:
    def test_inverted_year_range_is_loud(self) -> None:
        result = annotate_dates("1934 .", min_year=2100, max_year=1000)
        assert not result.ok
        assert result.diagnostics[0].code == "DATE_BAD_RANGE"

    def test_inverted_year_range_is_loud_for_the_corpus_too(self) -> None:
        result = annotate_corpus([("a.txt", "1", "1934 .")], min_year=2100, max_year=1000)
        assert not result.ok
        assert result.diagnostics[0].code == "DATE_BAD_RANGE"

    def test_malformed_doc_tuple_is_loud(self) -> None:
        result = annotate_corpus([("a.txt", "1")])  # type: ignore[arg-type]
        assert not result.ok
        assert result.diagnostics[0].code == "DATE_BAD_DOCS"

    def test_missing_summary_column_is_loud(self) -> None:
        result = summarize_dates(pd.DataFrame({"Document": ["a.txt"]}))
        assert not result.ok
        assert result.diagnostics[0].code == "DATE_MISSING_COLUMN"


class TestCorpus:
    def test_ids_continue_across_documents(self) -> None:
        docs = [
            ("a.txt", "1", "1934 ."),
            ("b.txt", "2", "1936 and 1938 ."),
        ]
        result = annotate_corpus(docs)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame["Date ID"]) == [1, 2, 3]
        assert list(frame["Document"]) == ["a.txt", "b.txt", "b.txt"]

    def test_skips_are_reported_per_document(self) -> None:
        docs = [
            ("a.txt", "1", "1500 and 1600 ."),
            ("b.txt", "2", "1934 ."),
        ]
        result = annotate_corpus(docs, min_year=1900, max_year=2100)
        assert result.ok
        assert list(result.unwrap()["Normalized"]) == ["1934"]
        diag = next(d for d in result.diagnostics if d.code == "DATE_YEAR_SKIPPED")
        assert diag.context["document"] == "a.txt"
        assert diag.context["skipped"] == 2


class TestSummary:
    def test_spread_of_dates_per_document(self) -> None:
        result = annotate_dates("1934 , 1936 , 1934 .", document="a.txt", document_id="1")
        summary = summarize_dates(result.unwrap())
        row = summary.unwrap().iloc[0]
        assert row["Dates"] == 3
        assert row["Distinct"] == 2
        assert row["Earliest"] == "1934"
        assert row["Latest"] == "1936"
