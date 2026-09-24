"""ngram_viewer: frequency over time (FR-1.6, CAP-NGRAM-10).

Offline contract tests: year aggregation rates, the smoothing window, the
EMPTY STATE that teaches (no dates anywhere -> NG_VIEWER_NO_DATES with the
homework's answer as its fix), the partial-dates warning, the no-hits info,
the query/smooth guards, the chart's empty state and its no-plotly fallback.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from core.analysis.ngram_viewer import SERIES_COLUMNS, frame_tokens, ngram_series
from core.viz.ngram_viewer import ngram_viewer_html

_TINY = pd.DataFrame(
    {
        "ID": [1, 2],
        "Form": ["War", "war"],
        "Lemma": ["war", "war"],
        "POS": ["NN", "NN"],
        "NER": ["O", "O"],
        "Head": [0, 0],
        "DepRel": ["root", "root"],
        "Sentence ID": [1, 1],
        "Document ID": ["7", "7"],
        "Document": ["1900_a.txt", "1900_a.txt"],
    }
)


class TestSeries:
    def test_counts_rates_and_share_per_year(self) -> None:
        doc_tokens = {
            "1": ["war", "and", "peace", "war"],
            "2": ["war"],
            "3": ["peace", "peace"],
        }
        dates = {"1": 1900, "2": 1901, "3": 1900}
        result = ngram_series(doc_tokens, dates, ["war", "peace", "cold war"])
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert list(frame.columns) == SERIES_COLUMNS
        war_1900 = frame[(frame["N-gram"] == "war") & (frame["Year"] == 1900)].iloc[0]
        assert war_1900["Count"] == 2
        assert war_1900["Per Million"] == round(1000000.0 * 2 / 6, 2)
        assert war_1900["Share of Documents"] == 0.5
        peace_1900 = frame[(frame["N-gram"] == "peace") & (frame["Year"] == 1900)].iloc[0]
        assert peace_1900["Count"] == 3
        assert peace_1900["Share of Documents"] == 1.0
        peace_1901 = frame[(frame["N-gram"] == "peace") & (frame["Year"] == 1901)].iloc[0]
        assert peace_1901["Count"] == 0
        assert peace_1901["Per Million"] == 0
        assert peace_1901["Share of Documents"] == 0
        assert any(d.code == "NG_VIEWER_NO_HITS" for d in result.diagnostics)
        assert "cold war" in str(next(d for d in result.diagnostics if d.code == "NG_VIEWER_NO_HITS").message)
        cold_war = frame[frame["N-gram"] == "cold war"]
        assert cold_war["Year"].tolist() == [1900, 1901]
        assert cold_war["Count"].tolist() == [0, 0]

    def test_exposure_years_include_zero_hits_but_no_corpus_years_stay_absent(self) -> None:
        doc_tokens = {
            "1": ["war", "x"],
            "2": ["peace", "x", "x", "x"],
            "3": ["war", "x"],
        }
        dates = {"1": 1900, "2": 1901, "3": 1903}
        frame = ngram_series(doc_tokens, dates, ["war"]).unwrap()
        war = frame[frame["N-gram"] == "war"].set_index("Year")
        assert list(war.index) == [1900, 1901, 1903]
        assert war["Count"].tolist() == [1, 0, 1]
        assert war["Per Million"].tolist() == [500000.0, 0.0, 500000.0]
        assert war["Share of Documents"].tolist() == [1.0, 0.0, 1.0]
        assert war["Corpus Tokens"].tolist() == [2, 4, 2]
        assert war["Corpus Documents"].tolist() == [1, 1, 1]
        assert war["Documents With Hit"].tolist() == [1, 0, 1]

    def test_smoothing_uses_zero_hit_exposure_years_and_does_not_bridge_gap(self) -> None:
        doc_tokens = {"1": ["war"], "2": ["peace"], "3": ["war"]}
        dates = {"1": 1900, "2": 1901, "3": 1903}
        frame = ngram_series(doc_tokens, dates, ["war"], smooth=3).unwrap()
        war = frame[frame["N-gram"] == "war"].set_index("Year")
        assert list(war.index) == [1900, 1901, 1903]
        assert war["Count"].tolist() == [0.5, 0.5, 1.0]
        assert war["Share of Documents"].tolist() == [1.0, 0.0, 1.0]
        assert war["Raw Count"].tolist() == [1, 0, 1]
        assert war["Raw Per Million"].tolist() == [1000000.0, 0.0, 1000000.0]

    def test_phrases_match_consecutive_tokens(self) -> None:
        doc_tokens = {"1": ["cold", "war", "cold", "peace"], "2": ["war", "cold"]}
        dates = {"1": 1900, "2": 1900}
        result = ngram_series(doc_tokens, dates, ["cold war"])
        assert result.ok, result.diagnostics
        assert result.unwrap()["Count"].tolist() == [1]

    def test_case_insensitive_by_default(self) -> None:
        doc_tokens = {"1": ["War", "WAR", "war"]}
        dates = {"1": 1900}
        loose = ngram_series(doc_tokens, dates, ["war"]).unwrap()
        strict = ngram_series(doc_tokens, dates, ["war"], case_sensitive=True).unwrap()
        assert loose["Count"].tolist() == [3]
        assert strict["Count"].tolist() == [1]

    def test_smoothing_is_a_centered_window_in_years_with_truncated_edges(self) -> None:
        doc_tokens = {"1": ["war"], "2": ["war"] * 2, "3": ["war"] * 3}
        dates = {"1": 1900, "2": 1901, "3": 1902}
        result = ngram_series(doc_tokens, dates, ["war"], smooth=3)
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert frame["Count"].tolist() == [1.5, 2.0, 2.5]

    def test_smooth_one_keeps_raw_integer_counts(self) -> None:
        doc_tokens = {"1": ["war"] * 4}
        dates = {"1": 1900}
        frame = ngram_series(doc_tokens, dates, ["war"], smooth=1).unwrap()
        assert frame["Count"].tolist() == [4]

    def test_empty_input_is_an_empty_table_not_a_failure(self) -> None:
        result = ngram_series({}, {}, ["war"])
        assert result.ok, result.diagnostics
        frame = result.unwrap()
        assert frame.empty
        assert list(frame.columns) == SERIES_COLUMNS


class TestDates:
    def test_no_dates_is_the_teaching_failure(self) -> None:
        doc_tokens = {"1": ["war"] * 5, "2": ["peace"] * 5}
        result = ngram_series(doc_tokens, {"1": None, "2": None}, ["war"])
        assert not result.ok
        diag = result.diagnostics[0]
        assert diag.code == "NG_VIEWER_NO_DATES"
        assert "co-occurrence viewer" in str(diag.context["fix"])
        assert "1934_address.txt" in str(diag.context["fix"])

    def test_some_dates_missing_warns_and_names_the_drop(self) -> None:
        doc_tokens = {"1": ["war"], "2": ["war"], "3": ["war"]}
        dates = {"1": 1900, "2": None, "3": 1901}
        result = ngram_series(doc_tokens, dates, ["war"])
        assert result.ok, result.diagnostics
        warning = next(d for d in result.diagnostics if d.code == "NG_VIEWER_UNDATED_DROPPED")
        assert warning.context["dropped"] == 1
        assert result.unwrap()["Count"].tolist() == [1, 1]

    def test_a_date_object_contributes_its_year(self) -> None:
        from datetime import date

        result = ngram_series({"1": ["war"]}, {"1": date(1934, 1, 3)}, ["war"])
        assert result.ok, result.diagnostics
        assert result.unwrap()["Year"].tolist() == [1934]


class TestGuards:
    def test_query_longer_than_three_words_fails_loudly(self) -> None:
        result = ngram_series({"1": ["war"]}, {"1": 1900}, ["the cold war years"])
        assert not result.ok
        assert result.diagnostics[0].code == "NG_VIEWER_BAD_QUERY"

    def test_no_queries_fails_loudly(self) -> None:
        result = ngram_series({"1": ["war"]}, {"1": 1900}, [])
        assert not result.ok
        assert result.diagnostics[0].code == "NG_VIEWER_BAD_QUERY"

    def test_smooth_below_one_fails_loudly(self) -> None:
        result = ngram_series({"1": ["war"]}, {"1": 1900}, ["war"], smooth=0)
        assert not result.ok
        assert result.diagnostics[0].code == "NG_VIEWER_BAD_SMOOTH"


class TestFrameTokens:
    def test_keys_are_document_ids_and_forms_stay_raw(self) -> None:
        tokens = frame_tokens(_TINY)
        assert list(tokens) == ["7"]
        assert tokens["7"] == ["War", "war"]

    def test_empty_frame_gives_no_tokens(self) -> None:
        assert frame_tokens(pd.DataFrame()) == {}


class TestChart:
    def test_html_line_inserts_missing_year_in_axis_order(self, monkeypatch) -> None:
        plotly = pytest.importorskip("plotly.express")
        captured: dict[str, pd.DataFrame] = {}

        class Figure:
            def update_traces(self, **_kwargs: object) -> None:
                pass

            def update_layout(self, **_kwargs: object) -> None:
                pass

            def to_html(self, **_kwargs: object) -> str:
                return "<div>chart</div>"

        def line(frame: pd.DataFrame, **_kwargs: object) -> Figure:
            captured["frame"] = frame
            return Figure()

        monkeypatch.setattr(plotly, "line", line)
        series = ngram_series({"a": ["war"], "b": ["war"]}, {"a": 1900, "b": 1902}, ["war"]).unwrap()
        assert ngram_viewer_html(series).ok
        ordered = captured["frame"].sort_values("Year", kind="stable")
        assert captured["frame"]["Year"].tolist() == ordered["Year"].tolist()
        assert captured["frame"]["Year"].tolist() == [1900.0, 1901.0, 1902.0]
        assert pd.isna(captured["frame"].iloc[1]["Per Million"])

    def test_empty_series_is_a_paragraph_not_a_failure(self) -> None:
        result = ngram_viewer_html(pd.DataFrame(columns=SERIES_COLUMNS))
        assert result.ok, result.diagnostics
        assert "<p>" in result.unwrap()
        assert "no hits" in result.unwrap().lower()

    def test_missing_columns_fail_loudly(self) -> None:
        result = ngram_viewer_html(pd.DataFrame({"Year": [1900], "Count": [1]}))
        assert not result.ok
        assert result.diagnostics[0].code == "NG_VIEWER_PLOT_BAD_COLUMN"

    def test_chart_renders_or_falls_back_with_a_warning(self) -> None:
        series = ngram_series({"1": ["war"]}, {"1": 1900}, ["war"]).unwrap()
        result = ngram_viewer_html(series, title="War over time")
        assert result.ok, result.diagnostics
        html = result.unwrap()
        assert "plotly" in html.lower() or "ngv-table" in html

    def test_no_plotly_falls_back_to_a_table_and_says_so(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "plotly.express", None)
        series = ngram_series({"1": ["war"]}, {"1": 1900}, ["war"]).unwrap()
        result = ngram_viewer_html(series)
        assert result.ok, result.diagnostics
        assert "ngv-table" in result.unwrap()
        warning = next(d for d in result.diagnostics if d.code == "NG_VIEWER_PLOTLY_UNAVAILABLE")
        assert warning.context["fix"] == "pip install plotly"


class TestCli:
    def test_queries_flag_parses_comma_separated_phrases(self) -> None:
        from tools.ngram_viewer import _parse_args

        args = _parse_args(["corpus", "out", "--queries", "war, peace"])
        assert args.queries == "war, peace"
        assert args.case_sensitive is False
        assert args.smooth == 1

    def test_smooth_and_case_sensitive_flags_exist(self) -> None:
        from tools.ngram_viewer import _parse_args

        args = _parse_args(["corpus", "out", "--queries", "war", "--smooth", "5", "--case-sensitive"])
        assert args.smooth == 5
        assert args.case_sensitive is True
