"""The scripting library (``import nlpsuite as nlp``) over a small dated corpus.

The acceptance test for smart scripts is at the bottom: on the real State of
the Union corpus, ``term_rates(..., match="exact-lowercase")`` must give the
per-speech rates the hand-written immigration script published, to the last
digit. It ran green when this file was written (max difference 4e-16 over 87
speeches, 85 yearly means and the five-year trend); it is skipped where the
corpus is not checked out.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

from core.script.session import FigureOutput, FolderSource, NoteOutput, Session, TableOutput, open_session
import nlpsuite as nlp

ROOT = Path(__file__).resolve().parents[1]

TEXTS = {
    "1946-01-21_harry s truman_sotu.txt": "Immigrants came to the border. The border was closed. We spoke of peace.",
    "1948-01-07_harry s truman_sotu.txt": "Displaced persons need homes. An immigrant family arrived! Nothing else.",
    "1995-01-24_william j clinton_sotu.txt": "Border security matters. Deportation of aliens. Immigration law.",
    "2007-01-23_george w bush_sotu.txt": "Secure borders and fair laws. The economy grows.",
}
LEXICON = ["immigration", "immigrant", "immigrants", "border", "borders", "deportation", "aliens"]


@pytest.fixture
def session(tmp_path: Path) -> Session:
    for name, text in TEXTS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return Session(FolderSource(tmp_path))


@pytest.fixture
def parsed_session(session: Session) -> Session:
    pytest.importorskip("spacy")
    try:
        with open_session(session):
            nlp.corpus().tokens()
    except nlp.SuiteError:
        pytest.skip("no English parser installed")
    return session


class DetailSource(FolderSource):
    """A FolderSource whose documents carry details, as the app's source does.

    Details (Order, Village, Chapter ...) are what ``by="order"`` and
    ``by="field:..."`` group on; a plain folder has none.
    """

    def __init__(self, directory: Path, details: dict[str, dict[str, str]]) -> None:
        super().__init__(directory)
        self._details = details

    def documents(self) -> list[dict[str, object]]:
        rows = super().documents()
        for row in rows:
            row["fields"] = self._details.get(str(row["name"]), {})
        return rows

    def corpus(self, ids):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        base = super().corpus(ids)
        docs = tuple(replace(doc, fields=tuple(self._details.get(doc.name, {}).items())) for doc in base.docs)
        return replace(base, docs=docs)


def _raw_rate(text: str) -> tuple[int, int]:
    words = re.findall(r"[A-Za-z]+", text.lower())
    return sum(words.count(term) for term in LEXICON), len(words)


class TestCorpus:
    def test_documents_carry_date_and_year(self, session: Session) -> None:
        with open_session(session):
            docs = nlp.corpus().documents
        assert list(docs.columns[:5]) == ["Document ID", "Document", "Date", "Year", "Words"]
        assert docs["Year"].tolist() == [1946, 1948, 1995, 2007]

    def test_filter_by_range_list_and_function(self, session: Session) -> None:
        with open_session(session):
            corpus = nlp.corpus()
            assert len(corpus.filter(Year=range(1940, 1950))) == 2
            assert len(corpus.filter(Year=[1995, 2007])) == 2
            assert corpus.documents["Words"].tolist() == [13, 10, 8, 8]
            assert len(corpus.filter(Words=lambda n: n > 9)) == 2
            assert len(nlp.corpus({"Year": 1995})) == 1

    def test_a_filter_matching_nothing_says_so(self, session: Session) -> None:
        with open_session(session), pytest.raises(nlp.SuiteError, match="No documents match"):
            nlp.corpus().filter(Year=1800)

    def test_unknown_column_lists_the_real_ones(self, session: Session) -> None:
        with open_session(session), pytest.raises(nlp.SuiteError, match="Columns: Document ID"):
            nlp.corpus().filter(Speaker="Truman")

    def test_no_session_is_a_readable_refusal(self) -> None:
        with pytest.raises(nlp.SuiteError, match="not connected to any documents"):
            nlp.corpus()


class TestTermRates:
    def test_exact_lowercase_is_the_raw_script_rule(self, session: Session) -> None:
        with open_session(session):
            rates = nlp.term_rates(nlp.corpus(), {"immigration": LEXICON}, match="exact-lowercase")
        for name, text in TEXTS.items():
            count, words = _raw_rate(text)
            row = rates.loc[rates["Document"] == name].iloc[0]
            assert row["immigration count"] == count
            assert row["Words counted"] == words
            assert row["immigration per 1000"] == pytest.approx(1000 * count / words)

    def test_by_year_pools_words(self, session: Session) -> None:
        with open_session(session):
            by_decade = nlp.term_rates(nlp.corpus(), LEXICON, by="decade", match="exact-lowercase", per=100)
        forties = by_decade.loc[by_decade["Decade"] == "1940s"].iloc[0]
        counts = [_raw_rate(t) for n, t in TEXTS.items() if n.startswith("194")]
        assert forties["Documents"] == 2
        assert forties["terms per 100"] == pytest.approx(100 * sum(c for c, _ in counts) / sum(w for _, w in counts))

    def test_lemma_counts_inflections_as_one_word(self, parsed_session: Session) -> None:
        with open_session(parsed_session):
            rates = nlp.term_rates(nlp.corpus(), {"immigrant": ["immigrant"]}, match="lemma")
        assert rates.loc[rates["Year"] == 1946, "immigrant count"].item() == 1  # "Immigrants"

    def test_bad_arguments_are_refused_in_words(self, session: Session) -> None:
        with open_session(session):
            with pytest.raises(nlp.SuiteError, match="match must be one of"):
                nlp.term_rates(nlp.corpus(), LEXICON, match="stem")
            with pytest.raises(nlp.SuiteError, match="by must be one of"):
                nlp.term_rates(nlp.corpus(), LEXICON, by="month")

    def test_by_order_groups_chapters_not_years(self, tmp_path: Path) -> None:
        details = {
            "1946-01-21_harry s truman_sotu.txt": {"Order": "1", "Village": "A"},
            "1948-01-07_harry s truman_sotu.txt": {"Order": "2", "Village": "B"},
            "1995-01-24_william j clinton_sotu.txt": {"Order": "10", "Village": "A"},
            "2007-01-23_george w bush_sotu.txt": {"Order": "11", "Village": "B"},
        }
        for name, text in TEXTS.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
        with open_session(Session(DetailSource(tmp_path, details))):
            by_order = nlp.term_rates(nlp.corpus(), LEXICON, by="order", match="exact-lowercase")
        # Chapter 10 belongs after chapter 2, not beside it: sorted 1, 2, 10, 11.
        assert by_order["Order"].tolist() == ["1", "2", "10", "11"]
        first = by_order.iloc[0]
        count, words = _raw_rate(TEXTS["1946-01-21_harry s truman_sotu.txt"])
        assert first["Documents"] == 1
        assert first["terms count"] == count
        assert first["terms per 1000"] == pytest.approx(1000 * count / words)

    def test_by_field_groups_the_named_detail(self, tmp_path: Path) -> None:
        details = {name: {"Village": "A" if index < 2 else "B"} for index, name in enumerate(TEXTS)}
        for name, text in TEXTS.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
        with open_session(Session(DetailSource(tmp_path, details))):
            by_village = nlp.term_rates(nlp.corpus(), LEXICON, by="field:Village", match="exact-lowercase", per=100)
        assert by_village["Village"].tolist() == ["A", "B"]
        counts = [_raw_rate(t) for t in list(TEXTS.values())[:2]]
        a = by_village.iloc[0]
        assert a["Documents"] == 2
        assert a["terms per 100"] == pytest.approx(100 * sum(c for c, _ in counts) / sum(w for _, w in counts))

    def test_by_period_makes_blocks_of_the_order(self, tmp_path: Path) -> None:
        details = {name: {"Order": str(index + 1)} for index, name in enumerate(TEXTS)}
        for name, text in TEXTS.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
        with open_session(Session(DetailSource(tmp_path, details))):
            by_period = nlp.term_rates(nlp.corpus(), LEXICON, by="period", match="exact-lowercase")
        assert len(by_period) == 2  # four documents make two blocks of two
        assert all(label.startswith("Documents ") for label in by_period["Period"])

    def test_a_document_without_the_grouping_value_is_refused(self, tmp_path: Path) -> None:
        details = {name: {} for name in TEXTS}
        details["1946-01-21_harry s truman_sotu.txt"] = {"Order": "1"}
        for name, text in TEXTS.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
        with (
            open_session(Session(DetailSource(tmp_path, details))),
            pytest.raises(nlp.SuiteError, match="needs a value for every document"),
        ):
            nlp.term_rates(nlp.corpus(), LEXICON, by="order", match="exact-lowercase")


class TestPassages:
    def test_every_counted_sentence_has_a_passage_with_context(self, session: Session) -> None:
        with open_session(session):
            found = nlp.passages(nlp.corpus(), LEXICON, match="exact-lowercase", context=1)
        truman = found.loc[found["Year"] == 1946]
        assert truman["Sentence ID"].tolist() == [1, 2]
        assert truman.iloc[0]["After"] == "The border was closed."
        assert truman.iloc[1]["Before"].startswith("Immigrants came")
        assert "Nothing else." not in set(found["Text"])


class TestRun:
    def test_run_returns_the_published_rows(self, session: Session) -> None:
        from core.profiler.executor import execute
        from core.profiler.plan import build_plan

        with open_session(session):
            corpus = nlp.corpus()
            result = nlp.run("readability", corpus)
        expected = execute(build_plan(["readability"], {}).unwrap(), corpus=corpus.core).outcomes[0].frames
        assert result.table.equals(expected["readability.csv"])

    def test_unknown_tool_and_parameter_suggest_the_right_name(self, session: Session) -> None:
        with open_session(session):
            with pytest.raises(nlp.SuiteError, match="Did you mean readability"):
                nlp.run("readabilty", nlp.corpus())
            with pytest.raises(nlp.SuiteError, match="did you mean seed"):
                nlp.run("lexical_diversity", nlp.corpus(), seeds=3)

    def test_describe_uses_python_names(self) -> None:
        assert "top_n (" in nlp.describe("tfidf")
        assert "top-n" not in nlp.describe("tfidf").split("Parameters:")[1]

    def test_keyness_between_two_filters(self, parsed_session: Session) -> None:
        with open_session(parsed_session):
            corpus = nlp.corpus()
            table = nlp.keyness(corpus.filter(Year=range(1940, 1950)), corpus.filter(Year=range(1990, 2010)), top_n=0)
        assert "Word" in table.columns
        assert any("A" in str(column) for column in table.columns)

    def test_keyness_group_a_is_corpus_a_even_when_it_is_not_the_earliest(self, tmp_path: Path) -> None:
        """Group A is corpus_a, not whichever documents happen to be numbered 1..N.

        The joined corpus's fingerprint is order-independent, so an earlier
        parse of the whole corpus can be read back with its own numbering --
        and a corpus whose documents come *after* the other's in that order
        silently became group B. The word counts pin which side is which.
        """
        pytest.importorskip("spacy")
        for name, text in {
            "1990-01-01_early.txt": "apples apples apples filler filler",
            "1990-02-01_early2.txt": "apples apples filler filler filler",
            "2000-01-01_late.txt": "oranges oranges oranges filler filler",
            "2000-02-01_late2.txt": "oranges oranges filler filler filler",
        }.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
        session = Session(FolderSource(tmp_path))
        try:
            with open_session(session):
                whole = nlp.corpus()
                whole.tokens()  # warm the whole corpus, whose order is by date
                late = whole.filter(Year=range(2000, 2001))  # A: the oranges
                early = whole.filter(Year=range(1990, 1991))  # B: the apples
                table = nlp.keyness(late, early, top_n=0).set_index("Word")
        except nlp.SuiteError:
            pytest.skip("no English parser installed")
        assert table.loc["orange", "Freq A"] == 5 and table.loc["orange", "Freq B"] == 0
        assert table.loc["apple", "Freq A"] == 0 and table.loc["apple", "Freq B"] == 5


class TestOutputs:
    def test_show_recommends_a_chart(self, session: Session) -> None:
        with open_session(session):
            nlp.show(nlp.term_rates(nlp.corpus(), LEXICON, match="exact-lowercase"))
        [output] = session.outputs
        assert isinstance(output, TableOutput)
        assert output.chart is not None

    def test_several_y_columns_are_drawn_as_series(self, session: Session) -> None:
        frame = pd.DataFrame({"Year": [1, 1, 2], "a": [1.0, 3.0, 2.0], "b": [4.0, 5.0, 6.0]})
        with open_session(session):
            nlp.show(frame, x="Year", y=["a", "b"], name="Two lines")
        [output] = session.outputs
        assert isinstance(output, TableOutput)
        assert output.name == "Two_lines"
        assert output.chart == {
            "kind": "line",
            "x": "Year",
            "y": "Value",
            "group": "Series",
            "agg": "mean",
            "top_n": 0,
            # The CSV keeps a and b side by side; copied matplotlib code must melt them first.
            "prepare": {"melt": {"id": "Year", "values": ["a", "b"]}},
        }
        assert output.chart_frame is not None and set(output.chart_frame["Series"]) == {"a", "b"}
        assert "mean" in output.chart_note

    def test_a_table_of_passages_is_charted_as_passages_per_year(self, session: Session) -> None:
        """It used to sum sentence numbers per document: a chart, and nothing to read in it."""
        with open_session(session):
            passages = nlp.passages(nlp.corpus(), LEXICON, match="exact-lowercase")
            nlp.show(passages)
        [output] = session.outputs
        assert isinstance(output, TableOutput) and output.chart is not None
        assert (output.chart["x"], output.chart["y"]) == ("Year", "Passages")
        assert output.chart_frame is not None
        counts = dict(zip(output.chart_frame["Year"], output.chart_frame["Passages"], strict=True))
        assert counts == passages["Year"].value_counts().to_dict()

    def test_a_reshaped_table_falls_back_to_the_recommender(self, session: Session) -> None:
        with open_session(session):
            passages = nlp.passages(nlp.corpus(), LEXICON, match="exact-lowercase")
            nlp.show(passages.drop(columns=["Year"]))
        [output] = session.outputs
        assert isinstance(output, TableOutput)
        assert output.chart is None or output.chart.get("y") != "Passages"

    def test_no_automatic_chart_measures_an_identifier(self, parsed_session: Session) -> None:
        """The class of the passages bug: every library table's automatic chart draws a measure.

        A row number, sentence number or document id summed per category is a
        chart with nothing in it. Each table the library returns is shown with
        no x/y, and whatever chart comes back must draw a real measure (or a
        count the library derived on purpose).
        """
        from core.insight.profile import ColumnRole, profile_frame

        with open_session(parsed_session):
            corpus = nlp.corpus()
            tables = {
                "documents": corpus.documents,
                "term_rates": nlp.term_rates(corpus, LEXICON, match="exact-lowercase"),
                "term_rates by year": nlp.term_rates(corpus, {"a": ["border"], "b": ["peace"]}, by="year"),
                "passages": nlp.passages(corpus, LEXICON, match="exact-lowercase"),
                "sentiment": nlp.sentiment(corpus),
                "keyness": nlp.keyness(corpus.filter(Year=range(1940, 1950)), corpus.filter(Year=range(1990, 2010))),
            }
            for name, table in tables.items():
                nlp.show(table, name=name)
        charts = {o.name: o for o in parsed_session.outputs if isinstance(o, TableOutput)}
        assert charts["sentiment"].chart is not None and charts["sentiment"].chart["y"] == "Compound"
        keyness = charts["keyness"].chart
        assert keyness is not None and (keyness["x"], keyness["y"]) == ("Word", "G2 (log-likelihood)")
        for name, output in charts.items():
            chart = output.chart
            if chart is None or (chart.get("prepare") or {}).get("count"):
                continue
            drawn = output.chart_frame if output.chart_frame is not None else output.frame
            role = profile_frame(drawn)[chart["y"]].role
            assert role is not ColumnRole.IDENTIFIER, f"{name}: the chart measures {chart['y']!r}, an identifier"

    def test_a_missing_column_is_named(self, session: Session) -> None:
        with open_session(session), pytest.raises(nlp.SuiteError, match="no column 'rate'"):
            nlp.show(pd.DataFrame({"Year": [1]}), x="Year", y="rate")

    def test_save_figure_and_note(self, session: Session) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        with open_session(session):
            nlp.save(pd.DataFrame({"x": [1]}), "rates")
            nlp.save(pd.DataFrame({"x": [2]}), "rates")
            nlp.figure(fig, "trend")
            nlp.note("Counts are exact lowercase forms.")
        saved, again, drawn, written = session.outputs
        assert isinstance(saved, TableOutput) and saved.saved and saved.name == "rates"
        assert isinstance(again, TableOutput) and again.name == "rates_2"
        assert isinstance(drawn, FigureOutput) and drawn.png.startswith(b"\x89PNG") and b"<svg" in drawn.svg
        assert isinstance(written, NoteOutput)
        assert not plt.fignum_exists(fig.number)

    def test_a_figure_survives_a_missing_svg_writer(self, session: Session) -> None:
        """A packaged runtime can lack matplotlib's SVG backend (it loads only
        when saving). The figure must be kept as its PNG, with a note -- not
        thrown away with the SVG that could not be made."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        real_savefig = fig.savefig

        def only_png(*args: object, **kwargs: object) -> object:
            if kwargs.get("format") == "svg":
                raise ModuleNotFoundError("No module named 'matplotlib.backends.backend_svg'")
            return real_savefig(*args, **kwargs)

        fig.savefig = only_png  # type: ignore[method-assign]
        with open_session(session):
            nlp.figure(fig, "trend")
        drawn, said = session.outputs
        assert isinstance(drawn, FigureOutput) and drawn.png.startswith(b"\x89PNG") and drawn.svg is None
        assert isinstance(said, NoteOutput) and "PNG only" in said.text and "backend_svg" in said.text
        assert not plt.fignum_exists(fig.number)

    def test_chart_draws_an_interactive_html_output(self, session: Session) -> None:
        """nlp.chart reaches the engine's plotly kinds (sankey and the rest)."""
        from core.script.session import HtmlOutput

        pytest.importorskip("plotly")
        frame = pd.DataFrame({"From": ["A", "A", "B"], "To": ["X", "Y", "X"], "N": [1, 2, 3]})
        with open_session(session):
            nlp.chart(frame, kind="sankey", x="From", y="N", group="To", agg="sum", name="flows")
        [output] = session.outputs
        assert isinstance(output, HtmlOutput) and output.name == "flows"
        assert "sankey" in output.html.lower()
        assert [call.function for call in session.calls] == ["chart"]

    def test_chart_refuses_what_the_engine_refuses(self, session: Session) -> None:
        with open_session(session), pytest.raises(nlp.SuiteError, match="needs --group"):
            nlp.chart(pd.DataFrame({"a": [1]}), kind="sankey", x="a", y="a")
        with open_session(session), pytest.raises(nlp.SuiteError, match="kind must be one of"):
            nlp.chart(pd.DataFrame({"a": [1]}), kind="spiral", x="a", y="a")

    def test_a_cell_run_again_keeps_its_names(self, session: Session) -> None:
        """In a kernel, running a cell a third time used to show "rates 3"."""
        frame = pd.DataFrame({"x": [1]})
        with open_session(session):
            for _ in range(3):
                session.start_cell()
                nlp.show(frame, name="rates")
        [output] = session.outputs  # earlier cells' outputs are not kept in memory either
        assert isinstance(output, TableOutput) and output.name == "rates"

    def test_calls_are_recorded_for_provenance(self, session: Session) -> None:
        with open_session(session):
            nlp.term_rates(nlp.corpus(), LEXICON, match="exact-lowercase")
        assert [call.function for call in session.calls] == ["corpus", "term_rates"]
