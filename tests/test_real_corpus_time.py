"""Can you ask "did this change over ninety years?" of a real dated corpus?

The 87 State of the Union addresses span 1934 to 2024, and until now no table
this suite produced carried a date, so that question could not be put to any
of them. Every chart of the corpus was a ranking of 87 documents against each
other in whatever order a measure happened to fall.

These tests run the real analyses over the real speeches and check the whole
path: the corpus's dates reach the result, they are the *right* dates, the
reading says what stretch of time the table covers, and the chart offered
first is a timeline rather than a ranking. A fixture cannot show any of this
-- a hand-built frame has whatever dates its author typed, in whatever order
they typed them, and would agree with a join that matched the wrong rows.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from conftest import real_corpus_dir

pytestmark = [
    pytest.mark.model_integration,
    pytest.mark.skipif(real_corpus_dir() is None, reason="no real corpus on this machine"),
]


def _run(snapshot: Any, corpus: Any, tool: str, params: dict[str, Any] | None = None) -> Any:
    """One analysis over the real parse, through the path a publish uses."""
    from core.profiler.executor import execute
    from core.profiler.plan import build_plan

    plan = build_plan([tool], {tool: params or {}}).unwrap()
    batch = execute(
        plan,
        corpus=corpus,
        table=snapshot.table,
        parse_diagnostics=snapshot.diagnostics,
        tokenizer=snapshot.tokenizer,
    )
    outcome = batch.outcomes[0]
    assert outcome.ok, [d.message for d in outcome.diagnostics]
    return outcome


class TestTheCorpusDatesReachTheResult:
    def test_a_per_document_result_carries_the_date_of_each_document(
        self, real_snapshot: Any, real_corpus: Any
    ) -> None:
        """The date on a row is the date of the document that row is about.

        Checked against the corpus rather than against a list of years written
        down here: a join on the wrong key still produces a full Date column,
        and one that is right about every row except which row it belongs to
        is exactly the failure a spot-check of the first row would pass.
        """
        outcome = _run(real_snapshot, real_corpus, "readability")
        frame = next(iter(outcome.frames.values()))
        assert "Date" in frame.columns and "Year" in frame.columns

        expected = {str(doc.doc_id): doc.date for doc in real_corpus.docs}
        assert any(value is not None for value in expected.values()), "the real corpus has no dated documents"
        for _, row in frame.iterrows():
            wanted = expected[str(row["Document ID"])]
            assert row["Date"] == (wanted.isoformat() if wanted else None)
            assert row["Year"] == (wanted.year if wanted else None)

    def test_the_dates_span_the_corpus_rather_than_one_end_of_it(self, real_snapshot: Any, real_corpus: Any) -> None:
        """A join that half-matched would still fill a Date column -- with the
        same few dates repeated, or with most of them empty."""
        outcome = _run(real_snapshot, real_corpus, "readability")
        frame = next(iter(outcome.frames.values()))
        dated = frame["Date"].dropna()
        corpus_dates = sorted({doc.date for doc in real_corpus.docs if doc.date})

        assert len(dated) == len(frame), "some rows lost their date in the join"
        assert sorted({date.fromisoformat(value) for value in dated}) == corpus_dates
        assert corpus_dates[-1].year - corpus_dates[0].year > 50, "expected a corpus spanning decades"

    def test_the_date_sits_beside_the_document_not_after_the_measures(
        self, real_snapshot: Any, real_corpus: Any
    ) -> None:
        """Where a column lands decides whether anyone finds it. Appended last
        it is below every measure in the axis pickers, which is where a column
        goes to be ignored."""
        outcome = _run(real_snapshot, real_corpus, "readability")
        columns = list(next(iter(outcome.frames.values())).columns)
        assert columns.index("Date") == columns.index("Document") + 1
        assert columns.index("Year") == columns.index("Date") + 1

    def test_every_per_document_frame_a_tool_produces_is_dated(self, real_snapshot: Any, real_corpus: Any) -> None:
        """Sentiment publishes a document table and a sentence table. Both are
        about documents, so a timeline should be askable of either -- dating
        only the first frame would make that depend on which one you opened."""
        outcome = _run(real_snapshot, real_corpus, "sentiment_vader_anew", {"analysis": "vader"})
        assert len(outcome.frames) > 1
        for name, frame in outcome.frames.items():
            assert "Document ID" in frame.columns, name
            assert "Date" in frame.columns, f"{name} is per-document but carries no date"
            assert frame["Date"].notna().all(), name


class TestAnUndatedCorpusIsLeftAlone:
    def test_no_date_columns_appear_when_the_corpus_knows_no_dates(self, real_snapshot: Any, real_corpus: Any) -> None:
        """Two empty columns are worse than none: they offer an axis whose
        every value is blank, and a reader cannot tell that from a bug."""
        from dataclasses import replace

        undated = replace(real_corpus, docs=tuple(replace(doc, date=None) for doc in real_corpus.docs))
        frame = next(iter(_run(real_snapshot, undated, "readability").frames.values()))
        assert "Date" not in frame.columns
        assert "Year" not in frame.columns


class TestTheChartOfferedFirstAnswersSomething:
    def test_sentiment_over_a_dated_corpus_opens_on_a_timeline(self, real_snapshot: Any, real_corpus: Any) -> None:
        """The complaint this exists to answer: a run over these speeches drew
        87 bars of how many sentences each contained, sorted by size."""
        from core.insight.recommend import recommend_charts

        frame = _run(real_snapshot, real_corpus, "sentiment_vader_anew", {"analysis": "vader"}).frames["vader.csv"]
        first = recommend_charts(frame, tool="sentiment_vader_anew", limit=4)[0]

        assert first.spec.kind == "line"
        assert first.spec.x == "Date"
        assert first.spec.y == "Compound", "a timeline of sentence counts is a chart of how long the speeches are"
        # A top-N cut keeps the largest categories, which on a timeline deletes
        # the quiet years and joins what is left into a line that never happened.
        assert first.spec.top_n is None
        assert first.spec.agg == "mean"

    def test_the_same_table_undated_still_recommends_what_it_always_did(
        self, real_snapshot: Any, real_corpus: Any
    ) -> None:
        from core.insight.recommend import recommend_charts

        frame = _run(real_snapshot, real_corpus, "sentiment_vader_anew", {"analysis": "vader"}).frames["vader.csv"]
        without = frame.drop(columns=["Date", "Year"])
        first = recommend_charts(without, tool="sentiment_vader_anew", limit=4)[0]
        assert first.spec.kind == "bar"
        assert (first.spec.x, first.spec.y) == ("Document", "Compound")

    def test_the_reading_says_what_stretch_of_time_the_table_covers(self, real_snapshot: Any, real_corpus: Any) -> None:
        from core.insight.readout import readout

        frame = _run(real_snapshot, real_corpus, "sentiment_vader_anew", {"analysis": "vader"}).frames["vader.csv"]
        reading = readout(frame, tool="sentiment_vader_anew")
        first, last = sorted(frame["Date"].dropna())[0], sorted(frame["Date"].dropna())[-1]
        assert any(first in line and last in line for line in reading.observations), reading.observations
