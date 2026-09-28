"""How a corpus lines up: time, order or none (docs/internal/PLAN_0.5.0.md 1.2, 1.6)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from core.corpus_axis import DASH, along, axis_of, period_key, plural
from core.io.reader import Corpus, Document
from core.profiler.executor import DETAIL_PREFIX, _with_details


def _doc(doc_id: int, name: str, when: date | None = None, **fields: str) -> Document:
    return Document(
        doc_id=doc_id,
        path=Path(name),
        text="Some words here.",
        date=when,
        fields=tuple(sorted(fields.items())),
    )


def _corpus(docs: tuple[Document, ...]) -> Corpus:
    return Corpus(docs=docs, sha256="test")


def _chapters(count: int, **extra: str) -> Corpus:
    return _corpus(tuple(_doc(i, f"ch{i:02d}.txt", Order=str(i), Work="Emma", **extra) for i in range(1, count + 1)))


def test_dated_documents_line_up_in_time() -> None:
    corpus = _corpus((_doc(1, "a.txt", date(1934, 1, 3)), _doc(2, "b.txt", date(1935, 7, 2)), _doc(3, "c.txt")))
    axis = axis_of(corpus)
    assert axis.kind == "time" and axis.noun == "Year"
    assert axis.positions["1"] == 1934.005479  # 3 January: two days into 1934
    assert axis.labels["2"] == "1935-07-02"
    assert "3" not in axis.positions, "an undated document has no place on a time axis"
    assert along(axis.kind, axis.noun) == "over time"


def test_ordered_documents_line_up_by_order_when_nothing_is_dated() -> None:
    axis = axis_of(_chapters(12), noun="Chapter")
    assert axis.kind == "order"
    assert axis.positions["3"] == 3.0 and axis.labels["3"] == "Chapter 3"
    assert along(axis.kind, axis.noun) == "across the chapters"
    periods = axis.periods()
    labels = sorted(set(periods.values()), key=period_key)
    dash = DASH
    assert labels == [f"Chapters {a}{dash}{b}" for a, b in ((1, 3), (4, 6), (7, 9), (10, 12))]
    assert sorted(["1990s", "1940s"], key=period_key) == ["1940s", "1990s"]


def test_order_periods_follow_a_grouping_detail_when_given() -> None:
    axis = axis_of(_chapters(6), noun="Chapter")
    volumes = {str(i): "Volume I" if i <= 3 else "Volume II" for i in range(1, 7)}
    assert axis.periods(grouping=volumes) == volumes


def test_no_axis_when_nothing_places_the_documents() -> None:
    corpus = _corpus((_doc(1, "essay.txt"), _doc(2, "another.txt")))
    axis = axis_of(corpus)
    assert axis.kind == "none" and not axis.placed
    assert axis.periods() == {}
    assert along("none") == ""


def test_a_chosen_axis_is_kept_even_when_it_places_nothing() -> None:
    """So a figure can say "add an order on the Corpus page" rather than silently drawing time."""
    corpus = _corpus((_doc(1, "a.txt", date(1934, 1, 3)), _doc(2, "b.txt", date(1935, 1, 3))))
    axis = axis_of(corpus, "order", "Session")
    assert axis.kind == "order" and not axis.placed


def test_plural_nouns() -> None:
    assert [plural(n) for n in ("Chapter", "Diary entry", "Day", "Series")] == [
        "Chapters",
        "Diary entries",
        "Days",
        "Series",
    ]


def test_every_per_document_table_gets_position_order_and_details() -> None:
    corpus = _chapters(3, Narrator="Emma")
    frame = pd.DataFrame({"Document ID": ["1", "2", "3"], "Document": ["ch01", "ch02", "ch03"], "Score": [1, 2, 3]})
    frames, notes = _with_details({"t.csv": frame}, corpus, axis_of(corpus, noun="Chapter"))
    out = frames["t.csv"]
    assert list(out.columns) == [
        "Document ID",
        "Document",
        "Order",
        "Position",
        "Position label",
        "Narrator",
        "Work",
        "Score",
    ]
    assert list(out["Position label"]) == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert notes == ()


def test_a_detail_named_like_a_tool_column_is_added_under_a_prefix_once() -> None:
    corpus = _corpus((_doc(1, "a.txt", Score="high"), _doc(2, "b.txt", Score="low")))
    frame = pd.DataFrame({"Document ID": ["1", "2"], "Score": [0.5, 0.7]})
    frames, notes = _with_details({"a.csv": frame, "b.csv": frame}, corpus, axis_of(corpus))
    assert list(frames["a.csv"][DETAIL_PREFIX + "Score"]) == ["high", "low"]
    assert list(frames["a.csv"]["Score"]) == [0.5, 0.7], "the tool's own column is untouched"
    assert len(notes) == 1 and notes[0].code == "DETAILS_COLUMN_RENAMED"


def test_a_detail_the_table_already_carries_is_not_repeated() -> None:
    """A comparison's own Side column and the Side detail are the same values."""
    corpus = _corpus((_doc(1, "a.txt", Side="A"), _doc(2, "b.txt", Side="B")))
    frame = pd.DataFrame({"Document ID": ["1", "2"], "Side": ["A", "B"]})
    frames, notes = _with_details({"a.csv": frame}, corpus, axis_of(corpus))
    assert list(frames["a.csv"].columns) == ["Document ID", "Side"]
    assert notes == ()


def test_dated_tables_keep_date_and_year_where_they_were() -> None:
    corpus = _corpus((_doc(1, "a.txt", date(1934, 1, 3), Speaker="Ada"), _doc(2, "b.txt", date(1950, 1, 3))))
    frame = pd.DataFrame({"Document ID": ["1", "2"], "Document": ["a", "b"], "Score": [1, 2]})
    out = _with_details({"t.csv": frame}, corpus, axis_of(corpus))[0]["t.csv"]
    assert list(out.columns)[:6] == ["Document ID", "Document", "Date", "Year", "Position", "Position label"]
    assert list(out["Speaker"]) == ["Ada", None]


def test_placement_columns_order_rows_and_are_never_averaged() -> None:
    """Profiled as a score, Position invited "mean Position by Speaker": a number about nothing."""
    from core.insight.profile import ColumnRole, profile_frame

    frame = pd.DataFrame(
        {
            "Position": [1934.005, 1950.2, 1990.7],
            "Position label": ["1934-01-03", "1950-03-15", "1990-09-12"],
            "Order": [1, 2, 3],
            "Score": [0.2, 0.5, 0.9],
        }
    )
    roles = {name: profile.role for name, profile in profile_frame(frame).items()}
    assert roles["Position"] is ColumnRole.DATE
    assert roles["Order"] is ColumnRole.DATE
    assert roles["Position label"] is ColumnRole.IDENTIFIER
    assert roles["Score"] is ColumnRole.PROPORTION


def test_an_unnamed_step_is_a_document() -> None:
    """ "across the orders" and "Orders 1-3" named nothing; until the project names a step, it is a document."""
    for noun in ("", "Order"):
        axis = axis_of(_chapters(6), noun=noun)
        assert axis.noun == "Document"
        assert along(axis.kind, axis.noun) == "across the documents"
        assert axis.labels["2"] == "Document 2"
        assert sorted(set(axis.periods().values()), key=period_key)[0] == f"Documents 1{DASH}2"
