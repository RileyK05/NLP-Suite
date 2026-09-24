"""A live answer offers every figure its tool has, and redraws them on request.

Before this the live bench drew one figure, with its defaults, and nothing
could change it; and the reading beside it proposed generic column pairings
for tools that have purpose-built figures. These pin the replacements:

* the session keeps recent answers so a figure can be redrawn with new
  settings, and refuses one whose corpus has been reloaded since;
* each figure is drawn from whichever of the answer's tables feeds it;
* a tool with a recipe gets no generic chart suggestions, and its reading
  names its figures by question;
* the generic timeline is not proposed for a table that pools unrelated items.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from core.insight.recommend import recommend_charts
from desktop_backend.live import LiveResult, Session
from desktop_backend.live_panels import draw_live_panel, live_panel, live_panels_offered
from desktop_backend.server import reading_of


def keyness_frame() -> pd.DataFrame:
    """The first rows of the real keyness table (1930s-60s against the rest)."""
    return pd.DataFrame(
        {
            "Word": ["of", "you", "america", "shall"],
            "Freq Group A (pattern docs)": [9876, 254, 189, 496],
            "Freq Group B (other docs)": [9140, 1579, 1333, 120],
            "G2 (log-likelihood)": [881.9132, 654.052, 610.3407, 310.2],
            "p-value": [0.0, 0.0, 0.0, 0.0],
            "Log Ratio": [0.6114, -2.1341, -2.3153, 1.9],
            "Pct Diff": [52.77, -77.22, -79.91, 70.1],
            "BIC": [868.7568, 640.8956, 597.1843, 297.0],
            "Overrepresented in": ["Group A", "Group B", "Group B", "Group A"],
        }
    )


class _Bench:
    def analyse(self, *_: Any) -> LiveResult:  # pragma: no cover - never called here
        raise AssertionError


def session_on(snapshot_id: str) -> Session:
    session = Session(_Bench())  # type: ignore[arg-type]
    session._snapshot = SimpleNamespace(id=snapshot_id)  # type: ignore[assignment]
    return session


class TestRememberedAnswers:
    def test_an_answer_is_kept_and_found_by_id(self) -> None:
        session = session_on("s1")
        result = LiveResult(tool="keyness", ok=True, frames={"keyness.csv": keyness_frame()})
        answer_id = session.remember(result, {"top-n": 4}, "s1")
        found = session.answer(answer_id)
        assert found is not None and found[0] is result and found[1] == {"top-n": 4}

    def test_an_answer_from_a_reloaded_corpus_is_refused(self) -> None:
        session = session_on("s1")
        answer_id = session.remember(LiveResult(tool="keyness", ok=True), {}, "s1")
        session._snapshot = SimpleNamespace(id="s2")  # type: ignore[assignment]
        assert session.answer(answer_id) is None

    def test_only_the_newest_answers_are_kept(self) -> None:
        session = session_on("s1")
        ids = [session.remember(LiveResult(tool="keyness", ok=True), {}, "s1") for _ in range(6)]
        assert session.answer(ids[0]) is None
        assert session.answer(ids[-1]) is not None


class TestLiveFigures:
    def test_every_figure_the_frames_can_feed_is_offered(self) -> None:
        result = LiveResult(tool="keyness", ok=True, frames={"keyness.csv": keyness_frame()})
        offered = live_panels_offered(result)
        assert [info["name"] for info in offered] == ["keyness_volcano"]
        assert offered[0]["question"]

    def test_a_figure_is_found_in_whichever_table_feeds_it(self) -> None:
        """The first table on the page is not the only one a figure can use."""
        result = LiveResult(
            tool="keyness",
            ok=True,
            frames={"summary.csv": pd.DataFrame({"x": [1]}), "keyness.csv": keyness_frame()},
        )
        assert [info["name"] for info in live_panels_offered(result)] == ["keyness_volcano"]
        first = live_panel(result, {})
        assert first is not None and first["ok"] is True

    def test_a_named_figure_is_drawn_with_the_readers_settings(self) -> None:
        result = LiveResult(tool="keyness", ok=True, frames={"keyness.csv": keyness_frame()})
        drawn = draw_live_panel(result, {}, "keyness_volcano", {"label-top": 1})
        assert drawn["ok"] is True
        assert sum(1 for mark in drawn["marks"] if mark["labelled"]) <= 1

    def test_another_tools_figure_is_refused_by_name(self) -> None:
        result = LiveResult(tool="keyness", ok=True, frames={"keyness.csv": keyness_frame()})
        drawn = draw_live_panel(result, {}, "lda_relevance", {})
        assert drawn["ok"] is False
        assert drawn["diagnostics"][0]["code"] == "PANEL_WRONG_TOOL"


class TestReadings:
    def test_a_tool_with_figures_gets_no_generic_charts(self) -> None:
        reading = reading_of(keyness_frame(), "keyness")
        assert reading["recommended_charts"] == []
        assert [figure["panel"] for figure in reading["figures"]] == ["keyness_volcano"]

    def test_a_table_first_tool_says_why(self) -> None:
        frame = pd.DataFrame({"Left": ["a"], "Keyword": ["freedom"], "Right": ["b"], "Document": ["d.txt"]})
        reading = reading_of(frame, "kwic")
        assert reading["recommended_charts"] == []
        assert "concordance" in reading["table_first"]

    def test_the_general_chart_builder_still_gets_suggestions(self) -> None:
        frame = pd.DataFrame({"Group": ["a", "b", "c", "a"], "Score": [1.0, 2.5, 3.0, 4.0]})
        assert reading_of(frame, "charts")["recommended_charts"]


class TestPooledTimeline:
    def test_no_timeline_over_a_table_of_items_per_document(self) -> None:
        """The ngram_cooccurrence shape: one row per word pair per speech."""
        pairs = pd.DataFrame(
            {
                "Word 1": ["of", "and", "be", "of", "free", "and"],
                "Word 2": ["the", "the", "will", "the", "nation", "be"],
                "Count": [594, 300, 120, 410, 12, 95],
                "Document": ["a.txt"] * 3 + ["b.txt"] * 3,
                "Date": ["1934-01-03"] * 3 + ["1935-01-04"] * 3,
            }
        )
        assert not [r for r in recommend_charts(pairs) if r.spec.kind == "line"]

    def test_a_timeline_over_one_row_per_document(self) -> None:
        documents = pd.DataFrame(
            {
                "Document": ["a.txt", "b.txt", "c.txt"],
                "Date": ["1934-01-03", "1935-01-04", "1936-01-03"],
                "Flesch Reading Ease": [52.1, 48.0, 55.3],
            }
        )
        assert [r for r in recommend_charts(documents) if r.spec.kind == "line"]
