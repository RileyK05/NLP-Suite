"""Chart recommendations and the output readout.

The degenerate-chart cases are pinned hard, because the whole point of the
module is that "corpus size against corpus size" must never be offered as a
chart worth drawing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.insight.profile import ColumnRole, profile_frame
from core.insight.readout import Readout, readout
from core.insight.recommend import degenerate_reasons, recommend_charts


@pytest.fixture
def messy() -> pd.DataFrame:
    """A table with one of every trap in it."""
    return pd.DataFrame(
        {
            "Document": [f"doc{i}.txt" for i in range(12)],
            "Corpus Size": [100 * (i + 1) for i in range(12)],
            "Tokens": [100 * (i + 1) for i in range(12)],  # the same quantity, renamed
            "Double": [200 * (i + 1) for i in range(12)],  # perfectly collinear
            "Rank": list(range(1, 13)),
            "Parser": ["spacy"] * 12,  # constant
            "Readability": [55.1, 61.2, 48.9, 72.3, 50.0, 66.6, 59.4, 44.2, 70.1, 63.3, 52.8, 68.0],
            "Share": [i / 11 for i in range(12)],
            "Notes": [f"a fairly long sentence of free prose about document number {i}" for i in range(12)],
        }
    )


class TestProfile:
    def test_roles_are_assigned_as_documented(self, messy: pd.DataFrame) -> None:
        roles = {name: profile.role for name, profile in profile_frame(messy).items()}
        assert roles["Parser"] is ColumnRole.CONSTANT
        assert roles["Rank"] is ColumnRole.IDENTIFIER
        assert roles["Corpus Size"] is ColumnRole.COUNT
        assert roles["Readability"] is ColumnRole.SCORE
        assert roles["Share"] is ColumnRole.PROPORTION
        assert roles["Notes"] is ColumnRole.TEXT
        assert roles["Document"] is ColumnRole.IDENTIFIER

    def test_an_all_null_column_is_empty_not_constant(self) -> None:
        frame = pd.DataFrame({"a": [None, None], "b": [1, 2]})
        assert profile_frame(frame)["a"].role is ColumnRole.EMPTY

    def test_profiling_never_raises_on_odd_content(self) -> None:
        frame = pd.DataFrame({"mixed": [1, "two", None, [3]], "ok": [1, 2, 3, 4]})
        assert profile_frame(frame)["mixed"].role in (ColumnRole.TEXT, ColumnRole.CATEGORICAL)


class TestDegenerateCharts:
    def test_a_column_against_itself_is_refused(self, messy: pd.DataFrame) -> None:
        """The case that started this: a perfect line that means nothing."""
        reasons = degenerate_reasons(messy, "Corpus Size", "Corpus Size")
        assert reasons
        assert "both axes" in reasons[0]

    def test_the_same_quantity_under_two_names_is_refused(self, messy: pd.DataFrame) -> None:
        reasons = degenerate_reasons(messy, "Corpus Size", "Tokens")
        assert reasons
        assert "same information" in reasons[0]

    def test_perfectly_collinear_columns_are_refused(self, messy: pd.DataFrame) -> None:
        assert degenerate_reasons(messy, "Corpus Size", "Double")

    def test_a_numeric_identifier_x_axis_is_refused(self, messy: pd.DataFrame) -> None:
        """Plotting against Rank plots against row position."""
        reasons = degenerate_reasons(messy, "Rank", "Readability")
        assert reasons
        assert "row identifier" in reasons[0]

    def test_a_text_label_x_axis_is_allowed(self, messy: pd.DataFrame) -> None:
        """A ranked bar chart of top-N items is the point, not a mistake.

        Document names are unique per row, but "readability by document" is
        the most useful chart that table has. Only numeric identifiers are
        meaningless as an axis.
        """
        assert degenerate_reasons(messy, "Document", "Readability") == []

    def test_a_constant_axis_is_refused(self, messy: pd.DataFrame) -> None:
        assert degenerate_reasons(messy, "Parser", "Readability")

    def test_free_text_is_not_an_axis(self, messy: pd.DataFrame) -> None:
        assert degenerate_reasons(messy, "Notes", "Readability")

    def test_an_unknown_column_is_reported(self, messy: pd.DataFrame) -> None:
        assert degenerate_reasons(messy, "Nope", "Readability")

    def test_a_real_relationship_is_allowed(self, messy: pd.DataFrame) -> None:
        assert degenerate_reasons(messy, "Corpus Size", "Readability") == []

    def test_monotonic_but_not_linear_is_still_refused(self) -> None:
        """Rank-identical columns can only produce a rising curve."""
        frame = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [1, 8, 27, 64, 125, 216]})
        reasons = degenerate_reasons(frame, "x", "y")
        assert reasons
        assert "rank" in reasons[0].lower()


class TestRecommendations:
    def test_no_recommendation_is_ever_degenerate(self, messy: pd.DataFrame) -> None:
        """The contract that ties the two halves of the module together."""
        for recommendation in recommend_charts(messy, limit=10):
            spec = recommendation.spec
            if spec.kind == "histogram":
                continue
            assert degenerate_reasons(messy, spec.x, spec.y) == [], recommendation.describe()

    def test_every_recommendation_carries_a_question(self, messy: pd.DataFrame) -> None:
        recommendations = recommend_charts(messy, limit=10)
        assert recommendations
        for recommendation in recommendations:
            assert recommendation.question.endswith("?")
            assert len(recommendation.why) > 20

    def test_tool_specific_recommendations_come_first(self) -> None:
        frame = pd.DataFrame(
            {
                "Pair": [f"w{i} w{i + 1}" for i in range(8)],
                "Word 1": [f"w{i}" for i in range(8)],
                "Word 2": [f"w{i + 1}" for i in range(8)],
                "Co-occurrences": [9, 8, 7, 6, 5, 4, 3, 2],
                "G2 (log-likelihood)": [40.0, 33.0, 21.0, 18.0, 12.0, 9.0, 5.0, 3.0],
                "PMI": [2.0, 9.0, 3.0, 1.0, 8.0, 4.0, 7.0, 6.0],
            }
        )
        first = recommend_charts(frame, tool="collocations", limit=5)[0]
        assert first.spec.y == "G2 (log-likelihood)"

    def test_limit_is_respected(self, messy: pd.DataFrame) -> None:
        assert len(recommend_charts(messy, limit=2)) <= 2

    def test_empty_frame_recommends_nothing(self) -> None:
        assert recommend_charts(pd.DataFrame(), limit=5) == []

    def test_recommendations_are_deterministic(self, messy: pd.DataFrame) -> None:
        first = [r.describe() for r in recommend_charts(messy, limit=5)]
        for _ in range(3):
            assert [r.describe() for r in recommend_charts(messy, limit=5)] == first


class TestReadout:
    def test_empty_result_says_so_and_suggests_the_cause(self) -> None:
        reading = readout(pd.DataFrame(), tool="collocations")
        assert "empty" in reading.headline
        assert reading.cautions
        assert "threshold" in reading.cautions[0]

    def test_empty_result_cautions_are_whole_strings(self) -> None:
        """A caution must be one sentence, not a tuple of its characters."""
        for tool in ("collocations", "dispersion", "keyness", "ngrams", "tfidf", ""):
            reading = readout(pd.DataFrame(), tool=tool)
            assert len(reading.cautions) == 1, tool
            assert len(reading.cautions[0]) > 60, tool

    def test_headline_names_the_tool_and_shape(self, messy: pd.DataFrame) -> None:
        reading = readout(messy, tool="readability")
        assert reading.headline.startswith("readability:")
        assert "12" in reading.headline

    def test_constant_columns_are_called_out(self, messy: pd.DataFrame) -> None:
        text = readout(messy).to_text()
        assert "Parser" in text

    def test_collocations_flags_pmi_on_rare_pairs(self) -> None:
        """The specific trap the collocations table is prone to."""
        frame = pd.DataFrame(
            {
                "Pair": [f"w{i} w{i + 1}" for i in range(10)],
                "Word 1": [f"w{i}" for i in range(10)],
                "Word 2": [f"w{i + 1}" for i in range(10)],
                "Co-occurrences": [1] * 10,
                "G2 (log-likelihood)": list(range(10, 0, -1)),
                "PMI": list(range(10, 0, -1)),
            }
        )
        cautions = " ".join(readout(frame, tool="collocations").cautions)
        assert "PMI" in cautions
        assert "--min-count" in cautions

    def test_collocations_flags_a_dominating_function_word(self) -> None:
        frame = pd.DataFrame(
            {
                "Pair": [f"the w{i}" for i in range(8)],
                "Word 1": ["the"] * 8,
                "Word 2": [f"w{i}" for i in range(8)],
                "Co-occurrences": [20] * 8,
                "G2 (log-likelihood)": list(range(8, 0, -1)),
                "PMI": [1.0] * 8,
            }
        )
        cautions = " ".join(readout(frame, tool="collocations").cautions)
        assert "stopword" in cautions

    def test_dispersion_names_the_concentrated_word(self) -> None:
        frame = pd.DataFrame(
            {
                "Term": ["spread", "clumped"],
                "Frequency": [30, 90],
                "Range": [8, 1],
                "Parts": [8, 8],
                "Gries DP": [0.05, 0.95],
                "DP norm": [0.05, 1.0],
                "Juilland D": [0.95, 0.0],
                "Adjusted Frequency": [28.5, 0.0],
            }
        )
        text = readout(frame, tool="dispersion").to_text()
        assert "clumped" in text
        assert "Raw frequency would have" in text

    def test_readout_never_claims_significance_or_cause(self, messy: pd.DataFrame) -> None:
        """The honesty contract, checked as a contract."""
        banned = ("significant", "significance", "causes", "because of", "proves", "p-value", "p <")
        for tool in ("", "collocations", "dispersion", "readability", "keyness", "tfidf"):
            text = readout(messy, tool=tool).to_text().lower()
            for word in banned:
                assert word not in text, f"{tool!r} readout used {word!r}"

    def test_markdown_and_text_carry_the_same_content(self, messy: pd.DataFrame) -> None:
        reading = readout(messy, tool="readability")
        for observation in reading.observations:
            assert observation in reading.to_text()
            assert observation in reading.to_markdown()

    def test_the_markdown_is_the_shape_the_desktop_renders(self) -> None:
        """``readout.md`` is rendered by the app, not handed to a Markdown library.

        ``desktop/src/Markdown.tsx`` covers exactly the constructs emitted
        here: a headline paragraph, ``**bold**`` section headings alone on a
        line, and ``- `` bullets. Emitting anything else -- a table, a fenced
        block, a numbered list -- would reach the reader as literal text, so
        the shape is pinned on this side where it is produced.
        """
        markdown = Readout(
            headline="A headline.",
            observations=("First.", "Second."),
            cautions=("Careful.",),
            tool="readability",
        ).to_markdown()
        assert markdown == (
            "A headline.\n\n**What the table says**\n\n- First.\n- Second.\n\n**Read with care**\n\n- Careful.\n"
        )
        # The published file adds one heading line on top (``_readout_for``),
        # which the viewer renders as a heading. Nothing else may appear: a
        # table, a fenced block or a numbered list would reach the reader as
        # literal text.
        for line in markdown.splitlines():
            assert not line.startswith(("|", "```", "> ", "1.")), (
                f"{line!r} is Markdown the desktop viewer does not render; extend desktop/src/Markdown.tsx first"
            )

    def test_the_published_file_is_the_same_shape(self) -> None:
        """What the viewer actually receives, heading and all."""
        from core.io.writer import _readout_for

        text = _readout_for(pd.DataFrame({"Word": ["a", "b", "c"], "Count": [3, 2, 1]}), "ngrams", "ngrams.csv")
        assert text is not None
        lines = text.splitlines()
        assert lines[0].startswith("# "), lines[0]
        for line in lines[1:]:
            assert not line.startswith(("|", "```", "> ", "1.")), line

    def test_readout_is_deterministic(self, messy: pd.DataFrame) -> None:
        first = readout(messy, tool="readability").to_text()
        for _ in range(3):
            assert readout(messy, tool="readability").to_text() == first

    def test_counts_are_pluralised_correctly(self) -> None:
        frame = pd.DataFrame(
            {
                "Pair": ["a b", "c d"],
                "Word 1": ["a", "c"],
                "Word 2": ["b", "d"],
                "Co-occurrences": [1, 5],
                "G2 (log-likelihood)": [9.0, 3.0],
                "PMI": [5.0, 1.0],
            }
        )
        text = readout(frame, tool="collocations").to_text()
        assert "seen 1 time)" in text
        assert "1 times" not in text


class TestExplainCli:
    def test_check_exits_three_on_a_degenerate_pairing(self, tmp_path, messy: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
        from tools.explain import main

        path = tmp_path / "t.csv"
        messy.to_csv(path, index=False)
        assert main([str(path), "--check", "Corpus Size", "Corpus Size"]) == 3

    def test_check_exits_zero_on_a_good_pairing(self, tmp_path, messy: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
        from tools.explain import main

        path = tmp_path / "t.csv"
        messy.to_csv(path, index=False)
        assert main([str(path), "--check", "Corpus Size", "Readability"]) == 0

    def test_json_output_is_parseable(self, tmp_path, capsys, messy: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
        import json

        from tools.explain import main

        path = tmp_path / "t.csv"
        messy.to_csv(path, index=False)
        assert main([str(path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "headline" in payload
        assert isinstance(payload["recommended_charts"], list)

    def test_a_missing_file_fails_cleanly(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from tools.explain import main

        assert main([str(tmp_path / "nope.csv")]) == 1

    def test_tool_is_inferred_from_a_run_directory(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from tools.explain import _infer_tool

        path = tmp_path / "collocations__2026-01-01T00-00-00-000+00-00-abc123" / "collocations.csv"
        assert _infer_tool(path) == "collocations"


class TestAutomaticReadoutArtifact:
    """Every run writes a plain-language reading beside its table."""

    @staticmethod
    def _writer(tmp_path, tool: str = "collocations"):  # type: ignore[no-untyped-def]
        from core.io.writer import OutputWriter

        return OutputWriter(tmp_path / "out", tool, {})

    def test_a_run_publishes_readout_markdown(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        writer = self._writer(tmp_path)
        writer.write_table(pd.DataFrame({"Term": ["a", "b"], "Frequency": [3, 1]}), "t.csv")
        envelope = writer.finalize().unwrap()
        paths = [artifact.path for artifact in envelope.artifacts]
        assert "readout.md" in paths
        assert (writer.run_dir / "readout.md").is_file()

    def test_the_readout_names_the_tool_and_table(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        writer = self._writer(tmp_path, tool="dispersion")
        writer.write_table(pd.DataFrame({"Term": ["a"], "Frequency": [1]}), "dispersion.csv")
        writer.finalize().unwrap()
        text = (writer.run_dir / "readout.md").read_text(encoding="utf-8")
        assert text.startswith("# dispersion - dispersion.csv")

    def test_only_the_first_table_produces_a_readout(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Multi-artifact tools get one reading, of their primary result."""
        writer = self._writer(tmp_path)
        writer.write_table(pd.DataFrame({"a": [1, 2]}), "first.csv")
        writer.write_table(pd.DataFrame({"b": [3, 4]}), "second.csv")
        envelope = writer.finalize().unwrap()
        assert [artifact.path for artifact in envelope.artifacts].count("readout.md") == 1

    def test_a_run_with_no_table_has_no_readout(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        writer = self._writer(tmp_path)
        writer.write_text("hello", "notes.txt", kind="text")
        envelope = writer.finalize().unwrap()
        assert "readout.md" not in [artifact.path for artifact in envelope.artifacts]

    def test_a_failing_readout_never_fails_the_run(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
        """The numbers are the artifact that matters; this is a convenience."""
        import core.io.writer as writer_module

        def explode(*_args: object, **_kwargs: object) -> str:
            raise RuntimeError("readout blew up")

        monkeypatch.setattr(writer_module, "_readout_for", explode)
        writer = self._writer(tmp_path)
        written = writer.write_table(pd.DataFrame({"a": [1]}), "t.csv")
        assert written.ok
        assert writer.finalize().ok

    def test_readout_content_matches_the_engine(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        frame = pd.DataFrame({"Term": ["a", "b", "c"], "Frequency": [5, 3, 1]})
        writer = self._writer(tmp_path, tool="dispersion")
        writer.write_table(frame, "dispersion.csv")
        writer.finalize().unwrap()
        text = (writer.run_dir / "readout.md").read_text(encoding="utf-8")
        assert readout(frame, tool="dispersion").to_markdown() in text


class TestTheReadingDescribesATableWithSomethingInIt:
    """A tool that publishes an empty table beside a populated one.

    ``clause_svo`` writes ``clauses.csv`` then ``svo.csv``. On a corpus with no
    tagged clauses the first is empty, and the reading attached to it -- so a
    run that found three subject-verb-object triples published "The result is
    empty: no rows met the settings used for this run", which is false about
    the run and sends the analyst off to loosen thresholds that were never the
    problem.
    """

    @staticmethod
    def _writer(tmp_path, tool: str = "clause_svo"):  # type: ignore[no-untyped-def]
        from core.io.writer import OutputWriter

        return OutputWriter(tmp_path / "out", tool, {})

    @staticmethod
    def _readout(writer) -> str:  # type: ignore[no-untyped-def]
        return (writer.run_dir / "readout.md").read_text(encoding="utf-8")

    def test_an_empty_first_table_does_not_claim_the_reading(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        writer = self._writer(tmp_path)
        writer.write_table(pd.DataFrame(columns=["Clause", "Tag"]), "clauses.csv")
        writer.write_table(pd.DataFrame({"Subject": ["Markets"], "Verb": ["reported"]}), "svo.csv")
        writer.finalize().unwrap()
        text = self._readout(writer)
        assert "svo.csv" in text
        assert "The result is empty" not in text

    def test_an_all_empty_run_still_says_it_is_empty(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The one case where "the result is empty" is the answer."""
        writer = self._writer(tmp_path)
        writer.write_table(pd.DataFrame(columns=["Clause", "Tag"]), "clauses.csv")
        writer.write_table(pd.DataFrame(columns=["Subject", "Verb"]), "svo.csv")
        writer.finalize().unwrap()
        assert "The result is empty" in self._readout(writer)

    def test_the_first_populated_table_wins_not_the_last(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        writer = self._writer(tmp_path)
        writer.write_table(pd.DataFrame({"Subject": ["Markets"]}), "svo.csv")
        writer.write_table(pd.DataFrame({"Other": ["x"]}), "later.csv")
        writer.finalize().unwrap()
        text = self._readout(writer)
        assert "svo.csv" in text
        assert "later.csv" not in text

    def test_a_run_with_no_table_at_all_publishes_no_reading(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        writer = self._writer(tmp_path)
        envelope = writer.finalize().unwrap()
        assert not any(artifact.path == "readout.md" for artifact in envelope.artifacts)
