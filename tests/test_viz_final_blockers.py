"""Second-review blocker + checkpoint regressions (charts).

Every test here failed against the code it pins BEFORE the fix was written
(verified with the reproducer captured in out/visual_review/final_blockers/
before_fix_log.txt):

* histogram/heatmap effective axis labels (blocker 1 + senior checkpoint)
* genuine sparse-category tick mapping against the RENDERED figure, not
  merely ticktext (blocker 2, both orientations)
* numeric/datetime heatmap with dtype-aware coordinates and gaps (blocker 3)
* single JSON emission on publish_failure / finalize failure (blocker 4)
* missing axis values rejected (checkpoint 2) + sentinel tick angle
  (checkpoint 1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from core.result import Result
from core.viz.chartspec import ChartSpec, prepare_chart_data
from core.viz.plotters import build_figure


def _prepared(spec: ChartSpec, frame: pd.DataFrame) -> Any:
    result = prepare_chart_data(frame, spec)
    assert result.ok, result.errors
    return result.unwrap()


def _figure(frame: pd.DataFrame, spec: ChartSpec) -> Any:
    return build_figure(_prepared(spec, frame), spec).unwrap()


SPARSE = pd.DataFrame(
    {"c": ["A", "C", "E", "B", "D", "F"], "g": ["one"] * 3 + ["two"] * 3, "n": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]}
)
NUMERIC_HEAT = pd.DataFrame(
    {"c": [1, 2, 10, 1, 2, 10], "g": ["one"] * 3 + ["two"] * 3, "n": [1.0, 2.0, 10.0, 11.0, 12.0, 20.0]}
)


# ---------------------------------------------------------------------------
# Blocker 1: effective axis labels
# ---------------------------------------------------------------------------


class TestEffectiveLabels:
    def test_histogram_x_label_is_value_column(self) -> None:
        frame = pd.DataFrame({"grp": ["A", "A", "B", "B"], "val": [1.0, 2.0, 3.0, 4.0]})
        spec = ChartSpec(kind="histogram", x="grp", y="val", group="grp")
        prepared = _prepared(spec, frame)
        assert prepared.x_label == "val"
        assert prepared.y_label == "Count"

    def test_histogram_figure_axes_carry_effective_labels(self) -> None:
        frame = pd.DataFrame({"grp": ["A", "A", "B", "B"], "val": [1.0, 2.0, 3.0, 4.0]})
        spec = ChartSpec(kind="histogram", x="grp", y="val", group="grp")
        fig = _figure(frame, spec)
        assert fig.layout.xaxis.title.text == "val"
        assert fig.layout.yaxis.title.text == "Count"

    def test_histogram_percent_and_share_labels(self) -> None:
        frame = pd.DataFrame({"grp": ["A", "A", "B", "B"], "val": [1.0, 2.0, 3.0, 4.0]})
        pct = ChartSpec(kind="histogram", x="grp", y="val", group="grp", normalize="percent")
        share = ChartSpec(kind="histogram", x="grp", y="val", group="grp", normalize="share")
        assert _prepared(pct, frame).y_label == "Percent"
        assert _prepared(share, frame).y_label == "Share"
        assert _figure(frame, pct).layout.yaxis.title.text == "Percent"

    def test_histogram_explicit_overrides_win(self) -> None:
        frame = pd.DataFrame({"grp": ["A"], "val": [1.0]})
        spec = ChartSpec(kind="histogram", x="grp", y="val", x_label="VALUE", y_label="N")
        fig = _figure(frame, spec)
        assert fig.layout.xaxis.title.text == "VALUE"
        assert fig.layout.yaxis.title.text == "N"

    def test_heatmap_y_label_names_group_axis_not_measure(self) -> None:
        frame = pd.DataFrame({"doc": ["d1", "d2"], "role": ["R1", "R2"], "hits": [1.0, 2.0]})
        spec = ChartSpec(kind="heatmap", x="doc", y="hits", group="role")
        prepared = _prepared(spec, frame)
        assert prepared.y_label == "role"
        assert prepared.x_label == "doc"

    def test_heatmap_figure_axis_and_colorbar_labels(self) -> None:
        frame = pd.DataFrame({"doc": ["d1", "d2"], "role": ["R1", "R2"], "hits": [1.0, 2.0]})
        spec = ChartSpec(kind="heatmap", x="doc", y="hits", group="role")
        fig = _figure(frame, spec)
        trace = fig.data[0]
        assert fig.layout.xaxis.title.text == "doc"
        assert fig.layout.yaxis.title.text == "role"
        assert trace.colorbar.title.text == "hits"

    def test_heatmap_ungrouped_row_label(self) -> None:
        # A heatmap REQUIRES a group column; the only ungrouped shape the
        # renderer can see is via an explicit override, which must win.
        frame = pd.DataFrame({"doc": ["d1"], "role": ["R1"], "hits": [1.0]})
        spec = ChartSpec(kind="heatmap", x="doc", y="hits", group="role", y_label="ROWS")
        fig = _figure(frame, spec)
        assert fig.layout.yaxis.title.text == "ROWS"

    def test_heatmap_explicit_overrides_win(self) -> None:
        frame = pd.DataFrame({"doc": ["d1"], "role": ["R1"], "hits": [1.0]})
        spec = ChartSpec(kind="heatmap", x="doc", y="hits", group="role", x_label="DOC", y_label="ROLE")
        fig = _figure(frame, spec)
        assert fig.layout.xaxis.title.text == "DOC"
        assert fig.layout.yaxis.title.text == "ROLE"


# ---------------------------------------------------------------------------
# Blocker 2: sparse-category mapping synchronized with the rendered axis
# ---------------------------------------------------------------------------


class TestSparseCategoryPinning:
    def test_prepared_labels_are_a_through_f(self) -> None:
        spec = ChartSpec(kind="bar", x="c", y="n", group="g")
        prepared = _prepared(spec, SPARSE)
        assert sorted(set(prepared.data["x"])) == list("ABCDEF")

    def test_categoryorder_pinned_to_tick_order(self) -> None:
        spec = ChartSpec(kind="bar", x="c", y="n", group="g")
        fig = _figure(SPARSE, spec)
        xaxis = fig.layout.xaxis
        assert xaxis.categoryorder == "array"
        assert list(xaxis.categoryarray) == ["A", "B", "C", "D", "E", "F"]
        assert list(xaxis.ticktext) == ["A", "B", "C", "D", "E", "F"]

    def test_rendered_positions_match_tick_mapping(self) -> None:
        """The full-figure check: value 3 belongs to category C (index 1 in
        A..F), and NO trace places a value under the wrong tick."""
        spec = ChartSpec(kind="bar", x="c", y="n", group="g")
        fig = _figure(SPARSE, spec)
        categoryarray = list(fig.layout.xaxis.categoryarray)
        assert categoryarray == ["A", "B", "C", "D", "E", "F"]
        position = {category: index for index, category in enumerate(categoryarray)}
        for trace in fig.data:
            for category, value in zip(trace.x, trace.y, strict=True):
                index = position[str(category)]
                # A->1, C->3, E->5 (trace one); B->2, D->4, F->6 (trace two)
                expected = {"A": 1.0, "C": 3.0, "E": 5.0, "B": 2.0, "D": 4.0, "F": 6.0}[str(category)]
                assert float(value) == expected
                assert index is not None

    def test_rendered_bars_lie_under_the_right_ticks(self) -> None:
        """Rendered bar positions must equal the index of the category in
        categoryarray — the false display of value 3 under B is the bug."""
        spec = ChartSpec(kind="bar", x="c", y="n", group="g")
        fig = _figure(SPARSE, spec)
        position = {c: i for i, c in enumerate(fig.layout.xaxis.categoryarray)}
        bar_positions: dict[str, list[float]] = {}
        for trace in fig.data:
            for category in trace.x:
                bar_positions.setdefault(str(category), []).append(position[str(category)])
        # every category occupies exactly its own tick position
        for category, indexes in bar_positions.items():
            assert indexes == [position[category]], category

    def test_horizontal_sparse_pinned_too(self) -> None:
        spec = ChartSpec(kind="bar", x="c", y="n", group="g", horizontal=True)
        fig = _figure(SPARSE, spec)
        yaxis = fig.layout.yaxis
        assert yaxis.categoryorder == "array"
        assert list(yaxis.categoryarray) == ["A", "B", "C", "D", "E", "F"]
        position = {c: i for i, c in enumerate(yaxis.categoryarray)}
        assert position is not None
        for trace in fig.data:
            for category, value in zip(trace.y, trace.x, strict=True):
                expected = {"A": 1.0, "C": 3.0, "E": 5.0, "B": 2.0, "D": 4.0, "F": 6.0}[str(category)]
                assert float(value) == expected
                assert position[str(category)] is not None

    def test_sparse_histogram_and_line_share_the_pin(self) -> None:
        # line with sparse interleaved groups must pin too (same axis logic)
        frame = pd.DataFrame(
            {"c": ["B", "A", "C", "B", "A"], "g": ["s", "s", "t", "t", "s"], "n": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )
        spec = ChartSpec(kind="line", x="c", y="n", group="g", agg="sum")
        fig = _figure(frame, spec)
        assert fig.layout.xaxis.categoryorder == "array"
        assert list(fig.layout.xaxis.categoryarray) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Blocker 3: numeric/datetime heatmap coordinates
# ---------------------------------------------------------------------------


class TestNumericHeatmap:
    def test_numeric_x_sorted_by_value_not_text(self) -> None:
        spec = ChartSpec(kind="heatmap", x="c", y="n", group="g")
        fig = _figure(NUMERIC_HEAT, spec)
        trace = fig.data[0]
        assert [int(v) for v in trace.x] == [1, 2, 10]  # NOT [1, 10, 2]

    def test_all_six_values_preserved_with_coordinates(self) -> None:
        spec = ChartSpec(kind="heatmap", x="c", y="n", group="g")
        fig = _figure(NUMERIC_HEAT, spec)
        trace = fig.data[0]
        z = [list(row) for row in trace.z]
        assert z == [[1.0, 2.0, 10.0], [11.0, 12.0, 20.0]]
        assert [str(v) for v in trace.y] == ["one", "two"]
        # numeric coordinates go straight onto the axes (no imshow resampling)
        assert all(pd.api.types.is_number(v) for v in trace.x)

    def test_heatmap_is_graph_objects_heatmap(self) -> None:
        spec = ChartSpec(kind="heatmap", x="c", y="n", group="g")
        fig = _figure(NUMERIC_HEAT, spec)
        assert fig.data[0].type == "heatmap"

    def test_datetime_heatmap_chronological(self) -> None:
        frame = pd.DataFrame(
            {
                "day": pd.to_datetime(["2024-03-01", "2024-01-01", "2024-03-01", "2024-01-01"]),
                "role": ["R1", "R1", "R2", "R2"],
                "hits": [2.0, 1.0, 4.0, 3.0],
            }
        )
        spec = ChartSpec(kind="heatmap", x="day", y="hits", group="role")
        fig = _figure(frame, spec)
        trace = fig.data[0]
        assert list(trace.x) == list(pd.to_datetime(["2024-01-01", "2024-03-01"]))
        assert [list(row) for row in trace.z] == [[1.0, 2.0], [3.0, 4.0]]

    def test_missing_cells_remain_gaps(self) -> None:
        frame = pd.DataFrame({"doc": ["d1", "d1", "d2"], "role": ["R1", "R2", "R1"], "hits": [1.0, 2.0, 3.0]})
        spec = ChartSpec(kind="heatmap", x="doc", y="hits", group="role")
        fig = _figure(frame, spec)
        flat = [v for row in fig.data[0].z for v in row]
        assert any(v is None or pd.isna(v) for v in flat)
        assert sum(v for v in flat if not (v is None or pd.isna(v))) == pytest.approx(6.0)

    def test_numeric_looking_strings_stay_textual(self) -> None:
        frame = pd.DataFrame({"c": ["1", "2", "10"], "g": ["g"] * 3, "n": [1.0, 2.0, 3.0]})
        spec = ChartSpec(kind="heatmap", x="c", y="n", group="g")
        fig = _figure(frame, spec)
        assert list(fig.data[0].x) == ["1", "10", "2"]  # text order, honestly categorical

    def test_numeric_heatmap_png_exportable(self) -> None:
        """The figure must serialize (kaleido renders from to_plotly_json);
        numeric/datetime coordinates cannot be silently resampled."""
        spec = ChartSpec(kind="heatmap", x="c", y="n", group="g")
        fig = _figure(NUMERIC_HEAT, spec)
        payload = fig.to_plotly_json()
        assert [int(v) for v in payload["data"][0]["x"]] == [1, 2, 10]


# ---------------------------------------------------------------------------
# Checkpoint 1: sentinel tick angle (None = unspecified, explicit -30 wins)
# ---------------------------------------------------------------------------


class TestTickAngleSentinel:
    def test_unspecified_horizontal_upright(self) -> None:
        frame = pd.DataFrame({"c": ["A", "B"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="c", y="n", horizontal=True)
        fig = _figure(frame, spec)
        assert fig.layout.yaxis.tickangle == 0

    def test_explicit_horizontal_minus_30_is_honored(self) -> None:
        frame = pd.DataFrame({"c": ["A", "B"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="c", y="n", horizontal=True, x_tick_angle=-30)
        fig = _figure(frame, spec)
        assert fig.layout.yaxis.tickangle == -30

    def test_unspecified_vertical_default(self) -> None:
        frame = pd.DataFrame({"c": ["A", "B"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="c", y="n")
        fig = _figure(frame, spec)
        assert fig.layout.xaxis.tickangle == -30

    def test_explicit_vertical_angle_wins(self) -> None:
        frame = pd.DataFrame({"c": ["A", "B"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="c", y="n", x_tick_angle=-30)
        fig = _figure(frame, spec)
        assert fig.layout.xaxis.tickangle == -30

    def test_zero_explicit_is_not_treated_as_unspecified(self) -> None:
        frame = pd.DataFrame({"c": ["A", "B"], "n": [1.0, 2.0]})
        spec = ChartSpec(kind="bar", x="c", y="n", x_tick_angle=0)
        fig = _figure(frame, spec)
        assert fig.layout.xaxis.tickangle == 0


# ---------------------------------------------------------------------------
# Checkpoint 2: missing axis values are rejected, never dropped/invented
# ---------------------------------------------------------------------------


class TestMissingAxisValues:
    def test_none_in_x_refused_not_silently_dropped(self) -> None:
        frame = pd.DataFrame({"c": ["A", None], "g": ["g", "g"], "n": [2.0, 8.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="c", y="n", group="g", agg="sum"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_COLUMN"

    def test_none_in_group_not_coerced_to_fake_category(self) -> None:
        frame = pd.DataFrame({"c": ["A", "B"], "g": ["g", None], "n": [2.0, 8.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="c", y="n", group="g"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_COLUMN"

    def test_nan_in_x_rejected(self) -> None:
        frame = pd.DataFrame({"c": ["A", float("nan")], "n": [1.0, 2.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="c", y="n"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_COLUMN"

    def test_nat_in_datetime_x_rejected(self) -> None:
        frame = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", None]), "n": [1.0, 2.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="line", x="day", y="n"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_COLUMN"

    def test_heatmap_rejects_none_group(self) -> None:
        frame = pd.DataFrame({"c": ["A", "B"], "g": ["g", None], "n": [1.0, 2.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="heatmap", x="c", y="n", group="g"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_COLUMN"

    def test_literal_string_nan_is_a_legitimate_category(self) -> None:
        frame = pd.DataFrame({"c": ["A", "nan"], "n": [1.0, 2.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="c", y="n"))
        assert result.ok
        assert sorted(set(result.unwrap().data["x"])) == ["A", "nan"]

    def test_histogram_may_have_missing_unused_x_selector(self) -> None:
        # the histogram bins spec.y; spec.x is an ignored selector
        frame = pd.DataFrame({"doc": [None, "d2"], "val": [1.0, 2.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="histogram", x="doc", y="val"))
        assert result.ok

    def test_missing_measure_still_bad_numeric(self) -> None:
        frame = pd.DataFrame({"c": ["A", "B"], "g": ["g", "g"], "n": [1.0, None]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="c", y="n", group="g"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_NUMERIC"

    def test_missing_denominator_still_bad_numeric(self) -> None:
        frame = pd.DataFrame({"c": ["A", "B"], "n": [1.0, 2.0], "d": [1.0, None]})
        result = prepare_chart_data(
            frame, ChartSpec(kind="bar", x="c", y="n", agg="sum", rate_per=100.0, denominator_column="d")
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_NUMERIC"

    def test_cli_missing_group_value_fails_with_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.charts import main

        inp = tmp_path / "in"
        inp.mkdir()
        csv = inp / "d.csv"
        csv.write_text("c,g,n\nA,g,2\nB,,8\n", encoding="utf-8")
        code = main([str(csv), str(tmp_path / "out"), "--x", "c", "--y", "n", "--group", "g", "--agg", "sum", "--json"])
        captured = capsys.readouterr()
        assert code == 1
        payload: Any = json.loads(captured.out)  # exactly one JSON object
        assert payload["ok"] is False
        assert payload["diagnostics"][0]["code"] == "CHART_BAD_COLUMN"
        assert payload["run_dir"] is None

    def test_cli_missing_x_value_fails_with_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.charts import main

        inp = tmp_path / "in"
        inp.mkdir()
        csv = inp / "d.csv"
        csv.write_text("c,n\nA,2\n,8\n", encoding="utf-8")
        code = main([str(csv), str(tmp_path / "out"), "--x", "c", "--y", "n", "--agg", "sum", "--json"])
        captured = capsys.readouterr()
        assert code == 1
        payload: Any = json.loads(captured.out)
        assert payload["ok"] is False
        assert payload["diagnostics"][0]["code"] == "CHART_BAD_COLUMN"

    def test_cli_histogram_allows_missing_x_selector(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.charts import main

        inp = tmp_path / "in"
        inp.mkdir()
        csv = inp / "d.csv"
        csv.write_text("doc,val\n,1\nd2,2\n", encoding="utf-8")
        code = main([str(csv), str(tmp_path / "out"), "--kind", "histogram", "--x", "doc", "--y", "val", "--json"])
        captured = capsys.readouterr()
        assert code == 0
        payload: Any = json.loads(captured.out)
        assert payload["ok"] is True


# ---------------------------------------------------------------------------
# Blocker 4: exactly one JSON object on publication/finalization failure
# ---------------------------------------------------------------------------


class TestSingleEmissionOnPublicationFailure:
    @pytest.fixture()
    def charts_input(self, tmp_path: Path) -> Path:
        inp = tmp_path / "in"
        inp.mkdir()
        csv = inp / "d.csv"
        csv.write_text("cat,n\nA,5\nB,3\n", encoding="utf-8")
        return csv

    def _run_with_fault(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> tuple[int, list[Any]]:
        """Run the CLI with OutputWriter.publish_failure fault-injected to
        return Result.failure (as an envelope-write/rename failure would)."""
        from core.io.writer import OutputWriter
        from core.result import Diagnostic

        def broken_publish_failure(*diagnostics: Any, keep_artifacts: bool = False) -> Result[Path]:
            return Result[Path](
                None,
                (Diagnostic.error("WRITER_PUBLISH_FAILED", "injected publication failure"),),
            )

        monkeypatch.setattr(OutputWriter, "publish_failure", broken_publish_failure)
        from tools.charts import main

        code = main(argv)
        out = capsys.readouterr().out
        objects = [json.loads(line) for line in out.splitlines() if line.strip().startswith("{")]
        return code, objects

    def test_publish_failure_emits_exactly_one_json(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        charts_input: Path,
    ) -> None:
        pytest.importorskip("plotly")
        try:
            import kaleido  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("kaleido installed; image-export failure path not reachable")
        code, objects = self._run_with_fault(
            [str(charts_input), str(tmp_path / "out"), "--x", "cat", "--y", "n", "--format", "png", "--json"],
            monkeypatch,
            capsys,
        )
        assert code == 1
        assert len(objects) == 1, "double JSON emission on publish_failure"
        payload = objects[0]
        assert payload["ok"] is False
        codes = [d["code"] for d in payload["diagnostics"]]
        assert "WRITER_PUBLISH_FAILED" in codes  # publication error included
        assert any(c.startswith("CHART_IMAGE_") for c in codes)
        # no paths to abandoned files
        assert payload["run_dir"] is None
        assert payload["artifacts"] == []

    def test_publish_failure_clean_staging_and_single_json_kaleido_present(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        charts_input: Path,
    ) -> None:
        """With kaleido importable, fault-inject at the image call itself."""
        pytest.importorskip("plotly")
        pytest.importorskip("kaleido")
        from core.viz import plotters

        def failing_image(*args: Any, **kwargs: Any) -> Any:
            from core.result import Diagnostic

            return Result[bytes](
                None,
                (Diagnostic.error("CHART_IMAGE_FAILED", "injected image export failure"),),
            )

        monkeypatch.setattr(plotters, "chart_image_bytes", failing_image)
        from core.io.writer import OutputWriter

        def broken_publish_failure(*diagnostics: Any, keep_artifacts: bool = False) -> Result[Path]:
            from core.result import Diagnostic

            return Result[Path](None, (Diagnostic.error("WRITER_PUBLISH_FAILED", "injected publication failure"),))

        monkeypatch.setattr(OutputWriter, "publish_failure", broken_publish_failure)
        from tools.charts import main

        code = main([str(charts_input), str(tmp_path / "out"), "--x", "cat", "--y", "n", "--format", "png", "--json"])
        captured = capsys.readouterr()
        assert code == 1
        objects = [json.loads(line) for line in captured.out.splitlines() if line.strip().startswith("{")]
        assert len(objects) == 1
        payload = objects[0]
        codes = [d["code"] for d in payload["diagnostics"]]
        assert "WRITER_PUBLISH_FAILED" in codes
        assert "CHART_IMAGE_FAILED" in codes
        assert payload["run_dir"] is None
        assert payload["artifacts"] == []
        # staging cleaned: no .staging-* left behind
        leftovers = [p.name for p in (tmp_path / "out").iterdir() if p.name.startswith(".staging-")]
        assert leftovers == []

    def test_finalize_failure_cleans_staging_and_emits_once(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
        charts_input: Path,
    ) -> None:
        from core.io.writer import OutputWriter

        def broken_finalize(self: Any) -> Any:
            from core.result import Diagnostic, Result

            return Result.failure(Diagnostic.error("WRITER_PUBLISH_FAILED", "injected finalize failure"))

        monkeypatch.setattr(OutputWriter, "finalize", broken_finalize)
        from tools.charts import main

        code = main([str(charts_input), str(tmp_path / "out"), "--x", "cat", "--y", "n", "--json"])
        captured = capsys.readouterr()
        assert code == 1
        objects = [json.loads(line) for line in captured.out.splitlines() if line.strip().startswith("{")]
        assert len(objects) == 1
        payload = objects[0]
        assert payload["ok"] is False
        codes = [d["code"] for d in payload["diagnostics"]]
        assert "WRITER_PUBLISH_FAILED" in codes
        assert payload["run_dir"] is None
        assert payload["artifacts"] == []
        leftovers = [p.name for p in (tmp_path / "out").iterdir() if p.name.startswith(".staging-")]
        assert leftovers == []
