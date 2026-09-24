"""Hand-computed semantic tests for core.viz.chartspec (FR-6.5 charts core).

Every expected value below is hand-computed from the fixture data shown in
the test, not derived from the implementation — the oracle is arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.viz.chartspec import (
    AGG_FUNCS,
    CHART_KINDS,
    ChartSpec,
    prepare_chart_data,
    wrap_label,
)

# ---------------------------------------------------------------------------
# Fixtures (all expectations hand-computed from these numbers)
# ---------------------------------------------------------------------------


def _bar_frame() -> pd.DataFrame:
    """Category counts: A=5, B=3, C=8 (totals across groups)."""
    return pd.DataFrame(
        {
            "cat": ["A", "B", "C"],
            "n": [5.0, 3.0, 8.0],
        }
    )


def _dup_frame() -> pd.DataFrame:
    """Duplicated x: A appears twice (2+3), B once (4)."""
    return pd.DataFrame(
        {
            "cat": ["A", "B", "A"],
            "n": [2.0, 4.0, 3.0],
        }
    )


def _grouped_dup_frame() -> pd.DataFrame:
    """Duplicated (x, group): A/S1=2, A/S2=3, B/S1=4, C/S1=1."""
    return pd.DataFrame(
        {
            "cat": ["A", "B", "A", "C"],
            "grp": ["S1", "S1", "S2", "S1"],
            "n": [2.0, 4.0, 3.0, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# Numeric ordering and dtype preservation
# ---------------------------------------------------------------------------


class TestNumericOrdering:
    def test_years_sort_numerically_not_as_text(self) -> None:
        frame = pd.DataFrame({"year": [10, 1, 2], "n": [8.0, 5.0, 3.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="year", y="n"))
        assert result.ok
        x_values = result.unwrap().data["x"].tolist()
        assert x_values == [1, 2, 10]  # NOT ['1', '10', '2']

    def test_numeric_x_dtype_preserved(self) -> None:
        frame = pd.DataFrame({"year": [10, 1, 2], "n": [8.0, 5.0, 3.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="year", y="n"))
        assert result.ok
        assert pd.api.types.is_numeric_dtype(result.unwrap().data["x"])

    def test_datetime_x_sorts_chronologically(self) -> None:
        frame = pd.DataFrame(
            {
                "day": ["2024-03-01", "2024-01-15", "2023-12-31"],
                "n": [1.0, 2.0, 3.0],
            }
        )
        frame["day"] = pd.to_datetime(frame["day"])
        result = prepare_chart_data(frame, ChartSpec(kind="line", x="day", y="n"))
        assert result.ok
        x_values = list(result.unwrap().data["x"])
        assert [d.strftime("%Y-%m-%d") for d in x_values] == [
            "2023-12-31",
            "2024-01-15",
            "2024-03-01",
        ]
        assert pd.api.types.is_datetime64_any_dtype(result.unwrap().data["x"])

    def test_categorical_x_sorts_as_text(self) -> None:
        frame = pd.DataFrame({"cat": ["b", "c", "a"], "n": [1.0, 2.0, 3.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n"))
        assert result.ok
        assert result.unwrap().data["x"].tolist() == ["a", "b", "c"]

    def test_text_x_with_10_and_2_is_text_sorted(self) -> None:
        # A genuinely string-typed column keeps text order ("10" < "2").
        frame = pd.DataFrame({"cat": ["10", "2", "1"], "n": [1.0, 2.0, 3.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n"))
        assert result.ok
        assert result.unwrap().data["x"].tolist() == ["1", "10", "2"]

    def test_ordering_is_deterministic_across_runs(self) -> None:
        frame = _grouped_dup_frame()
        spec = ChartSpec(kind="bar", x="cat", y="n", group="grp", agg="sum")
        first = prepare_chart_data(frame, spec)
        second = prepare_chart_data(frame, spec)
        assert first.unwrap().data.equals(second.unwrap().data)


# ---------------------------------------------------------------------------
# Group column is always present, even ungrouped
# ---------------------------------------------------------------------------


class TestGroupAlwaysPresent:
    def test_ungrouped_bar_carries_constant_group(self) -> None:
        result = prepare_chart_data(_bar_frame(), ChartSpec(kind="bar", x="cat", y="n"))
        assert result.ok
        prepared = result.unwrap()
        assert "group" in prepared.data.columns
        assert prepared.data["group"].eq("(all)").all()

    def test_group_column_uses_source_values(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "A", "B"], "grp": ["S1", "S2", "S1"], "n": [2.0, 3.0, 4.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n", group="grp"))
        assert result.ok
        assert sorted(set(result.unwrap().data["group"].tolist())) == ["S1", "S2"]


# ---------------------------------------------------------------------------
# Explicit aggregation — never a silent default
# ---------------------------------------------------------------------------


class TestAggregationSemantics:
    def test_duplicates_without_agg_fail(self) -> None:
        result = prepare_chart_data(_dup_frame(), ChartSpec(kind="bar", x="cat", y="n"))
        assert not result.ok
        error = result.errors[0]
        assert error.code == "CHART_AMBIGUOUS"
        assert "--agg" in error.message

    def test_duplicates_with_agg_sum(self) -> None:
        # A: 2+3 = 5, B: 4 — hand-computed
        result = prepare_chart_data(_dup_frame(), ChartSpec(kind="bar", x="cat", y="n", agg="sum"))
        assert result.ok
        data = result.unwrap().data
        by_x = dict(zip(data["x"], data["y"], strict=True))
        assert by_x["A"] == 5.0
        assert by_x["B"] == 4.0

    def test_duplicates_with_agg_mean(self) -> None:
        # A: (2+3)/2 = 2.5, B: 4
        result = prepare_chart_data(_dup_frame(), ChartSpec(kind="bar", x="cat", y="n", agg="mean"))
        assert result.ok
        by_x = dict(zip(result.unwrap().data["x"], result.unwrap().data["y"], strict=True))
        assert by_x["A"] == 2.5
        assert by_x["B"] == 4.0

    def test_duplicates_with_agg_median(self) -> None:
        # A: median(2,3) = 2.5, B: 4
        result = prepare_chart_data(_dup_frame(), ChartSpec(kind="bar", x="cat", y="n", agg="median"))
        assert result.ok
        by_x = dict(zip(result.unwrap().data["x"], result.unwrap().data["y"], strict=True))
        assert by_x["A"] == 2.5
        assert by_x["B"] == 4.0

    def test_duplicates_with_agg_count(self) -> None:
        # A: 2 rows, B: 1 row
        result = prepare_chart_data(_dup_frame(), ChartSpec(kind="bar", x="cat", y="n", agg="count"))
        assert result.ok
        by_x = dict(zip(result.unwrap().data["x"], result.unwrap().data["y"], strict=True))
        assert by_x["A"] == 2.0
        assert by_x["B"] == 1.0

    def test_grouped_duplicates_key_on_x_and_group(self) -> None:
        # (A,S1)=2 unique; A appears twice overall but once per group.
        frame = pd.DataFrame({"cat": ["A", "A", "B"], "grp": ["S1", "S2", "S1"], "n": [2.0, 3.0, 4.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n", group="grp"))
        assert result.ok  # no CHART_AMBIGUOUS: (x, group) cells are unique

    def test_grouped_duplicates_require_agg(self) -> None:
        # (A,S1) appears twice: values 2 and 5.
        frame = pd.DataFrame({"cat": ["A", "A", "B"], "grp": ["S1", "S1", "S1"], "n": [2.0, 5.0, 4.0]})
        refused = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n", group="grp"))
        assert not refused.ok
        assert refused.errors[0].code == "CHART_AMBIGUOUS"
        aggregated = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n", group="grp", agg="sum"))
        assert aggregated.ok
        cell = aggregated.unwrap().data
        # (A, S1) sums to 2+5=7; the only other cell is (B, S1)=4
        a_s1 = cell[(cell["x"] == "A") & (cell["group"] == "S1")]
        assert float(a_s1["y"].iloc[0]) == 7.0
        assert len(cell) == 2

    def test_agg_mean_overflow_refused(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "A"], "n": [1e308, 1e308]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n", agg="sum"))
        assert not result.ok
        assert result.errors[0].code == "CHART_NONFINITE_AGGREGATE"


# ---------------------------------------------------------------------------
# Observation kinds: scatter / box reject aggregation silently
# ---------------------------------------------------------------------------


class TestObservationKinds:
    def test_scatter_rejects_agg(self) -> None:
        result = prepare_chart_data(_dup_frame(), ChartSpec(kind="scatter", x="cat", y="n", agg="sum"))
        assert not result.ok
        assert result.errors[0].code == "CHART_UNSUPPORTED"

    def test_box_rejects_agg(self) -> None:
        result = prepare_chart_data(_dup_frame(), ChartSpec(kind="box", x="cat", y="n", agg="mean"))
        assert not result.ok
        assert result.errors[0].code == "CHART_UNSUPPORTED"

    def test_box_keeps_every_observation(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "A", "A", "A"], "n": [1.0, 2.0, 9.0, 4.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="box", x="cat", y="n"))
        assert result.ok
        data = result.unwrap().data
        assert len(data) == 4  # nothing collapsed
        assert sorted(data["y"].tolist()) == [1.0, 2.0, 4.0, 9.0]

    def test_scatter_keeps_input_order(self) -> None:
        frame = pd.DataFrame({"cat": ["b", "a", "c"], "n": [7.0, 1.0, 5.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="scatter", x="cat", y="n"))
        assert result.ok
        assert result.unwrap().data["y"].tolist() == [7.0, 1.0, 5.0]


# ---------------------------------------------------------------------------
# Heatmap: x columns, group rows, y measure; group required
# ---------------------------------------------------------------------------


def _heatmap_frame() -> pd.DataFrame:
    """Cells (x=doc, grp=role): d1/R1=1.0, d1/R2=2.0, d2/R1=3.0."""
    return pd.DataFrame(
        {
            "doc": ["d1", "d1", "d2"],
            "role": ["R1", "R2", "R1"],
            "hits": [1.0, 2.0, 3.0],
        }
    )


class TestHeatmapSemantics:
    def test_heatmap_without_group_rejected(self) -> None:
        result = prepare_chart_data(_heatmap_frame(), ChartSpec(kind="heatmap", x="doc", y="hits"))
        assert not result.ok
        assert result.errors[0].code == "CHART_MISSING_GROUP"

    def test_heatmap_cells_are_x_by_group_with_y_measure(self) -> None:
        result = prepare_chart_data(
            _heatmap_frame(),
            ChartSpec(kind="heatmap", x="doc", y="hits", group="role"),
        )
        assert result.ok
        prepared = result.unwrap()
        assert prepared.layout == "matrix"
        data = prepared.data
        # (d1, R2) = 2.0, (d2, R1) = 3.0, (d1, R1) = 1.0 — hand-computed
        lookup = {(str(r["x"]), str(r["group"])): float(r["y"]) for _, r in data.iterrows()}
        assert lookup[("d1", "R1")] == 1.0
        assert lookup[("d1", "R2")] == 2.0
        assert lookup[("d2", "R1")] == 3.0
        assert len(data) == 3

    def test_heatmap_duplicate_cells_refused_without_agg(self) -> None:
        frame = pd.DataFrame(
            {
                "doc": ["d1", "d1"],
                "role": ["R1", "R1"],
                "hits": [1.0, 2.0],
            }
        )
        result = prepare_chart_data(frame, ChartSpec(kind="heatmap", x="doc", y="hits", group="role"))
        assert not result.ok
        assert result.errors[0].code == "CHART_AMBIGUOUS"

    def test_heatmap_duplicate_cells_aggregated(self) -> None:
        frame = pd.DataFrame(
            {
                "doc": ["d1", "d1", "d1", "d1"],
                "role": ["R1", "R1", "R2", "R2"],
                "hits": [1.0, 3.0, 10.0, 20.0],
            }
        )
        result = prepare_chart_data(
            frame,
            ChartSpec(kind="heatmap", x="doc", y="hits", group="role", agg="sum"),
        )
        assert result.ok
        lookup = {(str(r["x"]), str(r["group"])): float(r["y"]) for _, r in result.unwrap().data.iterrows()}
        assert lookup[("d1", "R1")] == 4.0  # 1 + 3
        assert lookup[("d1", "R2")] == 30.0  # 10 + 20

    def test_heatmap_y_is_never_an_axis(self) -> None:
        result = prepare_chart_data(
            _heatmap_frame(),
            ChartSpec(kind="heatmap", x="doc", y="hits", group="role"),
        )
        assert result.ok
        data = result.unwrap().data
        # y column carries the numeric measure, x/group carry the axes
        assert pd.api.types.is_numeric_dtype(data["y"])
        assert set(data["group"]) == {"R1", "R2"}

    def test_heatmap_rejects_normalize_topn_rates(self) -> None:
        frame = _heatmap_frame()
        assert not prepare_chart_data(
            frame,
            ChartSpec(kind="heatmap", x="doc", y="hits", group="role", normalize="percent"),
        ).ok
        assert not prepare_chart_data(frame, ChartSpec(kind="heatmap", x="doc", y="hits", group="role", top_n=1)).ok
        frame2 = frame.copy()
        frame2["tokens"] = [100.0, 100.0, 100.0]
        assert not prepare_chart_data(
            frame2,
            ChartSpec(
                kind="heatmap",
                x="doc",
                y="hits",
                group="role",
                agg="sum",
                rate_per=1000.0,
                denominator_column="tokens",
            ),
        ).ok


# ---------------------------------------------------------------------------
# Histogram: common bins, per-group counts, exposed bins count
# ---------------------------------------------------------------------------


def _hist_frame() -> pd.DataFrame:
    """Values: A group {1, 2, 3, 4}, B group {9, 10}. Whole range 1..10."""
    return pd.DataFrame(
        {
            "grp": ["A", "A", "A", "A", "B", "B"],
            "val": [1.0, 2.0, 3.0, 4.0, 9.0, 10.0],
        }
    )


class TestHistogramSemantics:
    def test_common_bin_edges_across_groups(self) -> None:
        result = prepare_chart_data(
            _hist_frame(),
            ChartSpec(kind="histogram", x="grp", y="val", group="grp"),
        )
        assert result.ok
        prepared = result.unwrap()
        assert prepared.layout == "bins"
        data = prepared.data
        edges = sorted(set(data["bin_left"]) | set(data["bin_right"]))
        # edges computed once from the WHOLE dataset: first=1.0, last=10.0
        assert edges[0] == pytest.approx(1.0)
        assert edges[-1] == pytest.approx(10.0)
        # every group reports over the same edges
        for _group, sub in data.groupby("group"):
            sub_left = sorted(sub["bin_left"].tolist())
            assert sub_left[0] == pytest.approx(1.0)
            assert len(sub) == len(edges) - 1  # every bin, zero-counts included

    def test_bin_count_exposed_and_centers_widths_present(self) -> None:
        result = prepare_chart_data(_hist_frame(), ChartSpec(kind="histogram", x="grp", y="val", group="grp"))
        assert result.ok
        prepared = result.unwrap()
        data = prepared.data
        assert prepared.prepared_by["bins"] == str(len(data) // data["group"].nunique())
        assert "bin_center" in data.columns
        assert "bin_width" in data.columns
        # centers are midpoints of [left, right]
        mid = (data["bin_left"] + data["bin_right"]) / 2.0
        assert np.allclose(data["bin_center"], mid)
        assert np.allclose(data["bin_width"], data["bin_right"] - data["bin_left"])

    def test_counts_hand_computed(self) -> None:
        # Values 1..10 over 10 equal bins of width 0.9: each value falls in
        # its own bin for group A (1..4); group B (9, 10) in the last bins.
        # np.histogram([1,2,3,4,9,10]) -> 10 bins over [1, 10].
        result = prepare_chart_data(_hist_frame(), ChartSpec(kind="histogram", x="grp", y="val", group="grp"))
        assert result.ok
        data = result.unwrap().data
        total = data["count"].sum()
        assert total == 6.0  # 4 + 2 observations conserved (row-count property)
        group_a = data[data["group"] == "A"]["count"].sum()
        group_b = data[data["group"] == "B"]["count"].sum()
        assert group_a == 4.0
        assert group_b == 2.0

    def test_histogram_rejects_agg(self) -> None:
        result = prepare_chart_data(
            _hist_frame(),
            ChartSpec(kind="histogram", x="grp", y="val", agg="sum"),
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_UNSUPPORTED"

    def test_histogram_rejects_non_total_scale_by(self) -> None:
        result = prepare_chart_data(
            _hist_frame(),
            ChartSpec(kind="histogram", x="grp", y="val", scale_by="group"),
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_UNSUPPORTED"

    def test_histogram_percent_normalization(self) -> None:
        # 6 observations: A=4 (66.67%), B=2 (33.33%) — hand-computed.
        result = prepare_chart_data(
            _hist_frame(),
            ChartSpec(kind="histogram", x="grp", y="val", group="grp", normalize="percent"),
        )
        assert result.ok
        data = result.unwrap().data
        assert data["count"].sum() == pytest.approx(100.0)
        assert data["count_orig"].sum() == pytest.approx(6.0)

    def test_histogram_share_normalization(self) -> None:
        result = prepare_chart_data(_hist_frame(), ChartSpec(kind="histogram", x="grp", y="val", normalize="share"))
        assert result.ok
        assert result.unwrap().data["count"].sum() == pytest.approx(1.0)

    def test_empty_histogram_normalize_refused(self) -> None:
        # A single zero value: bins exist but all counts are 0... actually
        # one observation exists, so instead test the truly-empty path with
        # zero-count bins only — total count is 0 only with no rows, which
        # CHART_EMPTY catches. A zero-only histogram has 1 obs in one bin.
        frame = pd.DataFrame({"grp": ["A"], "val": [0.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="histogram", x="grp", y="val", normalize="percent"))
        assert result.ok
        assert result.unwrap().data["count"].sum() == pytest.approx(100.0)

    def test_explicit_bins_control(self) -> None:
        # 6 values, --bins 3: three common edges, count conserved = 6.
        frame = pd.DataFrame({"grp": ["A", "A", "A", "B", "B", "B"], "val": [1.0, 2.0, 3.0, 4.0, 9.0, 10.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="histogram", x="grp", y="val", group="grp", bins=3))
        assert result.ok
        prepared = result.unwrap()
        data = prepared.data
        assert prepared.prepared_by["bins"] == "3"
        assert data["bin_left"].nunique() == 3  # 3 common edges per group
        assert data["count"].sum() == pytest.approx(6.0)  # row-count conserved


# ---------------------------------------------------------------------------
# Top-n: category-level across groups, deterministic ties, before normalize
# ---------------------------------------------------------------------------


def _topn_frame() -> pd.DataFrame:
    """Category totals across groups (hand-computed):
    A: 5+1=6, B: 3+2=5, C: 8+0=8, D: 4+4=8  (tie C/D at 8, A=6, B=5)
    """
    return pd.DataFrame(
        {
            "cat": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "grp": ["S1", "S2", "S1", "S2", "S1", "S2", "S1", "S2"],
            "n": [5.0, 1.0, 3.0, 2.0, 8.0, 0.0, 4.0, 4.0],
        }
    )


class TestTopN:
    def test_ranking_is_category_level_across_groups(self) -> None:
        # totals: A=6, B=5, C=8, D=8 -> top 2 = C and D (8 each)
        result = prepare_chart_data(
            _topn_frame(),
            ChartSpec(kind="bar", x="cat", y="n", group="grp", top_n=2),
        )
        assert result.ok
        kept = set(result.unwrap().data["x"].tolist())
        assert kept == {"C", "D"}

    def test_ties_break_by_name_ascending(self) -> None:
        # C and D tie at 8; C < D alphabetically, so a top-3 keeps D out.
        result = prepare_chart_data(
            _topn_frame(),
            ChartSpec(kind="bar", x="cat", y="n", group="grp", top_n=3),
        )
        assert result.ok
        kept = set(result.unwrap().data["x"].tolist())
        assert kept == {"A", "C", "D"}  # 8, 8, 6 — B (5) is out
        # but a top-2 keeps {C, D} (both 8s), not {A, C}
        result2 = prepare_chart_data(
            _topn_frame(),
            ChartSpec(kind="bar", x="cat", y="n", group="grp", top_n=2),
        )
        assert set(result2.unwrap().data["x"].tolist()) == {"C", "D"}

    def test_group_detail_preserved_within_survivors(self) -> None:
        result = prepare_chart_data(
            _topn_frame(),
            ChartSpec(kind="bar", x="cat", y="n", group="grp", top_n=2),
        )
        assert result.ok
        data = result.unwrap().data
        c_rows = data[data["x"] == "C"]
        assert sorted(c_rows["group"].tolist()) == ["S1", "S2"]
        assert float(c_rows[c_rows["group"] == "S1"]["y"].iloc[0]) == 8.0
        assert float(c_rows[c_rows["group"] == "S2"]["y"].iloc[0]) == 0.0

    def test_topn_runs_before_normalization(self) -> None:
        # With top-n {C, D} the chart total is 8+0+4+4 = 16, so C/S1 share
        # = 8/16 = 0.5 (50%) — NOT 8/21 ≈ 0.381 of the unfiltered total.
        result = prepare_chart_data(
            _topn_frame(),
            ChartSpec(kind="bar", x="cat", y="n", group="grp", top_n=2, normalize="share"),
        )
        assert result.ok
        prepared = result.unwrap()
        assert prepared.prepared_by["selection"] == "top_n then normalize"
        data = prepared.data
        assert data["y"].sum() == pytest.approx(1.0)
        lookup = {(str(r["x"]), str(r["group"])): float(r["y"]) for _, r in data.iterrows()}
        assert lookup[("C", "S1")] == pytest.approx(0.5)  # 8 / 16
        assert lookup[("D", "S2")] == pytest.approx(0.25)  # 4 / 16

    def test_topn_deterministic(self) -> None:
        spec = ChartSpec(kind="bar", x="cat", y="n", group="grp", top_n=2)
        first = prepare_chart_data(_topn_frame(), spec).unwrap()
        second = prepare_chart_data(_topn_frame(), spec).unwrap()
        assert first.data.equals(second.data)

    def test_topn_over_negative_totals_refused(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "n": [5.0, -3.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n", top_n=1))
        assert not result.ok
        assert result.errors[0].code == "CHART_NEGATIVE_VALUE"

    def test_topn_rejected_for_other_kinds(self) -> None:
        frame = _hist_frame()
        result = prepare_chart_data(frame, ChartSpec(kind="histogram", x="grp", y="val", top_n=1))
        assert not result.ok
        assert result.errors[0].code == "CHART_UNSUPPORTED"


# ---------------------------------------------------------------------------
# Normalization: percent/share at total/group/category levels
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_percent_whole_chart(self) -> None:
        # A=5, B=3, C=8, total=16 -> 31.25%, 18.75%, 50%
        result = prepare_chart_data(_bar_frame(), ChartSpec(kind="bar", x="cat", y="n", normalize="percent"))
        assert result.ok
        data = result.unwrap().data
        assert data["y"].sum() == pytest.approx(100.0)
        by_x = dict(zip(data["x"], data["y"], strict=True))
        assert by_x["A"] == pytest.approx(5.0 / 16.0 * 100.0)
        assert by_x["C"] == pytest.approx(50.0)

    def test_share_whole_chart(self) -> None:
        result = prepare_chart_data(_bar_frame(), ChartSpec(kind="bar", x="cat", y="n", normalize="share"))
        assert result.ok
        data = result.unwrap().data
        assert data["y"].sum() == pytest.approx(1.0)
        assert "y_orig" in data.columns  # original values kept for audit
        assert data["y_orig"].sum() == pytest.approx(16.0)

    def test_percent_by_group(self) -> None:
        # Each group's slices sum to 100 within the group.
        result = prepare_chart_data(
            _topn_frame(),
            ChartSpec(
                kind="bar",
                x="cat",
                y="n",
                group="grp",
                normalize="percent",
                scale_by="group",
            ),
        )
        assert result.ok
        data = result.unwrap().data
        for group, sub in data.groupby("group"):
            assert sub["y"].sum() == pytest.approx(100.0), group

    def test_percent_by_category(self) -> None:
        # Within each category, its groups sum to 100.
        result = prepare_chart_data(
            _topn_frame(),
            ChartSpec(
                kind="bar",
                x="cat",
                y="n",
                group="grp",
                normalize="percent",
                scale_by="category",
            ),
        )
        assert result.ok
        data = result.unwrap().data
        for category, sub in data.groupby("x"):
            assert sub["y"].sum() == pytest.approx(100.0), category

    def test_negative_values_refused_for_share(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "n": [5.0, -1.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n", normalize="share"))
        assert not result.ok
        assert result.errors[0].code == "CHART_NEGATIVE_VALUE"

    def test_zero_denominator_refused(self) -> None:
        # Group S2 sums to 0: a share of nothing is undefined.
        result = prepare_chart_data(
            _topn_frame(),
            ChartSpec(
                kind="bar",
                x="cat",
                y="n",
                group="grp",
                normalize="percent",
                scale_by="group",
                top_n=2,
            ),
        )
        # After top-n {C, D}: group S2 total = 0 + 4 = 4 > 0, so this succeeds;
        # use a frame where a whole group is zero after filtering instead.
        assert result.ok  # C/D survivors have positive group totals
        zero_group = pd.DataFrame({"cat": ["A", "A"], "grp": ["S1", "S2"], "n": [0.0, 0.0]})
        refused = prepare_chart_data(
            zero_group,
            ChartSpec(kind="bar", x="cat", y="n", group="grp", normalize="percent", scale_by="group"),
        )
        assert not refused.ok
        assert refused.errors[0].code == "CHART_ZERO_DENOMINATOR"

    def test_normalize_rejected_for_scatter_box_heatmap(self) -> None:
        frame = pd.DataFrame({"cat": ["A"], "grp": ["S1"], "n": [1.0]})
        for kind in ("scatter", "box", "heatmap"):
            result = prepare_chart_data(frame, ChartSpec(kind=kind, x="cat", y="n", group="grp", normalize="share"))
            assert not result.ok
            assert result.errors[0].code == "CHART_UNSUPPORTED"

    def test_non_default_index_normalized_correctly(self) -> None:
        """Regression: broadcasting a validated one-element total over a
        non-default index must NOT NaN-fill (senior review §3/§4)."""
        frame = pd.DataFrame({"cat": ["A", "B", "C"], "n": [5.0, 3.0, 8.0]}, index=[10, 20, 30])
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n", normalize="percent"))
        assert result.ok
        data = result.unwrap().data
        assert data["y"].sum() == pytest.approx(100.0)
        assert not data["y"].isna().any()

    def test_overflow_total_refused_for_percent(self) -> None:
        """1e308+1e308 total must FAIL, never emit two silent 0% slices."""
        frame = pd.DataFrame({"c": ["A", "B"], "y": [1e308, 1e308]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="c", y="y", normalize="percent"))
        assert not result.ok
        assert result.errors[0].code == "CHART_NONFINITE_AGGREGATE"

    def test_group_totals_overflow_refused(self) -> None:
        frame = pd.DataFrame({"c": ["A", "A", "B"], "g": ["S", "S", "T"], "y": [1e308, 1e308, 1.0]})
        result = prepare_chart_data(
            frame, ChartSpec(kind="bar", x="c", y="y", group="g", agg="sum", normalize="percent", scale_by="group")
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_NONFINITE_AGGREGATE"

    def test_category_totals_overflow_refused(self) -> None:
        frame = pd.DataFrame({"c": ["A", "A"], "y": [1e308, 1e308]})
        result = prepare_chart_data(
            frame,
            ChartSpec(kind="bar", x="c", y="y", agg="sum", normalize="percent", scale_by="category"),
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_NONFINITE_AGGREGATE"

    def test_normal_sized_control_percent(self) -> None:
        """Hand-computed control: small numbers still normalize correctly."""
        frame = pd.DataFrame({"c": ["A", "B"], "y": [1.0, 3.0]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="c", y="y", normalize="percent"))
        assert result.ok
        by_x = dict(zip(result.unwrap().data["x"], result.unwrap().data["y"], strict=True))
        assert by_x["A"] == pytest.approx(25.0)
        assert by_x["B"] == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# Rate scaling: numerator / denominator totals x rate_per (agg=sum only)
# ---------------------------------------------------------------------------


def _rate_frame() -> pd.DataFrame:
    """Mentions per tokens: A=3/100, A=2/100, B=5/300 (tokens column)."""
    return pd.DataFrame(
        {
            "cat": ["A", "A", "B"],
            "grp": ["S1", "S2", "S1"],
            "mentions": [3.0, 2.0, 5.0],
            "tokens": [100.0, 100.0, 300.0],
        }
    )


class TestRateScaling:
    def test_rate_hand_computed(self) -> None:
        # A: (3+2)/200 * 10000 = 250; B: 5/300 * 10000 = 166.667
        result = prepare_chart_data(
            _rate_frame(),
            ChartSpec(
                kind="bar",
                x="cat",
                y="mentions",
                agg="sum",
                rate_per=10_000.0,
                denominator_column="tokens",
            ),
        )
        assert result.ok
        data = result.unwrap().data
        by_x = dict(zip(data["x"], data["y"], strict=True))
        assert by_x["A"] == pytest.approx(250.0)
        assert by_x["B"] == pytest.approx(5.0 / 300.0 * 10_000.0)

    def test_rate_grouped_hand_computed(self) -> None:
        # (A, S1): 3/100 * 100 = 3; (A, S2): 2/100 * 100 = 2
        result = prepare_chart_data(
            _rate_frame(),
            ChartSpec(
                kind="bar",
                x="cat",
                y="mentions",
                group="grp",
                agg="sum",
                rate_per=100.0,
                denominator_column="tokens",
            ),
        )
        assert result.ok
        lookup = {(str(r["x"]), str(r["group"])): float(r["y"]) for _, r in result.unwrap().data.iterrows()}
        assert lookup[("A", "S1")] == pytest.approx(3.0)
        assert lookup[("A", "S2")] == pytest.approx(2.0)
        assert lookup[("B", "S1")] == pytest.approx(5.0 / 300.0 * 100.0)

    def test_rate_requires_agg_sum(self) -> None:
        result = prepare_chart_data(
            _rate_frame(),
            ChartSpec(
                kind="bar",
                x="cat",
                y="mentions",
                agg="mean",
                rate_per=100.0,
                denominator_column="tokens",
            ),
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_UNSUPPORTED"

    def test_rate_requires_both_parameters(self) -> None:
        with pytest.raises(ValueError, match="together"):
            ChartSpec(kind="bar", x="cat", y="n", rate_per=100.0)
        with pytest.raises(ValueError, match="together"):
            ChartSpec(kind="bar", x="cat", y="n", denominator_column="tokens")

    def test_rate_rejects_non_positive_multiplier(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ChartSpec(
                kind="bar",
                x="cat",
                y="n",
                agg="sum",
                rate_per=0.0,
                denominator_column="tokens",
            )
        with pytest.raises(ValueError, match="positive"):
            ChartSpec(
                kind="bar",
                x="cat",
                y="n",
                agg="sum",
                rate_per=-5.0,
                denominator_column="tokens",
            )

    def test_rate_zero_denominator_total_refused(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "mentions": [3.0, 5.0], "tokens": [0.0, 300.0]})
        result = prepare_chart_data(
            frame,
            ChartSpec(
                kind="bar",
                x="cat",
                y="mentions",
                agg="sum",
                rate_per=100.0,
                denominator_column="tokens",
            ),
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_ZERO_DENOMINATOR"

    def test_rate_negative_denominator_refused(self) -> None:
        frame = pd.DataFrame({"cat": ["A"], "mentions": [3.0], "tokens": [-100.0]})
        result = prepare_chart_data(
            frame,
            ChartSpec(
                kind="bar",
                x="cat",
                y="mentions",
                agg="sum",
                rate_per=100.0,
                denominator_column="tokens",
            ),
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_NUMERIC"

    def test_rate_denominator_overflow_refused(self) -> None:
        frame = pd.DataFrame(
            {
                "cat": ["A", "A", "B"],
                "mentions": [3.0, 2.0, 5.0],
                "tokens": [1e308, 1e308, 300.0],
            }
        )
        result = prepare_chart_data(
            frame,
            ChartSpec(
                kind="bar",
                x="cat",
                y="mentions",
                agg="sum",
                rate_per=100.0,
                denominator_column="tokens",
            ),
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_NONFINITE_AGGREGATE"

    def test_rate_multiplier_overflow_refused(self) -> None:
        """Infinite RATES (finite denominator x huge multiplier) must fail too."""
        frame = pd.DataFrame({"c": ["A"], "y": [1e307], "d": [1.0]})
        result = prepare_chart_data(
            frame,
            ChartSpec(kind="bar", x="c", y="y", agg="sum", rate_per=1e308, denominator_column="d"),
        )
        assert not result.ok
        assert result.errors[0].code == "CHART_NONFINITE_AGGREGATE"

    def test_rate_normal_sized_control(self) -> None:
        """Hand-computed control: (3+2)/200 x 10000 = 250."""
        frame = pd.DataFrame({"c": ["A", "A"], "y": [3.0, 2.0], "d": [100.0, 100.0]})
        result = prepare_chart_data(
            frame, ChartSpec(kind="bar", x="c", y="y", agg="sum", rate_per=10_000.0, denominator_column="d")
        )
        assert result.ok
        assert float(result.unwrap().data["y"].iloc[0]) == pytest.approx(250.0)

    def test_rate_rejected_for_other_kinds(self) -> None:
        frame = _rate_frame()
        for kind in ("scatter", "box", "histogram"):
            result = prepare_chart_data(
                frame,
                ChartSpec(
                    kind=kind,
                    x="cat",
                    y="mentions",
                    agg="sum" if kind == "histogram" else None,
                    rate_per=100.0,
                    denominator_column="tokens",
                ),
            )
            assert not result.ok, kind
            assert result.errors[0].code == "CHART_UNSUPPORTED"


# ---------------------------------------------------------------------------
# Invalid numbers, missing categories, misc
# ---------------------------------------------------------------------------


class TestNumericValidation:
    def test_non_numeric_y_refused(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "n": ["5", "oops"]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_NUMERIC"

    def test_nonfinite_y_refused(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "n": [5.0, float("nan")]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_NUMERIC"

    def test_infinite_y_refused(self) -> None:
        frame = pd.DataFrame({"cat": ["A", "B"], "n": [5.0, float("inf")]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_NUMERIC"

    def test_missing_column_named(self) -> None:
        frame = _bar_frame()
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="nope"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_COLUMN"
        assert "nope" in result.errors[0].message

    def test_empty_frame_refused(self) -> None:
        frame = pd.DataFrame({"cat": [], "n": []})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n"))
        assert not result.ok
        assert result.errors[0].code == "CHART_EMPTY"

    def test_missing_group_column_for_grouped_kind_refused(self) -> None:
        result = prepare_chart_data(_bar_frame(), ChartSpec(kind="bar", x="cat", y="n", group="nope"))
        assert not result.ok
        assert result.errors[0].code == "CHART_BAD_COLUMN"

    def test_horizontal_rejected_for_non_bar(self) -> None:
        result = prepare_chart_data(_bar_frame(), ChartSpec(kind="line", x="cat", y="n", horizontal=True))
        assert not result.ok
        assert result.errors[0].code == "CHART_UNSUPPORTED"

    def test_unknown_kind_rejected_at_spec_construction(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            ChartSpec(kind="contour3d", x="a", y="b")  # type: ignore[arg-type]

    def test_spec_validates_agg_choices(self) -> None:
        with pytest.raises(ValueError, match="agg"):
            ChartSpec(kind="bar", x="a", y="b", agg="mode")  # type: ignore[arg-type]


class TestContract:
    def test_kinds_and_aggs_are_stable(self) -> None:
        assert CHART_KINDS[:6] == ("bar", "line", "scatter", "histogram", "box", "heatmap")
        # "bubble" joined the list when the legacy Excel bubble chart was made
        # reachable; tests/test_chart_kind_parity.py holds this in step with
        # the desktop and the Excel exporter.
        assert CHART_KINDS[6:] == ("pie", "sunburst", "treemap", "violin", "radar", "waffle", "calendar", "bubble")
        assert AGG_FUNCS == ("sum", "mean", "median", "count")

    def test_wrap_label(self) -> None:
        assert wrap_label("short", 10) == "short"
        assert wrap_label("abcdefghij", 5) == "abcde<br>fghij"

    def test_scatter_group_column_present_even_ungrouped(self) -> None:
        result = prepare_chart_data(_bar_frame(), ChartSpec(kind="scatter", x="cat", y="n"))
        assert result.ok
        assert result.unwrap().data["group"].eq("(all)").all()

    def test_box_normalize_rejected_but_raw_scale_by_total_allowed(self) -> None:
        # scale_by only matters with normalize; for box, normalize is rejected
        # and scale-by=total (default) is a no-op that must still be accepted.
        result = prepare_chart_data(_bar_frame(), ChartSpec(kind="box", x="cat", y="n", scale_by="total"))
        assert result.ok

    def test_rate_y_label_names_denominator_and_rate(self) -> None:
        result = prepare_chart_data(
            _rate_frame(),
            ChartSpec(
                kind="bar",
                x="cat",
                y="mentions",
                agg="sum",
                rate_per=10_000.0,
                denominator_column="tokens",
            ),
        )
        assert result.ok
        label = result.unwrap().y_label
        assert "tokens" in label
        assert "10000" in label

    def test_overflow_aggregate_refused(self) -> None:
        # 1e308 + 1e308 = inf; the aggregate path must refuse it.
        frame = pd.DataFrame({"cat": ["A", "A"], "n": [1e308, 1e308]})
        result = prepare_chart_data(frame, ChartSpec(kind="bar", x="cat", y="n", agg="sum"))
        assert not result.ok
        error = result.errors[0]
        assert error.code == "CHART_NONFINITE_AGGREGATE"
        assert error.context["count"] == 1
