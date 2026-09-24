"""The desktop's chart list and the engine's must be the same list.

``desktop/src/VisualizeDialog.tsx`` holds a hand-written copy of the chart
kinds, because the labels and hints are UI copy. Nothing kept it in step with
``core/viz/chartspec.py``, and the copy is on the far side of a language
boundary, so drift is invisible: an extra kind in the desktop is a job the
engine refuses, and a missing one is a capability nobody can reach.

Both happened. ``bubble`` was listed in ``EXCEL_KINDS`` and drawn by the Excel
exporter, but was absent from ``CHART_KINDS`` -- so no ``ChartSpec`` could
carry it, the exporter's bubble branch was unreachable, and the desktop never
offered it. One of the six chart kinds the original suite's Excel GUI produced
was quietly unavailable, while a test asserting ``EXCEL_KINDS`` held all six
passed the whole time.

These tests read the TypeScript rather than trusting it.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from core.result import Result
from core.viz.charts_excel import EXCEL_KINDS
from core.viz.chartspec import CHART_KINDS, PreparedChart

DIALOG = Path(__file__).resolve().parent.parent / "desktop" / "src" / "VisualizeDialog.tsx"


def _kind_entries() -> list[dict[str, object]]:
    """The desktop's CHART_KINDS entries, recovered from the source."""
    source = DIALOG.read_text(encoding="utf-8")
    body = source.split("const CHART_KINDS", 1)[1].split("];", 1)[0]
    entries: list[dict[str, object]] = []
    for block in re.findall(r"\{(.*?)\}", body, flags=re.DOTALL):
        kind = re.search(r'kind:\s*"([^"]+)"', block)
        if kind is None:
            continue
        entries.append(
            {
                "kind": kind.group(1),
                "label": (re.search(r'label:\s*"([^"]+)"', block) or [None, ""])[1],
                "hint": (re.search(r'hint:\s*"([^"]+)"', block) or [None, ""])[1],
                "legacy": bool(re.search(r"legacy:\s*true", block)),
            }
        )
    return entries


@pytest.fixture(scope="module")
def entries() -> list[dict[str, object]]:
    found = _kind_entries()
    assert found, f"recovered no chart kinds from {DIALOG}; the extractor needs updating"
    return found


class TestTheTwoListsAgree:
    def test_the_desktop_offers_every_kind_the_engine_draws(self, entries: list[dict[str, object]]) -> None:
        missing = sorted(set(CHART_KINDS) - {str(e["kind"]) for e in entries})
        assert not missing, f"the engine draws {missing} but the desktop does not offer them"

    def test_the_desktop_offers_nothing_the_engine_refuses(self, entries: list[dict[str, object]]) -> None:
        extra = sorted({str(e["kind"]) for e in entries} - set(CHART_KINDS))
        assert not extra, f"the desktop offers {extra}, which the engine would refuse as an invalid kind"

    def test_no_kind_is_listed_twice(self, entries: list[dict[str, object]]) -> None:
        kinds = [str(e["kind"]) for e in entries]
        assert len(kinds) == len(set(kinds)), kinds


class TestTheLegacyBadgeIsTrue:
    """A "legacy" badge is a claim about the original suite, not decoration."""

    def test_exactly_the_excel_kinds_are_badged(self, entries: list[dict[str, object]]) -> None:
        badged = {str(e["kind"]) for e in entries if e["legacy"]}
        assert badged == set(EXCEL_KINDS), (
            f"badged {sorted(badged)} but the Excel exporter covers {sorted(EXCEL_KINDS)}; "
            "a badge that does not match what xlsx actually produces misleads anyone "
            "choosing a chart for work that must be handed in as Excel"
        )

    def test_every_excel_kind_can_actually_be_built(self) -> None:
        """The defect itself: an exporter kind no ChartSpec could carry."""
        from core.viz.chartspec import ChartSpec

        for kind in EXCEL_KINDS:
            spec = ChartSpec(kind=kind, x="Word", y="Count")  # type: ignore[arg-type]
            assert spec.kind == kind

    def test_bubble_is_reachable_end_to_end(self) -> None:
        """Pinned by name, because this is the one that was broken."""
        from core.viz.chartspec import ChartSpec

        assert "bubble" in CHART_KINDS
        assert "bubble" in EXCEL_KINDS
        assert ChartSpec(kind="bubble", x="Word", y="Count").kind == "bubble"


class TestEveryKindIsDescribed:
    def test_each_entry_has_a_label_and_a_hint(self, entries: list[dict[str, object]]) -> None:
        bare = [e["kind"] for e in entries if not e["label"] or not e["hint"]]
        assert not bare, f"chart kinds with no label or hint: {bare}"

    def test_the_hints_are_distinct(self, entries: list[dict[str, object]]) -> None:
        """Two kinds with the same hint means one of them is not explained."""
        hints = [str(e["hint"]) for e in entries]
        duplicates = sorted({h for h in hints if hints.count(h) > 1})
        assert not duplicates, f"chart kinds sharing a hint: {json.dumps(duplicates)}"


class TestNothingRestatesTheList:
    """The list existed in five places; four of them could drift silently.

    ``CHART_KINDS`` is the engine's tuple, ``ChartKind`` is the Literal mypy
    checks against, and the CLI, the tool registry and the desktop each held a
    copy. The CLI and registry now derive theirs; the other two cannot, so they
    are checked here.
    """

    def test_the_literal_type_matches_the_tuple(self) -> None:
        import typing

        from core.viz.chartspec import ChartKind

        assert set(typing.get_args(ChartKind)) == set(CHART_KINDS)

    def test_the_cli_accepts_every_engine_kind(self) -> None:
        """Behavioural: each kind actually parses, rather than inspecting argparse."""
        from tools.charts import _parse_args

        for kind in CHART_KINDS:
            args = _parse_args(["in.csv", "out", "--kind", kind, "--x", "a", "--y", "b"])
            assert args.kind == kind

    def test_the_cli_still_refuses_a_kind_the_engine_cannot_draw(self) -> None:
        from tools.charts import _parse_args

        with pytest.raises(SystemExit):
            _parse_args(["in.csv", "out", "--kind", "spiral", "--x", "a", "--y", "b"])

    def test_every_spec_that_offers_chart_kinds_offers_all_of_them(self) -> None:
        """Both the CLI registry and the desktop's table workflow declare kinds.

        ``table_charts`` is a separate ``ToolSpec`` living in
        ``desktop_backend/tables.py``, and it is the one the desktop's own
        parameter form reads. It held its own hand-written copy of the list,
        which is how ``bubble`` stayed unreachable from the Visualize page
        even after the engine could draw it.
        """
        from core.profiler.registry import TOOL_REGISTRY
        from desktop_backend.tables import TABLE_TOOLS

        offered = {
            spec.name: tuple(param.choices)
            for spec in (*TOOL_REGISTRY, *TABLE_TOOLS.values())
            for param in spec.params
            if param.name == "kind" and "bar" in param.choices
        }
        assert set(offered) == {"charts", "table_charts"}, (
            f"a new chart-kind surface appeared: {sorted(offered)}; derive it from CHART_KINDS"
        )
        for name, choices in offered.items():
            assert choices == tuple(CHART_KINDS), f"{name} offers {choices}, the engine draws {CHART_KINDS}"

    def test_the_desktop_registry_and_engine_all_agree(self, entries: list[dict[str, object]]) -> None:
        """Adding a kind should be one edit, not five."""
        from core.profiler.registry import TOOL_REGISTRY
        from desktop_backend.tables import TABLE_TOOLS

        spec = next(s for s in TOOL_REGISTRY if s.name == "charts")
        registry = {str(c) for c in next(p for p in spec.params if p.name == "kind").choices}
        table = {str(c) for c in next(p for p in TABLE_TOOLS["table_charts"].params if p.name == "kind").choices}
        desktop = {str(e["kind"]) for e in entries}
        assert registry == table == desktop == set(CHART_KINDS)


class TestNothingCountsTheListByHand:
    """A count is the same restatement problem in miniature.

    The Visualize dialog's "Charts" tile read "13 kinds" while the grid beside
    it offered fourteen, because the number was typed rather than counted.
    """

    def test_the_dialog_does_not_hard_code_how_many_kinds_there_are(self) -> None:
        source = DIALOG.read_text(encoding="utf-8")
        # Comments may name the old number while explaining why it was wrong.
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
        stray = re.findall(r"(?<!\.)\b(\d{1,3})\s+kinds\b", code)
        assert not stray, f"{DIALOG.name} states a chart count of {stray} instead of counting the list"

    def test_the_dialog_counts_the_list_instead(self) -> None:
        source = DIALOG.read_text(encoding="utf-8")
        assert "CHART_KINDS.length" in source, (
            "the kind count should be derived from CHART_KINDS so it cannot fall behind it"
        )


class TestTheDialogRefusesWhatTheEngineRefuses:
    """``--agg`` is invalid for the kinds that draw every row as it stands.

    The dialog defaults the aggregation to "sum", so a kind it forgets to
    exclude is submitted with an aggregation the engine will refuse -- after
    the job has been queued and run. Its ``AGG_REJECTED`` set was declared and
    never consulted; the condition standing in its place listed three of the
    five kinds inline.
    """

    def _engine_rejects_agg(self) -> set[str]:
        """The kinds ``_resolve_semantics`` refuses an aggregation for."""
        import pandas as pd

        from core.viz.chartspec import ChartSpec, prepare_chart_data

        frame = pd.DataFrame(
            {
                "Word": ["a", "b", "c", "d"],
                "Count": [4, 3, 2, 1],
                "Set": ["x", "x", "y", "y"],
            }
        )
        refused = set()
        for kind in CHART_KINDS:
            # A few kinds need a group to be constructible at all; give every
            # spec one and let the semantic pass be the thing that refuses.
            for group in ("Set", None):
                try:
                    spec = ChartSpec(kind=kind, x="Word", y="Count", group=group, agg="sum")
                except ValueError:
                    continue
                outcome = prepare_chart_data(frame, spec)
                if not outcome.ok and any("agg" in d.message for d in outcome.diagnostics):
                    refused.add(kind)
                break
        return refused

    def _desktop_rejects_agg(self) -> set[str]:
        source = DIALOG.read_text(encoding="utf-8")
        body = source.split("const AGG_REJECTED", 1)[1].split("]", 1)[0]
        return set(re.findall(r'"([a-z]+)"', body))

    def test_the_two_sets_are_the_same(self) -> None:
        engine, desktop = self._engine_rejects_agg(), self._desktop_rejects_agg()
        assert engine, "no kind refused an aggregation; the probe needs updating"
        assert desktop == engine, (
            f"the dialog excludes {sorted(desktop)} from aggregation but the engine refuses "
            f"{sorted(engine)}; a kind missing here is submitted with agg='sum' and fails after running"
        )

    def test_the_set_is_actually_used(self) -> None:
        """It was declared and never read for as long as it existed."""
        source = DIALOG.read_text(encoding="utf-8")
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
        assert code.count("AGG_REJECTED") >= 3, (
            "AGG_REJECTED should be consulted where a chart is validated, where the "
            "aggregation control is shown, and where the job is submitted"
        )


class TestNoKindIsImpossibleToDraw:
    """A kind may not refuse ``--agg`` and demand it at the same time.

    ``bubble`` did. It was added to the list of kinds that reject ``--agg``
    (they draw one mark per row, so collapsing rows would change the meaning)
    but not to the dispatch that gives those kinds their own preparation, so it
    fell through to the bar/line path -- which raises ``CHART_AMBIGUOUS`` and
    asks for the very flag the other rule had just refused. Any table with a
    repeated x value made the kind undrawable, with the two diagnostics each
    telling the reader to do what the other forbids.
    """

    def _outcome(self, kind: str, **kwargs: object) -> Result[PreparedChart]:
        """Prepare a chart of ``kind`` over a table with one repeated x value."""
        import pandas as pd

        from core.viz.chartspec import ChartSpec, prepare_chart_data

        frame = pd.DataFrame({"Word": ["a", "a", "b", "c"], "Count": [4, 3, 2, 1], "Set": ["x", "x", "y", "y"]})
        return prepare_chart_data(frame, ChartSpec(kind=kind, x="Word", y="Count", **kwargs))  # type: ignore[arg-type]

    def _refuses_agg(self, kind: str) -> bool:
        outcome = self._outcome(kind, agg="sum", group="Set" if kind in ("sunburst", "treemap", "heatmap") else None)
        return not outcome.ok and any("agg" in d.message for d in outcome.diagnostics)

    def test_a_kind_that_refuses_agg_can_be_drawn_without_one(self) -> None:
        for kind in CHART_KINDS:
            if not self._refuses_agg(kind):
                continue
            outcome = self._outcome(kind)
            codes = [d.code for d in outcome.diagnostics]
            assert "CHART_AMBIGUOUS" not in codes, (
                f"{kind} refuses --agg and then demands it on duplicate x values; "
                "it is undrawable for any table with a repeated category"
            )

    def test_the_observation_kinds_draw_every_row(self) -> None:
        """They keep one row per observation rather than collapsing to categories."""
        from core.viz.chartspec import _OBSERVATION_KINDS

        for kind in _OBSERVATION_KINDS:
            outcome = self._outcome(kind)
            assert outcome.ok, (kind, [d.message for d in outcome.diagnostics])
            assert len(outcome.unwrap().data) == 4, f"{kind} collapsed rows it should have kept"

    def test_bubble_survives_a_repeated_category(self) -> None:
        """Pinned by name: the kind this was found on."""
        outcome = self._outcome("bubble")
        assert outcome.ok, [d.message for d in outcome.diagnostics]


class TestEveryKindDrawsFromACsv:
    """The only input a reader has is a file, so that is where to check.

    Every calendar test built its frame in memory with ``pd.to_datetime``. A
    CSV carries no dtypes, so ``pd.read_csv`` gives a date column back as text
    and ``prepare_chart_data`` refused it with "convert the column to dates
    first" -- which neither the CLI nor the desktop offered any way to do. The
    kind was implemented, covered and unreachable, exactly as ``bubble`` had
    been.
    """

    # What the desktop dialog sends for each kind, mirrored here so this test
    # exercises the combination a reader actually produces.
    AGG_REJECTED = frozenset({"scatter", "box", "violin", "histogram", "bubble"})
    GROUP_REQUIRED = frozenset({"heatmap", "sunburst", "treemap"})
    GROUP_REJECTED = frozenset({"pie", "waffle", "calendar"})

    @pytest.fixture
    def table(self, tmp_path: Path) -> Path:
        path = tmp_path / "observations.csv"
        path.write_text(
            "Word,Count,Set,Day\na,4,x,2024-01-01\na,3,x,2024-01-02\nb,2,y,2024-01-03\nc,1,y,2024-01-04\n",
            encoding="utf-8",
        )
        return path

    @pytest.mark.parametrize("kind", sorted(CHART_KINDS))
    def test_the_kind_draws(self, kind: str, table: Path) -> None:
        import pandas as pd

        from core.viz.chartspec import ChartSpec, coerce_date_axis, prepare_chart_data

        frame = pd.read_csv(table, encoding="utf-8-sig")
        spec = ChartSpec(
            kind=kind,  # type: ignore[arg-type]
            x="Day" if kind == "calendar" else "Word",
            y="Count",
            group=None if kind in self.GROUP_REJECTED or kind not in self.GROUP_REQUIRED else "Set",
            agg=None if kind in self.AGG_REJECTED else "sum",
        )
        dated = coerce_date_axis(frame, spec)
        assert dated.value is not None, [d.message for d in dated.diagnostics]
        prepared = prepare_chart_data(dated.unwrap(), spec)
        assert prepared.ok, f"{kind} cannot be drawn from a CSV: {[d.message for d in prepared.diagnostics]}"

    def test_a_date_column_read_from_a_file_becomes_dates(self, table: Path) -> None:
        import pandas as pd

        from core.viz.chartspec import ChartSpec, coerce_date_axis

        frame = pd.read_csv(table, encoding="utf-8-sig")
        assert not pd.api.types.is_datetime64_any_dtype(frame["Day"]), "the premise: a CSV has no dtypes"
        converted = coerce_date_axis(frame, ChartSpec(kind="calendar", x="Day", y="Count")).unwrap()
        assert pd.api.types.is_datetime64_any_dtype(converted["Day"])

    def test_text_that_is_not_dates_is_named_rather_than_dropped(self, tmp_path: Path) -> None:
        """A silently-NaT row would remove a day from a calendar without saying so."""
        import pandas as pd

        from core.viz.chartspec import ChartSpec, coerce_date_axis

        frame = pd.DataFrame({"Day": ["2024-01-01", "sometime"], "Count": [1, 2]})
        outcome = coerce_date_axis(frame, ChartSpec(kind="calendar", x="Day", y="Count"))
        assert outcome.value is None
        assert "sometime" in outcome.diagnostics[0].message

    def test_other_kinds_are_left_alone(self) -> None:
        """Only the kind that needs dates, and only its x column."""
        import pandas as pd

        from core.viz.chartspec import ChartSpec, coerce_date_axis

        frame = pd.DataFrame({"Day": ["2024-01-01", "2024-01-02"], "Count": [1, 2]})
        untouched = coerce_date_axis(frame, ChartSpec(kind="bar", x="Day", y="Count")).unwrap()
        assert untouched["Day"].dtype == object


class TestTheWorkbenchAgreesWithTheEngine:
    """The interactive chart draws in the browser; publishing draws here.

    ``desktop/src/chartLayout.ts`` renders a page of rows live so that changing
    a column is instant rather than a job submission. That makes it a second
    implementation of the same picture, which is only safe while the two agree
    on what they are drawing and what they will accept.
    """

    LAYOUT = Path(__file__).resolve().parent.parent / "desktop" / "src" / "chartLayout.ts"

    def _list(self, name: str) -> list[str]:
        source = self.LAYOUT.read_text(encoding="utf-8")
        body = source.split(f"export const {name} = [", 1)[1].split("]", 1)[0]
        return re.findall(r'"([^"]+)"', body)

    def test_every_live_kind_is_one_the_engine_can_draw(self) -> None:
        live = self._list("LIVE_KINDS")
        assert live, "recovered no live kinds; the extractor needs updating"
        unknown = sorted(set(live) - set(CHART_KINDS))
        assert not unknown, (
            f"the workbench draws {unknown} live, which the engine cannot draw at all — "
            "so its Publish button would submit a job that is refused"
        )

    def test_the_palette_is_the_engine_palette(self) -> None:
        """A preview and a published chart of the same column share colours."""
        from core.viz.plotters import OKABE_ITO

        assert tuple(self._list("OKABE_ITO")) == tuple(OKABE_ITO), (
            "the workbench's colours have drifted from core/viz/plotters.py"
        )

    def test_publishing_never_sends_an_aggregation_the_engine_refuses(self) -> None:
        """The defect that made five kinds fail, guarded on the live side too."""
        source = self.LAYOUT.read_text(encoding="utf-8")
        takes_agg = set(
            re.findall(r'"([a-z]+)"', source.split("ENGINE_TAKES_AGG = new Set<string>([", 1)[1].split("]", 1)[0])
        )
        refused = {
            kind for kind in self._list("LIVE_KINDS") if kind in {"scatter", "bubble", "box", "violin", "histogram"}
        }
        assert not (takes_agg & refused), (
            f"the workbench would send --agg for {sorted(takes_agg & refused)}, which the engine refuses"
        )

    def test_publishing_never_sends_a_ranking_the_engine_refuses(self) -> None:
        source = self.LAYOUT.read_text(encoding="utf-8")
        takes_top_n = set(
            re.findall(r'"([a-z]+)"', source.split("ENGINE_TAKES_TOP_N = new Set<string>([", 1)[1].split("]", 1)[0])
        )
        # chartspec: top-n applies to bar/line/pie/sunburst/treemap/waffle.
        allowed = {"bar", "line", "pie", "sunburst", "treemap", "waffle"}
        assert takes_top_n <= allowed, (
            f"the workbench would send --top-n for {sorted(takes_top_n - allowed)}, which the engine refuses"
        )
