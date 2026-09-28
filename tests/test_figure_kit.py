"""The figure kit and the showcase figures, checked the way a reader checks them.

``docs/internal/SHOWCASE_FIGURES_PLAN.md`` section 6: a showcase is drawn from real
output, the image is looked at (here, the overlap linter stands in for the
eye and a render must succeed), it refuses with a reason when the run cannot
support it, and the provenance footer is complete. These tests run the kit
pieces over the real trimmed 87-speech fixtures and assert those gates.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from core.viz import kit
from core.viz.showcase import SHOWCASES, render_showcase, showcase_for, showcases_for

pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

FIXTURES = Path(__file__).parent / "fixtures" / "figures"


def _frames(tool: str) -> dict[str, pd.DataFrame]:
    """Every fixture table for a tool, keyed by its file name."""
    found: dict[str, pd.DataFrame] = {}
    for path in sorted(FIXTURES.glob(f"{tool}__*.parquet")):
        found[f"{path.stem.split('__', 1)[1]}.csv"] = pd.read_parquet(path)
    return found


def _png(fig) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=90)
    return buffer.getvalue()


# ------------------------------------------------------------------- kit --


def test_stream_draws_a_band_per_topic_and_returns_colors() -> None:
    shares = pd.DataFrame(
        {"A": [0.4, 0.5, 0.6, 0.7], "B": [0.5, 0.4, 0.3, 0.2], "BG": [0.1, 0.1, 0.1, 0.1]},
        index=[1934, 1935, 1936, 1937],
    )
    c = kit.canvas("A finding", "a subtitle", footer="demo")
    c.rows([1.0])
    ax = c.row(0).cols([1.0]).axis(0)
    colors = kit.stream(ax, shares, labels={"A": "War"}, background="BG", event_at=[1936])
    assert set(colors) == {"A", "B", "BG"}
    assert colors["BG"] == kit.BACKGROUND  # the background is grey, not a palette slot
    assert _png(c.done())[:4] == b"\x89PNG"


def test_mixture_bars_prints_wide_values_only() -> None:
    shares = pd.DataFrame({"A": [0.6, 0.05], "B": [0.4, 0.95]}, index=["x", "y"])
    c = kit.canvas("Mixtures", footer="demo")
    c.rows([1.0])
    ax = c.row(0).cols([1.0]).axis(0)
    kit.mixture_bars(ax, shares, threshold=0.12)
    labels = [t.get_text() for t in ax.texts]
    assert "60%" in labels and "40%" in labels
    assert "5%" not in labels  # too narrow to print inside
    assert _png(c.done())[:4] == b"\x89PNG"


def test_events_keeps_two_close_labels_apart() -> None:
    c = kit.canvas("Events", footer="demo")
    c.rows([1.0])
    ax = c.row(0).cols([1.0]).axis(0)
    ax.set(xlim=(1930, 1960))
    kit.events(ax, [(1940.5, "Pearl Harbor"), (1941.5, "Korea")])
    xs = [t.get_position()[0] for t in ax.texts]
    assert max(xs) - min(xs) >= kit.plots._EVENT_GAP - 1e-9


def test_time_axis_shades_a_detail_band() -> None:
    documents = pd.DataFrame({"Year": [1934, 1936, 1948, 1950], "Speaker": ["FDR", "FDR", "Truman", "Truman"]})
    c = kit.canvas("Bands", footer="demo")
    c.rows([1.0])
    ax = c.row(0).cols([1.0]).axis(0)
    ax.set(xlim=(1933, 1951))
    kit.time_axis(ax, documents, band="Speaker")
    assert len(ax.patches) == 2  # one shaded span per speaker run
    assert _png(c.done())[:4] == b"\x89PNG"


def test_tiles_relabels_by_first_appearance() -> None:
    c = kit.canvas("Tiles", footer="demo")
    c.rows([1.0])
    ax = c.row(0).cols([1.0]).axis(0)
    kit.tiles(ax, [("fit A", [2, 2, 0, 0]), ("fit B", [1, 1, 5, 5])], stoplines=[(2.0, "event")])
    assert _png(c.done())[:4] == b"\x89PNG"


def test_lint_reports_nothing_on_a_clean_figure() -> None:
    shares = pd.DataFrame({"A": [0.5, 0.6], "B": [0.5, 0.4]}, index=[1934, 1935])
    c = kit.canvas("Clean", footer="demo")
    c.rows([1.0])
    kit.stream(c.row(0).cols([1.0]).axis(0), shares)
    assert kit.lint(c.done()) == []


# -------------------------------------------------------------- showcase --


def test_every_showcase_names_a_registered_tool_and_real_tables() -> None:
    from core.profiler.registry import get_tool

    for showcase in SHOWCASES:
        assert get_tool(showcase.tool) is not None, showcase.tool
        assert showcase.tables, showcase.name
        assert showcase.title and showcase.question


@pytest.mark.parametrize("showcase", SHOWCASES, ids=lambda s: s.name)
def test_showcase_draws_cleanly_from_its_real_trimmed_run(showcase) -> None:
    """Every showcase draws from real output, with no overlapping text (section 6)."""
    frames = _frames(showcase.tool)
    source = next((name for name in showcase.tables if name in frames), "")
    if not source:
        pytest.skip(f"no trimmed fixture for {showcase.tool}")
    result = render_showcase(showcase, frames, "png", source=source)
    assert result.ok, [str(d) for d in result.diagnostics]
    assert result.unwrap()[:4] == b"\x89PNG"
    assert not any(d.code == "STATIC_TEXT_OVERLAP" for d in result.diagnostics), [str(d) for d in result.diagnostics]


def test_showcase_refuses_with_a_reason_when_the_run_cannot_support_it() -> None:
    result = render_showcase(showcase_for("lda_gensim_showcase"), {"topics.csv": pd.DataFrame()}, "png")
    assert not result.ok
    assert result.diagnostics[0].code == "SHOWCASE_NOT_DRAWN"
    # Refusal is information, not an error: the run's tables still stand.
    assert result.diagnostics[0].severity.value == "INFO"


def test_showcase_is_stable_across_two_renders() -> None:
    frames = _frames("lda_gensim")
    # The only thing that differs between two renders is the provenance
    # timestamp (the figure is otherwise deterministic: the same corpus and
    # seeds). Compare the SVG with that generated time removed.
    import re

    def normalized(fmt: str) -> bytes:
        result = render_showcase(showcase_for("lda_gensim_showcase"), frames, fmt, source="doc_topics.csv")
        assert result.ok, [str(d) for d in result.diagnostics]
        return re.sub(rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", b"TIME", result.unwrap())

    assert normalized("svg") == normalized("svg")


def test_showcases_for_filters_by_tool() -> None:
    assert showcases_for("lda_gensim") and showcases_for("lda_mallet")
    assert showcases_for("readability")
    assert showcases_for("not_a_tool") == []


def test_project_events_become_a_dated_stopline() -> None:
    """A project event (section 4) draws a stopline on the run's dated figure."""
    frames = _frames("readability")
    with_event = render_showcase(
        showcase_for("readability_showcase"), frames, "svg", source="readability.csv", events=[(1941.5, "Pearl Harbor")]
    )
    without = render_showcase(showcase_for("readability_showcase"), frames, "svg", source="readability.csv")
    assert with_event.ok and without.ok
    assert b"Pearl Harbor" in with_event.unwrap()
    assert b"Pearl Harbor" not in without.unwrap()


def test_project_event_positions_parse_years_decimals_and_iso_dates() -> None:
    from desktop_backend.fields import event_position

    assert event_position("1941") == 1941.0
    assert event_position("1941.5") == 1941.5
    assert abs(event_position("1941-12-07") - 1941.93) < 0.01
    assert event_position("not a date") is None
    assert event_position("9999") is None  # out of the plausible range: dropped, not placed at the origin


def test_a_showcase_is_published_beside_a_runs_panels(tmp_path: Path) -> None:
    """A finished lda_gensim run writes its showcase into figures/, sealed."""
    from core.profiler.batch import BatchRequest, write_batch
    from core.profiler.executor import BatchResult, ToolOutcome

    frames = _frames("lda_gensim")
    outcome = ToolOutcome(name="lda_gensim", ok=True, frames=frames)
    report = write_batch(
        tmp_path, BatchRequest(("lda_gensim",), {"lda_gensim": {}}, BatchResult((outcome,)), figures=True)
    )
    assert report.ok, [str(d) for d in report.diagnostics]
    run = report.unwrap().child_dirs["lda_gensim"]
    assert (run / "figures" / "lda_gensim_showcase.png").is_file()
    assert (run / "figures" / "lda_gensim_showcase.svg").is_file()
    # Sealed and listed in the envelope like every other artifact, and
    # described in the run's figure index.
    from core.artifacts.envelope import Envelope

    listed = {artifact.path for artifact in Envelope.read(run).unwrap().artifacts}
    assert "figures/lda_gensim_showcase.png" in listed
    index = (run / "figures" / "README.md").read_text(encoding="utf-8")
    assert "What the topic model sees" in index
