"""Publication-only figures: each draws on real output, readably.

The bundles (``core/viz/static/bundles.py``) answer questions about a whole
table -- which measures say the same thing, whether a measure is length in
disguise, which topics each decade leaned on, where each speech sits among
the others. Checked on the trimmed real fixtures with the same drawn-text
lint as every panel's publication figure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("seaborn")

from core.viz.panels import best_table
from core.viz.static.bundles import BUNDLES, Bundle, bundles_for, render_bundle

FIXTURES = Path(__file__).parent / "fixtures" / "figures"


def fixture_for(bundle: Bundle) -> pd.DataFrame | None:
    frames = {path.stem: pd.read_parquet(path) for path in FIXTURES.glob(f"{bundle.tool}__*.parquet")}
    chosen = best_table(bundle, [(name, list(f.columns)) for name, f in frames.items() if not f.empty])
    return frames[chosen] if chosen else None


DRAWABLE = [bundle for bundle in BUNDLES if fixture_for(bundle) is not None]


def test_names_are_unique_and_each_says_what_it_answers() -> None:
    names = [bundle.name for bundle in BUNDLES]
    assert len(names) == len(set(names))
    assert all(bundle.question.endswith("?") for bundle in BUNDLES)


def test_the_document_measure_tools_get_a_correlation_matrix() -> None:
    assert [b.name for b in bundles_for("readability")] == [
        "readability_measure_correlations",
        "readability_against_length",
    ]


@pytest.mark.parametrize("bundle", DRAWABLE, ids=lambda b: b.name)
def test_every_bundle_draws_on_real_output_without_overprinted_text(bundle: Bundle) -> None:
    frame = fixture_for(bundle)
    assert frame is not None
    image = render_bundle(bundle, frame, "png", source="fixture.csv", dpi=60)
    assert image.value is not None, [str(d) for d in image.diagnostics]
    assert image.unwrap()[:4] == b"\x89PNG"
    assert [str(d) for d in image.diagnostics] == []


def test_the_correlation_caption_names_the_most_alike_pair() -> None:
    bundle = next(b for b in BUNDLES if b.name == "readability_measure_correlations")
    frame = fixture_for(bundle)
    svg = render_bundle(bundle, frame, "svg", source="readability.csv").unwrap().decode("utf-8")
    assert "Most alike:" in svg
    assert "readability.csv" in svg


def test_a_table_it_cannot_draw_is_a_notice_not_a_crash() -> None:
    bundle = next(b for b in BUNDLES if b.name == "lda_topics_by_decade")
    undated = pd.DataFrame({"Document": ["a.txt", "b.txt"], "Dominant topic": [0, 1], "Topic keywords": ["x", "y"]})
    result = render_bundle(bundle, undated, "png")
    assert result.value is None
    assert result.diagnostics[0].code == "BUNDLE_NOT_DRAWN"
