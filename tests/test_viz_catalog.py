"""CAP-VIZ-10 — visualization provenance, and legacy coverage that cannot silently regress.

The original suite's Excel workbooks and HTML charts are deliverables people
are required to produce, so "we replaced it with something better" is not an
acceptable outcome for any of them. These tests make losing one loud.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.viz.catalog import (
    LEGACY_VISUALIZATION_MODULES,
    VISUALIZATIONS,
    Origin,
    by_origin,
    get_visualization,
    legacy_visualizations,
    validate_catalog,
)

ROOT = Path(__file__).resolve().parent.parent
LEGACY_SRC = ROOT.parent / "NLP-Suite-1.6.38" / "src"


class TestCatalogConsistency:
    def test_catalog_validates(self) -> None:
        assert validate_catalog() == []

    def test_every_named_module_exists_on_disk(self) -> None:
        """A catalog entry pointing at nothing is worse than no entry."""
        missing: list[str] = []
        for visualization in VISUALIZATIONS:
            for part in visualization.module.split("+"):
                path = ROOT / part.strip()
                if not path.is_file():
                    missing.append(f"{visualization.name} -> {part.strip()}")
        assert not missing, f"catalog names modules that do not exist: {missing}"

    def test_names_are_unique(self) -> None:
        names = [v.name for v in VISUALIZATIONS]
        assert len(names) == len(set(names))

    def test_a_straight_port_claims_no_advantage(self) -> None:
        """If it genuinely improves on legacy it must be LEGACY_EXTENDED."""
        for visualization in by_origin(Origin.LEGACY_PORT):
            assert not visualization.advantage, visualization.name

    def test_anything_not_a_straight_port_states_its_advantage(self) -> None:
        for visualization in VISUALIZATIONS:
            if visualization.origin is Origin.LEGACY_PORT:
                continue
            assert len(visualization.advantage) > 40, f"{visualization.name} states no real advantage"

    def test_every_legacy_entry_describes_the_original(self) -> None:
        for visualization in legacy_visualizations():
            assert visualization.legacy_module
            assert len(visualization.legacy_note) > 20, visualization.name

    def test_lookup_by_name(self) -> None:
        assert get_visualization("excel_charts") is not None
        assert get_visualization("nope") is None


class TestLegacyCoverage:
    def test_every_legacy_visualization_module_is_accounted_for(self) -> None:
        """Carried over, or dropped for a stated reason. Never merely absent."""
        for module, target in LEGACY_VISUALIZATION_MODULES.items():
            if target.startswith("dropped:"):
                assert len(target) > len("dropped: ") + 20, f"{module} dropped without a real reason"
                continue
            assert get_visualization(target) is not None, f"{module} maps to unknown {target!r}"

    def test_excel_and_html_chart_exports_are_both_carried(self) -> None:
        """The two outputs the original suite's users actually hand in.

        Excel was once marked dropped in the ledger as "superseded by Plotly".
        It is not superseded: it is a required deliverable, and the exporter
        exists. This test exists so that cannot be quietly reversed again.
        """
        excel = get_visualization("excel_charts")
        html = get_visualization("html_charts")
        assert excel is not None and excel.origin.is_legacy
        assert html is not None and html.origin.is_legacy
        assert "xlsx" in excel.produces
        assert "html" in html.produces

    def test_the_excel_exporter_covers_the_legacy_chart_kinds(self) -> None:
        """The six kinds the legacy Excel GUI offered."""
        from core.viz.charts_excel import EXCEL_KINDS

        assert set(EXCEL_KINDS) == {"bar", "line", "pie", "scatter", "radar", "bubble"}

    def test_most_visualizations_are_legacy(self) -> None:
        """The point of the rebuild is to carry the old suite forward."""
        assert len(legacy_visualizations()) >= len(VISUALIZATIONS) - len(by_origin(Origin.NEW))
        assert len(legacy_visualizations()) > len(by_origin(Origin.NEW))

    @pytest.mark.skipif(not LEGACY_SRC.is_dir(), reason="legacy oracle not present")
    def test_no_legacy_visualization_module_is_unlisted(self) -> None:
        """Scan the legacy tree so a module cannot be forgotten rather than dropped.

        The legacy suite is read-only next to this repo. Any module whose name
        marks it as a visualization producer must appear in the coverage map.
        """
        markers = ("chart", "wordcloud", "gephi", "visualization", "tsne_plot", "heatmap", "kml", "folium")
        found = {
            path.name for path in LEGACY_SRC.glob("*.py") if any(marker in path.name.casefold() for marker in markers)
        }
        unlisted = sorted(found - set(LEGACY_VISUALIZATION_MODULES))
        assert not unlisted, (
            "legacy visualization modules missing from LEGACY_VISUALIZATION_MODULES "
            f"(carry them over or drop them with a reason): {unlisted}"
        )


class TestVisualizationsCli:
    def test_listing_everything_succeeds(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.visualizations import main

        assert main([]) == 0
        assert "excel_charts" in capsys.readouterr().out

    def test_legacy_filter_excludes_new_work(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.visualizations import main

        assert main(["--origin", "legacy"]) == 0
        out = capsys.readouterr().out
        assert "excel_charts" in out
        assert "dispersion_plot" not in out

    def test_new_filter_excludes_legacy(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.visualizations import main

        assert main(["--origin", "new"]) == 0
        out = capsys.readouterr().out
        assert "dispersion_plot" in out
        assert "excel_charts" not in out

    def test_json_output_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        from tools.visualizations import main

        assert main(["--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert {"name", "origin", "is_legacy", "produces"} <= set(payload[0])

    def test_detail_shows_provenance(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.visualizations import main

        assert main(["--name", "wordcloud"]) == 0
        out = capsys.readouterr().out
        assert "wordclouds_util.py" in out
        assert "adds" in out

    def test_unknown_name_fails_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.visualizations import main

        assert main(["--name", "nope"]) == 2
        assert "unknown visualization" in capsys.readouterr().err

    def test_legacy_map_lists_carried_and_dropped(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.visualizations import main

        assert main(["--legacy-map"]) == 0
        out = capsys.readouterr().out
        assert "charts_Excel_util.py" in out
        assert "Not carried" in out

    def test_format_filter(self, capsys: pytest.CaptureFixture[str]) -> None:
        from tools.visualizations import main

        assert main(["--format", "xlsx"]) == 0
        out = capsys.readouterr().out
        assert "excel_charts" in out
        assert "dispersion_plot" not in out
