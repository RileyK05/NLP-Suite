"""Every visualization the suite can produce, and where it came from.

This repository is a replacement for NLP Suite 1.6.38, and its visualizations
fall into three groups that an analyst has a real reason to tell apart:

* **Legacy port** -- the same picture the original suite drew, reproduced so
  that work depending on it keeps working. The Excel workbooks and the
  standalone HTML charts matter here: they are what the original suite's users
  actually hand in and present, and "we replaced it with something better" is
  no help when a specific output is expected.
* **Legacy, extended** -- the legacy picture plus something it could not do.
  The extension is named explicitly, because the reason to prefer the new
  version should be a stated advantage rather than novelty.
* **New** -- a visualization the original suite had no equivalent for.

The catalog is declarative so the distinction is machine-readable rather than
folklore: the CLI can list "everything legacy" for an assignment that requires
it, the desktop can badge a chart with its origin, and
``LEGACY_VISUALIZATION_MODULES`` lets a test assert that every visualization
module in the old suite is either carried over here or dropped for a stated
reason. Nothing may be quietly lost.

The canonical legacy inventory lives in ``docs/REPLACEMENT_LEDGER.md`` under
the CAP-VIZ rows; this module is the runtime view of the same facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "LEGACY_VISUALIZATION_MODULES",
    "VISUALIZATIONS",
    "Origin",
    "Visualization",
    "by_origin",
    "get_visualization",
    "legacy_visualizations",
    "validate_catalog",
]


class Origin(Enum):
    """Where a visualization came from."""

    LEGACY_PORT = "legacy"
    LEGACY_EXTENDED = "legacy-extended"
    NEW = "new"

    def __str__(self) -> str:
        return self.value

    @property
    def is_legacy(self) -> bool:
        """True for anything the original suite could also draw."""
        return self in (Origin.LEGACY_PORT, Origin.LEGACY_EXTENDED)


@dataclass(frozen=True, slots=True)
class Visualization:
    """One visualization, its provenance and what it produces."""

    name: str
    title: str
    origin: Origin
    module: str
    tool: str
    produces: tuple[str, ...]
    summary: str
    legacy_module: str = ""
    legacy_note: str = ""
    advantage: str = ""

    def describe(self) -> str:
        """One line for a CLI listing."""
        formats = "/".join(self.produces)
        return f"{self.name:<22} {'[' + str(self.origin) + ']':<19} {formats:<16} {self.summary}"


VISUALIZATIONS: tuple[Visualization, ...] = (
    Visualization(
        name="excel_charts",
        title="Native Excel charts",
        origin=Origin.LEGACY_PORT,
        module="core/viz/charts_excel.py",
        tool="charts",
        produces=("xlsx",),
        summary="Real Excel workbooks with native charts (bar, line, pie, scatter, radar, bubble).",
        legacy_module="charts_Excel_util.py",
        legacy_note=(
            "A 'Data' worksheet holding series and categories plus a first-positioned 'Chart' worksheet "
            "holding a native openpyxl chart, axis titles, category labels pinned low and the legend "
            "dropped for a single series. Reproduced so a workbook opened in Excel looks the way the "
            "original suite's workbooks look. The macro-enabled .xlsm hover path is deliberately not "
            "carried: it required shipping binary templates with embedded VBA."
        ),
    ),
    Visualization(
        name="html_charts",
        title="Interactive HTML charts",
        origin=Origin.LEGACY_EXTENDED,
        module="core/viz/charts.py + core/viz/plotters.py",
        tool="charts",
        produces=("html", "png", "svg", "pdf"),
        summary="Standalone interactive HTML charts, now across thirteen chart kinds.",
        legacy_module="charts_Plotly_util.py, charts_util.py",
        legacy_note="Plotly HTML for a handful of kinds, written per calling module.",
        advantage=(
            "Thirteen kinds rather than a handful, all behind one typed ChartSpec with explicit "
            "aggregation, normalization and rate semantics, so a chart states how its numbers were "
            "combined instead of leaving it implied. Static PNG/SVG/PDF export is available, and the "
            "plotted numbers are always written beside the chart so any figure can be traced to its rows."
        ),
    ),
    Visualization(
        name="wordcloud",
        title="Wordcloud",
        origin=Origin.LEGACY_EXTENDED,
        module="core/viz/wordcloud_gephi.py",
        tool="wordcloud_gephi",
        produces=("html", "png"),
        summary="Weighted wordcloud as interactive HTML and a raster image.",
        legacy_module="wordclouds_util.py",
        legacy_note="Spiral-layout raster wordcloud with optional image masks and per-group recolouring.",
        advantage=(
            "The raster output matches the legacy layout, and an interactive HTML version is produced "
            "alongside it. Layout is seeded, so the same input redraws identically instead of shifting "
            "between runs."
        ),
    ),
    Visualization(
        name="gephi_network",
        title="Gephi network (GEXF)",
        origin=Origin.LEGACY_EXTENDED,
        module="core/viz/wordcloud_gephi.py",
        tool="wordcloud_gephi",
        produces=("gexf",),
        summary="Network export for Gephi, schema- and weight-validated before writing.",
        legacy_module="Gephi_util.py",
        legacy_note="Wrote a GEXF edge list for opening in Gephi.",
        advantage=(
            "The graph is validated before it is written, so a malformed edge list fails with a named "
            "diagnostic here rather than as an unexplained error inside Gephi."
        ),
    ),
    Visualization(
        name="tsne_scatter",
        title="Word-vector t-SNE scatter",
        origin=Origin.LEGACY_PORT,
        module="core/viz/embeddings.py",
        tool="word_embeddings",
        produces=("html",),
        summary="2-D t-SNE projection of word vectors, each point labelled with its word.",
        legacy_module="word2vec_tsne_plot_util.py",
        legacy_note=(
            "A Plotly scatter of the 2-D t-SNE projection with the word beside each point. The legacy "
            "3-D variant is deliberately not carried: 3-D scatters read poorly and the 2-D projection is "
            "the research-useful form."
        ),
    ),
    Visualization(
        name="story_shape",
        title="Story shape",
        origin=Origin.LEGACY_PORT,
        module="core/viz/shapes.py + core/viz/shape_clusters.py",
        tool="shapes",
        produces=("csv", "html"),
        summary="Per-sentence shape metrics across a narrative, plus KMeans shape clustering.",
        legacy_module="shape_of_stories_visualization_util.py, shape_of_stories_vectorizer_util.py",
        legacy_note=(
            "Matplotlib trajectories over resampled per-document shape vectors, with a PCA-variance "
            "heuristic choosing the cluster count."
        ),
    ),
    Visualization(
        name="sankey",
        title="Sankey flow diagram",
        origin=Origin.LEGACY_PORT,
        module="core/viz/shapes.py",
        tool="shapes",
        produces=("html",),
        summary="Flow diagram between two categorical columns.",
        legacy_module="charts_util.py (Sankey path)",
        legacy_note="Plotly Sankey between a source and target column, weighted by a value column.",
    ),
    Visualization(
        name="map_kml",
        title="Google Earth KML",
        origin=Origin.LEGACY_PORT,
        module="core/gis/mapping.py",
        tool="earth",
        produces=("kml",),
        summary="Placemarks and tours for Google Earth from a geocoded table.",
        legacy_module="GIS_KML_util.py, GIS_Google_pin_util.py, GIS_Google_Maps_util.py",
        legacy_note=(
            "KML placemark and tour generation from geocoded rows, opened in Google Earth. Coordinates "
            "are validated against WGS-84 bounds here, so an ungeocoded row is skipped with a warning "
            "rather than silently placed at Null Island."
        ),
    ),
    Visualization(
        name="map_heatmap",
        title="Geographic heatmap",
        origin=Origin.LEGACY_PORT,
        module="core/gis/mapping.py",
        tool="earth",
        produces=("html",),
        summary="Density heatmap over geocoded points as standalone HTML.",
        legacy_module="GIS_folium_util.py, GIS_heatMapIntegration.py",
        legacy_note=(
            "Folium/gmaps heatmaps over geocoded rows. Reproduced without the Google Maps API "
            "dependency, so the output needs no API key and no network call."
        ),
    ),
    Visualization(
        name="html_annotation",
        title="Annotated document HTML",
        origin=Origin.LEGACY_PORT,
        module="core/analysis/html_annotator.py",
        tool="html_annotator",
        produces=("html",),
        summary="The source document rendered with its annotations highlighted in place.",
        legacy_module="parsers_annotators_visualization_util.py, html_annotator_*_util.py",
        legacy_note="Highlighted annotator output over the original text, viewed in a browser.",
    ),
    Visualization(
        name="dispersion_plot",
        title="Lexical dispersion plot",
        origin=Origin.NEW,
        module="core/viz/dispersion_plot.py",
        tool="dispersion",
        produces=("html",),
        summary="Where each word falls across the corpus, one tick per occurrence.",
        advantage=(
            "The original suite had no dispersion visualization of any kind. It answers the question a "
            "frequency table cannot: whether a frequent word is spread through the corpus or concentrated "
            "in one document. Document boundaries are marked, every tick carries its document and "
            "sentence, and the plot renders without plotly installed."
        ),
    ),
    Visualization(
        name="chart_recommendations",
        title="Recommended charts",
        origin=Origin.NEW,
        module="core/insight/recommend.py",
        tool="explain",
        produces=("text", "json"),
        summary="Which charts are worth drawing from a result, and which cannot inform.",
        advantage=(
            "The original suite offered every column against every other column, which invites the "
            "degenerate chart: a quantity plotted against itself produces a perfect line that looks like "
            "a finding and is not. This ranks the pairings that answer something and refuses the ones "
            "that cannot, with the reason stated."
        ),
    ),
)


# Every module in NLP Suite 1.6.38 that produced a visualization, mapped to the
# catalog entry that carries it, or to an explicit reason it was not carried.
# A test asserts this covers the legacy surface, so nothing is lost silently.
LEGACY_VISUALIZATION_MODULES: dict[str, str] = {
    "charts_Excel_util.py": "excel_charts",
    "charts_Plotly_util.py": "html_charts",
    "charts_util.py": "html_charts",
    "wordclouds_util.py": "wordcloud",
    "Gephi_util.py": "gephi_network",
    "word2vec_tsne_plot_util.py": "tsne_scatter",
    "shape_of_stories_visualization_util.py": "story_shape",
    "shape_of_stories_vectorizer_util.py": "story_shape",
    "GIS_KML_util.py": "map_kml",
    "GIS_Google_pin_util.py": "map_kml",
    "GIS_Google_Maps_util.py": "map_kml",
    "GIS_folium_util.py": "map_heatmap",
    "GIS_heatMapIntegration.py": "map_heatmap",
    "parsers_annotators_visualization_util.py": "html_annotation",
    # Not carried, each for a stated reason rather than by omission.
    "GIS_heatMapintegration_ver2.py": "dropped: a duplicate of GIS_heatMapIntegration.py differing only "
    "in casing and an abandoned second attempt; map_heatmap covers the capability",
    "data_visualization_main.py": "dropped: a Tkinter GUI launcher, not a visualization. Its role is "
    "filled by the desktop application and the charts CLI",
    "charts_Excel_main.py": "dropped: a Tkinter GUI launcher for charts_Excel_util; the export itself is "
    "carried as excel_charts",
    "wordclouds_main.py": "dropped: a Tkinter GUI launcher for wordclouds_util; the wordcloud itself is "
    "carried as wordcloud",
    "shape_of_stories_main.py": "dropped: a Tkinter GUI launcher; the analysis and plot are carried as story_shape",
    "GIS_main.py": "dropped: a Tkinter GUI launcher for the GIS family; the outputs are carried as "
    "map_kml and map_heatmap",
    "GIS_symbolic_main.py": "dropped: a Tkinter GUI launcher; symbolic typologies are an analysis "
    "(tools/symbolic.py), not a visualization",
    "GIS_Google_Earth_main.py": "dropped: a Tkinter GUI launcher for the KML path, carried as map_kml",
    "GIS_distance_main.py": "dropped: a Tkinter GUI launcher for a distance computation, not a visualization",
}


def get_visualization(name: str) -> Visualization | None:
    """Look one up by name."""
    for visualization in VISUALIZATIONS:
        if visualization.name == name:
            return visualization
    return None


def by_origin(origin: Origin) -> tuple[Visualization, ...]:
    """Every visualization with this exact origin."""
    return tuple(v for v in VISUALIZATIONS if v.origin is origin)


def legacy_visualizations() -> tuple[Visualization, ...]:
    """Everything the original suite could also draw, ported or extended.

    This is the set to reach for when an assignment or a reviewer expects the
    original suite's output specifically.
    """
    return tuple(v for v in VISUALIZATIONS if v.origin.is_legacy)


def validate_catalog() -> list[str]:
    """Internal consistency problems in the catalog. Empty means consistent."""
    problems: list[str] = []
    seen: set[str] = set()
    for visualization in VISUALIZATIONS:
        if visualization.name in seen:
            problems.append(f"duplicate visualization name: {visualization.name}")
        seen.add(visualization.name)
        if not visualization.produces:
            problems.append(f"{visualization.name} produces nothing")
        if visualization.origin.is_legacy and not visualization.legacy_module:
            problems.append(f"{visualization.name} is legacy but names no legacy module")
        if visualization.origin.is_legacy and not visualization.legacy_note:
            problems.append(f"{visualization.name} is legacy but does not say what the legacy version did")
        if visualization.origin is Origin.LEGACY_PORT and visualization.advantage:
            problems.append(
                f"{visualization.name} is a straight port but claims an advantage; "
                "use LEGACY_EXTENDED if it genuinely adds something"
            )
        if visualization.origin is not Origin.LEGACY_PORT and not visualization.advantage:
            problems.append(f"{visualization.name} claims to improve on the legacy suite but states no advantage")
        if visualization.origin is Origin.NEW and visualization.legacy_module:
            problems.append(f"{visualization.name} is marked new but names a legacy module")
    for module, target in LEGACY_VISUALIZATION_MODULES.items():
        if target.startswith("dropped:"):
            continue
        if get_visualization(target) is None:
            problems.append(f"legacy module {module} maps to unknown visualization {target!r}")
    return problems
