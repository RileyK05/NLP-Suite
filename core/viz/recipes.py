"""Which figures each tool's results are read through.

Every tool the suite can run must be accounted for here, and
``tests/test_figure_recipes.py`` fails for any that is not. A tool is
covered when it has registered panels (``core/viz/panels.py``), or when its
result is **read as a table** and this module says why, or both: a
concordance is read line by line, and a dispersion figure of its search term
is still worth offering beside it.

The rule exists because the alternative was the default. A tool with no
recipe fell through to ``core/insight/recommend.py``'s generic rules, which
see only column types, and those drew charts that were correct and
meaningless: the mean of 9,407 unrelated word-pair counts per date, or word
pairs ranked by whichever word comes first in the alphabet. A tool with a
recipe gets its own figures offered by the questions they answer, and no
generic guesses; the generic workbench stays available as an expert tool.

See ``docs/FIGURE_RECIPES.md`` for the per-tool plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.viz.panels import panels_for_tool
from core.viz.panelspec import PanelDefinition

__all__ = ["GENERIC_BUILDERS", "RECIPE_ALIASES", "TABLE_FIRST", "Recipe", "recipe_for"]

#: Tools whose result is read as a table first, and why. Worded for the
#: reader, because the reading shows it in place of chart suggestions.
TABLE_FIRST: dict[str, str] = {
    "convert": "Conversion reports which files were converted and which failed; there is nothing to chart.",
    "filenames": "A before-and-after list of file names, read row by row; there is nothing to chart.",
    "profiler": "A run summary that links to each tool's own results and figures. Unrelated analyses are not combined into one chart.",
    "k_sentences": "First and last sentences are read as text, side by side, with their matching terms.",
    "bert_extract": "An extractive summary is a ranked list of sentences to read, not a quantity to chart.",
    "quote_annotator": "Quotes and their speakers are read as text; counts per speaker follow from the table.",
    "search": "Search hits are read in context, line by line.",
    "table_search": "Matching rows are read as rows.",
    "kwic": "A concordance is read line by line; each line is one occurrence in its context.",
    "doc_duplicates": "Duplicate groups are a list to act on: which copies to keep and which to drop.",
    "spellcheck": "Findings are a list to review word by word before any are corrected.",
    "geocode": "Geocoding results are a status list (matched or not) until the offline map is built.",
    "gis_map": "The map is written as a KML/HTML file; the table lists what was placed and what was not.",
    "svo_map": "Mapped events are listed with their places until the offline map is built.",
    "knowledge_graph": "Entity relations are listed as triples when the graph has too few edges to draw.",
    "coreference": "Mention chains are read as text, so each resolved mention can be checked against its sentence.",
    "word_sense_induction": "The sense split is a baseline (a median split on one vector component, as the module "
    "says), not a clustering; drawn as a map it would look like senses the model found. Read the rows.",
    "csv_stats": "Descriptive statistics are a table of numbers per column, and the correlation is one coefficient "
    "for one pair; to see a column's distribution, chart it with the chart builder.",
    "table_rankcorr": "The result is one coefficient with its confidence interval, not the points behind it; chart the "
    "two columns against each other with the chart builder to see the relationship.",
    "wordcloud_gephi": "The tool's output is itself the figure: a word-cloud image and a GEXF graph to open in Gephi. "
    "There is no table to chart.",
}

#: Tools recorded under another tool's name. The desktop's table workflows
#: write their runs through the engine's own tool (a ``table_wordcloud_gephi``
#: job is recorded as ``wordcloud_gephi``), so the two share one recipe
#: rather than two that could drift.
RECIPE_ALIASES: dict[str, str] = {"table_wordcloud_gephi": "wordcloud_gephi"}

#: Not analyses: the general chart builder and the panel renderer. Their
#: whole purpose is to draw whatever table they are given.
GENERIC_BUILDERS: frozenset[str] = frozenset({"charts", "panels", "table_charts"})


@dataclass(frozen=True, slots=True)
class Recipe:
    """How one tool's results are shown."""

    tool: str
    panels: tuple[PanelDefinition, ...]
    #: Why the table is the primary view; empty when a figure is.
    table_first: str = ""
    generic: bool = False

    @property
    def covered(self) -> bool:
        return bool(self.panels or self.table_first or self.generic)

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "panels": [panel.name for panel in self.panels],
            "table_first": self.table_first,
            "generic": self.generic,
        }


def recipe_for(tool: str) -> Recipe:
    """The recipe for *tool*; uncovered when the tool has none (a gap)."""
    source = RECIPE_ALIASES.get(tool, tool)
    return Recipe(
        tool=tool,
        panels=panels_for_tool(source),
        table_first=TABLE_FIRST.get(source, ""),
        generic=source in GENERIC_BUILDERS,
    )
