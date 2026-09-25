"""The panel registry — every tool-specific figure, and the gate in front of it.

One place that knows which panels exist, so the CLI can list them, the
desktop can offer them, and a test can assert that each one is drawable.
:func:`prepare_panel` is the only supported way in: it validates the request
against the panel's declaration before any builder runs, which is what lets
a builder be a plain function of trusted inputs.

Everything a builder would otherwise have to re-check happens here, once:

* the panel exists (``PANEL_UNKNOWN``) and something can draw its shape
  (``PANEL_SHAPE_UNIMPLEMENTED``);
* the frame has rows (``PANEL_EMPTY``) and the columns the panel declared it
  reads (``PANEL_MISSING_COLUMN``) — so a panel pointed at the wrong
  artifact says which column is missing instead of raising ``KeyError`` from
  somewhere inside pandas;
* the parameters are known (``PANEL_UNKNOWN_PARAM``, strict rather than
  ignored: a misspelled parameter that silently does nothing is the quietest
  possible failure), correctly typed and within their declared bounds
  (``PANEL_BAD_PARAM``), with defaults filled in;
* the provenance is assembled from the definition plus what the caller knows,
  so a panel cannot be exported with an anonymous caption by accident
  (``PANEL_PROVENANCE_INCOMPLETE``).

A builder that raises anyway is contained (``PANEL_BUILD_FAILED``) rather
than allowed to take down the caller: panels are the newest and least
exercised surface here, and a broken figure must not break a run that also
produced good ones.

**Adding a panel:** write ``core/viz/panels_<tool>.py`` exporting a
``PanelDefinition`` (copy ``panels_keyness.py``, which is the reference
implementation), then add one line to :data:`PANELS` below. Builders import
``core.viz.panelspec`` and nothing else from this layer; this module is the
only thing that imports builders, which keeps the dependency acyclic and
keeps one list as the single answer to "what panels are there?".
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, Protocol

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_plotters import IMPLEMENTED_SHAPES
from core.viz.panels_collocations import COLLOCATION_STRENGTH
from core.viz.panels_document_measures import DOCUMENT_MEASURES_PANELS
from core.viz.panels_embeddings import (
    WORD2VEC_BERT_NEIGHBOURS,
    WORD2VEC_BERT_TSNE,
    WORD2VEC_BERT_VECTOR_QUERY,
    WORD2VEC_GENSIM_NEIGHBOURS,
    WORD2VEC_GENSIM_TSNE,
    WORD2VEC_GENSIM_VECTOR_QUERY,
)
from core.viz.panels_extras import EXTRAS_PANELS
from core.viz.panels_keyness import KEYNESS_VOLCANO
from core.viz.panels_lda_flow import LDA_FLOW
from core.viz.panels_lda_intertopic import LDA_INTERTOPIC
from core.viz.panels_lda_prevalence import LDA_PREVALENCE
from core.viz.panels_lda_relevance import LDA_RELEVANCE
from core.viz.panels_lda_stability import LDA_STABILITY
from core.viz.panels_mallet import MALLET_DOCUMENT_TOPICS, MALLET_TOPIC_TERMS
from core.viz.panels_meaning import MEANING_PANELS
from core.viz.panels_models import MODELS_PANELS
from core.viz.panels_ngram_viewer import NGRAM_FREQUENCY_OVER_TIME
from core.viz.panels_pairs import PAIRS_PANELS
from core.viz.panels_sentiment import ANNUAL_SENTIMENT_PANELS, SENTIMENT_PANELS
from core.viz.panels_sentiment_views import SENTIMENT_VIEWS_PANELS
from core.viz.panels_stats import STATS_PANELS
from core.viz.panels_svo_agency import SVO_AGENCY
from core.viz.panels_terms import TERMS_PANELS
from core.viz.panels_time_positions import TIME_POSITIONS_PANELS
from core.viz.panels_verb_profiles import VERB_PROFILE_PANELS
from core.viz.panels_word_meaning import WORD_MEANING_PANELS
from core.viz.panelspec import PanelDefinition, PanelParam, PreparedPanel, Provenance, Source

__all__ = [
    "PANELS",
    "best_table",
    "fit_to_rows",
    "get_panel",
    "panel_names",
    "panels_for_tool",
    "prepare_panel",
    "validate_registry",
]


#: Every registered panel. One line per panel; the definition itself lives
#: beside its builder.
PANELS: tuple[PanelDefinition, ...] = (
    KEYNESS_VOLCANO,
    NGRAM_FREQUENCY_OVER_TIME,
    LDA_RELEVANCE,
    LDA_INTERTOPIC,
    LDA_PREVALENCE,
    COLLOCATION_STRENGTH,
    LDA_FLOW,
    LDA_STABILITY,
    MALLET_DOCUMENT_TOPICS,
    MALLET_TOPIC_TERMS,
    SVO_AGENCY,
    # What the vectors mean first; the neighbour list and t-SNE map after.
    *WORD_MEANING_PANELS,
    WORD2VEC_GENSIM_NEIGHBOURS,
    WORD2VEC_GENSIM_VECTOR_QUERY,
    WORD2VEC_GENSIM_TSNE,
    WORD2VEC_BERT_NEIGHBOURS,
    WORD2VEC_BERT_VECTOR_QUERY,
    WORD2VEC_BERT_TSNE,
    # The annual means are left out: an average with no spread, drawn as a
    # line through single speeches. SENTIMENT_VIEWS_PANELS asks the same
    # question with every speech shown and a rolling median.
    *(panel for panel in SENTIMENT_PANELS if panel not in ANNUAL_SENTIMENT_PANELS),
    # The 0.4.0 shape families: per-document measures, term rankings,
    # pairwise similarity/duplicates, and positions over time and text.
    *DOCUMENT_MEASURES_PANELS,
    *TERMS_PANELS,
    *PAIRS_PANELS,
    *TIME_POSITIONS_PANELS,
    *VERB_PROFILE_PANELS,
    *MODELS_PANELS,
    *STATS_PANELS,
    *SENTIMENT_VIEWS_PANELS,
    *EXTRAS_PANELS,
    # doc_embeddings: documents by meaning (sentence models, core/models).
    *MEANING_PANELS,
)


def panel_names() -> tuple[str, ...]:
    return tuple(definition.name for definition in PANELS)


def get_panel(name: str) -> PanelDefinition | None:
    """The definition with this name, or None."""
    for definition in PANELS:
        if definition.name == name:
            return definition
    return None


def panels_for_tool(tool: str) -> tuple[PanelDefinition, ...]:
    """Every panel that draws this tool's results, in registration order."""
    return tuple(definition for definition in PANELS if definition.tool == tool)


def validate_registry() -> list[str]:
    """Problems with the registry itself, as messages. Empty means healthy.

    Called by ``tests/test_panels.py`` rather than at import time —
    R1 forbids import-time work, and a registry check that runs on every
    import would make a typo in one panel break every tool.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for definition in PANELS:
        if definition.name in seen:
            problems.append(f"duplicate panel name: {definition.name}")
        seen.add(definition.name)
        if definition.shape not in IMPLEMENTED_SHAPES:
            problems.append(
                f"{definition.name}: shape {definition.shape!r} is registered but no renderer draws it "
                f"(implemented: {', '.join(IMPLEMENTED_SHAPES)})"
            )
        if not definition.requires:
            problems.append(f"{definition.name}: declares no required columns, so nothing validates its input")
        if not definition.notes:
            problems.append(f"{definition.name}: declares no notes; every panel must state what it cannot show")
    return problems


def prepare_panel(
    name: str,
    frame: pd.DataFrame,
    params: Mapping[str, Any] | None = None,
    *,
    source: Source | None = None,
) -> Result[PreparedPanel]:
    """Validate a panel request and prepare the figure.

    ``source`` says where the numbers came from; the registry supplies the
    rest of the provenance from the panel's own definition. It is optional
    because a live preview has no artifact yet, but its absence is reported
    rather than assumed.
    """
    origin = source or Source()
    definition = get_panel(name)
    if definition is None:
        known = ", ".join(panel_names())
        return Result[PreparedPanel].failure(
            Diagnostic.error("PANEL_UNKNOWN", f"no panel named {name!r}; known panels: {known}", panel=name)
        )
    if definition.shape not in IMPLEMENTED_SHAPES:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_SHAPE_UNIMPLEMENTED",
                f"{name} declares shape {definition.shape!r}, which no renderer draws yet",
                panel=name,
                shape=definition.shape,
            )
        )
    if frame.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error("PANEL_EMPTY", f"{name}: the source table has no rows", panel=name)
        )
    missing = [column for column in definition.requires if column not in frame.columns]
    if missing:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_MISSING_COLUMN",
                f"{name} needs column(s) {missing} which the table does not have. "
                f"Columns present: {list(frame.columns)}",
                panel=name,
                missing=missing,
            )
        )

    resolved = _resolve_params(definition, params or {})
    if resolved.value is None:
        return Result[PreparedPanel](None, resolved.diagnostics)
    values = resolved.unwrap()

    provenance = Provenance(
        tool=definition.tool,
        panel=definition.name,
        source=origin.path,
        source_sha256=origin.sha256,
        params=dict(values),
        settings=dict(origin.settings),
        filters=origin.filters,
        library=origin.library,
    )
    notices = list(resolved.diagnostics)
    if not provenance.is_complete:
        notices.append(
            Diagnostic.warning(
                "PANEL_PROVENANCE_INCOMPLETE",
                f"{name} was prepared without naming its source artifact, so the exported caption cannot "
                "say where the numbers came from.",
                panel=name,
            )
        )

    try:
        built = definition.build(frame, values, provenance)
    except Exception as exc:  # a builder is new code; contain it (R4)
        return Result[PreparedPanel].failure(
            Diagnostic.error("PANEL_BUILD_FAILED", f"{name} failed while preparing: {exc}", panel=name),
            *notices,
        )
    if built.value is None:
        return Result[PreparedPanel](None, tuple(notices) + built.diagnostics)
    return Result.success(fit_to_rows(built.unwrap()), *notices, *built.diagnostics)


class _Requires(Protocol):
    """Anything that names the columns it reads: a panel, or a publication bundle."""

    @property
    def requires(self) -> tuple[str, ...]: ...


#: Pixels per labelled row, and the chrome around the rows (title, axis,
#: caption), for the row-banded shapes. Twelve px is one line of 10px text.
_ROW_PX = 13
_BAR_ROW_PX = 17
_ROW_CHROME = 150
_MAX_HEIGHT = 2400


def best_table(definition: _Requires, tables: Iterable[tuple[str, Iterable[str]]]) -> str | None:
    """Which of a run's tables feeds a panel: the one carrying every column
    it reads with the fewest others, the earliest on a tie.

    "The first that carries the columns" fed the topic-prevalence figure
    LDA's paragraph-by-paragraph topic flow whenever that table came first:
    it has the same Document and Dominant topic columns, one row per passage,
    and the figure counted passages as speeches. The table written for a
    figure carries little else, so the closest fit is the intended one.
    """
    needed = set(definition.requires)
    best: tuple[int, int, str] | None = None
    for position, (name, columns) in enumerate(tables):
        present = set(columns)
        if not needed <= present:
            continue
        candidate = (len(present - needed), position, name)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best else None


def fit_to_rows(prepared: PreparedPanel) -> PreparedPanel:
    """A row-banded figure made tall enough to label every row.

    Eighty-seven speeches in the default 500px left rows four pixels high:
    the renderer could label only every third one, and the heatmap's cells
    were slivers. Sizing is a layout rule, not a finding, so it lives here
    once rather than in every builder; a builder that asked for more height
    keeps it.
    """
    if prepared.shape in ("heatmap", "distribution", "positions"):
        rows = len(prepared.y_categories)
        per = _ROW_PX
    elif prepared.shape == "ribbon":
        rows = len({mark.y for mark in prepared.marks})
        per = _ROW_PX
    elif prepared.shape == "ranked_bars":
        rows = len({mark.label for mark in prepared.marks})
        per = _BAR_ROW_PX
    elif prepared.shape == "network" and prepared.edge_style == "elbow":
        rows = sum(1 for mark in prepared.marks if mark.labelled)
        per = _ROW_PX
    else:
        return prepared
    needed = min(_MAX_HEIGHT, _ROW_CHROME + rows * per)
    if needed <= prepared.height:
        return prepared
    return replace(prepared, height=needed)


def _resolve_params(definition: PanelDefinition, given: Mapping[str, Any]) -> Result[dict[str, Any]]:
    """Defaults merged, unknown names refused, declared bounds enforced."""
    declared = {param.name: param for param in definition.params}
    unknown = sorted(set(given) - set(declared))
    if unknown:
        known = ", ".join(sorted(declared)) or "(none)"
        return Result[dict[str, Any]].failure(
            Diagnostic.error(
                "PANEL_UNKNOWN_PARAM",
                f"{definition.name} has no parameter(s) {unknown}; it takes: {known}",
                panel=definition.name,
                unknown=unknown,
            )
        )
    values = definition.defaults()
    for key, raw in given.items():
        checked = _check_param(definition.name, declared[key], raw)
        if checked.value is None:
            return Result[dict[str, Any]](None, checked.diagnostics)
        values[key] = checked.unwrap()
    return Result.success(values)


def _check_param(panel: str, param: PanelParam, raw: Any) -> Result[Any]:
    """One parameter against its declaration: type first, then bounds."""

    def bad(detail: str) -> Result[Any]:
        return Result[Any].failure(
            Diagnostic.error(
                "PANEL_BAD_PARAM",
                f"{panel}: {param.name} {detail} (got {raw!r})",
                panel=panel,
                param=param.name,
            )
        )

    value: Any = raw
    if param.type == "bool":
        if not isinstance(raw, bool):
            return bad("must be true or false")
    elif param.type == "int":
        # bool is an int in Python; accepting True as 1 here would let a
        # checkbox silently satisfy a count.
        if isinstance(raw, bool) or not isinstance(raw, int):
            return bad("must be a whole number")
    elif param.type == "float":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return bad("must be a number")
        value = float(raw)
    elif param.type == "choice":
        if not isinstance(raw, str) or raw not in param.choices:
            return bad(f"must be one of {', '.join(param.choices)}")
    elif not isinstance(raw, str):
        return bad("must be text")

    if param.type in ("int", "float"):
        if param.minimum is not None and value < param.minimum:
            return bad(f"must be at least {param.minimum}")
        if param.maximum is not None and value > param.maximum:
            return bad(f"must be at most {param.maximum}")
    return Result.success(value)
