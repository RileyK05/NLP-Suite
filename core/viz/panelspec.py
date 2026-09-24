"""Tool-specific panels — figures whose meaning comes from one analysis.

``ChartSpec`` (``core/viz/chartspec.py``) draws any tidy table: name a kind,
map x, y and group to columns, and one set of rules for aggregation,
normalization and rates applies uniformly. That generality is the point, and
for frequency tables, lexical series and ranked counts it is the right tool.

It is the wrong tool for a result that is not a table of y by x. A topic
model produces two matrices (topic x term, document x topic) plus a geometry
over the simplex; asking which column is "x" has no honest answer. A word
embedding comparison's subject is the *difference between two spaces*. A
keyness table has a natural figure — effect size against strength of
evidence — that no combination of kind/x/y/agg expresses, so today it is
drawn as a bar chart of G2, which hides the one thing a reader needs to
judge it.

A **panel** is the second road: a named figure that belongs to one tool,
knows that tool's result shape, and declares its own parameters. Panels
deliberately do not extend ``CHART_KINDS``. A kind added there would inherit
x/y/group/agg validation that means nothing here, and the generic path's
strict semantics are worth keeping unpolluted. The two roads meet again at
rendering and export, not before.

The pattern is not new to this file. ``core/viz/dispersion_plot.py`` already
built one tool-specific figure by hand, and already stated the rule that
matters most: *a tick you cannot trace back to a document is not evidence*.
This module is that idea generalised, so the next five figures do not each
reinvent it.

Design contract (each rule has a named diagnostic and a test):

* **Preparation is pure; rendering is separate.** A builder is a function of
  (frame, params, provenance) returning a :class:`PreparedPanel` — marks,
  labels, annotations, and the numbers behind them. It imports no renderer,
  touches no filesystem (R3), and needs no browser. What a panel *claims* is
  therefore testable without plotly installed, which is the half worth
  testing. Rendering lives in ``core/viz/panel_plotters.py``.
* **Every mark carries its evidence.** :attr:`PanelMark.evidence` is
  required, not optional. A drawn thing that cannot say which rows,
  documents or sentences it stands for is decoration, and this contract is
  the difference between a figure and a finding. Evidence is expressed as
  *filters* — (column, value) pairs — never as row indices: indices do not
  survive a re-run, a re-sort, or the viewer's 500-row page, and a stale
  index silently points at the wrong text.
* **Every panel carries its provenance.** :class:`Provenance` is a required
  field, so a caption naming the source artifact, the tool, the settings and
  the renderer can be attached to any export without the builder having
  remembered to. An incomplete provenance is a WARNING
  (``PANEL_PROVENANCE_INCOMPLETE``), never a silent blank caption.
* **Shapes are a small closed vocabulary, and unimplemented ones are
  rejected up front** (``PANEL_SHAPE_UNIMPLEMENTED``). A panel is never
  half-served: the renderer's :data:`~core.viz.panel_plotters.
  IMPLEMENTED_SHAPES` is the authority, ``prepare_panel`` checks it before
  doing any work, and ``tests/test_panels.py`` asserts the two agree.
  This mirrors ``chartspec``'s ``CHART_UNSUPPORTED`` rule for the same
  reason: a figure that renders as something else is worse than no figure.
* **Parameters are declared, defaulted and bounds-checked centrally.** Each
  panel declares :class:`PanelParam` entries; the registry merges defaults,
  rejects unknown names (``PANEL_UNKNOWN_PARAM``) and enforces
  type/range/choice (``PANEL_BAD_PARAM``) before the builder runs. A builder
  receives values it can trust, so thirteen builders do not re-implement one
  validator thirteen slightly different ways.
* **Groups have a declared draw order.** :attr:`PreparedPanel.groups` fixes
  legend and colour assignment, so the same panel over the same data colours
  the same category the same way on every run.
* **A panel says what it cannot show.** :attr:`PreparedPanel.notes` carries
  the limits of the figure in plain language and travels with it into the
  export. A volcano plot that does not mention that G2 scales with corpus
  size is inviting a wrong reading.

No filesystem writes here (R3), no rendering imports, and no I/O: this
module is data shapes and validation only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from core.result import Result

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "COLOR_SCALES",
    "EDGE_STYLES",
    "EVIDENCE_SCOPES",
    "PANEL_SHAPES",
    "Annotation",
    "Evidence",
    "PanelBuilder",
    "PanelDefinition",
    "PanelEdge",
    "PanelMark",
    "PanelParam",
    "PreparedPanel",
    "Provenance",
    "Source",
    "counted_on_lemmas",
]


# --------------------------------------------------------------- vocabulary --

PanelShape = Literal[
    "scatter_labelled",
    "ranked_bars",
    "stream",
    "line_series",
    "ribbon",
    "small_multiples",
    "heatmap",
    "distribution",
    "network",
    "positions",
]
#: Every shape a panel may declare. Declaring one here does NOT make it
#: drawable — the renderer's ``IMPLEMENTED_SHAPES`` decides that, and
#: ``prepare_panel`` refuses a shape the renderer cannot draw. The vocabulary
#: is listed in full so the roadmap is legible and two panels needing the same
#: geometry cannot invent two names for it.
PANEL_SHAPES: tuple[str, ...] = (
    # Points in a plane, selectively labelled, with reference lines. Keyness
    # volcano; intertopic distance map.
    "scatter_labelled",
    # Horizontal bars in a computed (not alphabetical) order, optionally over
    # a second ghost series. Topic-term relevance.
    "ranked_bars",
    # Stacked bands over an ordered axis. Topic prevalence through time.
    "stream",
    # Independent annual series. A missing year is a gap, never a zero or a
    # straight-line estimate; builders supply explicit zero rows when the
    # corpus had dated documents but no matches.
    "line_series",
    # A sequence of coloured segments along one axis. Paragraph topic flow.
    "ribbon",
    # A grid of small point-and-line plots, one per ``PanelMark.facet`` in
    # ``PreparedPanel.facets`` order, each with its OWN y scale over a shared
    # x axis. Several measures that do not share units (length, sentence
    # count, average sentence length) side by side without one flattening the
    # rest.
    "small_multiples",
    # A matrix of cells. ``x``/``y`` are integer indexes into
    # ``x_categories``/``y_categories`` (row 0 at the top); ``value`` is the
    # cell's number, coloured on ``color_scale``. Document similarity,
    # document-by-term weights, topic matches across seeds, crosstabs.
    "heatmap",
    # One mark per observation: ``x`` is the value, ``y`` the index of its
    # row in ``y_categories``. Each row is drawn as a box (quartiles by linear
    # interpolation, whiskers to min and max) with every point over it,
    # because an average without its spread describes nothing.
    "distribution",
    # Nodes and edges. Marks are nodes placed by the builder (``x``, ``y``
    # from a deterministic layout, so both renderers agree); ``edges`` join
    # mark keys and carry their own evidence. ``edge_style="elbow"`` draws a
    # dendrogram's right-angled links.
    "network",
    # Where things fall along a document. ``x`` is a position (0..1 through
    # the document, or a token offset), ``y`` the index of the document's row
    # in ``y_categories``; each mark is a tick. Lexical dispersion plots,
    # coreference chains.
    "positions",
)

#: Colour scales a heatmap may declare. ``sequential`` runs from the lowest
#: value to the highest; ``diverging`` is centred on zero and symmetric, for
#: signed values such as correlations. The stops are shared with the app
#: (``panelLayout.ts``) and pinned by ``tests/test_panel_parity.py``.
COLOR_SCALES: dict[str, tuple[str, ...]] = {
    "sequential": ("#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"),
    "diverging": ("#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"),
}
EDGE_STYLES: tuple[str, ...] = ("straight", "elbow")

EvidenceScope = Literal["rows", "documents", "sentences", "terms"]
#: What a mark's evidence points at. ``rows`` filters the result table;
#: ``terms`` names words to look up in a concordance; ``documents`` and
#: ``sentences`` address the corpus itself.
EVIDENCE_SCOPES: tuple[str, ...] = ("rows", "documents", "sentences", "terms")


# ----------------------------------------------------------------- evidence --


@dataclass(frozen=True, slots=True)
class Evidence:
    """What one mark stands for, in terms a caller can resolve.

    Expressed as filters rather than row positions on purpose. ``[("Word",
    "freedom")]`` still means the same thing after the table is re-sorted,
    re-run with a different top-N, or paged; row 17 does not, and a stale
    index points at the wrong text without ever looking wrong.

    ``count`` is how many underlying units the mark summarises — rows behind
    a bar, occurrences behind a point — so a figure can say "this is 3
    sentences" before someone builds an argument on it.

    ``phrase`` is the text to look up in the corpus to read this mark in
    context, for ``terms`` evidence. The filters say which *row* a mark came
    from; they cannot say what to search the documents for, because only the
    builder knows that a keyness term lives in ``Word`` while a collocation is
    ``Word 1`` then ``Word 2``, and that a windowed collocation is not a
    phrase at all. Empty means "no honest lookup exists", and the app offers
    none rather than a misleading one. ``lemma`` says the table was counted
    over lemmas, so the lookup must also match every inflected form: a term
    counted as ``be`` occurs in the text as "is" and "was".
    """

    scope: EvidenceScope
    filters: tuple[tuple[str, str], ...]
    count: int
    describe: str
    phrase: str = ""
    lemma: bool = False

    def __post_init__(self) -> None:
        if self.scope not in EVIDENCE_SCOPES:
            raise ValueError(f"scope must be one of {EVIDENCE_SCOPES}, got {self.scope!r}")
        if not self.filters:
            raise ValueError("Evidence needs at least one (column, value) filter")
        for pair in self.filters:
            if len(pair) != 2 or not isinstance(pair[0], str) or not isinstance(pair[1], str):
                raise TypeError(f"each filter must be a (column, value) pair of strings, got {pair!r}")
            if not pair[0]:
                raise ValueError("filter column must be a non-empty string")
        if self.count < 0:
            raise ValueError(f"count must not be negative, got {self.count}")
        if not self.describe.strip():
            raise ValueError("Evidence needs a describe string")
        if self.phrase and self.scope != "terms":
            raise ValueError(f"only terms evidence carries a lookup phrase, not {self.scope!r} evidence")
        if self.lemma and not self.phrase:
            raise ValueError("lemma matching describes a phrase lookup; set a phrase or leave lemma off")

    def as_query(self) -> dict[str, str]:
        """The filters as a mapping, for a caller building a table query."""
        return {column: value for column, value in self.filters}

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "filters": [list(pair) for pair in self.filters],
            "count": self.count,
            "describe": self.describe,
            "phrase": self.phrase,
            "lemma": self.lemma,
        }


def counted_on_lemmas(settings: Mapping[str, Any]) -> bool:
    """Whether a run's table was counted over lemmas, from its own settings.

    The ``field`` parameter keyness and collocations record in their envelope,
    defaulting to lemma as both tools do. A panel drawn from the CLI has no
    envelope settings, and the tools' default is the honest guess there.
    """
    return str(settings.get("field", "lemma")).lower() == "lemma"


# --------------------------------------------------------------- provenance --


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a panel's numbers came from, as a caption can state it.

    Carried by every :class:`PreparedPanel` so that export does not depend on
    a builder having remembered to describe itself. The fields are the ones a
    reader needs to decide whether to trust the figure and a colleague needs
    to reproduce it: the artifact and its hash, the tool, the panel's own
    parameters, the analysis settings that produced the artifact, any filter
    narrowing the data, and what drew it.
    """

    tool: str
    panel: str
    source: str = ""
    source_sha256: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    filters: str = ""
    library: str = ""
    generated: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def is_complete(self) -> bool:
        """True when the caption can name where the numbers came from.

        A live preview legitimately has no artifact yet, so an incomplete
        provenance is a warning rather than a failure — but it is never
        silent, because an exported figure with an anonymous source is the
        one a reader cannot check.
        """
        return bool(self.source)

    def caption(self) -> str:
        """One line naming source, settings and renderer, for the figure."""
        parts = [f"{self.panel} · {self.tool}"]
        if self.source:
            digest = f" ({self.source_sha256[:12]})" if self.source_sha256 else ""
            parts.append(f"from {self.source}{digest}")
        else:
            parts.append("source not recorded")
        if self.filters:
            parts.append(self.filters)
        settings = _describe_mapping(self.settings)
        if settings:
            parts.append(settings)
        params = _describe_mapping(self.params)
        if params:
            parts.append(params)
        if self.library:
            parts.append(f"drawn by {self.library}")
        if self.generated:
            parts.append(self.generated)
        return " · ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "panel": self.panel,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "params": dict(self.params),
            "settings": dict(self.settings),
            "filters": self.filters,
            "library": self.library,
            "generated": self.generated,
        }


@dataclass(frozen=True, slots=True)
class Source:
    """What the caller knows about where a panel's numbers came from.

    The registry supplies the rest of :class:`Provenance` (which tool, which
    panel, which parameters) because it already knows it. This is the half
    only the caller can know, gathered into one object so a CLI or a server
    can assemble it once and hand the same record to every panel it draws.
    """

    path: str = ""
    sha256: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    filters: str = ""
    library: str = ""


def _describe_mapping(values: Mapping[str, Any]) -> str:
    """``a=1 b=2`` in key order, so two captions of one run read alike."""
    return " ".join(f"{key}={values[key]}" for key in sorted(values))


# ------------------------------------------------------------------- pieces --


@dataclass(frozen=True, slots=True)
class PanelMark:
    """One drawn thing, and what it stands for.

    ``labelled`` asks the renderer to write the text beside this mark.
    Labelling every point in a 200-word volcano produces an unreadable smear,
    so selection is the builder's decision (it knows which points are the
    finding) rather than a rendering accident.
    """

    key: str
    label: str
    x: float
    y: float
    evidence: Evidence
    group: str = ""
    size: float | None = None
    labelled: bool = False
    #: A heatmap cell's number, which its colour encodes. Unused elsewhere.
    value: float | None = None
    #: Which small-multiple panel this mark belongs to (``PreparedPanel.facets``).
    facet: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("PanelMark.key must be a non-empty string")
        if self.size is not None and self.size < 0:
            raise ValueError(f"PanelMark.size must not be negative, got {self.size}")


@dataclass(frozen=True, slots=True)
class PanelEdge:
    """A link between two node marks of a ``network`` panel.

    Clickable like a mark, because in a co-occurrence network the link *is*
    the finding: "war" and "peace" sharing 40 sentences is a claim about
    those 40 sentences, and a reader must be able to open them.
    """

    key: str
    source: str
    target: str
    weight: float
    evidence: Evidence
    label: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("PanelEdge.key must be a non-empty string")
        if self.source == self.target:
            raise ValueError(f"edge {self.key!r} joins {self.source!r} to itself")
        if self.weight < 0:
            raise ValueError(f"edge weight must not be negative, got {self.weight}")


@dataclass(frozen=True, slots=True)
class Annotation:
    """A reference line or note, with the reason it is there.

    ``note`` is not decoration. A line at G2 = 3.84 means nothing to most
    readers; "p < 0.05 at 1 degree of freedom" means something, and it is the
    difference between a figure that informs and one that merely looks
    rigorous.
    """

    kind: Literal["vline", "hline", "note"]
    value: float
    label: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("vline", "hline", "note"):
            raise ValueError(f"Annotation.kind must be vline, hline or note, got {self.kind!r}")


@dataclass(frozen=True, slots=True)
class PanelParam:
    """One declared parameter: its default, its bounds and its explanation.

    The registry validates against this before a builder runs, so every panel
    enforces its bounds identically and the desktop can build controls from
    the declaration instead of hard-coding them per panel.
    """

    name: str
    type: Literal["int", "float", "str", "bool", "choice"]
    default: Any
    help: str
    label: str = ""
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PanelParam.name must be a non-empty string")
        if self.type not in ("int", "float", "str", "bool", "choice"):
            raise ValueError(f"unsupported PanelParam type {self.type!r}")
        if self.type == "choice" and not self.choices:
            raise ValueError(f"choice parameter {self.name!r} needs choices")
        if self.type != "choice" and self.choices:
            raise ValueError(f"only a choice parameter may declare choices ({self.name!r})")
        if not self.help.strip():
            raise ValueError(f"PanelParam {self.name!r} needs help text")

    @property
    def display(self) -> str:
        return self.label or self.name.replace("-", " ").replace("_", " ").capitalize()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "help": self.help,
            "label": self.display,
            "choices": list(self.choices),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


# ------------------------------------------------------------------ prepared --


@dataclass(frozen=True, slots=True)
class PreparedPanel:
    """A figure, decided but not yet drawn.

    Everything a renderer needs and nothing it has to compute: the marks with
    their evidence, the axis labels, the reference lines, the group draw
    order, the provenance caption, the limits worth stating, and ``data`` —
    the numbers behind the picture, so a figure can always be published
    beside the rows that produced it.
    """

    panel: str
    shape: PanelShape
    title: str
    marks: tuple[PanelMark, ...]
    x_label: str
    y_label: str
    provenance: Provenance
    data: pd.DataFrame
    subtitle: str = ""
    annotations: tuple[Annotation, ...] = ()
    groups: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    width: int = 900
    height: int = 500
    #: Category axes, for the shapes whose ``x``/``y`` are indexes into them
    #: (heatmap: both; distribution and positions: ``y_categories``).
    x_categories: tuple[str, ...] = ()
    y_categories: tuple[str, ...] = ()
    #: Heatmap only: a key of :data:`COLOR_SCALES`.
    color_scale: str = ""
    #: Heatmap only: what a cell's colour stands for ("Cosine similarity, %"),
    #: the colour bar's title. Empty on an older builder; the bar is then
    #: labelled only by its end values.
    value_label: str = ""
    #: Network only.
    edges: tuple[PanelEdge, ...] = ()
    edge_style: str = "straight"
    #: Small multiples only: the panels, in draw order.
    facets: tuple[str, ...] = ()
    #: ``line_series`` and ``small_multiples``: a gap in ``x`` wider than this
    #: breaks the line, because a line across it would invent observations.
    #: One, for annual series; a builder over dated documents sets its own.
    line_gap: float = 1.0
    #: x holds log10 of the quantity (a count spanning 10 to 10,000); every
    #: renderer labels its ticks with the quantity itself -- 10, 100, 1,000 --
    #: never with "1.5", which a reader has to exponentiate in their head.
    x_log10: bool = False
    #: Groups drawn as points only, never joined: the individual documents
    #: behind a rolling median. Connecting single speeches in date order
    #: draws a trend out of noise.
    points_only: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.shape not in PANEL_SHAPES:
            raise ValueError(f"shape must be one of {PANEL_SHAPES}, got {self.shape!r}")
        if self.width < _MIN_DIMENSION or self.height < _MIN_DIMENSION:
            raise ValueError(f"panel dimensions must be at least {_MIN_DIMENSION}px")
        seen: set[str] = set()
        for mark in self.marks:
            if mark.key in seen:
                raise ValueError(f"duplicate mark key {mark.key!r}: keys identify marks and must be unique")
            seen.add(mark.key)
        unknown = sorted({mark.group for mark in self.marks if mark.group} - set(self.groups))
        if unknown:
            raise ValueError(f"marks use groups missing from the declared draw order: {unknown}")
        if self.line_gap <= 0:
            raise ValueError(f"line_gap must be positive, got {self.line_gap}")
        stray = sorted(set(self.points_only) - set(self.groups))
        if stray:
            raise ValueError(f"points_only names groups missing from the declared draw order: {stray}")
        self._check_shape_fields(seen)

    def _check_shape_fields(self, keys: set[str]) -> None:
        """The fields a shape reads must be present and consistent.

        Checked here, once, so neither renderer has to guess what a heatmap
        with no categories or an edge to a missing node was meant to be.
        """
        if self.shape == "heatmap":
            if self.color_scale not in COLOR_SCALES:
                raise ValueError(f"a heatmap needs color_scale in {sorted(COLOR_SCALES)}, got {self.color_scale!r}")
            _check_indexes(self.marks, "x", self.x_categories)
            _check_indexes(self.marks, "y", self.y_categories)
            missing = [mark.key for mark in self.marks if mark.value is None]
            if missing:
                raise ValueError(f"heatmap cells need a value: {missing[:5]}")
        elif self.shape in ("distribution", "positions"):
            _check_indexes(self.marks, "y", self.y_categories)
        elif self.shape == "small_multiples":
            if not self.facets:
                raise ValueError("small multiples need their facets declared in draw order")
            stray = sorted({mark.facet for mark in self.marks} - set(self.facets))
            if stray:
                raise ValueError(f"marks use facets missing from the declared order: {stray}")
        if self.shape == "network":
            if self.edge_style not in EDGE_STYLES:
                raise ValueError(f"edge_style must be one of {EDGE_STYLES}, got {self.edge_style!r}")
            edge_keys: set[str] = set()
            for edge in self.edges:
                if edge.key in edge_keys or edge.key in keys:
                    raise ValueError(f"duplicate key {edge.key!r}: edge keys must be unique among marks and edges")
                edge_keys.add(edge.key)
                for end in (edge.source, edge.target):
                    if end not in keys:
                        raise ValueError(f"edge {edge.key!r} names node {end!r}, which is not a mark")
        elif self.edges:
            raise ValueError(f"only a network panel draws edges, not {self.shape!r}")

    @property
    def caption(self) -> str:
        """The provenance line that travels with every export."""
        return self.provenance.caption()

    def evidence_for(self, key: str) -> Evidence | None:
        """The evidence behind one mark or edge, by key."""
        for mark in self.marks:
            if mark.key == key:
                return mark.evidence
        for edge in self.edges:
            if edge.key == key:
                return edge.evidence
        return None


def _check_indexes(marks: tuple[PanelMark, ...], axis: str, categories: tuple[str, ...]) -> None:
    """Every mark's *axis* coordinate is a whole-number index into *categories*."""
    if not categories:
        raise ValueError(f"this shape reads {axis} as an index into {axis}_categories, which is empty")
    for mark in marks:
        at = getattr(mark, axis)
        if at != int(at) or not 0 <= at < len(categories):
            raise ValueError(f"mark {mark.key!r}: {axis}={at} is not an index into {len(categories)} categories")


_MIN_DIMENSION: int = 200


#: What every panel builder is. Receives a validated frame, parameters the
#: registry has already defaulted and bounds-checked, and the provenance the
#: registry assembled; returns the prepared figure or diagnostics.
PanelBuilder = Callable[
    ["pd.DataFrame", Mapping[str, Any], Provenance],
    Result["PreparedPanel"],
]


@dataclass(frozen=True, slots=True)
class PanelDefinition:
    """One registered panel: what it needs, what it takes, what draws it.

    ``requires`` names the columns the builder reads. The registry checks
    them before calling, so a panel pointed at the wrong artifact fails with
    the missing column named rather than a ``KeyError`` from inside pandas.
    """

    name: str
    title: str
    tool: str
    shape: PanelShape
    summary: str
    requires: tuple[str, ...]
    params: tuple[PanelParam, ...]
    build: PanelBuilder
    notes: tuple[str, ...] = ()
    #: The question this figure answers, in the reader's words ("Which
    #: speeches are most alike?"). The reading offers a tool's figures by
    #: their questions, in place of generic column pairings.
    question: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PanelDefinition.name must be a non-empty string")
        if self.shape not in PANEL_SHAPES:
            raise ValueError(f"{self.name}: shape must be one of {PANEL_SHAPES}, got {self.shape!r}")
        if not self.tool:
            raise ValueError(f"{self.name}: a panel must name the tool it belongs to")
        if not self.summary.strip():
            raise ValueError(f"{self.name}: a panel must summarise itself")
        names = [param.name for param in self.params]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"{self.name}: duplicate parameter names {duplicates}")

    def defaults(self) -> dict[str, Any]:
        return {param.name: param.default for param in self.params}

    def describe(self) -> str:
        """One line for a CLI listing."""
        return f"{self.name:<26} {self.tool:<14} {self.shape:<17} {self.summary}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "tool": self.tool,
            "shape": self.shape,
            "summary": self.summary,
            "requires": list(self.requires),
            "params": [param.to_dict() for param in self.params],
            "notes": list(self.notes),
            "question": self.question,
        }
