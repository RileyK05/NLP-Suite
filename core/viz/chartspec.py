"""Typed chart specification + data preparation for publication-ready charts.

The deterministic core of the ``charts`` tool (FR-6.5): one :class:`ChartSpec`
describes what to draw; :func:`prepare_chart_data` turns an analyst's CSV into
the tidy frame a renderer needs — with **explicit, auditable semantics**.

Design contract (each rule has a named diagnostic and a test):

* **Deterministic, dtype-preserving ordering.** Numeric/datetime x values
  keep their dtype and sort numerically (years ``1, 2, 10`` never become the
  text order ``"1", "10", "2"``); only truly categorical x is text-sorted.
  The same input always produces the same prepared frame.
* **Aggregation is explicit opt-in, never a silent default.** Duplicate
  observations in one (x, group) cell fail with ``CHART_AMBIGUOUS`` unless
  ``agg`` names a reduction (sum/mean/median/count). Observation-shaped kinds
  (``_OBSERVATION_KINDS``: scatter, bubble, box, violin) reject ``agg``
  outright — collapsing rows would change the chart's meaning, so there is
  nothing to default to — and for the same reason they are exempt from the
  ambiguity check: a kind that refuses ``agg`` must never go on to demand it.
* **Heatmap shape is fixed: x columns by group rows, y as the numeric
  cells.** A heatmap without ``group`` is rejected (``CHART_MISSING_GROUP``);
  duplicate (x, group) cells require ``agg`` (``CHART_AMBIGUOUS``); the y
  column is always the measure, never an axis. The prepared labels name the
  RENDERED axes: x names the column axis, y names the group (row) axis, and
  the measure names the colorbar — the renderer owns that split.
* **Group is always present.** The prepared frame carries a ``group`` column
  even when no group column was requested (constant ``"(all)"``), so
  renderers and downstream CSV consumers see one stable schema.
* **Histograms use common bin edges.** Edges are computed ONCE from the whole
  dataset, so grouped histograms are comparable across groups; every group is
  reported over the same bins (zero counts included); the bin count is
  exposed. Bin centers and widths are exposed for renderers. ``agg`` is not
  defined for histograms and is rejected; ``scale_by`` does not exist for
  histograms and is rejected. Prepared labels describe the rendered axes:
  x names the binned VALUE column (``spec.y`` — the categorical ``spec.x``,
  if given, is an ignored selector), y names Count/Percent/Share.
* **Rates are explicit.** ``rate_per`` + ``denominator_column`` scale
  bar/line numerators by the aggregated denominator total times a positive
  multiplier (mentions per 10,000 tokens: ``rate_per=10000``). Only ``sum``
  aggregation is supported for rates — a mean of ratios is not a ratio of
  sums.
* **Ranking is category-level across groups with deterministic ties.**
  ``top_n`` ranks whole x-categories by their total y across every group;
  ties are broken by category name ascending; the ranking happens BEFORE
  normalization (shares are computed over what remains) and that order is
  recorded in ``prepared_by``.
* **Invalid numbers are diagnostics, not zeros.** Non-numeric/non-finite y
  values fail loudly (``CHART_BAD_NUMERIC``); an aggregate sum that
  overflows to +inf (1e308 + 1e308) fails (``CHART_NONFINITE_AGGREGATE``);
  zero/negative normalization denominators fail (``CHART_ZERO_DENOMINATOR``);
  negative values fail share/percent (a negative share is not a chart).
* **Unsupported combinations are rejected up front** (``CHART_UNSUPPORTED``)
  — the envelope never carries a chart that means something else. Anything a
  renderer does not yet implement is rejected, never half-served
  (heatmap normalize/rates/top-n, histogram agg/rates, observation-kind
  rates).
* **Missing categories are diagnosed, never silently dropped.** Every group
  observed in the data appears in the prepared frame; empty histogram bins
  are reported as zero counts rather than being omitted. Missing axis values
  (None/NaN/NaT in the x or group column) fail with ``CHART_BAD_COLUMN`` —
  dropping the row would misreport totals and coercing NaN to ``"nan"``
  would invent a category; the measure/denominator columns were already
  diagnosed per-row, and a histogram may still carry missing values in its
  unused ``x`` selector column.

No filesystem writes here (R3): renderers return HTML strings; the
OutputWriter stays the only writer. Image export is deferred — no renderer
writes bytes until the writing path is decided.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "AGG_FUNCS",
    "CHART_KINDS",
    "ChartSpec",
    "PreparedChart",
    "coerce_date_axis",
    "prepare_chart_data",
]

ChartKind = Literal[
    "bar",
    "line",
    "scatter",
    "histogram",
    "box",
    "heatmap",
    "pie",
    "sunburst",
    "treemap",
    "violin",
    "radar",
    "waffle",
    "calendar",
    "bubble",
]
AggFunc = Literal["sum", "mean", "median", "count"]
Normalize = Literal["none", "percent", "share"]
ScaleBy = Literal["none", "total", "group", "category"]

AGG_FUNCS: tuple[str, ...] = ("sum", "mean", "median", "count")
CHART_KINDS: tuple[str, ...] = (
    "bar",
    "line",
    "scatter",
    "histogram",
    "box",
    "heatmap",
    "pie",
    "sunburst",
    "treemap",
    "violin",
    "radar",
    "waffle",
    "calendar",
    # Excel has drawn bubbles since the legacy suite; it was listed in
    # EXCEL_KINDS but missing here, so no ChartSpec could carry it and the
    # exporter's bubble branch was unreachable.
    "bubble",
)
# Kinds that draw one mark per input row rather than one per x category.
# They reject ``agg`` outright, and they must also be exempt from the
# ``CHART_AMBIGUOUS`` demand for one -- otherwise duplicate x values make the
# kind unusable, refusing ``--agg`` and requiring it at the same time. That is
# exactly what happened to ``bubble``: it was added to the rejection list but
# left out of the dispatch below, so it fell through to the bar/line path.
_OBSERVATION_KINDS: tuple[str, ...] = ("scatter", "box", "violin", "bubble")
# Kinds whose x must be real calendar dates, not date-shaped text.
_DATE_AXIS_KINDS: tuple[str, ...] = ("calendar",)
# Kinds whose x is a hierarchical categorical path (one or two levels).
_HIERARCHY_KINDS: tuple[str, ...] = ("sunburst", "treemap")
# Kinds that aggregate y by x category before drawing (share bar/line rules).
_AGG_KINDS: tuple[str, ...] = ("bar", "line", "pie", "sunburst", "treemap", "radar", "waffle", "calendar")
_NORMALIZE_MODES: tuple[str, ...] = ("none", "percent", "share")
_SCALE_BY_MODES: tuple[str, ...] = ("none", "total", "group", "category")

_UNGROUPED: str = "(all)"
_MIN_DIMENSION: int = 200


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """What to draw — the typed contract between CLI, core, and renderer.

    Attributes mirror the CLI flags 1:1 (``--kind``, ``--x``, ``--y``, ...)
    so the registry spec can be validated against the CLI without translation.

    * ``denominator_column`` names a numeric column whose **aggregated
      total** divides the aggregated numerator (rate scaling); ``rate_per``
      is the positive multiplier applied on top (10_000 = "per 10k tokens").
      Both must be given together and are valid only for bar/line with
      ``--agg sum``.
    * ``scale_by`` names the normalization denominator level for bar/line
      (``total`` = whole chart, ``group`` = per group, ``category`` = per x);
      rejected for histogram (whose normalization is always the whole-chart
      observation total), scatter/box, and heatmap.
    * ``bar_mode`` selects the grouped-bar layout: ``group`` (side-by-side,
      the default — stacking means is never implied), ``stack`` or
      ``relative`` as explicit opt-ins. Only valid for bar charts.
    * ``x_tick_angle`` is the explicit category tick rotation in degrees;
      ``None`` (the default) means "not specified" and picks per-orientation
      defaults in the renderer (vertical -30, horizontal upright). Any
      explicitly supplied angle — including -30 — is honored verbatim.
    """

    kind: ChartKind
    x: str
    y: str
    group: str | None = None
    agg: AggFunc | None = None
    top_n: int | None = None
    normalize: Normalize = "none"
    scale_by: ScaleBy = "total"
    rate_per: float | None = None
    denominator_column: str | None = None
    horizontal: bool = False
    bar_mode: str = "group"
    bins: int | None = None
    title: str = ""
    subtitle: str = ""
    x_label: str = ""
    y_label: str = ""
    width: int = 900
    height: int = 500
    wrap_labels: int = 24
    x_tick_angle: int | None = None
    offline: bool = True

    def __post_init__(self) -> None:
        if self.kind not in CHART_KINDS:
            raise ValueError(f"kind must be one of {CHART_KINDS}, got {self.kind!r}")
        if self.agg is not None and self.agg not in AGG_FUNCS:
            raise ValueError(f"agg must be one of {AGG_FUNCS}, got {self.agg!r}")
        if self.normalize not in _NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {_NORMALIZE_MODES}, got {self.normalize!r}")
        if self.scale_by not in _SCALE_BY_MODES:
            raise ValueError(f"scale_by must be one of {_SCALE_BY_MODES}, got {self.scale_by!r}")
        if self.top_n is not None and self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")
        if self.bins is not None and self.bins < 1:
            raise ValueError(f"bins must be >= 1, got {self.bins}")
        if self.bar_mode not in ("group", "stack", "relative"):
            raise ValueError(f"bar_mode must be one of ('group', 'stack', 'relative'), got {self.bar_mode!r}")
        if self.bar_mode != "group" and self.kind not in ("bar", "waffle"):
            raise ValueError(f"bar_mode is only valid for bar charts, got {self.bar_mode} for {self.kind}")
        if self.kind in _HIERARCHY_KINDS and self.group is None:
            raise ValueError(f"{self.kind} needs --group: the hierarchy is x (outer) by group (inner)")
        if self.width < _MIN_DIMENSION or self.height < _MIN_DIMENSION:
            raise ValueError(f"width/height must be >= {_MIN_DIMENSION}, got {self.width}x{self.height}")
        if self.wrap_labels < 1:
            raise ValueError(f"wrap_labels must be >= 1, got {self.wrap_labels}")
        if not self.x:
            raise ValueError("x must be a non-empty column name")
        if not self.y:
            raise ValueError("y must be a non-empty column name")
        if (self.rate_per is None) != (self.denominator_column is None):
            raise ValueError("rate_per and denominator_column must be given together (rate scaling needs both)")
        if self.rate_per is not None:
            if self.rate_per <= 0:
                raise ValueError(f"rate_per must be a positive multiplier, got {self.rate_per}")
            if not math.isfinite(self.rate_per):
                raise ValueError(f"rate_per must be finite, got {self.rate_per}")


@dataclass(frozen=True, slots=True)
class PreparedChart:
    """The tidy data a renderer draws, plus metadata for the CSV artifact.

    ``data`` columns (one stable schema, ``group`` always present):

    * bar/line/heatmap: ``x``, ``group``, ``y`` (heatmap y IS the cell value)
    * scatter/box: one row per observation, same three columns
    * histogram: ``bin_center``, ``bin_left``, ``bin_right``, ``bin_width``,
      ``count``, ``group``
    * rate-scaled bar/line additionally carries ``denominator`` (the
      aggregated per-cell denominator totals) so rates are independently
      auditable from the CSV alone.

    ``prepared_by`` records the effective data-preparation parameters —
    including the actual bin edges/count, whether normalization ran before or
    after top-n selection, and the rate/normalization stage order — so the
    reproducibility CSV can carry its own provenance in the envelope.

    ``spec`` is the typed ChartSpec that produced this preparation (kept
    explicitly, not an optional back-reference, so callers never lose the
    effective parameters).
    """

    data: pd.DataFrame
    layout: str  # "wide" (x, group, y) | "bins" (histogram) | "matrix" (heatmap)
    prepared_by: dict[str, str]
    x_label: str
    y_label: str
    spec: ChartSpec


def wrap_label(value: object, width: int) -> str:
    """Insert ``<br>`` breaks every *width* characters (HTML label wrap)."""
    text = str(value)
    if width <= 0 or len(text) <= width:
        return text
    chunks = [text[i : i + width] for i in range(0, len(text), width)]
    return "<br>".join(chunks)


def _validate_columns(frame: pd.DataFrame, spec: ChartSpec) -> list[Diagnostic]:
    """Column presence; empty frame; heatmap group requirement."""
    missing = [c for c in (spec.x, spec.y) if c not in frame.columns]
    if spec.group is not None and spec.group not in frame.columns:
        missing.append(spec.group)
    if spec.denominator_column is not None and spec.denominator_column not in frame.columns:
        missing.append(spec.denominator_column)
    if missing:
        return [
            Diagnostic.error(
                "CHART_BAD_COLUMN",
                f"column(s) not in frame: {missing}",
                missing=missing,
            )
        ]
    if frame.empty:
        return [Diagnostic.error("CHART_EMPTY", "frame is empty")]
    missing_values = _check_missing_axis_values(frame, spec)
    if missing_values:
        return missing_values
    if spec.kind == "heatmap" and spec.group is None:
        return [
            Diagnostic.error(
                "CHART_MISSING_GROUP",
                "a heatmap needs --group: cells are (x columns by group rows) with y as the "
                "measure; x is the column axis, the group column is the row axis, and y is "
                "never an axis",
                kind=spec.kind,
            )
        ]
    return []


def _check_missing_axis_values(frame: pd.DataFrame, spec: ChartSpec) -> list[Diagnostic]:
    """Axis columns must be COMPLETE: no None/NaN/NaT cells (silently dropping
    a row would misreport the chart; coercing NaN to the string ``'nan'``
    would invent a category). The ``group`` column is checked whenever given;
    ``x`` is checked for every axis-using kind — histograms ignore ``spec.x``
    entirely (values are binned from ``spec.y``), so an unused x may contain
    missing values. A literal string ``'nan'`` is a legitimate category and
    stays allowed (only actual missing values — NaN/None/NaT — fail).
    """
    columns: list[str] = []
    if spec.group is not None:
        columns.append(spec.group)
    if spec.kind != "histogram":
        columns.append(spec.x)
    for column in columns:
        mask = frame[column].isna()
        if bool(mask.any()):
            bad_rows = frame.index[mask.to_numpy()].tolist()
            return [
                Diagnostic.error(
                    "CHART_BAD_COLUMN",
                    f"column {column!r} contains {int(mask.sum())} missing value(s) (None/NaN/NaT) "
                    "on a chart axis; missing axis values would be silently dropped or coerced "
                    "into a fake category — filter or impute them first",
                    rows=bad_rows[:20],
                )
            ]
    return []


def _numeric_values(frame: pd.DataFrame, spec: ChartSpec) -> Result[pd.Series]:
    """The y column coerced to finite floats, or a diagnostic naming the rows."""
    y_num = pd.to_numeric(frame[spec.y], errors="coerce")
    y_np = y_num.to_numpy(dtype=float, na_value=np.nan)
    bad_mask = y_num.isna() | ~np.isfinite(y_np)
    if bool(bad_mask.any()):
        bad_rows = frame.index[bad_mask.to_numpy()].tolist()
        return Result.failure(
            Diagnostic.error(
                "CHART_BAD_NUMERIC",
                f"column {spec.y!r} contains {int(bad_mask.sum())} non-numeric or non-finite "
                "value(s); refusing to chart — a zero-fill would be misleading",
                rows=bad_rows[:20],
            )
        )
    return Result.success(y_num.astype(float))


def _numeric_denominators(frame: pd.DataFrame, denominator_column: str) -> Result[pd.Series]:
    """The denominator column coerced to finite, non-negative floats."""
    column = frame[denominator_column]
    den_num = pd.to_numeric(column, errors="coerce")
    den_np = den_num.to_numpy(dtype=float, na_value=np.nan)
    bad_mask = den_num.isna() | ~np.isfinite(den_np) | (den_np < 0)
    if bool(bad_mask.any()):
        bad_rows = frame.index[bad_mask.to_numpy()].tolist()
        return Result.failure(
            Diagnostic.error(
                "CHART_BAD_NUMERIC",
                f"denominator column {denominator_column!r} contains "
                f"{int(bad_mask.sum())} non-numeric or negative value(s); rates need finite, "
                "non-negative denominators",
                rows=bad_rows[:20],
            )
        )
    return Result.success(den_num.astype(float))


def _check_finite(values: pd.Series, label: str) -> Result[pd.Series]:
    """An aggregate that overflows to +inf (1e308 + 1e308) is not a number."""
    as_float = values.astype(float)
    bad = as_float[~as_float.map(math.isfinite)]
    if not bad.empty:
        return Result.failure(
            Diagnostic.error(
                "CHART_NONFINITE_AGGREGATE",
                f"{label} produced non-finite value(s) (overflow); refusing to chart: "
                f"{[float(v) for v in bad.head(5)]}",
                count=len(bad),
            )
        )
    return Result.success(as_float)


def _finite_sum(values: pd.Series) -> float:
    """Sum a float series WITHOUT the numpy overflow RuntimeWarning.

    ``sum(1e308, 1e308)`` overflows to +inf; numpy emits a RuntimeWarning
    that warnings-as-errors would abort on. We WANT the +inf here (the
    finiteness check turns it into a diagnostic), so the warning is locally
    suppressed and the value checked — the overflow is never a silent zero.
    """
    with np.errstate(over="ignore"), suppress(RuntimeWarning):
        return float(values.astype(float).sum())
    return float("nan")


def _unsupported(param: str, kind: str, message: str) -> Diagnostic:
    return Diagnostic.error("CHART_UNSUPPORTED", message, kind=kind, param=param)


def _resolve_semantics(spec: ChartSpec) -> list[Diagnostic]:
    """Reject unsupported kind/option combinations, one clear diagnostic each."""
    kind = spec.kind
    diags: list[Diagnostic] = []

    if spec.agg is not None and kind in _OBSERVATION_KINDS:
        diags.append(
            _unsupported(
                "agg",
                kind,
                f"--agg is not valid for {kind}: it draws every observation as-is; aggregation "
                "would silently change the chart's meaning. Use a bar or line chart (with --agg) "
                "for grouped summaries.",
            )
        )
    if spec.agg is not None and kind == "histogram":
        diags.append(
            _unsupported(
                "agg",
                kind,
                "--agg is not valid for a histogram: bins count observations directly; aggregation is undefined.",
            )
        )
    if spec.normalize != "none" and kind in (*_OBSERVATION_KINDS, "heatmap", "radar", "calendar"):
        diags.append(
            _unsupported(
                "normalize",
                kind,
                f"--normalize is not valid for {kind}: shares/percentages are defined over "
                "category totals (bar/line/pie/sunburst/treemap/waffle) or histogram counts, "
                "not raw observations or cell grids.",
            )
        )
    if kind == "pie" and spec.group is not None:
        diags.append(
            _unsupported(
                "group",
                kind,
                "--group is not valid for pie: one pie shows one set of slices; use sunburst "
                "for a two-level x-by-group hierarchy.",
            )
        )
    if kind in ("pie", "sunburst", "treemap") and spec.normalize == "none" and spec.top_n is None:
        pass  # slices may show raw values; percent/share is optional like bar
    if kind in _HIERARCHY_KINDS and spec.normalize in ("percent", "share"):
        pass  # handled in preparation: branch sums are normalized by plotly
    if kind in ("pie", "sunburst", "treemap") and spec.horizontal:
        diags.append(
            _unsupported(
                "horizontal",
                kind,
                f"--horizontal is only valid for bar charts, not {kind}.",
            )
        )
    if kind in ("pie", "sunburst", "treemap", "waffle", "radar", "calendar", "violin") and (
        spec.rate_per is not None or spec.denominator_column is not None
    ):
        diags.append(
            _unsupported(
                "rate",
                kind,
                f"rate scaling is not defined for {kind}: a numerator/denominator ratio needs "
                "aggregated bar/line cells.",
            )
        )
    if kind in ("pie", "sunburst", "treemap", "radar") and spec.bins is not None:
        diags.append(_unsupported("bins", kind, f"--bins is only valid for histograms, not {kind}."))
    if kind == "waffle" and spec.normalize in ("percent", "share"):
        pass  # waffle encodes shares natively; normalization is the drawing
    if kind == "calendar" and spec.bins is not None:
        diags.append(_unsupported("bins", kind, "--bins is only valid for histograms, not calendar."))
    if kind == "histogram" and spec.scale_by != "total":
        diags.append(
            _unsupported(
                "scale-by",
                kind,
                f"--scale-by is not implemented for {kind}: histogram normalization is always "
                "over the whole-chart observation total; pass --scale-by total or none.",
            )
        )
    if kind in (*_OBSERVATION_KINDS, "calendar") and spec.scale_by != "total":
        diags.append(
            _unsupported(
                "scale-by",
                kind,
                f"--scale-by is not valid for {kind}: per-group/per-category scaling is defined "
                "only over aggregated bar/line cells.",
            )
        )
    if kind == "heatmap" and spec.scale_by != "total":
        diags.append(
            _unsupported(
                "scale-by",
                kind,
                f"--scale-by is not valid for {kind}.",
            )
        )
    rates_requested = spec.rate_per is not None or spec.denominator_column is not None
    if rates_requested and kind in (*_OBSERVATION_KINDS, "heatmap", "histogram", "radar", "calendar"):
        diags.append(
            _unsupported(
                "rate",
                kind,
                f"rate scaling is not defined for {kind}: a numerator/denominator ratio needs "
                "aggregated bar/line cells.",
            )
        )
    if spec.top_n is not None and kind not in ("bar", "line", "pie", "sunburst", "treemap", "waffle"):
        diags.append(
            _unsupported(
                "top-n",
                kind,
                f"--top-n is not valid for {kind}: ranking applies to categorical slices (bar/line/pie/sunburst/treemap/waffle).",
            )
        )
    if spec.horizontal and kind != "bar":
        diags.append(
            _unsupported(
                "horizontal",
                kind,
                f"--horizontal is only valid for bar charts, not {kind}.",
            )
        )
    return diags


def _x_sort_key(series: pd.Series) -> pd.Series:
    """Numeric/datetime series sort by value; everything else by string."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return series
    return series.astype(str)


def _order_x(series: pd.Series) -> list[Any]:
    """Deterministic x order: numeric/datetime ascending, else text-sorted."""
    return list(series.drop_duplicates().sort_values(key=_x_sort_key).tolist())


def _sorted_by_x(prepared: pd.DataFrame) -> pd.DataFrame:
    """Deterministic frame order: x by value (numeric-aware), group by text."""
    keys = [c for c in ("x", "group") if c in prepared.columns]
    return prepared.sort_values(
        keys,
        key=lambda s: _x_sort_key(s) if s.name == "x" else s.astype(str),
        kind="stable",
    ).reset_index(drop=True)


def coerce_date_axis(frame: pd.DataFrame, spec: ChartSpec) -> Result[pd.DataFrame]:
    """Parse the x column into real dates for the kinds that need a date axis.

    A CSV carries no dtypes. ``pd.read_csv`` reads ``2024-01-01`` as the string
    ``"2024-01-01"``, so a calendar chart built from a file always failed with
    ``CHART_NOT_DATETIME`` -- telling the reader to "convert the column to
    dates first", which no entry point offered any way to do. Every calendar
    test built its frame with ``pd.to_datetime`` in memory, so the kind was
    covered, implemented and unreachable at the same time.

    Conversion is deliberately narrow: only the one column named as the x axis,
    only for a kind that requires dates, and only at the boundary where a file
    is read. ``prepare_chart_data`` stays pure and still refuses a column that
    is not dates -- this is what gives it one it can accept. A value that does
    not parse is named rather than dropped, because a silently-NaT row would
    remove a day from a calendar without saying so.
    """
    if spec.kind not in _DATE_AXIS_KINDS or spec.x not in frame.columns:
        return Result.success(frame)
    column = frame[spec.x]
    if pd.api.types.is_datetime64_any_dtype(column):
        return Result.success(frame)
    with suppress(Exception):
        parsed = pd.to_datetime(column, errors="coerce")
        unparsed = column[parsed.isna() & column.notna()]
        if unparsed.empty:
            converted = frame.copy()
            converted[spec.x] = parsed
            return Result.success(converted)
        examples = ", ".join(repr(str(value)) for value in unparsed.unique()[:3])
        return Result.failure(
            Diagnostic.error(
                "CHART_NOT_DATETIME",
                f"{spec.kind} needs a date column for --x, and {spec.x!r} holds values that are "
                f"not dates: {examples}. Use a column of calendar dates, or correct those rows.",
                kind=spec.kind,
            )
        )
    return Result.success(frame)


def prepare_chart_data(
    frame: pd.DataFrame,
    spec: ChartSpec,
) -> Result[PreparedChart]:
    """Turn an analyst's CSV into the tidy frame the renderer draws.

    Semantics (deliberate, tested, documented in ``docs/viz-charts.md``):

    * **bar/line**: one point per (x, group); duplicates must be aggregated
      with ``--agg`` — refusing to guess is the whole point
      (``CHART_AMBIGUOUS``). Numeric or datetime x keeps its dtype and sorts
      numerically/chronologically; categorical x is text-sorted.
    * **histogram**: common bin edges computed once from the whole dataset;
      every group reports counts over the same bins (zero bins included);
      bin centers/widths are exposed; ``--agg`` is rejected. Only
      ``normalize`` (percent/share over all observations) applies.
    * **box/scatter**: one row per observation, input order preserved;
      ``agg``/``top-n``/``normalize``/rate scaling are rejected for these kinds.
    * **heatmap**: requires ``group``; cells are (x columns by group rows)
      with y as the numeric measure; duplicates require ``agg``.
    * **top-n**: ranks whole x-categories by their total y across all groups
      (ties -> x ascending); runs BEFORE normalization; the order is recorded
      in ``prepared_by["selection"]``.
    * **normalize** (bar/line/histogram): ``percent`` (slices sum to 100) or
      ``share`` (slices sum to 1), computed at the level ``scale_by`` names
      (``total`` | ``group`` | ``category``). Negative values are rejected.
    * **rate scaling** (bar/line, agg=sum): ``y_total / denominator_total *
      rate_per`` per (x, group) — e.g. mentions per 10k tokens. Mean/median/
      count aggregates are rejected for rates; zero denominator totals fail.
    """
    checks = _validate_columns(frame, spec)
    if checks:
        return Result[PreparedChart](None, tuple(checks))
    semantic = _resolve_semantics(spec)
    if semantic:
        return Result[PreparedChart](None, tuple(semantic))
    numeric = _numeric_values(frame, spec)
    if numeric.value is None:
        return Result[PreparedChart](None, numeric.diagnostics)

    return _prepare_chart(frame, spec, numeric.unwrap())


def _prepare_chart(
    frame: pd.DataFrame,
    spec: ChartSpec,
    y_num: pd.Series,
) -> Result[PreparedChart]:
    """The preparation pipeline once inputs are validated."""
    prepared_by = _prepared_by(spec)
    work = _working_frame(frame, spec, y_num)
    if work.value is None:
        return Result[PreparedChart](None, work.diagnostics)
    data = work.unwrap()

    prepared = _prepare_by_kind(data, spec, prepared_by)
    if prepared.value is None:
        return Result[PreparedChart](None, prepared.diagnostics)
    selected, y_label = prepared.unwrap()

    finalized = _finalize_selection(selected, spec, frame, prepared_by, y_label)
    if finalized.value is None:
        return Result[PreparedChart](None, finalized.diagnostics)
    selected, y_label = finalized.unwrap()

    return Result.success(
        PreparedChart(
            data=selected,
            layout=_layout_name(spec),
            prepared_by=prepared_by,
            # A histogram's x axis is the binned VALUE column (spec.x is the
            # ignored category selector, if any was passed at all) — the
            # prepared label names what the axis actually shows. Explicit
            # overrides still win over everything.
            x_label=spec.x_label or (str(spec.y) if spec.kind == "histogram" else str(spec.x)),
            y_label=spec.y_label or y_label,
            spec=spec,
        )
    )


def _finalize_selection(
    prepared: pd.DataFrame,
    spec: ChartSpec,
    frame: pd.DataFrame,
    prepared_by: dict[str, str],
    y_label: str,
) -> Result[tuple[pd.DataFrame, str]]:
    """Post-preparation pipeline: top-n (before scaling), then normalize, then rates.

    ``y_label`` is the default label the kind-specific preparation chose
    (histogram Count, heatmap group axis); it is preserved unless a
    transformation below renames the measure.
    """
    selected = prepared

    if spec.top_n is not None and spec.kind in ("bar", "line", "pie", "sunburst", "treemap", "waffle"):
        ranked = _apply_top_n(selected, spec)
        if ranked.value is None:
            return Result[tuple[pd.DataFrame, str]](None, ranked.diagnostics)
        selected = ranked.unwrap()
        _record_selection(prepared_by, spec, "top_n")

    if spec.kind in ("bar", "line", "pie", "sunburst", "treemap") and spec.normalize in ("percent", "share"):
        level = spec.scale_by if spec.scale_by in ("group", "category") else "total"
        normalized = _apply_normalization(selected, spec, level)
        if normalized.value is None:
            return Result[tuple[pd.DataFrame, str]](None, normalized.diagnostics)
        selected = normalized.unwrap()
        y_label = "Percent" if spec.normalize == "percent" else "Share"
        _record_selection(prepared_by, spec, "normalize")

    if spec.kind in ("bar", "line") and spec.rate_per is not None:
        rates = _apply_rate_scaling(selected, spec, frame)
        if rates.value is None:
            return Result[tuple[pd.DataFrame, str]](None, rates.diagnostics)
        selected = rates.unwrap()
        y_label = f"{spec.y} per {spec.rate_per:g} {spec.denominator_column}"
        _record_selection(prepared_by, spec, "rate scaling")

    if spec.kind == "histogram" and spec.normalize in ("percent", "share"):
        y_label = "Percent" if spec.normalize == "percent" else "Share"
        _record_selection(prepared_by, spec, "normalize")

    return Result.success((selected, y_label))


def _prepared_by(spec: ChartSpec) -> dict[str, str]:
    """The effective data-preparation parameters (provenance for the CSV)."""
    info = {
        "kind": spec.kind,
        "x": spec.x,
        "y": spec.y,
        "group": spec.group or "(none)",
        "agg": spec.agg or "(none)",
        "top_n": str(spec.top_n) if spec.top_n is not None else "(none)",
        "normalize": spec.normalize,
        "rate_per": str(spec.rate_per) if spec.rate_per is not None else "(none)",
        "denominator_column": spec.denominator_column or "(none)",
        "bins": str(spec.bins) if spec.bins is not None else "(auto)",
        "selection": "(none)",
    }
    if spec.top_n is not None and spec.normalize in ("percent", "share"):
        info["selection"] = "top_n then normalize"
    elif spec.top_n is not None:
        info["selection"] = "top_n"
    elif spec.normalize in ("percent", "share"):
        info["selection"] = "normalize"
    return info


def _record_selection(prepared_by: dict[str, str], spec: ChartSpec, stage: str) -> None:
    """Record the actual pipeline order (top-n always runs before scaling)."""
    if spec.top_n is not None:
        prepared_by["selection"] = f"top_n then {stage}"
    else:
        prepared_by["selection"] = stage


def _layout_name(spec: ChartSpec) -> str:
    if spec.kind == "histogram":
        return "bins"
    if spec.kind == "heatmap":
        return "matrix"
    if spec.kind in _HIERARCHY_KINDS:
        return "hierarchy"
    if spec.kind == "waffle":
        return "grid"
    if spec.kind == "calendar":
        return "calendar"
    return "wide"


def _working_frame(frame: pd.DataFrame, spec: ChartSpec, y_num: pd.Series) -> Result[pd.DataFrame]:
    """The tidy working frame: x (dtype-preserved), group, y — group always present."""
    group_values = (
        frame[spec.group].astype(str)
        if spec.group is not None
        else pd.Series([_UNGROUPED] * len(frame), index=frame.index)
    )
    data = pd.DataFrame(
        {"x": frame[spec.x], "group": group_values, "y": y_num.to_numpy()},
        index=frame.index,
    )
    if spec.denominator_column is not None:
        denom = _numeric_denominators(frame, spec.denominator_column)
        if denom.value is None:
            return Result[pd.DataFrame](None, denom.diagnostics)
        data["denominator"] = denom.unwrap().to_numpy()
    return Result.success(data)


def _prepare_by_kind(
    data: pd.DataFrame, spec: ChartSpec, prepared_by: dict[str, str]
) -> Result[tuple[pd.DataFrame, str]]:
    """Dispatch per kind; returns (prepared frame, default y-axis label)."""
    if spec.kind == "histogram":
        # x IS the binned value column: the histogram ignores spec.x as an
        # axis entirely, so the rendered x axis is named by the VALUE column
        # (spec.y), never by the ignored category column.
        return _prepare_histogram(data[["y", "group"]].copy(), spec, prepared_by)
    if spec.kind in _OBSERVATION_KINDS:
        return Result.success((data[["x", "group", "y"]].copy(), spec.y))
    if spec.kind == "heatmap":
        return _prepare_heatmap(data, spec)
    if spec.kind == "calendar":
        return _prepare_calendar(data, spec, prepared_by)
    if spec.kind in ("pie", "sunburst", "treemap"):
        return _prepare_hierarchy(data, spec, single=spec.kind == "pie")
    if spec.kind == "radar":
        return _prepare_radar(data, spec)
    if spec.kind == "waffle":
        return _prepare_waffle(data, spec, prepared_by)
    return _prepare_bar_line(data, spec)  # bar / line


def _aggregate_or_refuse(
    data: pd.DataFrame,
    keys: list[str],
    spec: ChartSpec,
    what: str,
) -> Result[pd.DataFrame]:
    """Shared bar/line/heatmap/pie aggregation: refuse duplicates unless --agg."""
    dup_mask = data.duplicated(subset=keys, keep=False)
    if bool(dup_mask.any()) and spec.agg is None:
        dupes = data.loc[dup_mask, keys].drop_duplicates().head(20)
        examples = [
            f"{x!r} (group {g!r})" if "group" in keys else f"{x!r}"
            for x, *rest in dupes.itertuples(index=False, name=None)
            for g in [rest[0] if rest else ""]
        ]
        label = "(x, group)" if "group" in keys else "x"
        return Result.failure(
            Diagnostic.error(
                "CHART_AMBIGUOUS",
                f"{what} has duplicate observations per {label}; pass --agg "
                f"(one of {AGG_FUNCS}) to say how to combine them — refusing to guess. "
                f"Examples: {examples}",
                examples=examples,
            )
        )
    if spec.agg is None:
        return Result.success(_sorted_by_x(data[["x", "group", "y"]].copy()))
    aggregated = data.groupby(keys, sort=False, as_index=False)["y"].agg(spec.agg)
    checked = _check_finite(aggregated["y"], f"--agg {spec.agg}")
    if checked.value is None:
        return Result[pd.DataFrame](None, checked.diagnostics)
    return Result.success(_sorted_by_x(aggregated))


def _prepare_bar_line(data: pd.DataFrame, spec: ChartSpec) -> Result[tuple[pd.DataFrame, str]]:
    """One row per (x, group); explicit aggregation or refuse duplicates."""
    result = _aggregate_or_refuse(data, ["x", "group"], spec, spec.kind)
    if result.value is None:
        return Result[tuple[pd.DataFrame, str]](None, result.diagnostics)
    y_label = spec.y
    if spec.normalize in ("percent", "share"):
        y_label = "Percent" if spec.normalize == "percent" else "Share"
    elif spec.rate_per is not None:
        y_label = f"{spec.y} per {spec.rate_per:g} {spec.denominator_column}"
    return Result.success((result.unwrap(), y_label))


def _prepare_heatmap(data: pd.DataFrame, spec: ChartSpec) -> Result[tuple[pd.DataFrame, str]]:
    """Cells = (x, group) pairs; y is the numeric measure, never an axis.

    The default y-axis label is the GROUP (row) axis — the rendered heatmap
    has x columns and group rows, while the measure belongs to the colorbar
    (the renderer maps ``prepared.y_label`` there, not to an axis).
    """
    result = _aggregate_or_refuse(data, ["x", "group"], spec, "heatmap")
    if result.value is None:
        return Result[tuple[pd.DataFrame, str]](None, result.diagnostics)
    row_label = spec.group or _UNGROUPED
    return Result.success((result.unwrap(), row_label))


def _prepare_hierarchy(data: pd.DataFrame, spec: ChartSpec, *, single: bool) -> Result[tuple[pd.DataFrame, str]]:
    """Pie / sunburst / treemap: slices from aggregated category values.

    * pie: one level (x categories); --group is rejected at the spec level.
    * sunburst/treemap: two levels — x is the OUTER ring/section, group is
      the INNER child; values aggregate per (x, group) with --agg (refused
      on duplicates otherwise, like bar/line).
    * Negative values are refused (slices encode magnitude only).
    """
    keys = ["x"] if single else ["x", "group"]
    result = _aggregate_or_refuse(data, keys, spec, spec.kind)
    if result.value is None:
        return Result[tuple[pd.DataFrame, str]](None, result.diagnostics)
    prepared = result.unwrap()
    if bool((prepared["y"] < 0).any()):
        bad = prepared.loc[prepared["y"] < 0, keys].head(5).itertuples(index=False, name=None)
        examples = [f"{values!r}" for values in bad]
        return Result.failure(
            Diagnostic.error(
                "CHART_NEGATIVE_SLICE",
                f"{spec.kind} slices encode magnitude; negative values have no slice. Negative cells: {examples}",
                examples=examples,
            )
        )
    y_label = spec.y
    return Result.success((prepared, y_label))


def _prepare_radar(data: pd.DataFrame, spec: ChartSpec) -> Result[tuple[pd.DataFrame, str]]:
    """Radar (line_polar): one spoke per x category, one polygon per group.

    Values are drawn as-is per (x, group); duplicates require --agg. The
    angular axis is categorical (x); the radial axis is y.
    """
    result = _aggregate_or_refuse(data, ["x", "group"], spec, "radar")
    if result.value is None:
        return Result[tuple[pd.DataFrame, str]](None, result.diagnostics)
    return Result.success((result.unwrap(), spec.y))


def _prepare_waffle(
    data: pd.DataFrame, spec: ChartSpec, prepared_by: dict[str, str]
) -> Result[tuple[pd.DataFrame, str]]:
    """Waffle: category shares drawn as a 10x10 (or bar_mode-relative) grid.

    Values aggregate per x category (duplicates require --agg); shares are
    computed over the total, negatives refused. ``prepared_by`` records the
    grid size the renderer will use (100 squares by default).
    """
    result = _aggregate_or_refuse(data, ["x"], spec, "waffle")
    if result.value is None:
        return Result[tuple[pd.DataFrame, str]](None, result.diagnostics)
    prepared = result.unwrap()
    if bool((prepared["y"] < 0).any()):
        return Result.failure(
            Diagnostic.error(
                "CHART_NEGATIVE_SLICE",
                "waffle squares encode a share of the whole; negative values have no square",
            )
        )
    prepared_by["waffle_grid"] = "10x10 (100 squares)"
    return Result.success((prepared, f"{spec.y} share"))


def _prepare_calendar(
    data: pd.DataFrame, spec: ChartSpec, prepared_by: dict[str, str]
) -> Result[tuple[pd.DataFrame, str]]:
    """Calendar heatmap: x is a DATE column, cells are per-day y totals.

    Requires datetime x (``CHART_NOT_DATETIME`` otherwise) and aggregation
    for duplicate dates; y is the cell measure. The renderer draws a
    month-by-(weekday) matrix per year.
    """
    x = data["x"]
    if not pd.api.types.is_datetime64_any_dtype(x):
        return Result.failure(
            Diagnostic.error(
                "CHART_NOT_DATETIME",
                f"calendar needs a date column for --x, got {spec.x!r} "
                f"(dtype {x.dtype}); convert the column to dates first",
                kind="calendar",
            )
        )
    result = _aggregate_or_refuse(data, ["x"], spec, "calendar")
    if result.value is None:
        return Result[tuple[pd.DataFrame, str]](None, result.diagnostics)
    prepared_by["calendar_span"] = f"{x.min():%Y-%m-%d}..{x.max():%Y-%m-%d}"
    return Result.success((result.unwrap(), spec.y))


def _prepare_histogram(
    data: pd.DataFrame, spec: ChartSpec, prepared_by: dict[str, str]
) -> Result[tuple[pd.DataFrame, str]]:
    """Counts per COMMON bin (edges from the whole dataset), per group.

    The histogram binned ``spec.y`` values; the default y-axis label is the
    measure name (Count/Percent/Share), the x axis is the VALUE column —
    finalized below (the renderer labels with the effective labels).
    """
    values = data["y"].to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        return Result.failure(
            Diagnostic.error(
                "CHART_BAD_NUMERIC",
                "histogram values must be finite; non-finite values cannot be binned",
            )
        )
    if data["y"].nunique() == 1 and float(data["y"].iloc[0]) == 0.0:
        edges = np.array([-0.5, 0.5])
    elif spec.bins is not None:
        _, edges = np.histogram(values, bins=spec.bins)
    else:
        _, edges = np.histogram(values)
    prepared = _histogram_counts(data, edges)
    prepared_by["bins"] = str(len(edges) - 1)
    prepared_by["bin_edges"] = ",".join(f"{e:.6g}" for e in edges)
    y_label = "Count"
    if spec.normalize != "none":
        factor = 100.0 if spec.normalize == "percent" else 1.0
        total = float(prepared["count"].sum())
        if total <= 0:
            return Result.failure(
                Diagnostic.error(
                    "CHART_ZERO_DENOMINATOR",
                    "histogram has no observations to normalize (total count is 0)",
                )
            )
        prepared["count_orig"] = prepared["count"]
        prepared["count"] = prepared["count"] / total * factor
        y_label = "Percent" if spec.normalize == "percent" else "Share"
    return Result.success((prepared, y_label))


def _histogram_counts(data: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Count per (bin, group) over shared edges; zero-count bins included."""
    rows: list[dict[str, Any]] = []
    for group in sorted(data["group"].astype(str).unique()):
        mask = data["group"].astype(str) == group
        counts, _ = np.histogram(data.loc[mask, "y"].to_numpy(dtype=float), bins=edges)
        for i, count in enumerate(counts):
            rows.append(
                {
                    "bin_center": (edges[i] + edges[i + 1]) / 2.0,
                    "bin_left": edges[i],
                    "bin_right": edges[i + 1],
                    "bin_width": float(edges[i + 1] - edges[i]),
                    "count": float(count),
                    "group": group,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["bin_center", "bin_left", "bin_right", "bin_width", "count", "group"],
    )


def _apply_top_n(prepared: pd.DataFrame, spec: ChartSpec) -> Result[pd.DataFrame]:
    """Rank whole categories by total y across groups; ties break by x ascending."""
    top_n = spec.top_n
    if top_n is None:
        return Result.failure(Diagnostic.error("CHART_UNSUPPORTED", "--top-n is required for ranking", param="top-n"))
    totals = prepared.groupby("x", sort=False)["y"].sum()
    checked = _check_finite(totals, "top-n category totals")
    if checked.value is None:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if bool((totals < 0).any()):
        return Result.failure(
            Diagnostic.error(
                "CHART_NEGATIVE_VALUE",
                "top-n ranking is undefined over negative totals (ranking against a negative base is meaningless)",
            )
        )
    ranked = sorted(totals.items(), key=lambda item: (-item[1], str(item[0])))
    keep_values = {value for value, _total in ranked[:top_n]}
    selected = prepared[prepared["x"].isin(keep_values)].reset_index(drop=True)
    return Result.success(selected)


def _normalization_denominator(out: pd.DataFrame, level: str) -> Result[tuple[pd.Series, list[str], str]]:
    """Per-row denominators + failing-slice labels, all finiteness-checked.

    Totals are computed with ``_finite_sum`` (no numpy overflow warning);
    an overflowing total becomes a diagnostic, never a silent 0%.
    """
    if level == "category":
        totals = out.groupby("x", sort=False)["y"].agg(_finite_sum)
        checked = _check_finite(totals, "normalization category totals")
        if checked.value is None:
            return Result[tuple[pd.Series, list[str], str]](None, checked.diagnostics)
        denom = out["x"].map(checked.unwrap()).astype(float)
        bad = out.loc[denom <= 0, "x"].astype(str).unique().tolist()
        return Result.success((denom, bad, "category totals"))
    if level == "group":
        totals = out.groupby("group", sort=False)["y"].agg(_finite_sum)
        checked = _check_finite(totals, "normalization group totals")
        if checked.value is None:
            return Result[tuple[pd.Series, list[str], str]](None, checked.diagnostics)
        denom = out["group"].map(checked.unwrap()).astype(float)
        bad = out.loc[denom <= 0, "group"].astype(str).unique().tolist()
        return Result.success((denom, bad, "group totals"))
    total = _finite_sum(out["y"])
    checked = _check_finite(pd.Series([total]), "normalization total")
    if checked.value is None:
        return Result[tuple[pd.Series, list[str], str]](None, checked.diagnostics)
    # Broadcast the VALIDATED SCALAR total; reindexing a one-element Series
    # against out.index would NaN-fill every non-matching index.
    validated_total = float(checked.unwrap().iloc[0])
    denom = pd.Series(validated_total, index=out.index)
    bad = ["(whole chart)"] if validated_total <= 0 else []
    return Result.success((denom, bad, "chart total"))


def _apply_normalization(prepared: pd.DataFrame, spec: ChartSpec, denom_level: str) -> Result[pd.DataFrame]:
    """Percent/share at the named level; negative values and zero bases fail.

    The FINAL transformed values are re-checked for finiteness: a total that
    overflows to +inf (1e308 + 1e308) must fail here, not silently produce
    "0%" for every slice.
    """
    out = prepared.copy()
    y = out["y"].astype(float)
    if bool((y < 0).any()):
        return Result.failure(
            Diagnostic.error(
                "CHART_NEGATIVE_VALUE",
                "--normalize percent/share is undefined over negative values: a negative share "
                "is not a meaningful chart. Filter or transform the data first.",
            )
        )
    denom_result = _normalization_denominator(out, denom_level)
    if denom_result.value is None:
        return Result[pd.DataFrame](None, denom_result.diagnostics)
    denom, bad, label = denom_result.unwrap()
    if bad:
        return Result.failure(
            Diagnostic.error(
                "CHART_ZERO_DENOMINATOR",
                f"normalization denominator ({label}) is zero or negative for: {bad}; refusing "
                "to scale — a share of nothing is not a chart",
                slices=bad,
            )
        )
    factor = 100.0 if spec.normalize == "percent" else 1.0
    out["y_orig"] = y
    transformed = y / denom * factor
    final_check = _check_finite(transformed, "normalized values")
    if final_check.value is None:
        return Result[pd.DataFrame](None, final_check.diagnostics)
    out["y"] = transformed
    return Result.success(out)


def _apply_rate_scaling(prepared: pd.DataFrame, spec: ChartSpec, frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """Rate = aggregated y total / aggregated denominator total x rate_per.

    Requires ``--agg sum`` (a mean/median/count of ratios is not a ratio of
    sums): the denominator is aggregated over the SAME (x, group) cells as
    the numerator, so the ratio is apples-to-apples.
    """
    if spec.rate_per is None or spec.denominator_column is None:
        return Result.failure(
            _unsupported(
                "rate",
                spec.kind,
                "rate scaling needs both rate_per and denominator_column",
            )
        )
    if spec.agg != "sum":
        return Result.failure(
            _unsupported(
                "agg",
                spec.kind,
                f"rate scaling supports --agg sum only (a mean/median/count of ratios is not a "
                f"ratio of sums), got --agg {spec.agg!r}",
            )
        )
    denom = _numeric_denominators(frame, spec.denominator_column)
    if denom.value is None:
        return Result[pd.DataFrame](None, denom.diagnostics)
    work = pd.DataFrame(
        {
            "x": frame[spec.x],
            "group": (
                frame[spec.group].astype(str)
                if spec.group is not None
                else pd.Series([_UNGROUPED] * len(frame), index=frame.index)
            ),
            "denominator": denom.unwrap().to_numpy(),
        },
        index=frame.index,
    )
    return _rate_from_totals(prepared, work, spec.rate_per)


def _rate_from_totals(prepared: pd.DataFrame, work: pd.DataFrame, rate: float) -> Result[pd.DataFrame]:
    """Merge per-cell denominator totals into the prepared frame and scale."""
    denom_totals = work.groupby(["x", "group"], sort=False, as_index=False)["denominator"].sum()
    checked = _check_finite(denom_totals["denominator"], "denominator total")
    if checked.value is None:
        return Result[pd.DataFrame](None, checked.diagnostics)
    merged = prepared.merge(denom_totals, on=["x", "group"], how="left", validate="one_to_one")
    if merged["denominator"].isna().any():
        return Result.failure(
            Diagnostic.error(
                "CHART_RATE_MISMATCH",
                "denominator totals missing for some prepared cells — internal invariant "
                "violated (every cell must carry a denominator)",
            )
        )
    if bool((merged["denominator"] <= 0).any()):
        bad = merged.loc[merged["denominator"] <= 0, ["x", "group"]].astype(str).agg("-".join, axis=1).tolist()
        return Result.failure(
            Diagnostic.error(
                "CHART_ZERO_DENOMINATOR",
                "denominator totals are zero for one or more (x, group) cells; a rate of "
                f"nothing is undefined. Cells: {bad}",
                cells=bad,
            )
        )
    merged["y_orig"] = merged["y"]
    transformed = merged["y"] / merged["denominator"] * rate
    final_check = _check_finite(transformed, "rate values")
    if final_check.value is None:
        return Result[pd.DataFrame](None, final_check.diagnostics)
    merged["y"] = transformed
    out = merged[["x", "group", "y", "y_orig", "denominator"]].reset_index(drop=True)
    return Result.success(out)


def order_x_values(series: pd.Series) -> list[Any]:
    """Public deterministic x ordering (numeric/datetime ascending, else text)."""
    return _order_x(series)
