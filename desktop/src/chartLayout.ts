/**
 * Turning a result table into chart geometry, without React or the DOM.
 *
 * Everything here is a pure function of (rows, settings). That is what makes
 * the workbench interactive: changing a column or a chart kind is a recompute,
 * not a job submission, so the picture follows the pointer instead of a round
 * trip through the engine.
 *
 * It is deliberately separate from rendering. The numbers a chart claims —
 * what a bar is worth, how many rows it stands for, where a median falls — are
 * the part worth testing, and they are testable here without a browser.
 *
 * This draws the table *currently loaded in the viewer*, which is a page of at
 * most 500 rows. It is a way of looking, not a published figure: every caller
 * says so on screen, and the workbench's Publish button hands the same
 * settings to the engine, which draws the whole file and records provenance.
 */

import { tickLabels } from "./panelTicks";

/** The engine's palette (core/viz/plotters.py), so a preview and a published
 *  chart of the same column do not disagree on sight. Pinned by
 *  tests/test_chart_kind_parity.py. */
export const OKABE_ITO = [
  "#0072B2", // blue
  "#E69F00", // orange
  "#009E73", // green
  "#D55E00", // vermillion
  "#CC79A7", // pink
  "#F0E442", // yellow
  "#56B4E9", // sky blue
  "#000000", // black
] as const;

export type Agg = "" | "sum" | "mean" | "median" | "count";

/** Chart kinds the workbench draws itself, live. This is every kind the
 *  engine draws: the workbench no longer sends a reader away to a job to see
 *  a pie, a treemap or a Sankey. Checked against the engine's CHART_KINDS by
 *  tests/test_chart_kind_parity.py. */
export const LIVE_KINDS = [
  "bar",
  "line",
  "scatter",
  "bubble",
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
  "sankey",
  "ecdf",
] as const;
export type LiveKind = (typeof LIVE_KINDS)[number];

export const isLiveKind = (kind: string): kind is LiveKind =>
  (LIVE_KINDS as readonly string[]).includes(kind);

export const usesMeasureOnly = (kind: string) =>
  kind === "histogram" || kind === "ecdf";

export type Settings = {
  kind: LiveKind;
  x: string;
  y: string;
  group: string;
  agg: Agg;
  topN: number;
  /** Sort categories by value rather than leaving them in table order. */
  sort: "table" | "high" | "low";
  bins: number;
};

/** One drawn thing: a bar, a point, a cell. Carries what it stands for, so a
 *  click can filter the table to exactly those rows. */
export type Mark = {
  /** Stable across redraws: identifies the mark for hover and selection. */
  key: string;
  label: string;
  group: string;
  x: number;
  y: number;
  /** How many table rows this mark summarises. */
  n: number;
  /** Bubble radius input; absent for kinds that do not size their marks. */
  size?: number;
  rowIndices?: number[];
};

export type ChartSelection = {
  label: string;
  group: string;
  key?: string;
  parent?: string | null;
  rowIndices?: number[];
};

export type Series = { group: string; color: string; marks: Mark[] };

export type Box = {
  key: string;
  label: string;
  group: string;
  color: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  n: number;
  /** Every observation behind the box, so the spread is visible rather than
   *  implied by two whiskers. */
  points: number[];
};

export type Cell = {
  key: string;
  row: string;
  column: string;
  value: number;
  n: number;
};

export type Axis = {
  kind: "category" | "number";
  label: string;
  min: number;
  max: number;
  /** Category names in draw order; empty for a numeric axis. */
  categories: string[];
  ticks: { at: number; label: string }[];
};

/** One slice of a pie or doughnut. Angles are radians, clockwise from 12
 *  o'clock; the renderer never recomputes a share. */
export type PieSlice = {
  key: string;
  label: string;
  group: string;
  value: number;
  share: number;
  startAngle: number;
  endAngle: number;
  color: string;
  n: number;
};

/** One arc of a sunburst: an x category on the inner ring, or a group inside
 *  it on the outer ring. `parent` names the arc an outer arc belongs to. */
export type SunburstArc = {
  key: string;
  label: string;
  group: string;
  value: number;
  share: number;
  startAngle: number;
  endAngle: number;
  inner: number;
  outer: number;
  color: string;
  n: number;
  parent: string | null;
};

/** One leaf tile of a treemap, in the plot's unit square. */
export type TreemapTile = {
  key: string;
  label: string;
  group: string;
  value: number;
  share: number;
  /** Unit-square geometry, 0..1. */
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  n: number;
  /** 1 = an x category, 2 = a group inside it. */
  depth: 1 | 2;
  parent: string | null;
};

/** One violin: every observation behind it, its box statistics, and a
 *  density kernel normalised to 0..1. `x` positions are in row units. */
export type ViolinGroup = {
  key: string;
  label: string;
  group: string;
  color: string;
  values: number[];
  stats: {
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
  };
  density: { x: number; y: number }[];
  n: number;
  /** Index of the x category, and the slot for a grouped violin. */
  xIndex: number;
  slot: number;
  slots: number;
};

/** A radar polygon: one value per spoke, in spoke order. */
export type RadarSeries = {
  group: string;
  color: string;
  values: number[];
  n: number;
};

/** One unit square of a waffle. `row` and `column` run 0..9. */
export type WaffleSquare = {
  key: string;
  label: string;
  group: string;
  row: number;
  column: number;
  color: string;
  share: number;
  n: number;
};

/** One day cell of a calendar heatmap, placed on a month row and day-of-month
 *  column. `value` is the aggregated measure for that date. */
export type CalendarCell = {
  key: string;
  label: string;
  group: string;
  value: number;
  n: number;
  month: number;
  day: number;
};

/** A Sankey node, in a two-column layout. Heights are normalised to 0..1. */
export type FlowNode = {
  key: string;
  label: string;
  side: "source" | "target";
  /** Top and bottom in the plot's 0..1 vertical space. */
  y0: number;
  y1: number;
  value: number;
  color: string;
};

/** A ribbon from a source node to a target node. */
export type FlowLink = {
  key: string;
  label: string;
  group: string;
  source: string;
  target: string;
  value: number;
  color: string;
  /** Top and bottom at the source and target ends, 0..1. */
  sy0: number;
  sy1: number;
  ty0: number;
  ty1: number;
};

export type Layout = {
  shape:
    | "bars"
    | "line"
    | "points"
    | "box"
    | "heatmap"
    | "pie"
    | "sunburst"
    | "treemap"
    | "violin"
    | "radar"
    | "waffle"
    | "calendar"
    | "sankey";
  series: Series[];
  boxes: Box[];
  cells: Cell[];
  heatmapMeasure?: string;
  step?: boolean;
  rows: string[];
  columns: string[];
  x: Axis;
  y: Axis;
  /** How many table rows the picture is built from. */
  used: number;
  /** Rows left out, and why — never silently dropped. */
  dropped: number;
  /** Categories removed by the top-N limit. */
  hidden: number;
  /** Bars run left-to-right. Set for a ranking whose labels are too long to
   *  read on a rotated bottom axis; the same categories and values. */
  horizontal?: boolean;
  /** A reason to doubt the picture, shown above it. Drawn anyway: the
   *  chart is not wrong, but reading it as a trend would be. */
  caution?: string;
  /** Geometry for the non-Cartesian kinds. Only the field for `shape` is set. */
  slices?: PieSlice[];
  arcs?: SunburstArc[];
  tiles?: TreemapTile[];
  violins?: ViolinGroup[];
  radar?: { spokes: string[]; series: RadarSeries[]; min: number; max: number };
  waffle?: { squares: WaffleSquare[]; maxShare: number };
  calendar?: {
    months: { key: string; label: string; index: number }[];
    cells: CalendarCell[];
    min: number;
    max: number;
  };
  flows?: { nodes: FlowNode[]; links: FlowLink[] };
};

export type LayoutResult =
  { ok: true; layout: Layout } | { ok: false; reason: string };

// ---------------------------------------------------------------- numbers --

/** A cell's number, or null when it is not one. Blank is missing, not zero. */
export function numberOf(raw: string | undefined): number | null {
  if (raw === undefined || raw === null) return null;
  const text = String(raw).trim();
  if (!text) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
}

export function isNumericColumn(
  rows: Record<string, string>[],
  column: string,
): boolean {
  let seen = 0;
  for (const row of rows) {
    const raw = row[column];
    if (raw === undefined || String(raw).trim() === "") continue;
    if (numberOf(raw) === null) return false;
    seen++;
  }
  return seen > 0;
}

/** Columns worth offering as a measure, in table order. */
export function numericColumns(table: {
  columns: string[];
  rows: Record<string, string>[];
}): string[] {
  return table.columns.filter((column) => isNumericColumn(table.rows, column));
}

/** How many distinct values a column holds, capped so a scan stays cheap. */
export function distinctCount(
  rows: Record<string, string>[],
  column: string,
  cap = 200,
): number {
  const seen = new Set<string>();
  for (const row of rows) {
    seen.add(String(row[column] ?? ""));
    if (seen.size > cap) return cap + 1;
  }
  return seen.size;
}

/** Columns that make sense as a category axis: few enough distinct values. */
export function categoricalColumns(
  table: { columns: string[]; rows: Record<string, string>[] },
  cap = 60,
): string[] {
  return table.columns.filter(
    (column) => distinctCount(table.rows, column, cap) <= cap,
  );
}

/** Columns carrying calendar dates, offered as the calendar's x axis. */
export function dateColumns(table: {
  columns: string[];
  rows: Record<string, string>[];
}): string[] {
  return table.columns.filter((column) => {
    const present = table.rows
      .map((row) => String(row[column] ?? "").trim())
      .filter(Boolean);
    if (!present.length) return false;
    return present
      .slice(0, 50)
      .every((value) => parseCalendarDate(value) !== null);
  });
}

/**
 * Whether a column is a row identifier rather than something to measure.
 *
 * Mirrors `core/insight/profile.py`, including the part that is easy to get
 * wrong: for a **numeric** column the name has to say so as well, because
 * every value being distinct is completely normal in a score column. Judging
 * on uniqueness alone would throw out `Flesch Reading Ease` — the most useful
 * thing in the table — along with `Document ID`.
 *
 * For a **text** column near-uniqueness is enough on its own.
 *
 * Note what this is *not* used for. `core/insight/recommend.py` allows a text
 * identifier as an axis on purpose: document names are unique per row, and
 * "readability by document" is still the chart worth drawing. Only numeric
 * identifiers are meaningless to plot.
 */
const IDENTIFIER_NAMES = ["id", "ids", "rank", "index", "key", "row"];
const UNIQUE_RATIO_FOR_IDENTIFIER = 0.95;
/** Years a corpus of documents could plausibly span, as `profile.py` has it. */
const EARLIEST_YEAR = 1000;
const LATEST_YEAR = 2999;

/**
 * A position in time, which is never the thing being measured.
 *
 * Mirrors the DATE rule in `core/insight/profile.py`. It matters here for one
 * concrete reason: every per-document result over a dated corpus now carries
 * a `Year`, and a column of four-digit integers is, to every other test in
 * this file, an ordinary count. Left alone it became the first measure in the
 * table and the chart opened on "Year by Document" — a staircase, and a
 * worse opening chart than the one this work set out to replace.
 *
 * Summing or averaging a year produces a number about nothing. It belongs on
 * an axis, in order.
 */
export function nameSaysYear(column: string): boolean {
  const words = column
    .toLowerCase()
    .split(/[^a-z]+/)
    .filter(Boolean);
  return words.length > 0 && words[words.length - 1] === "year";
}

export function isYearColumn(
  rows: Record<string, string>[],
  column: string,
): boolean {
  if (!nameSaysYear(column) || !isNumericColumn(rows, column)) return false;
  const present = rows.filter((row) => String(row[column] ?? "").trim() !== "");
  if (!present.length) return false;
  return present.every((row) => {
    const value = numberOf(row[column]);
    return (
      value !== null &&
      Number.isInteger(value) &&
      value >= EARLIEST_YEAR &&
      value <= LATEST_YEAR
    );
  });
}

export function nameSaysIdentifier(column: string): boolean {
  return column
    .toLowerCase()
    .split(/[^a-z]+/)
    .filter(Boolean)
    .some((word) => IDENTIFIER_NAMES.includes(word));
}

export function looksLikeIdentifier(
  rows: Record<string, string>[],
  column: string,
): boolean {
  const present = rows.filter((row) => String(row[column] ?? "").trim() !== "");
  const distinct = new Set(present.map((row) => String(row[column]))).size;
  // One distinct value is CONSTANT, not IDENTIFIER -- the same order the
  // engine checks them in, and the reason isConstant exists below.
  if (distinct <= 1) return false;
  const nearUnique = distinct / present.length >= UNIQUE_RATIO_FOR_IDENTIFIER;
  if (isNumericColumn(rows, column)) {
    const whole = present.every((row) =>
      Number.isInteger(numberOf(row[column]) ?? 0.5),
    );
    return nameSaysIdentifier(column) && (nearUnique || whole);
  }
  return nameSaysIdentifier(column) || nearUnique;
}

/**
 * A column holding the same value in every row.
 *
 * The engine calls this CONSTANT and never charts it: there is nothing to
 * compare. A one-document run publishes a table where *every* column is
 * constant, which is how `Document ID` came to be the measure a live chart
 * opened on.
 */
export function isConstant(
  rows: Record<string, string>[],
  column: string,
): boolean {
  const present = rows.filter((row) => String(row[column] ?? "").trim() !== "");
  return new Set(present.map((row) => String(row[column]))).size <= 1;
}

/**
 * Measures worth opening on: numeric, not a row number, not all one value.
 *
 * The name test does the work that the value test cannot. `looksLikeIdentifier`
 * mirrors the engine faithfully, and the engine reaches "do not chart this"
 * by two different routes: IDENTIFIER when an ID column's values vary, and
 * CONSTANT when they do not. A one-document run publishes a table where every
 * column holds exactly one value, so `Document ID` took the second route,
 * came back "not an identifier", and became the measure the chart opened on.
 * For a numeric column the engine's identifier rule requires the name to say
 * so anyway, so leaning on the name here agrees with it and survives a table
 * with one row.
 *
 * The fallbacks are ordered by which mistake is worse. Charting an identifier
 * is meaningless; charting a constant is merely dull.
 *
 * A year is excluded for the same reason as an identifier: it is where a row
 * sits, not what the row is worth.
 */
export function measureColumns(table: {
  columns: string[];
  rows: Record<string, string>[];
}): string[] {
  const numeric = numericColumns(table);
  const measurable = numeric.filter(
    (column) =>
      !nameSaysIdentifier(column) &&
      !looksLikeIdentifier(table.rows, column) &&
      !isYearColumn(table.rows, column),
  );
  const varying = measurable.filter(
    (column) => !isConstant(table.rows, column),
  );
  return varying.length ? varying : measurable.length ? measurable : numeric;
}

export function median(sorted: number[]): number {
  if (!sorted.length) return 0;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Linear-interpolated quantile, matching the convention numpy uses. */
export function quantile(sorted: number[], fraction: number): number {
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export function reduce(values: number[], agg: Agg): number {
  if (agg === "count") return values.length;
  if (!values.length) return 0;
  if (agg === "mean") return values.reduce((a, b) => a + b, 0) / values.length;
  if (agg === "median") return median([...values].sort((a, b) => a - b));
  return values.reduce((a, b) => a + b, 0); // sum, and the default
}

// ------------------------------------------------------------------ ticks --

/** Axis ticks at round numbers, chosen so the labels stay readable.
 *
 *  Every tick on one axis is printed at the same precision: formatting each
 *  value on its own produced "2.50" next to "5" (FIGURE_RECIPES R1). The
 *  shared rule lives in `panelTicks.tickLabels`, used by the panel layer, so
 *  the two chart surfaces cannot disagree about an axis. */
export function niceTicks(
  min: number,
  max: number,
  count = 5,
): { at: number; label: string }[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max)
    return [{ at: min, label: tickLabels([min])[0] ?? formatNumber(min) }];
  const span = max - min;
  const rough = span / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ??
    magnitude * 10;
  const values: number[] = [];
  for (
    let at = Math.ceil(min / step) * step;
    at <= max + step / 1000;
    at += step
  ) {
    // Floating-point accumulation drifts; snap each tick back to the step grid.
    values.push(Math.round(at / step) * step);
  }
  const labels = tickLabels(values);
  return values.map((at, index) => ({
    at,
    label: labels[index] ?? formatNumber(at),
  }));
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Object.is(value, -0) || Math.abs(value) < 1e-12) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e4) return `${(value / 1e3).toFixed(0)}k`;
  if (magnitude >= 1) {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
  }
  return value.toPrecision(2);
}

// ----------------------------------------------------------------- layout --

const colorFor = (index: number) => OKABE_ITO[index % OKABE_ITO.length];
const UNGROUPED = "(all)";

type Row = Record<string, string>;

/** Rows that carry a usable measure, plus how many were set aside. */
function usable(
  rows: Row[],
  y: string,
  agg: Agg,
): { kept: { row: Row; value: number }[]; dropped: number } {
  const kept: { row: Row; value: number }[] = [];
  let dropped = 0;
  for (const row of rows) {
    const value = numberOf(row[y]);
    // Counting asks how many rows there are, so a row with no measure still counts.
    if (value === null && agg !== "count") {
      dropped++;
      continue;
    }
    kept.push({ row, value: value ?? 0 });
  }
  return { kept, dropped };
}

function orderCategories(
  totals: Map<string, number>,
  order: string[],
  sort: Settings["sort"],
  topN: number,
): { categories: string[]; hidden: number } {
  let categories = [...order];
  if (sort !== "table") {
    categories.sort((a, b) => {
      const difference = (totals.get(b) ?? 0) - (totals.get(a) ?? 0);
      return sort === "high" ? difference : -difference;
    });
  }
  const limit = topN > 0 ? topN : categories.length;
  const hidden = Math.max(0, categories.length - limit);
  if (hidden) {
    // Rank by value even in table order: "top 20" means the largest twenty,
    // not the first twenty that happen to appear.
    //
    // Ties break by name, which is what chartspec's top_n rule does. Without
    // it a tie at the cut is broken by table order here and alphabetically
    // there, so the preview and the published chart would hold *different
    // categories* — a disagreement about the data, not just its arrangement.
    const ranked = [...categories]
      .sort(
        (a, b) =>
          (totals.get(b) ?? 0) - (totals.get(a) ?? 0) || a.localeCompare(b),
      )
      .slice(0, limit);
    const keep = new Set(ranked);
    categories = categories.filter((name) => keep.has(name));
  }
  return { categories, hidden };
}

function categoryAxis(categories: string[], label: string): Axis {
  return {
    kind: "category",
    label,
    min: 0,
    max: Math.max(1, categories.length),
    categories,
    ticks: categories.map((name, index) => ({ at: index + 0.5, label: name })),
  };
}

function numberAxis(label: string, min: number, max: number): Axis {
  // A bar chart is read against zero; a scatter of a narrow range is not.
  const low = Math.min(min, max);
  const high = Math.max(min, max);
  const padded =
    low === high ? { min: low - 1, max: high + 1 } : { min: low, max: high };
  return {
    kind: "number",
    label,
    min: padded.min,
    max: padded.max,
    categories: [],
    ticks: niceTicks(padded.min, padded.max),
  };
}

const empty = (reason: string): LayoutResult => ({ ok: false, reason });

/** Group values, in first-seen order, so colours stay put between redraws. */
function groupsOf(rows: { row: Row }[], group: string): string[] {
  if (!group) return [UNGROUPED];
  const seen: string[] = [];
  for (const { row } of rows) {
    const name = String(row[group] ?? "");
    if (!seen.includes(name)) seen.push(name);
  }
  return seen;
}

function buildCategorical(
  rows: Row[],
  settings: Settings,
  shape: "bars" | "line",
): LayoutResult {
  const { x, y, group, agg } = settings;
  const { kept, dropped } = usable(rows, y, agg);
  if (!kept.length) return empty(`No rows have a number in "${y}".`);

  const groups = groupsOf(kept, group);
  const order: string[] = [];
  const totals = new Map<string, number>();
  const buckets = new Map<string, number[]>();
  for (const { row, value } of kept) {
    const category = String(row[x] ?? "");
    const name = group ? String(row[group] ?? "") : UNGROUPED;
    if (!order.includes(category)) order.push(category);
    const key = `${category}\u0000${name}`;
    const bucket = buckets.get(key);
    if (bucket) bucket.push(value);
    else buckets.set(key, [value]);
    totals.set(category, (totals.get(category) ?? 0) + value);
  }
  // A line is read left to right, so it runs in the axis's own order and is
  // never cut: "largest first" shuffles the years, and keeping the twenty
  // largest deletes the quiet ones and draws straight through where they were.
  if (shape === "line") {
    order.sort((a, b) => (orderedValue(a) ?? 0) - (orderedValue(b) ?? 0));
  }
  const { categories, hidden } = orderCategories(
    totals,
    order,
    shape === "line" ? "table" : settings.sort,
    shape === "line" ? 0 : settings.topN,
  );
  if (!categories.length) return empty("Nothing left to draw after the limit.");

  const series: Series[] = groups.map((name, index) => ({
    group: name,
    color: colorFor(index),
    marks: categories.flatMap((category) => {
      const values = buckets.get(`${category}\u0000${name}`);
      if (!values) return [];
      return [
        {
          key: `${category}\u0000${name}`,
          label: category,
          group: name,
          x: categories.indexOf(category) + 0.5,
          y: reduce(values, settings.agg || "sum"),
          n: values.length,
        },
      ];
    }),
  }));
  const values = series.flatMap((s) => s.marks.map((m) => m.y));
  // Twenty-five words on a rotated bottom axis is a wall of near-vertical
  // text. The same ranking reads in one pass as horizontal bars, with the
  // words where the eye starts. Lines are never turned on their side.
  const horizontal =
    shape === "bars" &&
    categories.length > 10 &&
    Math.max(0, ...categories.map((category) => category.length)) > 9;
  return {
    ok: true,
    layout: {
      shape,
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(categories, x),
      // Bars and lines over categories are read against zero.
      y: numberAxis(y, Math.min(0, ...values), Math.max(0, ...values)),
      used: kept.length,
      dropped,
      hidden,
      horizontal,
    },
  };
}

function buildPoints(rows: Row[], settings: Settings): LayoutResult {
  const { x, y, group, kind } = settings;
  const points: { row: Row; index: number; xv: number; yv: number }[] = [];
  let dropped = 0;
  for (const [index, row] of rows.entries()) {
    const xv = numberOf(row[x]);
    const yv = numberOf(row[y]);
    if (xv === null || yv === null) {
      dropped++;
      continue;
    }
    points.push({ row, index, xv, yv });
  }
  if (!points.length)
    return empty(`"${x}" and "${y}" must both hold numbers for a ${kind}.`);

  const groups = groupsOf(
    points.map(({ row }) => ({ row })),
    group,
  );
  const magnitudes = points.map((p) => Math.abs(p.yv));
  const largest = Math.max(...magnitudes) || 1;
  const series: Series[] = groups.map((name, index) => ({
    group: name,
    color: colorFor(index),
    marks: points
      .filter(({ row }) => (group ? String(row[group] ?? "") === name : true))
      .map(({ row, index, xv, yv }) => ({
        key: `point-${index}`,
        label: String(row[x] ?? ""),
        group: name,
        x: xv,
        y: yv,
        n: 1,
        rowIndices: [index],
        size: kind === "bubble" ? Math.abs(yv) / largest : undefined,
      })),
  }));
  const xs = points.map((p) => p.xv);
  const ys = points.map((p) => p.yv);
  return {
    ok: true,
    layout: {
      shape: "points",
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: numberAxis(x, Math.min(...xs), Math.max(...xs)),
      y: numberAxis(y, Math.min(...ys), Math.max(...ys)),
      used: points.length,
      dropped,
      hidden: 0,
    },
  };
}

function buildHistogram(rows: Row[], settings: Settings): LayoutResult {
  const { y, group } = settings;
  const kept: { row: Row; index: number; value: number }[] = [];
  let dropped = 0;
  for (const [index, row] of rows.entries()) {
    const value = numberOf(row[y]);
    if (value === null) {
      dropped++;
      continue;
    }
    kept.push({ row, index, value });
  }
  if (!kept.length) return empty(`No rows have a number in "${y}".`);

  const values = kept.map((k) => k.value);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const count =
    high === low
      ? 1
      : Math.max(1, Math.min(60, Math.round(settings.bins) || 12));
  // Edges are computed once over every value, so grouped bars are comparable.
  const width = (high - low) / count || 1;
  const edges = Array.from({ length: count + 1 }, (_, i) => low + i * width);
  const groups = groupsOf(kept, group);
  const series: Series[] = groups.map((name, index) => {
    const mine = kept.filter(({ row }) =>
      group ? String(row[group] ?? "") === name : true,
    );
    const bins = Array.from({ length: count }, () => [] as number[]);
    for (const { value, index: rowIndex } of mine) {
      const slot =
        value === high ? count - 1 : Math.floor((value - low) / width);
      bins[Math.max(0, Math.min(count - 1, slot))].push(rowIndex);
    }
    return {
      group: name,
      color: colorFor(index),
      marks: bins.map((rowIndices, slot) => ({
        key: `${name}\u0000${slot}`,
        label: `${formatNumber(edges[slot])} – ${formatNumber(edges[slot + 1])}`,
        group: name,
        x: slot + 0.5,
        y: rowIndices.length,
        n: rowIndices.length,
        rowIndices,
      })),
    };
  });
  const tallest = Math.max(
    ...series.flatMap((s) => s.marks.map((m) => m.y)),
    1,
  );
  return {
    ok: true,
    layout: {
      shape: "bars",
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: {
        kind: "category",
        label: y,
        min: 0,
        max: count,
        categories: series[0]?.marks.map((m) => m.label) ?? [],
        // Every bin labelled would be unreadable; label the ends and middle.
        ticks: [...new Set([0, Math.floor(count / 2), count])].map((slot) => ({
          at: slot,
          label: formatNumber(edges[Math.min(slot, count)]),
        })),
      },
      y: numberAxis("Rows", 0, tallest),
      used: kept.length,
      dropped,
      hidden: 0,
    },
  };
}

function buildEcdf(rows: Row[], settings: Settings): LayoutResult {
  const { y, group } = settings;
  const { kept, dropped } = usable(rows, y, "");
  if (!kept.length) return empty(`No rows have a number in "${y}".`);
  const groups = groupsOf(kept, group).sort();
  const series = groups.map((name, index) => {
    const counts = new Map<number, number>();
    const values = kept.filter(
      ({ row }) => !group || String(row[group] ?? "") === name,
    );
    for (const { value } of values)
      counts.set(value, (counts.get(value) ?? 0) + 1);
    let n = 0;
    const marks = [...counts]
      .sort(([a], [b]) => a - b)
      .map(([x, count]) => {
        n += count;
        return {
          key: `${name}\u0000${x}`,
          label: String(x),
          group: name,
          x,
          y: (n / values.length) * 100,
          n,
        };
      });
    return { group: name, color: colorFor(index), marks };
  });
  const values = kept.map(({ value }) => value);
  return {
    ok: true,
    layout: {
      shape: "line",
      step: true,
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: numberAxis(y, Math.min(...values), Math.max(...values)),
      y: numberAxis("At or below (%)", 0, 100),
      used: kept.length,
      dropped,
      hidden: 0,
    },
  };
}

function buildBox(rows: Row[], settings: Settings): LayoutResult {
  const { x, y } = settings;
  const { kept, dropped } = usable(rows, y, "");
  if (!kept.length) return empty(`No rows have a number in "${y}".`);

  const order: string[] = [];
  const buckets = new Map<string, number[]>();
  const totals = new Map<string, number>();
  for (const { row, value } of kept) {
    const category = String(row[x] ?? "");
    if (!order.includes(category)) order.push(category);
    const bucket = buckets.get(category);
    if (bucket) bucket.push(value);
    else buckets.set(category, [value]);
    totals.set(category, (totals.get(category) ?? 0) + value);
  }
  const { categories, hidden } = orderCategories(
    totals,
    order,
    settings.sort,
    settings.topN,
  );
  const boxes: Box[] = categories.map((category, index) => {
    const values = [...(buckets.get(category) ?? [])].sort((a, b) => a - b);
    return {
      key: category,
      label: category,
      group: category,
      color: colorFor(index),
      min: values[0] ?? 0,
      q1: quantile(values, 0.25),
      median: median(values),
      q3: quantile(values, 0.75),
      max: values[values.length - 1] ?? 0,
      n: values.length,
      points: values,
    };
  });
  if (!boxes.length) return empty("Nothing left to draw after the limit.");
  const lows = boxes.map((b) => b.min);
  const highs = boxes.map((b) => b.max);
  return {
    ok: true,
    layout: {
      shape: "box",
      series: [],
      boxes,
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(categories, x),
      y: numberAxis(y, Math.min(...lows), Math.max(...highs)),
      used: kept.length,
      dropped,
      hidden,
    },
  };
}

function buildHeatmap(rows: Row[], settings: Settings): LayoutResult {
  const { x, y, group, agg } = settings;
  if (!group)
    return empty("A heatmap needs a Group column: it is the row axis.");
  const { kept, dropped } = usable(rows, y, agg);
  if (!kept.length) return empty(`No rows have a number in "${y}".`);

  const columns: string[] = [];
  const rowNames: string[] = [];
  const buckets = new Map<string, number[]>();
  const totals = new Map<string, number>();
  for (const { row, value } of kept) {
    const column = String(row[x] ?? "");
    const rowName = String(row[group] ?? "");
    if (!columns.includes(column)) columns.push(column);
    if (!rowNames.includes(rowName)) rowNames.push(rowName);
    const key = `${rowName}\u0000${column}`;
    const bucket = buckets.get(key);
    if (bucket) bucket.push(value);
    else buckets.set(key, [value]);
    totals.set(column, (totals.get(column) ?? 0) + value);
  }
  const { categories, hidden } = orderCategories(
    totals,
    columns,
    settings.sort,
    settings.topN,
  );
  const cells: Cell[] = [];
  for (const rowName of rowNames) {
    for (const column of categories) {
      const values = buckets.get(`${rowName}\u0000${column}`);
      if (!values) continue;
      cells.push({
        key: `${rowName}\u0000${column}`,
        row: rowName,
        column,
        value: reduce(values, agg || "sum"),
        n: values.length,
      });
    }
  }
  if (!cells.length) return empty("Nothing left to draw after the limit.");
  return {
    ok: true,
    layout: {
      shape: "heatmap",
      series: [],
      boxes: [],
      cells,
      heatmapMeasure: agg === "count" ? "Rows" : y,
      rows: rowNames,
      columns: categories,
      x: categoryAxis(categories, x),
      y: categoryAxis(rowNames, group),
      used: kept.length,
      dropped,
      hidden,
    },
  };
}

// ------------------------------------------------ non-Cartesian kinds --
//
// The engine draws these too (core/viz/plotters.py). The geometry below is
// computed here, never in the drawing component, so the numbers a slice or a
// ribbon claims are testable without a browser. Angles are radians clockwise
// from twelve o'clock; unit-square coordinates run left-to-right, top-to-bottom.

const TAU = Math.PI * 2;
const TWELVE = -Math.PI / 2;

/** Mix a hex colour toward white (positive) or black (negative). */
function shade(hex: string, amount: number): string {
  const value = hex.replace("#", "");
  if (!/^[0-9a-f]{6}$/i.test(value)) return hex;
  const parts = [0, 2, 4].map((at) => parseInt(value.slice(at, at + 2), 16));
  const toward = amount >= 0 ? 255 : 0;
  const mix = Math.min(1, Math.abs(amount));
  return `#${parts
    .map((part) =>
      Math.round(part + (toward - part) * mix)
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
}

type CategoryBucket = {
  categories: string[];
  totals: Map<string, number>;
  values: Map<string, number[]>;
  byGroup: Map<string, number[]>;
  groups: string[];
  hidden: number;
  kept: number;
  dropped: number;
};

/** Aggregate a measure by x category, keeping the group split as well. */
function categoryBuckets(
  rows: Row[],
  settings: Settings,
  sort: Settings["sort"] = settings.sort,
  topN = settings.topN,
): CategoryBucket {
  const { x, y, group, agg } = settings;
  const { kept, dropped } = usable(rows, y, agg);
  const order: string[] = [];
  const totals = new Map<string, number>();
  const values = new Map<string, number[]>();
  const byGroup = new Map<string, number[]>();
  const groups: string[] = [];
  for (const { row, value } of kept) {
    const category = String(row[x] ?? "");
    const name = group ? String(row[group] ?? "") : UNGROUPED;
    if (!order.includes(category)) order.push(category);
    if (!groups.includes(name)) groups.push(name);
    totals.set(category, (totals.get(category) ?? 0) + value);
    const valueBucket = values.get(category);
    if (valueBucket) valueBucket.push(value);
    else values.set(category, [value]);
    const key = `${category}\u0000${name}`;
    const groupBucket = byGroup.get(key);
    if (groupBucket) groupBucket.push(value);
    else byGroup.set(key, [value]);
  }
  const { categories, hidden } = orderCategories(totals, order, sort, topN);
  return {
    categories,
    totals,
    values,
    byGroup,
    groups,
    hidden,
    kept: kept.length,
    dropped,
  };
}

function buildPie(rows: Row[], settings: Settings): LayoutResult {
  const data = categoryBuckets(rows, settings);
  const total = data.categories.reduce(
    (sum, category) => sum + (data.totals.get(category) ?? 0),
    0,
  );
  if (total <= 0)
    return empty(
      "A pie needs a positive total; these values sum to zero or less.",
    );
  if (data.categories.some((category) => (data.totals.get(category) ?? 0) < 0))
    return empty(
      "A pie encodes magnitude: negative values cannot be drawn as slices.",
    );

  let angle = TWELVE;
  const slices: PieSlice[] = data.categories.map((category, index) => {
    const value = data.totals.get(category) ?? 0;
    const share = value / total;
    const startAngle = angle;
    angle += share * TAU;
    return {
      key: `${category}\u0000slice`,
      label: category,
      group: category,
      value,
      share,
      startAngle,
      endAngle: angle,
      color: colorFor(index),
      n: data.values.get(category)?.length ?? 0,
    };
  });
  const series: Series[] = slices.map((slice) => ({
    group: slice.label,
    color: slice.color,
    marks: [
      {
        key: slice.key,
        label: slice.label,
        group: slice.group,
        x: 0.5,
        y: slice.value,
        n: slice.n,
      },
    ],
  }));
  return {
    ok: true,
    layout: {
      shape: "pie",
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(data.categories, settings.x),
      y: numberAxis(settings.y, 0, total),
      used: data.kept,
      dropped: data.dropped,
      hidden: data.hidden,
      slices,
    },
  };
}

function buildSunburst(rows: Row[], settings: Settings): LayoutResult {
  const { group } = settings;
  if (!group)
    return empty(
      "A sunburst needs two levels: choose a Group column for the outer ring.",
    );
  const data = categoryBuckets(rows, settings);
  const total = data.categories.reduce(
    (sum, category) => sum + (data.totals.get(category) ?? 0),
    0,
  );
  if (total <= 0) return empty("A sunburst needs a positive total to divide.");

  const arcs: SunburstArc[] = [];
  let angle = TWELVE;
  const series: Series[] = [];
  data.categories.forEach((category, index) => {
    const value = data.totals.get(category) ?? 0;
    const share = value / total;
    const startAngle = angle;
    const endAngle = angle + share * TAU;
    const color = colorFor(index);
    arcs.push({
      key: `${category}\u0000ring`,
      label: category,
      group: category,
      value,
      share,
      startAngle,
      endAngle,
      inner: 0.3,
      outer: 0.62,
      color,
      n: data.values.get(category)?.length ?? 0,
      parent: null,
    });
    series.push({
      group: category,
      color,
      marks: [
        {
          key: `${category}\u0000ring`,
          label: category,
          group: category,
          x: 0.5,
          y: value,
          n: data.values.get(category)?.length ?? 0,
        },
      ],
    });
    // Children inside the category's wedge, in first-seen group order.
    let childAngle = startAngle;
    data.groups.forEach((name, slot) => {
      const values = data.byGroup.get(`${category}\u0000${name}`);
      if (!values?.length) return;
      const childValue = reduce(values, settings.agg || "sum");
      const childShare = childValue / total;
      if (childShare <= 0) return;
      const childStart = childAngle;
      childAngle += childShare * TAU;
      arcs.push({
        key: `${category}\u0000${name}`,
        label: `${category} · ${name}`,
        group: name,
        value: childValue,
        share: childShare,
        startAngle: childStart,
        endAngle: childAngle,
        inner: 0.66,
        outer: 0.98,
        color: shade(color, 0.12 + slot * 0.08),
        n: values.length,
        parent: category,
      });
    });
    angle = endAngle;
  });
  return {
    ok: true,
    layout: {
      shape: "sunburst",
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(data.categories, settings.x),
      y: numberAxis(settings.y, 0, total),
      used: data.kept,
      dropped: data.dropped,
      hidden: data.hidden,
      arcs,
    },
  };
}

type Rect = { x: number; y: number; width: number; height: number };

/** Slice a rectangle into proportional strips, alternating direction at each
 *  level: top-level categories across, their groups down. */
function sliceRect(
  values: number[],
  box: Rect,
  direction: "row" | "column",
): Rect[] {
  const total = values.reduce((sum, value) => sum + Math.max(0, value), 0);
  if (total <= 0) return values.map(() => ({ ...box, width: 0, height: 0 }));
  const rects: Rect[] = [];
  let offset = 0;
  for (const value of values) {
    const share = Math.max(0, value) / total;
    if (direction === "row") {
      rects.push({
        x: box.x + offset,
        y: box.y,
        width: box.width * share,
        height: box.height,
      });
      offset += box.width * share;
    } else {
      rects.push({
        x: box.x,
        y: box.y + offset,
        width: box.width,
        height: box.height * share,
      });
      offset += box.height * share;
    }
  }
  return rects;
}

function buildTreemap(rows: Row[], settings: Settings): LayoutResult {
  const { group } = settings;
  if (!group)
    return empty(
      "A treemap needs two levels: choose a Group column for the tiles inside each block.",
    );
  const data = categoryBuckets(rows, settings);
  const total = data.categories.reduce(
    (sum, category) => sum + (data.totals.get(category) ?? 0),
    0,
  );
  if (total <= 0) return empty("A treemap needs a positive total to divide.");
  if (data.categories.some((category) => (data.totals.get(category) ?? 0) < 0))
    return empty(
      "A treemap encodes magnitude: negative values cannot be drawn as area.",
    );

  const blocks = sliceRect(
    data.categories.map((category) => data.totals.get(category) ?? 0),
    { x: 0, y: 0, width: 1, height: 1 },
    "row",
  );
  const tiles: TreemapTile[] = [];
  const series: Series[] = [];
  data.categories.forEach((category, index) => {
    const box = blocks[index];
    const color = colorFor(index);
    const children = data.groups
      .map((name) => ({
        name,
        values: data.byGroup.get(`${category}\u0000${name}`) ?? [],
      }))
      .filter((child) => child.values.length);
    const childBoxes = sliceRect(
      children.map((child) => reduce(child.values, settings.agg || "sum")),
      box,
      "column",
    );
    children.forEach((child, childIndex) => {
      const value = reduce(child.values, settings.agg || "sum");
      if (value <= 0) return;
      const rect = childBoxes[childIndex];
      tiles.push({
        key: `${category}\u0000${child.name}`,
        label: `${category} · ${child.name}`,
        group: child.name,
        value,
        share: value / total,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        color: shade(color, 0.06 + (childIndex % 4) * 0.08),
        n: child.values.length,
        depth: 2,
        parent: category,
      });
    });
    series.push({
      group: category,
      color,
      marks: [
        {
          key: `${category}\u0000block`,
          label: category,
          group: category,
          x: 0,
          y: data.totals.get(category) ?? 0,
          n: data.values.get(category)?.length ?? 0,
        },
      ],
    });
  });
  return {
    ok: true,
    layout: {
      shape: "treemap",
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(data.categories, settings.x),
      y: numberAxis(settings.y, 0, total),
      used: data.kept,
      dropped: data.dropped,
      hidden: data.hidden,
      tiles,
    },
  };
}

function boxStats(values: number[]): ViolinGroup["stats"] {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    min: sorted[0] ?? 0,
    q1: quantile(sorted, 0.25),
    median: median(sorted),
    q3: quantile(sorted, 0.75),
    max: sorted[sorted.length - 1] ?? 0,
  };
}

/** A Gaussian kernel density on a fixed grid, normalised to 0..1. */
function density(values: number[]): { x: number; y: number }[] {
  if (values.length < 2) return [];
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    Math.max(1, values.length - 1);
  const sd = Math.sqrt(variance);
  const bandwidth = Math.max(1.06 * sd * values.length ** -0.2, 1e-9);
  const low = Math.min(...values) - bandwidth * 3;
  const high = Math.max(...values) + bandwidth * 3;
  const steps = 48;
  const points = Array.from({ length: steps }, (_, index) => {
    const x = low + ((high - low) * index) / (steps - 1);
    const y =
      values.reduce(
        (sum, value) =>
          sum +
          Math.exp(-0.5 * ((x - value) / bandwidth) ** 2) /
            (bandwidth * Math.sqrt(TAU)),
        0,
      ) / values.length;
    return { x, y };
  });
  const peak = Math.max(...points.map((point) => point.y), 1e-12);
  return points.map((point) => ({ x: point.x, y: point.y / peak }));
}

function buildViolin(rows: Row[], settings: Settings): LayoutResult {
  const data = categoryBuckets(rows, settings, settings.sort, settings.topN);
  if (!data.categories.length)
    return empty("Nothing left to draw after the limit.");
  const grouped = data.groups[0] !== UNGROUPED;
  const slots = grouped ? data.groups.length : 1;
  const violins: ViolinGroup[] = [];
  const series: Series[] = [];
  let low = Infinity;
  let high = -Infinity;
  data.categories.forEach((category, xIndex) => {
    data.groups.forEach((name, slot) => {
      const values = data.byGroup.get(`${category}\u0000${name}`);
      if (!values?.length) return;
      const stats = boxStats(values);
      low = Math.min(low, stats.min);
      high = Math.max(high, stats.max);
      const color = grouped ? colorFor(slot) : colorFor(xIndex);
      violins.push({
        key: `${category}\u0000${name}`,
        label: category,
        group: grouped ? name : category,
        color,
        values,
        stats,
        density: density(values),
        n: values.length,
        xIndex,
        slot: grouped ? slot : 0,
        slots,
      });
    });
    if (!grouped) {
      const values = data.values.get(category) ?? [];
      if (values.length) {
        series.push({
          group: category,
          color: colorFor(xIndex),
          marks: [
            {
              key: `${category}\u0000violin`,
              label: category,
              group: category,
              x: xIndex + 0.5,
              y: boxStats(values).median,
              n: values.length,
            },
          ],
        });
      }
    }
  });
  if (grouped) {
    data.groups.forEach((name, slot) => {
      series.push({
        group: name,
        color: colorFor(slot),
        marks: [],
      });
    });
  }
  if (!violins.length) return empty("Nothing left to draw after the limit.");
  if (!Number.isFinite(low) || !Number.isFinite(high))
    return empty(`No rows have a number in "${settings.y}".`);
  return {
    ok: true,
    layout: {
      shape: "violin",
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(data.categories, settings.x),
      y: numberAxis(settings.y, low, high),
      used: data.kept,
      dropped: data.dropped,
      hidden: data.hidden,
      violins,
    },
  };
}

function buildRadar(rows: Row[], settings: Settings): LayoutResult {
  const data = categoryBuckets(rows, settings, settings.sort, 0);
  if (data.categories.length < 3)
    return empty(
      "A radar needs at least three measures on its spokes; this table has fewer.",
    );
  const grouped = data.groups[0] !== UNGROUPED;
  const series: RadarSeries[] = [];
  const markSeries: Series[] = [];
  let low = 0;
  let high = 0;
  data.groups.forEach((name, index) => {
    const values = data.categories.map((category) => {
      const bucket = data.byGroup.get(`${category}\u0000${name}`);
      if (!bucket?.length) return 0;
      return reduce(bucket, settings.agg || "sum");
    });
    for (const value of values) {
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
    const color = grouped ? colorFor(index) : colorFor(0);
    series.push({ group: name, color, values, n: values.length });
    markSeries.push({
      group: name,
      color,
      marks: data.categories.map((category, spoke) => ({
        key: `${category}\u0000${name}`,
        label: category,
        group: name,
        x: spoke + 0.5,
        y: values[spoke],
        n: data.byGroup.get(`${category}\u0000${name}`)?.length ?? 0,
      })),
    });
  });
  if (high === low)
    return empty(
      "Every radar value is the same, so the shape would say nothing.",
    );
  return {
    ok: true,
    layout: {
      shape: "radar",
      series: markSeries,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(data.categories, settings.x),
      y: numberAxis(settings.y, low, high),
      used: data.kept,
      dropped: data.dropped,
      hidden: data.hidden,
      radar: {
        spokes: data.categories,
        series,
        min: low,
        max: high,
      },
    },
  };
}

function buildWaffle(rows: Row[], settings: Settings): LayoutResult {
  const data = categoryBuckets(rows, settings);
  const total = data.categories.reduce(
    (sum, category) => sum + (data.totals.get(category) ?? 0),
    0,
  );
  if (total <= 0)
    return empty("A waffle needs a positive total; these values sum to zero.");
  if (data.categories.some((category) => (data.totals.get(category) ?? 0) < 0))
    return empty(
      "A waffle encodes share: negative values cannot fill squares.",
    );

  const shares = data.categories.map(
    (category) => (data.totals.get(category) ?? 0) / total,
  );
  const counts = shares.map((share) => Math.floor(share * 100));
  let remainder = 100 - counts.reduce((sum, count) => sum + count, 0);
  const ranked = shares
    .map((share, index) => ({ share, index }))
    .sort((a, b) => b.share - a.share || a.index - b.index);
  for (const entry of ranked) {
    if (remainder <= 0) break;
    counts[entry.index] += 1;
    remainder -= 1;
  }
  const squares: WaffleSquare[] = [];
  let cursor = 0;
  data.categories.forEach((category, index) => {
    for (let unit = 0; unit < counts[index]; unit++) {
      const row = Math.floor(cursor / 10);
      const column = cursor % 10;
      squares.push({
        key: `${category}\u0000${cursor}`,
        label: category,
        group: category,
        row,
        column,
        color: colorFor(index),
        share: shares[index],
        n: data.values.get(category)?.length ?? 0,
      });
      cursor++;
    }
  });
  const series: Series[] = data.categories.map((category, index) => ({
    group: category,
    color: colorFor(index),
    marks: [
      {
        key: `${category}\u0000waffle`,
        label: category,
        group: category,
        x: 0.5,
        y: (data.totals.get(category) ?? 0) / total,
        n: data.values.get(category)?.length ?? 0,
      },
    ],
  }));
  return {
    ok: true,
    layout: {
      shape: "waffle",
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(data.categories, settings.x),
      y: numberAxis(settings.y, 0, total),
      used: data.kept,
      dropped: data.dropped,
      hidden: data.hidden,
      waffle: { squares, maxShare: Math.max(...shares) },
    },
  };
}

/** A calendar date from an ISO-ish cell, or null. */
export function parseCalendarDate(
  raw: string,
): { year: number; month: number; day: number; iso: string } | null {
  const text = String(raw ?? "").trim();
  const match = text.match(
    /^(\d{4})([-/])(\d{1,2})\2(\d{1,2})(?:$|[T ]\d{2}:\d{2})/,
  );
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[3]);
  const day = Number(match[4]);
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;
  const iso = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
  if (new Date(`${iso}T00:00:00Z`).getUTCDate() !== day) return null;
  return { year, month, day, iso };
}

function buildCalendar(rows: Row[], settings: Settings): LayoutResult {
  const { x, y, agg } = settings;
  const byDate = new Map<string, { n: number; values: number[] }>();
  let dropped = 0;
  for (const row of rows) {
    const parsed = parseCalendarDate(String(row[x] ?? ""));
    const value = numberOf(row[y]);
    if (!parsed || (value === null && agg !== "count")) {
      dropped++;
      continue;
    }
    const bucket = byDate.get(parsed.iso) ?? { n: 0, values: [] };
    bucket.n++;
    if (value !== null) bucket.values.push(value);
    byDate.set(parsed.iso, bucket);
  }
  if (!byDate.size)
    return empty(
      `A calendar needs dates in "${x}"; no cell looked like YYYY-MM-DD.`,
    );
  const days = [...byDate.keys()].sort();
  const months = [...new Set(days.map((day) => day.slice(0, 7)))].sort();
  const cells: CalendarCell[] = [];
  let low = Infinity;
  let high = -Infinity;
  for (const day of days) {
    const parsed = parseCalendarDate(day);
    const bucket = byDate.get(day);
    if (!parsed || !bucket) continue;
    const value =
      bucket.values.length || agg === "count"
        ? reduce(bucket.values, agg || "sum")
        : 0;
    low = Math.min(low, value);
    high = Math.max(high, value);
    cells.push({
      key: day,
      label: day,
      group: "",
      value,
      n: bucket.n,
      month: months.indexOf(day.slice(0, 7)),
      day: parsed.day,
    });
  }
  return {
    ok: true,
    layout: {
      shape: "calendar",
      series: [],
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(months, x),
      y: numberAxis(y, Math.min(0, low), high),
      used: rows.length - dropped,
      dropped,
      hidden: 0,
      calendar: {
        months: months.map((key, index) => ({ key, label: key, index })),
        cells,
        min: low,
        max: high,
      },
    },
  };
}

function buildSankey(rows: Row[], settings: Settings): LayoutResult {
  const { x, y, group } = settings;
  if (!group)
    return empty(
      "A Sankey needs a target: choose a Group column for the flow's right-hand side.",
    );
  const links = new Map<
    string,
    { source: string; target: string; value: number; n: number }
  >();
  let dropped = 0;
  for (const row of rows) {
    const value = numberOf(row[y]);
    if (value === null) {
      dropped++;
      continue;
    }
    if (value < 0)
      return empty("A Sankey encodes flow: negative weights cannot be drawn.");
    const source = String(row[x] ?? "");
    const target = String(row[group] ?? "");
    const key = `${source}\u0000${target}`;
    const link = links.get(key) ?? { source, target, value: 0, n: 0 };
    link.value += value;
    link.n++;
    links.set(key, link);
  }
  if (!links.size) return empty("No rows have a flow to draw.");
  const kept = [...links.values()].filter((link) => link.value > 0);
  const total = kept.reduce((sum, link) => sum + link.value, 0);
  if (total <= 0) return empty("The flow's weights sum to zero.");
  if (!Number.isFinite(total))
    return empty("The flow's weights exceed the supported numeric range.");

  const sources = [...new Set(kept.map((link) => link.source))];
  const targets = [...new Set(kept.map((link) => link.target))];
  const sourceTotals = new Map(sources.map((name) => [name, 0]));
  const targetTotals = new Map(targets.map((name) => [name, 0]));
  for (const link of kept) {
    sourceTotals.set(
      link.source,
      (sourceTotals.get(link.source) ?? 0) + link.value,
    );
    targetTotals.set(
      link.target,
      (targetTotals.get(link.target) ?? 0) + link.value,
    );
  }
  const maxGaps = Math.max(sources.length, targets.length) - 1;
  const gap = maxGaps ? Math.min(0.025, 0.25 / maxGaps) : 0;
  const available = 1 - gap * maxGaps;
  const stack = (
    names: string[],
    totals: Map<string, number>,
    side: "source" | "target",
  ): FlowNode[] => {
    let cursor = (1 - available - gap * (names.length - 1)) / 2;
    return names.map((name, index) => {
      const value = totals.get(name) ?? 0;
      const height = (value / total) * available;
      const node: FlowNode = {
        key: `${side}\u0000${name}`,
        label: name,
        side,
        y0: cursor,
        y1: cursor + height,
        value,
        color: colorFor(index),
      };
      cursor += height + gap;
      return node;
    });
  };
  const sourceNodes = stack(sources, sourceTotals, "source");
  const targetNodes = stack(targets, targetTotals, "target");
  const sourceAt = new Map(
    sourceNodes.map((node) => [node.label, { ...node, cursor: node.y0 }]),
  );
  const targetAt = new Map(
    targetNodes.map((node) => [node.label, { ...node, cursor: node.y0 }]),
  );
  const flows: FlowLink[] = kept.map((link) => {
    const source = sourceAt.get(link.source);
    const target = targetAt.get(link.target);
    if (!source || !target) {
      return {
        key: `${link.source}\u0000${link.target}`,
        label: link.source,
        group: link.target,
        source: link.source,
        target: link.target,
        value: 0,
        color: "#999999",
        sy0: 0,
        sy1: 0,
        ty0: 0,
        ty1: 0,
      };
    }
    const height = (link.value / total) * available;
    const sy0 = source.cursor;
    const ty0 = target.cursor;
    source.cursor += height;
    target.cursor += height;
    return {
      key: `${link.source}\u0000${link.target}`,
      label: link.source,
      group: link.target,
      source: link.source,
      target: link.target,
      value: link.value,
      color: source.color,
      sy0,
      sy1: sy0 + height,
      ty0,
      ty1: ty0 + height,
    };
  });
  const series: Series[] = sourceNodes.map((node) => ({
    group: node.label,
    color: node.color,
    marks: [
      {
        key: node.key,
        label: node.label,
        group: "",
        x: 0,
        y: node.value,
        n: 0,
      },
    ],
  }));
  return {
    ok: true,
    layout: {
      shape: "sankey",
      series,
      boxes: [],
      cells: [],
      rows: [],
      columns: [],
      x: categoryAxis(sources, x),
      y: categoryAxis(targets, group),
      used: rows.length - dropped,
      dropped,
      hidden: 0,
      flows: { nodes: [...sourceNodes, ...targetNodes], links: flows },
    },
  };
}

const ISO_DAY = /^\d{4}(-\d{2}(-\d{2})?)?$/;

/**
 * Where a value sits on an ordered axis: a date or a number. Null for a
 * label, which has no place in an order ("of" does not come before "and" in
 * any sense a line could show).
 */
export function orderedValue(raw: string): number | null {
  const text = String(raw ?? "").trim();
  if (ISO_DAY.test(text)) {
    const [year, month = 0, day = 0] = text.split("-").map(Number);
    return year * 10_000 + month * 100 + day;
  }
  return numberOf(text);
}

/** The first value of *column* that has no order, or null if every value has one. */
function unorderedValue(rows: Row[], column: string): string | null {
  for (const row of rows) {
    const text = String(row[column] ?? "").trim();
    if (text && orderedValue(text) === null) return text;
  }
  return null;
}

/** Columns naming the document a row came from. Averaging over documents
 *  that share a date is a statement about that date; averaging over the
 *  items inside them is not. */
const DOCUMENT_COLUMN = /^(document|document id|doc|file|file ?name)$/i;
/** Past this many distinct items per point on average, a point is a pool. */
const POOLED = 1.5;

/**
 * Whether each point of a line pools rows for many *different items*.
 *
 * The co-occurrence table has one row per word pair per speech. Averaged by
 * Date, each point is the mean count of about a hundred unrelated pairs,
 * which moves with the table's min-count cut and with speech length, not
 * with anything the pairs have in common. The line is drawn, because the
 * arithmetic is right; this says why it is not a trend.
 */
function pooledItems(rows: Row[], settings: Settings): string | null {
  const { x, y, group } = settings;
  const candidates = Object.keys(rows[0] ?? {}).filter(
    (column) =>
      column !== x &&
      column !== y &&
      column !== group &&
      !DOCUMENT_COLUMN.test(column) &&
      !nameSaysYear(column) &&
      !isNumericColumn(rows, column),
  );
  let worst: { column: string; perPoint: number } | null = null;
  for (const column of candidates) {
    const items = new Map<string, Set<string>>();
    for (const row of rows) {
      const at = String(row[x] ?? "");
      const seen = items.get(at) ?? new Set<string>();
      seen.add(String(row[column] ?? ""));
      items.set(at, seen);
    }
    const perPoint =
      [...items.values()].reduce((total, seen) => total + seen.size, 0) /
      Math.max(1, items.size);
    if (perPoint > POOLED && (!worst || perPoint > worst.perPoint)) {
      worst = { column, perPoint };
    }
  }
  if (!worst) return null;
  return `Each point combines rows for about ${Math.round(worst.perPoint)} different "${worst.column}" values, so it mixes unrelated items into one number. Colour by "${worst.column}", or filter the table to one value, before reading this as a trend.`;
}

/**
 * The one entry point: settings in, geometry or a plain reason out.
 *
 * A refusal is a sentence to show the reader, not an exception — choosing a
 * text column as a measure is an ordinary thing to do while exploring, and the
 * answer is to say so and keep the controls live.
 */
export function buildLayout(
  rows: Record<string, string>[],
  settings: Settings,
): LayoutResult {
  if (!rows.length) return empty("This page of the table has no rows.");
  if (!settings.y) return empty("Choose a measure to draw.");
  switch (settings.kind) {
    case "bar":
      return buildCategorical(rows, settings, "bars");
    case "line": {
      const unordered = unorderedValue(rows, settings.x);
      if (unordered !== null) {
        return empty(
          `A line joins its points in order, and "${settings.x}" has no order: its values are labels, such as "${unordered}". Draw bars to compare them, or choose a date or a number to run along the bottom.`,
        );
      }
      const built = buildCategorical(rows, settings, "line");
      if (!built.ok) return built;
      const caution = pooledItems(rows, settings);
      return caution
        ? { ok: true, layout: { ...built.layout, caution } }
        : built;
    }
    case "scatter":
    case "bubble":
      return buildPoints(rows, settings);
    case "histogram":
      return buildHistogram(rows, settings);
    case "ecdf":
      return buildEcdf(rows, settings);
    case "box":
      return buildBox(rows, settings);
    case "heatmap":
      return buildHeatmap(rows, settings);
    case "pie":
      return buildPie(rows, settings);
    case "sunburst":
      return buildSunburst(rows, settings);
    case "treemap":
      return buildTreemap(rows, settings);
    case "violin":
      return buildViolin(rows, settings);
    case "radar":
      return buildRadar(rows, settings);
    case "waffle":
      return buildWaffle(rows, settings);
    case "calendar":
      return buildCalendar(rows, settings);
    case "sankey":
      return buildSankey(rows, settings);
  }
}

/** Rows a clicked mark stands for, for filtering the table beneath the chart. */
export function rowsBehind(
  rows: Record<string, string>[],
  settings: Settings,
  mark: ChartSelection,
): Record<string, string>[] {
  if (mark.rowIndices)
    return mark.rowIndices.map((index) => rows[index]).filter(Boolean);
  if (settings.kind === "ecdf")
    return rows.filter((row) => {
      const value = numberOf(row[settings.y]);
      return (
        value !== null &&
        value <= Number(mark.label) &&
        (!settings.group || String(row[settings.group] ?? "") === mark.group)
      );
    });
  if (settings.kind === "calendar")
    return rows.filter(
      (row) =>
        parseCalendarDate(row[settings.x])?.iso === mark.label &&
        (settings.agg === "count" || numberOf(row[settings.y]) !== null),
    );
  // A hierarchy mark is either a top-level category ("Word") or a leaf
  // ("Word · Group"). The leaf's x is the part before the separator, not the
  // whole label; filtering on the label would find nothing.
  const hierarchy = settings.kind === "sunburst" || settings.kind === "treemap";
  const parent = hierarchy ? (mark.parent ?? mark.label) : mark.label;
  return rows.filter((row) => {
    if (settings.agg !== "count" && numberOf(row[settings.y]) === null)
      return false;
    const category = String(row[settings.x] ?? "");
    if (settings.kind !== "histogram" && category !== parent) return false;
    // A top-level hierarchy mark is one category, all its groups.
    if (hierarchy && mark.parent == null) return true;
    if (settings.group && String(row[settings.group] ?? "") !== mark.group)
      return false;
    return true;
  });
}

// ---------------------------------------------------------------- publish --

/** Kinds the engine will accept an aggregation for (chartspec `_AGG_KINDS`,
 *  plus heatmap, whose cells are aggregated by the engine's own path). */
const ENGINE_TAKES_AGG = new Set<string>([
  "bar",
  "line",
  "heatmap",
  "pie",
  "sunburst",
  "treemap",
  "radar",
  "waffle",
  "calendar",
  "sankey",
]);
/** Kinds the engine will rank (chartspec's top-n rule). */
const ENGINE_TAKES_TOP_N = new Set<string>([
  "bar",
  "line",
  "pie",
  "sunburst",
  "treemap",
  "waffle",
]);

/**
 * These settings as `table_charts` parameters, for the engine to draw properly.
 *
 * The workbench draws a page of rows in the browser; publishing draws the
 * whole file and records a run. The translation has to respect the engine's
 * rules exactly, because a parameter it refuses fails *after* the job has been
 * queued and run — sending `--agg` to a scatter, or `--top-n` to a heatmap,
 * would turn a click into a failed run with a diagnostic about a control the
 * reader never touched. `tests/test_chart_kind_parity.py` checks these two
 * sets against what the engine actually accepts.
 */
export function publishParams(
  settings: Settings,
  title = "",
): Record<string, unknown> {
  return {
    kind: settings.kind,
    x: usesMeasureOnly(settings.kind) ? settings.y : settings.x,
    y: settings.y,
    group: settings.group,
    agg: ENGINE_TAKES_AGG.has(settings.kind) ? settings.agg || "sum" : "",
    normalize: "none",
    "top-n":
      ENGINE_TAKES_TOP_N.has(settings.kind) && settings.topN > 0
        ? settings.topN
        : null,
    bins:
      settings.kind === "histogram"
        ? Math.max(1, Math.round(settings.bins))
        : null,
    title,
    format: "html",
  };
}
