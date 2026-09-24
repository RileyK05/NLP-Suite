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

/** Chart kinds the workbench draws itself, live. The rest are engine-only and
 *  the workbench says so rather than pretending. Checked against the engine's
 *  CHART_KINDS by tests/test_chart_kind_parity.py. */
export const LIVE_KINDS = [
  "bar",
  "line",
  "scatter",
  "bubble",
  "histogram",
  "box",
  "heatmap",
] as const;
export type LiveKind = (typeof LIVE_KINDS)[number];

export const isLiveKind = (kind: string): kind is LiveKind =>
  (LIVE_KINDS as readonly string[]).includes(kind);

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

export type Layout = {
  shape: "bars" | "line" | "points" | "box" | "heatmap";
  series: Series[];
  boxes: Box[];
  cells: Cell[];
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
  /** A reason to doubt the picture, shown above it. Drawn anyway: the
   *  chart is not wrong, but reading it as a trend would be. */
  caution?: string;
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

/** Axis ticks at round numbers, chosen so the labels stay readable. */
export function niceTicks(
  min: number,
  max: number,
  count = 5,
): { at: number; label: string }[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [{ at: min, label: formatNumber(min) }];
  const span = max - min;
  const rough = span / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ??
    magnitude * 10;
  const ticks: { at: number; label: string }[] = [];
  for (
    let at = Math.ceil(min / step) * step;
    at <= max + step / 1000;
    at += step
  ) {
    // Floating-point accumulation drifts; snap each tick back to the step grid.
    const snapped = Math.round(at / step) * step;
    ticks.push({ at: snapped, label: formatNumber(snapped) });
  }
  return ticks;
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e4) return `${(value / 1e3).toFixed(0)}k`;
  if (magnitude === 0 || magnitude >= 1) {
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
    },
  };
}

function buildPoints(rows: Row[], settings: Settings): LayoutResult {
  const { x, y, group, kind } = settings;
  const points: { row: Row; xv: number; yv: number }[] = [];
  let dropped = 0;
  for (const row of rows) {
    const xv = numberOf(row[x]);
    const yv = numberOf(row[y]);
    if (xv === null || yv === null) {
      dropped++;
      continue;
    }
    points.push({ row, xv, yv });
  }
  if (!points.length)
    return empty(`"${x}" and "${y}" must both hold numbers for a ${kind}.`);

  const groups = groupsOf(
    points.map(({ row }) => ({ row })),
    group,
  );
  const magnitudes = points.map((p) => Math.abs(p.yv));
  const largest = Math.max(...magnitudes, 1);
  const series: Series[] = groups.map((name, index) => ({
    group: name,
    color: colorFor(index),
    marks: points
      .filter(({ row }) => (group ? String(row[group] ?? "") === name : true))
      .map(({ row, xv, yv }, position) => ({
        key: `${name}\u0000${position}\u0000${xv}\u0000${yv}`,
        label: String(row[x] ?? ""),
        group: name,
        x: xv,
        y: yv,
        n: 1,
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
  const kept: { row: Row; value: number }[] = [];
  let dropped = 0;
  for (const row of rows) {
    const value = numberOf(row[y]);
    if (value === null) {
      dropped++;
      continue;
    }
    kept.push({ row, value });
  }
  if (!kept.length) return empty(`No rows have a number in "${y}".`);

  const values = kept.map((k) => k.value);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const count = Math.max(1, Math.min(60, Math.round(settings.bins) || 12));
  // Edges are computed once over every value, so grouped bars are comparable.
  const width = (high - low) / count || 1;
  const edges = Array.from({ length: count + 1 }, (_, i) => low + i * width);
  const groups = groupsOf(kept, group);
  const series: Series[] = groups.map((name, index) => {
    const mine = kept.filter(({ row }) =>
      group ? String(row[group] ?? "") === name : true,
    );
    const counts = new Array(count).fill(0);
    for (const { value } of mine) {
      const slot =
        value === high ? count - 1 : Math.floor((value - low) / width);
      counts[Math.max(0, Math.min(count - 1, slot))]++;
    }
    return {
      group: name,
      color: colorFor(index),
      marks: counts.map((n, slot) => ({
        key: `${name}\u0000${slot}`,
        label: `${formatNumber(edges[slot])} – ${formatNumber(edges[slot + 1])}`,
        group: name,
        x: slot + 0.5,
        y: n,
        n,
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
        ticks: [0, Math.floor(count / 2), count].map((slot) => ({
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
    case "box":
      return buildBox(rows, settings);
    case "heatmap":
      return buildHeatmap(rows, settings);
  }
}

/** Rows a clicked mark stands for, for filtering the table beneath the chart. */
export function rowsBehind(
  rows: Record<string, string>[],
  settings: Settings,
  mark: { label: string; group: string },
): Record<string, string>[] {
  return rows.filter((row) => {
    const category = String(row[settings.x] ?? "");
    if (settings.kind !== "histogram" && category !== mark.label) return false;
    if (settings.group && String(row[settings.group] ?? "") !== mark.group)
      return false;
    return true;
  });
}

// ---------------------------------------------------------------- publish --

/** Kinds the engine will accept an aggregation for (chartspec `_AGG_KINDS`). */
const ENGINE_TAKES_AGG = new Set<string>(["bar", "line", "heatmap"]);
/** Kinds the engine will rank (chartspec's top-n rule). */
const ENGINE_TAKES_TOP_N = new Set<string>(["bar", "line"]);

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
    x: settings.x,
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
