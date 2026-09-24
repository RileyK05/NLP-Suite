/**
 * Turning a prepared panel into drawable geometry, without React or the DOM.
 *
 * The `panelLayout.ts` counterpart of `chartLayout.ts`, and deliberately the
 * same shape: pure functions of their inputs, no browser, so the numbers a
 * panel claims are testable without rendering anything. The engine
 * (desktop_backend/panels.py) decides what a panel *says* — marks, evidence,
 * annotations, caption — and this file decides only *where* it goes: scales,
 * band positions and colours.
 *
 * Two rules cross the language boundary here and are pinned on both sides by
 * tests:
 *
 * * **Stream stacking happens here, in the renderer's order, with zero-fill
 *   for gaps.** The Python renderer (`core/viz/panel_plotters.py::_stream`)
 *   fills a missing (bucket, group) cell with 0 rather than drawing a line
 *   across the gap, because a straight line through a missing period invents
 *   data and, on a stack, lifts every band above it. `tests/test_panels.py::
 *   TestStream` pins the Python side; `panelLayout.test.ts` pins this one.
 * * **`ranked_bars` order: rank 0 is the first row.** `PanelMark.y` carries
 *   the rank (0 = best). SVG's y axis grows downward, so rank 0 is simply
 *   drawn first, top down. The Python renderer reverses its category array
 *   because *Plotly's* category axis grows upward; do not copy that reversal
 *   here — it would flip the panel. `TestRankedBars` pins Plotly's order and
 *   `panelLayout.test.ts` pins this one independently.
 */

import { OKABE_ITO } from "./chartLayout";
import type {
  PanelAnnotation,
  PanelEvidence,
  PanelMark,
  PreparedPanel,
} from "./api";

/** One laid-out mark: where it goes, what colour it is, and everything a
 *  click has to be answerable with — carried through untouched, because the
 *  wording lives engine-side in `mark.evidence.describe`. */
export type PlacedMark = {
  key: string;
  label: string;
  /** Data coordinates. For `ranked_bars`, y is the rank (0 = first row). */
  x: number;
  y: number;
  group: string;
  size: number | null;
  labelled: boolean;
  color: string;
  /** The row's slot in the layout's own order, so a click can be answered
   *  without recomputing anything. */
  rank: number;
  evidence: PanelEvidence;
};

/** A stacked band for the `stream` shape: one polygon between cumulative
 *  lower and upper y, drawn as its own laid-out series. */
export type StreamBand = {
  group: string;
  color: string;
  /** One x per axis value; lower/upper are cumulative stacked heights. */
  xs: number[];
  lower: number[];
  upper: number[];
  /** One laid-out mark per axis position, in axis order; null where the
   *  band contributes nothing. Already placed (colour, rank, evidence), so a
   *  click on the band still answers. */
  marks: (PlacedMark | null)[];
};

/** An annual series. Separate segments prevent a line through years without
 * dated documents; explicit zero-hit years remain ordinary marks. */
export type LineSeries = {
  group: string;
  color: string;
  segments: PlacedMark[][];
};

/** A laid-out panel: everything `PanelCanvas` draws, decided here. */
export type PanelLayout =
  | { shape: "scatter_labelled"; marks: PlacedMark[]; groups: string[] }
  | {
      shape: "line_series";
      marks: PlacedMark[];
      series: LineSeries[];
      groups: string[];
    }
  | {
      shape: "ranked_bars";
      /** Rows in draw order: rank 0 first. */
      rows: { label: string; rank: number; marks: PlacedMark[] }[];
      groups: string[];
      valueMin: number;
      valueMax: number;
    }
  | {
      shape: "stream";
      /** The ordered numeric axis every band spans. */
      axis: number[];
      bands: StreamBand[];
      groups: string[];
      /** Total stacked height per axis position, for the y scale. */
      totals: number[];
    }
  | {
      shape: "ribbon";
      /** One band per document, top down: y 0 first. */
      bands: RibbonBand[];
      /** The maximum end position any segment reaches, for the x scale. */
      positionMax: number;
      /** True when segment starts are shares (0..1); false when they are
       *  paragraph numbers. The x axis labels follow. */
      relative: boolean;
      groups: string[];
    }
  | {
      shape: "heatmap";
      /** One cell per mark; `x`/`y` are column/row indexes, row 0 at the top. */
      cells: HeatCell[];
      xCategories: string[];
      yCategories: string[];
      /** The declared scale's stops, low to high. */
      stops: string[];
      /** The value range the stops span (`heatDomain`). */
      domain: [number, number];
      /** Colour encodes value here, not group, so there is no group legend. */
      groups: string[];
    }
  | {
      shape: "distribution";
      /** One row per `yCategories` entry, top down; a row with no
       *  observations has no box. */
      rows: DistributionRow[];
      valueMin: number;
      valueMax: number;
      groups: string[];
    }
  | {
      shape: "positions";
      rows: { label: string; row: number; marks: PlacedMark[] }[];
      /** At least 0..1: the whole document, even where nothing falls. */
      positionMin: number;
      positionMax: number;
      groups: string[];
    }
  | {
      shape: "network";
      nodes: PlacedMark[];
      /** Drawn beneath the nodes, in the engine's order. */
      edges: PlacedEdge[];
      groups: string[];
    }
  | {
      shape: "small_multiples";
      /** In `prepared.facets` order, laid out row by row. */
      facets: Facet[];
      columns: number;
      rows: number;
      /** Shared by every facet. */
      xMin: number;
      xMax: number;
      groups: string[];
    };

/** A heatmap cell: a placed mark plus the number its colour encodes. */
export type HeatCell = PlacedMark & { value: number };

/** A distribution point: a placed mark plus its vertical jitter, in row
 *  units (positive is downward, toward the next row). */
export type JitteredMark = PlacedMark & { offset: number };

export type BoxStats = {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
};

export type DistributionRow = {
  label: string;
  row: number;
  stats: BoxStats | null;
  points: JitteredMark[];
};

/** A network link, placed. Carries `key` and `evidence` like a mark, so the
 *  canvas selects and describes it the same way. */
export type PlacedEdge = {
  key: string;
  label: string;
  source: string;
  target: string;
  weight: number;
  /** Stroke width in px (`edgeWidth`). */
  width: number;
  /** The path in the builder's [0, 1] layout coordinates. */
  points: { x: number; y: number }[];
  evidence: PanelEvidence;
};

/** One small multiple: its marks, its series, and its OWN y range. */
export type Facet = {
  name: string;
  marks: PlacedMark[];
  series: LineSeries[];
  yMin: number;
  yMax: number;
};

/** A band of the `ribbon` shape: one document's segments in paragraph
 *  order. A segment with no topic (too short to score) is a gap: no
 *  segment occupies its place on the band. */
export type RibbonBand = {
  /** The document name — the band's row label. */
  label: string;
  /** The band's row (the mark's y, 0 at the top). */
  row: number;
  /** Segments in position order, already placed (colour, evidence). */
  segments: PlacedMark[];
};

export type PanelLayoutResult =
  | { ok: true; layout: PanelLayout; annotations: PanelAnnotation[] }
  | { ok: false; reason: string };

const UNGROUPED = "(all)";

/** Colour per group, by index into the panel's *declared* draw order.
 *
 * Declared order, never a re-sort, so the same topic is the same colour on
 * every run. This mirrors `panel_plotters._group_color_map` exactly, down to
 * the two edge cases, because the app's drawing and the exported PNG of the
 * same panel must not disagree on sight:
 *
 * * a panel with no groups (the intertopic map) colours its one implicit
 *   group, `(all)`, with the first colour;
 * * a group the panel did not declare falls back to the first colour. The
 *   engine refuses to build such a panel, so this is a guard, not a feature.
 *
 * An earlier version hashed undeclared names instead, which coloured every
 * ungrouped panel orange in the app while its export came out blue.
 */
export function groupColors(groups: string[]): Map<string, string> {
  const colors = new Map<string, string>();
  const ordered = groups.length ? groups : [UNGROUPED];
  let index = 0;
  for (const group of ordered) {
    // A remainder group, named in parentheses ("(smaller clusters)"), is
    // grey and takes no palette colour (`static/style.group_colors`).
    if (isRemainder(group)) {
      colors.set(group, REMAINDER);
      continue;
    }
    colors.set(group, OKABE_ITO[index % OKABE_ITO.length]);
    index += 1;
  }
  return colors;
}

const REMAINDER = "#9aa39a";

function isRemainder(group: string): boolean {
  return group.startsWith("(") && group.endsWith(")") && group !== UNGROUPED;
}

function colorFor(colors: Map<string, string>, group: string): string {
  return colors.get(group) ?? OKABE_ITO[0];
}

/** Marks bucketed by group, in the panel's declared order; a group the panel
 *  did not declare follows, sorted, rather than vanishing. Mirrors
 *  `panel_plotters._grouped` so the SVG and the published figure agree on
 *  what order bands and points are drawn in. */
function groupOrder(prepared: PreparedPanel): {
  byGroup: Map<string, PanelMark[]>;
  ordered: string[];
} {
  const byGroup = new Map<string, PanelMark[]>();
  for (const mark of prepared.marks) {
    const name = mark.group || UNGROUPED;
    const bucket = byGroup.get(name);
    if (bucket) bucket.push(mark);
    else byGroup.set(name, [mark]);
  }
  const declared = prepared.groups.length ? prepared.groups : [UNGROUPED];
  const ordered = declared.filter((group) => byGroup.has(group));
  ordered.push(
    ...[...byGroup.keys()].filter((group) => !ordered.includes(group)).sort(),
  );
  return { byGroup, ordered };
}

/** The engine's marks as laid-out points, colours assigned by declared order. */
function layoutScatter(prepared: PreparedPanel): PanelLayout {
  const { byGroup, ordered } = groupOrder(prepared);
  const colors = groupColors(prepared.groups);
  const marks: PlacedMark[] = [];
  for (const group of ordered) {
    const color = colorFor(colors, group);
    for (const mark of byGroup.get(group) ?? []) {
      marks.push(placed(mark, group, color, marks.length));
    }
  }
  return { shape: "scatter_labelled", marks, groups: ordered };
}

/**
 * One group's series, placed into `sink` in x order.
 *
 * A gap in x wider than `prepared.lineGap` (1 for an annual series) starts a
 * new segment, the rule `panel_plotters._broken_line` applies to the export.
 * A group in `pointsOnly` gets no segments at all: its marks are single
 * documents, and joining them in date order draws a trend out of noise.
 */
function lineSeries(
  prepared: PreparedPanel,
  group: string,
  groupMarks: PanelMark[],
  color: string,
  sink: PlacedMark[],
): LineSeries {
  const gap = prepared.lineGap ?? 1;
  const joined = !(prepared.pointsOnly ?? []).includes(group);
  const sorted = [...groupMarks].sort((a, b) => a.x - b.x);
  const segments: PlacedMark[][] = [];
  let previous: number | null = null;
  for (const mark of sorted) {
    const item = placed(mark, group, color, sink.length);
    sink.push(item);
    if (!joined) continue;
    if (previous === null || mark.x - previous > gap) segments.push([]);
    segments[segments.length - 1].push(item);
    previous = mark.x;
  }
  return { group, color, segments };
}

function layoutLineSeries(prepared: PreparedPanel): PanelLayout {
  const { byGroup, ordered } = groupOrder(prepared);
  const colors = groupColors(prepared.groups);
  const marks: PlacedMark[] = [];
  const series = ordered.map((group) =>
    lineSeries(
      prepared,
      group,
      byGroup.get(group) ?? [],
      colorFor(colors, group),
      marks,
    ),
  );
  return { shape: "line_series", marks, series, groups: ordered };
}

function placed(
  mark: PanelMark,
  group: string,
  color: string,
  rank: number,
): PlacedMark {
  return {
    key: mark.key,
    label: mark.label,
    x: mark.x,
    y: mark.y,
    group,
    size: mark.size,
    labelled: mark.labelled,
    color,
    rank,
    evidence: mark.evidence,
  };
}

/** Ranked bars: rank 0 is the first row. SVG y grows downward, so draw order
 *  is rank order — no reversal, unlike the Plotly renderer. */
function layoutRankedBars(prepared: PreparedPanel): PanelLayout {
  const { byGroup, ordered } = groupOrder(prepared);
  const colors = groupColors(prepared.groups);
  // One row per label, ranked by the mark's y (0 = best). A label's rank is
  // taken from the first mark that carries it; the builder puts both series'
  // marks for one term on the same row, so any of them says the same thing.
  const rankOf = new Map<string, number>();
  for (const mark of prepared.marks) {
    if (!rankOf.has(mark.label)) rankOf.set(mark.label, mark.y);
  }
  const rows = [...rankOf.keys()]
    .map((label) => ({ label, rank: rankOf.get(label) ?? 0 }))
    .sort((a, b) => a.rank - b.rank || a.label.localeCompare(b.label));

  const laidOut = rows.map((row, index) => ({
    label: row.label,
    rank: row.rank,
    marks: ordered.flatMap((group) => {
      const color = colorFor(colors, group);
      return (byGroup.get(group) ?? [])
        .filter((mark) => mark.label === row.label)
        .map((mark) => placed(mark, group, color, index));
    }),
  }));
  const valueMin = Math.min(...prepared.marks.map((mark) => mark.x), 0);
  const valueMax = Math.max(...prepared.marks.map((mark) => mark.x), 0);
  return {
    shape: "ranked_bars",
    rows: laidOut,
    groups: ordered,
    valueMin,
    valueMax,
  };
}

/**
 * Stream: cumulative stacked bands over the panel's ordered numeric axis,
 * stacked in `PreparedPanel.groups` order.
 *
 * A group missing at an x value contributes **zero**, not a gap — the same
 * rule `core/viz/panel_plotters.py::_stream` applies. Band order is the
 * declared order, so the same topic is the same colour and the same stacking
 * position on every run.
 */
function layoutStream(prepared: PreparedPanel): PanelLayout {
  const { byGroup, ordered } = groupOrder(prepared);
  const colors = groupColors(prepared.groups);
  const axis = [...new Set(prepared.marks.map((mark) => mark.x))].sort(
    (a, b) => a - b,
  );
  const totals = new Array(axis.length).fill(0);
  const bands: StreamBand[] = [];
  let rank = 0;
  for (const group of ordered) {
    const color = colorFor(colors, group);
    const at = new Map<number, PanelMark>();
    for (const mark of byGroup.get(group) ?? []) {
      // Later marks at the same x would overwrite; builders emit one mark per
      // (bucket, group) cell, and the renderer's own trace holds one y per x.
      at.set(mark.x, mark);
    }
    const lower: number[] = [];
    const upper: number[] = [];
    const marks: (PlacedMark | null)[] = [];
    for (let index = 0; index < axis.length; index++) {
      const value = axis[index];
      const mark = at.get(value) ?? null;
      const contribution = mark ? mark.y : 0;
      lower.push(totals[index]);
      upper.push(totals[index] + contribution);
      marks.push(mark ? placed(mark, group, color, rank++) : null);
      totals[index] += contribution;
    }
    bands.push({ group, color, xs: axis, lower, upper, marks });
  }
  return { shape: "stream", axis, bands, groups: ordered, totals };
}

/**
 * Ribbon: each document as a band of its paragraphs' topics, top down.
 *
 * A mark's `x` is where its segment starts along the band and `size` is how
 * wide it is (same units); `y` is the band's row. SVG y grows downward, so
 * row 0 draws first — no reversal, unlike the Plotly renderer, whose
 * category axis grows upward. Segments are drawn in position order, and a
 * paragraph too short to score is simply absent: a gap, not a recoloured
 * guess.
 */
function layoutRibbon(prepared: PreparedPanel): PanelLayout {
  const { byGroup, ordered } = groupOrder(prepared);
  const colors = groupColors(prepared.groups);
  const byRow = new Map<number, RibbonBand>();
  let rank = 0;
  for (const group of ordered) {
    const color = colorFor(colors, group);
    for (const mark of byGroup.get(group) ?? []) {
      let band = byRow.get(mark.y);
      if (!band) {
        band = { label: mark.label, row: mark.y, segments: [] };
        byRow.set(mark.y, band);
      }
      band.segments.push(placed(mark, group, color, rank++));
    }
  }
  const bands = [...byRow.values()]
    .map((band) => ({
      ...band,
      // Draw segments left to right within the band.
      segments: band.segments.sort((a, b) => a.x - b.x),
    }))
    .sort((a, b) => a.row - b.row);
  const positionMax = Math.max(
    ...prepared.marks.map((mark) => mark.x + (mark.size ?? 0)),
    0,
  );
  // The builder stretches bands to one length in relative alignment; the
  // axis labels differ, and the panel's x label already says which.
  const relative = positionMax <= 1.0001;
  return { shape: "ribbon", bands, positionMax, relative, groups: ordered };
}

// ------------------------------------------------ rules shared with the engine
//
// Each function below restates one rule from `core/viz/panel_plotters.py`,
// with the same arithmetic in the same order so both produce the same
// doubles. `tests/test_panel_parity.py` checks that this file's tests pin the
// numbers the Python tests pin.

/** Heatmap colour scales, low to high. The same stops as
 *  `panelspec.COLOR_SCALES`, which `test_panel_parity.py` compares against
 *  this declaration. */
export const COLOR_SCALES: Record<string, string[]> = {
  sequential: ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
  diverging: ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"],
};

/** The fractional part of the golden ratio: its multiples, mod 1, spread
 *  evenly over [0, 1) at any count (`panel_plotters._JITTER_STEP`). */
const JITTER_STEP = 0.6180339887498949;
const JITTER_SPREAD = 0.25;
const EDGE_MIN_PX = 1;
const EDGE_MAX_PX = 6;

/** The value range a heatmap's colours span: lowest to highest for
 *  `sequential`, symmetric [-m, m] around zero for `diverging`
 *  (`panel_plotters._heat_domain`). */
export function heatDomain(values: number[], scale: string): [number, number] {
  if (!values.length) return [0, 0];
  if (scale === "diverging") {
    const largest = Math.max(...values.map((value) => Math.abs(value)));
    return [-largest, largest];
  }
  return [Math.min(...values), Math.max(...values)];
}

/** A value's colour: linear in RGB between evenly spaced stops, half-up
 *  rounding per channel. A one-value domain maps to the middle of the scale
 *  (`panel_plotters._heat_color`). */
export function heatColor(
  value: number,
  low: number,
  high: number,
  stops: string[],
): string {
  let t = high === low ? 0.5 : (value - low) / (high - low);
  t = Math.min(1, Math.max(0, t));
  const scaled = t * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const fraction = scaled - index;
  const start = rgb(stops[index]);
  const end = rgb(stops[index + 1]);
  return (
    "#" +
    start
      .map((a, channel) =>
        Math.floor(a + (end[channel] - a) * fraction + 0.5)
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")
  );
}

function rgb(hex: string): number[] {
  return [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16));
}

/** Linear interpolation at position (n-1)*p of the sorted values: numpy's
 *  default, "type 7" (`panel_plotters._quantile`). */
export function quantile(ordered: number[], p: number): number {
  const position = (ordered.length - 1) * p;
  const below = Math.floor(position);
  const above = Math.ceil(position);
  return (
    ordered[below] + (ordered[above] - ordered[below]) * (position - below)
  );
}

/** Quartiles, and whiskers to the extremes rather than 1.5 IQR: every point
 *  is drawn over the box, so none is hidden as an outlier. */
export function boxStats(values: number[]): BoxStats {
  const ordered = [...values].sort((a, b) => a - b);
  return {
    min: ordered[0],
    q1: quantile(ordered, 0.25),
    median: quantile(ordered, 0.5),
    q3: quantile(ordered, 0.75),
    max: ordered[ordered.length - 1],
  };
}

/** The vertical offset of a row's `index`-th point, within ±0.25 of the
 *  row's centre. Deterministic, and not stepped in index order -- a builder
 *  that sorts by value would otherwise draw every row as a staircase
 *  (`panel_plotters._jitter`). */
export function jitter(index: number): number {
  const step = index * JITTER_STEP;
  return (step - Math.floor(step) - 0.5) * 2 * JITTER_SPREAD;
}

/** 1px for no weight up to 6px for the heaviest link, linear on
 *  weight / max weight (`panel_plotters._edge_width`). */
export function edgeWidth(weight: number, heaviest: number): number {
  const share = heaviest > 0 ? weight / heaviest : 0;
  return EDGE_MIN_PX + (EDGE_MAX_PX - EDGE_MIN_PX) * share;
}

/** The path an edge takes. `elbow`: source, then (source.x, target.y), then
 *  target -- a dendrogram's right-angled link (`panel_plotters._edge_points`). */
export function edgePoints(
  source: { x: number; y: number },
  target: { x: number; y: number },
  style: string,
): { x: number; y: number }[] {
  if (style === "elbow")
    return [
      { x: source.x, y: source.y },
      { x: source.x, y: target.y },
      { x: target.x, y: target.y },
    ];
  return [
    { x: source.x, y: source.y },
    { x: target.x, y: target.y },
  ];
}

/** Rows and columns for `count` small multiples: at most three across
 *  (`panel_plotters._facet_grid`). */
export function facetGrid(count: number): { rows: number; columns: number } {
  const columns = Math.max(1, Math.min(3, count));
  return { rows: Math.ceil(count / columns), columns };
}

/** Heatmap: one cell per mark, coloured on the declared scale. */
function layoutHeatmap(prepared: PreparedPanel, stops: string[]): PanelLayout {
  const scale = prepared.colorScale ?? "";
  const values = prepared.marks.map((mark) => mark.value ?? 0);
  const domain = heatDomain(values, scale);
  const cells = prepared.marks.map((mark, index) => ({
    ...placed(
      mark,
      mark.group || UNGROUPED,
      heatColor(mark.value ?? 0, domain[0], domain[1], stops),
      index,
    ),
    value: mark.value ?? 0,
  }));
  return {
    shape: "heatmap",
    cells,
    xCategories: prepared.xCategories ?? [],
    yCategories: prepared.yCategories ?? [],
    stops,
    domain,
    groups: [],
  };
}

/** Distribution: a box per row from its observations, and every point with
 *  its jitter. A point's jitter index is its order among its row's marks in
 *  the engine's mark order, as `panel_plotters._jitter_offsets` numbers it. */
function layoutDistribution(prepared: PreparedPanel): PanelLayout {
  const colors = groupColors(prepared.groups);
  const categories = prepared.yCategories ?? [];
  const rows: DistributionRow[] = categories.map((label, row) => ({
    label,
    row,
    stats: null,
    points: [],
  }));
  const seen = new Map<number, number>();
  prepared.marks.forEach((mark, rank) => {
    const row = rows[mark.y];
    if (!row) return;
    const index = seen.get(mark.y) ?? 0;
    seen.set(mark.y, index + 1);
    const group = mark.group || UNGROUPED;
    row.points.push({
      ...placed(mark, group, colorFor(colors, group), rank),
      offset: jitter(index),
    });
  });
  for (const row of rows) {
    if (row.points.length) row.stats = boxStats(row.points.map((p) => p.x));
  }
  const xs = prepared.marks.map((mark) => mark.x);
  return {
    shape: "distribution",
    rows,
    valueMin: Math.min(...xs),
    valueMax: Math.max(...xs),
    groups: groupOrder(prepared).ordered,
  };
}

/** Positions: a tick per mark in its document's row. The x range is at
 *  least 0..1 so the whole document shows, including where nothing falls. */
function layoutPositions(prepared: PreparedPanel): PanelLayout {
  const colors = groupColors(prepared.groups);
  const rows = (prepared.yCategories ?? []).map((label, row) => ({
    label,
    row,
    marks: [] as PlacedMark[],
  }));
  prepared.marks.forEach((mark, rank) => {
    const group = mark.group || UNGROUPED;
    rows[mark.y]?.marks.push(
      placed(mark, group, colorFor(colors, group), rank),
    );
  });
  const xs = prepared.marks.map((mark) => mark.x);
  return {
    shape: "positions",
    rows,
    positionMin: Math.min(0, ...xs),
    positionMax: Math.max(1, ...xs),
    groups: groupOrder(prepared).ordered,
  };
}

/** Network: nodes where the builder put them, edges between their keys. An
 *  edge naming a missing node is skipped; the engine refuses to build one. */
function layoutNetwork(prepared: PreparedPanel): PanelLayout {
  const { byGroup, ordered } = groupOrder(prepared);
  const colors = groupColors(prepared.groups);
  const nodes: PlacedMark[] = [];
  for (const group of ordered) {
    for (const mark of byGroup.get(group) ?? []) {
      nodes.push(placed(mark, group, colorFor(colors, group), nodes.length));
    }
  }
  const byKey = new Map(nodes.map((node) => [node.key, node]));
  const links = prepared.edges ?? [];
  const heaviest = Math.max(0, ...links.map((edge) => edge.weight));
  const style = prepared.edgeStyle ?? "straight";
  const edges: PlacedEdge[] = [];
  for (const edge of links) {
    const source = byKey.get(edge.source);
    const target = byKey.get(edge.target);
    if (!source || !target) continue;
    edges.push({
      key: edge.key,
      label: edge.label,
      source: edge.source,
      target: edge.target,
      weight: edge.weight,
      width: edgeWidth(edge.weight, heaviest),
      points: edgePoints(source, target, style),
      evidence: edge.evidence,
    });
  }
  return { shape: "network", nodes, edges, groups: ordered };
}

/** Small multiples: one facet per declared name, each with its own y range
 *  over one shared x range. Within a facet, series follow `line_series`'s
 *  rules (`lineSeries`). */
function layoutSmallMultiples(prepared: PreparedPanel): PanelLayout {
  const { ordered } = groupOrder(prepared);
  const colors = groupColors(prepared.groups);
  const names = prepared.facets ?? [];
  let rank = 0;
  const facets: Facet[] = names.map((name) => {
    const inFacet = prepared.marks.filter(
      (mark) => (mark.facet ?? "") === name,
    );
    const marks: PlacedMark[] = [];
    const series: LineSeries[] = [];
    for (const group of ordered) {
      const groupMarks = inFacet.filter(
        (mark) => (mark.group || UNGROUPED) === group,
      );
      if (!groupMarks.length) continue;
      const start = marks.length;
      series.push(
        lineSeries(prepared, group, groupMarks, colorFor(colors, group), marks),
      );
      // Ranks run across the whole panel, not restarting per facet.
      for (let index = start; index < marks.length; index++)
        marks[index].rank = rank++;
    }
    const ys = marks.map((mark) => mark.y);
    return {
      name,
      marks,
      series,
      yMin: ys.length ? Math.min(...ys) : 0,
      yMax: ys.length ? Math.max(...ys) : 1,
    };
  });
  const xs = prepared.marks.map((mark) => mark.x);
  const { rows, columns } = facetGrid(facets.length);
  return {
    shape: "small_multiples",
    facets,
    columns,
    rows,
    xMin: Math.min(...xs),
    xMax: Math.max(...xs),
    groups: ordered,
  };
}

/**
 * The one entry point: a prepared panel in, geometry or a plain reason out.
 *
 * A refusal is a sentence to show the reader, not an exception — the engine
 * already refused everything it cannot draw, so what is left is the empty
 * case (a prepared panel with no marks) and nothing else.
 */
export function buildPanelLayout(prepared: PreparedPanel): PanelLayoutResult {
  if (!prepared.marks.length)
    return { ok: false, reason: "This panel has no marks to draw." };
  const annotations = prepared.annotations;
  switch (prepared.shape) {
    case "scatter_labelled":
      return { ok: true, layout: layoutScatter(prepared), annotations };
    case "line_series":
      return { ok: true, layout: layoutLineSeries(prepared), annotations };
    case "ranked_bars":
      return { ok: true, layout: layoutRankedBars(prepared), annotations };
    case "stream":
      return { ok: true, layout: layoutStream(prepared), annotations };
    case "ribbon":
      return { ok: true, layout: layoutRibbon(prepared), annotations };
    case "heatmap": {
      // The engine refuses a heatmap without a known scale; if one arrives
      // anyway, say so rather than colouring it on a scale nobody chose.
      const stops = COLOR_SCALES[prepared.colorScale ?? ""];
      if (!stops)
        return {
          ok: false,
          reason: `This heatmap names an unknown colour scale "${prepared.colorScale ?? ""}".`,
        };
      return { ok: true, layout: layoutHeatmap(prepared, stops), annotations };
    }
    case "distribution":
      return { ok: true, layout: layoutDistribution(prepared), annotations };
    case "positions":
      return { ok: true, layout: layoutPositions(prepared), annotations };
    case "network":
      return { ok: true, layout: layoutNetwork(prepared), annotations };
    case "small_multiples":
      return { ok: true, layout: layoutSmallMultiples(prepared), annotations };
    default:
      return {
        ok: false,
        reason: `The app cannot draw the "${prepared.shape}" panel shape yet.`,
      };
  }
}

/** Rows a clicked mark stands for, as (column, value) pairs ANDed against the
 *  table beside the figure. Evidence is filters, never row indices — see
 *  `docs/viz-panels.md`'s two rules — so this is a plain filter pass. */
export function rowsBehind(
  table: { rows: Record<string, string>[] },
  evidence: PanelEvidence,
): Record<string, string>[] {
  return table.rows.filter((row) =>
    evidence.filters.every(
      ([column, value]) => String(row[column] ?? "") === value,
    ),
  );
}
