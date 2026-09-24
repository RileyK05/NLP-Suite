import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  ChartColumnBig,
  Download,
  Filter,
  Info,
  RotateCcw,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import type { RecommendedChart, Table } from "./api";
import {
  LIVE_KINDS,
  isLiveKind,
  buildLayout,
  categoricalColumns,
  distinctCount,
  formatNumber,
  isNumericColumn,
  measureColumns,
  numericColumns,
  rowsBehind,
  type Agg,
  type LiveKind,
  type Settings,
} from "./chartLayout";
import { ChartCanvas, type Selection } from "./ChartCanvas";
import { ResultTable } from "./ResultTable";

/**
 * Looking at a result, rather than asking for a picture of one.
 *
 * Before this, seeing a chart meant filling in a dialog, submitting a job,
 * waiting for the engine, and opening a static file — for every change of
 * mind. Choosing the wrong column cost a round trip. So the charts people
 * actually made were the ones they already knew they wanted, which is the
 * opposite of exploring.
 *
 * Here the picture follows the controls. Everything redraws from the table
 * already in the browser, so switching measure, kind or grouping is instant,
 * and clicking a bar filters the rows underneath it.
 *
 * Two things it is careful about:
 *
 * * **It draws the page on screen, not the whole file**, and says so plainly.
 *   A preview that quietly showed 500 of 40,000 rows would be a way to reach
 *   a wrong conclusion quickly.
 * * **It publishes nothing.** A run directory carries provenance — inputs,
 *   hashes, settings, diagnostics — and an interactive view has none of that.
 *   "Publish this chart" hands these exact settings to the engine, which
 *   draws the complete file and records the result as a run.
 */

const KIND_LABELS: Record<LiveKind, { label: string; hint: string }> = {
  bar: { label: "Bar", hint: "compare categories" },
  line: { label: "Line", hint: "trend across an axis" },
  scatter: { label: "Scatter", hint: "two measures, one point per row" },
  bubble: { label: "Bubble", hint: "scatter sized by value" },
  histogram: { label: "Histogram", hint: "how one measure is distributed" },
  box: { label: "Box plot", hint: "spread and middle per category" },
  heatmap: {
    label: "Heatmap",
    hint: "two category axes, colour is the measure",
  },
};

/**
 * What the engine draws that this browser does not, and what it writes.
 *
 * Kept beside `LIVE_KINDS` rather than imported from `VisualizeDialog`, whose
 * list is the dialog's own concern; `tests/test_chart_kind_parity.py` checks
 * both against `core/viz/chartspec.py`, so a kind added to the engine shows up
 * as a failure here rather than as a quietly missing button.
 *
 * `format` is what the engine is asked to produce. A chart kind produces the
 * interactive HTML figure; the two exports reproduce whatever is on screen.
 */
const ENGINE_KINDS: { kind: string; label: string; hint: string }[] = [
  { kind: "pie", label: "Pie", hint: "shares of a whole" },
  { kind: "sunburst", label: "Sunburst", hint: "hierarchical shares" },
  { kind: "treemap", label: "Treemap", hint: "nested area rectangles" },
  { kind: "violin", label: "Violin", hint: "distribution shape per group" },
  { kind: "radar", label: "Radar", hint: "several measures on spokes" },
  { kind: "waffle", label: "Waffle", hint: "square-unit share of a whole" },
  { kind: "calendar", label: "Calendar", hint: "daily values on a year grid" },
];

const EXPORTS: { format: string; label: string; hint: string }[] = [
  {
    format: "xlsx",
    label: "Excel workbook",
    hint: "a native Excel chart over this table, editable in Excel",
  },
  {
    format: "html",
    label: "Interactive HTML",
    hint: "a standalone page that works with no NLP Suite installed",
  },
];

const AGGS: { id: Agg; label: string }[] = [
  { id: "sum", label: "Total" },
  { id: "mean", label: "Average" },
  { id: "median", label: "Median" },
  { id: "count", label: "How many rows" },
];

/** Kinds where x is a category to group by, rather than a second measure. */
const CATEGORICAL_X = new Set<LiveKind>(["bar", "line", "box", "heatmap"]);
/** Kinds that summarise several rows into one mark, so they need an aggregation. */
const AGGREGATES = new Set<LiveKind>(["bar", "line", "heatmap"]);

/**
 * A recommended chart as the controls that draw it.
 *
 * The recommendation carries more than three column names, and the extra two
 * decide whether the drawn chart is the recommended one. A timeline is
 * averaged and uncut: summed, two documents dated the same day become a spike
 * about how many documents that day had; cut to its twenty largest
 * categories, the quiet years are deleted and the line drawn straight through
 * where they were.
 *
 * Returns null for a kind this workbench does not draw itself, rather than
 * silently substituting one it does.
 */
export function chartSettings(chart: RecommendedChart): Settings | null {
  if (!isLiveKind(chart.kind)) return null;
  const named = ["sum", "mean", "median", "count"].includes(chart.agg);
  return {
    kind: chart.kind,
    x: chart.x,
    y: chart.y,
    group: "",
    agg: named ? (chart.agg as Agg) : "sum",
    topN: chart.top_n ?? 0,
    // A timeline reads in the order it happened. Ranking its categories by
    // value is the same chart with the years shuffled, which is not a
    // timeline at all.
    sort: chart.kind === "line" ? "table" : "high",
    bins: 12,
  };
}

/**
 * Where the chart opens.
 *
 * Given a recommendation whose columns are all present, that — because the
 * recommendation carries the question it answers, and column order does not.
 * Otherwise the old guess: the first label column against the first measure.
 *
 * That guess is what put "how many sentences are in each speech, longest
 * first" on screen as the result of a sentiment analysis. It was not wrong
 * about the data; it just answered nothing anybody had asked.
 */
export function openingSettings(
  table: Table,
  recommended: RecommendedChart[] = [],
): Settings {
  for (const chart of recommended) {
    const settings = chartSettings(chart);
    if (!settings) continue;
    const needed =
      chart.kind === "histogram" ? [settings.y] : [settings.x, settings.y];
    if (needed.every((column) => table.columns.includes(column))) {
      return settings;
    }
  }
  return initialSettings(table);
}

/** A first guess that is usually right: the first label column against the
 *  first measure, drawn as bars. */
export function initialSettings(table: Table): Settings {
  // The measure must not be a row number: `Document ID` is the first numeric
  // column of nearly every result this suite produces, and a chart of it is a
  // staircase. The *axis* may well be unique per row — "readability by
  // document" is the chart that table is for — which is why only the measure
  // is filtered, matching what core/insight/recommend.py allows.
  const measures = measureColumns(table);
  const numeric = numericColumns(table);
  const labels = table.columns.filter((column) => !numeric.includes(column));
  return {
    kind: "bar",
    x: labels[0] ?? table.columns[0] ?? "",
    y: measures[0] ?? "",
    group: "",
    agg: "sum",
    topN: 25,
    sort: "high",
    bins: 12,
  };
}

/**
 * The settings and drill-down for one table, for a caller that owns neither.
 *
 * The Runs pane wants a chart of whichever artifact is open and nothing more.
 * Explore has to put a *saved* arrangement into these same controls, which it
 * cannot do while the controls keep their own private copy. So the state lives
 * with the caller, and this hook is the version for the caller that has
 * nothing to restore.
 */
export function useChartSettings(
  table: Table,
  recommended: RecommendedChart[] = [],
) {
  const [settings, setSettings] = useState<Settings>(() =>
    openingSettings(table, recommended),
  );
  const [selected, setSelected] = useState<Selection>(null);

  // A new artifact is a new table: its columns are different, so the previous
  // choice of axis would name a column that is no longer there.
  //
  // Keyed on the columns and on which chart is recommended first, never on
  // the recommendation array: a live analysis re-answers the same question on
  // every keystroke and hands back a fresh array each time, so depending on
  // the array would reset the axes out from under anyone adjusting them.
  const opening = recommended[0];
  const openingKey = opening ? `${opening.kind}:${opening.x}:${opening.y}` : "";
  useEffect(() => {
    setSettings(openingSettings(table, recommended));
    setSelected(null);
  }, [openingKey, table.columns.join("\u0000")]); // eslint-disable-line react-hooks/exhaustive-deps

  return { settings, setSettings, selected, setSelected };
}

/**
 * The panel's own heading, at whatever depth the page puts it.
 *
 * Under Runs & results the workbench sits below the run's own `<h2>`, so `h3`
 * is right. On Explore it sits directly under the page `<h1>`, where `h3`
 * skips a level and anyone navigating by heading hears a gap. The level is the
 * caller's to say because only the caller knows what is above it.
 */
function Heading({ level, children }: { level: 2 | 3; children: ReactNode }) {
  const Tag = level === 2 ? "h2" : "h3";
  return (
    <Tag>
      <ChartColumnBig size={16} aria-hidden="true" /> {children}
    </Tag>
  );
}

export function Workbench({
  table,
  artifactPath,
  settings,
  onSettings,
  selected,
  onSelect,
  canPublish,
  onPublish,
  gaps = [],
  headingLevel = 3,
  onHandOff,
}: {
  table: Table;
  artifactPath: string;
  settings: Settings;
  onSettings: (settings: Settings) => void;
  selected: Selection;
  onSelect: (selection: Selection) => void;
  canPublish: boolean;
  /** Hand these settings to the engine, which draws the whole file. */
  onPublish: (settings: Settings) => void;
  /**
   * What publishing will not reproduce about this arrangement, in the
   * server's words. Shown beside the button that would do it: a caution
   * nobody meets before acting is not a caution.
   */
  gaps?: string[];
  /** Depth of this panel's heading; see {@link Heading}. */
  headingLevel?: 2 | 3;
  /**
   * Hand this arrangement to the engine, which draws kinds the browser cannot
   * and writes formats it cannot. Omitted where there is nothing to hand off
   * to -- a live table has no artifact on disk for the engine to read, so the
   * live bench leaves this out rather than offering a button that cannot work.
   */
  onHandOff?: (seed: {
    kind: string;
    x: string;
    y: string;
    format: string;
  }) => void;
}) {
  // Open by default. The chart is the point of this panel, and a result you
  // have to expand before you can look at it is a result most people never
  // look at.
  const [open, setOpen] = useState(true);
  const setSelected = onSelect;

  // Measures, not identifiers: the same distinction initialSettings makes.
  const numeric = useMemo(() => measureColumns(table), [table]);
  const categorical = useMemo(() => categoricalColumns(table), [table]);
  const result = useMemo(
    () => buildLayout(table.rows, settings),
    [table.rows, settings],
  );
  const filtered = useMemo(
    () => (selected ? rowsBehind(table.rows, settings, selected) : null),
    [selected, table.rows, settings],
  );

  const set = (patch: Partial<Settings>) =>
    onSettings({ ...settings, ...patch });
  const categoricalX = CATEGORICAL_X.has(settings.kind);
  const xOptions = categoricalX ? categorical : numeric;
  const partial =
    table.truncated ||
    (table.filtered_total ?? table.total) > table.rows.length;

  // An empty result is a result. `duplicates.csv` with no rows means the run
  // found no duplicates, and saying "this page of the table has no rows" reads
  // like a paging fault rather than the answer it is.
  if (!table.total) {
    return (
      <section className="panel workbench">
        <Heading level={headingLevel}>Explore this table</Heading>
        <p className="muted">
          <strong>{artifactPath}</strong> is empty — the run completed and found
          nothing to report in it. There is nothing to chart.
        </p>
      </section>
    );
  }

  if (!numeric.length) {
    return (
      <section className="panel workbench">
        <Heading level={headingLevel}>Explore this table</Heading>
        <p className="muted">
          Nothing in <strong>{artifactPath}</strong> is a number, so there is no
          measure to draw.
        </p>
      </section>
    );
  }

  return (
    <section className="panel workbench" aria-label="Explore this table">
      <div className="workbench-head">
        <Heading level={headingLevel}>Explore this table</Heading>
        <button
          className="text-button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
        >
          {open ? "Hide chart" : "Show chart"}
        </button>
      </div>

      {open && (
        <>
          <div className="workbench-kinds" role="group" aria-label="Chart type">
            {LIVE_KINDS.map((kind) => (
              <button
                key={kind}
                type="button"
                title={KIND_LABELS[kind].hint}
                className={
                  settings.kind === kind ? "viz-kind selected" : "viz-kind"
                }
                onClick={() => {
                  // Moving between a category axis and a second measure changes
                  // what x means, so re-pick a column that fits the new kind.
                  const nextCategorical = CATEGORICAL_X.has(kind);
                  const options = nextCategorical ? categorical : numeric;
                  const keep = options.includes(settings.x);
                  set({
                    kind,
                    x: keep
                      ? settings.x
                      : (options.find((c) => c !== settings.y) ??
                        options[0] ??
                        settings.x),
                    group:
                      kind === "heatmap" && !settings.group
                        ? (categorical[1] ?? "")
                        : settings.group,
                  });
                  setSelected(null);
                }}
              >
                {KIND_LABELS[kind].label}
              </button>
            ))}
          </div>

          {onHandOff && (
            <div className="workbench-handoff">
              <span className="handoff-label">
                Drawn by the engine, from these same settings:
              </span>
              <div role="group" aria-label="Other chart types">
                {ENGINE_KINDS.map((entry) => (
                  <button
                    key={entry.kind}
                    type="button"
                    className="viz-kind"
                    title={entry.hint}
                    onClick={() =>
                      onHandOff({
                        kind: entry.kind,
                        x: settings.x,
                        y: settings.y,
                        format: "html",
                      })
                    }
                  >
                    {entry.label}
                  </button>
                ))}
              </div>
              <div role="group" aria-label="Export this chart">
                {EXPORTS.map((entry) => (
                  <button
                    key={entry.format}
                    type="button"
                    className="viz-kind viz-export"
                    title={entry.hint}
                    onClick={() =>
                      onHandOff({
                        kind: settings.kind,
                        x: settings.x,
                        y: settings.y,
                        format: entry.format,
                      })
                    }
                  >
                    <Download size={13} /> {entry.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="workbench-controls">
            <label className="field-label">
              {settings.kind === "histogram" ? "Measure to bin" : "Measure"}
              <select
                value={settings.y}
                onChange={(e) => set({ y: e.target.value })}
              >
                {numeric.map((column) => (
                  <option key={column}>{column}</option>
                ))}
              </select>
            </label>

            {settings.kind !== "histogram" && (
              <label className="field-label">
                {categoricalX ? "Group rows by" : "Against"}
                <select
                  value={settings.x}
                  onChange={(e) => set({ x: e.target.value })}
                >
                  {xOptions.map((column) => (
                    <option key={column}>{column}</option>
                  ))}
                </select>
              </label>
            )}

            {AGGREGATES.has(settings.kind) && (
              <label className="field-label">
                Combine repeated rows by
                <select
                  value={settings.agg}
                  onChange={(e) => set({ agg: e.target.value as Agg })}
                >
                  {AGGS.map((agg) => (
                    <option key={agg.id} value={agg.id}>
                      {agg.label}
                    </option>
                  ))}
                </select>
              </label>
            )}

            {settings.kind !== "box" && (
              <label className="field-label">
                {settings.kind === "heatmap" ? "Row axis" : "Colour by"}
                <select
                  value={settings.group}
                  onChange={(e) => set({ group: e.target.value })}
                >
                  <option value="">
                    {settings.kind === "heatmap"
                      ? "Choose a column"
                      : "Nothing"}
                  </option>
                  {categorical
                    .filter((column) => column !== settings.x)
                    .map((column) => (
                      <option key={column}>{column}</option>
                    ))}
                </select>
              </label>
            )}

            {settings.kind === "histogram" && (
              <label className="field-label">
                Bins
                <input
                  type="range"
                  min={4}
                  max={40}
                  value={settings.bins}
                  onChange={(e) => set({ bins: Number(e.target.value) })}
                />
                <small>{settings.bins} bins</small>
              </label>
            )}

            {categoricalX && settings.kind !== "heatmap" && (
              <label className="field-label">
                Order
                <select
                  value={settings.sort}
                  onChange={(e) =>
                    set({ sort: e.target.value as Settings["sort"] })
                  }
                >
                  <option value="high">Largest first</option>
                  <option value="low">Smallest first</option>
                  <option value="table">Table order</option>
                </select>
              </label>
            )}

            {categoricalX && (
              <label className="field-label">
                Show at most
                <input
                  type="range"
                  min={5}
                  max={Math.min(
                    80,
                    Math.max(5, distinctCount(table.rows, settings.x)),
                  )}
                  value={settings.topN}
                  onChange={(e) => set({ topN: Number(e.target.value) })}
                />
                <small>{settings.topN} categories</small>
              </label>
            )}
          </div>

          {result.ok ? (
            <>
              {result.layout.caution && (
                <p className="alert warning" role="note">
                  <TriangleAlert size={16} aria-hidden="true" />
                  <span>{result.layout.caution}</span>
                </p>
              )}
              <ChartCanvas
                layout={result.layout}
                selected={selected}
                onSelect={setSelected}
                title={`${settings.y} by ${settings.x}`}
                name={
                  artifactPath
                    .split("/")
                    .pop()
                    ?.replace(/\.csv$/i, "") ?? ""
                }
              />
              <p className="workbench-note">
                {/* What the picture is built from, always, so a page of a long
                    table is never mistaken for the whole of it. */}
                Drawn from {formatNumber(result.layout.used)} row
                {result.layout.used === 1 ? "" : "s"} on this page
                {partial && (
                  <strong>
                    {" "}
                    of {formatNumber(table.filtered_total ?? table.total)} in
                    the file
                  </strong>
                )}
                .
                {result.layout.dropped > 0 &&
                  ` ${formatNumber(result.layout.dropped)} row${
                    result.layout.dropped === 1 ? " has" : "s have"
                  } no value in ${settings.y} and are left out.`}
                {result.layout.hidden > 0 &&
                  ` ${formatNumber(result.layout.hidden)} smaller categor${
                    result.layout.hidden === 1 ? "y is" : "ies are"
                  } not shown.`}
              </p>
            </>
          ) : (
            <p className="alert warning" role="status">
              <TriangleAlert size={16} aria-hidden="true" />
              <span>{result.reason}</span>
            </p>
          )}

          {gaps.length > 0 && result.ok && (
            <div className="workbench-gaps" role="note">
              <Info size={15} aria-hidden="true" />
              <div>
                <strong>Publishing will not match this exactly.</strong>
                <ul>
                  {gaps.map((gap) => (
                    <li key={gap}>{gap}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          <div className="workbench-actions">
            {selected && (
              <button className="secondary" onClick={() => setSelected(null)}>
                <RotateCcw size={15} /> Show all rows
              </button>
            )}
            <button
              className="primary"
              disabled={!canPublish || !result.ok}
              title="Draw this over the whole file and record it as a run"
              onClick={() => onPublish(settings)}
            >
              <Sparkles size={15} /> Publish this chart
            </button>
          </div>
          <p className="muted workbench-provenance">
            Saving a view keeps this arrangement and runs nothing. Publishing
            runs the same settings over the complete file and records the
            inputs, settings and diagnostics with the result. The SVG and PNG
            buttons save the picture exactly as drawn above — this page&apos;s
            rows, at the current zoom — for dropping into notes or slides.
          </p>
        </>
      )}

      {filtered && (
        <p className="workbench-filter" role="status">
          <Filter size={14} aria-hidden="true" />
          Showing the {formatNumber(filtered.length)} row
          {filtered.length === 1 ? "" : "s"} behind{" "}
          <strong>{selected?.label || "(blank)"}</strong>
          {selected?.group && selected.group !== "(all)"
            ? ` · ${selected.group}`
            : ""}
          .
        </p>
      )}

      <FilteredRows table={table} rows={filtered} />
    </section>
  );
}

/** The rows a clicked mark stands for, shown beneath the chart. */
function FilteredRows({
  table,
  rows,
}: {
  table: Table;
  rows: Record<string, string>[] | null;
}) {
  const [offset, setOffset] = useState(0);
  // A different mark is a different evidence set: carrying the old offset
  // into it would open another mark's rows midway through.
  useEffect(() => setOffset(0), [rows]);
  if (!rows) return null;
  const PAGE = 50;
  const shown = rows.slice(offset, offset + PAGE);
  return (
    <ResultTable
      columns={table.columns}
      rows={shown}
      copyable
      footerNote={
        rows.length > PAGE ? (
          <>
            Rows {offset + 1}–{offset + shown.length} of{" "}
            {formatNumber(rows.length)} behind this mark.
            <div className="drop-actions">
              <button
                className="text-button"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE))}
              >
                Previous rows
              </button>
              <button
                className="text-button"
                disabled={offset + PAGE >= rows.length}
                onClick={() => setOffset(offset + PAGE)}
              >
                More rows
              </button>
            </div>
          </>
        ) : undefined
      }
    />
  );
}

export { isNumericColumn };
