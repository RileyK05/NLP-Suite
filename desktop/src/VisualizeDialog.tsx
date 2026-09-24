import { useMemo, useState } from "react";
import { ChartColumnBig, Cloud, Network, TriangleAlert } from "lucide-react";
import { type Component, type Table, type Tool } from "./api";

/**
 * Every chart the engine can draw.
 *
 * `legacy` marks the kinds NLP Suite 1.6.38 also produced — the six its Excel
 * chart GUI offered. Work built on the original suite often has to hand in its
 * output specifically, so "did the old suite draw this one?" is a question with
 * a practical answer, and it belongs here, at the moment of choosing, rather
 * than only in a reference list elsewhere.
 *
 * Kept in step with `core/viz/chartspec.py` and `core/viz/charts_excel.py` by
 * `tests/test_chart_kind_parity.py`, which fails if either drifts. Copying this
 * list by hand is how `bubble` came to be missing from the desktop entirely
 * while the Excel exporter still advertised it.
 */
const CHART_KINDS: {
  kind: string;
  label: string;
  hint: string;
  legacy?: boolean;
}[] = [
  { kind: "bar", label: "Bar", hint: "compare categories", legacy: true },
  { kind: "line", label: "Line", hint: "trends across an axis", legacy: true },
  {
    kind: "scatter",
    label: "Scatter",
    hint: "two measures, one point each",
    legacy: true,
  },
  {
    kind: "bubble",
    label: "Bubble",
    hint: "scatter sized by value",
    legacy: true,
  },
  {
    kind: "histogram",
    label: "Histogram",
    hint: "how a measure is distributed",
  },
  { kind: "box", label: "Box plot", hint: "spread and outliers per group" },
  {
    kind: "heatmap",
    label: "Heatmap",
    hint: "two category axes, colored density",
  },
  { kind: "pie", label: "Pie", hint: "shares of a whole", legacy: true },
  { kind: "sunburst", label: "Sunburst", hint: "hierarchical shares" },
  { kind: "treemap", label: "Treemap", hint: "nested area rectangles" },
  { kind: "violin", label: "Violin", hint: "distribution shape per group" },
  {
    kind: "radar",
    label: "Radar",
    hint: "several measures on spokes",
    legacy: true,
  },
  { kind: "waffle", label: "Waffle", hint: "square-unit share of a whole" },
  { kind: "calendar", label: "Calendar", hint: "daily values on a year grid" },
];

/** The chart's name as this dialog prints it, so a refusal names the button
 * the reader just pressed ("Box plot", not "box"). */
const kindLabel = (kind: string): string =>
  CHART_KINDS.find((entry) => entry.kind === kind)?.label ?? kind;

/** Formats the legacy Excel exporter can draw natively. */
export const EXCEL_CHART_KINDS = CHART_KINDS.filter((k) => k.legacy).map(
  (k) => k.kind,
);

const WORDCLOUD_KINDS = [
  { id: "wordcloud", label: "Wordcloud", hint: "size = frequency" },
  { id: "gexf", label: "GEXF network", hint: "open in Gephi" },
] as const;

type Family = "charts" | "wordcloud" | "gexf";

type Choice = {
  family: Family;
  kind: string;
  x: string;
  y: string;
  format: string;
  title: string;
  group: string;
  agg: string;
  wordCol: string;
  weightCol: string;
  sourceCol: string;
  targetCol: string;
};

// Mirrors core/viz/chartspec.py validation so the dialog disables every
// combination the engine would refuse, before a job is ever submitted.
const AGG_KINDS = new Set([
  "bar",
  "line",
  "pie",
  "sunburst",
  "treemap",
  "radar",
  "waffle",
  "calendar",
  "heatmap",
]);
const GROUP_REQUIRED = new Set(["heatmap", "sunburst", "treemap"]);
const GROUP_REJECTED = new Set(["pie", "waffle", "calendar"]);
const AGG_REJECTED = new Set([
  "scatter",
  "box",
  "violin",
  "histogram",
  "bubble",
]);
const DATE_REQUIRED = new Set(["calendar"]);

export function chartBlockedReasons(
  kind: string,
  x: string,
  y: string,
  group: string,
  agg: string,
  table: Table | null,
): string[] {
  const reasons: string[] = [];
  const numericX = x && isNumericColumn(table, x);
  const yNumeric = y && isNumericColumn(table, y);
  if (DATE_REQUIRED.has(kind) && x && !isDateColumn(table, x))
    reasons.push(
      `Calendar needs a date column for X ("${x}" does not look like dates).`,
    );
  if (!DATE_REQUIRED.has(kind) && x === y && x)
    reasons.push("X and Y must be different columns.");
  if (!yNumeric)
    reasons.push(`Y must be a numeric column ("${y || "none"}" is not).`);
  if (GROUP_REQUIRED.has(kind) && !group)
    reasons.push(
      `${kindLabel(kind)} needs a Group column (rows = group, columns = X).`,
    );
  if (GROUP_REJECTED.has(kind) && group)
    reasons.push(
      `${kindLabel(kind)} takes no Group column; clear Group to use this kind.`,
    );
  // The engine refuses --agg outright for the kinds that draw every
  // observation as it stands. This dialog defaulted the aggregation to "sum"
  // and sent it whatever the kind, so scatter, bubble, box, violin and
  // histogram all failed after the job had been submitted. AGG_REJECTED was
  // declared for exactly this and never consulted; the inline list that stood
  // here in its place had already fallen behind it by one kind.
  if (AGG_REJECTED.has(kind)) {
    if (agg)
      reasons.push(
        `${kindLabel(kind)} draws every row as it stands and takes no aggregation.`,
      );
    if (kind === "histogram" && group && !columnIsCategorical(table, group))
      reasons.push(
        "Histogram Group should be a category column (few distinct values).",
      );
  } else if (
    !agg &&
    AGG_KINDS.has(kind) &&
    hasDuplicateKeys(table, x, group, kind)
  ) {
    reasons.push(
      `${kindLabel(kind)} has duplicate X values; pick an aggregation (sum/mean/…).`,
    );
  }
  if (
    (kind === "pie" ||
      kind === "sunburst" ||
      kind === "treemap" ||
      kind === "waffle") &&
    hasNegativeColumn(table, y)
  )
    reasons.push(
      `${kindLabel(kind)} needs non-negative values (slices encode magnitude).`,
    );
  void numericX;
  return reasons;
}

/** The engine's own label for a setting, so this dialog and the run form agree. */
const paramLabel = (tool: Tool | undefined, name: string, fallback: string) =>
  tool?.params.find((param) => param.name === name)?.label ?? fallback;

/**
 * Which export formats this installation can actually produce.
 *
 * PNG, SVG and PDF are rendered by Kaleido, which is an optional extra. The
 * dialog offered all five regardless, so on a machine without it the only
 * sign was a job that failed after running. A format that cannot be produced
 * is now shown, named, and not selectable.
 */
const STATIC_FORMATS = new Set(["png", "svg", "pdf"]);
export function exportFormats(
  choices: (string | number)[],
  components: Component[],
): { value: string; label: string; available: boolean }[] {
  // An empty component list means the environment has not been read yet, not
  // that nothing is installed; assume available rather than block the reader.
  const kaleido = components.find((c) => c.name === "kaleido");
  const canRender = kaleido ? kaleido.present : true;
  return choices.map((choice) => {
    const value = String(choice);
    const available = canRender || !STATIC_FORMATS.has(value);
    return {
      value,
      label: available ? value : `${value} — needs Kaleido`,
      available,
    };
  });
}

/** A chart the recommender proposed, used to open this dialog already filled in.
 *
 * `format` is carried for the workbench's handoff, where the reader has
 * already said what they want out -- an Excel workbook, a standalone page --
 * by pressing a button that says so. A seed without one opens on HTML, which
 * is the format that needs nothing installed.
 */
export type ChartSeed = {
  kind: string;
  x: string;
  y: string;
  format?: string;
};

/**
 * Where a recommended chart puts the dialog when it opens.
 *
 * A seeded axis is only honoured if the artifact currently on screen actually
 * has that column, and a seeded kind only if this dialog can draw it. A
 * recommendation carried over from another artifact therefore degrades to the
 * ordinary defaults rather than leaving the dialog pointing at a column that
 * is not there.
 */
export function resolveSeed(
  seed: ChartSeed | null | undefined,
  columns: string[],
  fallbackX: string,
  fallbackY: string,
): ChartSeed & { format: string } {
  const column = (name: string | undefined, fallback: string) =>
    name && columns.includes(name) ? name : fallback;
  return {
    kind:
      seed && CHART_KINDS.some((k) => k.kind === seed.kind) ? seed.kind : "bar",
    x: column(seed?.x, fallbackX),
    y: column(seed?.y, fallbackY),
    format: seed?.format ?? "html",
  };
}

export function VisualizeDialog({
  tools,
  table,
  artifactPath,
  canRun,
  components,
  seed,
  onSubmit,
}: {
  tools: Tool[];
  table: Table | null;
  artifactPath: string;
  canRun: boolean;
  /**
   * What this installation actually has. A static export needs Kaleido; the
   * dialog used to offer PNG, SVG and PDF regardless and let the job fail
   * after it had run.
   */
  components: Component[];
  /**
   * Pre-selection from a recommended chart. Read once, when the dialog opens:
   * it is a starting point, not a lock, and every control stays editable so a
   * recommendation can be adjusted rather than only accepted.
   */
  seed?: ChartSeed | null;
  onSubmit: (tool: string, params: Record<string, unknown>) => void;
}) {
  const charts = useMemo(
    () => tools.find((tool) => tool.name === "table_charts"),
    [tools],
  );
  const clouds = useMemo(
    () => tools.find((tool) => tool.name === "table_wordcloud_gephi"),
    [tools],
  );
  const formatChoices = useMemo(
    () =>
      exportFormats(
        charts?.params.find((param) => param.name === "format")?.choices ?? [
          "html",
        ],
        components,
      ),
    [charts, components],
  );
  const columns = table?.columns ?? [];
  const textCol =
    columns.find((c) => !isNumericColumn(table, c)) ?? columns[0] ?? "";
  const numCol =
    columns.find((c) => isNumericColumn(table, c)) ??
    columns[1] ??
    columns[0] ??
    "";
  const start = resolveSeed(seed, columns, textCol, numCol);
  const [family, setFamily] = useState<Choice>({
    family: charts ? "charts" : "wordcloud",
    kind: start.kind,
    x: start.x,
    y: start.y,
    format: start.format,
    title: artifactPath.replace(/\.csv$/i, ""),
    group: "",
    agg: "sum",
    wordCol: textCol,
    weightCol: numCol,
    sourceCol: textCol,
    targetCol: numCol,
  });
  if (!charts || !clouds) return null;
  const chartChoice = family.family === "charts";
  const cloudChoice = family.family !== "charts";
  // The aggregation that will actually be submitted. A kind that refuses one
  // never receives it, so the reader is not shown a blocking reason about a
  // control that is not on screen for them to change.
  const effectiveAgg = AGG_REJECTED.has(family.kind) ? "" : family.agg;
  const blocked =
    chartChoice &&
    chartBlockedReasons(
      family.kind,
      family.x,
      family.y,
      family.group,
      effectiveAgg,
      table,
    );
  return (
    <>
      <p className="muted">
        Visualizing <strong>{artifactPath}</strong> — pick a picture for this
        table. Every chart keeps its data next to it in the run folder.
      </p>
      <div className="viz-families">
        <button
          type="button"
          className={
            family.family === "charts" ? "viz-family selected" : "viz-family"
          }
          onClick={() =>
            setFamily({
              ...family,
              family: "charts",
            })
          }
        >
          <ChartColumnBig size={18} />
          Charts
          {/* Counted, not restated: this said "13 kinds" while the list beside
              it offered fourteen. */}
          <small>{CHART_KINDS.length} kinds</small>
        </button>
        <button
          type="button"
          className={
            family.family !== "charts" ? "viz-family selected" : "viz-family"
          }
          onClick={() =>
            setFamily({
              ...family,
              family: "wordcloud",
            })
          }
        >
          <Cloud size={18} />
          Words &amp; networks
          <small>wordcloud · GEXF</small>
        </button>
      </div>
      {chartChoice ? (
        <>
          <div className="viz-kind-grid">
            {CHART_KINDS.map((k) => {
              const probe = chartBlockedReasons(
                k.kind,
                family.x,
                family.y,
                family.group,
                family.agg,
                table,
              );
              const disabled = probe.length > 0;
              return (
                <button
                  key={k.kind}
                  type="button"
                  title={disabled ? probe.join(" ") : k.hint}
                  disabled={disabled}
                  className={
                    family.kind === k.kind ? "viz-kind selected" : "viz-kind"
                  }
                  onClick={() => setFamily({ ...family, kind: k.kind })}
                >
                  {k.label}
                  {k.legacy && (
                    <span
                      className="viz-kind-legacy"
                      title="NLP Suite 1.6.38 drew this kind too, and exports it to Excel"
                    >
                      legacy
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          <p className="viz-kind-note">
            Kinds marked <span className="viz-kind-legacy">legacy</span> are the
            ones NLP&nbsp;Suite&nbsp;1.6.38 also drew, and the only ones its
            Excel export covers. Pick the <code>xlsx</code> format below for a
            real Excel workbook with a native chart in it.
          </p>
          <label className="field-label">
            {paramLabel(charts, "x", "Horizontal axis")}
            <select
              value={family.x}
              onChange={(e) => setFamily({ ...family, x: e.target.value })}
            >
              {columns.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
          <label className="field-label">
            {paramLabel(charts, "y", "Vertical axis")}
            <select
              value={family.y}
              onChange={(e) => setFamily({ ...family, y: e.target.value })}
            >
              {columns.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
          <label className="field-label">
            {paramLabel(charts, "group", "Group by column")}
            <select
              value={family.group}
              onChange={(e) => setFamily({ ...family, group: e.target.value })}
            >
              <option value="">No grouping</option>
              {columns.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
          {/* Shown only where it applies. Leaving it on screen for a kind
              that refuses it is what let the default "sum" reach the engine. */}
          <label className="field-label" hidden={AGG_REJECTED.has(family.kind)}>
            {paramLabel(charts, "agg", "Combine repeated rows by")}
            {/* Read from the tool, not restated: the engine's AGG_FUNCS is the
                list that actually governs what a job will accept. */}
            <select
              value={family.agg}
              onChange={(e) => setFamily({ ...family, agg: e.target.value })}
            >
              {(
                charts?.params.find((param) => param.name === "agg")
                  ?.choices ?? [""]
              ).map((value) => (
                <option key={String(value)} value={String(value)}>
                  {value === "" ? "Do not combine" : String(value)}
                </option>
              ))}
            </select>
          </label>
          <label className="field-label">
            {paramLabel(charts, "format", "Export format")}
            <select
              value={family.format}
              onChange={(e) => setFamily({ ...family, format: e.target.value })}
            >
              {formatChoices.map((f) => (
                <option key={f.value} value={f.value} disabled={!f.available}>
                  {f.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field-label">
            Chart title
            <input
              value={family.title}
              onChange={(e) => setFamily({ ...family, title: e.target.value })}
            />
          </label>
        </>
      ) : (
        <>
          <div className="viz-kind-grid">
            {WORDCLOUD_KINDS.map((k) => (
              <button
                key={k.id}
                type="button"
                title={k.hint}
                className={
                  (
                    k.id === "wordcloud"
                      ? cloudChoice && family.family !== "gexf"
                      : family.family === "gexf"
                  )
                    ? "viz-kind selected"
                    : "viz-kind"
                }
                onClick={() =>
                  setFamily({
                    ...family,
                    family: k.id === "wordcloud" ? "wordcloud" : "gexf",
                  })
                }
              >
                {k.label}
              </button>
            ))}
          </div>
          {family.family !== "gexf" ? (
            <>
              <label className="field-label">
                Word column
                <select
                  value={family.wordCol}
                  onChange={(e) =>
                    setFamily({ ...family, wordCol: e.target.value })
                  }
                >
                  {columns.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
              <label className="field-label">
                Weight column
                <select
                  value={family.weightCol}
                  onChange={(e) =>
                    setFamily({ ...family, weightCol: e.target.value })
                  }
                >
                  {columns.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
              <label className="field-label">
                Cloud title
                <input
                  value={family.title}
                  onChange={(e) =>
                    setFamily({ ...family, title: e.target.value })
                  }
                />
              </label>
            </>
          ) : (
            <>
              <label className="field-label">
                Source column (edges)
                <select
                  value={family.sourceCol}
                  onChange={(e) =>
                    setFamily({ ...family, sourceCol: e.target.value })
                  }
                >
                  {columns.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
              <label className="field-label">
                Target column (edges)
                <select
                  value={family.targetCol}
                  onChange={(e) =>
                    setFamily({ ...family, targetCol: e.target.value })
                  }
                >
                  {columns.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
            </>
          )}
        </>
      )}
      {blocked && blocked.length > 0 && (
        <div className="alert error" role="alert">
          <TriangleAlert size={17} />
          <div>
            {blocked.map((reason, i) => (
              <p key={i}>{reason}</p>
            ))}
          </div>
        </div>
      )}
      <button
        className="primary"
        disabled={!canRun || (!!blocked && blocked.length > 0)}
        onClick={() => {
          if (family.family === "charts") {
            onSubmit("table_charts", {
              kind: family.kind,
              x: family.x,
              y: family.y,
              group: family.group,
              // Never send an aggregation to a kind that refuses one.
              agg: effectiveAgg,
              normalize: "none",
              "top-n": null,
              bins: null,
              title: family.title,
              format: family.format,
            });
          } else if (family.family === "wordcloud") {
            onSubmit("table_wordcloud_gephi", {
              mode: "wordcloud",
              "word-col": family.wordCol,
              "weight-col": family.weightCol,
              "source-col": "",
              "target-col": "",
              title: family.title,
              "max-words": 100,
              image: true,
              shape: "",
            });
          } else {
            onSubmit("table_wordcloud_gephi", {
              mode: "gexf",
              "word-col": family.wordCol,
              "weight-col": family.weightCol,
              "source-col": family.sourceCol,
              "target-col": family.targetCol,
              title: "",
              "max-words": 100,
              image: false,
              shape: "",
            });
          }
        }}
      >
        <Network size={16} />
        Build visualization
      </button>
    </>
  );
}

function isNumericColumn(table: Table | null, column: string): boolean {
  if (!table || !column) return false;
  const values = table.rows
    .slice(0, 20)
    .map((row) => row[column]?.trim() ?? "");
  return (
    values.some((v) => v !== "" && Number.isFinite(Number(v))) &&
    values.every((v) => v === "" || Number.isFinite(Number(v)))
  );
}

function isDateColumn(table: Table | null, column: string): boolean {
  if (!table || !column) return false;
  const values = table.rows
    .slice(0, 20)
    .map((row) => row[column]?.trim() ?? "")
    .filter((value) => value !== "");
  if (!values.length) return false;
  const parsed = values.filter((value) => !Number.isNaN(Date.parse(value)));
  return parsed.length === values.length;
}

function columnIsCategorical(table: Table | null, column: string): boolean {
  if (!table || !column) return false;
  const values = table.rows
    .slice(0, 20)
    .map((row) => row[column]?.trim() ?? "");
  const distinct = new Set(values.filter((v) => v !== ""));
  return distinct.size > 0 && distinct.size <= 12;
}

function hasDuplicateKeys(
  table: Table | null,
  x: string,
  group: string,
  kind: string,
): boolean {
  if (!table || !x || !AGG_KINDS.has(kind)) return false;
  const keys = kind === "heatmap" ? [x, group] : [x];
  if (keys.some((key) => !key)) return false;
  const seen = new Set<string>();
  for (const row of table.rows.slice(0, 20)) {
    const key = keys.map((k) => row[k] ?? "").join("\u0000");
    if (seen.has(key)) return true;
    seen.add(key);
  }
  return false;
}

function hasNegativeColumn(table: Table | null, column: string): boolean {
  if (!table || !column) return false;
  return table.rows
    .slice(0, 20)
    .some(
      (row) =>
        row[column]?.trim() !== "" &&
        row[column] !== undefined &&
        Number.isFinite(Number(row[column])) &&
        Number(row[column]) < 0,
    );
}
