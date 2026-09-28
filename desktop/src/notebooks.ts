/**
 * The Scripts page's data: notebooks, their kernel's events, and what each
 * cell has shown.
 *
 * Kept apart from the page so the part that decides what a cell's output *is*
 * -- which event belongs to which cell, when a cell counts as finished, what
 * a stopped kernel means for the cells still waiting -- is tested without
 * drawing anything (scripts.test.ts).
 */
import { api, post, type Job, type Table } from "./api";
import type { Agg, LiveKind, Settings } from "./chartLayout";
import { isLiveKind, nameSaysYear } from "./chartLayout";

export type CellKind = "code" | "markdown";

/** One cell as stored: nbformat 4, without outputs (those belong to a run). */
export type NotebookCell = {
  cell_type: CellKind;
  id: string;
  source: string;
  metadata: { nlpsuite?: { origin?: "pasted" | "import" } };
  execution_count?: null;
  outputs?: unknown[];
};

export type NotebookContent = {
  nbformat: 4;
  nbformat_minor: number;
  metadata: {
    nlpsuite?: { library?: string; created_by?: "app" | "import" | "pasted" };
    [key: string]: unknown;
  };
  cells: NotebookCell[];
};

export type Notebook = {
  id: string;
  project_id: string;
  name: string;
  content: NotebookContent;
  revision: number;
  created: string;
  updated: string;
};

export type LibraryFunction = {
  name: string;
  kind: "function" | "class";
  signature: string;
  doc: string;
  /** A class's properties and methods, one per line. */
  members: string;
  /** What "Insert" puts in the cell: the call with its required arguments. */
  insert?: string;
};

/** A notebook the app offers to start from, and the guide's worked example. */
export type ScriptTemplate = {
  id: string;
  name: string;
  description: string;
  /** The request, in a researcher's words, this notebook answers. */
  request: string;
  /** The notebook's cells in order; `kind` is "code" or "markdown". */
  cells: { kind: string; text: string }[];
};

export type ScriptReference = {
  markdown: string;
  functions: LibraryFunction[];
  templates: ScriptTemplate[];
};

/** How a chart beside a shown table is drawn: `nlp.show`'s choice, or the recommender's. */
export type ShownChart = {
  kind: string;
  x: string;
  y: string;
  group: string;
  agg: string;
  top_n: number;
  question?: string;
  /** A step between the table (as its CSV holds it) and what is drawn. */
  prepare?: {
    /** Several y columns drawn as one line each: melted into Series/Value. */
    melt?: { id: string; values: string[] };
    /** Rows counted per value of a column (a table of passages has no measure). */
    count?: { by: string; name: string };
  };
};

export type CellOutput =
  | { kind: "stream"; name: "stdout" | "stderr"; text: string }
  | { kind: "text"; text: string }
  | {
      kind: "table";
      name: string;
      title: string;
      file: string;
      saved: boolean;
      preview: Table;
      chart: ShownChart | null;
      chart_note: string;
      chart_rows?: Table;
    }
  | { kind: "figure"; name: string; png: string; svg: string | null }
  | { kind: "html"; name: string; file: string };

export type CellError = { type: string; message: string; trace: string[] };

export type CellRun = {
  /** The kernel's number for this run of the cell; null while it is only queued. */
  exec: number | null;
  state: "queued" | "running" | "done" | "error";
  outputs: CellOutput[];
  seconds: number | null;
  error: CellError | null;
  started: number;
};

export type KernelEvent =
  | { event: "ready"; library?: string }
  | { event: "ended" }
  | { event: "stream"; id: number; name: "stdout" | "stderr"; text: string }
  | ({ event: "output"; id: number } & Record<string, unknown>)
  | {
      event: "done";
      id: number;
      ok: boolean;
      seconds: number | null;
      error: CellError | null;
    };

export type KernelState = {
  events: KernelEvent[];
  next: number;
  running: boolean;
  busy: number | null;
  ready: boolean;
};

// ------------------------------------------------------------------ api --

const base = (projectId: string) => `/projects/${projectId}/notebooks`;

export const listNotebooks = (projectId: string) =>
  api<Notebook[]>(base(projectId));
export const getNotebook = (projectId: string, id: string) =>
  api<Notebook>(`${base(projectId)}/${id}`);
export const createNotebook = (
  projectId: string,
  body: { name: string; template?: string; imported?: unknown },
) => post<Notebook>(base(projectId), body);
export const saveNotebook = (
  projectId: string,
  notebook: Pick<Notebook, "id" | "name" | "content" | "revision">,
) =>
  post<Notebook>(`${base(projectId)}/${notebook.id}`, {
    name: notebook.name,
    content: notebook.content,
    expected_revision: notebook.revision,
  });
export const duplicateNotebook = (projectId: string, id: string) =>
  post<Notebook>(`${base(projectId)}/${id}/duplicate`, {});
export const deleteNotebook = (projectId: string, id: string) =>
  post<{ deleted: boolean }>(`${base(projectId)}/${id}/delete`, {});
export const runAndSave = (projectId: string, id: string, parser: string) =>
  post<Job>(`${base(projectId)}/${id}/run`, { parser });
export const executeCell = (
  projectId: string,
  id: string,
  cell: string,
  code: string,
  parser: string,
) =>
  post<{ exec: number; after: number }>(
    `${base(projectId)}/${id}/kernel/exec`,
    {
      cell,
      code,
      parser,
    },
  );
export const kernelEvents = (projectId: string, id: string, after: number) =>
  api<KernelState>(`${base(projectId)}/${id}/kernel/events?after=${after}`);
export const stopKernel = (projectId: string, id: string) =>
  post<{ stopped: boolean }>(`${base(projectId)}/${id}/kernel/stop`, {});
export const kernelFilePath = (projectId: string, id: string, name: string) =>
  `${base(projectId)}/${id}/kernel/files/${encodeURIComponent(name)}`;
/** Every table this notebook's kernel has produced, as one zip. */
export const notebookTablesPath = (projectId: string, id: string) =>
  `${base(projectId)}/${id}/tables.zip`;
export const scriptReference = () => api<ScriptReference>("/script/reference");
export const guidePath = (
  projectId: string,
  version: "short" | "full",
  corpus: boolean,
) =>
  `/projects/${projectId}/script-guide?version=${version}&corpus=${corpus ? "true" : "false"}`;
export const exportPath = (projectId: string, id: string) =>
  `${base(projectId)}/${id}/export`;

/** What the corpus has, for the reference panel and completion: names and counts, never text. */
export type CorpusSummary = {
  name: string;
  documents: number;
  words: number;
  years: [number, number] | null;
  columns: string[];
  /** Columns that are the project's own document details (app-only). */
  details: string[];
  samples: Record<string, string[]>;
};
export const scriptCorpus = (projectId: string) =>
  api<CorpusSummary | null>(`/projects/${projectId}/script-corpus`);

export type LintWarning = { line: number; message: string };
export const lintCode = (code: string) =>
  post<{ warnings: LintWarning[] }>("/script/lint", { code });

// ------------------------------------------------------------- notebooks --

let counter = 0;
/** A cell id nbformat accepts: short, unique within the notebook. */
export function newCellId(): string {
  counter += 1;
  return `${Date.now().toString(36).slice(-5)}${counter.toString(36)}`;
}

export function newCell(
  kind: CellKind,
  source = "",
  origin?: "pasted",
): NotebookCell {
  return {
    cell_type: kind,
    id: newCellId(),
    source,
    metadata: origin ? { nlpsuite: { origin } } : {},
    ...(kind === "code" ? { execution_count: null, outputs: [] } : {}),
  };
}

/** True when the notebook as a whole came from outside the app. */
export const cameFromOutside = (content: NotebookContent): boolean =>
  ["import", "pasted"].includes(content.metadata.nlpsuite?.created_by ?? "app");

/** Read a `.ipynb` file's text into JSON the server will clean, or say why not. */
export function parseIpynb(text: string): unknown {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("That file is not a Jupyter notebook: it is not JSON.");
  }
  if (
    !parsed ||
    typeof parsed !== "object" ||
    (parsed as { nbformat?: unknown }).nbformat !== 4
  )
    throw new Error(
      "That file is not a Jupyter notebook in the current format (nbformat 4).",
    );
  return parsed;
}

// --------------------------------------------------------------- outputs --

/** Adjacent printed text of one stream merges, as a terminal would show it. */
function append(outputs: CellOutput[], output: CellOutput): CellOutput[] {
  const last = outputs[outputs.length - 1];
  if (
    output.kind === "stream" &&
    last?.kind === "stream" &&
    last.name === output.name
  )
    return [
      ...outputs.slice(0, -1),
      { ...last, text: last.text + output.text },
    ];
  return [...outputs, output];
}

/**
 * Fold kernel events into the cells they belong to.
 *
 * `runs` is keyed by cell id; `execToCell` says which cell each kernel exec
 * number was for. Events for an exec the page no longer tracks (a cell that
 * was deleted while it ran) are dropped rather than attached to whichever
 * cell happens to be nearby.
 */
export function applyEvents(
  runs: Record<string, CellRun>,
  execToCell: Record<number, string>,
  events: KernelEvent[],
): Record<string, CellRun> {
  let next = runs;
  for (const event of events) {
    if (!("id" in event)) continue;
    const cellId = execToCell[event.id];
    const run = cellId ? next[cellId] : undefined;
    if (!run || run.exec !== event.id) continue;
    let updated: CellRun = run;
    if (event.event === "stream") {
      updated = {
        ...run,
        outputs: append(run.outputs, {
          kind: "stream",
          name: event.name,
          text: event.text,
        }),
      };
    } else if (event.event === "output") {
      const { event: _event, id: _id, ...output } = event;
      updated = { ...run, outputs: append(run.outputs, output as CellOutput) };
    } else if (event.event === "done") {
      updated = {
        ...run,
        state: event.ok ? "done" : "error",
        seconds: event.seconds,
        error: event.error,
      };
    }
    next = { ...next, [cellId]: updated };
  }
  return next;
}

/**
 * What Stop, or a kernel that went away, means for the cells not finished.
 *
 * The one running ends with the reason; the queued ones are not run at all
 * (running them in a fresh kernel would run them without the variables the
 * cells above them made).
 */
export function endUnfinished(
  runs: Record<string, CellRun>,
  message: string,
): Record<string, CellRun> {
  const next = { ...runs };
  for (const [id, run] of Object.entries(runs)) {
    if (run.state === "running")
      next[id] = {
        ...run,
        state: "error",
        error: { type: "Stopped", message, trace: [] },
      };
    else if (run.state === "queued") delete next[id];
  }
  return next;
}

// ----------------------------------------------------------------- charts --

/**
 * A shown table's chart as the chart canvas's settings.
 *
 * Unlike the workbench's `chartSettings`, this keeps `group`: `nlp.show(t,
 * x="Year", y=["a", "b"])` arrives as one Value column split by Series, and
 * dropping the split would draw the two lines as one.
 */
export function shownChartSettings(chart: ShownChart): Settings | null {
  if (!isLiveKind(chart.kind)) return null;
  const kind: LiveKind = chart.kind;
  const named = ["sum", "mean", "median", "count"].includes(chart.agg);
  return {
    kind,
    x: chart.x,
    y: chart.y,
    group: chart.group ?? "",
    agg: named ? (chart.agg as Agg) : "sum",
    topN: chart.top_n ?? 0,
    // Years and dates read in their own order, as bars too: passages per
    // year sorted by size put 1981 before 2018 before 1956.
    sort:
      kind === "line" ||
      kind === "scatter" ||
      nameSaysYear(chart.x) ||
      /\bdate\b/i.test(chart.x)
        ? "table"
        : "high",
    bins: 12,
  };
}

const py = (text: string) => JSON.stringify(text);

/**
 * About a dozen lines of matplotlib that draw the same chart from the saved CSV.
 *
 * The requirement it answers: every chart can leave the app as a CSV and a
 * script the researcher can edit. It is also a lesson -- the obvious next
 * step after "the app drew this" is "here is how you would draw it".
 */
export function matplotlibCode(
  chart: ShownChart,
  csv: string,
  title = "",
): string {
  const lines = [
    "import matplotlib.pyplot as plt",
    "import pandas as pd",
    "",
    `table = pd.read_csv(${py(csv)})`,
  ];
  const melt = chart.prepare?.melt;
  const count = chart.prepare?.count;
  if (melt)
    lines.push(
      `table = table.melt(id_vars=[${py(melt.id)}], value_vars=[${melt.values.map(py).join(", ")}], var_name="Series", value_name="Value")`,
    );
  if (count)
    lines.push(
      `table = table.groupby(${py(count.by)}).size().reset_index(name=${py(count.name)})`,
    );
  const x = py(chart.x);
  const y = py(chart.y);
  const agg = ["sum", "mean", "median", "count"].includes(chart.agg)
    ? chart.agg
    : "";
  if (chart.kind === "histogram") {
    lines.push(
      "fig, ax = plt.subplots(figsize=(8, 4.5))",
      `ax.hist(table[${y}].dropna(), bins=12, color="#0072B2")`,
      `ax.set_xlabel(${y})`,
      `ax.set_ylabel("Rows")`,
    );
  } else if (chart.kind === "scatter" || chart.kind === "bubble") {
    lines.push("fig, ax = plt.subplots(figsize=(8, 4.5))");
    if (chart.group)
      lines.push(
        `for name, part in table.groupby(${py(chart.group)}):`,
        `    ax.scatter(part[${x}], part[${y}], label=str(name), alpha=0.8)`,
        "ax.legend(frameon=False)",
      );
    else
      lines.push(
        `ax.scatter(table[${x}], table[${y}], color="#0072B2", alpha=0.8)`,
      );
    lines.push(`ax.set_xlabel(${x})`, `ax.set_ylabel(${y})`);
  } else if (chart.kind === "heatmap" && chart.group) {
    lines.push(
      `data = table.pivot_table(index=${x}, columns=${py(chart.group)}, values=${y}, aggfunc=${py(agg || "sum")})`,
      "fig, ax = plt.subplots(figsize=(9, 6))",
      'image = ax.imshow(data.values, aspect="auto", cmap="viridis")',
      "ax.set_xticks(range(len(data.columns)), data.columns, rotation=90)",
      "ax.set_yticks(range(len(data.index)), data.index)",
      `fig.colorbar(image, ax=ax, label=${y})`,
    );
  } else if (chart.kind === "box") {
    lines.push(
      `groups = [(str(name), part[${y}].dropna()) for name, part in table.groupby(${x})]`,
      "fig, ax = plt.subplots(figsize=(8, 4.5))",
      "ax.boxplot([values for _, values in groups])",
      "ax.set_xticks(range(1, len(groups) + 1), [name for name, _ in groups])",
      `ax.set_xlabel(${x})`,
      `ax.set_ylabel(${y})`,
    );
  } else {
    const keys = chart.group ? `[${x}, ${py(chart.group)}]` : x;
    const how = agg || "sum";
    lines.push(
      agg
        ? `data = table.groupby(${keys})[${y}].${how}()${chart.group ? ".unstack()" : ""}`
        : chart.group
          ? `data = table.pivot_table(index=${x}, columns=${py(chart.group)}, values=${y})`
          : `data = table.set_index(${x})[${y}]`,
    );
    if (chart.kind === "bar" && !chart.group)
      lines.push(
        `data = data.sort_values(ascending=False)${chart.top_n ? `.head(${chart.top_n})` : ""}`,
      );
    lines.push(
      "fig, ax = plt.subplots(figsize=(9, 4.5))",
      chart.kind === "line"
        ? `data.plot(ax=ax, marker="o", markersize=3${chart.group ? "" : ', color="#0072B2"'})`
        : `data.plot.bar(ax=ax${chart.group ? "" : ', color="#0072B2"'})`,
      `ax.set_xlabel(${x})`,
      `ax.set_ylabel(${py(agg ? `${chart.y} (${agg})` : chart.y)})`,
    );
  }
  if (title) lines.push(`ax.set_title(${py(title)})`);
  lines.push(
    'ax.spines[["top", "right"]].set_visible(False)',
    "fig.tight_layout()",
    `fig.savefig(${py(csv.replace(/\.csv$/i, "") + ".png")}, dpi=200)`,
  );
  return lines.join("\n") + "\n";
}
