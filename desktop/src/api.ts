import { convertFileSrc, invoke, isTauri } from "@tauri-apps/api/core";
import type { SourcePassage } from "./phrase";

export type Connection = { baseUrl: string; token: string };
export type Project = {
  id: string;
  name: string;
  created: string;
  documents: number;
  words: number;
  archived?: number;
  trashed?: number;
};
export type Document = {
  id: string;
  name: string;
  stored_name?: string;
  bytes: number;
  words: number;
  sha256: string;
  document_date?: string | null;
  date_source?: "filename" | null;
  text?: string;
  truncated?: boolean;
};
export type Diagnostic = { severity: string; code: string; message: string };
export type Job = {
  scope?: {
    document_count: number;
    date_from: string | null;
    date_to: string | null;
    include_undated: boolean;
    explicit_documents: boolean;
  } | null;
  id: string;
  tool: string;
  state: string;
  stage: string;
  created: string;
  finished: string | null;
  run_dir: string | null;
  diagnostics: Diagnostic[];
  trashed?: number;
};
export type Param = {
  name: string;
  /** What to print above the input. Resolved by the engine, never derived here. */
  label: string;
  type: string;
  default: unknown;
  required: boolean;
  help: string;
  choices: (string | number)[];
  /** Words for each choice where its value is an id (models), from the engine. */
  choice_labels?: Record<string, string>;
  minimum: number | null;
  maximum: number | null;
};
export type Tool = {
  availability?: {
    state: "available" | "needs_setup" | "needs_input" | "needs_model";
    message: string;
    missing: string[];
    /** needs_model: the registered model to download from the Models page. */
    model_id?: string;
    model_name?: string;
    size_mb?: number;
  };
  name: string;
  /** The tool's name in words, from core/profiler/labels.py. */
  label: string;
  description: string;
  requires_parse: boolean;
  params: Param[];
  input_kind: string;
  outputs: string[];
  category?: "analysis" | "visualization";
  /** Syllabus family id (topics, words, sentiment, ...), from the engine. */
  family?: string;
  /** The family in words, resolved engine-side (core/profiler/registry.py). */
  family_label?: string;
};
/** One installed component, named and explained by the engine. */
export type Component = {
  name: string;
  label: string;
  purpose: string;
  present: boolean;
};
export type Setup = {
  python: string;
  workspace: string;
  offline: boolean;
  components: Component[];
  sample_available: boolean;
};
export type Envelope = {
  tool: string;
  created: string;
  artifacts: { path: string; kind: string; description?: string }[];
  inputs: { path: string; sha256: string }[];
  params: Record<string, unknown>;
  diagnostics: Diagnostic[];
  corpus_sha256: string;
};
export type RecommendedChart = {
  kind: string;
  x: string;
  y: string;
  /** How rows sharing an x value are combined; "" when the kind takes none. */
  agg: string;
  /** Categories to keep, or null for all of them. A timeline keeps all. */
  top_n?: number | null;
  question: string;
  why: string;
};
/** The words half of a reading, without the fetching. Shared by a finished
 *  run's insight and a live analysis, which produce it from the same code. */
export type Reading = {
  headline: string;
  observations: string[];
  cautions: string[];
  recommended_charts: RecommendedChart[];
  /** The tool's own figures this table can feed, by the question each
   *  answers. A tool with figures gets no generic recommendations. */
  figures?: FigureOffer[];
  /** Why this result is read as a table first; empty when a figure leads. */
  table_first?: string;
};
export type FigureOffer = {
  panel: string;
  title: string;
  question: string;
  shape: string;
};
/** A plain-language reading of a finished run, produced by the Python engine. */
export type Insight = Reading & {
  tool: string;
  available: boolean;
  reason?: string;
  table?: string;
};
/** Provenance of a visualization: carried over from NLP Suite 1.6.38, or new. */
export type VisualizationInfo = {
  name: string;
  title: string;
  origin: "legacy" | "legacy-extended" | "new";
  is_legacy: boolean;
  tool: string;
  produces: string[];
  summary: string;
  legacy_module: string;
  legacy_note: string;
  advantage: string;
};
/** A panel's declaration: what it draws and what it takes. Built by the
 *  engine (core/viz/panelspec.py) so the app never hard-codes a panel's
 *  controls. */
export type PanelInfo = {
  name: string;
  title: string;
  tool: string;
  shape: string;
  summary: string;
  requires: string[];
  params: Param[];
  notes: string[];
  /** The question this figure answers, in the reader's words. */
  question?: string;
};

/** What one mark stands for, as filters a caller can resolve. Never row
 *  indices: an index does not survive a re-run or a re-sort. */
export type PanelEvidence = {
  scope: string;
  /** (column, value) pairs, ANDed. */
  filters: [string, string][];
  count: number;
  describe: string;
  /** The text to search the corpus for, to read this mark in context. Set
   *  by the builder for terms evidence; empty when no honest lookup exists
   *  (a windowed collocation is not a phrase). */
  phrase?: string;
  /** The run counted lemmas, so the lookup must match every form. */
  lemma?: boolean;
};

/** One drawn thing, and what it stands for. */
export type PanelMark = {
  key: string;
  label: string;
  x: number;
  y: number;
  group: string;
  size: number | null;
  labelled: boolean;
  evidence: PanelEvidence;
  /** A heatmap cell's number, which its colour encodes. */
  value?: number | null;
  /** The small-multiple panel this mark belongs to. */
  facet?: string;
};

/** A link between two node marks of a `network` panel; clickable, because
 *  in a co-occurrence network the link is the finding. */
export type PanelEdge = {
  key: string;
  source: string;
  target: string;
  weight: number;
  label: string;
  evidence: PanelEvidence;
};

/** A reference line, carrying the reason it is there. */
export type PanelAnnotation = {
  kind: "vline" | "hline" | "note";
  value: number;
  label: string;
  note: string;
};

/**
 * A prepared panel, as the engine hands it over: geometry plus evidence,
 * not a rendered picture. The app draws it so every mark stays clickable —
 * the same reason `ChartCanvas` draws charts rather than embedding them.
 */
export type PreparedPanel = {
  ok: true;
  panel: string;
  shape: string;
  title: string;
  subtitle: string;
  xLabel: string;
  yLabel: string;
  groups: string[];
  caption: string;
  notes: string[];
  marks: PanelMark[];
  annotations: PanelAnnotation[];
  table: Table;
  diagnostics: Diagnostic[];
  /** Category axes for the shapes whose x/y are indexes into them. */
  xCategories?: string[];
  yCategories?: string[];
  /** Heatmap: "sequential" or "diverging". */
  colorScale?: string;
  edges?: PanelEdge[];
  /** Network: "straight" or "elbow" (a dendrogram). */
  edgeStyle?: string;
  /** Small multiples, in draw order. */
  facets?: string[];
  /** A gap in x wider than this breaks a line. */
  lineGap?: number;
  /** Groups drawn as points and never joined. */
  pointsOnly?: string[];
  /** x holds log10 of the quantity; ticks are labelled with the quantity. */
  xLog10?: boolean;
  /** Heatmap: what a cell's colour stands for, the colour scale's title. */
  valueLabel?: string;
  /** The builder's preferred drawing size; a heatmap of 87 documents needs
   *  more height than a ranking of twenty words. */
  width?: number;
  height?: number;
};

/** A refusal in the same shape as a success (R-C7). */
export type PanelRefusal = { ok: false; diagnostics: Diagnostic[] };

export type PanelAnswer = PreparedPanel | PanelRefusal;

export type Table = {
  columns: string[];
  rows: Record<string, string>[];
  total: number;
  truncated: boolean;
  /** Null when the engine does not paginate this table; a number when it does. */
  filtered_total?: number | null;
  offset?: number | null;
};
/**
 * One CSV result table somewhere in this project, as the Explore page lists
 * them. `manifest` marks the runner's record of which documents were read:
 * worth being able to open, wrong to open by default.
 */
export type TableSource = {
  job: string;
  index: number;
  tool: string;
  label: string;
  path: string;
  description: string;
  created: string;
  state: string;
  manifest: boolean;
};
export type ImportReport = {
  imported: number;
  duplicates: number;
  errors: { name: string; message: string }[];
};

/**
 * A timestamp as a reader reads it. Shared rather than redefined per page:
 * the same run shown in two places was showing two different dates.
 */
export const when = (value: string): string =>
  new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

/** Keep required dropdown state consistent with the first visible option. */
export function desktopParams(tool: Tool): Param[] {
  return tool.params
    .filter(
      (param) => !(tool.name === "spellcheck" && param.name === "correct"),
    )
    .map((param) =>
      tool.name === "search" && param.name === "mode"
        ? {
            ...param,
            choices: param.choices.filter((choice) => choice !== "csv"),
          }
        : param,
    );
}

export function initialParams(tool: Tool): Record<string, unknown> {
  return Object.fromEntries(
    desktopParams(tool).map((param) => [
      param.name,
      param.default ??
        (param.required && param.choices.length ? param.choices[0] : null),
    ]),
  );
}

let connection: Connection;
export async function connect(): Promise<void> {
  if (isTauri()) {
    connection = await invoke<Connection>("connection");
  } else {
    const hash = new URLSearchParams(location.hash.slice(1));
    const token =
      hash.get("token") || sessionStorage.getItem("nlp-token") || "";
    connection = { baseUrl: location.origin, token };
    if (token) sessionStorage.setItem("nlp-token", token);
    if (hash.has("token")) history.replaceState(null, "", location.pathname);
  }
  if (!connection.token)
    throw new Error(
      "Open the desktop app, or use the authenticated preview link printed by the launcher.",
    );
  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      await api("/health");
      return;
    } catch (error) {
      if (attempt === 19) throw error;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
}

/** Fetch a catalog that may take a while on first call (cold package scan). */
export async function apiWithTimeout<T>(
  path: string,
  seconds: number,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () =>
        reject(
          new Error(
            `The engine did not answer ${path} within ${seconds}s. It may still be loading packages — wait a moment and reload.`,
          ),
        ),
      seconds * 1000,
    );
  });
  try {
    return await Promise.race([api<T>(path), timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function request(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const response = await fetch(`${connection.baseUrl}/api${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${connection.token}`, ...init.headers },
  });
  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({ detail: `Request failed (${response.status})` }));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "Some settings are invalid. Please check your inputs.",
    );
  }
  return response;
}
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  return (await request(path, init)).json();
}
export function post<T>(path: string, body: unknown): Promise<T> {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
/** A publication figure's formats (core/viz/static). */
export type FigureFormat = "png" | "svg" | "pdf";

/** A publication figure, or why it could not be drawn. */
export type FigureImage =
  | { ok: true; blob: Blob; format: FigureFormat; warnings: number }
  | { ok: false; diagnostics: Diagnostic[] };

/**
 * One publication figure (matplotlib + seaborn), drawn by the engine.
 *
 * The server answers with the image itself, or -- like every panel refusal
 * -- with `ok: false` and diagnostics as JSON, so the content type says
 * which came back.
 */
export async function publicationFigure(
  path: string,
  body: Record<string, unknown> & { format: FigureFormat },
): Promise<FigureImage> {
  const response = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if ((response.headers.get("content-type") ?? "").includes("json")) {
    const refused = (await response.json()) as { diagnostics?: Diagnostic[] };
    return { ok: false, diagnostics: refused.diagnostics ?? [] };
  }
  return {
    ok: true,
    blob: await response.blob(),
    format: body.format,
    warnings: Number(response.headers.get("x-figure-warnings") ?? 0),
  };
}

export function del<T>(path: string): Promise<T> {
  return api(path, { method: "DELETE" });
}
/** Two runs compared artifact by artifact (the graded backend questions). */
export type CompareAnswer = {
  ok: boolean;
  /** True when the two runs agree on everything `core/compare` checks. */
  passed: boolean;
  expected: string;
  actual: string;
  compared_rows?: number;
  compared_columns?: number;
  /** Where they disagree: Row / Column / Expected / Actual / Issue. */
  diff?: Table | null;
  diagnostics?: { severity: string; code: string; message: string }[];
};
export function compareRuns(
  project: string,
  expected: string,
  actual: string,
): Promise<CompareAnswer> {
  return post<CompareAnswer>(`/projects/${project}/compare`, {
    expected,
    actual,
  });
}
export async function upload(
  project: string,
  files: File[],
): Promise<ImportReport> {
  const report: ImportReport = { imported: 0, duplicates: 0, errors: [] };
  for (const file of files) {
    try {
      const result = await api<{ duplicate?: boolean }>(
        `/projects/${project}/documents?name=${encodeURIComponent(file.name)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: file,
        },
      );
      if (result.duplicate) report.duplicates++;
      else report.imported++;
    } catch (error) {
      report.errors.push({
        name: file.name,
        message: String(error instanceof Error ? error.message : error),
      });
    }
  }
  return report;
}
export async function download(path: string, name: string): Promise<void> {
  if (isTauri()) {
    await invoke("save_download", { name, path });
  } else {
    const blob = await (await request(path)).blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

/**
 * A wider window of one loaded document, around verified occurrence offsets.
 *
 * GET with query parameters rather than a POST body: it is a pure read whose
 * answer depends only on the loaded snapshot, the document and the offsets.
 */
export async function sourcePassage(
  project: string,
  documentId: string,
  characterStart: number,
  characterEnd: number | null,
  radius = 100,
): Promise<SourcePassage> {
  const params = new URLSearchParams({
    document_id: documentId,
    character_start: String(characterStart),
    radius: String(radius),
  });
  if (characterEnd !== null) {
    params.set("character_end", String(characterEnd));
  }
  return api<SourcePassage>(
    `/projects/${project}/live/source?${params.toString()}`,
  );
}

/** Artifact text, for the small set of artifacts the app renders itself. */
export async function artifactText(path: string): Promise<string> {
  return (await request(path)).text();
}

export async function artifactBlobUrl(path: string): Promise<string> {
  const response = await request(path);
  return URL.createObjectURL(await response.blob());
}

/** HTML needs its own response policy: blob/srcdoc inherit the app's CSP. */
export async function artifactFrameUrl(path: string): Promise<string> {
  if (
    !/^\/projects\/[a-f0-9]{32}\/jobs\/[a-f0-9]{32}\/artifacts\/\d{1,9}$/i.test(
      path,
    )
  ) {
    throw new Error("Invalid artifact preview path");
  }
  // convertFileSrc encodes its entire argument (including slashes). Use it
  // only for the platform-specific origin, then append our validated route.
  if (isTauri()) return convertFileSrc("", "nlp-viz") + path.slice(1);
  const ticket = await post<{ path: string }>(`${path}/preview`, {});
  if (!/^\/api\/previews\/[A-Za-z0-9_-]{43}$/.test(ticket.path)) {
    throw new Error("Invalid preview response");
  }
  return connection.baseUrl + ticket.path;
}

export function filterRows(table: Table, query: string): Table["rows"] {
  const needle = query.toLocaleLowerCase();
  return table.rows.filter((row) =>
    Object.values(row).some((value) =>
      String(value ?? "")
        .toLocaleLowerCase()
        .includes(needle),
    ),
  );
}

/**
 * Tool names in words.
 *
 * These used to be a hand-written table in this file. It fell out of step with
 * the engine in both directions at once: three tools added to the desktop
 * catalog later had no entry and appeared as `collocations`, `tfidf` and
 * `dispersion` in lower case, while three entries named tools the desktop does
 * not publish. Nothing could notice, because nothing compared the two.
 *
 * The engine now sends `label` with every tool. This map is filled from that
 * catalog on load and read back when only a tool's name is in hand — a job
 * row, a run record — so a name is never invented on this side.
 */
const labels = new Map<string, string>();

export function rememberLabels(tools: Tool[]): void {
  for (const tool of tools) if (tool.label) labels.set(tool.name, tool.label);
}

/** Last resort for a name the catalog has never carried (a tool since removed). */
export function humanize(name: string): string {
  const words = name.replaceAll("_", " ").replaceAll("-", " ").trim();
  return words ? words[0].toUpperCase() + words.slice(1) : name;
}

export const title = (name: string): string =>
  labels.get(name) ?? humanize(name);

/** One pretrained model, as the Models page shows it (desktop_backend/models.py). */
export type ModelInfo = {
  id: string;
  name: string;
  kind: "token_embeddings" | "sentence_embeddings" | "classifier";
  kindLabel: string;
  description: string;
  sizeMb: number;
  status: "ready" | "not_downloaded" | "corrupt" | "unpublished" | "downloading";
  /** Ships inside the app (and so cannot be removed). */
  bundled: boolean;
  /** A downloaded copy the reader can delete to free space. */
  removable: boolean;
  license: string;
  source: string;
  precision: string;
  /** Tool names that can run this model. */
  usedBy: string[];
  download: {
    state: "downloading" | "failed" | "cancelled";
    done: number;
    total: number;
    message: string;
  } | null;
  freedMb?: number;
  error?: string;
};

export type ModelListing = { models: ModelInfo[]; folder: string };

export const listModels = (): Promise<ModelListing> => api("/models");

export const downloadModel = (id: string): Promise<ModelInfo> =>
  api(`/models/${encodeURIComponent(id)}/download`, { method: "POST" });

export const cancelModelDownload = (id: string): Promise<ModelInfo> =>
  api(`/models/${encodeURIComponent(id)}/cancel`, { method: "POST" });

export const deleteModel = (id: string): Promise<ModelInfo> =>
  del(`/models/${encodeURIComponent(id)}`);

/** Corpus at a glance (desktop_backend/glance.py): the newest result and what it found. */
export type GlanceFigure = { tool: string; label: string; panel: string; path: string };
export type Glance = {
  state: "none" | "queued" | "running" | "ready" | "stale" | "failed" | "partial" | "cancelled" | "interrupted";
  key: string;
  job?: Job;
  summary?: string[];
  figures?: GlanceFigure[];
};

export const glanceStatus = (projectId: string): Promise<Glance> => api(`/projects/${projectId}/glance`);

export const startGlance = (projectId: string, parser: string): Promise<Job> =>
  post(`/projects/${projectId}/glance`, { parser });

export const glanceFigureUrl = (projectId: string, path: string): Promise<string> =>
  artifactBlobUrl(`/projects/${projectId}/glance/figure?path=${encodeURIComponent(path)}`);
