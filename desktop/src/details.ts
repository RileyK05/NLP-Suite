/**
 * Document details on the Corpus page (docs/internal/PLAN_0.5.0.md 1.8): each
 * document's Date, Speaker, Kind, Chapter ..., where each value came from, and
 * how the project's documents line up (the axis).
 */
import { api, del, post } from "./api";

export type DetailSource = "filename" | "csv" | "split" | "user";

export type DetailValue = {
  value: string;
  source: DetailSource;
  overridden: { value: string; source: DetailSource }[];
};

export type AxisChoice = "auto" | "time" | "order" | "none";

/** What is left out of the text the tools read (plan 5.2). */
export type TextCleaning = {
  /** Bracketed stage directions like "(Applause.)" are removed at read time. */
  stage_directions: boolean;
  /** The project's own words that make a bracketed span a stage direction. */
  extra_terms: string[];
};

/** A dated event drawn as a stopline on every time figure (plan section 4). */
export type ProjectEvent = {
  /** What happened ("Pearl Harbor"). */
  name: string;
  /** A year (1941), a decimal year (1941.5), or an ISO date (1941-12-07). */
  date: string;
  /** An optional note, kept with the event. */
  note: string;
};

export type DetailSettings = {
  axis: AxisChoice;
  order_label: string;
  filename_detection: boolean;
  part_names: Record<string, string>;
  text_cleaning?: TextCleaning;
  /** Dated events for time figures; empty until the reader adds one. */
  events: ProjectEvent[];
};

export type ResolvedAxis = {
  kind: "time" | "order" | "none";
  noun: string;
  /** How many documents the axis places. */
  placed: number;
  first: string | null;
  last: string | null;
};

export type ProjectDetails = {
  names: {
    name: string;
    count: number;
    distinct: number;
    sources: DetailSource[];
    sample_values: string[];
  }[];
  documents: Record<string, Record<string, DetailValue>>;
  template: {
    parts: string[];
    separator: string;
    fits: number;
    total: number;
    misses: string[];
    describe: string;
  } | null;
  settings: DetailSettings;
  revision: number;
  axis: ResolvedAxis;
};

export const SOURCE_WORDS: Record<DetailSource, string> = {
  filename: "from the file name",
  csv: "from a spreadsheet",
  split: "set when the book was split",
  user: "typed by you",
};

const base = (projectId: string) => `/projects/${projectId}`;

export const loadDetails = (projectId: string) =>
  api<ProjectDetails>(`${base(projectId)}/fields`);

export const setDetail = (
  projectId: string,
  documentId: string,
  name: string,
  value: string,
  expectedRevision: number,
) =>
  post<{ revision: number }>(`${base(projectId)}/fields`, {
    rows: [{ document_id: documentId, name, value, source: "user" }],
    expected_revision: expectedRevision,
  });

/** What importing a spreadsheet would do (or did, when `applied`). */
export type ImportPreview = {
  name_column: string;
  /** "" when rows name documents by file name; else the detail ("Speaker"). */
  matched_on: string;
  columns: {
    column: string;
    /** The detail it becomes; "" leaves it out. */
    target: string;
    kind: "date" | "number" | "text";
    filled: number;
    samples: string[];
    /** Why the target cannot name a detail, or "". */
    problem: string;
  }[];
  matched: number;
  matched_by: Record<string, number>;
  unmatched_rows: string[];
  unmatched_row_count: number;
  unmatched_documents: string[];
  unmatched_document_count: number;
  duplicate_rows: string[];
  preview: { document: string; values: Record<string, string> }[];
  diagnostics: { severity: string; code: string; message: string }[];
  applied: boolean;
  revision?: number;
};

export type ImportRequest = {
  filename: string;
  /** The file's bytes, base64. */
  content: string;
  name_column?: string | null;
  targets?: Record<string, string> | null;
  dry_run: boolean;
  expected_revision?: number | null;
};

export const importDetails = (projectId: string, body: ImportRequest) =>
  post<ImportPreview>(`${base(projectId)}/fields/import`, body);

/** A chosen file's bytes as base64, for `importDetails`. */
export function fileToBase64(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result ?? "");
      resolve(url.slice(url.indexOf(",") + 1));
    };
    reader.onerror = () => reject(reader.error ?? new Error("unreadable"));
    reader.readAsDataURL(file);
  });
}

/** "4 documents, by Speaker" / "12 documents by file name", for the preview's headline. */
export function matchWords(preview: ImportPreview): string {
  const count = `${preview.matched} document${preview.matched === 1 ? "" : "s"}`;
  return preview.matched_on
    ? `${count}, through each one's ${preview.matched_on}`
    : `${count} by file name`;
}

/** How to cut a book into its sections (`desktop_backend/sections.py` SplitBody). */
export type SplitOptions = {
  rule: "auto" | "headings" | "blank-gap" | "words" | "pattern";
  pattern: string;
  regex: boolean;
  block_words: number;
  keep_front_matter: boolean;
  keep_whole: boolean;
  expected_sha256?: string | null;
  /** A transcript, cut by speaker instead of by chapter. */
  transcript?: boolean;
  per_turn?: boolean;
  /** The speakers to keep; null keeps everyone. */
  speakers?: string[] | null;
};

export type TurnPreview = {
  document: { id: string; name: string; sha256: string };
  event: string;
  speakers: { name: string; turns: number; words: number; kept: boolean }[];
  directions: { text: string; count: number }[];
  preamble_words: number;
  /** How many documents the split would make. */
  documents: number;
};

export const previewTurns = (
  projectId: string,
  documentId: string,
  options: SplitOptions,
) =>
  post<TurnPreview>(`${sectionsPath(projectId, documentId)}/preview`, {
    ...options,
    transcript: true,
  });

export type SplitPreview = {
  document: { id: string; name: string; sha256: string };
  rule: SplitOptions["rule"];
  work: string;
  /** What one section is called: "Chapter", "Section". */
  level: string;
  sections: {
    order: number;
    title: string;
    level: string;
    parents: [string, string][];
    words: number;
    opening: string;
  }[];
  left_out: { what: string; size: number }[];
  diagnostics: {
    severity: string;
    code: string;
    message: string;
    order?: number;
  }[];
};

const sectionsPath = (projectId: string, documentId: string) =>
  `${base(projectId)}/documents/${documentId}/sections`;

export const previewSplit = (
  projectId: string,
  documentId: string,
  options: SplitOptions,
) =>
  post<SplitPreview>(`${sectionsPath(projectId, documentId)}/preview`, options);

export const applySplit = (
  projectId: string,
  documentId: string,
  options: SplitOptions,
) =>
  post<{ documents: { id: string; name: string }[] }>(
    `${sectionsPath(projectId, documentId)}/apply`,
    options,
  );

/** "Split into 61 chapters". */
export function splitButtonWords(preview: SplitPreview): string {
  const count = preview.sections.length;
  const noun = preview.level.toLowerCase();
  return `Split into ${count} ${count === 1 ? noun : `${noun}s`}`;
}

export const removeTypedDetail = (projectId: string, name: string) =>
  del<{ deleted: boolean }>(
    `${base(projectId)}/fields/${encodeURIComponent(name)}`,
  );

export const saveDetailSettings = (
  projectId: string,
  settings: DetailSettings,
  expectedRevision: number,
) =>
  post<{ settings: DetailSettings; revision: number }>(
    `${base(projectId)}/settings`,
    {
      settings,
      expected_revision: expectedRevision,
    },
  );

/** What the project's text cleaning would take out, counted and unchanged. */
export type CleaningPreview = {
  found: number;
  samples: string[];
  documents: number;
  cleaning: TextCleaning;
};

export const loadCleaningPreview = (projectId: string) =>
  api<CleaningPreview>(`${base(projectId)}/cleaning_preview`);

/** "lined up by time", "lined up by chapter", "not lined up". */
export function axisWords(axis: ResolvedAxis): string {
  if (axis.kind === "time") return "lined up by time";
  if (axis.kind === "order")
    // An unnamed step: "lined up in order", not "lined up by document".
    return axis.noun === "Document"
      ? "lined up in order"
      : `lined up by ${axis.noun.toLowerCase()}`;
  return "not lined up";
}

/** The corpus in one line: "87 documents · dated 1934–2024 · lined up by time · 2 details: Speaker, Kind". */
export function shapeLine(details: ProjectDetails, documents: number): string {
  const parts = [
    `${documents.toLocaleString()} document${documents === 1 ? "" : "s"}`,
  ];
  const { axis } = details;
  if (axis.kind === "time" && axis.first && axis.last)
    parts.push(`dated ${axis.first.slice(0, 4)}–${axis.last.slice(0, 4)}`);
  if (axis.kind === "order" && axis.first && axis.last)
    parts.push(`${axis.noun.toLowerCase()} ${axis.first}–${axis.last}`);
  parts.push(axisWords(axis));
  const shown = details.names.filter(
    (item) => !["date", "order"].includes(item.name.toLowerCase()),
  );
  if (shown.length)
    parts.push(
      `${shown.length} detail${shown.length === 1 ? "" : "s"}: ${shown.map((item) => item.name).join(", ")}`,
    );
  return parts.join(" · ");
}

/** A cell's hover text: where its value came from, and what it hides. */
export function sourceNote(item: DetailValue | undefined): string {
  if (!item) return "No value";
  const hidden = item.overridden.map(
    (under) => `${SOURCE_WORDS[under.source]}: ${under.value}`,
  );
  return [
    SOURCE_WORDS[item.source],
    ...(hidden.length ? [`replaces ${hidden.join("; ")}`] : []),
  ].join(" · ");
}
