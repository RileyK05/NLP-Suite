/**
 * The Compare page's data: saved comparisons, a preview of their sides, runs.
 *
 * A comparison is a definition only (desktop_backend/comparisons.py): its
 * results are ordinary runs of the `contrast` tool in the comparison's home
 * project, so they open in Past runs and draw with the same panels as every
 * other run.
 */
import { api, post, type Envelope, type Job, type Table } from "./api";

export type Selection = {
  document_ids?: string[] | null;
  date_from?: string | null;
  date_to?: string | null;
  include_undated?: boolean;
  fields?: Record<string, string[]> | null;
};

export type Side = {
  name: string;
  project_id: string;
  selection: Selection | null;
};

export type Method =
  | "measures"
  | "keyness"
  | "focus"
  | "topics"
  | "tone"
  | "meaning"
  | "carryover";

export type Definition = {
  sides: Side[];
  alignment: string;
  methods: Method[];
  focus: Record<string, string[]> | null;
  match: "lemma" | "form" | "exact-lowercase";
  topics: number;
};

export type Comparison = {
  id: string;
  name: string;
  definition: Definition;
  revision: number;
  created: string;
  updated: string;
};

export type SidePreview =
  | {
      name: string;
      documents: number;
      words: number;
      first_date: string | null;
      last_date: string | null;
      /** Each detail's number of distinct values, and the first few. */
      details: Record<string, { count: number; examples: string[] }>;
      error?: undefined;
    }
  | { name: string; error: string };

export type Preview = {
  sides: SidePreview[];
  alignments: { name: string; shared_values: number; examples: string[] }[];
  dated: boolean;
};

export type ComparisonRun = Job & {
  comparison_id: string;
  comparison_name: string;
};

/** A project's documents' details, for the side selection editors. */
export type ProjectFields = {
  names: {
    name: string;
    count: number;
    distinct: number;
    sample_values: string[];
  }[];
  documents: Record<string, Record<string, { value: string; source: string }>>;
};

/** Every method, in the order the page lists them (plan 3.5). */
export const METHODS: {
  id: Method;
  label: string;
  cost: string;
  help: string;
  available: boolean;
}[] = [
  {
    id: "measures",
    label: "Size and style",
    cost: "fast",
    help: "Sentence and word length, reading ease, vocabulary diversity, with effect sizes.",
    available: true,
  },
  {
    id: "keyness",
    label: "Distinctive words",
    cost: "fast",
    help: "The words each side uses far more often, per 10,000 words.",
    available: true,
  },
  {
    id: "focus",
    label: "Focus word rates",
    cost: "fast",
    help: "How often each side uses your focus words, per group.",
    available: true,
  },
  {
    id: "tone",
    label: "Tone",
    cost: "fast",
    help: "VADER's overall tone per document, side by side.",
    available: true,
  },
  {
    id: "topics",
    label: "Shared and distinct topics",
    cost: "a minute or two",
    help: "One topic model over every side, and each topic's share of each side.",
    available: true,
  },
  {
    id: "meaning",
    label: "Meaning map",
    cost: "slow",
    help: "A sentence map of similar meaning across the two sides is planned.",
    available: false,
  },
  {
    id: "carryover",
    label: "Carried-over passages",
    cost: "slow",
    help: "Reciprocal nearest sentence matches within each shared group, with both source document and sentence locations.",
    available: true,
  },
];

// ------------------------------------------------------------------ api --

const base = (projectId: string) => `/projects/${projectId}/comparisons`;

export const listComparisons = (projectId: string) =>
  api<Comparison[]>(base(projectId));
export const createComparison = (
  projectId: string,
  name: string,
  definition: Definition,
) => post<Comparison>(base(projectId), { name, definition });
export const saveComparison = (
  projectId: string,
  comparison: Pick<Comparison, "id" | "name" | "definition" | "revision">,
) =>
  post<Comparison>(`${base(projectId)}/${comparison.id}`, {
    name: comparison.name,
    definition: comparison.definition,
    expected_revision: comparison.revision,
  });
export const duplicateComparison = (projectId: string, id: string) =>
  post<Comparison>(`${base(projectId)}/${id}/duplicate`, {});
export const deleteComparison = (projectId: string, id: string) =>
  post<{ deleted: boolean }>(`${base(projectId)}/${id}/delete`, {});
export const previewComparison = (projectId: string, definition: Definition) =>
  post<Preview>(`${base(projectId)}/preview`, { definition });
export const runComparison = (projectId: string, id: string, parser: string) =>
  post<Job>(`${base(projectId)}/${id}/run`, { parser });
export const comparisonRuns = (projectId: string) =>
  api<ComparisonRun[]>(`${base(projectId)}/runs`);
export const projectFields = (projectId: string) =>
  api<ProjectFields>(`/projects/${projectId}/fields`);
export const runEnvelope = (projectId: string, jobId: string) =>
  api<Envelope>(`/projects/${projectId}/jobs/${jobId}/results`);
export const runTable = (projectId: string, jobId: string, index: number) =>
  api<Table>(`/projects/${projectId}/jobs/${jobId}/artifacts/${index}`);

// ---------------------------------------------------------------- focus --

/**
 * Focus words as the lexicon tool writes them: `immigration: immigration,
 * border, asylum; economy: jobs, wages`. Groups are separated by `;` or a
 * new line; a line with no name becomes the group "focus".
 */
export function parseFocus(text: string): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const chunk of text.split(/[;\n]/)) {
    const line = chunk.trim();
    if (!line) continue;
    const colon = line.indexOf(":");
    const name = colon > 0 ? line.slice(0, colon).trim() : "focus";
    const terms = (colon > 0 ? line.slice(colon + 1) : line)
      .split(",")
      .map((term) => term.trim().toLowerCase())
      .filter(Boolean);
    if (!terms.length) continue;
    groups[name] = [...(groups[name] ?? []), ...terms];
  }
  return groups;
}

export function formatFocus(focus: Record<string, string[]> | null): string {
  if (!focus) return "";
  return Object.entries(focus)
    .map(([name, terms]) => `${name}: ${terms.join(", ")}`)
    .join("\n");
}

/** A fresh comparison: the current project on both sides, told apart by a detail. */
export function newDefinition(projectId: string): Definition {
  return {
    sides: [
      { name: "Side A", project_id: projectId, selection: null },
      { name: "Side B", project_id: projectId, selection: null },
    ],
    alignment: "none",
    methods: ["measures", "keyness", "tone"],
    focus: null,
    match: "lemma",
    topics: 8,
  };
}

/** Every value each detail takes in a project, for the side's filter menus. */
export function detailValues(fields: ProjectFields): Record<string, string[]> {
  const found: Record<string, Set<string>> = {};
  for (const perDocument of Object.values(fields.documents))
    for (const [name, item] of Object.entries(perDocument)) {
      if (name === "Date") continue;
      (found[name] ??= new Set()).add(item.value);
    }
  return Object.fromEntries(
    Object.entries(found).map(([name, values]) => [
      name,
      [...values].sort((a, b) => a.localeCompare(b)),
    ]),
  );
}

/** What went wrong with a definition before it is sent, in words; empty when it can be run. */
export function definitionProblems(definition: Definition): string[] {
  const problems: string[] = [];
  const names = definition.sides.map((side) => side.name.trim().toLowerCase());
  if (names.some((name) => !name)) problems.push("Give every side a name.");
  if (new Set(names).size !== names.length)
    problems.push("Two sides have the same name.");
  if (!definition.methods.length) problems.push("Choose at least one method.");
  if (
    definition.methods.includes("focus") &&
    !Object.keys(definition.focus ?? {}).length
  )
    problems.push("Focus word rates need focus words.");
  if (
    definition.methods.includes("carryover") &&
    !Object.keys(definition.focus ?? {}).length
  )
    problems.push("Carried-over passages need focus words.");
  return problems;
}
