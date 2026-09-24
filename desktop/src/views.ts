/**
 * Saved views: the workbench's arrangement, written down and reopened.
 *
 * The workbench holds its settings in a `Settings` object shaped for the
 * controls that edit it. The server stores a `ViewSettings` record shaped for
 * the model that validates it. They describe the same thing in two casings,
 * which is one casing too many — so the translation lives here, in one pair of
 * functions, rather than being spelled out at each call site.
 *
 * `tests/test_view_parity.py` reads this file and checks the two shapes still
 * correspond, because the other half of the contract is in Python and nothing
 * else would notice a field added on one side only.
 */

import { isLiveKind, type Agg, type Settings } from "./chartLayout";

/** A chart setting as the server stores it. Field names are the wire names. */
export type ViewSettings = {
  kind: string;
  x: string;
  y: string;
  group: string;
  agg: Agg;
  top_n: number;
  sort: Settings["sort"];
  bins: number;
};

/** One column equals one value: what clicking a mark means, stored. */
export type RowFilter = { column: string; equals: string };

/** Whether a view's source table is still the file it was saved over. */
export type SourceState = {
  state: "ok" | "changed" | "missing";
  detail: string;
};

export type View = {
  id: string;
  name: string;
  job: string;
  index: number;
  artifact_path: string;
  source_sha256: string;
  settings: ViewSettings;
  filters: RowFilter[];
  revision: number;
  created: string;
  updated: string;
  source: SourceState;
  /** What publishing this view would not reproduce. Sentences, not codes. */
  publication: string[];
};

/**
 * What a published chart will not reproduce, in the server's words.
 *
 * A plain lookup on purpose. The rule lives in `desktop_backend/views.py` and
 * arrives as data from `/api/chart-contract`, so adding a gap there makes it
 * appear here with no change to this file. Restating the rule in TypeScript is
 * how the desktop's copy of the tool names came to drift from the registry's.
 */
export type ChartContract = Record<string, Record<string, string>>;

export function publicationGaps(
  contract: ChartContract,
  settings: Settings,
): string[] {
  const note = contract[settings.kind]?.[settings.sort];
  return note ? [note] : [];
}

export const toViewSettings = (settings: Settings): ViewSettings => ({
  kind: settings.kind,
  x: settings.x,
  y: settings.y,
  group: settings.group,
  agg: settings.agg,
  top_n: settings.topN,
  sort: settings.sort,
  bins: settings.bins,
});

/**
 * A stored view's settings as the workbench's controls want them.
 *
 * `fallback` supplies anything a stored record cannot be trusted for: a view
 * saved against an older or newer build, or restored from a backup, may name a
 * chart kind this workbench does not draw. Falling back beats rendering
 * nothing, and beats trusting the string into a `LiveKind` cast that would fail
 * somewhere further in with no clue where it came from.
 */
export function fromViewSettings(
  stored: ViewSettings,
  fallback: Settings,
): Settings {
  return {
    kind: isLiveKind(stored.kind) ? stored.kind : fallback.kind,
    x: stored.x,
    y: stored.y,
    group: stored.group,
    agg: stored.agg,
    topN: stored.top_n,
    sort: stored.sort,
    bins: stored.bins,
  };
}

/** Whether what is on screen still matches what was saved. */
export function isDirty(
  view: View | null,
  settings: Settings,
  filters: RowFilter[],
): boolean {
  if (!view) return false;
  return (
    JSON.stringify(toViewSettings(settings)) !==
      JSON.stringify(view.settings) ||
    JSON.stringify(filters) !== JSON.stringify(view.filters)
  );
}

/**
 * The drill-down as stored predicates.
 *
 * A clicked mark means "the rows where the x column is this label" — plus the
 * group column when one is chosen. That is exactly what `rowsBehind` filters
 * on, so the two must agree; a stored predicate the workbench cannot reproduce
 * would reopen a view showing different rows than it was saved with.
 */
export function filtersFor(
  settings: Settings,
  selected: { label: string; group: string } | null,
): RowFilter[] {
  if (!selected) return [];
  const filters: RowFilter[] = [];
  if (settings.kind !== "histogram" && settings.x) {
    filters.push({ column: settings.x, equals: selected.label });
  }
  if (settings.group) {
    filters.push({ column: settings.group, equals: selected.group });
  }
  return filters;
}

/**
 * The reverse: the mark a stored drill-down stands for, or nothing.
 *
 * Takes the two column names rather than a whole `Settings`, so it can be
 * handed a stored `ViewSettings` without a cast. Those are the only fields
 * either shape contributes here.
 */
export function selectionFor(
  settings: { x: string; group: string },
  filters: RowFilter[],
): { label: string; group: string } | null {
  if (!filters.length) return null;
  const label = filters.find((f) => f.column === settings.x)?.equals ?? "";
  const group =
    filters.find((f) => f.column === settings.group)?.equals ?? "(all)";
  return { label, group };
}
