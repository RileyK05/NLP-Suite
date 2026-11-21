import type { TableSource } from "./api";

/** Results first: a run's input manifest is a record, not a finding. */
export function orderSources(sources: TableSource[]): TableSource[] {
  return [
    ...sources.filter((source) => !source.manifest),
    ...sources.filter((source) => source.manifest),
  ];
}

/** The table to land on when the page opens with nothing chosen yet. */
export function firstWorthOpening(sources: TableSource[]): TableSource | null {
  return orderSources(sources)[0] ?? null;
}

export const sourceKey = (source: TableSource): string =>
  `${source.job}:${source.index}`;
