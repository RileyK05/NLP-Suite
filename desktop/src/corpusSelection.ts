import type { Document } from "./api";

export type CorpusSelection = {
  document_ids: string[] | null;
  date_from: string | null;
  date_to: string | null;
  include_undated: boolean;
  /** The same window along an order axis (chapters, sessions). */
  order_from: number | null;
  order_to: number | null;
  include_unordered: boolean;
  /** Keep documents whose detail is one of the listed values, ANDed across
   *  details. `"(empty)"` selects documents without that detail. */
  fields: Record<string, string[]> | null;
};

export const allDocuments = (): CorpusSelection => ({
  document_ids: null,
  date_from: null,
  date_to: null,
  include_undated: false,
  order_from: null,
  order_to: null,
  include_unordered: false,
  fields: null,
});

const EMPTY = "(empty)";

/** The document's detail value, or "(empty)" when it has none. */
function fieldValue(doc: Document, name: string): string {
  const value = doc.fields?.[name] ?? "";
  return value ? value : EMPTY;
}

/** Preview only. The server independently resolves and freezes the selection. */
export function selectedDocuments(
  documents: Document[],
  selection: CorpusSelection,
): Document[] {
  const ids =
    selection.document_ids === null ? null : new Set(selection.document_ids);
  const window = !!(selection.date_from || selection.date_to);
  const orderWindow = !!(
    selection.order_from != null || selection.order_to != null
  );
  const filters = Object.entries(selection.fields ?? {});
  return documents.filter((doc) => {
    if (ids && !ids.has(doc.id)) return false;
    for (const [name, allowed] of filters) {
      const wanted = allowed.map((value) => value.toLowerCase());
      if (!wanted.includes(fieldValue(doc, name).toLowerCase())) return false;
    }
    if (window) {
      if (!doc.document_date) {
        if (!selection.include_undated) return false;
      } else if (
        (selection.date_from && doc.document_date < selection.date_from) ||
        (selection.date_to && doc.document_date > selection.date_to)
      ) {
        return false;
      }
    }
    if (orderWindow) {
      const order = doc.document_order ?? null;
      if (order === null) {
        if (!selection.include_unordered) return false;
      } else if (
        (selection.order_from != null && order < selection.order_from) ||
        (selection.order_to != null && order > selection.order_to)
      ) {
        return false;
      }
    }
    return true;
  });
}

/** What a finished run's scope narrowed beyond dates, as a sentence or two
 *  for its reproducibility record: the order window and the detail filters. */
export function scopeFilters(scope: {
  order_from?: number | null;
  order_to?: number | null;
  include_unordered?: boolean;
  fields?: Record<string, string[]> | null;
}): string {
  const parts: string[] = [];
  const from = scope.order_from,
    to = scope.order_to;
  if (from != null || to != null) {
    const window =
      from != null && to != null
        ? `Order ${from} through ${to}, inclusive.`
        : from != null
          ? `Order ${from} and later.`
          : `Order up to ${to}.`;
    parts.push(
      ` ${window}${
        scope.include_unordered
          ? " Documents with no order allowed."
          : " Documents with no order excluded."
      }`,
    );
  }
  for (const [name, values] of Object.entries(scope.fields ?? {})) {
    parts.push(` Only documents whose ${name} is ${values.join(" or ")}.`);
  }
  return parts.join("");
}

export function selectionError(
  documents: Document[],
  selection: CorpusSelection,
): string | null {
  for (const value of [selection.date_from, selection.date_to]) {
    if (
      value &&
      (!/^\d{4}-\d{2}-\d{2}$/.test(value) ||
        Number.isNaN(Date.parse(value)) ||
        new Date(value).toISOString().slice(0, 10) !== value)
    ) {
      return "Choose valid calendar dates.";
    }
  }
  if (
    selection.date_from &&
    selection.date_to &&
    selection.date_from > selection.date_to
  ) {
    return "The start date must be on or before the end date.";
  }
  for (const value of [selection.order_from, selection.order_to]) {
    if (value != null && !Number.isFinite(value)) {
      return "Order bounds must be real numbers.";
    }
  }
  if (
    selection.order_from != null &&
    selection.order_to != null &&
    selection.order_from > selection.order_to
  ) {
    return "The first order must be on or before the last.";
  }
  for (const [name, allowed] of Object.entries(selection.fields ?? {})) {
    if (!name.trim()) return "A detail filter needs the detail's name.";
    if (!allowed.length) {
      return `The filter on “${name}” lists no values; remove it or choose at least one.`;
    }
  }
  if (
    selection.document_ids?.some(
      (id) => !documents.some((doc) => doc.id === id),
    )
  ) {
    return "The document list changed. Choose the documents again.";
  }
  return selectedDocuments(documents, selection).length
    ? null
    : "No documents match this selection.";
}
