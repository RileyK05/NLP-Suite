import type { Document } from "./api";

export type CorpusSelection = {
  document_ids: string[] | null;
  date_from: string | null;
  date_to: string | null;
  include_undated: boolean;
};

export const allDocuments = (): CorpusSelection => ({
  document_ids: null,
  date_from: null,
  date_to: null,
  include_undated: false,
});

/** Preview only. The server independently resolves and freezes the selection. */
export function selectedDocuments(
  documents: Document[],
  selection: CorpusSelection,
): Document[] {
  const ids =
    selection.document_ids === null ? null : new Set(selection.document_ids);
  const window = !!(selection.date_from || selection.date_to);
  return documents.filter((doc) => {
    if (ids && !ids.has(doc.id)) return false;
    if (!window) return true;
    if (!doc.document_date) return selection.include_undated;
    return (
      (!selection.date_from || doc.document_date >= selection.date_from) &&
      (!selection.date_to || doc.document_date <= selection.date_to)
    );
  });
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
