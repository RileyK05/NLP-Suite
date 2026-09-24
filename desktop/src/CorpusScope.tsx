import { useState } from "react";
import type { Document } from "./api";
import {
  allDocuments,
  selectedDocuments,
  selectionError,
  type CorpusSelection as Selection,
} from "./corpusSelection";

export function CorpusScope({
  documents,
  value,
  onChange,
  disabled = false,
}: {
  documents: Document[];
  value: Selection;
  onChange: (selection: Selection) => void;
  disabled?: boolean;
}) {
  const [query, setQuery] = useState("");
  const visible = documents.filter((doc) =>
    doc.name.toLowerCase().includes(query.toLowerCase()),
  );
  const selected = selectedDocuments(documents, value);
  const error = selectionError(documents, value);
  const set = (patch: Partial<Selection>) => onChange({ ...value, ...patch });
  const undated = documents.filter((doc) => !doc.document_date).length;
  return (
    <fieldset className="corpus-selection" disabled={disabled}>
      <legend>Which documents should this analysis read?</legend>
      <p role="status">
        <strong>
          {selected.length} of {documents.length} documents
        </strong>{" "}
        · {selected.reduce((sum, doc) => sum + doc.words, 0).toLocaleString()}{" "}
        words
      </p>
      <div className="drop-actions">
        <label>
          <input
            type="radio"
            name="corpus-choice"
            checked={value.document_ids === null}
            onChange={() => set({ document_ids: null })}
          />{" "}
          All documents before date filtering
        </label>
        <label>
          <input
            type="radio"
            name="corpus-choice"
            checked={value.document_ids !== null}
            onChange={() => set({ document_ids: [] })}
          />{" "}
          Choose specific documents
        </label>
      </div>
      {value.document_ids !== null && (
        <>
          <label className="field-label">
            Find documents to select
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <div className="drop-actions">
            <button
              type="button"
              className="secondary"
              onClick={() =>
                set({
                  document_ids: [
                    ...new Set([
                      ...value.document_ids!,
                      ...visible.map((doc) => doc.id),
                    ]),
                  ],
                })
              }
            >
              Select matching names
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => set({ document_ids: [] })}
            >
              Clear document choices
            </button>
          </div>
          <div className="corpus-selection-list">
            {visible.map((doc) => (
              <label key={doc.id}>
                <input
                  type="checkbox"
                  checked={value.document_ids!.includes(doc.id)}
                  onChange={(event) =>
                    set({
                      document_ids: event.target.checked
                        ? [...value.document_ids!, doc.id]
                        : value.document_ids!.filter((id) => id !== doc.id),
                    })
                  }
                />
                <span>
                  {doc.name}
                  <small>{doc.document_date ?? "Undated"}</small>
                </span>
              </label>
            ))}
            {!visible.length && (
              <p>No names match. Existing choices are retained.</p>
            )}
          </div>
        </>
      )}
      <div className="corpus-selection-dates">
        <label className="field-label">
          From (inclusive)
          <input
            type="date"
            value={value.date_from ?? ""}
            onChange={(event) => set({ date_from: event.target.value || null })}
          />
        </label>
        <label className="field-label">
          Through (inclusive)
          <input
            type="date"
            value={value.date_to ?? ""}
            onChange={(event) => set({ date_to: event.target.value || null })}
          />
        </label>
      </div>
      <p className="muted">
        Dates come from document filenames, never the import date. {undated}{" "}
        documents have no recognized date. Leave both dates blank for no time
        restriction.
      </p>
      {!!(value.date_from || value.date_to) && (
        <label>
          <input
            type="checkbox"
            checked={value.include_undated}
            onChange={(event) => set({ include_undated: event.target.checked })}
          />{" "}
          Also include undated documents
        </label>
      )}
      {error && (
        <p className="alert warning" role="alert">
          {error}
        </p>
      )}
      <button
        type="button"
        className="text-button"
        onClick={() => onChange(allDocuments())}
      >
        Reset to the whole corpus
      </button>
    </fieldset>
  );
}
