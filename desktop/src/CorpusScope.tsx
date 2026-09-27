import { useMemo, useState } from "react";
import type { Document } from "./api";
import {
  allDocuments,
  selectedDocuments,
  selectionError,
  type CorpusSelection as Selection,
} from "./corpusSelection";

/** Details the scope filters by range, not by value. */
const WINDOWED = new Set(["date", "order"]);

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
  // The details this corpus carries, and each one's values, for the filters.
  // Date and Order are left out: each has its own window above and below.
  const details = useMemo(() => {
    const found = new Map<string, Set<string>>();
    for (const doc of documents) {
      for (const [name, item] of Object.entries(doc.fields ?? {})) {
        if (WINDOWED.has(name.toLowerCase())) continue;
        if (!found.has(name)) found.set(name, new Set());
        found.get(name)!.add(item);
      }
    }
    return found;
  }, [documents]);
  const filters = value.fields ?? {};
  const setFilter = (name: string, chosen: string[] | null) => {
    const next = { ...filters };
    if (chosen === null) delete next[name];
    else next[name] = chosen;
    set({ fields: Object.keys(next).length ? next : null });
  };
  const optionsFor = (name: string) => {
    const values = [...(details.get(name) ?? [])].sort((a, b) =>
      a.localeCompare(b),
    );
    const someEmpty = documents.some((doc) => !doc.fields?.[name]);
    return someEmpty ? [...values, "(empty)"] : values;
  };
  const unordered = documents.filter(
    (doc) => doc.document_order === null || doc.document_order === undefined,
  ).length;
  const ordered = documents.length > unordered;
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
      {details.size > 0 && (
        <div className="corpus-detail-filters">
          {Object.entries(filters).map(([name, chosen]) => (
            <fieldset key={name} className="side-filter">
              <legend>
                {name}
                <button
                  type="button"
                  className="link-button"
                  onClick={() => setFilter(name, null)}
                  aria-label={`Remove the ${name} filter`}
                >
                  ×
                </button>
              </legend>
              <div className="side-filter-values">
                {optionsFor(name).map((item) => (
                  <label key={item}>
                    <input
                      type="checkbox"
                      checked={chosen.includes(item)}
                      onChange={(event) => {
                        const next = event.target.checked
                          ? [...chosen, item]
                          : chosen.filter((one) => one !== item);
                        setFilter(name, next.length ? next : null);
                      }}
                    />
                    {item}
                  </label>
                ))}
              </div>
            </fieldset>
          ))}
          {[...details.keys()].some((name) => !(name in filters)) && (
            <label className="field-label">
              Keep only documents whose detail is
              <select
                value=""
                onChange={(event) => {
                  const name = event.target.value;
                  if (name) setFilter(name, optionsFor(name));
                }}
              >
                <option value="">Choose a detail…</option>
                {[...details.keys()]
                  .filter((name) => !(name in filters))
                  .sort((a, b) => a.localeCompare(b))
                  .map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
              </select>
            </label>
          )}
        </div>
      )}
      {ordered && (
        <>
          <div className="corpus-selection-dates">
            <label className="field-label">
              From order (inclusive)
              <input
                type="number"
                step="any"
                value={value.order_from ?? ""}
                onChange={(event) =>
                  set({
                    order_from:
                      event.target.value === ""
                        ? null
                        : Number(event.target.value),
                  })
                }
              />
            </label>
            <label className="field-label">
              Through order (inclusive)
              <input
                type="number"
                step="any"
                value={value.order_to ?? ""}
                onChange={(event) =>
                  set({
                    order_to:
                      event.target.value === ""
                        ? null
                        : Number(event.target.value),
                  })
                }
              />
            </label>
          </div>
          {!!(value.order_from != null || value.order_to != null) &&
            unordered > 0 && (
              <label>
                <input
                  type="checkbox"
                  checked={value.include_unordered}
                  onChange={(event) =>
                    set({ include_unordered: event.target.checked })
                  }
                />{" "}
                Also include the {unordered} document(s) with no order
              </label>
            )}
        </>
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
