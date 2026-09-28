import { useCallback, useEffect, useMemo, useState } from "react";
import { Plus, Tags, X } from "lucide-react";
import type { Document } from "./api";
import { DetailImport } from "./DetailImport";
import { SplitBook } from "./SplitBook";
import {
  SOURCE_WORDS,
  loadCleaningPreview,
  loadDetails,
  removeTypedDetail,
  saveDetailSettings,
  setDetail,
  shapeLine,
  sourceNote,
  type AxisChoice,
  type CleaningPreview,
  type ProjectDetails,
  type ProjectEvent,
} from "./details";

const errorText = (caught: unknown) =>
  caught instanceof Error ? caught.message : String(caught);
/** Rows drawn before "Show all": a plain table stays quick at a few hundred inputs. */
const FIRST_ROWS = 100;

/**
 * The Corpus page's "Document details": what each document is (Date,
 * Speaker, Kind, Chapter ...), where each value came from, and how the
 * documents line up. Details read from file names need no clicks; this is
 * for the few names that were odd, and for details no name carries.
 */
export function CorpusDetails({
  projectId,
  documents,
  onChanged,
}: {
  projectId: string;
  documents: (Pick<Document, "id" | "name"> & { words?: number })[];
  /** Called after a change, so pages that show dates can reload them. */
  onChanged?: () => void;
}) {
  const [details, setDetails] = useState<ProjectDetails | null>(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);
  const [added, setAdded] = useState<string[]>([]);
  const [newName, setNewName] = useState("");
  const [saving, setSaving] = useState(false);
  const [cleaning, setCleaning] = useState<CleaningPreview | null>(null);

  const reload = useCallback(async () => {
    try {
      setDetails(await loadDetails(projectId));
      setError("");
    } catch (caught) {
      setError(errorText(caught));
    }
  }, [projectId]);

  useEffect(() => {
    setDetails(null);
    setAdded([]);
    void reload();
  }, [reload, documents.length]);

  // What text cleaning would take out: read once per project, never changed
  // by the reading (plan 5.2's offer on the Corpus page).
  useEffect(() => {
    let alive = true;
    setCleaning(null);
    loadCleaningPreview(projectId)
      .then((found) => alive && setCleaning(found))
      .catch(() => alive && setCleaning(null));
    return () => {
      alive = false;
    };
  }, [projectId, documents.length]);

  const columns = useMemo(() => {
    const names = (details?.names ?? []).map((item) => item.name);
    const extra = added.filter(
      (name) =>
        !names.some((known) => known.toLowerCase() === name.toLowerCase()),
    );
    return [...names, ...extra];
  }, [details, added]);

  const rows = useMemo(() => {
    const wanted = query.trim().toLowerCase();
    return documents.filter((doc) => {
      if (!wanted) return true;
      const values = Object.values(details?.documents[doc.id] ?? {}).map(
        (item) => item.value,
      );
      return [doc.name, ...values].some((text) =>
        text.toLowerCase().includes(wanted),
      );
    });
  }, [documents, details, query]);

  if (!details)
    return (
      <section className="panel corpus-details" aria-label="Document details">
        <h2>Document details</h2>
        <p className="muted">{error || "Reading the documents' details…"}</p>
      </section>
    );

  const change = async (work: () => Promise<unknown>) => {
    setSaving(true);
    try {
      await work();
      await reload();
      onChanged?.();
    } catch (caught) {
      setError(errorText(caught));
      await reload();
    } finally {
      setSaving(false);
    }
  };

  const saveCell = (documentId: string, name: string, value: string) => {
    const current = details.documents[documentId]?.[name];
    if ((current?.value ?? "") === value.trim()) return;
    void change(() =>
      setDetail(projectId, documentId, name, value, details.revision),
    );
  };

  const chooseAxis = (
    axis: AxisChoice,
    orderLabel = details.settings.order_label,
  ) =>
    void change(() =>
      saveDetailSettings(
        projectId,
        { ...details.settings, axis, order_label: orderLabel || "Document" },
        details.revision,
      ),
    );

  const chooseCleaning = (stage_directions: boolean) =>
    void change(() =>
      saveDetailSettings(
        projectId,
        {
          ...details.settings,
          text_cleaning: {
            stage_directions,
            extra_terms: details.settings.text_cleaning?.extra_terms ?? [],
          },
        },
        details.revision,
      ),
    );

  const chooseEvents = (events: ProjectEvent[]) =>
    void change(() => saveDetailSettings(projectId, { ...details.settings, events }, details.revision));

  const shown = showAll ? rows : rows.slice(0, FIRST_ROWS);
  const template = details.template;

  return (
    <section className="panel corpus-details" aria-label="Document details">
      <div className="section-title">
        <h2>
          <Tags size={18} /> Document details
        </h2>
      </div>
      <p className="corpus-shape" role="status">
        {shapeLine(details, documents.length)}
      </p>
      {error && (
        <div className="alert error" role="alert">
          <span>{error}</span>
          <button aria-label="Dismiss error" onClick={() => setError("")}>
            <X size={16} />
          </button>
        </div>
      )}
      <div className="corpus-axis">
        <label className="field-label">
          Line the documents up by
          <select
            value={details.settings.axis}
            disabled={saving}
            aria-label="Line the documents up by"
            onChange={(e) => chooseAxis(e.target.value as AxisChoice)}
          >
            <option value="auto">
              Choose for me (now:{" "}
              {details.axis.kind === "none" ? "nothing" : details.axis.kind})
            </option>
            <option value="time">Time (the Date detail)</option>
            <option value="order">
              Order (the Order detail: chapters, sessions)
            </option>
            <option value="none">
              Nothing: the documents are not a sequence
            </option>
          </select>
          <small>
            Figures that show change use this: &ldquo;over time&rdquo; for dated
            documents, &ldquo;across the chapters&rdquo; for a book.
          </small>
        </label>
        {(details.settings.axis === "order" ||
          details.axis.kind === "order") && (
          <label className="field-label">
            One step is called
            <input
              defaultValue={details.settings.order_label}
              aria-label="What one step along the order is called"
              placeholder="Chapter"
              maxLength={40}
              onBlur={(e) => {
                const label = e.target.value.trim();
                if (label && label !== details.settings.order_label)
                  chooseAxis(details.settings.axis, label);
              }}
            />
            <small>
              Chapter, Session, Episode… Figures say &ldquo;across the
              chapters&rdquo;.
            </small>
          </label>
        )}
      </div>
      <p className="muted corpus-template">
        {template
          ? `Read from the file names: ${template.describe}.`
          : details.settings.filename_detection
            ? "The file names share no pattern the suite can read details from."
            : "Reading details from file names is off."}{" "}
        <label className="inline-check">
          <input
            type="checkbox"
            checked={details.settings.filename_detection}
            disabled={saving}
            onChange={(e) =>
              void change(() =>
                saveDetailSettings(
                  projectId,
                  { ...details.settings, filename_detection: e.target.checked },
                  details.revision,
                ),
              )
            }
          />
          Read details from file names
        </label>
      </p>
      {template && template.misses.length > 0 && (
        <p className="muted corpus-template">
          Not matching the pattern: {template.misses.slice(0, 5).join(", ")}
          {template.misses.length > 5
            ? ` and ${template.misses.length - 5} more`
            : ""}
          . Type their details below.
        </p>
      )}
      <div className="corpus-cleaning">
        <h3>Stage directions</h3>
        <label className="inline-check">
          <input
            type="checkbox"
            checked={details.settings.text_cleaning?.stage_directions ?? false}
            disabled={saving}
            onChange={(event) => chooseCleaning(event.target.checked)}
          />
          Leave bracketed stage directions out of analyses
        </label>
        <p className="muted">
          {cleaning === null
            ? "Counting this corpus's bracketed stage directions…"
            : cleaning.found === 0
              ? "No bracketed stage directions were found in this corpus."
              : `Found ${cleaning.found.toLocaleString()} bracketed stage directions like ${cleaning.samples.join(", ")}; ${
                  details.settings.text_cleaning?.stage_directions
                    ? "they are left out of every analysis. The imported files are unchanged."
                    : "leave them out of analyses?"
                }`}
        </p>
      </div>
      <EventsEditor
        events={details.settings.events ?? []}
        disabled={saving}
        axisKind={details.axis.kind}
        onChange={chooseEvents}
      />
      <div className="corpus-details-tools">
        <label className="search-box">
          <input
            placeholder="Find a document or value…"
            aria-label="Find a document or detail value"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <form
          className="add-detail"
          onSubmit={(e) => {
            e.preventDefault();
            const name = newName.trim();
            if (!name) return;
            setAdded((current) => [...current, name]);
            setNewName("");
          }}
        >
          <input
            value={newName}
            aria-label="New detail name"
            placeholder="Party, Genre, Session…"
            maxLength={40}
            onChange={(e) => setNewName(e.target.value)}
          />
          <button
            className="secondary"
            type="submit"
            disabled={!newName.trim()}
          >
            <Plus size={14} /> Add a detail
          </button>
        </form>
        <SplitBook
          projectId={projectId}
          documents={documents}
          disabled={saving}
          onSplit={() => {
            void reload();
            onChanged?.();
          }}
        />
        <DetailImport
          projectId={projectId}
          revision={details.revision}
          existing={details.names.map((item) => item.name)}
          disabled={saving}
          onImported={() => {
            void reload();
            onChanged?.();
          }}
        />
      </div>
      <div className="table-scroll">
        <table className="document-table details-table">
          <thead>
            <tr>
              <th>DOCUMENT</th>
              {columns.map((name) => {
                const typed = Object.values(details.documents).some(
                  (doc) => doc[name] && doc[name].source !== "filename",
                );
                return (
                  <th key={name}>
                    {name}
                    {typed && (
                      <button
                        className="link-button"
                        aria-label={`Remove the values of ${name} that were typed or imported`}
                        title="Remove typed and imported values (file-name values stay)"
                        disabled={saving}
                        onClick={() => {
                          if (
                            window.confirm(
                              `Remove every typed or imported value of “${name}”?`,
                            )
                          )
                            void change(() =>
                              removeTypedDetail(projectId, name),
                            );
                        }}
                      >
                        <X size={12} />
                      </button>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {shown.map((doc) => (
              <tr key={doc.id}>
                <td className="details-name">{doc.name}</td>
                {columns.map((name) => {
                  const item = details.documents[doc.id]?.[name];
                  return (
                    <td key={name}>
                      <input
                        key={`${details.revision}:${item?.value ?? ""}`}
                        className={
                          item
                            ? `detail-cell source-${item.source}`
                            : "detail-cell"
                        }
                        defaultValue={item?.value ?? ""}
                        aria-label={`${name} of ${doc.name}`}
                        title={sourceNote(item)}
                        disabled={saving}
                        maxLength={500}
                        onBlur={(e) => saveCell(doc.id, name, e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") e.currentTarget.blur();
                        }}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > shown.length && (
        <button className="text-button" onClick={() => setShowAll(true)}>
          Show all {rows.length.toLocaleString()} documents
        </button>
      )}
      <p className="muted details-legend">
        {(["filename", "csv", "user"] as const).map((source) => (
          <span key={source} className={`detail-legend source-${source}`}>
            {SOURCE_WORDS[source]}
          </span>
        ))}
        <span>Emptying a typed value brings back the file name&rsquo;s.</span>
      </p>
    </section>
  );
}

/**
 * Dated events for the project's time figures (SHOWCASE_FIGURES_PLAN section 4).
 *
 * "The world on the axis": a stopline at Pearl Harbor or 9/11 turns a trend
 * into a test the reader can check. Events are saved with the project's other
 * settings and frozen into each run, so a published figure draws the ones the
 * reader set when it was submitted.
 */
function EventsEditor({
  events,
  disabled,
  axisKind,
  onChange,
}: {
  events: ProjectEvent[];
  disabled: boolean;
  axisKind: "time" | "order" | "none";
  onChange: (events: ProjectEvent[]) => void;
}) {
  const [name, setName] = useState("");
  const [date, setDate] = useState("");

  const add = () => {
    const trimmed = name.trim();
    const when = date.trim();
    if (!trimmed || !when) return;
    if (events.some((event) => event.name.toLowerCase() === trimmed.toLowerCase()))
      return;
    onChange([...events, { name: trimmed, date: when, note: "" }]);
    setName("");
    setDate("");
  };

  return (
    <div className="corpus-events">
      <h3>Events</h3>
      <p className="muted">
        {axisKind === "time"
          ? "Dated events are drawn as stoplines on every figure over time, so a change can be checked against what happened. Use a year (1941), a decimal year (1941.5) or a date (1941-12-07)."
          : "This corpus is not lined up by time, so events will not appear on its figures yet. Set the axis to time above to use them."}
      </p>
      {events.length > 0 && (
        <ul className="event-list">
          {events.map((event, index) => (
            <li key={`${event.name}-${event.date}`}>
              <span className="event-when">{event.date}</span>
              <span className="event-name">{event.name}</span>
              <button
                className="icon-button"
                aria-label={`Remove event ${event.name}`}
                disabled={disabled}
                onClick={() => onChange(events.filter((_, i) => i !== index))}
              >
                <X size={14} />
              </button>
            </li>
          ))}
        </ul>
      )}
      <form
        className="add-event"
        onSubmit={(e) => {
          e.preventDefault();
          add();
        }}
      >
        <input
          aria-label="Event date"
          placeholder="1941"
          value={date}
          disabled={disabled}
          onChange={(e) => setDate(e.target.value)}
        />
        <input
          aria-label="Event name"
          placeholder="Pearl Harbor"
          value={name}
          disabled={disabled}
          onChange={(e) => setName(e.target.value)}
        />
        <button type="submit" className="text-button" disabled={disabled || !name.trim() || !date.trim()}>
          <Plus size={14} /> Add event
        </button>
      </form>
    </div>
  );
}
