import { useRef, useState } from "react";
import { FileSpreadsheet, X } from "lucide-react";
import {
  fileToBase64,
  importDetails,
  matchWords,
  type ImportPreview,
} from "./details";

const errorText = (caught: unknown) =>
  caught instanceof Error ? caught.message : String(caught);

/** The menu value that leaves a column out. */
const LEAVE_OUT = "";

/** What a column can become: its own name, the two axis details, or a detail
 *  the project already has -- each once, whatever its case. */
export function targetOptions(own: string, existing: string[]): string[] {
  const seen = new Set<string>();
  return [own, "Date", "Order", ...existing].filter((name) => {
    const key = name.trim().toLowerCase();
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/**
 * "Import details from a spreadsheet" (docs/PLAN_0.5.0.md 1.4.2). A sheet
 * the researcher already keeps -- one row per speech, or one per president --
 * is previewed first: which documents its rows reach, what each column
 * becomes, which rows matched nothing. Only "Import" stores anything.
 */
export function DetailImport({
  projectId,
  revision,
  existing,
  onImported,
  disabled = false,
}: {
  projectId: string;
  /** The details revision the preview was made against. */
  revision: number;
  /** The project's detail names, offered as targets. */
  existing: string[];
  onImported: () => void;
  disabled?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<{
    filename: string;
    content: string;
  } | null>(null);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  // Each column's name as the sheet proposed it, so "keep as" survives a
  // change of mind; and the choices made since.
  const [proposed, setProposed] = useState<Record<string, string>>({});
  const [targets, setTargets] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState("");

  const ask = async (
    chosen: { filename: string; content: string },
    wanted: Record<string, string>,
    apply: boolean,
  ) => {
    setBusy(true);
    setError("");
    try {
      const answer = await importDetails(projectId, {
        ...chosen,
        targets: Object.keys(wanted).length ? wanted : null,
        dry_run: !apply,
        expected_revision: apply ? revision : null,
      });
      if (answer.applied) {
        setDone(
          `Imported ${answer.columns
            .filter((column) => column.target)
            .map((column) => column.target)
            .join(", ")} for ${matchWords(answer)}.`,
        );
        close();
        onImported();
      } else setPreview(answer);
      return answer;
    } catch (caught) {
      setError(errorText(caught));
      return null;
    } finally {
      setBusy(false);
    }
  };

  const close = () => {
    setFile(null);
    setPreview(null);
    setTargets({});
    setProposed({});
    if (input.current) input.current.value = "";
  };

  const choose = async (picked: File | undefined) => {
    if (!picked) return;
    setDone("");
    const chosen = {
      filename: picked.name,
      content: await fileToBase64(picked),
    };
    setFile(chosen);
    setTargets({});
    const first = await ask(chosen, {}, false);
    if (first)
      setProposed(
        Object.fromEntries(
          first.columns.map((column) => [column.column, column.target]),
        ),
      );
  };

  const retarget = (column: string, target: string) => {
    if (!file) return;
    const next = { ...targets, [column]: target };
    setTargets(next);
    void ask(file, next, false);
  };

  const blocked =
    !preview ||
    preview.matched === 0 ||
    preview.columns.every((column) => !column.target) ||
    preview.columns.some((column) => column.target && column.problem);

  return (
    <div className="detail-import">
      <input
        ref={input}
        type="file"
        accept=".csv,.tsv,.txt,.xlsx"
        hidden
        aria-label="Spreadsheet of document details"
        onChange={(e) => void choose(e.target.files?.[0])}
      />
      <button
        className="secondary"
        type="button"
        disabled={disabled || busy}
        onClick={() => input.current?.click()}
      >
        <FileSpreadsheet size={14} /> Import from a spreadsheet
      </button>
      {done && (
        <p className="muted" role="status">
          {done}
        </p>
      )}
      {error && (
        <div className="alert error" role="alert">
          <span>{error}</span>
          <button aria-label="Dismiss error" onClick={() => setError("")}>
            <X size={16} />
          </button>
        </div>
      )}
      {preview && file && (
        <section
          className="detail-import-preview"
          aria-label="Spreadsheet preview"
        >
          <h3>{file.filename}</h3>
          <p>
            Rows reach <strong>{matchWords(preview)}</strong>, read from the
            column &ldquo;{preview.name_column}&rdquo;.
            {preview.unmatched_document_count > 0 &&
              ` ${preview.unmatched_document_count} document(s) get no row and keep their details as they are.`}
          </p>
          {preview.unmatched_row_count > 0 && (
            <p className="muted">
              {preview.unmatched_row_count} row(s) name nothing in this project
              and are left out: {preview.unmatched_rows.slice(0, 8).join(", ")}
              {preview.unmatched_row_count > 8 ? "…" : ""}
            </p>
          )}
          {preview.diagnostics
            .filter((d) => d.code !== "METADATA_UNMATCHED_ROWS")
            .map((d) => (
              <p key={d.code} className="alert warning">
                {d.message}
              </p>
            ))}
          <table className="document-table detail-import-columns">
            <thead>
              <tr>
                <th>COLUMN</th>
                <th>BECOMES</th>
                <th>LOOKS LIKE</th>
              </tr>
            </thead>
            <tbody>
              {preview.columns.map((column) => {
                const own = proposed[column.column] ?? column.target;
                const options = targetOptions(own, existing);
                return (
                  <tr key={column.column}>
                    <td>{column.column}</td>
                    <td>
                      <select
                        aria-label={`What ${column.column} becomes`}
                        value={column.target}
                        disabled={busy}
                        onChange={(e) =>
                          retarget(column.column, e.target.value)
                        }
                      >
                        {options.map((name) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                        <option value={LEAVE_OUT}>Leave out</option>
                      </select>
                      {column.target && column.problem && (
                        <small className="detail-import-problem">
                          {column.problem}
                        </small>
                      )}
                    </td>
                    <td className="muted">
                      {column.kind} · {column.samples.slice(0, 3).join(", ")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {preview.preview.length > 0 && (
            <details>
              <summary>
                First {preview.preview.length} documents as imported
              </summary>
              <ul className="detail-import-rows">
                {preview.preview.map((row) => (
                  <li key={row.document}>
                    <strong>{row.document}</strong>:{" "}
                    {Object.entries(row.values)
                      .map(([name, value]) => `${name} ${value}`)
                      .join(" · ") || "no values"}
                  </li>
                ))}
              </ul>
            </details>
          )}
          <div className="dialog-actions">
            <button
              type="button"
              className="primary"
              disabled={busy || blocked}
              onClick={() => void ask(file, targets, true)}
            >
              Import details for {preview.matched} document
              {preview.matched === 1 ? "" : "s"}
            </button>
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={close}
            >
              Cancel
            </button>
          </div>
          <p className="muted">
            Imported values sit under anything typed here, and importing a
            corrected sheet replaces this one.
          </p>
        </section>
      )}
    </div>
  );
}
