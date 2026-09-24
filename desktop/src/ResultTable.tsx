import { useMemo, useState, type ReactNode } from "react";
import { ArrowDown, ArrowUp, Check, Copy } from "lucide-react";
import { isNumericColumn, numberOf } from "./chartLayout";

/**
 * One result table, everywhere a result table appears.
 *
 * The runs pane and the workbench's drill-down each grew their own copy of
 * this markup, and each copy answered a different subset of the same three
 * questions a reader asks of any table: which column decides the order, is
 * this a number or a name, and how do I get this into my notes. So the table
 * lives here once:
 *
 * * Headers sort on click (numeric columns numerically, text columns by
 *   name), cycling ascending → descending → back to the run's own order.
 *   Sorting is a way of looking at the rows already loaded — it never fetches
 *   more — and the header says so rather than letting a sorted page be
 *   mistaken for a sorted file.
 * * Numeric columns align right, the convention that makes magnitudes
 *   comparable down a column.
 * * "Copy rows" puts the visible rows on the clipboard as tab-separated
 *   text, which pastes straight into a spreadsheet or notebook.
 */

export type Sort = { column: string; direction: "asc" | "desc" } | null;

/**
 * Rows in the requested order. Numbers compare as numbers — "9" sorts before
 * "10" — and anything that is not a number on both sides falls back to a
 * text comparison, so a column of mixed values still has a stable order.
 */
export function sortRows(
  rows: Record<string, string>[],
  column: string,
  direction: "asc" | "desc",
): Record<string, string>[] {
  const factor = direction === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const left = a[column] ?? "";
    const right = b[column] ?? "";
    // A blank is a missing value, not the smallest thing there is: it sorts
    // after everything else in both directions, the way spreadsheets treat an
    // empty cell.
    const leftBlank = left.trim() === "";
    const rightBlank = right.trim() === "";
    if (leftBlank && rightBlank) return 0;
    if (leftBlank) return 1;
    if (rightBlank) return -1;
    const leftNumber = numberOf(left);
    const rightNumber = numberOf(right);
    if (leftNumber !== null && rightNumber !== null)
      return (leftNumber - rightNumber) * factor;
    return left.localeCompare(right) * factor;
  });
}

/**
 * The rows as tab-separated text: header line first, one line per row. Tabs
 * and line breaks inside a cell are flattened to spaces — a pasted row must
 * stay one row.
 */
export function rowsToTsv(
  columns: string[],
  rows: Record<string, string>[],
): string {
  const flat = (value: string | undefined) =>
    String(value ?? "")
      .replaceAll("\t", " ")
      .replace(/\r?\n/g, " ");
  const lines = [columns.join("\t")];
  for (const row of rows)
    lines.push(columns.map((c) => flat(row[c])).join("\t"));
  return lines.join("\n");
}

export function ResultTable({
  columns,
  rows,
  copyable = false,
  footerNote,
}: {
  columns: string[];
  rows: Record<string, string>[];
  copyable?: boolean;
  footerNote?: ReactNode;
}) {
  const [sort, setSort] = useState<Sort>(null);
  const [copied, setCopied] = useState(false);

  // Judged on the rows in hand, exactly as the chart's measure list judges
  // them: a column is numeric when everything present in it parses, so a
  // single stray word does not silently flip a whole column's alignment.
  const numeric = useMemo(
    () => new Set(columns.filter((column) => isNumericColumn(rows, column))),
    [columns, rows],
  );

  const visible = useMemo(
    () => (sort ? sortRows(rows, sort.column, sort.direction) : rows),
    [rows, sort],
  );

  const cycle = (column: string) =>
    setSort((current) =>
      !current || current.column !== column
        ? { column, direction: "asc" }
        : current.direction === "asc"
          ? { column, direction: "desc" }
          : // Third click restores the run's own order.
            null,
    );

  const copy = async () => {
    const text = rowsToTsv(columns, visible);
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Non-secure contexts (an edge the desktop webview does not hit, but a
      // dev browser over http might) deny the async clipboard; the selection
      // trick still works there.
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <>
      {copyable && (
        <div className="result-table-tools">
          <button
            type="button"
            className="text-button"
            title="Copy the visible rows to the clipboard as tab-separated text"
            onClick={() => void copy()}
          >
            {copied ? (
              <>
                <Check size={14} aria-hidden="true" /> Copied
              </>
            ) : (
              <>
                <Copy size={14} aria-hidden="true" /> Copy rows
              </>
            )}
          </button>
        </div>
      )}
      <div className="result-table table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((column) => {
                const active = sort?.column === column;
                return (
                  <th
                    key={column}
                    aria-sort={
                      active
                        ? sort.direction === "asc"
                          ? "ascending"
                          : "descending"
                        : undefined
                    }
                  >
                    <button
                      type="button"
                      className={active ? "column-sort active" : "column-sort"}
                      // The honest scope of the sort, before anyone mistakes a
                      // rearranged page for a rearranged file.
                      title={
                        active
                          ? `Sorted ${sort.direction}; sorting covers the rows on this page. Click to reverse, again to restore the run's order.`
                          : "Sort by this column (this page's rows)"
                      }
                      onClick={() => cycle(column)}
                    >
                      {column}
                      {active &&
                        (sort.direction === "asc" ? (
                          <ArrowUp size={12} aria-hidden="true" />
                        ) : (
                          <ArrowDown size={12} aria-hidden="true" />
                        ))}
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {visible.map((row, index) => (
              <tr key={index}>
                {columns.map((column) => (
                  <td
                    key={column}
                    className={numeric.has(column) ? "numeric" : undefined}
                  >
                    {row[column]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {footerNote && <div className="table-footnote">{footerNote}</div>}
    </>
  );
}
