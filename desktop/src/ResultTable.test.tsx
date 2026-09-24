import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ResultTable, rowsToTsv, sortRows } from "./ResultTable";

const columns = ["Document", "Score", "Note"];

const rows = [
  { Document: "b.txt", Score: "9", Note: "has\ttab" },
  { Document: "a.txt", Score: "10", Note: "two\nlines" },
  { Document: "c.txt", Score: "", Note: "" },
];

describe("sorting rows", () => {
  it("compares numbers numerically, not as text", () => {
    const sorted = sortRows(rows, "Score", "asc");
    expect(sorted.map((row) => row.Score)).toEqual(["9", "10", ""]);
  });

  it("reverses on demand without touching the input", () => {
    const sorted = sortRows(rows, "Document", "desc");
    expect(sorted.map((row) => row.Document)).toEqual([
      "c.txt",
      "b.txt",
      "a.txt",
    ]);
    expect(rows[0].Document).toBe("b.txt");
  });

  it("keeps a stable, deterministic order for mixed values", () => {
    const mixed = [
      { Label: "10", V: "1" },
      { Label: "2", V: "1" },
      { Label: "x", V: "1" },
    ];
    const sorted = sortRows(mixed, "Label", "asc");
    // Numbers first, in numeric order; the non-number after them.
    expect(sorted.map((row) => row.Label)).toEqual(["2", "10", "x"]);
  });
});

describe("copying rows as TSV", () => {
  it("emits the header then one line per row", () => {
    const text = rowsToTsv(
      ["A", "B"],
      [
        { A: "1", B: "2" },
        { A: "3", B: "4" },
      ],
    );
    expect(text).toBe("A\tB\n1\t2\n3\t4");
  });

  it("flattens tabs and line breaks so a pasted row stays one row", () => {
    const [header, line] = rowsToTsv(columns, [rows[0]]).split("\n");
    expect(header).toBe("Document\tScore\tNote");
    expect(line).toBe("b.txt\t9\thas tab");
    expect(rowsToTsv(columns, [rows[1]]).split("\n")[1]).toBe(
      "a.txt\t10\ttwo lines",
    );
  });

  it("treats a missing cell as empty rather than the word undefined", () => {
    const [, line] = rowsToTsv(columns, [{ Document: "only" }]).split("\n");
    expect(line).toBe("only\t\t");
  });
});

describe("rendering a result table", () => {
  const draw = (props: Partial<Parameters<typeof ResultTable>[0]> = {}) =>
    renderToStaticMarkup(
      <ResultTable columns={columns} rows={rows} {...props} />,
    );

  it("renders every column and row", () => {
    const html = draw();
    for (const column of columns) expect(html).toContain(column);
    for (const row of rows) expect(html).toContain(row.Document);
  });

  it("marks numeric columns for right alignment", () => {
    expect(draw()).toContain("numeric");
  });

  it("offers the copy button only when asked", () => {
    expect(draw({ copyable: true })).toContain("Copy rows");
    expect(draw()).not.toContain("Copy rows");
  });

  it("sort headers are buttons with an accessible sort state once active", () => {
    const html = draw();
    expect(html).toContain('class="column-sort');
    // Nothing is sorted yet, so no aria-sort is claimed.
    expect(html).not.toContain("aria-sort");
  });
});
