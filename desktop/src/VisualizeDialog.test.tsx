import { describe, expect, it } from "vitest";
import {
  chartBlockedReasons,
  exportFormats,
  resolveSeed,
  EXCEL_CHART_KINDS,
} from "./VisualizeDialog";
import { viewableArtifact } from "./ArtifactViewer";
import type { Table } from "./api";

const table: Table = {
  columns: ["word", "count", "date"],
  rows: [
    { word: "freedom", count: "120", date: "2020-01-01" },
    { word: "liberty", count: "95", date: "2020-01-02" },
    { word: "freedom", count: "5", date: "2020-01-03" },
  ],
  total: 3,
  truncated: false,
};

describe("chart compatibility (mirrors chartspec validation)", () => {
  it("blocks a heatmap without a group column", () => {
    const reasons = chartBlockedReasons(
      "heatmap",
      "word",
      "count",
      "",
      "sum",
      table,
    );
    expect(reasons.join(" ")).toMatch(/needs a Group column/);
  });
  it("allows a heatmap with a group", () => {
    const wide: Table = {
      columns: ["word", "count", "group"],
      rows: [
        { word: "a", count: "1", group: "A" },
        { word: "b", count: "2", group: "B" },
      ],
      total: 2,
      truncated: false,
    };
    expect(
      chartBlockedReasons("heatmap", "word", "count", "group", "sum", wide),
    ).toEqual([]);
  });
  it("blocks pie/waffle/calendar with a group column", () => {
    for (const kind of ["pie", "waffle", "calendar"]) {
      const reasons = chartBlockedReasons(
        kind,
        "word",
        "count",
        "count",
        "sum",
        table,
      );
      expect(reasons.join(" ")).toMatch(/takes no Group column/);
    }
  });
  it("blocks calendar when x is not a date", () => {
    const reasons = chartBlockedReasons(
      "calendar",
      "word",
      "count",
      "",
      "sum",
      table,
    );
    expect(reasons.join(" ")).toMatch(/date column/);
  });
  it("allows calendar on a date column", () => {
    expect(
      chartBlockedReasons("calendar", "date", "count", "", "sum", table),
    ).toEqual([]);
  });
  it("flags duplicate x keys without aggregation", () => {
    const reasons = chartBlockedReasons("bar", "word", "count", "", "", table);
    expect(reasons.join(" ")).toMatch(/duplicate X values/);
    expect(
      chartBlockedReasons("bar", "word", "count", "", "sum", table),
    ).toEqual([]);
  });
  it("requires a numeric y", () => {
    const reasons = chartBlockedReasons(
      "bar",
      "word",
      "word",
      "",
      "sum",
      table,
    );
    expect(reasons.join(" ")).toMatch(/numeric column/);
  });
  it("rejects group for pie but allows distinct-group heatmap", () => {
    const reasons = chartBlockedReasons(
      "pie",
      "word",
      "count",
      "count",
      "sum",
      table,
    );
    expect(reasons.length).toBeGreaterThan(0);
  });
  it("blocks negative slices for pie/waffle/hierarchy kinds", () => {
    const negatives: Table = {
      columns: ["word", "delta"],
      rows: [
        { word: "a", delta: "-5" },
        { word: "b", delta: "3" },
      ],
      total: 2,
      truncated: false,
    };
    expect(
      chartBlockedReasons("pie", "word", "delta", "", "sum", negatives).join(
        " ",
      ),
    ).toMatch(/non-negative/);
  });
  it("blocks x === y", () => {
    const reasons = chartBlockedReasons(
      "bar",
      "word",
      "word",
      "",
      "sum",
      table,
    );
    expect(reasons.join(" ")).toMatch(/different columns/);
  });
});

describe("viewableArtifact", () => {
  it("accepts chart/html/png/pdf/svg/txt", () => {
    for (const p of ["chart.html", "wordcloud.png", "a.pdf", "b.svg", "c.txt"])
      expect(viewableArtifact(p)).toBe(true);
  });
  it("rejects csv/gexf/docx", () => {
    for (const p of ["table.csv", "graph.gexf", "a.docx"])
      expect(viewableArtifact(p)).toBe(false);
  });
});
describe("a recommended chart seeds the dialog", () => {
  const columns = ["Term", "TF-IDF", "Document Frequency"];

  it("opens on the recommended kind and axes", () => {
    const seed = { kind: "scatter", x: "Document Frequency", y: "TF-IDF" };
    expect(resolveSeed(seed, columns, "Term", "TF-IDF")).toEqual({
      ...seed,
      format: "html",
    });
  });

  it("falls back to the defaults with no recommendation", () => {
    expect(resolveSeed(null, columns, "Term", "TF-IDF")).toEqual({
      kind: "bar",
      x: "Term",
      y: "TF-IDF",
      format: "html",
    });
  });

  it("ignores an axis this artifact does not have", () => {
    const seed = { kind: "bar", x: "Gries DP", y: "TF-IDF" };
    expect(resolveSeed(seed, columns, "Term", "TF-IDF")).toEqual({
      kind: "bar",
      x: "Term",
      y: "TF-IDF",
      format: "html",
    });
  });

  it("ignores a kind this dialog cannot draw", () => {
    const seed = { kind: "dispersion", x: "Term", y: "TF-IDF" };
    expect(resolveSeed(seed, columns, "Term", "TF-IDF").kind).toBe("bar");
  });

  it("opens on the format the workbench asked for", () => {
    // The handoff buttons say what comes out -- "Excel workbook" -- so the
    // dialog has already been told, and asking again in a dropdown the reader
    // has to find is asking twice.
    const seed = { kind: "bar", x: "Term", y: "TF-IDF", format: "xlsx" };
    expect(resolveSeed(seed, columns, "Term", "TF-IDF").format).toBe("xlsx");
  });

  it("defaults to the format that needs nothing installed", () => {
    // Static images need Kaleido and workbooks need openpyxl; interactive HTML
    // is the one that always works, so an unasked-for format is that one.
    const seed = { kind: "bar", x: "Term", y: "TF-IDF" };
    expect(resolveSeed(seed, columns, "Term", "TF-IDF").format).toBe("html");
  });
});

describe("legacy chart kinds", () => {
  it("names exactly the kinds the original suite's Excel export covers", () => {
    expect([...EXCEL_CHART_KINDS].sort()).toEqual([
      "bar",
      "bubble",
      "line",
      "pie",
      "radar",
      "scatter",
    ]);
  });

  it("accepts bubble as a drawable kind", () => {
    const columns = ["Word", "Count"];
    expect(
      resolveSeed(
        { kind: "bubble", x: "Word", y: "Count" },
        columns,
        "Word",
        "Count",
      ).kind,
    ).toBe("bubble");
  });

  it("refuses an aggregation on bubble, as the engine does", () => {
    const table = {
      columns: ["word", "count"],
      rows: [
        { word: "a", count: "1" },
        { word: "b", count: "2" },
      ],
      total: 2,
      truncated: false,
    };
    // Mirrors chartspec: bubble draws every observation, so --agg is invalid.
    expect(
      chartBlockedReasons("bubble", "word", "count", "", "", table),
    ).toEqual([]);
  });
});

describe("export formats the machine can produce", () => {
  const choices = ["html", "png", "svg", "pdf", "xlsx"];
  const present = (name: string, present: boolean) => ({
    name,
    label: name,
    purpose: "",
    present,
  });

  it("offers every format when Kaleido is installed", () => {
    const formats = exportFormats(choices, [present("kaleido", true)]);
    expect(formats.every((f) => f.available)).toBe(true);
    expect(formats.map((f) => f.label)).toEqual(choices);
  });

  it("blocks only the static image formats when Kaleido is missing", () => {
    const formats = exportFormats(choices, [present("kaleido", false)]);
    const blocked = formats.filter((f) => !f.available).map((f) => f.value);
    expect(blocked).toEqual(["png", "svg", "pdf"]);
  });

  it("says why a format cannot be chosen rather than hiding it", () => {
    const png = exportFormats(choices, [present("kaleido", false)]).find(
      (f) => f.value === "png",
    );
    expect(png?.label).toContain("Kaleido");
  });

  it("leaves HTML and Excel alone, which do not use Kaleido", () => {
    const formats = exportFormats(choices, [present("kaleido", false)]);
    for (const value of ["html", "xlsx"])
      expect(formats.find((f) => f.value === value)?.available).toBe(true);
  });

  it("assumes available before the environment has been read", () => {
    // An empty list means "not checked yet", not "nothing installed".
    expect(exportFormats(choices, []).every((f) => f.available)).toBe(true);
  });
});

/**
 * The engine refuses `--agg` outright for the kinds that draw every row as it
 * stands. This dialog defaulted the aggregation to "sum" and sent it whatever
 * the kind, so five of the fourteen kinds failed after the job had run —
 * including `bubble`, which had only just become reachable at all.
 *
 * `AGG_REJECTED` was declared for exactly this and never consulted; the
 * condition standing in its place listed scatter, box and violin inline and
 * had already fallen a kind behind.
 */
describe("kinds that take no aggregation", () => {
  const table = {
    columns: ["Word", "Count"],
    rows: [
      { Word: "a", Count: "4" },
      { Word: "a", Count: "3" },
      { Word: "b", Count: "2" },
    ],
    total: 3,
    truncated: false,
  };

  for (const kind of ["scatter", "bubble", "box", "violin", "histogram"]) {
    it(`refuses an aggregation on ${kind}, as the engine does`, () => {
      const reasons = chartBlockedReasons(
        kind,
        "Word",
        "Count",
        "",
        "sum",
        table,
      );
      expect(reasons.join(" ")).toMatch(/takes no aggregation/);
    });

    it(`allows ${kind} once the aggregation is cleared`, () => {
      const reasons = chartBlockedReasons(kind, "Word", "Count", "", "", table);
      expect(reasons.join(" ")).not.toMatch(/aggregation/);
    });
  }

  it("still asks for an aggregation where one is needed", () => {
    // Duplicate X values in a bar chart are genuinely ambiguous.
    const reasons = chartBlockedReasons("bar", "Word", "Count", "", "", table);
    expect(reasons.join(" ")).toMatch(/duplicate X values/);
  });

  it("accepts a bar chart that has been given one", () => {
    expect(
      chartBlockedReasons("bar", "Word", "Count", "", "sum", table),
    ).toEqual([]);
  });
});
