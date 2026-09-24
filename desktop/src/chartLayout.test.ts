import { describe, expect, it } from "vitest";
import {
  LIVE_KINDS,
  buildLayout,
  categoricalColumns,
  distinctCount,
  formatNumber,
  isNumericColumn,
  isConstant,
  looksLikeIdentifier,
  nameSaysIdentifier,
  measureColumns,
  median,
  niceTicks,
  numberOf,
  numericColumns,
  publishParams,
  quantile,
  reduce,
  rowsBehind,
  type Settings,
} from "./chartLayout";

/**
 * The workbench draws from these numbers, so these are the numbers to check.
 * A chart that is merely plausible is worse than no chart: it is a wrong
 * reading that looks like a finding.
 */

const settings = (over: Partial<Settings> = {}): Settings => ({
  kind: "bar",
  x: "Word",
  y: "Count",
  group: "",
  agg: "sum",
  topN: 0,
  sort: "table",
  bins: 4,
  ...over,
});

const rows = [
  { Word: "a", Count: "4", Set: "x" },
  { Word: "a", Count: "3", Set: "y" },
  { Word: "b", Count: "2", Set: "x" },
  { Word: "c", Count: "1", Set: "y" },
];

describe("reading numbers out of a table", () => {
  it("treats a blank cell as missing, never as zero", () => {
    expect(numberOf("")).toBeNull();
    expect(numberOf("   ")).toBeNull();
    expect(numberOf(undefined)).toBeNull();
    expect(numberOf("0")).toBe(0);
  });
  it("refuses text that is not a number", () => {
    expect(numberOf("twelve")).toBeNull();
    expect(numberOf("12abc")).toBeNull();
  });
  it("accepts the shapes a CSV actually holds", () => {
    expect(numberOf("-3.5")).toBe(-3.5);
    expect(numberOf("1e3")).toBe(1000);
    expect(numberOf(" 42 ")).toBe(42);
  });
  it("does not call an all-blank column numeric", () => {
    expect(isNumericColumn([{ a: "" }, { a: "" }], "a")).toBe(false);
  });
  it("allows gaps in an otherwise numeric column", () => {
    expect(isNumericColumn([{ a: "1" }, { a: "" }, { a: "3" }], "a")).toBe(
      true,
    );
  });
  it("finds the measures a table offers", () => {
    expect(numericColumns({ columns: ["Word", "Count", "Set"], rows })).toEqual(
      ["Count"],
    );
  });
  it("offers a column with few values as a category", () => {
    expect(
      categoricalColumns({ columns: ["Word", "Count", "Set"], rows }),
    ).toContain("Set");
  });
  it("stops counting distinct values once the cap is passed", () => {
    const many = Array.from({ length: 300 }, (_, i) => ({ a: String(i) }));
    expect(distinctCount(many, "a", 50)).toBe(51);
  });
});

describe("summarising", () => {
  it("adds, averages and takes the middle", () => {
    expect(reduce([1, 2, 3, 4], "sum")).toBe(10);
    expect(reduce([1, 2, 3, 4], "mean")).toBe(2.5);
    expect(reduce([1, 2, 3, 4], "median")).toBe(2.5);
    expect(reduce([1, 2, 3], "median")).toBe(2);
  });
  it("counts rows rather than adding their values", () => {
    expect(reduce([5, 5, 5], "count")).toBe(3);
    expect(reduce([], "count")).toBe(0);
  });
  it("does not invent a value for nothing", () => {
    expect(reduce([], "mean")).toBe(0);
    expect(median([])).toBe(0);
  });
  it("interpolates quartiles the way numpy does", () => {
    const sorted = [1, 2, 3, 4];
    expect(quantile(sorted, 0.25)).toBeCloseTo(1.75);
    expect(quantile(sorted, 0.75)).toBeCloseTo(3.25);
    expect(quantile(sorted, 0)).toBe(1);
    expect(quantile(sorted, 1)).toBe(4);
  });
});

describe("axis ticks", () => {
  it("lands on round numbers", () => {
    expect(niceTicks(0, 100).map((t) => t.at)).toEqual([
      0, 20, 40, 60, 80, 100,
    ]);
  });
  it("does not drift on fractional steps", () => {
    // Accumulating 0.1 fifty times is famously not 5; ticks must still be round.
    for (const tick of niceTicks(0, 5))
      expect(tick.at).toBeCloseTo(Math.round(tick.at * 100) / 100);
  });
  it("survives a flat range", () => {
    expect(niceTicks(7, 7)).toHaveLength(1);
  });
  it("shortens long numbers rather than overflowing the axis", () => {
    expect(formatNumber(1_500_000)).toBe("1.5M");
    expect(formatNumber(12_345)).toBe("12k");
    expect(formatNumber(42)).toBe("42");
    expect(formatNumber(0.5)).toBe("0.50");
  });
});

describe("bars", () => {
  it("adds up the rows behind each category", () => {
    const result = buildLayout(rows, settings());
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const marks = result.layout.series[0].marks;
    expect(marks.map((m) => [m.label, m.y])).toEqual([
      ["a", 7],
      ["b", 2],
      ["c", 1],
    ]);
  });

  it("says how many rows each bar stands for", () => {
    const result = buildLayout(rows, settings());
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.series[0].marks[0].n).toBe(2);
  });

  it("averages instead of adding when asked", () => {
    const result = buildLayout(rows, settings({ agg: "mean" }));
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.series[0].marks[0].y).toBe(3.5);
  });

  it("counts rows when the measure is not the point", () => {
    const result = buildLayout(rows, settings({ agg: "count" }));
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.series[0].marks[0].y).toBe(2);
  });

  it("splits into one series per group, each with its own colour", () => {
    const result = buildLayout(rows, settings({ group: "Set" }));
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.series.map((s) => s.group)).toEqual(["x", "y"]);
    expect(result.layout.series[0].color).not.toBe(
      result.layout.series[1].color,
    );
  });

  it("is read against zero", () => {
    const result = buildLayout(rows, settings());
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.y.min).toBe(0);
  });

  it("keeps negative values visible below the line", () => {
    const result = buildLayout([{ Word: "a", Count: "-5" }], settings());
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.y.min).toBe(-5);
    expect(result.layout.y.max).toBe(0);
  });

  it("orders by value when asked, and leaves table order alone otherwise", () => {
    const shuffled = [
      { Word: "small", Count: "1" },
      { Word: "big", Count: "9" },
    ];
    const asIs = buildLayout(shuffled, settings());
    const ranked = buildLayout(shuffled, settings({ sort: "high" }));
    if (!asIs.ok || !ranked.ok) throw new Error("layout failed");
    expect(asIs.layout.x.categories).toEqual(["small", "big"]);
    expect(ranked.layout.x.categories).toEqual(["big", "small"]);
  });

  it("keeps the largest categories under a limit, not the first ones", () => {
    const many = [
      { Word: "tiny", Count: "1" },
      { Word: "huge", Count: "99" },
      { Word: "mid", Count: "50" },
    ];
    const result = buildLayout(many, settings({ topN: 2 }));
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.x.categories).toEqual(["huge", "mid"]);
    expect(result.layout.hidden).toBe(1);
  });

  it("reports rows it could not use rather than dropping them quietly", () => {
    const result = buildLayout([...rows, { Word: "d", Count: "" }], settings());
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.dropped).toBe(1);
    expect(result.layout.used).toBe(4);
  });

  it("still counts a row with no measure when counting rows", () => {
    const result = buildLayout(
      [{ Word: "a", Count: "" }],
      settings({ agg: "count" }),
    );
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.dropped).toBe(0);
    expect(result.layout.series[0].marks[0].y).toBe(1);
  });
});

describe("points", () => {
  const numeric = [
    { A: "1", B: "10", Set: "x" },
    { A: "2", B: "20", Set: "y" },
  ];

  it("draws one point per row, not one per category", () => {
    const result = buildLayout(
      numeric,
      settings({ kind: "scatter", x: "A", y: "B" }),
    );
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.series[0].marks).toHaveLength(2);
    expect(result.layout.series[0].marks[0].n).toBe(1);
  });

  it("needs both axes to be numbers, and says which is not", () => {
    const result = buildLayout(
      rows,
      settings({ kind: "scatter", x: "Word", y: "Count" }),
    );
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.reason).toContain("Word");
  });

  it("sizes bubbles by magnitude, relative to the largest", () => {
    const result = buildLayout(
      numeric,
      settings({ kind: "bubble", x: "A", y: "B" }),
    );
    if (!result.ok) throw new Error(result.reason);
    const sizes = result.layout.series[0].marks.map((m) => m.size);
    expect(sizes[1]).toBe(1);
    expect(sizes[0]).toBeCloseTo(0.5);
  });

  it("leaves scatter marks unsized", () => {
    const result = buildLayout(
      numeric,
      settings({ kind: "scatter", x: "A", y: "B" }),
    );
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.series[0].marks[0].size).toBeUndefined();
  });

  it("does not force a point chart to include zero", () => {
    const far = [
      { A: "100", B: "500" },
      { A: "200", B: "600" },
    ];
    const result = buildLayout(
      far,
      settings({ kind: "scatter", x: "A", y: "B" }),
    );
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.y.min).toBe(500);
  });
});

describe("histogram", () => {
  const spread = Array.from({ length: 10 }, (_, i) => ({
    Word: "w",
    Count: String(i),
  }));

  it("puts every row in exactly one bin", () => {
    const result = buildLayout(
      spread,
      settings({ kind: "histogram", bins: 5 }),
    );
    if (!result.ok) throw new Error(result.reason);
    const total = result.layout.series[0].marks.reduce(
      (sum, m) => sum + m.y,
      0,
    );
    expect(total).toBe(spread.length);
  });

  it("includes the largest value rather than falling off the end", () => {
    const result = buildLayout(
      spread,
      settings({ kind: "histogram", bins: 5 }),
    );
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.series[0].marks.at(-1)?.y).toBeGreaterThan(0);
  });

  it("reports empty bins as zero rather than omitting them", () => {
    const gapped = [{ Count: "0" }, { Count: "10" }];
    const result = buildLayout(
      gapped,
      settings({ kind: "histogram", bins: 5, x: "Count" }),
    );
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.series[0].marks).toHaveLength(5);
    expect(
      result.layout.series[0].marks.filter((m) => m.y === 0).length,
    ).toBeGreaterThan(0);
  });

  it("uses one set of bin edges across groups so they can be compared", () => {
    const grouped = [
      { Count: "0", Set: "x" },
      { Count: "10", Set: "y" },
    ];
    const result = buildLayout(
      grouped,
      settings({ kind: "histogram", group: "Set", bins: 4 }),
    );
    if (!result.ok) throw new Error(result.reason);
    const [first, second] = result.layout.series;
    expect(first.marks.map((m) => m.label)).toEqual(
      second.marks.map((m) => m.label),
    );
  });
});

describe("box plot", () => {
  const spread = [1, 2, 3, 4, 100].map((n) => ({
    Word: "a",
    Count: String(n),
  }));

  it("reports the middle and the quartiles", () => {
    const result = buildLayout(spread, settings({ kind: "box" }));
    if (!result.ok) throw new Error(result.reason);
    const box = result.layout.boxes[0];
    expect(box.median).toBe(3);
    expect(box.q1).toBe(2);
    expect(box.q3).toBe(4);
    expect(box.n).toBe(5);
  });

  it("keeps an outlier in range rather than clipping it out of sight", () => {
    const result = buildLayout(spread, settings({ kind: "box" }));
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.boxes[0].max).toBe(100);
    expect(result.layout.y.max).toBeGreaterThanOrEqual(100);
  });
});

describe("heatmap", () => {
  it("needs a row axis and says so", () => {
    const result = buildLayout(rows, settings({ kind: "heatmap", group: "" }));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("Group");
  });

  it("makes one cell per row-and-column pair that has data", () => {
    const result = buildLayout(
      rows,
      settings({ kind: "heatmap", group: "Set" }),
    );
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.cells).toHaveLength(4);
    expect(result.layout.rows).toEqual(["x", "y"]);
  });

  it("does not invent a cell where there are no rows", () => {
    const sparse = [
      { Word: "a", Count: "1", Set: "x" },
      { Word: "b", Count: "2", Set: "y" },
    ];
    const result = buildLayout(
      sparse,
      settings({ kind: "heatmap", group: "Set" }),
    );
    if (!result.ok) throw new Error(result.reason);
    expect(result.layout.cells).toHaveLength(2);
  });
});

describe("refusals are sentences, not exceptions", () => {
  it("explains an empty page", () => {
    const result = buildLayout([], settings());
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("no rows");
  });
  it("explains a measure with nothing in it", () => {
    const result = buildLayout([{ Word: "a", Count: "" }], settings());
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("Count");
  });
  it("never throws, whatever the columns hold", () => {
    const nasty = [{ Word: "", Count: "NaN", Set: "" }];
    for (const kind of LIVE_KINDS) {
      expect(() => buildLayout(nasty, settings({ kind }))).not.toThrow();
    }
  });
});

describe("clicking a mark", () => {
  it("finds exactly the rows behind a category", () => {
    expect(
      rowsBehind(rows, settings(), { label: "a", group: "" }),
    ).toHaveLength(2);
  });
  it("narrows to the group as well when one is in use", () => {
    const found = rowsBehind(rows, settings({ group: "Set" }), {
      label: "a",
      group: "x",
    });
    expect(found).toHaveLength(1);
    expect(found[0].Set).toBe("x");
  });
  it("finds nothing for a category that is not there", () => {
    expect(
      rowsBehind(rows, settings(), { label: "zzz", group: "" }),
    ).toHaveLength(0);
  });
});

/**
 * Publishing hands these settings to the engine. A parameter it refuses fails
 * *after* the job has run, so the translation has to respect its rules — this
 * is the same defect that made five chart kinds fail from the Visualize dialog.
 */
describe("publishing to the engine", () => {
  it("sends an aggregation only where the engine takes one", () => {
    expect(publishParams(settings({ kind: "bar" })).agg).toBe("sum");
    expect(publishParams(settings({ kind: "heatmap" })).agg).toBe("sum");
    for (const kind of ["scatter", "bubble", "histogram", "box"] as const) {
      expect(publishParams(settings({ kind })).agg).toBe("");
    }
  });

  it("sends a ranking limit only for the kinds the engine ranks", () => {
    expect(publishParams(settings({ kind: "bar", topN: 20 }))["top-n"]).toBe(
      20,
    );
    for (const kind of [
      "scatter",
      "bubble",
      "histogram",
      "box",
      "heatmap",
    ] as const) {
      expect(publishParams(settings({ kind, topN: 20 }))["top-n"]).toBeNull();
    }
  });

  it("sends bins only for a histogram", () => {
    expect(publishParams(settings({ kind: "histogram", bins: 9 })).bins).toBe(
      9,
    );
    expect(publishParams(settings({ kind: "bar", bins: 9 })).bins).toBeNull();
  });

  it("carries the axes and grouping through unchanged", () => {
    const params = publishParams(
      settings({ x: "Word", y: "Count", group: "Set" }),
    );
    expect(params).toMatchObject({
      x: "Word",
      y: "Count",
      group: "Set",
      kind: "bar",
    });
  });

  it("names every parameter the tool requires", () => {
    const params = publishParams(settings());
    for (const name of [
      "kind",
      "x",
      "y",
      "group",
      "agg",
      "normalize",
      "top-n",
      "bins",
      "title",
      "format",
    ]) {
      expect(params).toHaveProperty(name);
    }
  });
});

/**
 * Found by running the workbench against a real `readability.csv` rather than
 * a fixture: it opened on `Document ID` — the first numeric column of nearly
 * every result this suite produces — so the first chart anyone saw was a
 * staircase of row numbers. `core/insight/profile.py` already refuses to chart
 * these; this is the same rule on the other side of the wire.
 */
describe("row identifiers are not measures", () => {
  const readability = [
    {
      "Document ID": "1",
      Document: "a.txt",
      Sentences: "3",
      "Flesch Reading Ease": "79.81",
    },
    {
      "Document ID": "2",
      Document: "b.txt",
      Sentences: "3",
      "Flesch Reading Ease": "63.11",
    },
    {
      "Document ID": "3",
      Document: "c.txt",
      Sentences: "3",
      "Flesch Reading Ease": "98.38",
    },
    {
      "Document ID": "4",
      Document: "d.txt",
      Sentences: "4",
      "Flesch Reading Ease": "55.02",
    },
  ];
  const table = {
    columns: ["Document ID", "Document", "Sentences", "Flesch Reading Ease"],
    rows: readability,
  };

  it("knows a column named like an identifier", () => {
    expect(looksLikeIdentifier(readability, "Document ID")).toBe(true);
  });

  it("knows one by its shape, whatever it is called", () => {
    expect(looksLikeIdentifier(readability, "Document")).toBe(true);
  });

  it("does not mistake a repeated measure for one", () => {
    expect(looksLikeIdentifier(readability, "Sentences")).toBe(false);
  });

  it("leaves a real measure alone even when every value differs", () => {
    // Distinct values are normal in a score column; only the name saves this
    // one, which is why both rules exist.
    expect(measureColumns(table)).toContain("Flesch Reading Ease");
  });

  it("does not offer a row number as something to chart", () => {
    expect(measureColumns(table)).not.toContain("Document ID");
  });

  it("still offers something when every measure looks like an identifier", () => {
    const onlyIds = {
      columns: ["Rank"],
      rows: [{ Rank: "1" }, { Rank: "2" }, { Rank: "3" }],
    };
    expect(measureColumns(onlyIds)).toEqual(["Rank"]);
  });

  it("does not guess from too few rows", () => {
    expect(looksLikeIdentifier([{ a: "1" }, { a: "2" }], "a")).toBe(false);
  });

  it("does not chart the ID of a corpus that holds one document", () => {
    // Found by opening the app on a real one-document project. Every column
    // of that table holds exactly one value, so `Document ID` came back "not
    // an identifier" -- it is a constant -- and became the measure the chart
    // opened on. The name is what saves it when the values cannot.
    const single = {
      columns: ["Document ID", "Document", "Sentences", "Flesch Reading Ease"],
      rows: [
        {
          "Document ID": "1",
          Document: "only.txt",
          Sentences: "12",
          "Flesch Reading Ease": "61.3",
        },
      ],
    };
    expect(measureColumns(single)[0]).toBe("Sentences");
    expect(measureColumns(single)).not.toContain("Document ID");
  });

  it("knows a column that never changes", () => {
    const rows = [{ Year: "2020" }, { Year: "2020" }, { Year: "2020" }];
    expect(isConstant(rows, "Year")).toBe(true);
    expect(isConstant([{ Year: "2020" }, { Year: "2021" }], "Year")).toBe(
      false,
    );
  });

  it("prefers a measure that varies over one that does not", () => {
    const table = {
      columns: ["Tokens", "Corpus Size"],
      rows: [
        { Tokens: "10", "Corpus Size": "500" },
        { Tokens: "20", "Corpus Size": "500" },
        { Tokens: "30", "Corpus Size": "500" },
      ],
    };
    expect(measureColumns(table)).toEqual(["Tokens"]);
  });

  it("still offers a constant when that is all the table has", () => {
    const table = {
      columns: ["Corpus Size"],
      rows: [{ "Corpus Size": "500" }, { "Corpus Size": "500" }],
    };
    expect(measureColumns(table)).toEqual(["Corpus Size"]);
  });

  it("reads an identifier name whatever the separator", () => {
    expect(nameSaysIdentifier("Document ID")).toBe(true);
    expect(nameSaysIdentifier("doc_id")).toBe(true);
    expect(nameSaysIdentifier("Rank")).toBe(true);
    expect(nameSaysIdentifier("Sentences")).toBe(false);
    expect(nameSaysIdentifier("Identity")).toBe(false);
  });
});

/**
 * A line needs an order to join its points in, and a point that pools many
 * unrelated items is not a trend. Rows shaped like the real
 * ngram_cooccurrence table: one row per word pair per speech.
 */
describe("line charts", () => {
  const pairs = [
    {
      "Word 1": "of",
      "Word 2": "the",
      Count: "594",
      Document: "a.txt",
      Date: "1934-01-03",
    },
    {
      "Word 1": "and",
      "Word 2": "the",
      Count: "300",
      Document: "a.txt",
      Date: "1934-01-03",
    },
    {
      "Word 1": "be",
      "Word 2": "will",
      Count: "120",
      Document: "a.txt",
      Date: "1934-01-03",
    },
    {
      "Word 1": "of",
      "Word 2": "the",
      Count: "410",
      Document: "b.txt",
      Date: "1935-01-04",
    },
    {
      "Word 1": "free",
      "Word 2": "nation",
      Count: "12",
      Document: "b.txt",
      Date: "1935-01-04",
    },
    {
      "Word 1": "and",
      "Word 2": "be",
      Count: "95",
      Document: "b.txt",
      Date: "1935-01-04",
    },
  ];
  const line = (over: Partial<Settings>) =>
    buildLayout(pairs, settings({ kind: "line", agg: "mean", ...over }));

  it("refuses a line across labels, which have no order", () => {
    const result = line({ x: "Word 1", group: "Word 2" });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toContain('"Word 1" has no order');
      expect(result.reason).toContain("Draw bars");
    }
  });

  it("draws a line over dates", () => {
    expect(line({ x: "Date" }).ok).toBe(true);
  });

  it("warns when each point pools many different items", () => {
    const result = line({ x: "Date" });
    expect(result.ok && result.layout.caution).toContain('"Word 1"');
  });

  it("does not warn when each point is one item", () => {
    const perDocument = [
      { Document: "a.txt", Date: "1934-01-03", Flesch: "60" },
      { Document: "b.txt", Date: "1935-01-04", Flesch: "55" },
    ];
    const result = buildLayout(
      perDocument,
      settings({ kind: "line", x: "Date", y: "Flesch", agg: "mean" }),
    );
    expect(result.ok && result.layout.caution).toBeFalsy();
  });

  it("runs in date order and is never cut, whatever the sort and limit say", () => {
    const shuffled = [
      { Date: "1990-01-01", Score: "3" },
      { Date: "1950-01-01", Score: "9" },
      { Date: "1970-01-01", Score: "1" },
    ];
    const result = buildLayout(
      shuffled,
      settings({ kind: "line", x: "Date", y: "Score", sort: "high", topN: 2 }),
    );
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.layout.x.categories).toEqual([
        "1950-01-01",
        "1970-01-01",
        "1990-01-01",
      ]);
      expect(result.layout.hidden).toBe(0);
    }
  });

  it("orders numbers numerically, not as text", () => {
    const years = [
      { Year: "1999", Score: "1" },
      { Year: "200", Score: "2" },
      { Year: "1000", Score: "3" },
    ];
    const result = buildLayout(
      years,
      settings({ kind: "line", x: "Year", y: "Score" }),
    );
    expect(result.ok && result.layout.x.categories).toEqual([
      "200",
      "1000",
      "1999",
    ]);
  });
});
