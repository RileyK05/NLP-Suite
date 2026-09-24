import { describe, expect, it } from "vitest";
import {
  boxStats,
  buildPanelLayout,
  COLOR_SCALES,
  edgePoints,
  edgeWidth,
  facetGrid,
  groupColors,
  heatColor,
  heatDomain,
  jitter,
  quantile,
  rowsBehind,
  type PlacedMark,
} from "./panelLayout";
import { OKABE_ITO } from "./chartLayout";
import type { PanelEvidence, PanelMark, PreparedPanel } from "./api";

/**
 * The numbers a panel claims are the part worth testing, and they are all
 * decided here — the same reason `chartLayout.test.ts` exists for charts.
 *
 * Two of these pin rules that also have a Python side, because the two
 * implementations must agree across the language boundary (pinned jointly by
 * `tests/test_panel_parity.py`):
 *
 * * **Stream stacking in declared group order with zero-fill** — mirrors
 *   `core/viz/panel_plotters.py::_stream`, pinned Python-side by
 *   `TestStream.test_a_missing_bucket_is_zero_not_a_line_across_the_gap`.
 * * **`ranked_bars` rank 0 first** — deliberately *not* the Plotly
 *   renderer's reversal: SVG y grows downward, so rank 0 is simply the first
 *   row. `TestRankedBars` pins Plotly's order; this file pins SVG's.
 */

const evidence = (
  describe: string,
  filters: [string, string][] = [["Word", "a"]],
): PanelEvidence => ({
  scope: "rows",
  filters,
  count: 1,
  describe,
});

const mark = (over: Partial<PanelMark> & { key: string }): PanelMark => ({
  label: over.key,
  x: 0,
  y: 0,
  group: "",
  size: null,
  labelled: false,
  evidence: evidence(`${over.key} stands for something`),
  ...over,
});

const prepared = (
  over: Partial<PreparedPanel> & { marks: PanelMark[] },
): PreparedPanel => ({
  ok: true,
  panel: "test",
  shape: "scatter_labelled",
  title: "Test panel",
  subtitle: "",
  xLabel: "X",
  yLabel: "Y",
  groups: [],
  caption: "test panel · test tool",
  notes: [],
  annotations: [],
  table: { columns: [], rows: [], total: 0, truncated: false },
  diagnostics: [],
  ...over,
});

describe("group colours", () => {
  it("assigns colours by index into the DECLARED order, never a re-sort", () => {
    // The engine assigns the same way (panel_plotters._group_color_map); a
    // sorted copy here would give the same topic a different colour than the
    // published figure.
    const colors = groupColors(["Topic 2", "Topic 0", "Topic 1"]);
    expect(colors.get("Topic 2")).toBe(OKABE_ITO[0]);
    expect(colors.get("Topic 0")).toBe(OKABE_ITO[1]);
    expect(colors.get("Topic 1")).toBe(OKABE_ITO[2]);
  });

  it("wraps the palette rather than running out", () => {
    const many = Array.from({ length: 12 }, (_, i) => `g${i}`);
    const colors = groupColors(many);
    expect(colors.get("g8")).toBe(OKABE_ITO[0]);
  });

  it("reuses the engine palette", () => {
    expect(OKABE_ITO.length).toBe(8);
  });
});

describe("scatter_labelled", () => {
  const marks = [
    mark({
      key: "a",
      label: "freedom",
      x: 2.3,
      y: 88,
      group: "A",
      size: 132,
      labelled: true,
    }),
    mark({ key: "b", label: "the", x: 0.04, y: 41, group: "B", size: 17700 }),
  ];
  const panel = prepared({ marks, groups: ["A", "B"] });

  it("carries evidence through untouched, so a click is answerable", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "scatter_labelled")
      throw new Error("wrong shape");
    const placed = result.layout.marks;
    expect(placed).toHaveLength(2);
    expect(placed[0].evidence.describe).toBe("a stands for something");
    expect(placed[0].evidence.filters).toEqual([["Word", "a"]]);
    expect(placed[0].color).toBe(OKABE_ITO[0]);
    expect(placed[1].color).toBe(OKABE_ITO[1]);
  });

  it("keeps marks in draw order: declared groups first, then undeclared", () => {
    const stray = prepared({
      marks: [mark({ key: "z", label: "z", group: "Undeclared" }), ...marks],
      groups: ["A", "B"],
    });
    const result = buildPanelLayout(stray);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "scatter_labelled")
      throw new Error("wrong shape");
    expect(result.layout.groups).toEqual(["A", "B", "Undeclared"]);
  });

  it("colours an ungrouped panel with the first colour, as its export does", () => {
    // The intertopic map declares no groups. panel_plotters._group_color_map
    // gives its implicit "(all)" group OKABE_ITO[0]; an earlier version of
    // this file hashed the name instead and drew it orange while the exported
    // PNG of the same panel was blue.
    const ungrouped = prepared({
      marks: [mark({ key: "t0", label: "Topic 0", group: "" })],
      groups: [],
    });
    const result = buildPanelLayout(ungrouped);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "scatter_labelled")
      throw new Error("wrong shape");
    expect(result.layout.marks[0].color).toBe(OKABE_ITO[0]);
  });

  it("refuses an empty panel with a sentence, not an exception", () => {
    const result = buildPanelLayout(prepared({ marks: [] }));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("no marks");
  });

  it("refuses a shape the app cannot draw", () => {
    // Every shape the engine declares is drawn now, so the stand-in is one
    // outside the vocabulary (test_panel_parity checks it stays outside).
    const result = buildPanelLayout(
      prepared({ marks: [mark({ key: "a" })], shape: "sankey" }),
    );
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("sankey");
  });
});

describe("line_series", () => {
  it("keeps zero-hit years but breaks where the corpus had no dated year", () => {
    const result = buildPanelLayout(
      prepared({
        shape: "line_series",
        groups: ["war", "peace"],
        marks: [
          mark({ key: "war-2001", x: 2001, y: 4, group: "war" }),
          mark({ key: "war-2002", x: 2002, y: 0, group: "war" }),
          mark({ key: "war-2004", x: 2004, y: 6, group: "war" }),
          mark({ key: "peace-2001", x: 2001, y: 2, group: "peace" }),
        ],
      }),
    );
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "line_series") throw new Error("wrong shape");
    expect(
      result.layout.series[0].segments.map((segment) =>
        segment.map((m) => m.x),
      ),
    ).toEqual([[2001, 2002], [2004]]);
    expect(result.layout.series[0].segments[0][1].y).toBe(0);
    expect(result.layout.series[0].segments[0][1].evidence.describe).toContain(
      "stands for something",
    );
    expect(result.layout.series[1].color).toBe(OKABE_ITO[1]);
  });

  // The same samples as tests/test_panel_shapes.py::TestLineSeriesGap.
  const dated = (over: Partial<PreparedPanel> = {}) =>
    buildPanelLayout(
      prepared({
        shape: "line_series",
        groups: ["war"],
        marks: [1990.1, 1990.6, 1992.0].map((x, index) =>
          mark({ key: `m${index}`, x, y: index, group: "war" }),
        ),
        ...over,
      }),
    );
  const segmentsOf = (result: ReturnType<typeof dated>) => {
    if (!result.ok || result.layout.shape !== "line_series")
      throw new Error("wrong shape");
    return result.layout.series[0].segments.map((s) => s.map((m) => m.x));
  };

  it("breaks at a gap wider than one when the builder sets none", () => {
    expect(segmentsOf(dated())).toEqual([[1990.1, 1990.6], [1992.0]]);
  });

  it("uses the builder's line gap instead of a hard-coded one", () => {
    expect(segmentsOf(dated({ lineGap: 2 }))).toEqual([
      [1990.1, 1990.6, 1992.0],
    ]);
  });

  it("never joins a points-only group, but still places its marks", () => {
    const result = dated({ pointsOnly: ["war"] });
    expect(segmentsOf(result)).toEqual([]);
    if (result.ok && result.layout.shape === "line_series")
      expect(result.layout.marks).toHaveLength(3);
  });
});

describe("heatmap", () => {
  const cells = [
    [0, 0, 0.1],
    [1, 0, 0.5],
    [2, 0, 0.9],
    [0, 1, 0.3],
    [1, 1, 0.7],
  ];
  const panel = (colorScale = "sequential") =>
    prepared({
      shape: "heatmap",
      colorScale,
      xCategories: ["alpha", "beta", "gamma"],
      yCategories: ["first row", "second row"],
      marks: cells.map(([x, y, value]) =>
        mark({ key: `r${y}c${x}`, x, y, value }),
      ),
    });

  it("interpolates in RGB between evenly spaced stops, rounding half up", () => {
    // The sample tests/test_panel_shapes.py pins: 1/8 of the way is halfway
    // from #f7fbff to #c6dbef, (222.5, 235, 247), and half up is 0xdf.
    const stops = COLOR_SCALES.sequential;
    expect(heatColor(0.125, 0, 1, stops)).toBe("#dfebf7");
    expect(heatColor(0, 0, 1, stops)).toBe(stops[0]);
    expect(heatColor(0.5, 0, 1, stops)).toBe(stops[2]);
    expect(heatColor(1, 0, 1, stops)).toBe(stops[4]);
    expect(heatColor(-5, 0, 1, stops)).toBe(stops[0]);
    expect(heatColor(3, 3, 3, COLOR_SCALES.diverging)).toBe(
      COLOR_SCALES.diverging[2],
    );
  });

  it("spans min..max when sequential and a symmetric range when diverging", () => {
    expect(heatDomain([0.2, -0.4, 0.9], "sequential")).toEqual([-0.4, 0.9]);
    expect(heatDomain([0.2, -0.4, 0.9], "diverging")).toEqual([-0.9, 0.9]);
  });

  it("places one cell per mark, coloured over the panel's own domain", () => {
    const result = buildPanelLayout(panel());
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "heatmap") throw new Error("wrong shape");
    expect(result.layout.cells).toHaveLength(5);
    expect(result.layout.domain).toEqual([0.1, 0.9]);
    expect(result.layout.cells[0].color).toBe(COLOR_SCALES.sequential[0]);
    expect(result.layout.cells[2].color).toBe(COLOR_SCALES.sequential[4]);
    expect(result.layout.cells[1].value).toBe(0.5);
    expect(result.layout.cells[3].evidence.describe).toBe(
      "r1c0 stands for something",
    );
  });

  it("centres a diverging scale on zero", () => {
    const result = buildPanelLayout(
      prepared({
        shape: "heatmap",
        colorScale: "diverging",
        xCategories: ["a", "b"],
        yCategories: ["r"],
        marks: [
          mark({ key: "neg", x: 0, y: 0, value: -0.4 }),
          mark({ key: "pos", x: 1, y: 0, value: 0.9 }),
        ],
      }),
    );
    if (!result.ok || result.layout.shape !== "heatmap")
      throw new Error("wrong shape");
    expect(result.layout.domain).toEqual([-0.9, 0.9]);
    expect(heatColor(0, -0.9, 0.9, result.layout.stops)).toBe("#f7f7f7");
  });

  it("refuses a heatmap on a scale nobody declared", () => {
    const result = buildPanelLayout(panel("rainbow"));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toContain("rainbow");
  });
});

describe("distribution", () => {
  it("takes quartiles by linear interpolation, whiskers to the extremes", () => {
    // The sample tests/test_panel_shapes.py pins.
    expect(boxStats([1, 2, 3, 4, 10])).toEqual({
      min: 1,
      q1: 2,
      median: 3,
      q3: 4,
      max: 10,
    });
    expect(boxStats([4, 1, 3, 2])).toEqual({
      min: 1,
      q1: 1.75,
      median: 2.5,
      q3: 3.25,
      max: 4,
    });
    expect(quantile([1, 2, 3, 4], 0.25)).toBe(1.75);
  });

  it("jitters deterministically within a quarter row", () => {
    expect(jitter(0)).toBe(-0.25);
    expect(jitter(1)).toBeCloseTo(0.059017, 6);
    expect(jitter(2)).toBeCloseTo(-0.131966, 6);
    for (let index = 0; index < 200; index++) {
      expect(Math.abs(jitter(index))).toBeLessThanOrEqual(0.25);
    }
  });

  it("builds a box per row and numbers jitter within each row", () => {
    const values = [1, 2, 3, 4, 10];
    const result = buildPanelLayout(
      prepared({
        shape: "distribution",
        groups: ["G", "H"],
        yCategories: ["row a", "row b", "row c"],
        marks: [
          ...values.map((x, i) => mark({ key: `a${i}`, x, y: 0, group: "G" })),
          mark({ key: "b0", x: 5, y: 1, group: "H" }),
          mark({ key: "b1", x: 7, y: 1, group: "H" }),
        ],
      }),
    );
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "distribution") throw new Error("wrong shape");
    const [a, b, c] = result.layout.rows;
    expect(a.stats).toEqual({ min: 1, q1: 2, median: 3, q3: 4, max: 10 });
    expect(b.stats?.median).toBe(6);
    expect(c.stats).toBeNull();
    expect(a.points.map((p) => p.offset)).toEqual(
      [0, 1, 2, 3, 4].map((i) => jitter(i)),
    );
    expect(b.points.map((p) => p.offset)).toEqual([jitter(0), jitter(1)]);
    expect(b.points[0].color).toBe(OKABE_ITO[1]);
    expect([result.layout.valueMin, result.layout.valueMax]).toEqual([1, 10]);
  });
});

describe("positions", () => {
  it("puts ticks in their rows over at least the whole 0..1 document", () => {
    const result = buildPanelLayout(
      prepared({
        shape: "positions",
        groups: ["war", "peace"],
        yCategories: ["1934", "1942"],
        marks: [
          mark({ key: "d0:1", x: 0.1, y: 0, group: "war" }),
          mark({ key: "d0:2", x: 0.4, y: 0, group: "peace" }),
          mark({ key: "d1:1", x: 0.2, y: 1, group: "war" }),
        ],
      }),
    );
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "positions") throw new Error("wrong shape");
    expect(result.layout.rows.map((row) => row.label)).toEqual([
      "1934",
      "1942",
    ]);
    expect(result.layout.rows[0].marks.map((m) => m.key)).toEqual([
      "d0:1",
      "d0:2",
    ]);
    expect(result.layout.rows[0].marks[1].color).toBe(OKABE_ITO[1]);
    expect([result.layout.positionMin, result.layout.positionMax]).toEqual([
      0, 1,
    ]);
  });

  it("widens past 1 for token offsets", () => {
    const result = buildPanelLayout(
      prepared({
        shape: "positions",
        yCategories: ["doc"],
        marks: [mark({ key: "t", x: 540, y: 0 })],
      }),
    );
    if (!result.ok || result.layout.shape !== "positions")
      throw new Error("wrong shape");
    expect(result.layout.positionMax).toBe(540);
  });
});

describe("network", () => {
  const nodes = [
    mark({ key: "war", x: 0, y: 0, group: "A", size: 40, labelled: true }),
    mark({ key: "peace", x: 1, y: 1, group: "A", size: 10 }),
    mark({ key: "treaty", x: 0.5, y: 0.2, group: "B" }),
  ];
  const edge = (
    key: string,
    source: string,
    target: string,
    weight: number,
  ) => ({
    key,
    source,
    target,
    weight,
    label: "",
    evidence: evidence(`${key} stands for something`, [["Pair", key]]),
  });
  const panel = (edgeStyle = "straight") =>
    prepared({
      shape: "network",
      groups: ["A", "B"],
      marks: nodes,
      edgeStyle,
      edges: [
        edge("war~peace", "war", "peace", 2),
        edge("war~treaty", "war", "treaty", 4),
      ],
    });

  it("scales edge width linearly from 1px to 6px on weight / max weight", () => {
    expect([0, 2, 4].map((weight) => edgeWidth(weight, 4))).toEqual([
      1, 3.5, 6,
    ]);
    expect(edgeWidth(3, 0)).toBe(1);
  });

  it("turns an elbow at the source's x and the target's y", () => {
    const source = { x: 0.2, y: 0.9 };
    const target = { x: 0.7, y: 0.1 };
    expect(edgePoints(source, target, "elbow")).toEqual([
      { x: 0.2, y: 0.9 },
      { x: 0.2, y: 0.1 },
      { x: 0.7, y: 0.1 },
    ]);
    expect(edgePoints(source, target, "straight")).toHaveLength(2);
  });

  it("places edges with their own evidence, between the nodes' positions", () => {
    const result = buildPanelLayout(panel("elbow"));
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "network") throw new Error("wrong shape");
    expect(result.layout.nodes.map((n) => n.key)).toEqual([
      "war",
      "peace",
      "treaty",
    ]);
    expect(result.layout.nodes[2].color).toBe(OKABE_ITO[1]);
    const [first, second] = result.layout.edges;
    expect(first.points).toEqual([
      { x: 0, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ]);
    expect([first.width, second.width]).toEqual([3.5, 6]);
    expect(second.evidence.filters).toEqual([["Pair", "war~treaty"]]);
  });

  it("skips an edge to a node that is not there rather than drawing to 0,0", () => {
    const result = buildPanelLayout(
      prepared({
        shape: "network",
        marks: nodes,
        groups: ["A", "B"],
        edges: [edge("ghost", "war", "nowhere", 1)],
      }),
    );
    if (!result.ok || result.layout.shape !== "network")
      throw new Error("wrong shape");
    expect(result.layout.edges).toEqual([]);
  });
});

describe("small_multiples", () => {
  const facets = ["Words", "Sentences", "Words per sentence", "Commas"];
  const scales = [1000, 40, 25, 3];
  const panel = prepared({
    shape: "small_multiples",
    facets,
    groups: ["median", "speech"],
    pointsOnly: ["speech"],
    lineGap: 2,
    marks: facets.flatMap((facet, index) => [
      ...[2001, 2002, 2005].map((year) =>
        mark({
          key: `${facet}:${year}`,
          x: year,
          y: scales[index] * (year - 2000),
          facet,
          group: "median",
        }),
      ),
      mark({
        key: `${facet}:doc`,
        x: 2003.5,
        y: scales[index] * 2.5,
        facet,
        group: "speech",
      }),
    ]),
  });

  it("lays facets out at most three across", () => {
    expect(facetGrid(1)).toEqual({ rows: 1, columns: 1 });
    expect(facetGrid(3)).toEqual({ rows: 1, columns: 3 });
    expect(facetGrid(4)).toEqual({ rows: 2, columns: 3 });
    expect(facetGrid(7)).toEqual({ rows: 3, columns: 3 });
  });

  it("keeps facets in declared order, each with its own y range", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "small_multiples")
      throw new Error("wrong shape");
    expect(result.layout.facets.map((f) => f.name)).toEqual(facets);
    expect([result.layout.rows, result.layout.columns]).toEqual([2, 3]);
    expect(result.layout.facets.map((f) => [f.yMin, f.yMax])).toEqual([
      [1000, 5000],
      [40, 200],
      [25, 125],
      [3, 15],
    ]);
    // One x range for every facet.
    expect([result.layout.xMin, result.layout.xMax]).toEqual([2001, 2005]);
  });

  it("breaks lines at the builder's gap and never joins points-only groups", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok || result.layout.shape !== "small_multiples")
      throw new Error("wrong shape");
    const words = result.layout.facets[0];
    const median = words.series.find((s) => s.group === "median");
    const speech = words.series.find((s) => s.group === "speech");
    expect(median?.segments.map((s) => s.map((m) => m.x))).toEqual([
      [2001, 2002],
      [2005],
    ]);
    expect(speech?.segments).toEqual([]);
    expect(words.marks).toHaveLength(4);
    // Ranks run across the whole panel, not restarting per facet.
    const ranks = result.layout.facets.flatMap((f) =>
      f.marks.map((m) => m.rank),
    );
    expect(new Set(ranks).size).toBe(16);
  });
});

describe("ranked_bars", () => {
  // Rank 0 = best, exactly as the engine builds it (panels_lda_relevance).
  const marks = [
    mark({
      key: "freedom:rel",
      label: "freedom",
      x: 0.9,
      y: 0,
      group: "Relevance",
    }),
    mark({
      key: "freedom:sal",
      label: "freedom",
      x: 0.5,
      y: 0,
      group: "Saliency",
    }),
    mark({ key: "war:rel", label: "war", x: 0.7, y: 1, group: "Relevance" }),
    mark({ key: "war:sal", label: "war", x: 0.3, y: 1, group: "Saliency" }),
    mark({
      key: "shall:rel",
      label: "shall",
      x: 0.5,
      y: 2,
      group: "Relevance",
    }),
    mark({ key: "shall:sal", label: "shall", x: 0.4, y: 2, group: "Saliency" }),
  ];
  const panel = prepared({
    shape: "ranked_bars",
    marks,
    groups: ["Relevance", "Saliency"],
  });

  it("puts rank 0 in the first row — no Plotly-style reversal in SVG", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "ranked_bars") throw new Error("wrong shape");
    expect(result.layout.rows.map((row) => row.label)).toEqual([
      "freedom",
      "war",
      "shall",
    ]);
    expect(result.layout.rows[0].rank).toBe(0);
  });

  it("puts both series' marks on the same row", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "ranked_bars") throw new Error("wrong shape");
    const first = result.layout.rows[0].marks;
    expect(first.map((m) => m.group)).toEqual(["Relevance", "Saliency"]);
    expect(first[0].y).toBe(first[1].y);
  });

  it("reports the value range the bars are read against", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "ranked_bars") throw new Error("wrong shape");
    expect(result.layout.valueMax).toBeCloseTo(0.9);
  });

  it("carries each mark's own evidence", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "ranked_bars") throw new Error("wrong shape");
    const saliency = result.layout.rows[0].marks[1] as PlacedMark;
    expect(saliency.key).toBe("freedom:sal");
    expect(saliency.evidence.describe).toContain("freedom");
  });
});

describe("stream stacking", () => {
  // Topic 1 is absent in 1940 on purpose — the same fixture the Python
  // TestStream pins, so the two implementations face the same data.
  const marks = [
    mark({ key: "1930:0", label: "T0", x: 1930, y: 0.6, group: "Topic 0" }),
    mark({ key: "1940:0", label: "T0", x: 1940, y: 0.4, group: "Topic 0" }),
    mark({ key: "1930:1", label: "T1", x: 1930, y: 0.4, group: "Topic 1" }),
  ];
  const panel = prepared({
    shape: "stream",
    marks,
    groups: ["Topic 0", "Topic 1"],
  });

  it("fills a missing bucket with zero, not a gap", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "stream") throw new Error("wrong shape");
    const topicOne = result.layout.bands.find(
      (band) => band.group === "Topic 1",
    );
    expect(topicOne).toBeDefined();
    expect(topicOne?.xs).toEqual([1930, 1940]);
    // At 1940 Topic 1 contributes nothing: its band is zero-height there
    // (lower 0.4 to upper 0.4), not a line carried across the gap.
    expect(topicOne?.lower).toEqual([0.6, 0.4]);
    expect(topicOne?.upper).toEqual([1.0, 0.4]);
    expect(topicOne?.marks[1]).toBeNull();
  });

  it("stacks bands in the declared group order", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "stream") throw new Error("wrong shape");
    // Topic 0 sits at the bottom: its lower edge is zero everywhere.
    const topicZero = result.layout.bands[0];
    expect(topicZero.group).toBe("Topic 0");
    expect(topicZero.lower).toEqual([0, 0]);
    expect(topicZero.upper).toEqual([0.6, 0.4]);
    // The total is the same with or without the gap: nothing is invented.
    expect(result.layout.totals).toEqual([1.0, 0.4]);
  });

  it("spans the whole numeric axis on every band", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "stream") throw new Error("wrong shape");
    for (const band of result.layout.bands) {
      expect(band.xs).toEqual([1930, 1940]);
    }
  });

  it("assigns band colours by declared order", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "stream") throw new Error("wrong shape");
    expect(result.layout.bands[0].color).toBe(OKABE_ITO[0]);
    expect(result.layout.bands[1].color).toBe(OKABE_ITO[1]);
  });
});

/**
 * Ribbon: each speech a band of its paragraphs' topics. The builder's
 * geometry — x is where the segment starts, size is its width, y is the
 * band's row (0 at the top) — is the same data the Python `_ribbon`
 * renderer draws, pinned here on the SVG side. Paragraph 3 of the 1934
 * speech is unscored on purpose: a gap, not a coloured guess.
 */
describe("ribbon", () => {
  const marks = [
    mark({
      key: "1934:1",
      label: "1934",
      x: 0.0,
      y: 0,
      size: 0.5,
      group: "Topic 0",
    }),
    mark({
      key: "1934:2",
      label: "1934",
      x: 0.5,
      y: 0,
      size: 0.25,
      group: "Topic 1",
    }),
    mark({
      key: "1934:3",
      label: "1934",
      x: 0.75,
      y: 0,
      size: 0.25,
      group: "Topic 0",
    }),
    mark({
      key: "1942:1",
      label: "1942",
      x: 0.0,
      y: 1,
      size: 1.0,
      group: "Topic 1",
    }),
  ];
  const panel = prepared({
    shape: "ribbon",
    marks,
    groups: ["Topic 0", "Topic 1"],
  });

  it("puts row 0 in the first band — no Plotly reversal on an SVG axis", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "ribbon") throw new Error("wrong shape");
    expect(result.layout.bands.map((band) => band.label)).toEqual([
      "1934",
      "1942",
    ]);
  });

  it("draws each band's segments in position order", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "ribbon") throw new Error("wrong shape");
    expect(result.layout.bands[0].segments.map((s) => s.key)).toEqual([
      "1934:1",
      "1934:2",
      "1934:3",
    ]);
  });

  it("carries each segment's evidence and declared colour through", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "ribbon") throw new Error("wrong shape");
    const second = result.layout.bands[0].segments[1];
    expect(second.evidence.describe).toBe("1934:2 stands for something");
    expect(second.color).toBe(OKABE_ITO[1]);
  });

  it("reports the axis extent and whether it is relative", () => {
    const result = buildPanelLayout(panel);
    if (!result.ok) throw new Error(result.reason);
    if (result.layout.shape !== "ribbon") throw new Error("wrong shape");
    expect(result.layout.positionMax).toBeCloseTo(1.0);
    expect(result.layout.relative).toBe(true);
    const numbered = buildPanelLayout(
      prepared({
        shape: "ribbon",
        marks: [
          mark({ key: "a:1", label: "a", x: 0, y: 0, size: 1, group: "T" }),
          mark({ key: "a:2", label: "a", x: 1, y: 0, size: 1, group: "T" }),
        ],
        groups: ["T"],
      }),
    );
    if (!numbered.ok) throw new Error(numbered.reason);
    if (numbered.layout.shape !== "ribbon") throw new Error("wrong shape");
    expect(numbered.layout.relative).toBe(false);
    expect(numbered.layout.positionMax).toBe(2);
  });
});

describe("evidence filters against the table", () => {
  const table = {
    rows: [
      { Word: "freedom", G2: "88.4" },
      { Word: "the", G2: "41.2" },
      { Word: "freedom", G2: "12.0" },
    ],
  };

  it("selects exactly the rows the filters name, ANDed", () => {
    expect(
      rowsBehind(table, evidence("one", [["Word", "freedom"]])),
    ).toHaveLength(2);
    expect(
      rowsBehind(
        table,
        evidence("one", [
          ["Word", "freedom"],
          ["G2", "88.4"],
        ]),
      ),
    ).toHaveLength(1);
  });

  it("selects nothing for a filter that matches no row", () => {
    expect(
      rowsBehind(table, evidence("none", [["Word", "tariff"]])),
    ).toHaveLength(0);
  });

  it("treats a blank cell as blank, not as matching", () => {
    const sparse = {
      rows: [
        { Word: "a", Note: "" },
        { Word: "a", Note: "kept" },
      ],
    };
    expect(
      rowsBehind(sparse, evidence("one", [["Note", "kept"]])),
    ).toHaveLength(1);
  });
});
