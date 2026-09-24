// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { PanelCanvas } from "./PanelCanvas";
import { PanelAnswerView } from "./PanelSection";
import type { PanelMark, PreparedPanel } from "./api";

/**
 * What the drawing owes: accessible marks, annotations that explain
 * themselves, the caption and the limits on the page, and a click that says
 * what it picked. The geometry behind these assertions is pinned separately
 * in `panelLayout.test.ts`; this file checks what rendering adds to it.
 */

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
afterEach(() => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
});
function render(element: ReactElement) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  mounted.push({ root, container });
  return {
    container,
    rerender: (next: ReactElement) => act(() => root.render(next)),
  };
}

const evidence = (describe: string, key: string) => ({
  scope: "rows",
  filters: [["Word", key]] as [string, string][],
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
  evidence: evidence(`${over.key}: 12 in group A, 3 in group B`, over.key),
  ...over,
});

const prepared = (over: Partial<PreparedPanel>): PreparedPanel => ({
  ok: true,
  panel: "keyness_volcano",
  shape: "scatter_labelled",
  title: "Keyness volcano",
  subtitle: "87 addresses",
  xLabel: "Log Ratio",
  yLabel: "G2",
  groups: [],
  caption: "keyness_volcano · keyness from keyness.csv (0123456789ab)",
  notes: ["G2 measures evidence rather than effect size."],
  annotations: [],
  diagnostics: [],
  table: {
    columns: ["Word"],
    rows: [{ Word: "freedom" }],
    total: 1,
    truncated: false,
  },
  marks: [
    mark({ key: "freedom", label: "freedom", x: 2.3, y: 88, labelled: true }),
    mark({ key: "the", label: "the", x: 0.04, y: 41 }),
  ],
  ...over,
});

function draw(
  over: Partial<PreparedPanel> = {},
  onSelect: (s: { key: string } | null) => void = () => {},
) {
  const panel = prepared(over);
  const { container, rerender } = render(
    <PanelCanvas prepared={panel} selected={null} onSelect={onSelect} />,
  );
  const svg = container.querySelector("svg[role=img]") as SVGSVGElement;
  return { container, svg, panel, rerender };
}

describe("what the panel owes its reader", () => {
  it("names itself for a screen reader", () => {
    const { svg } = draw();
    expect(svg.getAttribute("aria-label")).toBe("Keyness volcano");
  });

  it("shows the evidence description a hover shows as the first readout state", () => {
    const { container } = draw();
    const readout = container.querySelector(".chart-readout");
    expect(readout?.textContent).toContain("Point to read a mark");
  });

  it("makes every mark a button carrying the evidence sentence as its name", () => {
    const { container } = draw();
    const buttons = container.querySelectorAll('[role="button"]');
    expect(buttons.length).toBe(2);
    // The name a screen reader hears is the same sentence the hover shows —
    // written once, engine-side, not recomposed here.
    expect(buttons[0].getAttribute("aria-label")).toContain("12 in group A");
  });

  it("keeps exactly one mark in the tab order, arrows for the rest", () => {
    const { container } = draw();
    expect(container.querySelectorAll('[tabindex="0"]')).toHaveLength(1);
    expect(container.querySelectorAll('[tabindex="-1"]')).toHaveLength(1);
  });

  it("renders the provenance caption beneath the figure", () => {
    const { container } = draw();
    const caption = container.querySelector(".panel-caption");
    expect(caption?.textContent).toContain("keyness.csv");
    expect(caption?.textContent).toContain("0123456789ab");
  });

  it("renders the limits as 'What this can and cannot show'", () => {
    const { container } = draw();
    const details = container.querySelector("details.panel-notes");
    expect(details?.querySelector("summary")?.textContent).toContain(
      "What this can and cannot show",
    );
    expect(details?.textContent).toContain("effect size");
  });

  it("renders no notes block when the panel declares none", () => {
    const { container } = draw({ notes: [] });
    expect(container.querySelector("details.panel-notes")).toBeNull();
  });
});

describe("reference lines explain themselves", () => {
  it("draws the significance line with its note visible on the page", () => {
    // A line at G2 = 3.84 means nothing; this note is the whole reason
    // Annotation.note exists, so it must reach the page, not just aria.
    const { container } = draw({
      annotations: [
        {
          kind: "hline",
          value: 3.84,
          label: "p < 0.05",
          note: "p < 0.05 at 1 degree of freedom; a conventional cut, not a decision",
        },
      ],
    });
    const annotation = container.querySelector(".panel-annotation text");
    expect(annotation?.textContent).toContain("p < 0.05");
    expect(annotation?.textContent).toContain("1 degree of freedom");
    expect(container.querySelector(".panel-annotation line")).not.toBeNull();
  });

  it("keeps a reference line on the plot when every mark is beyond it", () => {
    // The test above passes on the element existing. Its marks sit at G2 41
    // and 88 and its line at 3.84, so when the scale covered the marks alone
    // the line was drawn off the plot and clipped -- present in the DOM,
    // invisible on screen. That is the pre-truncated keyness table, the one
    // case PANEL_ALL_ABOVE_THRESHOLD exists to warn about.
    const { container } = draw({
      annotations: [
        { kind: "hline", value: 3.84, label: "p < 0.05", note: "" },
      ],
    });
    const line = container.querySelector(
      ".panel-annotation line",
    ) as SVGLineElement;
    const y = Number(line.getAttribute("y1"));
    // The plot area: PAD.top (22) down to PAD.top + PLOT.height (354).
    expect(y).toBeGreaterThanOrEqual(22);
    expect(y).toBeLessThanOrEqual(354);
  });

  it("keeps a vertical reference line on the plot too", () => {
    const { container } = draw({
      marks: [
        mark({ key: "a", x: 6.9, y: 10 }),
        mark({ key: "b", x: 17.4, y: 20 }),
      ],
      annotations: [
        { kind: "vline", value: 0, label: "no association", note: "" },
      ],
    });
    const line = container.querySelector(
      ".panel-annotation line",
    ) as SVGLineElement;
    const x = Number(line.getAttribute("x1"));
    // PAD.left (84) to WIDTH - PAD.right (874).
    expect(x).toBeGreaterThanOrEqual(84);
    expect(x).toBeLessThanOrEqual(874);
  });

  it("draws a vline as a vertical dashed line", () => {
    const { container } = draw({
      annotations: [
        {
          kind: "vline",
          value: 0,
          label: "no difference",
          note: "a doubling is a log ratio of 1",
        },
      ],
    });
    expect(container.querySelectorAll(".panel-annotation line")).toHaveLength(
      1,
    );
  });

  it("marks without notes still show their label", () => {
    const { container } = draw({
      annotations: [
        { kind: "hline", value: 3.84, label: "p < 0.05", note: "" },
      ],
    });
    expect(container.querySelector(".panel-annotation text")?.textContent).toBe(
      "p < 0.05",
    );
  });
});

describe("clicking a mark", () => {
  it("reports the picked mark with its key, so the table can be filtered", () => {
    const onSelect = vi.fn();
    const { svg } = draw({}, onSelect);
    const first = svg.querySelector('[role="button"]') as Element;
    act(() => first.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onSelect).toHaveBeenCalledWith({ key: "freedom" });
  });

  it("unpicks the selection when the same mark is clicked again", () => {
    const onSelect = vi.fn();
    const { svg, rerender } = draw({}, onSelect);
    const first = svg.querySelector('[role="button"]') as Element;
    act(() => first.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onSelect).toHaveBeenCalledWith({ key: "freedom" });
    // The parent re-renders with the selection applied; a second click is a
    // request to put it back.
    rerender(
      <PanelCanvas
        prepared={prepared({})}
        selected={{ key: "freedom" }}
        onSelect={onSelect}
      />,
    );
    const again = svg.querySelector('[role="button"]') as Element;
    act(() => again.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onSelect.mock.calls.at(-1)![0]).toBeNull();
  });

  it("shows the evidence sentence in the readout on hover", () => {
    const { container, svg } = draw();
    const first = svg.querySelector('[role="button"]') as Element;
    // mouseenter does not bubble, and React listens for the bubbling
    // mouseover to synthesise it — so dispatch the event it actually
    // delegates on.
    act(() =>
      first.dispatchEvent(new MouseEvent("mouseover", { bubbles: true })),
    );
    expect(container.querySelector(".chart-readout")?.textContent).toContain(
      "12 in group A",
    );
  });

  it("focus shows the description too, so the keyboard reads the same panel", () => {
    const { container, svg } = draw();
    const first = svg.querySelector('[role="button"]') as HTMLElement;
    act(() => first.focus());
    expect(container.querySelector(".chart-readout")?.textContent).toContain(
      "12 in group A",
    );
  });
});

describe("drawing each shape", () => {
  it("draws independent year lines with a visible break at an uncovered year", () => {
    const { svg } = draw({
      shape: "line_series",
      xLabel: "Year",
      yLabel: "Occurrences per million",
      groups: ["war"],
      marks: [
        mark({ key: "2001", x: 2001, y: 4, group: "war" }),
        mark({ key: "2002", x: 2002, y: 0, group: "war" }),
        mark({ key: "2004", x: 2004, y: 6, group: "war" }),
      ],
    });
    expect(svg.querySelectorAll("polyline")).toHaveLength(2);
    expect(svg.querySelectorAll('circle[role="button"]')).toHaveLength(3);
    expect(svg.textContent).toContain("Occurrences per million");
    expect(svg.textContent).toContain("2002");
  });

  it("keeps signed sentiment year lines inside the plot", () => {
    const { svg } = draw({
      shape: "line_series",
      xLabel: "Year",
      yLabel: "Mean compound",
      groups: ["Sentiment"],
      marks: [
        mark({ key: "2001", x: 2001, y: -0.5, group: "Sentiment" }),
        mark({ key: "2002", x: 2002, y: 0.5, group: "Sentiment" }),
      ],
    });
    const dots = [...svg.querySelectorAll('circle[role="button"]')];
    expect(dots).toHaveLength(2);
    expect(Number(dots[0].getAttribute("cy"))).toBeGreaterThan(
      Number(dots[1].getAttribute("cy")),
    );
    expect(Number(dots[0].getAttribute("cy"))).toBeLessThanOrEqual(420);
    expect(svg.textContent).toMatch(/-0\.[0-9]+/);
  });

  it("keeps negative cosine bars inside the ranked value axis", () => {
    const { svg } = draw({
      shape: "ranked_bars",
      xLabel: "Cosine similarity",
      groups: [],
      marks: [
        mark({ key: "negative", label: "opposite", x: -0.4, y: 0 }),
        mark({ key: "positive", label: "related", x: 0.8, y: 1 }),
      ],
    });
    const bars = [...svg.querySelectorAll('rect[role="button"]')];
    expect(bars).toHaveLength(2);
    const firstX = Number(bars[0].getAttribute("x"));
    const secondX = Number(bars[1].getAttribute("x"));
    expect(firstX).toBeGreaterThanOrEqual(84);
    expect(firstX).toBeLessThan(secondX);
    // A negative tick, a zero that is not "-0", one precision throughout.
    const ticks = [...svg.querySelectorAll("text")].map((t) => t.textContent);
    expect(ticks).toContain("-0.5");
    expect(ticks).toContain("0");
    expect(ticks).not.toContain("-0");
  });

  it("gives ranked bars a value axis and says what a length is", () => {
    // The bars were drawn with no scale and no axis title. The title is
    // where the relevance panel says its bars are occurrences, not the
    // log-scale scores that order the rows -- so without it, the one
    // sentence that makes the figure readable never reached the page.
    const { container } = draw({
      shape: "ranked_bars",
      xLabel:
        "Occurrences (in this topic, estimated, against the whole corpus)",
      yLabel: "Terms, ordered by relevance",
      groups: ["In this topic", "In the whole corpus"],
      marks: [
        mark({
          key: "war:topic",
          label: "war",
          x: 36.5,
          y: 0,
          group: "In this topic",
        }),
        mark({
          key: "war:corpus",
          label: "war",
          x: 40,
          y: 0,
          group: "In the whole corpus",
        }),
      ],
    });
    const text = container.querySelector("svg[role=img]")?.textContent ?? "";
    expect(text).toContain("Occurrences (in this topic");
    expect(text).toContain("0");
    expect(text).toContain("40");
    // The rows label themselves where a rotated y title would sit.
    expect(text).not.toContain("Terms, ordered by relevance");
  });

  it("draws ranked bars with one row per rank, rank 0 first", () => {
    const { container } = draw({
      shape: "ranked_bars",
      xLabel: "Relevance / Saliency",
      yLabel: "Terms, ranked by relevance",
      groups: ["Relevance", "Saliency"],
      marks: [
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
        mark({
          key: "war:rel",
          label: "war",
          x: 0.7,
          y: 1,
          group: "Relevance",
        }),
      ],
    });
    // Row labels are drawn top down in rank order.
    const labels = [...container.querySelectorAll("svg text")].map(
      (node) => node.textContent ?? "",
    );
    expect(labels.indexOf("freedom")).toBeGreaterThan(-1);
    expect(labels.indexOf("freedom")).toBeLessThan(labels.indexOf("war"));
  });

  it("draws stream bands as polygons, one per declared group", () => {
    const { container } = draw({
      shape: "stream",
      xLabel: "Decade",
      yLabel: "Share of addresses",
      groups: ["Topic 0", "Topic 1"],
      marks: [
        mark({ key: "1930:0", label: "T0", x: 1930, y: 0.6, group: "Topic 0" }),
        mark({ key: "1940:0", label: "T0", x: 1940, y: 0.4, group: "Topic 0" }),
        mark({ key: "1930:1", label: "T1", x: 1930, y: 0.4, group: "Topic 1" }),
      ],
    });
    expect(container.querySelectorAll("polygon")).toHaveLength(2);
    expect(container.querySelector(".chart-legend")?.textContent).toContain(
      "Topic 0",
    );
  });

  it("draws labelled scatter marks as visible text", () => {
    const { container } = draw();
    const texts = [...container.querySelectorAll("svg text")].map(
      (node) => node.textContent ?? "",
    );
    expect(texts).toContain("freedom");
    expect(texts).not.toContain("the");
  });

  it("emits no NaN geometry for any shape", () => {
    for (const shape of [
      "scatter_labelled",
      "ranked_bars",
      "stream",
    ] as const) {
      const { container } = draw({
        shape,
        groups: ["Topic 0"],
        marks: [
          mark({ key: "a", label: "T0", x: 1, y: 0.5, group: "Topic 0" }),
        ],
      });
      expect(container.innerHTML, shape).not.toContain("NaN");
      expect(container.innerHTML, shape).not.toContain("Infinity");
    }
  });

  it("survives a single-mark panel without dividing by zero", () => {
    const { container } = draw({
      marks: [mark({ key: "only", label: "only", x: 5, y: 5 })],
    });
    expect(container.innerHTML).not.toContain("NaN");
  });

  it("refuses a shape the app cannot draw with a sentence, not a crash", () => {
    // Every declared shape is drawn now; the stand-in is one outside the
    // vocabulary (test_panel_parity keeps it there).
    const { container } = draw({ shape: "sankey" });
    expect(container.querySelector(".empty-copy")?.textContent).toContain(
      "sankey",
    );
  });

  it("draws ribbon bands as labelled rows of segment rectangles", () => {
    const { container } = draw({
      shape: "ribbon",
      xLabel: "Position in the speech",
      yLabel: "Speeches",
      groups: ["Topic 0", "Topic 1"],
      marks: [
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
          key: "1942:1",
          label: "1942",
          x: 0.0,
          y: 1,
          size: 1.0,
          group: "Topic 1",
        }),
      ],
    });
    // One rect per scored segment (counted inside the clip group: the
    // clipPath holds a rect of its own): a paragraph too short to score is
    // a gap.
    const clipped = container.querySelector("g[clipPath], g[clip-path]");
    expect(clipped?.querySelectorAll("rect")).toHaveLength(3);
    const labels = [...container.querySelectorAll("svg text")].map(
      (node) => node.textContent ?? "",
    );
    // Band labels are drawn top down in row order.
    expect(labels.indexOf("1934")).toBeGreaterThan(-1);
    expect(labels.indexOf("1934")).toBeLessThan(labels.indexOf("1942"));
    // ...and outside the clip group: they sit left of the plot, where the
    // clip rectangle would cut them away.
    expect(clipped?.querySelectorAll("text")).toHaveLength(0);
    expect(container.querySelector(".chart-legend")?.textContent).toContain(
      "Topic 0",
    );
  });
});

describe("drawing the matrix, distribution, position, network and grid shapes", () => {
  const heatmap = (over: Partial<PreparedPanel> = {}) =>
    draw({
      shape: "heatmap",
      colorScale: "diverging",
      xLabel: "Speech",
      xCategories: ["1934 Roosevelt", "a speech title far too long to print"],
      yCategories: ["1934 Roosevelt", "1942 Roosevelt"],
      marks: [
        mark({ key: "0:0", x: 0, y: 0, value: 1 }),
        mark({ key: "1:0", x: 1, y: 0, value: -0.5 }),
        mark({ key: "0:1", x: 0, y: 1, value: 0.25 }),
      ],
      ...over,
    });

  it("draws every heatmap cell as a button named by its evidence", () => {
    const { svg } = heatmap();
    const cells = [...svg.querySelectorAll('rect[role="button"]')];
    expect(cells).toHaveLength(3);
    expect(cells[1].getAttribute("aria-label")).toContain("1:0");
    // Row 0 at the top: the (0, 1) cell sits below the (0, 0) cell.
    expect(Number(cells[2].getAttribute("y"))).toBeGreaterThan(
      Number(cells[0].getAttribute("y")),
    );
    // Diverging on [-1, 1]: 1 is the last stop.
    expect(cells[0].getAttribute("fill")).toBe("#2166ac");
  });

  it("labels rows and columns, cutting long names, with a colour scale", () => {
    const { svg } = heatmap();
    const texts = [...svg.querySelectorAll("text")].map((t) => t.textContent);
    expect(texts).toContain("1942 Roosevelt");
    // Too long for the slanted column room: cut to fit, never overprinted.
    const cut = texts.find((t) => t?.startsWith("a speech title"));
    expect(cut?.endsWith("…")).toBe(true);
    // The scale says which values its ends stand for: [-1, 1] around 0.
    const scale = [...svg.querySelectorAll(".panel-scale text")].map(
      (t) => t.textContent,
    );
    expect(scale).toEqual(["1", "0", "-1"]);
    expect(svg.querySelectorAll("linearGradient stop")).toHaveLength(5);
  });

  it("draws a box per row with every point over it", () => {
    const { svg } = draw({
      shape: "distribution",
      groups: ["G"],
      yCategories: ["row a", "row b"],
      marks: [1, 2, 3, 4, 10].map((x, i) =>
        mark({ key: `p${i}`, x, y: 0, group: "G" }),
      ),
    });
    expect(svg.querySelectorAll('circle[role="button"]')).toHaveLength(5);
    // One box: row b has no observations, so none is drawn for it.
    expect(svg.querySelectorAll('g[aria-hidden="true"] rect')).toHaveLength(1);
    const texts = [...svg.querySelectorAll("text")].map((t) => t.textContent);
    expect(texts).toEqual(expect.arrayContaining(["row a", "row b"]));
  });

  it("draws position ticks that carry a wide hit area", () => {
    const { svg } = draw({
      shape: "positions",
      groups: ["war", "peace"],
      yCategories: ["1934", "1942"],
      marks: [
        mark({ key: "d0:1", x: 0.1, y: 0, group: "war" }),
        mark({ key: "d1:1", x: 0.9, y: 1, group: "peace" }),
      ],
    });
    const ticks = [...svg.querySelectorAll('g[role="button"]')];
    expect(ticks).toHaveLength(2);
    expect(ticks[0].querySelector('line[stroke="transparent"]')).not.toBeNull();
    // The second document's tick is lower and further right.
    const [a, b] = ticks.map((tick) => tick.querySelector("line") as Element);
    expect(Number(b.getAttribute("x1"))).toBeGreaterThan(
      Number(a.getAttribute("x1")),
    );
    expect(Number(b.getAttribute("y1"))).toBeGreaterThan(
      Number(a.getAttribute("y1")),
    );
  });

  const network = (): Partial<PreparedPanel> => ({
    shape: "network",
    groups: ["A"],
    edgeStyle: "elbow",
    marks: [
      mark({ key: "war", x: 0, y: 0, group: "A", labelled: true }),
      mark({ key: "peace", x: 1, y: 1, group: "A" }),
    ],
    edges: [
      {
        key: "war~peace",
        source: "war",
        target: "peace",
        weight: 3,
        label: "",
        evidence: evidence("war and peace share 40 sentences", "war~peace"),
      },
    ],
  });

  it("makes a network edge a button, selectable like a mark", () => {
    const onSelect = vi.fn();
    const { svg } = draw(network(), onSelect);
    // The edge and both nodes, one tab stop between them.
    expect(svg.querySelectorAll('[role="button"]')).toHaveLength(3);
    expect(svg.querySelectorAll('[tabindex="0"]')).toHaveLength(1);
    const link = svg.querySelector('[data-key="war~peace"]') as Element;
    expect(link.getAttribute("aria-label")).toBe(
      "war and peace share 40 sentences",
    );
    // An elbow: three points, turning at (source.x, target.y), with a wide
    // transparent twin that takes the pointer.
    const lines = link.querySelectorAll("polyline");
    expect(lines[0].getAttribute("points")?.split(" ")).toHaveLength(3);
    expect(lines[1].getAttribute("stroke")).toBe("transparent");
    act(() => link.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onSelect).toHaveBeenCalledWith({ key: "war~peace" });
  });

  it("lays out small multiples with a title per facet and its own y scale", () => {
    const { svg } = draw({
      shape: "small_multiples",
      xLabel: "Year",
      groups: ["median"],
      facets: ["Words", "Sentences"],
      marks: [
        mark({ key: "w1", x: 2001, y: 1000, facet: "Words", group: "median" }),
        mark({ key: "w2", x: 2002, y: 5000, facet: "Words", group: "median" }),
        mark({
          key: "s1",
          x: 2001,
          y: 40,
          facet: "Sentences",
          group: "median",
        }),
        mark({
          key: "s2",
          x: 2002,
          y: 200,
          facet: "Sentences",
          group: "median",
        }),
      ],
    });
    const facets = [...svg.querySelectorAll(".panel-facet")];
    expect(facets).toHaveLength(2);
    expect(facets[0].textContent).toContain("Words");
    // Each facet's own scale: the larger value in each sits at the same
    // height, although one is 5000 and the other 200.
    const tops = facets.map((facet) =>
      Math.min(
        ...[...facet.querySelectorAll("circle")].map((c) =>
          Number(c.getAttribute("cy")),
        ),
      ),
    );
    expect(tops[0]).toBeCloseTo(tops[1]);
    expect(svg.querySelectorAll("polyline")).toHaveLength(2);
  });

  it("gives fractional-date line series round ticks, not one per document", () => {
    const years = Array.from({ length: 40 }, (_, i) => 1990 + i * 0.73);
    const { svg } = draw({
      shape: "line_series",
      groups: ["speech"],
      pointsOnly: ["speech"],
      marks: years.map((x, i) =>
        mark({ key: `d${i}`, x, y: i, group: "speech" }),
      ),
    });
    const texts = [...svg.querySelectorAll("text")].map(
      (t) => t.textContent ?? "",
    );
    expect(texts).toContain("2000");
    expect(texts.some((t) => t.includes("1990.73"))).toBe(false);
    // Whole years stay whole: 2000, never "2,000".
    expect(texts.some((t) => t.includes(","))).toBe(false);
    // Points only: never joined.
    expect(svg.querySelectorAll("polyline")).toHaveLength(0);
  });

  it("uses the builder's height, mapping the engine default onto 420", () => {
    expect(heatmap({ height: 500 }).svg.getAttribute("viewBox")).toBe(
      "0 0 900 420",
    );
    expect(heatmap({ height: 1400 }).svg.getAttribute("viewBox")).toBe(
      "0 0 900 1320",
    );
    expect(heatmap().svg.getAttribute("viewBox")).toBe("0 0 900 420");
  });

  it("emits no NaN geometry for any new shape", () => {
    const single = [mark({ key: "a", x: 1, y: 0, value: 2, facet: "F" })];
    const shapes: Partial<PreparedPanel>[] = [
      {
        shape: "heatmap",
        colorScale: "sequential",
        xCategories: ["c"],
        yCategories: ["r"],
      },
      { shape: "distribution", yCategories: ["r"] },
      { shape: "positions", yCategories: ["r"] },
      { shape: "network" },
      { shape: "small_multiples", facets: ["F", "Empty"] },
    ];
    for (const over of shapes) {
      const { container } = draw({ ...over, groups: [], marks: single });
      expect(container.innerHTML, over.shape).not.toContain("NaN");
      expect(container.innerHTML, over.shape).not.toContain("Infinity");
      expect(container.querySelector(".empty-copy"), over.shape).toBeNull();
    }
  });
});

describe("a selected network edge opens its evidence", () => {
  it("finds the edge, not only marks, and shows the rows behind it", () => {
    const panel = prepared({
      shape: "network",
      groups: [],
      marks: [
        mark({ key: "war", x: 0, y: 0 }),
        mark({ key: "peace", x: 1, y: 1 }),
      ],
      edges: [
        {
          key: "war~peace",
          source: "war",
          target: "peace",
          weight: 1,
          label: "",
          evidence: evidence("war and peace share 1 sentence", "war"),
        },
      ],
      table: {
        columns: ["Word"],
        rows: [{ Word: "war" }, { Word: "peace" }],
        total: 2,
        truncated: false,
      },
    });
    const { container } = render(
      <PanelAnswerView
        projectId="p"
        answer={panel}
        selected={{ key: "war~peace" }}
        onSelect={() => {}}
      />,
    );
    const shown = container.querySelector(".panel-evidence")?.textContent;
    expect(shown).toContain("war and peace share 1 sentence");
    expect(shown).toContain("1 of 2 shown rows");
  });
});

describe("reading a figure with the pointer", () => {
  const grouped = {
    groups: ["Group A", "Group B"],
    marks: [
      mark({ key: "freedom", x: 2.3, y: 88, group: "Group A" }),
      mark({ key: "tariff", x: -3.2, y: 12, group: "Group B" }),
    ],
  };

  it("shows the mark's sentence in a tooltip by the pointer", () => {
    const { container, svg } = draw(grouped);
    const circle = svg.querySelector('[data-key="freedom"]') as SVGElement;
    act(() => {
      circle.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
      circle.dispatchEvent(
        new MouseEvent("mousemove", {
          bubbles: true,
          clientX: 40,
          clientY: 30,
        }),
      );
    });
    const tooltip = container.querySelector(".panel-tooltip");
    expect(tooltip?.textContent).toContain("freedom: 12 in group A");
    act(() => {
      svg.dispatchEvent(
        new MouseEvent("mouseout", {
          bubbles: true,
          relatedTarget: document.body,
        }),
      );
    });
  });

  it("hides a group from its legend entry and brings it back", () => {
    const { container, svg } = draw(grouped);
    const entry = [...container.querySelectorAll(".panel-legend button")].find(
      (button) => button.textContent?.includes("Group B"),
    ) as HTMLButtonElement;
    act(() => entry.click());
    expect(svg.querySelector('[data-key="tariff"]')).toBeNull();
    expect(svg.querySelector('[data-key="freedom"]')).not.toBeNull();
    expect(entry.getAttribute("aria-pressed")).toBe("false");
    act(() => entry.click());
    expect(svg.querySelector('[data-key="tariff"]')).not.toBeNull();
  });

  it("never hides the last group", () => {
    const { container, svg } = draw(grouped);
    const buttons = [
      ...container.querySelectorAll(".panel-legend button"),
    ] as HTMLButtonElement[];
    act(() => buttons[0].click());
    act(() => buttons[1].click());
    expect(svg.querySelectorAll("circle[data-key]").length).toBe(1);
  });

  it("keeps a group's colour when another is hidden", () => {
    const { container, svg } = draw(grouped);
    const before = svg
      .querySelector('[data-key="tariff"]')
      ?.getAttribute("fill");
    const entry = [...container.querySelectorAll(".panel-legend button")].find(
      (button) => button.textContent?.includes("Group A"),
    ) as HTMLButtonElement;
    act(() => entry.click());
    expect(svg.querySelector('[data-key="tariff"]')?.getAttribute("fill")).toBe(
      before,
    );
  });
});

describe("zooming", () => {
  it("offers zoom on scatter figures and says how", () => {
    const { container } = draw();
    expect(container.querySelector(".panel-zoom-hint")?.textContent).toContain(
      "Drag a box",
    );
  });

  it("does not offer zoom where it means nothing", () => {
    const { container } = draw({
      shape: "ranked_bars",
      marks: [mark({ key: "a", x: 3, y: 0 }), mark({ key: "b", x: 2, y: 1 })],
    });
    expect(container.querySelector(".panel-zoom-hint")).toBeNull();
  });
});
