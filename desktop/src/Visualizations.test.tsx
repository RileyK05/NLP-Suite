// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { CHART_EXAMPLES, CHART_TASKS } from "./chartExamples";
import { buildLayout, LIVE_KINDS } from "./chartLayout";
import { Visualizations } from "./Visualizations";
import { CHART_KINDS } from "./VisualizeDialog";

vi.mock("./api", () => ({ api: vi.fn(), title: (name: string) => name }));
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const view = <Visualizations onExplore={() => {}} onRuns={() => {}} />;

describe("Visualizations", () => {
  it("lists every chart once and renders each example through the live layout", () => {
    const html = renderToStaticMarkup(view);
    const listed = CHART_TASKS.flatMap(({ kinds }) => [...kinds]);
    expect([...listed].sort()).toEqual([...LIVE_KINDS].sort());
    for (const kind of CHART_KINDS) {
      expect(html).toContain(kind.label);
      expect(html).toContain(kind.hint);
    }
    for (const kind of LIVE_KINDS) {
      const sample = CHART_EXAMPLES[kind];
      const result = buildLayout(sample.rows, {
        ...sample.settings,
        kind,
        agg: "",
        bins: 8,
        sort: "table",
        topN: 0,
      });
      expect(result.ok, kind).toBe(true);
    }
    expect(html).toContain("Theme mentions by genre");
    expect(html).toContain("Open Interactive");
    expect(html).toContain("View example data");
  });

  it("shows only the evidence behind a selected bin and can open the figure directory", async () => {
    vi.mocked(api).mockResolvedValue([
      {
        name: "trend",
        title: "Topic trend",
        shape: "line_series",
        summary: "Topic prevalence",
        tool: "topics",
      },
      {
        name: "places",
        title: "Places mentioned",
        shape: "geographic",
        summary: "Geocoded place mentions",
        tool: "geocode",
      },
    ]);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    try {
      await act(async () => root.render(view));
      const button = (label: string) =>
        [...container.querySelectorAll("button")].find((item) =>
          item.textContent?.includes(label),
        )!;
      act(() => button("Histogram").click());
      const firstBin = container.querySelector("[data-mark]")!;
      act(() =>
        firstBin.dispatchEvent(new MouseEvent("click", { bubbles: true })),
      );
      const rows = container.querySelectorAll(".viz-source tbody tr");
      expect(rows.length).toBeGreaterThan(0);
      expect(rows.length).toBeLessThan(CHART_EXAMPLES.histogram.rows.length);
      expect(
        container.querySelector(".viz-evidence-head")?.textContent,
      ).toContain(`${rows.length} `);
      act(() => button("Analysis figures").click());
      expect(container.textContent).toContain("Topic trend");
      expect(container.textContent).toContain("Open Past runs");
      expect(container.querySelector('input[type="search"]')).not.toBeNull();
      const [analysis, shape] = container.querySelectorAll("select");
      act(() => {
        analysis.value = "geocode";
        analysis.dispatchEvent(new Event("change", { bubbles: true }));
      });
      expect(container.textContent).toContain("Places mentioned");
      expect(container.textContent).not.toContain("Topic trend");
      expect([...shape.options].map((option) => option.textContent)).toEqual([
        "All shapes",
        "Geographic map",
      ]);
    } finally {
      act(() => root.unmount());
      container.remove();
    }
  });
});
