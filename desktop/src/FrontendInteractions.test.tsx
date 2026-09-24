// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, type ComponentProps, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { ChartCanvas } from "./ChartCanvas";
import { buildLayout, type Settings } from "./chartLayout";
import { Explore } from "./Explore";
import { SourceReader } from "./PhraseExplorer";
import type { Table, TableSource } from "./api";

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
const table: Table = {
  columns: ["Word", "Count"],
  rows: [
    { Word: "alpha", Count: "4" },
    { Word: "beta", Count: "2" },
  ],
  total: 2,
  truncated: false,
};
const settings: Settings = {
  kind: "bar",
  x: "Word",
  y: "Count",
  group: "",
  agg: "sum",
  topN: 0,
  sort: "table",
  bins: 4,
};
function chart() {
  const result = buildLayout(table.rows, settings);
  if (!result.ok) throw new Error(result.reason);
  const select = vi.fn();
  const { container } = render(
    <ChartCanvas
      layout={result.layout}
      selected={null}
      onSelect={select}
      title="Counts"
    />,
  );
  const svg = container.querySelector("svg[role=img]") as SVGSVGElement;
  const capture = vi.fn();
  svg.setPointerCapture = capture;
  svg.getBoundingClientRect = () => ({
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    right: 900,
    bottom: 380,
    width: 900,
    height: 380,
    toJSON() {},
  });
  const mark = svg.querySelector("[data-mark]") as SVGElement;
  return { container, svg, mark, capture, select };
}
function pointer(target: Element, type: string, x: number) {
  const event = new MouseEvent(type, {
    bubbles: true,
    clientX: x,
    clientY: 100,
    button: 0,
  });
  Object.defineProperty(event, "pointerId", { value: 1 });
  act(() => target.dispatchEvent(event));
}
describe("chart gestures", () => {
  it("leaves normal clicks on the mark instead of capturing them at the SVG root", () => {
    const { mark, capture, select } = chart();
    pointer(mark, "pointerdown", 200);
    expect(capture).not.toHaveBeenCalled();
    pointer(mark, "pointerup", 200);
    act(() => mark.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(select).toHaveBeenCalledWith(
      expect.objectContaining({ label: "alpha" }),
    );
  });
  it("captures a drag only after movement, without selecting a mark", () => {
    const { svg, mark, capture, select } = chart();
    pointer(mark, "pointerdown", 200);
    pointer(svg, "pointermove", 250);
    expect(capture).toHaveBeenCalledWith(1);
    pointer(svg, "pointerup", 250);
    act(() => mark.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(select).not.toHaveBeenCalled();
  });
  it("shows the mark's value on keyboard focus", () => {
    const { mark, container } = chart();
    act(() => mark.focus());
    expect(container.querySelector(".chart-readout")?.textContent).toContain(
      "alpha",
    );
  });
  it("lets the page scroll over axis labels and zooms only inside the plot", () => {
    const { svg, container } = chart();
    const axis = new WheelEvent("wheel", {
      bubbles: true,
      cancelable: true,
      clientX: 200,
      clientY: 350,
      deltaY: -100,
    });
    act(() => svg.dispatchEvent(axis));
    expect(axis.defaultPrevented).toBe(false);
    const plot = new WheelEvent("wheel", {
      bubbles: true,
      cancelable: true,
      clientX: 200,
      clientY: 100,
      deltaY: -100,
    });
    act(() => svg.dispatchEvent(plot));
    expect(plot.defaultPrevented).toBe(true);
    expect(container.textContent).toContain("Reset zoom");
  });
});

const first: TableSource = {
  job: "one",
  index: 0,
  tool: "ngrams",
  label: "First run",
  path: "counts.csv",
  description: "Counts",
  created: "2026-09-20",
  state: "DONE",
  manifest: false,
};
const second: TableSource = { ...first, job: "two", label: "Second run" };
function exploration(over: Partial<ComponentProps<typeof Explore>> = {}) {
  return (
    <Explore
      sources={[first, second]}
      loading={false}
      selected={first}
      onSelect={() => {}}
      table={table}
      tableLoading={false}
      canPublish
      onPublish={() => {}}
      onBrowseAnalyses={() => {}}
      liveBench={null}
      defaultMode="results"
      views={[]}
      contract={{}}
      onSaveView={async () => null}
      onUpdateView={async () => null}
      onDuplicateView={async () => null}
      onDeleteView={async () => false}
      onCompareSource={() => {}}
      {...over}
    />
  );
}
describe("finished result comparison", () => {
  it("connects the picker, draws the loaded comparison and removes it on close", () => {
    const choose = vi.fn();
    const { container, rerender } = render(
      exploration({ onCompareSource: choose }),
    );
    const picker = container.querySelector(
      '[aria-label="Table to compare"]',
    ) as HTMLSelectElement;
    act(() => {
      picker.value = "two:0";
      picker.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(choose).toHaveBeenCalledWith(second);
    rerender(exploration({ compareSource: second, compare: table }));
    expect(container.querySelectorAll('svg[role="img"]')).toHaveLength(2);
    rerender(exploration());
    expect(container.querySelectorAll('svg[role="img"]')).toHaveLength(1);
  });
  it("explains incompatible columns rather than drawing an empty comparison", () => {
    const { container } = render(
      exploration({
        compareSource: second,
        compare: { ...table, columns: ["Other"], rows: [{ Other: "5" }] },
      }),
    );
    expect(container.textContent).toContain("missing columns");
    expect(container.querySelectorAll('svg[role="img"]')).toHaveLength(1);
  });
  it("shows loading and failed fetch states", () => {
    const { container, rerender } = render(
      exploration({ compareSource: second, compareLoading: true }),
    );
    expect(container.textContent).toContain("Loading comparison");
    rerender(exploration({ compareSource: second }));
    expect(container.textContent).toContain(
      "comparison table could not be read",
    );
  });
});

it("highlights code-point source offsets correctly after an emoji", () => {
  const passage = {
    document_id: "doc",
    document: "Speech",
    text: "😀 alpha beta",
    character_start: 10,
    character_end: 22,
  } as ComponentProps<typeof SourceReader>["passage"];
  const { container } = render(
    <SourceReader
      passage={passage}
      anchor={{ documentId: "doc", start: 12, end: 17, side: "primary" }}
      loading={false}
      error=""
      radius={100}
      onWiden={() => {}}
      onClose={() => {}}
    />,
  );
  expect(container.querySelector("mark")?.textContent).toBe("alpha");
  expect(container.querySelector("pre")?.textContent).toBe("😀 alpha beta");
});
