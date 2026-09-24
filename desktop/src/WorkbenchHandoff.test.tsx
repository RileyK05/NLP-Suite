// @vitest-environment jsdom
/**
 * The way out of the workbench.
 *
 * The workbench draws seven chart kinds in the browser. The engine draws
 * fourteen, writes native Excel workbooks and standalone interactive HTML,
 * and for a long time none of that could be reached from a table on screen:
 * the only route was the Visualize page, which starts over with a tool card
 * and a file picker instead of the arrangement you are looking at. So the
 * seven kinds you could reach were the seven you got, and the Excel export
 * the catalog advertised was, from the workbench, imaginary.
 *
 * These tests are about reachability, not about drawing. They check that the
 * other kinds are offered, that pressing one carries the current arrangement
 * rather than resetting it, and that the offer is absent exactly where it
 * cannot work -- a live table has no artifact on disk for the engine to read.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Workbench } from "./Workbench";
import type { Settings } from "./chartLayout";
import type { Table } from "./api";

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
  return container;
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

/** Mount a workbench, optionally with somewhere to hand off to. */
function bench(onHandOff?: (seed: Record<string, string>) => void) {
  return render(
    <Workbench
      table={table}
      artifactPath="counts.csv"
      settings={settings}
      onSettings={() => {}}
      selected={null}
      onSelect={() => {}}
      canPublish
      onPublish={() => {}}
      onHandOff={onHandOff}
    />,
  );
}

/** Every button inside a labelled group, by its trimmed text. */
function groupButtons(container: HTMLElement, label: string): string[] {
  const group = container.querySelector(`[aria-label="${label}"]`);
  if (!group) return [];
  return [...group.querySelectorAll("button")].map((b) =>
    (b.textContent ?? "").trim(),
  );
}

function press(container: HTMLElement, label: string, text: string) {
  const group = container.querySelector(`[aria-label="${label}"]`)!;
  const button = [...group.querySelectorAll("button")].find(
    (b) => (b.textContent ?? "").trim() === text,
  );
  if (!button) throw new Error(`no "${text}" button under "${label}"`);
  act(() => {
    button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

describe("the kinds the browser cannot draw", () => {
  it("offers the engine-only chart kinds beside the live ones", () => {
    const container = bench(vi.fn());
    const offered = groupButtons(container, "Other chart types");
    // Pie was the specific gap: the original suite drew it, the engine draws
    // it, the catalog advertised it, and the workbench had no button for it.
    expect(offered).toContain("Pie");
    expect(offered).toContain("Treemap");
    expect(offered).toContain("Radar");
    expect(offered.length).toBeGreaterThanOrEqual(7);
  });

  it("does not repeat a kind the workbench already draws itself", () => {
    // A second "Bar" that queues a job and waits, next to the one that redraws
    // instantly, is two buttons with one name and different costs.
    const offered = groupButtons(bench(vi.fn()), "Other chart types");
    for (const live of ["Bar", "Line", "Scatter", "Histogram", "Heatmap"]) {
      expect(offered).not.toContain(live);
    }
  });

  it("hands off the arrangement on screen, not a fresh one", () => {
    const handOff = vi.fn();
    press(bench(handOff), "Other chart types", "Pie");
    expect(handOff).toHaveBeenCalledWith({
      kind: "pie",
      x: "Word",
      y: "Count",
      format: "html",
    });
  });
});

describe("exporting what is on screen", () => {
  it("offers an Excel workbook and a standalone page", () => {
    const offered = groupButtons(bench(vi.fn()), "Export this chart");
    expect(offered).toContain("Excel workbook");
    expect(offered).toContain("Interactive HTML");
  });

  it("keeps the current chart kind when exporting", () => {
    // Exporting is "this, as a file", so the kind is the one being looked at.
    const handOff = vi.fn();
    press(bench(handOff), "Export this chart", "Excel workbook");
    expect(handOff).toHaveBeenCalledWith({
      kind: "bar",
      x: "Word",
      y: "Count",
      format: "xlsx",
    });
  });
});

describe("where there is nothing to hand off to", () => {
  it("offers neither row without a handler", () => {
    // The live bench computes its table in memory; the engine reads a CSV off
    // disk. Offering the button there would be a control that cannot work,
    // which is worse than the absence it replaces.
    const container = bench(undefined);
    expect(container.querySelector(".workbench-handoff")).toBeNull();
    expect(groupButtons(container, "Other chart types")).toEqual([]);
    expect(groupButtons(container, "Export this chart")).toEqual([]);
  });

  it("still draws the live kinds", () => {
    // The absence above is specific. The workbench itself is unaffected.
    expect(
      groupButtons(bench(undefined), "Chart type").length,
    ).toBeGreaterThanOrEqual(7);
  });
});
