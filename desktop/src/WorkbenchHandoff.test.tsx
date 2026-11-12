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
import { act, useState, type ReactElement } from "react";
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

describe("every engine kind is a live button", () => {
  it("offers all sixteen kinds in the workbench itself", () => {
    const container = bench(vi.fn());
    const offered = groupButtons(container, "Chart type");
    // These used to sit under "Other chart types" and queue a job. They are
    // now drawn from the page on screen, like every other kind.
    for (const label of [
      "Pie",
      "Sunburst",
      "Treemap",
      "Violin",
      "Radar",
      "Waffle",
      "Calendar",
      "Sankey",
      "Cumulative distribution",
    ]) {
      expect(offered).toContain(label);
    }
    expect(offered).toHaveLength(16);
  });

  it("no longer keeps a second row of kinds that cannot be previewed", () => {
    const container = bench(vi.fn());
    expect(
      container.querySelector(".workbench-handoff")?.textContent,
    ).not.toContain("Drawn by the engine");
    expect(groupButtons(container, "Other chart types")).toEqual([]);
  });

  it("changes kind in one update, so a parallel selection clear cannot overwrite it", () => {
    // Explore applies settings and the cleared drill-down through two props.
    // Calling onSelect(null) and then onSettings() built both edits from the
    // same old render, and the second quietly restored the previous kind.
    const onSettings = vi.fn();
    const onSelect = vi.fn();
    const container = render(
      <Workbench
        table={table}
        artifactPath="counts.csv"
        settings={settings}
        onSettings={onSettings}
        selected={null}
        onSelect={onSelect}
        canPublish
        onPublish={() => {}}
      />,
    );
    press(container, "Chart type", "Pie");
    expect(onSettings).toHaveBeenCalledTimes(1);
    expect(onSettings.mock.calls[0][0]).toMatchObject({ kind: "pie" });
    expect(onSettings.mock.calls[0][1]).toBe(true);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("redraws when a newly live kind is pressed", () => {
    function Harness() {
      const [current, setCurrent] = useState(settings);
      return (
        <Workbench
          table={table}
          artifactPath="counts.csv"
          settings={current}
          onSettings={setCurrent}
          selected={null}
          onSelect={() => {}}
          canPublish
          onPublish={() => {}}
          onHandOff={vi.fn()}
        />
      );
    }
    const container = render(<Harness />);
    press(container, "Chart type", "Pie");
    expect(container.querySelector(".chart-kicker")?.textContent).toBe(
      "Share of a whole",
    );
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
