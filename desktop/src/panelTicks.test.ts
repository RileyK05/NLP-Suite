import { describe, expect, it } from "vitest";
import {
  axisTicks,
  fitText,
  labelMargin,
  niceTicks,
  placeLabels,
  textWidth,
  tickLabels,
} from "./panelTicks";

describe("niceTicks", () => {
  it("never produces negative zero", () => {
    // The VADER trend axis: a span straddling zero at a 0.25 step.
    for (const [low, high] of [
      [-0.4, 0.9],
      [-0.12, 0.31],
      [-1e-9, 1],
      [-0.05, 0.05],
    ]) {
      for (const tick of niceTicks(low, high)) {
        expect(Object.is(tick, -0)).toBe(false);
      }
    }
  });

  it("puts zero exactly at zero, with no floating residue", () => {
    const ticks = niceTicks(-0.3, 0.3);
    expect(ticks).toContain(0);
    expect(ticks.every((tick) => Number(tick.toFixed(10)) === tick)).toBe(
      true,
    );
  });

  it("stays inside the range it was given", () => {
    const ticks = niceTicks(1.07, 2.43);
    expect(Math.min(...ticks)).toBeGreaterThanOrEqual(1.07);
    expect(Math.max(...ticks)).toBeLessThanOrEqual(2.43);
  });
});

describe("tickLabels", () => {
  it("never prints -0", () => {
    const labels = tickLabels([-0.5, -0, 0.5, 1]);
    expect(labels).not.toContain("-0");
    expect(labels).not.toContain("−0");
    expect(labels).toContain("0");
  });

  it("uses one precision for a whole axis", () => {
    // The co-occurrence axis printed "1.50" beside "2".
    expect(tickLabels([1.5, 2, 2.5])).toEqual(["1.5", "2.0", "2.5"]);
    expect(tickLabels([0, 0.25, 0.5, 0.75, 1])).toEqual([
      "0",
      "0.25",
      "0.50",
      "0.75",
      "1.00",
    ]);
  });

  it("keeps a year axis ungrouped", () => {
    expect(tickLabels([1980, 2000, 2020], { plain: true })).toEqual([
      "1980",
      "2000",
      "2020",
    ]);
  });

  it("gives large values one shared suffix", () => {
    expect(tickLabels([0, 500000, 1000000])).toEqual(["0", "0.5M", "1.0M"]);
    expect(tickLabels([0, 20000, 40000])).toEqual(["0", "20k", "40k"]);
  });

  it("groups thousands below the suffix threshold", () => {
    expect(tickLabels([0, 2500, 5000])).toEqual(["0", "2,500", "5,000"]);
  });
});

describe("axisTicks", () => {
  it("pairs values with labels", () => {
    const ticks = axisTicks(-0.2, 0.2);
    expect(ticks.find((tick) => tick.value === 0)?.label).toBe("0");
  });
});

describe("label widths", () => {
  it("measures longer text as wider", () => {
    expect(textWidth("Franklin D. Roosevelt")).toBeGreaterThan(
      textWidth("Roosevelt"),
    );
  });

  it("keeps text that fits and cuts text that does not", () => {
    expect(fitText("war", 100)).toBe("war");
    const cut = fitText("the united states of america and the world", 80);
    expect(cut.endsWith("…")).toBe(true);
    expect(textWidth(cut)).toBeLessThanOrEqual(80);
  });

  it("sizes a margin to its labels within a floor and a share", () => {
    expect(labelMargin(["a", "b"], 900)).toBe(84);
    const long = labelMargin(["Union of Soviet Socialist Republics"], 900);
    expect(long).toBeGreaterThan(84);
    expect(labelMargin(["x".repeat(400)], 900)).toBe(Math.round(900 * 0.38));
  });
});

describe("placeLabels", () => {
  const bounds = { left: 0, top: 0, right: 600, bottom: 400 };

  it("places no two labels on top of each other", () => {
    // Points close together, the "income · private · sector" case.
    const requests = [
      { x: 200, y: 200, radius: 4, text: "high · priority" },
      { x: 210, y: 198, radius: 4, text: "income · tax" },
      { x: 220, y: 202, radius: 4, text: "private · sector" },
      { x: 205, y: 210, radius: 4, text: "labor · management" },
      { x: 215, y: 190, radius: 4, text: "business · small" },
    ];
    const placed = placeLabels(requests, bounds);
    for (const a of placed) {
      for (const b of placed) {
        if (a === b) continue;
        const overlap =
          a.box.left < b.box.right &&
          b.box.left < a.box.right &&
          a.box.top < b.box.bottom &&
          b.box.top < a.box.bottom;
        expect(overlap).toBe(false);
      }
    }
    expect(placed.length).toBeGreaterThanOrEqual(3);
  });

  it("keeps labels inside the plot", () => {
    const placed = placeLabels(
      [{ x: 598, y: 2, radius: 4, text: "states · united" }],
      bounds,
    );
    expect(placed).toHaveLength(1);
    const { box } = placed[0];
    expect(box.left).toBeGreaterThanOrEqual(0);
    expect(box.right).toBeLessThanOrEqual(600);
    expect(box.top).toBeGreaterThanOrEqual(0);
  });

  it("is deterministic", () => {
    const requests = [
      { x: 100, y: 100, radius: 5, text: "one" },
      { x: 104, y: 100, radius: 5, text: "two" },
    ];
    expect(placeLabels(requests, bounds)).toEqual(
      placeLabels(requests, bounds),
    );
  });
});
