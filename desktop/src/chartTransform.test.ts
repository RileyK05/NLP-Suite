import { describe, expect, it } from "vitest";
import {
  IDENTITY,
  MAX_SCALE,
  clampView,
  isIdentity,
  panBy,
  zoomAt,
} from "./chartTransform";

describe("clamping a view", () => {
  it("treats the fitted plot as the floor", () => {
    expect(clampView({ scale: 0.5, pan: -2 })).toEqual(IDENTITY);
    expect(clampView({ scale: 1, pan: -0.4 })).toEqual(IDENTITY);
  });

  it("caps how far in a reader can zoom", () => {
    expect(clampView({ scale: 500, pan: 0 }).scale).toBe(MAX_SCALE);
  });

  it("never lets the plot leave the viewport", () => {
    // Panned fully left, then dragged further: the left edges align.
    expect(panBy({ scale: 4, pan: -3 }, -2)).toEqual({ scale: 4, pan: -3 });
    // Panned fully right, then dragged further: the right edges align.
    expect(panBy({ scale: 4, pan: 0 }, 2)).toEqual({ scale: 4, pan: 0 });
  });
});

describe("zooming about a point", () => {
  it("does not pan when zoom is already at its limit", () => {
    const view = { scale: MAX_SCALE, pan: -10 };
    expect(zoomAt(view, 1.25, 0.5)).toEqual(view);
  });

  it("keeps the anchor stationary when a zoom step reaches the limit", () => {
    const view = { scale: MAX_SCALE - 1, pan: -10 };
    const anchor = 0.5;
    const data = (anchor - view.pan) / view.scale;
    const next = zoomAt(view, 1.25, anchor);
    expect(next.pan + data * next.scale).toBeCloseTo(anchor);
  });
  it("keeps the data under the anchor stationary on screen", () => {
    const view = { scale: 1, pan: 0 };
    const anchor = 0.3;
    const dataUnderAnchor = (anchor - view.pan) / view.scale;
    const zoomed = zoomAt(view, 2, anchor);
    // The same data point must still render at the anchor's screen position.
    expect(zoomed.pan + dataUnderAnchor * zoomed.scale).toBeCloseTo(anchor);
  });

  it("round-trips: zooming out undoes zooming in about the same anchor", () => {
    const start = { scale: 2, pan: -1 };
    const anchor = 0.7;
    const back = zoomAt(zoomAt(start, 1.6, anchor), 1 / 1.6, anchor);
    expect(back.scale).toBeCloseTo(start.scale);
    expect(back.pan).toBeCloseTo(start.pan);
  });

  it("collapses to identity when zooming all the way out", () => {
    expect(zoomAt({ scale: 8, pan: -4 }, 0.01, 0.5)).toEqual(IDENTITY);
  });
});

describe("panning", () => {
  it("is a no-op when the plot is fitted", () => {
    expect(panBy(IDENTITY, -0.5)).toEqual(IDENTITY);
  });

  it("moves the window without changing the zoom", () => {
    const moved = panBy({ scale: 2, pan: -0.5 }, -0.25);
    expect(moved).toEqual({ scale: 2, pan: -0.75 });
  });
});

describe("isIdentity", () => {
  it("knows when there is nothing to reset", () => {
    expect(isIdentity(IDENTITY)).toBe(true);
    expect(isIdentity({ scale: 2, pan: -1 })).toBe(false);
  });
});
