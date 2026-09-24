import { describe, expect, it } from "vitest";
import {
  filtersFor,
  fromViewSettings,
  isDirty,
  publicationGaps,
  selectionFor,
  toViewSettings,
  type ChartContract,
  type View,
  type ViewSettings,
} from "./views";
import type { Settings } from "./chartLayout";

const settings: Settings = {
  kind: "bar",
  x: "Word",
  y: "Count",
  group: "",
  agg: "sum",
  topN: 25,
  sort: "high",
  bins: 12,
};

const stored: ViewSettings = {
  kind: "bar",
  x: "Word",
  y: "Count",
  group: "",
  agg: "sum",
  top_n: 25,
  sort: "high",
  bins: 12,
};

const view = (over: Partial<View> = {}): View => ({
  id: "v1",
  name: "Counts",
  job: "j1",
  index: 0,
  artifact_path: "ngrams.csv",
  source_sha256: "a".repeat(64),
  settings: stored,
  filters: [],
  revision: 1,
  created: "2026-09-16T09:00:00+00:00",
  updated: "2026-09-16T09:00:00+00:00",
  source: { state: "ok", detail: "" },
  publication: [],
  ...over,
});

describe("the two shapes of one arrangement", () => {
  it("survives the round trip unchanged", () => {
    expect(fromViewSettings(toViewSettings(settings), settings)).toEqual(
      settings,
    );
  });

  it("carries every control across, not just the ones with matching names", () => {
    const fiddled: Settings = {
      ...settings,
      kind: "heatmap",
      group: "Decade",
      agg: "mean",
      topN: 7,
      sort: "low",
      bins: 30,
    };
    expect(fromViewSettings(toViewSettings(fiddled), settings)).toEqual(
      fiddled,
    );
  });

  it("renames topN to the field the server stores", () => {
    expect(toViewSettings({ ...settings, topN: 9 }).top_n).toBe(9);
  });

  it("falls back rather than trusting a kind this build cannot draw", () => {
    // A view saved by a newer build, or restored from one. Drawing the
    // fallback beats casting the string through and failing further in, where
    // nothing says where it came from.
    const restored = fromViewSettings(
      { ...stored, kind: "sunburst" },
      { ...settings, kind: "line" },
    );
    expect(restored.kind).toBe("line");
    expect(restored.y).toBe("Count");
  });
});

describe("knowing whether anything has changed", () => {
  it("is not dirty with nothing open", () => {
    expect(isDirty(null, settings, [])).toBe(false);
  });

  it("is not dirty when the controls match what was saved", () => {
    expect(isDirty(view(), settings, [])).toBe(false);
  });

  it("notices a changed setting", () => {
    expect(isDirty(view(), { ...settings, y: "Rank" }, [])).toBe(true);
  });

  it("notices a drill-down that was not saved", () => {
    expect(isDirty(view(), settings, [{ column: "Word", equals: "the" }])).toBe(
      true,
    );
  });
});

describe("a drill-down, stored and restored", () => {
  it("records the category a clicked mark stands for", () => {
    expect(filtersFor(settings, { label: "the", group: "(all)" })).toEqual([
      { column: "Word", equals: "the" },
    ]);
  });

  it("records the group as well when one is chosen", () => {
    expect(
      filtersFor(
        { ...settings, group: "Decade" },
        { label: "the", group: "1990s" },
      ),
    ).toEqual([
      { column: "Word", equals: "the" },
      { column: "Decade", equals: "1990s" },
    ]);
  });

  it("stores nothing for a histogram, whose marks are bins not values", () => {
    // `rowsBehind` skips the x predicate for a histogram, so storing one would
    // reopen the view showing rows it was never filtered to.
    expect(
      filtersFor({ ...settings, kind: "histogram" }, { label: "3", group: "" }),
    ).toEqual([]);
  });

  it("stores nothing when nothing is selected", () => {
    expect(filtersFor(settings, null)).toEqual([]);
  });

  it("comes back as the mark it was", () => {
    const mark = { label: "the", group: "1990s" };
    const grouped = { ...settings, group: "Decade" };
    expect(selectionFor(grouped, filtersFor(grouped, mark))).toEqual(mark);
  });

  it("is nothing when the view was saved without one", () => {
    expect(selectionFor(settings, [])).toBeNull();
  });
});

describe("what a published chart will not reproduce", () => {
  const contract: ChartContract = {
    bar: { high: "Ordering: largest first becomes category order.", low: "…" },
  };

  it("names the gap the server named", () => {
    expect(publicationGaps(contract, settings)).toEqual([
      "Ordering: largest first becomes category order.",
    ]);
  });

  it("says nothing about an arrangement the engine does reproduce", () => {
    expect(publicationGaps(contract, { ...settings, sort: "table" })).toEqual(
      [],
    );
  });

  it("says nothing about a kind the contract does not mention", () => {
    expect(publicationGaps(contract, { ...settings, kind: "scatter" })).toEqual(
      [],
    );
  });

  it("says nothing at all when the contract could not be fetched", () => {
    // A missing contract must not stop anybody charting; the cost is an
    // unshown caution, which is where this started.
    expect(publicationGaps({}, settings)).toEqual([]);
  });
});
