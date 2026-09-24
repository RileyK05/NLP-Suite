import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ChartCanvas } from "./ChartCanvas";
import { LIVE_KINDS, buildLayout, type Settings } from "./chartLayout";

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
  { Word: "alpha", Count: "4", Set: "x" },
  { Word: "beta", Count: "3", Set: "y" },
  { Word: "gamma", Count: "2", Set: "x" },
];

const draw = (over: Partial<Settings> = {}) => {
  const result = buildLayout(rows, settings(over));
  if (!result.ok) throw new Error(result.reason);
  return renderToStaticMarkup(
    <ChartCanvas
      layout={result.layout}
      selected={null}
      onSelect={() => {}}
      title="test chart"
    />,
  );
};

describe("drawing a layout", () => {
  it("draws every kind without failing", () => {
    for (const kind of LIVE_KINDS) {
      const group = kind === "heatmap" ? "Set" : "";
      const x = kind === "scatter" || kind === "bubble" ? "Count" : "Word";
      const y = "Count";
      const result = buildLayout(rows, settings({ kind, group, x, y }));
      // scatter/bubble need two numeric columns; this table has one, so a
      // refusal is the correct answer and is shown as a sentence instead.
      if (!result.ok) continue;
      expect(() =>
        renderToStaticMarkup(
          <ChartCanvas
            layout={result.layout}
            selected={null}
            onSelect={() => {}}
            title="t"
          />,
        ),
      ).not.toThrow();
    }
  });

  it("puts one bar on the page per category", () => {
    const html = draw();
    expect((html.match(/<rect/g) ?? []).length).toBeGreaterThanOrEqual(3);
  });

  it("labels the axes with the columns being drawn", () => {
    const html = draw();
    expect(html).toContain("Word");
    expect(html).toContain("Count");
  });

  it("names the chart for a screen reader", () => {
    expect(draw()).toContain('aria-label="test chart"');
  });

  it("shows a legend only when there is more than one series", () => {
    expect(draw()).not.toContain("chart-legend");
    expect(draw({ group: "Set" })).toContain("chart-legend");
  });

  it("invites the reader to point at it before anything is hovered", () => {
    expect(draw()).toContain("Point to read a value");
  });

  it("offers SVG and PNG downloads of the drawn chart", () => {
    const html = draw();
    expect(html).toContain("SVG");
    expect(html).toContain("PNG");
    // Fitted to the page there is nothing to reset; the button appears only
    // once the view has moved.
    expect(html).not.toContain("Reset zoom");
  });

  it("dims the marks that a selection excludes", () => {
    const result = buildLayout(rows, settings());
    if (!result.ok) throw new Error(result.reason);
    const html = renderToStaticMarkup(
      <ChartCanvas
        layout={result.layout}
        selected={{ label: "alpha", group: "(all)" }}
        onSelect={() => {}}
        title="t"
      />,
    );
    expect(html).toContain('opacity="0.25"');
    expect(html).toContain('opacity="1"');
  });

  it("draws a box plot as a box with a median line", () => {
    const spread = [1, 2, 3, 9].map((n) => ({ Word: "a", Count: String(n) }));
    const result = buildLayout(spread, settings({ kind: "box" }));
    if (!result.ok) throw new Error(result.reason);
    const html = renderToStaticMarkup(
      <ChartCanvas
        layout={result.layout}
        selected={null}
        onSelect={() => {}}
        title="t"
      />,
    );
    expect(html).toContain("<rect");
    expect(html).toContain("<line");
  });

  it("survives a single-row table without dividing by zero", () => {
    const result = buildLayout([{ Word: "only", Count: "1" }], settings());
    if (!result.ok) throw new Error(result.reason);
    const html = renderToStaticMarkup(
      <ChartCanvas
        layout={result.layout}
        selected={null}
        onSelect={() => {}}
        title="t"
      />,
    );
    expect(html).not.toContain("NaN");
    expect(html).not.toContain("Infinity");
  });

  it("never emits NaN geometry for any kind", () => {
    for (const kind of LIVE_KINDS) {
      const group = kind === "heatmap" ? "Set" : "";
      const result = buildLayout(rows, settings({ kind, group }));
      if (!result.ok) continue;
      const html = renderToStaticMarkup(
        <ChartCanvas
          layout={result.layout}
          selected={null}
          onSelect={() => {}}
          title="t"
        />,
      );
      expect(html, kind).not.toContain("NaN");
    }
  });
});

describe("keyboard access to marks", () => {
  it("makes every mark a button with the hover description as its name", () => {
    const result = buildLayout(rows, settings());
    if (!result.ok) throw new Error(result.reason);
    const html = renderToStaticMarkup(
      <ChartCanvas
        layout={result.layout}
        selected={null}
        onSelect={() => {}}
        title="t"
      />,
    );
    expect(html).toContain('role="button"');
    // The name a screen reader hears is the same sentence the hover shows.
    expect(html).toContain("aria-label=");
    expect(html).toContain("alpha");
  });

  it("keeps exactly one mark in the tab order, whoever has focus", () => {
    const result = buildLayout(rows, settings());
    if (!result.ok) throw new Error(result.reason);
    const html = renderToStaticMarkup(
      <ChartCanvas
        layout={result.layout}
        selected={null}
        onSelect={() => {}}
        title="t"
      />,
    );
    const tabbable = html.match(/tabindex="0"/g) ?? [];
    expect(tabbable.length).toBe(1);
    // The rest stay focusable by arrow key but out of the tab sequence.
    expect(html).toContain('tabindex="-1"');
  });

  it("counts box and heatmap cells as marks too", () => {
    for (const kind of ["box", "heatmap"] as const) {
      const group = kind === "heatmap" ? "Set" : "";
      const result = buildLayout(rows, settings({ kind, group }));
      if (!result.ok) throw new Error(result.reason);
      const html = renderToStaticMarkup(
        <ChartCanvas
          layout={result.layout}
          selected={null}
          onSelect={() => {}}
          title="t"
        />,
      );
      expect(html, kind).toContain('role="button"');
    }
  });
});
