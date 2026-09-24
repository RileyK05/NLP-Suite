import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { VisualizationCatalog } from "./VisualizationCatalog";
import type { VisualizationInfo } from "./api";

const items: VisualizationInfo[] = [
  {
    name: "excel_charts",
    title: "Native Excel charts",
    origin: "legacy",
    is_legacy: true,
    tool: "charts",
    produces: ["xlsx"],
    summary: "Real Excel workbooks with native charts.",
    legacy_module: "charts_Excel_util.py",
    legacy_note: "A Data sheet plus a first-positioned Chart sheet.",
    advantage: "",
  },
  {
    name: "html_charts",
    title: "Interactive HTML charts",
    origin: "legacy-extended",
    is_legacy: true,
    tool: "charts",
    produces: ["html", "png"],
    summary: "Standalone interactive HTML charts.",
    legacy_module: "charts_Plotly_util.py",
    legacy_note: "Plotly HTML for a handful of kinds.",
    advantage: "Thirteen kinds behind one typed spec.",
  },
  {
    name: "dispersion_plot",
    title: "Lexical dispersion plot",
    origin: "new",
    is_legacy: false,
    tool: "dispersion",
    produces: ["html"],
    summary: "Where each word falls across the corpus.",
    legacy_module: "",
    legacy_note: "",
    advantage: "The original suite had no dispersion visualization.",
  },
];

describe("Visualization catalog", () => {
  it("counts how many the original suite also drew", () => {
    const html = renderToStaticMarkup(<VisualizationCatalog items={items} />);
    expect(html).toContain("2 of 3");
  });

  it("badges each visualization with its origin", () => {
    const html = renderToStaticMarkup(<VisualizationCatalog items={items} />);
    expect(html).toContain("viz-origin-legacy");
    expect(html).toContain("viz-origin-legacy-extended");
    expect(html).toContain("viz-origin-new");
    expect(html).toContain("legacy + extended");
  });

  it("names the legacy module a port came from", () => {
    const html = renderToStaticMarkup(<VisualizationCatalog items={items} />);
    expect(html).toContain("charts_Excel_util.py");
  });

  it("states what an extended visualization adds", () => {
    const html = renderToStaticMarkup(<VisualizationCatalog items={items} />);
    expect(html).toContain("Thirteen kinds behind one typed spec.");
  });

  it("offers filters for legacy and new", () => {
    const html = renderToStaticMarkup(<VisualizationCatalog items={items} />);
    expect(html).toContain("From the original suite");
    expect(html).toContain("New here");
  });

  it("renders nothing when the catalog is empty", () => {
    expect(renderToStaticMarkup(<VisualizationCatalog items={[]} />)).toBe("");
  });
});
