import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Learn, guides, matchingGuides } from "./Learn";
import type { Tool } from "./api";

const tools: Tool[] = Object.keys(guides).map((name) => ({
  name,
  label: name,
  description: name,
  input_kind: name.startsWith("table_") ? "csv" : "corpus",
  requires_parse: false,
  params: [],
  outputs: [`${name}.csv`],
}));
describe("Learn field guide", () => {
  it("includes an in-depth guide for all current desktop tools", () => {
    expect(tools).toHaveLength(57);
    for (const guide of Object.values(guides)) {
      expect(Object.keys(guide)).toEqual([
        "what",
        "how",
        "question",
        "interpretation",
        "limits",
      ]);
      for (const section of Object.values(guide))
        expect(section.length).toBeGreaterThan(60);
    }
  });
  it("searches concepts and questions, not just tool names", () => {
    expect(matchingGuides(tools, "  TF-IDF ").map((t) => t.name)).toContain(
      "doc_similarity",
    );
    expect(matchingGuides(tools, "unmatchable-xyz")).toEqual([]);
  });
  it.each(tools)(
    "renders the $name guide and actual output reference",
    (tool) => {
      const html = renderToStaticMarkup(
        <Learn tools={[tool]} onChoose={() => {}} canRun={false} />,
      );
      expect(html).toContain("Limitations &amp; common mistakes");
      expect(html).toContain(`${tool.name}.csv`);
      expect(html).toContain("disabled");
    },
  );
  it("renders an empty catalog without crashing", () => {
    expect(
      renderToStaticMarkup(<Learn tools={[]} onChoose={() => {}} canRun />),
    ).toContain("No matching guides");
  });
  it("renders live settings without inventing defaults", () => {
    const tool = {
      ...tools[0],
      params: [
        {
          name: "seed",
          type: "int",
          label: "Random seed",
          default: 42,
          required: false,
          help: "Sampling seed",
          choices: [],
          minimum: 0,
          maximum: null,
        },
      ],
    };
    const html = renderToStaticMarkup(
      <Learn tools={[tool]} onChoose={() => {}} canRun />,
    );
    expect(html).toContain("Sampling seed");
    expect(html).toContain("Default: 42");
    expect(html).toContain("Minimum: 0");
    // The reference names a setting the way the run form does, and gives the
    // command-line flag beside it rather than instead of it.
    expect(html).toContain("Random seed");
    expect(html).toContain("--seed");
  });
});
