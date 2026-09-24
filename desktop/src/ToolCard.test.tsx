import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { FAMILY_ORDER, ToolCard, toolFamily } from "./ToolCard";
import type { Tool } from "./api";

const tool = (over: Partial<Tool> = {}): Tool =>
  ({
    name: "readability",
    label: "Readability",
    description: "How hard each document is to read.",
    requires_parse: false,
    params: [],
    input_kind: "corpus",
    outputs: [],
    category: "analysis",
    family: "style",
    family_label: "Style & readability",
    ...over,
  }) as Tool;

describe("what kind of work a tool is", () => {
  /**
   * Identity is the engine's family, not the card's position in a list and
   * not a guess at the input format. The old derivation could not tell topic
   * modeling from word embeddings -- the exact comparison the course grades.
   */
  it("does not depend on where the tool appears in a list", () => {
    const first = toolFamily(tool());
    const again = toolFamily(tool());
    expect(first.key).toBe(again.key);
    expect(first.Icon).toBe(again.Icon);
  });

  it("gives each family its own key and names the shelf from the engine", () => {
    const seen = new Set<unknown>();
    for (const family of FAMILY_ORDER) {
      const result = toolFamily(
        tool({ family, family_label: `Label for ${family}` }),
      );
      expect(result.key).toBe(family);
      expect(result.caption).toBe(`LABEL FOR ${family}`.toUpperCase());
      seen.add(result.Icon);
    }
    // Distinct shelves read distinctly: one glyph per family, not one per
    // position in the grid. (Identity, not .name: bundled lucide components
    // do not keep their export name.)
    expect(seen.size).toBe(FAMILY_ORDER.length);
  });

  it("keeps a table visualization on the visualization shelf", () => {
    const tableViz = toolFamily(
      tool({
        family: "visualization",
        family_label: "Visualization",
        input_kind: "csv",
      }),
    );
    expect(tableViz.key).toBe("visualization");
    expect(tableViz.caption).toBe("VISUALIZATION");
  });

  it("falls back to the old derivation only when the catalog sends no family", () => {
    // A payload that predates the taxonomy still has to draw something true.
    const legacy = toolFamily(
      tool({ family: undefined, family_label: undefined, requires_parse: true }),
    );
    expect(legacy.caption).toBe("PARSER-ASSISTED");
  });
});

describe("a tool card", () => {
  const render = (t: Tool) =>
    renderToStaticMarkup(
      <ToolCard tool={t} onOpen={() => {}} disabled={false} />,
    );

  it("shows the engine's name and sentence, never the identifier", () => {
    const html = render(tool());
    expect(html).toContain("Readability");
    expect(html).toContain("How hard each document is to read.");
    expect(html).not.toContain("readability<");
  });

  it("agrees with its own caption", () => {
    const parser = tool({
      family: "parsers_conll",
      family_label: "Parsers and the CoNLL table",
    });
    const html = render(parser);
    expect(html).toContain(toolFamily(parser).caption);
    expect(html).toContain(`family-${toolFamily(parser).key}`);
  });

  it("flags a tool whose components are missing", () => {
    const html = render(
      tool({
        availability: {
          state: "needs_setup",
          message: "Needs Gensim.",
          missing: ["gensim"],
        },
      }),
    );
    expect(html).toContain("Needs setup");
    expect(html).toContain("Needs Gensim.");
  });

  it("does not flag a tool that is ready", () => {
    const html = render(
      tool({
        availability: { state: "available", message: "Ready", missing: [] },
      }),
    );
    expect(html).not.toContain("Needs setup");
  });
});
