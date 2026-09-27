// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { Learn, guides, matchingGuides } from "./Learn";
import type { Tool } from "./api";

/** The engine's answer: a template with its real cells, as /script/reference sends it. */
const reference = {
  markdown: "",
  functions: [],
  templates: [
    {
      id: "word-group-over-time",
      name: "Word group over time",
      description: "How often a list of words is used.",
      request: "How often is immigration mentioned over time?",
      cells: [
        { kind: "markdown", text: "A remark the example page must not show." },
        { kind: "code", text: "import nlpsuite as nlp\n\nnlp.show(rates)" },
      ],
    },
  ],
};
vi.mock("./notebooks", () => ({ scriptReference: async () => reference }));

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
    expect(tools).toHaveLength(61);
    for (const guide of Object.values(guides)) {
      expect(Object.keys(guide)).toEqual([
        "what",
        "how",
        "formula",
        "question",
        "interpretation",
        "limits",
        "settings",
        "example",
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
      // The fleshed-out sections: the formula, the settings prose and a
      // script example, all from this very tool's guide. The markup is
      // entity-escaped, so decode before matching the source strings.
      const text = html
        .replaceAll("&amp;", "&")
        .replaceAll("&#x27;", "'")
        .replaceAll("&quot;", '"');
      expect(html).toContain("How it is measured");
      expect(html).toContain("Use it in scripts");
      expect(text).toContain(guides[tool.name].example.split("\n")[0]);
      expect(text).toContain(guides[tool.name].formula.split(".")[0]);
      expect(text).toContain(guides[tool.name].settings.split(".")[0]);
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

describe("Learn as the workspace tour", () => {
  it("tours the workspace with how-to guidance, not a feature list", () => {
    const html = renderToStaticMarkup(
      <Learn tools={tools} onChoose={() => {}} canRun={false} />,
    );
    expect(html).toContain("What this workspace does");
    for (const name of [
      "Import a corpus",
      "Compare two collections",
      "Write it yourself",
      "Everything stays on this machine",
    ])
      expect(html).toContain(name);
    // Each card says how to use the feature, not only that it exists.
    expect(html).toContain("Corpus → add files or a whole folder");
  });

  it("teaches the scripts with calls the library really has", () => {
    const html = renderToStaticMarkup(
      <Learn tools={tools} onChoose={() => {}} canRun={false} />,
    );
    expect(html).toContain("How to use this tool in scripts");
    expect(html).toContain("import nlpsuite as nlp");
    expect(html).toContain("nlp.term_rates");
    expect(html).toContain("nlp.show(rates");
    expect(html).toContain("nlp.save(rates");
    expect(html).toContain("nlp.run(&quot;readability&quot;, corpus)");
    // The first script is plain text inside the markup; no template
    // literals leak into the code block.
    expect(html).not.toContain("${");
  });

  it("shows the engine's worked examples, code cells only", async () => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    await act(async () => {
      root.render(<Learn tools={tools} onChoose={() => {}} canRun={false} />);
    });
    expect(host.textContent).toContain(
      "How often is immigration mentioned over time?",
    );
    expect(host.textContent).toContain("Word group over time");
    expect(host.textContent).toContain("nlp.show(rates)");
    expect(host.textContent).not.toContain(
      "A remark the example page must not show.",
    );
    root.unmount();
    host.remove();
  });
});
