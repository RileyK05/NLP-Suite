import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Markdown } from "./Markdown";

/**
 * A published `readout.md`, exactly as `core/io/writer.py` writes it: the
 * heading that names the tool and table, then `Readout.to_markdown`. Copied
 * from the engine rather than invented, so this fails if the artifact stops
 * looking like what the viewer is built to render. `tests/test_insight.py`
 * holds the same shape from the other side.
 */
const READOUT =
  "# readability - readability.csv\n\n" +
  "Readability varies widely across 5 documents.\n\n" +
  "**What the table says**\n\n" +
  "- Flesch ranges from 12.0 to 71.0.\n" +
  "- One document is far harder than the rest.\n\n" +
  "**Read with care**\n\n" +
  "- Scores assume English prose.\n";

const render = (source: string) =>
  renderToStaticMarkup(<Markdown source={source} />);

describe("the readout a run publishes", () => {
  const html = render(READOUT);

  it("renders the writer's heading as a heading", () => {
    expect(html).toContain("<h3>readability - readability.csv</h3>");
  });
  it("shows the headline as a sentence, not as a bullet", () => {
    expect(html).toContain(
      "<p>Readability varies widely across 5 documents.</p>",
    );
  });
  it("turns each section into a heading", () => {
    expect(html).toContain("<h4>What the table says</h4>");
    expect(html).toContain("<h4>Read with care</h4>");
  });
  it("turns the observations into a list", () => {
    expect(html).toContain("<li>Flesch ranges from 12.0 to 71.0.</li>");
    expect(html).toContain("<li>Scores assume English prose.</li>");
  });
  it("keeps the two lists separate", () => {
    expect(html.match(/<ul>/g)).toHaveLength(2);
  });
  it("leaves no Markdown punctuation for the reader to decode", () => {
    // The defect: served into an iframe, this file showed its own asterisks.
    expect(html).not.toContain("**");
    expect(html).not.toContain("- Flesch");
  });
});

describe("markdown rendering", () => {
  it("renders bold and code inside a sentence", () => {
    const html = render("A **strong** claim about `readability.csv`.");
    expect(html).toContain("<strong>strong</strong>");
    expect(html).toContain("<code>readability.csv</code>");
  });
  it("renders headings beneath the panel's own heading level", () => {
    expect(render("# Title")).toContain("<h3>Title</h3>");
    expect(render("## Section")).toContain("<h4>Section</h4>");
  });
  it("joins wrapped lines into one paragraph", () => {
    expect(render("one\ntwo")).toContain("<p>one two</p>");
  });
  it("starts a new paragraph at a blank line", () => {
    expect(render("one\n\ntwo").match(/<p>/g)).toHaveLength(2);
  });
  it("accepts either bullet character", () => {
    expect(render("* starred\n- dashed").match(/<li>/g)).toHaveLength(2);
  });
  it("survives an empty file", () => {
    expect(render("")).toBe('<div class="markdown-view"></div>');
  });
  it("shows text it does not understand rather than dropping it", () => {
    expect(render("| a | b |")).toContain("| a | b |");
  });
  it("never lets an artifact become markup", () => {
    const html = render("<img src=x onerror=alert(1)> and **bold**");
    expect(html).toContain("&lt;img");
    expect(html).not.toContain("<img");
    expect(html).toContain("<strong>bold</strong>");
  });
  it("does not choke on an unclosed bold marker", () => {
    expect(render("a ** dangling")).toContain("a ** dangling");
  });
  it("handles CRLF line endings", () => {
    expect(render("one\r\n\r\n- two")).toContain("<li>two</li>");
  });
});
