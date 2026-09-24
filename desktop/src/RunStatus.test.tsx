import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Badge, Diagnostics, STATE_LABELS, stateLabel } from "./RunStatus";
import type { Diagnostic } from "./api";

const diagnostic = (severity: string, code: string, message: string) =>
  ({ severity, code, message }) as Diagnostic;

describe("what a run's state is called", () => {
  /**
   * Only DONE had a word for it. Everything else was printed as the engine's
   * identifier in lower case, so a finished run read "Completed" beside a
   * failed one reading "failed".
   */
  it("gives every state a word, in one register", () => {
    for (const [state, label] of Object.entries(STATE_LABELS)) {
      expect(label[0]).toBe(label[0].toUpperCase());
      expect(label).not.toBe(state);
      expect(label).not.toBe(state.toLowerCase());
    }
  });

  it("distinguishes a run that finished from one that finished badly", () => {
    expect(stateLabel("DONE")).toBe("Completed");
    expect(stateLabel("PARTIAL")).toBe("Finished with errors");
    expect(stateLabel("PARTIAL")).not.toBe(stateLabel("FAILED"));
  });

  it("still reads for a state nobody has named yet", () => {
    expect(stateLabel("NEEDS_REVIEW")).toBe("Needs review");
  });

  it("prints the word, not the identifier", () => {
    const html = renderToStaticMarkup(<Badge state="FAILED" />);
    expect(html).toContain("Failed");
    expect(html).not.toContain("FAILED");
  });
});

describe("what a run reported", () => {
  const render = (items: Diagnostic[]) =>
    renderToStaticMarkup(<Diagnostics items={items} />);

  /**
   * Everything went into one collapsed drawer, so a run that quietly fell back
   * to a different parser looked exactly like a clean one.
   */
  it("shows a warning rather than folding it away", () => {
    const html = render([
      diagnostic(
        "WARNING",
        "PARSER_FALLBACK",
        "Stanza has no usable English model; parsed with spaCy instead.",
      ),
    ]);
    expect(html).toContain("parsed with spaCy instead");
    expect(html).not.toContain("<details");
  });

  it("shows an error as an alert", () => {
    const html = render([
      diagnostic("ERROR", "NO_ROWS", "Nothing to analyse."),
    ]);
    expect(html).toContain('role="alert"');
    expect(html).toContain("Nothing to analyse.");
  });

  it("keeps routine notes in the drawer", () => {
    const html = render([diagnostic("INFO", "EMPTY_DOC", "Skipped one file.")]);
    expect(html).toContain("<details");
    expect(html).toContain("1 note");
  });

  it("separates the two rather than choosing between them", () => {
    const html = render([
      diagnostic("INFO", "EMPTY_DOC", "Skipped one file."),
      diagnostic("WARNING", "PARSER_FALLBACK", "Used a different parser."),
      diagnostic("INFO", "ROUNDED", "Values rounded."),
    ]);
    expect(html).toContain("Used a different parser.");
    expect(html).toContain("2 notes");
  });

  it("names the code so it can be looked up, without leading with it", () => {
    const html = render([
      diagnostic("WARNING", "PARSER_FALLBACK", "Used a different parser."),
    ]);
    expect(html.indexOf("Used a different parser.")).toBeLessThan(
      html.indexOf("PARSER_FALLBACK"),
    );
  });

  it("renders nothing when a run had nothing to report", () => {
    expect(render([])).toBe("");
  });
});
