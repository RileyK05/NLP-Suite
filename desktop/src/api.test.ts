import { describe, expect, it } from "vitest";
import {
  desktopParams,
  filterRows,
  humanize,
  initialParams,
  rememberLabels,
  title,
  type Tool,
  type Param,
} from "./api";

describe("desktop mode boundaries", () => {
  it("does not offer CSV corpus-search mode or mutate the shared catalog", () => {
    const tool = {
      name: "search",
      params: [{ name: "mode", choices: ["text", "csv", "conll"] }],
    } as Tool;
    expect(desktopParams(tool)[0].choices).toEqual(["text", "conll"]);
    expect(tool.params[0].choices).toContain("csv");
  });
  it("does not offer unsupported corrected-copy output", () => {
    const tool = {
      name: "spellcheck",
      params: [{ name: "correct" }, { name: "distance" }],
    } as Tool;
    expect(desktopParams(tool).map((param) => param.name)).toEqual([
      "distance",
    ]);
  });
});

describe("analysis defaults", () => {
  const defaults = (param: Partial<Param>) =>
    initialParams({
      params: [
        {
          name: "mode",
          type: "str",
          default: null,
          required: true,
          choices: ["text", "conll"],
          help: "Mode",
          minimum: null,
          maximum: null,
          ...param,
        },
      ],
    } as Tool);
  it("initializes a required dropdown to its visible first option", () =>
    expect(defaults({}).mode).toBe("text"));
  it("preserves explicit false and zero defaults", () => {
    expect(defaults({ default: false }).mode).toBe(false);
    expect(defaults({ default: 0 }).mode).toBe(0);
  });
  it("does not invent values for optional or free-text fields", () => {
    expect(defaults({ required: false }).mode).toBeNull();
    expect(defaults({ choices: [] }).mode).toBeNull();
  });
});

describe("result table filtering", () => {
  const table = {
    columns: ["Document", "Score"],
    rows: [
      { Document: "Speech A.txt", Score: "12.5" },
      { Document: "Speech B.txt", Score: "7" },
    ],
    total: 2,
    truncated: false,
  };
  it("searches document names case-insensitively", () =>
    expect(filterRows(table, "speech a")).toHaveLength(1));
  it("keeps numeric cells searchable without coercing source values", () =>
    expect(filterRows(table, "12.5")[0].Score).toBe("12.5"));
  it("returns all rows for an empty search", () =>
    expect(filterRows(table, "")).toHaveLength(2));
  it("does not invent results for an unmatched query", () =>
    expect(filterRows(table, "missing")).toHaveLength(0));
});

/**
 * Tool names used to live in a hand-written map here, which drifted out of
 * step with the engine in both directions: tools added later showed as raw
 * identifiers, and entries stayed behind for tools the desktop had dropped.
 * The engine now sends the name with the tool, and this side only remembers it.
 */
describe("tool names", () => {
  it("uses the name the engine sent", () => {
    rememberLabels([
      { name: "doc_similarity", label: "Document similarity" } as Tool,
    ]);
    expect(title("doc_similarity")).toBe("Document similarity");
  });
  it("still reads for a tool the catalog never carried", () =>
    expect(title("some_new_tool")).toBe("Some new tool"));
  it("does not invent a name from an empty label", () => {
    rememberLabels([{ name: "blank_tool", label: "" } as Tool]);
    expect(title("blank_tool")).toBe("Blank tool");
  });
  it("capitalises without re-spelling flags as sentences", () => {
    expect(humanize("max-df-ratio")).toBe("Max df ratio");
    expect(humanize("")).toBe("");
  });
});
