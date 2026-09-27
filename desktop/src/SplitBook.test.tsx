// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { SplitOptions, SplitPreview } from "./details";

/**
 * The book splitter against a fake engine: it is offered for long documents
 * only, previews before storing, re-asks when the rule changes, and applies
 * with the sha256 the preview was made from.
 */

type Call = { path: string; body: SplitOptions };
const calls: Call[] = [];

function answer(body: SplitOptions): SplitPreview {
  const blocks = body.rule === "words";
  return {
    document: { id: "book", name: "Alice.txt", sha256: "abc123" },
    rule: body.rule === "auto" ? "headings" : body.rule,
    work: "Alice",
    level: blocks ? "Section" : "Chapter",
    sections: (blocks ? [1, 2, 3, 4] : [1, 2, 3]).map((order) => ({
      order,
      title: blocks ? `Words ${order}` : `Chapter ${order}`,
      level: blocks ? "section" : "chapter",
      parents: [],
      words: order === 3 ? 120 : 2000,
      opening: `CHAPTER ${order}. It began`,
    })),
    left_out: [{ what: "Project Gutenberg header", size: 137 }],
    diagnostics: [
      {
        severity: "INFO",
        code: "SECTIONS_SHORT",
        message: "Chapter 3 is only 120 words.",
        order: 3,
      },
    ],
  };
}

vi.mock("./api", () => ({
  api: async () => ({}),
  del: async () => ({}),
  post: async (path: string, body: SplitOptions) => {
    calls.push({ path, body });
    return path.endsWith("/apply") ? { documents: [] } : answer(body);
  },
}));

const { SplitBook, splitCandidates } = await import("./SplitBook");

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
afterEach(() => {
  calls.length = 0;
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
});

async function until(check: () => boolean, what: string) {
  const deadline = Date.now() + 3000;
  while (!check()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await act(async () => new Promise((r) => setTimeout(r, 20)));
  }
}

describe("which documents are offered", () => {
  it("offers long documents, or the only one a project has", () => {
    expect(
      splitCandidates([
        { id: "a", name: "speech", words: 5000 },
        { id: "b", name: "novel", words: 120000 },
      ]).map((doc) => doc.id),
    ).toEqual(["b"]);
    expect(
      splitCandidates([{ id: "a", name: "story", words: 900 }]),
    ).toHaveLength(1);
    expect(
      splitCandidates([
        { id: "a", name: "one", words: 900 },
        { id: "b", name: "two", words: 800 },
      ]),
    ).toEqual([]);
  });
});

describe("splitting a book", () => {
  it("previews, re-asks on a new rule, and applies with the previewed sha", async () => {
    let splits = 0;
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() =>
      root.render(
        <SplitBook
          projectId="p"
          documents={[{ id: "book", name: "Alice.txt", words: 26000 }]}
          onSplit={() => {
            splits += 1;
          }}
        />,
      ),
    );
    mounted.push({ root, container });
    const open = [...container.querySelectorAll("button")].find((button) =>
      button.textContent?.includes("Split a long document"),
    )!;
    act(() => open.click());
    await until(
      () => !!container.querySelector('[aria-label="Sections found"] li'),
      "the sections",
    );
    expect(calls[0].path).toBe("/projects/p/documents/book/sections/preview");
    expect(calls[0].body.rule).toBe("auto");
    const list = container.querySelector('[aria-label="Sections found"]')!;
    expect(list.querySelectorAll("li")).toHaveLength(3);
    // A short chapter is flagged beside it, not hidden.
    expect(list.textContent).toContain("Chapter 3 is only 120 words.");
    expect(container.textContent).toContain("Project Gutenberg header: 137");

    const rule = container.querySelector<HTMLSelectElement>(
      '[aria-label="Cut at"]',
    )!;
    const setter = Object.getOwnPropertyDescriptor(
      HTMLSelectElement.prototype,
      "value",
    )!.set!;
    act(() => {
      setter.call(rule, "words");
      rule.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await until(
      () =>
        list.isConnected &&
        container.textContent!.includes("Split into 4 sections"),
      "the block preview",
    );
    const apply = [...container.querySelectorAll("button")].find((button) =>
      button.textContent?.startsWith("Split into"),
    )!;
    act(() => apply.click());
    await until(() => splits === 1, "the split");
    const last = calls.at(-1)!;
    expect(last.path).toBe("/projects/p/documents/book/sections/apply");
    expect(last.body.rule).toBe("words");
    expect(last.body.expected_sha256).toBe("abc123");
  });
});
