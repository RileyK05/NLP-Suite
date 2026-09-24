// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { PanelPassages } from "./PanelPassages";
import { api, post } from "./api";
import type { PanelEvidence } from "./api";
import type { PhraseAnswer, PhraseOccurrence } from "./phrase";

/**
 * Reading a panel mark in the text it came from.
 *
 * What these pin: the lookup is the one the builder wrote into the evidence
 * (never composed here); nothing is searched until the reader asks, and
 * nothing is parsed until they ask for that too; a lemma count is read back
 * as a lemma; and the page never implies the passages are the run's own when
 * the loaded documents may differ.
 */

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: vi.fn(),
  post: vi.fn(),
}));

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
afterEach(() => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
  vi.mocked(api).mockReset();
  vi.mocked(post).mockReset();
});

function render(evidence: PanelEvidence) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(<PanelPassages projectId="p1" evidence={evidence} />));
  mounted.push({ root, container });
  return container;
}

function click(container: HTMLElement, text: string) {
  const button = [...container.querySelectorAll("button")].find((b) =>
    b.textContent?.includes(text),
  );
  expect(button, `a button reading "${text}" must be on screen`).toBeDefined();
  act(() => button!.dispatchEvent(new MouseEvent("click", { bubbles: true })));
}

const shall: PanelEvidence = {
  scope: "terms",
  filters: [["Word", "shall"]],
  count: 380,
  describe: "shall: 339 in group A, 41 in group B",
  phrase: "shall",
  lemma: true,
};

function occurrence(
  id: string,
  document: string,
  match: string,
): PhraseOccurrence {
  return {
    id,
    document_id: id,
    document,
    date: "1934-01-03",
    sentence_id: "1",
    token_start: 0,
    token_end: 1,
    character_start: null,
    character_end: null,
    browser_character_start: null,
    browser_character_end: null,
    character_offset_unit: null,
    word_start: null,
    relative_position: null,
    left: "the Congress",
    match,
    right: "consider these measures",
    left_source: null,
    match_source: null,
    right_source: null,
    exact_source_highlight: false,
  } as PhraseOccurrence;
}

function answer(rows: PhraseOccurrence[], total = rows.length): PhraseAnswer {
  return {
    summary: {
      occurrences: total,
      word_tokens: 1000,
      occurrences_per_10000: null,
      eligible_documents: 87,
      matching_documents: 30,
      document_prevalence_percent: null,
      dated_documents: 87,
      undated_documents: 0,
    },
    evidence: {
      rows,
      total,
      filtered_total: total,
      offset: 0,
      limit: 8,
      truncated: false,
    },
  } as unknown as PhraseAnswer;
}

const ready = { state: "ready", error: "" };
const cold = { state: "cold", error: "" };

describe("reading a mark in context", () => {
  it("offers nothing when the builder wrote no lookup", () => {
    // A windowed collocation is not a phrase; offering "w1 w2" would search
    // only its adjacent cases and present the undercount as the evidence.
    const container = render({ ...shall, phrase: "", lemma: false });
    expect(container.textContent).toBe("");
  });

  it("searches nothing until the reader asks", () => {
    render(shall);
    expect(vi.mocked(api)).not.toHaveBeenCalled();
    expect(vi.mocked(post)).not.toHaveBeenCalled();
  });

  it("reads the builder's own lookup, as lemmas when the run counted lemmas", async () => {
    vi.mocked(api).mockResolvedValue(ready);
    vi.mocked(post).mockResolvedValue(
      answer([occurrence("a", "1934 FDR", "shall")]),
    );
    const container = render(shall);
    click(container, "in context");
    await act(async () => {});
    expect(vi.mocked(post)).toHaveBeenCalledWith("/projects/p1/live/phrase", {
      text: "shall",
      match_lemma: true,
      evidence_limit: 8,
    });
    expect(container.querySelector(".passage-list mark")?.textContent).toBe(
      "shall",
    );
    expect(container.textContent).toContain("counting every form of the word");
  });

  it("reads a form count literally", async () => {
    vi.mocked(api).mockResolvedValue(ready);
    vi.mocked(post).mockResolvedValue(
      answer([occurrence("a", "1934 FDR", "shall")]),
    );
    const container = render({ ...shall, lemma: false });
    click(container, "in context");
    await act(async () => {});
    expect(vi.mocked(post).mock.calls[0][1]).toMatchObject({
      match_lemma: false,
    });
    expect(container.textContent).not.toContain("every form");
  });

  it("says the loaded documents may not be the ones the run read", async () => {
    vi.mocked(api).mockResolvedValue(ready);
    vi.mocked(post).mockResolvedValue(
      answer([occurrence("a", "1934 FDR", "shall")]),
    );
    const container = render(shall);
    click(container, "in context");
    await act(async () => {});
    expect(container.textContent).toContain(
      "30 of the 87 documents loaded now",
    );
    expect(container.textContent).toContain(
      "may differ from the ones this run read",
    );
  });

  it("points at the phrase explorer when there are more passages than shown", async () => {
    vi.mocked(api).mockResolvedValue(ready);
    vi.mocked(post).mockResolvedValue(
      answer([occurrence("a", "1934 FDR", "shall")], 380),
    );
    const container = render(shall);
    click(container, "in context");
    await act(async () => {});
    expect(container.textContent).toContain("Showing 1 of 380");
    expect(container.textContent).toContain("phrase explorer");
  });

  it("an empty answer says where it looked", async () => {
    vi.mocked(api).mockResolvedValue(ready);
    vi.mocked(post).mockResolvedValue(answer([]));
    const container = render(shall);
    click(container, "in context");
    await act(async () => {});
    expect(container.textContent).toContain(
      "No passages found in the 87 documents loaded now",
    );
  });
});

describe("when the corpus is not loaded", () => {
  it("says so and parses only when asked", async () => {
    vi.mocked(api).mockResolvedValue(cold);
    const container = render(shall);
    click(container, "in context");
    await act(async () => {});
    expect(container.textContent).toContain("not loaded yet");
    // Control: no search and no parse happened on the first click.
    expect(vi.mocked(post)).not.toHaveBeenCalled();

    vi.mocked(post)
      .mockResolvedValueOnce(ready as never)
      .mockResolvedValueOnce(
        answer([occurrence("a", "1934 FDR", "shall")]) as never,
      );
    click(container, "Load the corpus");
    await act(async () => {});
    expect(vi.mocked(post).mock.calls[0][0]).toBe("/projects/p1/live/warm");
    expect(vi.mocked(post).mock.calls[1][0]).toBe("/projects/p1/live/phrase");
    expect(container.querySelector(".passage-list mark")?.textContent).toBe(
      "shall",
    );
  });

  it("a load that does not finish says why instead of searching", async () => {
    vi.mocked(api).mockResolvedValue(cold);
    const container = render(shall);
    click(container, "in context");
    await act(async () => {});
    vi.mocked(post).mockResolvedValueOnce({
      state: "failed",
      error: "spaCy model missing",
    } as never);
    click(container, "Load the corpus");
    await act(async () => {});
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      "spaCy model missing",
    );
    expect(vi.mocked(post)).toHaveBeenCalledTimes(1);
  });

  it("a failed request becomes a message, never a crash", async () => {
    vi.mocked(api).mockRejectedValue(new Error("server went away"));
    const container = render(shall);
    click(container, "in context");
    await act(async () => {});
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      "server went away",
    );
  });
});
