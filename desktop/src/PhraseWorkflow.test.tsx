// @vitest-environment jsdom
/**
 * The phrase workspace driven the way a researcher drives it: by clicking.
 *
 * Every defect pinned here was in a callback, not in a helper. The helpers
 * that decide these questions -- `refined`, `comparable`, `differsFromSaved` --
 * were already correct and already tested; what went wrong was a component
 * calling them with the wrong argument, or not calling them at all. A test
 * that renders the markup once and reads it cannot see that, so these render
 * the component and press its buttons.
 *
 * Three sequences matter, and each corresponds to a way the interface could
 * answer a question the reader did not ask:
 *
 * - open a saved question, load something else, turn a page
 * - compare two phrases while the corpus changes between the two requests
 * - highlight one saved record in the list while another is open, then update
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactElement } from "react";
import { PhraseExplorer } from "./PhraseExplorer";
import type {
  PhraseAnswer,
  PhraseRequest,
  SavedPhraseQuestion,
} from "./phrase";

declare global {
  // eslint-disable-next-line no-var
  var IS_REACT_ACT_ENVIRONMENT: boolean;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const mounted: Array<{ root: Root; container: HTMLElement }> = [];

afterEach(() => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
});

function render(ui: ReactElement) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(ui));
  mounted.push({ root, container });
  return container;
}

/** Click and let every promise the handler started settle. */
async function click(element: Element) {
  await act(async () => {
    element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

function control<T extends Element>(
  container: HTMLElement,
  selector: string,
  match: (element: T) => boolean,
): T {
  const found = Array.from(container.querySelectorAll<T>(selector)).find(match);
  if (!found)
    throw new Error(`no ${selector} matching in:\n${container.textContent}`);
  return found;
}

function button(container: HTMLElement, label: string): HTMLButtonElement {
  return control<HTMLButtonElement>(container, "button", (element) =>
    (element.textContent ?? "").includes(label),
  );
}

function type(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    "value",
  )!.set!;
  act(() => {
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

/** The saved-question list. The first select on the page is position detail. */
function shelf(container: HTMLElement): HTMLSelectElement {
  return container.querySelector<HTMLSelectElement>(".question-shelf select")!;
}

function choose(select: HTMLSelectElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLSelectElement.prototype,
    "value",
  )!.set!;
  act(() => {
    setter.call(select, value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

function answer(
  text: string,
  snapshot: string,
  overrides: Partial<PhraseAnswer["evidence"]> = {},
): PhraseAnswer {
  return {
    schema_version: 1,
    kind: "phrase_distribution",
    snapshot_id: snapshot,
    question: {
      subject: {
        text,
        tokens: text.split(" "),
        case_sensitive: false,
        boundary: "sentence",
        punctuation: "preserve",
        overlapping_matches: true,
        counted_forms: text.split(" ").map((token) => [token]),
      },
      matching_profile: {
        version: 2,
        tokenizer: "spacy/en_core_web_sm",
        resolved_by_snapshot: true,
        source: "snapshot",
        normalize: false,
        match_lemma: false,
        match_nominalization: false,
        unavailable: [],
      },
    },
    summary: {
      occurrences: 120,
      word_tokens: 10000,
      occurrences_per_10000: 120,
      eligible_documents: 8,
      matching_documents: 5,
      document_prevalence_percent: 62.5,
      dated_documents: 8,
      undated_documents: 0,
    },
    time: [
      {
        year: 1999,
        observed: true,
        occurrences: 4,
        word_tokens: 900,
        eligible_documents: 1,
        matching_documents: 1,
        occurrences_per_10000: 44,
        document_prevalence_percent: 100,
      },
    ],
    position: [
      {
        bin: 1,
        start_percent: 0,
        end_percent: 10,
        occurrences: 3,
        word_tokens: 1000,
        contributing_documents: 2,
        occurrences_per_10000: 30,
      },
    ],
    documents: [
      {
        document_id: "doc-1",
        document: "1999 speech.txt",
        content_sha256: "a".repeat(64),
        date: "1999-01-19",
        year: 1999,
        word_tokens: 900,
        occurrences: 4,
        occurrences_per_10000: 44,
        contains_phrase: true,
      },
    ],
    tracks: [
      {
        document_id: "doc-1",
        document: "1999 speech.txt",
        content_sha256: "a".repeat(64),
        date: "1999-01-19",
        year: 1999,
        word_tokens: 900,
        occurrences: 4,
        position_bins: 10,
        positions_truncated: false,
        positions: [
          {
            word_start: 360,
            relative_position: 0.4,
            position_bin: 5,
            sentence_id: "3",
            occurrence_id: "occurrence-1",
            character_start: 120,
            character_end: 133,
            browser_character_start: 120,
            browser_character_end: 133,
            exact_source_highlight: true,
          },
        ],
      },
    ],
    evidence: {
      rows: [
        {
          id: "occurrence-1",
          document_id: "doc-1",
          document: "1999 speech.txt",
          date: "1999-01-19",
          sentence_id: "3",
          token_start: 40,
          token_end: 42,
          word_start: 360,
          character_start: 120,
          character_end: 133,
          browser_character_start: 120,
          browser_character_end: 133,
          character_offset_unit: "unicode_code_point",
          relative_position: 0.4,
          left: "and the",
          match: text,
          right: "again",
          left_source: "and the",
          match_source: text,
          right_source: "again",
          exact_source_highlight: true,
        },
      ],
      total: 120,
      filtered_total: 120,
      offset: 0,
      limit: 50,
      truncated: true,
      filter: {
        year: null,
        document_id: null,
        position_start: null,
        position_end: null,
      },
      source_offsets: {
        status: "partial",
        unit: "unicode_code_point",
        browser_unit: "utf16_code_unit",
        verified: 100,
        total: 120,
        filtered_status: "verified",
        filtered_verified: 1,
        filtered_total: 1,
      },
      ...overrides,
    },
  };
}

function saved(
  id: string,
  name: string,
  text: string,
  revision = 1,
  comparison: string | null = null,
): SavedPhraseQuestion {
  return {
    id,
    name,
    snapshot_id: "s".repeat(64),
    parser: "spacy",
    document_ids: ["doc-1"],
    specification: {
      schema_version: 1,
      kind: "phrase_distribution",
      text,
      comparison_text: comparison,
      case_sensitive: false,
      normalize: false,
      match_lemma: false,
      match_nominalization: false,
      position_bins: 10,
      evidence_year: null,
      evidence_document_id: null,
    },
    matching: {
      version: 2,
      tokenizer: "spacy/en_core_web_sm",
      source: "snapshot",
      tokens: text.split(" "),
      comparison_tokens: comparison ? comparison.split(" ") : [],
    },
    revision,
    created: "2026-09-20T00:00:00+00:00",
    updated: "2026-09-20T00:00:00+00:00",
  };
}

type Props = Parameters<typeof PhraseExplorer>[0];

function workspace(overrides: Partial<Props> = {}) {
  // The server echoes the evidence filter back on the answer, so the mock
  // does too: the brushed status line reads the answer, not the request.
  const onTrack = vi.fn(async (request: PhraseRequest) =>
    answer(request.text, "snapshot-live", {
      filter: {
        year: request.evidence_year ?? null,
        document_id: request.evidence_document_id ?? null,
        position_start: request.evidence_position_start ?? null,
        position_end: request.evidence_position_end ?? null,
      },
    }),
  );
  const props: Props = {
    ready: true,
    onTrack,
    onReadSource: vi.fn(async () => null),
    savedQuestions: [],
    onSaveQuestion: vi.fn(async () => null),
    onOpenQuestion: vi.fn(async () => null),
    onDuplicateQuestion: vi.fn(async () => null),
    onDeleteQuestion: vi.fn(async () => false),
    onPublishQuestion: vi.fn(),
    canPublish: true,
    ...overrides,
  };
  return { props, container: render(<PhraseExplorer {...props} />) };
}

async function track(container: HTMLElement, phrase: string, compare = "") {
  const inputs = container.querySelectorAll<HTMLInputElement>(
    ".phrase-query input",
  );
  type(inputs[0], phrase);
  if (compare) type(inputs[1], compare);
  await click(button(container, "Track phrase"));
}

describe("paging and drilling stay on the question that was asked", () => {
  it("pages the reopened question against the snapshot that answered it", async () => {
    const record = saved("A", "American people", "the American people");
    const onOpenQuestion = vi.fn(async () => ({
      primary: answer("the American people", "snapshot-from-the-saved-corpus"),
      comparison: null,
    }));
    const { props, container } = workspace({
      savedQuestions: [record],
      onOpenQuestion,
    });
    choose(shelf(container), "A");
    await click(button(container, "Open"));
    expect(onOpenQuestion).toHaveBeenCalledWith(record);

    await click(button(container, "Next"));

    // The page request must cite the corpus the reopened answer came from. It
    // used to cite nothing, so a corpus loaded in between would answer it.
    const paged = vi.mocked(props.onTrack).mock.calls.at(-1)![0];
    expect(paged.snapshot_id).toBe("snapshot-from-the-saved-corpus");
    expect(paged.text).toBe("the American people");
    expect(paged.evidence_offset).toBe(50);
  });

  it("pages a tracked question against its own snapshot", async () => {
    const { props, container } = workspace();
    await track(container, "public health");
    await click(button(container, "Next"));
    const paged = vi.mocked(props.onTrack).mock.calls.at(-1)![0];
    expect(paged.snapshot_id).toBe("snapshot-live");
  });

  it("keeps paging the tracked question after the form is edited", async () => {
    const { props, container } = workspace();
    await track(container, "public health");
    const inputs = container.querySelectorAll<HTMLInputElement>(
      ".phrase-query input",
    );
    type(inputs[0], "private health");
    await click(button(container, "Next"));
    const paged = vi.mocked(props.onTrack).mock.calls.at(-1)![0];
    expect(paged.text).toBe("public health");
    expect(container.textContent).toContain("The form has been changed");
  });
});

describe("a comparison is two subjects in one corpus", () => {
  it("refuses a comparison answered from a corpus that changed underneath", async () => {
    const onTrack = vi.fn(async (request: PhraseRequest) =>
      answer(
        request.text,
        request.text === "private health" ? "reloaded" : "snapshot-live",
      ),
    );
    const { container } = workspace({ onTrack });
    await track(container, "public health", "private health");

    expect(container.textContent).toContain(
      "could not be answered from the same documents",
    );
    // Only the subject that survived is summarized, never the two side by side.
    expect(container.querySelectorAll(".subject-label")).toHaveLength(1);
  });

  it("shows both subjects when they describe the same corpus", async () => {
    const { container } = workspace();
    await track(container, "public health", "private health");
    expect(container.querySelectorAll(".subject-label")).toHaveLength(2);
    expect(container.textContent).not.toContain(
      "could not be answered from the same documents",
    );
  });
});

describe("the record on screen and the record in the list", () => {
  it("updates the open record, not whatever is highlighted in the list", async () => {
    const first = saved("A", "American people", "the American people", 3);
    const second = saved("B", "Public health", "public health", 7);
    const onOpenQuestion = vi.fn(async () => ({
      primary: answer("the American people", "snapshot-live"),
      comparison: null,
    }));
    const { props, container } = workspace({
      savedQuestions: [first, second],
      onOpenQuestion,
    });

    choose(shelf(container), "A");
    await click(button(container, "Open"));
    // Highlighting another row must not re-aim Update at it.
    choose(shelf(container), "B");
    expect(container.textContent).toContain(
      "is selected in the list but not open",
    );

    await click(button(container, "Update"));
    const call = vi.mocked(props.onSaveQuestion).mock.calls.at(-1)!;
    expect(call[4]).toBe("A");
    expect(call[5]).toBe(3);
    expect(call[0]).toBe("American people");
  });

  it("adopts nothing when the open fails", async () => {
    const record = saved("A", "American people", "the American people");
    const { props, container } = workspace({
      savedQuestions: [record],
      onOpenQuestion: vi.fn(async () => null),
    });
    await track(container, "public health");
    choose(shelf(container), "A");
    await click(button(container, "Open"));

    expect(props.onOpenQuestion).toHaveBeenCalledWith(record);
    // The tracked answer is still the one on screen, so there is nothing to
    // update and nothing named it. An open that failed used to leave the
    // interface offering to replace a record it never loaded.
    expect(container.textContent).not.toContain("Update “");
    expect(container.textContent).toContain("public health");
    const name = control<HTMLInputElement>(
      container,
      ".question-save input",
      () => true,
    );
    expect(name.value).toBe("");
  });

  it("fills the save box from the record it opened", async () => {
    const record = saved("A", "American people", "the American people", 4);
    const { container } = workspace({
      savedQuestions: [record],
      onOpenQuestion: vi.fn(async () => ({
        primary: answer("the American people", "snapshot-live"),
        comparison: null,
      })),
    });
    choose(shelf(container), "A");
    await click(button(container, "Open"));
    const name = control<HTMLInputElement>(
      container,
      ".question-save input",
      () => true,
    );
    expect(name.value).toBe("American people");
    expect(container.textContent).toContain("revision 4");
  });

  it("saves as new without naming a record to replace", async () => {
    const { props, container } = workspace();
    await track(container, "public health");
    const name = control<HTMLInputElement>(
      container,
      ".question-save input",
      () => true,
    );
    type(name, "Public health over time");
    await click(button(container, "Save question"));
    const call = vi.mocked(props.onSaveQuestion).mock.calls.at(-1)!;
    expect(call[0]).toBe("Public health over time");
    expect(call[4]).toBeNull();
    expect(call[5]).toBeNull();
  });

  it("selects a duplicate without claiming it is open", async () => {
    const record = saved("A", "American people", "the American people");
    const copy = saved("C", "American people (copy)", "the American people");
    const { props, container } = workspace({
      savedQuestions: [record, copy],
      onDuplicateQuestion: vi.fn(async () => copy),
      onOpenQuestion: vi.fn(async () => ({
        primary: answer("the American people", "snapshot-live"),
        comparison: null,
      })),
    });
    choose(shelf(container), "A");
    await click(button(container, "Open"));
    await click(button(container, "Duplicate"));

    expect(props.onDuplicateQuestion).toHaveBeenCalledWith(record);
    expect(shelf(container).value).toBe("C");
    // The answer on screen is still the original's, so Update still replaces
    // the original. Working on the copy means opening it.
    expect(button(container, "Update").textContent).toContain(
      "American people",
    );
    await click(button(container, "Update"));
    expect(vi.mocked(props.onSaveQuestion).mock.calls.at(-1)![4]).toBe("A");
  });
});

describe("the evidence says what it covers", () => {
  it("describes the selection on screen, not the whole corpus", async () => {
    const { container } = workspace();
    await track(container, "public health");
    expect(container.textContent).toContain(
      "Every highlight in this selection is verified",
    );
    expect(container.textContent).toContain(
      "Across the whole corpus, 100 of 120 are verified",
    );
  });
});

describe("following a finding into its documents and passages", () => {
  it("clicks a position bin and filters the evidence by that range", async () => {
    const { props, container } = workspace();
    await track(container, "public health");
    // The position chart's bins are buttons once they can filter. The
    // "Within texts" panel is the second section inside the phrase grid.
    const grid = container.querySelector(".phrase-grid")!;
    const within = grid.querySelectorAll<HTMLElement>("section")[1];
    const bins = Array.from(
      within.querySelectorAll<HTMLButtonElement>(".research-bar"),
    );
    expect(bins.length).toBeGreaterThan(0);
    await click(bins[0]);

    const brushed = vi.mocked(props.onTrack).mock.calls.at(-1)![0];
    expect(brushed.evidence_position_start).toBeCloseTo(0);
    expect(brushed.evidence_position_end).toBeCloseTo(0.1);
    expect(container.textContent).toContain(
      "Evidence is filtered to positions",
    );

    await click(button(container, "Clear position filter"));
    const cleared = vi.mocked(props.onTrack).mock.calls.at(-1)![0];
    expect(cleared.evidence_position_start).toBeNull();
    expect(cleared.evidence_position_end).toBeNull();
  });

  it("opens the source reader at an occurrence's verified offset", async () => {
    const onReadSource = vi.fn(async () => ({
      document_id: "doc-1",
      document: "1999 speech.txt",
      content_sha256: "a".repeat(64),
      character_start: 20,
      character_end: 300,
      browser_character_start: 20,
      browser_character_end: 300,
      character_offset_unit: "unicode_code_point" as const,
      text: "and the public health again",
    }));
    const { container } = workspace({ onReadSource });
    await track(container, "public health");
    const trackToggle = container.querySelector<HTMLButtonElement>(
      ".document-track .track-toggle",
    )!;
    await click(trackToggle);
    await click(button(container, "sentence 3"));

    expect(onReadSource).toHaveBeenCalledWith("doc-1", 120, 133, 100);
    expect(container.textContent).toContain("Source reader");
    expect(container.querySelector("mark")?.textContent).toBe("public health");

    await click(button(container, "±1,000"));
    const widened = onReadSource.mock.calls.at(-1)! as unknown as [
      string,
      number,
      number | null,
      number,
    ];
    expect(widened[3]).toBe(1000);
  });

  it("keeps zero-occurrence documents on their tracks", async () => {
    const { container } = workspace();
    await track(container, "public health");
    const empty = Array.from(
      container.querySelectorAll(".document-track"),
    ).length;
    expect(empty).toBe(1);
    expect(container.textContent).toContain(
      "0 of 1 documents contain none of the tokens",
    );
  });
});
