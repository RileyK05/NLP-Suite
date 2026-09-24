// @vitest-environment jsdom
/**
 * The four tick boxes that decide what counts as the phrase.
 *
 * Each one changes the question, so each one has to travel: into the request,
 * into the saved specification, back out when the question is reopened, and
 * into every "this has been edited" comparison. A widening that reaches the
 * engine but not `differsFromSaved` is an unsaved change the save button
 * reports as saved; one that reaches the request but not the reopen is a
 * saved question that answers something else the second time it is opened.
 *
 * Rendered and clicked rather than called, because the helpers underneath
 * were right the last time this went wrong and the component called them with
 * the wrong argument.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { PhraseExplorer } from "./PhraseExplorer";
import { differsFromSaved, isDraftDifferent, sameOptions } from "./phrase";
import type {
  PhraseAnswer,
  PhraseRequest,
  SavedPhraseQuestion,
} from "./phrase";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
afterEach(() => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
});
function render(element: ReactElement) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  mounted.push({ root, container });
  return container;
}

function tick(container: HTMLElement, label: string) {
  const field = Array.from(
    container.querySelectorAll(".phrase-matching .check-field"),
  ).find((element) => (element.textContent ?? "").includes(label));
  const box = field?.querySelector("input[type=checkbox]");
  if (!box) throw new Error(`no option labelled ${label}`);
  act(() => {
    (box as HTMLInputElement).click();
  });
}

function type(container: HTMLElement, value: string) {
  const input = container.querySelector(
    "input[placeholder*='phrase'], .phrase-form input[type=text]",
  ) as HTMLInputElement | null;
  const field =
    input ?? (container.querySelector("form input") as HTMLInputElement);
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    "value",
  )!.set!;
  act(() => {
    setter.call(field, value);
    field.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

function submit(container: HTMLElement) {
  const button = Array.from(container.querySelectorAll("button")).find((b) =>
    (b.textContent ?? "").includes("Track phrase"),
  );
  act(() => {
    button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

/** An answer shaped like the engine's, with whatever the caller widened. */
function answer(
  overrides: Partial<PhraseAnswer["question"]> = {},
): PhraseAnswer {
  return {
    schema_version: 1,
    kind: "phrase_distribution",
    snapshot_id: "snap",
    question: {
      subject: {
        text: "promise",
        tokens: ["promise"],
        case_sensitive: false,
        boundary: "sentence",
        punctuation: "preserve",
        overlapping_matches: true,
        counted_forms: [["promise"]],
      },
      matching_profile: {
        version: 3,
        tokenizer: "spacy/en_core_web_sm",
        resolved_by_snapshot: true,
        source: "snapshot",
        normalize: false,
        match_lemma: false,
        match_nominalization: false,
        unavailable: [],
      },
      ...overrides,
    },
    summary: {
      occurrences: 112,
      word_tokens: 1000,
      occurrences_per_10000: 1120,
      eligible_documents: 2,
      matching_documents: 2,
      document_prevalence_percent: 100,
      dated_documents: 2,
      undated_documents: 0,
    },
    time: [],
    position: [],
    documents: [],
    tracks: [],
    evidence: {
      rows: [],
      total: 0,
      filtered_total: 0,
      offset: 0,
      limit: 50,
      truncated: false,
      filter: {
        year: null,
        document_id: null,
        position_start: null,
        position_end: null,
      },
      source_offsets: {
        status: "verified",
        unit: "unicode_code_point",
        browser_unit: "utf16_code_unit",
        verified: 0,
        total: 0,
        filtered_status: "verified",
        filtered_verified: 0,
        filtered_total: 0,
      },
    },
  } as unknown as PhraseAnswer;
}

function explorer(onTrack = vi.fn().mockResolvedValue(answer())) {
  const container = render(
    <PhraseExplorer
      ready
      onTrack={onTrack}
      onReadSource={vi.fn().mockResolvedValue(null)}
      savedQuestions={[]}
      onSaveQuestion={vi.fn().mockResolvedValue(null)}
      onOpenQuestion={vi.fn().mockResolvedValue(null)}
      onDuplicateQuestion={vi.fn().mockResolvedValue(null)}
      onDeleteQuestion={vi.fn().mockResolvedValue(false)}
      onPublishQuestion={vi.fn()}
      canPublish
    />,
  );
  return { container, onTrack };
}

describe("the options travel with the question", () => {
  it("offers all four", () => {
    const { container } = explorer();
    const text = container.textContent ?? "";
    expect(text).toContain("Match capitalization exactly");
    expect(text).toContain("Ignore typography");
    expect(text).toContain("Count every inflection");
    expect(text).toContain("Count nominalizations");
  });

  it("sends each one that is ticked", async () => {
    const { container, onTrack } = explorer();
    type(container, "promise");
    tick(container, "Count every inflection");
    tick(container, "Ignore typography");
    submit(container);
    await act(async () => {});

    const sent = onTrack.mock.calls.at(-1)?.[0] as PhraseRequest;
    expect(sent.match_lemma).toBe(true);
    expect(sent.normalize).toBe(true);
    expect(sent.match_nominalization).toBe(false);
  });

  it("asks nothing different until the question is tracked again", async () => {
    // Ticking a box changes what would be asked, not what is on screen. The
    // answer below still answers the question it answered.
    const { container, onTrack } = explorer();
    type(container, "promise");
    submit(container);
    await act(async () => {});
    const before = onTrack.mock.calls.length;

    tick(container, "Count every inflection");
    await act(async () => {});
    expect(onTrack.mock.calls.length).toBe(before);
    expect(container.textContent).toContain("The form has been changed");
  });
});

describe("what a widened question is counting", () => {
  it("lists the forms when the question was widened", async () => {
    const widened = answer({
      subject: {
        text: "promise",
        tokens: ["promise"],
        case_sensitive: false,
        boundary: "sentence",
        punctuation: "preserve",
        overlapping_matches: true,
        counted_forms: [["promise", "promised", "promises"]],
      },
      matching_profile: {
        version: 3,
        tokenizer: "spacy/en_core_web_sm",
        resolved_by_snapshot: true,
        source: "snapshot",
        normalize: false,
        match_lemma: true,
        match_nominalization: false,
        unavailable: [],
      },
    } as never);
    const { container } = explorer(vi.fn().mockResolvedValue(widened));
    type(container, "promise");
    submit(container);
    await act(async () => {});

    const listed = container.querySelector(".phrase-counted-list");
    expect(listed).not.toBeNull();
    expect(listed?.textContent).toContain("promised");
    expect(listed?.textContent).toContain("promises");
  });

  it("says nothing on a literal question", async () => {
    const { container } = explorer();
    type(container, "promise");
    submit(container);
    await act(async () => {});
    expect(container.querySelector(".phrase-counted")).toBeNull();
  });

  it("reports an option that could not be applied", async () => {
    // The defect this was found with: WordNet was returning nothing and the
    // option looked like it had worked. An option that cannot work and says
    // nothing is indistinguishable from one that is switched off.
    const blocked = answer({
      subject: {
        text: "govern",
        tokens: ["govern"],
        case_sensitive: false,
        boundary: "sentence",
        punctuation: "preserve",
        overlapping_matches: true,
        counted_forms: [["govern"]],
      },
      matching_profile: {
        version: 3,
        tokenizer: "spacy/en_core_web_sm",
        resolved_by_snapshot: true,
        source: "snapshot",
        normalize: false,
        match_lemma: false,
        match_nominalization: true,
        unavailable: ["Nominalizations were not matched: no WordNet data."],
      },
    } as never);
    const { container } = explorer(vi.fn().mockResolvedValue(blocked));
    type(container, "govern");
    submit(container);
    await act(async () => {});
    expect(container.textContent).toContain("Nominalizations were not matched");
  });
});

describe("an option is an edit", () => {
  const request: PhraseRequest = {
    text: "promise",
    case_sensitive: false,
    normalize: false,
    match_lemma: false,
    match_nominalization: false,
    position_bins: 10,
    evidence_offset: 0,
    evidence_limit: 50,
    evidence_year: null,
    evidence_document_id: null,
  };
  const draft = {
    text: "promise",
    case_sensitive: false,
    normalize: false,
    match_lemma: false,
    match_nominalization: false,
    position_bins: 10,
  };

  it("counts every option, not only the one that existed first", () => {
    for (const key of [
      "case_sensitive",
      "normalize",
      "match_lemma",
      "match_nominalization",
    ] as const) {
      expect(
        isDraftDifferent({ ...draft, [key]: true }, "", request, null),
      ).toBe(true);
    }
  });

  it("is an unsaved change against the saved record too", () => {
    const saved = {
      specification: {
        schema_version: 1,
        kind: "phrase_distribution",
        text: "promise",
        comparison_text: null,
        case_sensitive: false,
        normalize: false,
        match_lemma: false,
        match_nominalization: false,
        position_bins: 10,
        evidence_year: null,
        evidence_document_id: null,
      },
    } as SavedPhraseQuestion;
    expect(differsFromSaved(saved, request, null)).toBe(false);
    expect(
      differsFromSaved(saved, { ...request, match_lemma: true }, null),
    ).toBe(true);
  });

  it("treats a question saved before the options as the literal one", () => {
    // A record written before these existed has no keys for them, and the
    // question it was saved as is the one with nothing ticked.
    const old = { case_sensitive: false } as Partial<PhraseRequest>;
    expect(sameOptions(old, request)).toBe(true);
    expect(sameOptions(old, { ...request, normalize: true })).toBe(false);
  });
});
