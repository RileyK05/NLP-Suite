// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ImportPreview, ImportRequest, ProjectDetails } from "./details";
import { shapeLine, sourceNote } from "./details";

/**
 * The Corpus page's details against a fake engine: the shape line, editing a
 * cell, where a value came from, choosing the axis, adding a detail.
 */

type Call = { method: string; path: string; body?: unknown };
const calls: Call[] = [];
let state: ProjectDetails;

function fresh(): ProjectDetails {
  return {
    names: [
      {
        name: "Date",
        count: 3,
        distinct: 3,
        sources: ["filename"],
        sample_values: [],
      },
      {
        name: "Kind",
        count: 3,
        distinct: 1,
        sources: ["filename"],
        sample_values: ["sotu"],
      },
      {
        name: "Speaker",
        count: 3,
        distinct: 2,
        sources: ["filename", "user"],
        sample_values: [],
      },
    ],
    documents: {
      d1: {
        Date: { value: "1934-01-03", source: "filename", overridden: [] },
        Speaker: {
          value: "Franklin D Roosevelt",
          source: "filename",
          overridden: [],
        },
        Kind: { value: "sotu", source: "filename", overridden: [] },
      },
      d2: {
        Date: { value: "1953-02-02", source: "filename", overridden: [] },
        Speaker: {
          value: "Dwight D. Eisenhower",
          source: "user",
          overridden: [{ value: "Dwight D Eisenhower", source: "filename" }],
        },
        Kind: { value: "sotu", source: "filename", overridden: [] },
      },
      d3: {
        Date: { value: "2024-03-07", source: "filename", overridden: [] },
        Speaker: {
          value: "Joseph R Biden",
          source: "filename",
          overridden: [],
        },
        Kind: { value: "sotu", source: "filename", overridden: [] },
      },
    },
    template: {
      parts: ["Date", "Speaker", "Kind"],
      separator: "_",
      fits: 3,
      total: 3,
      misses: [],
      describe: "Date, then Speaker, then Kind, separated by _ (fits 3 of 3)",
    },
    settings: {
      axis: "auto",
      order_label: "Document",
      filename_detection: true,
      part_names: {},
    },
    revision: 4,
    axis: {
      kind: "time",
      noun: "Year",
      placed: 3,
      first: "1934-01-03",
      last: "2024-03-07",
    },
  };
}

function importAnswer(body: ImportRequest): ImportPreview {
  const target = body.targets?.party ?? "Party";
  if (!body.dry_run) {
    for (const id of ["d1", "d2", "d3"])
      state.documents[id].Party = {
        value: id === "d1" ? "Democratic" : "Republican",
        source: "csv",
        overridden: [],
      };
    state.revision += 1;
  }
  return {
    name_column: "president",
    matched_on: "Speaker",
    columns: [
      {
        column: "party",
        target,
        kind: "text",
        filled: 3,
        samples: ["Democratic", "Republican"],
        problem: "",
      },
    ],
    matched: 3,
    matched_by: { "by Speaker": 3 },
    unmatched_rows: ["Abraham Lincoln"],
    unmatched_row_count: 1,
    unmatched_documents: [],
    unmatched_document_count: 0,
    duplicate_rows: [],
    preview: [
      {
        document: DOCS[0].name,
        values: target ? { [target]: "Democratic" } : {},
      },
    ],
    diagnostics: [],
    applied: !body.dry_run,
    revision: state.revision,
  };
}

function route(method: string, path: string, body?: unknown): unknown {
  calls.push({ method, path, body });
  if (path.endsWith("/fields/import"))
    return importAnswer(body as ImportRequest);
  if (path.endsWith("/fields") && method === "GET") return state;
  if (path.endsWith("/fields") && method === "POST") {
    const [row] = (
      body as { rows: { document_id: string; name: string; value: string }[] }
    ).rows;
    const doc = (state.documents[row.document_id] ??= {});
    doc[row.name] = { value: row.value, source: "user", overridden: [] };
    if (!state.names.some((n) => n.name === row.name))
      state.names.push({
        name: row.name,
        count: 1,
        distinct: 1,
        sources: ["user"],
        sample_values: [],
      });
    state.revision += 1;
    return { revision: state.revision };
  }
  if (path.endsWith("/settings")) {
    const chosen = (body as { settings: ProjectDetails["settings"] }).settings;
    state.settings = chosen;
    if (chosen.axis === "none")
      state.axis = {
        kind: "none",
        noun: "",
        placed: 0,
        first: null,
        last: null,
      };
    state.revision += 1;
    return { settings: chosen, revision: state.revision };
  }
  if (method === "DELETE") return { deleted: true };
  throw new Error(`unrouted ${method} ${path}`);
}

vi.mock("./api", () => ({
  api: async (path: string) => route("GET", path),
  post: async (path: string, body: unknown) => route("POST", path, body),
  del: async (path: string) => route("DELETE", path),
}));

const { CorpusDetails } = await import("./CorpusDetails");

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
beforeEach(() => {
  calls.length = 0;
  state = fresh();
});
afterEach(() => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
});
const DOCS = [
  { id: "d1", name: "1934-01-03_franklin d roosevelt_sotu.txt" },
  { id: "d2", name: "1953-02-02_dwight d eisenhower_sotu.txt" },
  { id: "d3", name: "2024-03-07_joseph r biden_sotu.txt" },
];
function page() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(<CorpusDetails projectId="p" documents={DOCS} />));
  mounted.push({ root, container });
  return container;
}
async function until(check: () => boolean, what: string) {
  const deadline = Date.now() + 3000;
  while (!check()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await act(async () => new Promise((r) => setTimeout(r, 20)));
  }
}
const labelled = <T extends HTMLElement>(view: HTMLElement, label: string) => {
  const found = view.querySelector<T>(`[aria-label="${label}"]`);
  if (!found) throw new Error(`nothing labelled ${label}`);
  return found;
};
function type(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )!.set!;
  act(() => {
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
function choose(select: HTMLSelectElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLSelectElement.prototype,
    "value",
  )!.set!;
  act(() => {
    setter.call(select, value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
}
const blur = (input: HTMLElement) =>
  act(() => input.dispatchEvent(new FocusEvent("focusout", { bubbles: true })));

describe("document details on the Corpus page", () => {
  it("says what the corpus is in one line", async () => {
    const view = page();
    await until(() => !!view.querySelector(".corpus-shape"), "the shape line");
    expect(view.querySelector(".corpus-shape")!.textContent).toBe(
      "3 documents · dated 1934–2024 · lined up by time · 2 details: Kind, Speaker",
    );
    expect(view.textContent).toContain(
      "Read from the file names: Date, then Speaker, then Kind",
    );
  });

  it("saves a typed value as the user's, and says where each value came from", async () => {
    const view = page();
    await until(() => !!view.querySelector(".details-table"), "the grid");
    const typed = labelled<HTMLInputElement>(
      view,
      `Speaker of ${DOCS[1].name}`,
    );
    expect(typed.title).toBe(
      "typed by you · replaces from the file name: Dwight D Eisenhower",
    );
    expect(typed.className).toContain("source-user");
    const cell = labelled<HTMLInputElement>(view, `Speaker of ${DOCS[0].name}`);
    expect(cell.title).toBe("from the file name");
    type(cell, "Franklin Delano Roosevelt");
    blur(cell);
    await until(() => calls.some((c) => c.method === "POST"), "the save");
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({
      rows: [
        {
          document_id: "d1",
          name: "Speaker",
          value: "Franklin Delano Roosevelt",
          source: "user",
        },
      ],
      expected_revision: 4,
    });
    await until(
      () =>
        labelled<HTMLInputElement>(view, `Speaker of ${DOCS[0].name}`).title ===
        "typed by you",
      "reload",
    );
  });

  it("does not save a cell that did not change", async () => {
    const view = page();
    await until(() => !!view.querySelector(".details-table"), "the grid");
    blur(labelled<HTMLInputElement>(view, `Kind of ${DOCS[2].name}`));
    await act(async () => new Promise((r) => setTimeout(r, 30)));
    expect(calls.filter((c) => c.method === "POST")).toEqual([]);
  });

  it("lets the reader choose how the documents line up", async () => {
    const view = page();
    await until(() => !!view.querySelector(".details-table"), "the grid");
    const select = labelled<HTMLSelectElement>(
      view,
      "Line the documents up by",
    );
    expect(select.options[0].textContent).toBe("Choose for me (now: time)");
    choose(select, "none");
    await until(
      () =>
        view
          .querySelector(".corpus-shape")!
          .textContent!.includes("not lined up"),
      "the new axis",
    );
    expect(calls.find((c) => c.path.endsWith("/settings"))!.body).toMatchObject(
      {
        settings: { axis: "none" },
        expected_revision: 4,
      },
    );
  });

  it("adds a detail as an empty column that fills as values are typed", async () => {
    const view = page();
    await until(() => !!view.querySelector(".details-table"), "the grid");
    type(labelled<HTMLInputElement>(view, "New detail name"), "Party");
    act(() =>
      view.querySelector<HTMLFormElement>(".add-detail")!.requestSubmit(),
    );
    const party = labelled<HTMLInputElement>(view, `Party of ${DOCS[0].name}`);
    expect(party.value).toBe("");
    type(party, "Democratic");
    blur(party);
    await until(
      () => state.documents.d1.Party?.value === "Democratic",
      "the new value",
    );
  });

  it("previews a spreadsheet, lets a column be renamed, then imports it", async () => {
    // 1.4.2: nothing is stored until Import; the preview says which documents
    // the rows reach (here through Speaker) and which rows reach nothing.
    const view = page();
    await until(() => !!view.querySelector(".details-table"), "the grid");
    const input = labelled<HTMLInputElement>(
      view,
      "Spreadsheet of document details",
    );
    const sheet = new File(
      ["president,party\nharry s truman,Democratic\n"],
      "parties.csv",
      { type: "text/csv" },
    );
    Object.defineProperty(input, "files", { value: [sheet] });
    act(() => input.dispatchEvent(new Event("change", { bubbles: true })));
    await until(
      () => !!view.querySelector(".detail-import-preview"),
      "the preview",
    );
    const preview = view.querySelector(".detail-import-preview")!;
    expect(preview.textContent).toContain(
      "3 documents, through each one's Speaker",
    );
    expect(preview.textContent).toContain("Abraham Lincoln");
    const first = calls.find((call) => call.path.endsWith("/fields/import"))!;
    expect((first.body as ImportRequest).dry_run).toBe(true);
    expect((first.body as ImportRequest).content).toBe(
      btoa("president,party\nharry s truman,Democratic\n"),
    );
    expect(state.documents.d1.Party).toBeUndefined();

    // The column can become any detail the project has, or be left out.
    const becomes = labelled<HTMLSelectElement>(view, "What party becomes");
    expect([...becomes.options].map((option) => option.value)).toEqual([
      "Party",
      "Date",
      "Order",
      "Kind",
      "Speaker",
      "",
    ]);

    const importButton = [...preview.querySelectorAll("button")].find(
      (button) => button.textContent?.startsWith("Import details"),
    )!;
    act(() => importButton.click());
    await until(
      () => state.documents.d1.Party?.value === "Democratic",
      "the imported value",
    );
    const applied = calls.filter((call) =>
      call.path.endsWith("/fields/import"),
    );
    expect((applied.at(-1)!.body as ImportRequest).dry_run).toBe(false);
    expect((applied.at(-1)!.body as ImportRequest).expected_revision).toBe(4);
    await until(
      () => !view.querySelector(".detail-import-preview"),
      "the preview to close",
    );
    expect(view.textContent).toContain("Imported Party");
  });
});

describe("the words for details", () => {
  it("names an order axis by its noun", () => {
    const details = fresh();
    details.names = [];
    details.axis = {
      kind: "order",
      noun: "Chapter",
      placed: 61,
      first: "1",
      last: "61",
    };
    expect(shapeLine(details, 61)).toBe(
      "61 documents · chapter 1–61 · lined up by chapter",
    );
  });

  it("says when a value has nothing behind it", () => {
    expect(sourceNote(undefined)).toBe("No value");
  });
});
