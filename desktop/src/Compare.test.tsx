// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { Job, Project } from "./api";
import {
  definitionProblems,
  detailValues,
  formatFocus,
  newDefinition,
  parseFocus,
  type Comparison,
  type ComparisonRun,
  type Definition,
  type Preview,
} from "./comparisons";

/**
 * The Compare page against a fake engine: saved comparisons, the preview that
 * counts each side, running one, and what a finished run says above its
 * figures. The figures themselves are PanelSection's, tested there.
 */

type Route = { method: "GET" | "POST"; path: string; body?: unknown };
const calls: Route[] = [];
let saved: Comparison[] = [];
let runs: ComparisonRun[] = [];
let preview: (definition: Definition) => Preview = () => ({
  sides: [],
  alignments: [],
  dated: false,
});
let testsRows: Record<string, string>[] = [];
let carryoverRows: Record<string, string>[] = [];

const PROJECTS = [
  { id: "p", name: "State of the Union" },
  { id: "q", name: "Inaugurals" },
] as Project[];

function route(method: "GET" | "POST", path: string, body?: unknown): unknown {
  calls.push({ method, path, body });
  const parts = path.split("/").filter(Boolean);
  if (parts[2] === "fields")
    return {
      names: [],
      documents: {
        d1: { Speaker: { value: "Truman", source: "filename" } },
        d2: { Speaker: { value: "Eisenhower", source: "filename" } },
        d3: { Kind: { value: "sotu", source: "user" } },
      },
    };
  if (parts[2] === "jobs") {
    if (parts[4] === "results")
      return {
        artifacts: [
          { path: "contrast_keyness.csv" },
          { path: "contrast_measure_tests.csv" },
          { path: "contrast_carryover.csv" },
        ],
      };
    if (parts[4] === "artifacts" && parts[5] === "2")
      return { columns: [], rows: carryoverRows };
    return { columns: ["Group", "Reading"], rows: testsRows };
  }
  if (parts[2] === "documents")
    return {
      name: parts[3] === "d1" ? "Before speech" : "After speech",
      found: true,
      excerpt:
        parts[3] === "d1"
          ? "Before context. We protect our families today. After context."
          : "Intro context. Families deserve protection today. Closing context.",
    };
  if (parts[2] !== "comparisons") throw new Error(`unrouted ${method} ${path}`);
  const id = parts[3];
  if (!id) {
    if (method === "GET") return saved;
    const input = body as { name: string; definition: Definition };
    const made = comparison(
      `c${saved.length + 1}`,
      input.name,
      input.definition,
    );
    saved = [...saved, made];
    return made;
  }
  if (id === "preview")
    return preview((body as { definition: Definition }).definition);
  if (id === "runs") return runs;
  const action = parts[4];
  const found = saved.find((item) => item.id === id)!;
  if (!action) {
    const input = body as { name: string; definition: Definition };
    const stored = {
      ...found,
      name: input.name,
      definition: input.definition,
      revision: found.revision + 1,
    };
    saved = saved.map((item) => (item.id === id ? stored : item));
    return stored;
  }
  if (action === "run")
    return {
      id: "job1",
      state: "QUEUED",
      tool: "contrast",
      stage: "",
      diagnostics: [],
    };
  throw new Error(`unrouted ${method} ${path}`);
}

vi.mock("./api", () => ({
  api: async (path: string) => route("GET", path),
  post: async (path: string, body: unknown) => route("POST", path, body),
  when: (value: string) => `on ${value}`,
}));
vi.mock("./PanelSection", () => ({
  PanelSection: ({ jobId }: { jobId: string }) => (
    <div>figures for {jobId}</div>
  ),
}));

const { Compare } = await import("./Compare");

function comparison(
  id: string,
  name: string,
  definition: Definition,
): Comparison {
  return {
    id,
    name,
    definition,
    revision: 1,
    created: "2026-09-26T10:00:00",
    updated: "2026-09-26T10:00:00",
  };
}
const crossProject = (): Definition => ({
  ...newDefinition("p"),
  sides: [
    { name: "SOTU", project_id: "p", selection: null },
    { name: "Inaugural", project_id: "q", selection: null },
  ],
});
const counted = (definition: Definition): Preview => ({
  sides: definition.sides.map((side) => ({
    name: side.name,
    documents: side.project_id === "p" ? 87 : 31,
    words: side.project_id === "p" ? 1_600_000 : 200_000,
    first_date: "1946-01-21",
    last_date: "2019-02-05",
    details: {
      Speaker: { count: 14, examples: ["Barack Obama", "Bill Clinton"] },
      Kind: { count: 1, examples: ["sotu"] },
    },
  })),
  alignments: [
    {
      name: "Speaker",
      shared_values: 14,
      examples: ["Barack Obama", "Bill Clinton"],
    },
  ],
  dated: true,
});

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
beforeEach(() => {
  calls.length = 0;
  saved = [];
  runs = [];
  testsRows = [];
  carryoverRows = [];
  preview = counted;
});
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
const page = (jobs: Job[] = []) =>
  render(
    <Compare
      projectId="p"
      projects={PROJECTS}
      parser="spacy"
      jobs={jobs}
      onOpenJob={() => undefined}
    />,
  );

async function until(check: () => boolean, what: string, ms = 4000) {
  const deadline = Date.now() + ms;
  while (!check()) {
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await act(async () => new Promise((r) => setTimeout(r, 40)));
  }
}
function button(
  container: HTMLElement,
  name: string | RegExp,
): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((b) => {
    const label = b.getAttribute("aria-label") ?? b.textContent ?? "";
    return typeof name === "string" ? label.trim() === name : name.test(label);
  });
  if (!found) throw new Error(`no button ${name}`);
  return found;
}
function labelled<T extends HTMLElement>(
  container: HTMLElement,
  label: string,
): T {
  const found = container.querySelector<T>(`[aria-label="${label}"]`);
  if (!found) throw new Error(`nothing labelled ${label}`);
  return found;
}
function change(
  element: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement,
  value: string,
) {
  const prototype = Object.getPrototypeOf(element) as object;
  const setter = Object.getOwnPropertyDescriptor(prototype, "value")!.set!;
  act(() => {
    setter.call(element, value);
    element.dispatchEvent(
      new Event(element instanceof HTMLSelectElement ? "change" : "input", {
        bubbles: true,
      }),
    );
  });
}
const click = (element: HTMLElement) => act(() => element.click());

describe("the Compare page", () => {
  it("starts a comparison with two sides and counts what each holds", async () => {
    const view = page();
    await until(
      () => view.textContent!.includes("No comparisons yet"),
      "the empty list",
    );
    click(button(view, /New comparison/));
    await until(
      () => view.querySelectorAll(".side-card").length === 2,
      "two side cards",
    );
    expect(labelled<HTMLInputElement>(view, "Side 1 name").value).toBe(
      "Side A",
    );
    await until(
      () =>
        view.textContent!.includes("87 documents, 1,600,000 words, 1946–2019"),
      "the preview",
    );
    expect(view.textContent).toContain("14 Speaker values");
    // One value is named, not counted: "1 Kind" told the reader nothing.
    expect(view.textContent).toContain("Kind: sotu");
    // A third side, up to six.
    click(button(view, /Add a side/));
    expect(view.querySelectorAll(".side-card").length).toBe(3);
    expect(labelled<HTMLInputElement>(view, "Side 3 name").value).toBe(
      "Side C",
    );
    expect(view.textContent).toContain("Unsaved changes");
  });

  it("offers only what every side can be lined up by", async () => {
    saved = [comparison("c1", "SOTU vs Inaugural", crossProject())];
    const view = page();
    await until(
      () => view.textContent!.includes("Speaker: 14 on every side"),
      "the alignment choices",
    );
    const select = labelled<HTMLSelectElement>(view, "Line the sides up by");
    const options = [...select.options].map((o) => o.textContent);
    expect(options).toContain("Decade");
    change(select, "field:Speaker");
    click(button(view, /Save/));
    await until(() => saved[0].revision === 2, "the save");
    expect(saved[0].definition.alignment).toBe("field:Speaker");
  });

  it("names the side it cannot read and will not run until it is fixed", async () => {
    saved = [comparison("c1", "Broken", crossProject())];
    preview = (definition) => ({
      ...counted(definition),
      sides: [
        counted(definition).sides[0],
        {
          name: "Inaugural",
          error: "Side 'Inaugural': its project is not in this workspace.",
        },
      ],
    });
    const view = page();
    await until(
      () => view.textContent!.includes("its project is not in this workspace"),
      "the side's error",
    );
    const compare = button(view, /^\s*Compare$/);
    expect(compare.disabled).toBe(true);
    expect(compare.title).toContain("Inaugural");
    expect(view.querySelector(".side-summary.error")).not.toBeNull();
  });

  it("turns focus words into word groups and asks for them when chosen", async () => {
    saved = [comparison("c1", "Focus", crossProject())];
    const view = page();
    await until(
      () => view.querySelectorAll(".side-card").length === 2,
      "the sides",
    );
    const focus = view.querySelector<HTMLLabelElement>(".compare-methods")!;
    const box = [...focus.querySelectorAll("label")].find((l) =>
      l.textContent!.includes("Focus word rates"),
    )!;
    click(box.querySelector("input")!);
    await until(
      () => view.textContent!.includes("Focus word rates need focus words."),
      "the problem",
    );
    change(
      labelled<HTMLTextAreaElement>(view, "Focus words"),
      "immigration: Immigration, border\neconomy: jobs",
    );
    await until(
      () => !view.textContent!.includes("need focus words"),
      "the problem to clear",
    );
    click(button(view, /Save/));
    await until(() => saved[0].revision === 2, "the save");
    expect(saved[0].definition.focus).toEqual({
      immigration: ["immigration", "border"],
      economy: ["jobs"],
    });
    expect(saved[0].definition.methods).toContain("focus");
  });

  it("offers carried-over passages and keeps the meaning map unavailable", async () => {
    saved = [comparison("c1", "Methods", crossProject())];
    const view = page();
    await until(
      () => view.querySelectorAll(".method").length > 0,
      "the methods",
    );
    const unavailable = [...view.querySelectorAll(".method.unavailable")];
    expect(unavailable.map((l) => l.textContent).join(" ")).toContain(
      "Meaning map",
    );
    expect(unavailable.map((l) => l.textContent).join(" ")).not.toContain(
      "Carried-over passages",
    );
    for (const label of view.querySelectorAll(".method.unavailable"))
      expect(label.querySelector("input")!.disabled).toBe(true);
    const carryover = [...view.querySelectorAll(".method")].find((label) =>
      label.textContent?.includes("Carried-over passages"),
    )!;
    expect(carryover.classList.contains("unavailable")).toBe(false);
    expect(carryover.querySelector("input")!.disabled).toBe(false);
  });

  it("saves unsaved changes before it runs, with the chosen parser", async () => {
    saved = [comparison("c1", "Run me", crossProject())];
    const view = page();
    await until(
      () => view.textContent!.includes("87 documents"),
      "the preview",
    );
    change(labelled<HTMLInputElement>(view, "Comparison name"), "Run me now");
    await until(
      () => !button(view, /^\s*Compare$/).disabled,
      "Compare to be enabled",
    );
    click(button(view, /^\s*Compare$/));
    await until(() => calls.some((c) => c.path.endsWith("/run")), "the run");
    const order = calls
      .filter((c) => c.method === "POST" && !c.path.endsWith("/preview"))
      .map((c) => c.path);
    expect(order).toEqual([
      "/projects/p/comparisons/c1",
      "/projects/p/comparisons/c1/run",
    ]);
    expect(calls.find((c) => c.path.endsWith("/run"))!.body).toEqual({
      parser: "spacy",
    });
    expect(saved[0].name).toBe("Run me now");
    await until(() => view.textContent!.includes("Waiting…"), "the run status");
  });

  it("says what a finished run found before its figures", async () => {
    saved = [comparison("c1", "Done", crossProject())];
    testsRows = [
      { Group: "Barack Obama", Reading: "per speaker" },
      {
        Group: "All groups",
        Reading: "SOTU has longer words than Inaugural in 12 of 14 groups.",
      },
    ];
    carryoverRows = [
      {
        Group: "Barack Obama",
        Similarity: "0.91",
        "Project ID A": "p",
        "Document ID A": "d1",
        "Document A": "Before speech",
        "Sentence ID A": "12",
        "Passage A": "We protect our families today.",
        "Project ID B": "q",
        "Document ID B": "d2",
        "Document B": "After speech",
        "Sentence ID B": "8",
        "Passage B": "Families deserve protection today.",
      },
    ];
    runs = [
      {
        id: "job9",
        state: "DONE",
        tool: "contrast",
        stage: "",
        created: "2026-09-26",
        finished: "2026-09-26",
        run_dir: "runs/job9",
        comparison_id: "c1",
        comparison_name: "Done",
        diagnostics: [
          {
            severity: "INFO",
            code: "CONTRAST_NOTE",
            message: "SOTU and Inaugural are different settings.",
          },
        ],
      } as unknown as ComparisonRun,
    ];
    const view = page();
    await until(
      () => view.textContent!.includes("in 12 of 14 groups"),
      "the reading",
    );
    expect(view.textContent).toContain("Read these first");
    expect(view.textContent).toContain("different settings");
    expect(view.textContent).not.toContain("per speaker");
    expect(view.textContent).toContain("figures for job9");
    expect(view.textContent).toContain("Compared on 2026-09-26.");
    await until(
      () => view.textContent!.includes("Read in context"),
      "the carried-over passage",
    );
    click(button(view, "Read in context"));
    await until(
      () => view.textContent!.includes("Sentence IDs 12 and 8"),
      "both source contexts",
    );
    expect(view.textContent).toContain("Before context.");
    expect(view.textContent).toContain("Closing context.");
    expect(calls.map((call) => call.path)).toContain(
      "/projects/p/documents/d1/context",
    );
    expect(calls.map((call) => call.path)).toContain(
      "/projects/q/documents/d2/context",
    );
    expect(calls).toContainEqual({
      method: "POST",
      path: "/projects/p/documents/d1/context",
      body: { passage: "We protect our families today." },
    });
  });
});

describe("comparison definitions", () => {
  it("reads focus words the way the lexicon tool writes them", () => {
    expect(
      parseFocus(
        "immigration: Immigrant, border; economy: jobs, wages\nfreedom",
      ),
    ).toEqual({
      immigration: ["immigrant", "border"],
      economy: ["jobs", "wages"],
      focus: ["freedom"],
    });
    expect(parseFocus(" ; \n")).toEqual({});
    const groups = { a: ["x", "y"], b: ["z"] };
    expect(parseFocus(formatFocus(groups))).toEqual(groups);
  });

  it("says what is wrong before anything is sent", () => {
    const definition = newDefinition("p");
    expect(definitionProblems(definition)).toEqual([]);
    definition.sides[1].name = " side a ";
    definition.methods = [];
    expect(definitionProblems(definition)).toEqual([
      "Two sides have the same name.",
      "Choose at least one method.",
    ]);
    definition.sides[0].name = "";
    definition.methods = ["focus"];
    expect(definitionProblems(definition)).toContain("Give every side a name.");
    expect(definitionProblems(definition)).toContain(
      "Focus word rates need focus words.",
    );
    definition.methods = ["carryover"];
    expect(definitionProblems(definition)).toContain(
      "Carried-over passages need focus words.",
    );
  });

  it("lists each detail's values, leaving out the date", () => {
    expect(
      detailValues({
        names: [],
        documents: {
          a: {
            Speaker: { value: "Truman", source: "filename" },
            Date: { value: "1946", source: "filename" },
          },
          b: { Speaker: { value: "Adams", source: "user" } },
          c: { Speaker: { value: "Truman", source: "filename" } },
        },
      }),
    ).toEqual({ Speaker: ["Adams", "Truman"] });
  });
});
