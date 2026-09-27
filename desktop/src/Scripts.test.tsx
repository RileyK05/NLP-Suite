// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { KernelEvent, Notebook, NotebookCell } from "./notebooks";

/**
 * The Scripts page against a fake engine: notebooks, a kernel that answers
 * each cell with whatever events the test scripts, "Run and save", and the
 * chatbot dialog. CodeMirror is replaced by a textarea -- what is tested is
 * the page, not the editor.
 */

type Route = { method: "GET" | "POST"; path: string; body?: unknown };
const calls: Route[] = [];
let notebooks: Notebook[] = [];
let events: KernelEvent[] = [];
let exec = 0;
/** What the kernel does with a cell's code: the events it sends, without ids. */
let kernel: (code: string) => Array<Record<string, unknown>> = () => [];
let lintWarnings: { line: number; message: string }[] = [];

function route(method: "GET" | "POST", path: string, body?: unknown): unknown {
  calls.push({ method, path, body });
  const url = new URL(path, "http://x");
  const parts = url.pathname.split("/").filter(Boolean);
  if (url.pathname === "/script/reference")
    return {
      markdown: "",
      functions: [
        {
          name: "term_rates",
          kind: "function",
          signature: "nlp.term_rates(corpus, groups, *, per=1000)",
          doc: "How often each group of words is used.\n\nReturns a table.",
          members: "",
          insert: "nlp.term_rates(corpus, groups)",
        },
      ],
      templates: [
        {
          id: "word-group-over-time",
          name: "Word group over time",
          description: "How often a list of words is used.",
          request: "How often is immigration mentioned over time?",
          cells: [
            { kind: "code", text: "import nlpsuite as nlp\n\nnlp.show(rates)" },
          ],
        },
      ],
    };
  if (url.pathname === "/script/lint") return { warnings: lintWarnings };
  if (url.pathname.endsWith("/script-corpus"))
    return {
      name: "Speeches",
      documents: 4,
      words: 1200,
      years: [1946, 2007],
      columns: ["Document", "Year"],
      samples: { Year: ["1946", "1955"] },
    };
  if (parts[2] !== "notebooks") throw new Error(`unrouted ${method} ${path}`);
  const id = parts[3];
  const found = notebooks.find((n) => n.id === id);
  if (!id) {
    if (method === "GET") return notebooks;
    const input = body as { name: string; template?: string };
    const made = notebook(`nb${notebooks.length + 1}`, input.name, [
      cell("code", "import nlpsuite as nlp"),
    ]);
    notebooks = [...notebooks, made];
    return made;
  }
  const action = parts.slice(4).join("/");
  if (!action) {
    if (method === "GET") return found;
    const input = body as {
      name: string;
      content: Notebook["content"];
      expected_revision: number;
    };
    const saved = {
      ...found!,
      name: input.name,
      content: input.content,
      revision: found!.revision + 1,
    };
    notebooks = notebooks.map((n) => (n.id === id ? saved : n));
    return saved;
  }
  if (action === "kernel/exec") {
    exec += 1;
    const after = events.length;
    const code = (body as { code: string }).code;
    for (const event of kernel(code))
      events.push({ id: exec, ...event } as KernelEvent);
    return { exec, after };
  }
  if (action === "kernel/events") {
    const after = Number(url.searchParams.get("after"));
    return {
      events: events.slice(after),
      next: events.length,
      running: true,
      busy: null,
      ready: true,
    };
  }
  if (action === "kernel/stop") {
    exec = 0;
    events = [];
    return { stopped: true };
  }
  if (action === "run")
    return {
      id: "job1",
      state: "QUEUED",
      tool: "notebook",
      stage: "",
      diagnostics: [],
    };
  throw new Error(`unrouted ${method} ${path}`);
}

vi.mock("./api", () => ({
  api: async (path: string) => route("GET", path),
  post: async (path: string, body: unknown) => route("POST", path, body),
  artifactBlobUrl: async () => "blob:figure",
  artifactText: async (path: string) => `guide for ${path}`,
  download: vi.fn(async () => undefined),
  when: (value: string) => value,
}));

vi.mock("./CodeEditor", () => ({
  CodeEditor: ({
    value,
    onChange,
    label,
  }: {
    value: string;
    onChange: (value: string) => void;
    label: string;
  }) => (
    <textarea
      aria-label={label}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  ),
}));

const { Scripts } = await import("./Scripts");

let counter = 0;
function cell(
  kind: "code" | "markdown",
  source: string,
  pasted = false,
): NotebookCell {
  counter += 1;
  return {
    cell_type: kind,
    id: `c${counter}`,
    source,
    metadata: pasted ? { nlpsuite: { origin: "pasted" } } : {},
    ...(kind === "code" ? { execution_count: null, outputs: [] } : {}),
  };
}
function notebook(
  id: string,
  name: string,
  cells: NotebookCell[],
  createdBy: "app" | "import" = "app",
): Notebook {
  return {
    id,
    project_id: "p",
    name,
    revision: 1,
    created: "2026-09-25T10:00:00",
    updated: "2026-09-25T10:00:00",
    content: {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: { nlpsuite: { library: "1", created_by: createdBy } },
      cells,
    },
  };
}

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
const clipboard = vi.fn(async (_text: string) => undefined);
beforeEach(() => {
  calls.length = 0;
  events = [];
  exec = 0;
  kernel = () => [];
  lintWarnings = [];
  localStorage.clear();
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText: clipboard },
    configurable: true,
  });
  clipboard.mockClear();
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
const page = () =>
  render(
    <Scripts
      projectId="p"
      projectName="Speeches"
      parser="spacy"
      jobs={[]}
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
const click = (element: HTMLElement) => act(() => element.click());
function type(element: HTMLTextAreaElement | HTMLInputElement, value: string) {
  const proto = Object.getPrototypeOf(element);
  Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(element, value);
  act(() => {
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
const execs = () => calls.filter((c) => c.path.endsWith("/kernel/exec"));

describe("Scripts", () => {
  it("opens the project's notebook, runs a cell and shows what it printed", async () => {
    notebooks = [
      notebook("nb1", "Immigration", [
        cell("markdown", "# Immigration over time"),
        cell("code", "print('hello')"),
      ]),
    ];
    kernel = () => [
      { event: "stream", name: "stdout", text: "hello\n" },
      { event: "done", ok: true, seconds: 0.2, error: null },
    ];
    const container = page();
    await until(() => !!container.querySelector("h1, h2"), "the notebook");
    await until(
      () => container.textContent!.includes("Immigration over time"),
      "markdown",
    );
    await click(button(container, "Run cell 2"));
    await until(
      () => container.textContent!.includes("hello"),
      "the printed text",
    );
    await until(
      () => container.textContent!.includes("Done in 0.2s"),
      "the status",
    );
    expect(execs()[0].body).toEqual({
      cell: "2",
      code: "print('hello')",
      parser: "spacy",
    });
  });

  it("draws a shown table's chart and copies matplotlib code for it", async () => {
    notebooks = [notebook("nb1", "Rates", [cell("code", "nlp.show(rates)")])];
    kernel = () => [
      {
        event: "output",
        kind: "table",
        name: "rates",
        title: "Rate in each document",
        file: "0001_001_rates.csv",
        saved: false,
        preview: {
          columns: ["Year", "rate"],
          rows: [
            { Year: "1990", rate: "1.5" },
            { Year: "1991", rate: "2.5" },
          ],
          total: 2,
          truncated: false,
        },
        chart: {
          kind: "line",
          x: "Year",
          y: "rate",
          group: "",
          agg: "",
          top_n: 0,
        },
        chart_note: "",
      },
      { event: "done", ok: true, seconds: 0.1, error: null },
    ];
    const container = page();
    await until(
      () =>
        container.textContent!.includes("Rates") &&
        !!container.querySelector("textarea"),
      "the notebook",
    );
    await click(button(container, "Run cell 1"));
    await until(
      () => container.textContent!.includes("Rate in each document"),
      "the table",
    );
    expect(container.querySelector(".cell-chart svg")).not.toBeNull();
    await act(async () => button(container, /Copy matplotlib code/).click());
    expect(clipboard).toHaveBeenCalledTimes(1);
    expect(clipboard.mock.calls[0][0]).toContain('pd.read_csv("rates.csv")');
    await click(button(container, "Table"));
    expect(container.querySelector("table")).not.toBeNull();
  });

  it("shows an error by its type, with the user's line on request", async () => {
    notebooks = [notebook("nb1", "Broken", [cell("code", "1 / 0")])];
    kernel = () => [
      {
        event: "done",
        ok: false,
        seconds: 0,
        error: {
          type: "ZeroDivisionError",
          message: "division by zero",
          trace: ["cell 1, line 1: 1 / 0"],
        },
      },
    ];
    const container = page();
    await until(() => !!container.querySelector("textarea"), "the notebook");
    await click(button(container, "Run cell 1"));
    await until(() => !!container.querySelector(".cell-error"), "the error");
    expect(container.querySelector(".cell-error")!.textContent).toContain(
      "ZeroDivisionError division by zero",
    );
    expect(container.textContent).not.toContain("cell 1, line 1");
    await click(button(container, "Where"));
    expect(container.textContent).toContain("cell 1, line 1: 1 / 0");
  });

  it("Run all stops at the first cell that fails", async () => {
    notebooks = [
      notebook("nb1", "Three", [
        cell("code", "a = 1"),
        cell("code", "raise ValueError('bad')"),
        cell("code", "never"),
      ]),
    ];
    kernel = (code) => [
      code.startsWith("raise")
        ? {
            event: "done",
            ok: false,
            seconds: 0,
            error: { type: "ValueError", message: "bad", trace: [] },
          }
        : { event: "done", ok: true, seconds: 0, error: null },
    ];
    const container = page();
    await until(
      () => container.querySelectorAll("textarea").length === 3,
      "three cells",
    );
    await click(button(container, /Run all/));
    // Not the text "ValueError": the cell's own source contains it.
    await until(() => !!container.querySelector(".cell-error"), "the failure");
    await act(async () => new Promise((r) => setTimeout(r, 500)));
    expect(execs().map((c) => (c.body as { code: string }).code)).toEqual([
      "a = 1",
      "raise ValueError('bad')",
    ]);
    const third = container.querySelector('[aria-label="Cell 3"]')!;
    expect(third.textContent).toContain("Not run");
  });

  it("Run all stops above pasted code nobody has read", async () => {
    notebooks = [
      notebook("nb1", "Mixed", [
        cell("code", "a = 1"),
        cell("code", "import os", true),
        cell("code", "b = 2"),
      ]),
    ];
    kernel = () => [{ event: "done", ok: true, seconds: 0, error: null }];
    const container = page();
    await until(
      () => container.querySelectorAll("textarea").length === 3,
      "three cells",
    );
    await click(button(container, /Run all/));
    await until(
      () => container.textContent!.includes("Done in 0.0s"),
      "the first cell",
    );
    await act(async () => new Promise((r) => setTimeout(r, 500)));
    expect(execs().map((c) => (c.body as { code: string }).code)).toEqual([
      "a = 1",
    ]);
    expect(container.textContent).toContain("Run all stops above cell 2");
  });

  it("Stop ends a cell that would run forever, and says what it cost", async () => {
    notebooks = [notebook("nb1", "Loop", [cell("code", "while True: pass")])];
    kernel = () => [];
    const container = page();
    await until(() => !!container.querySelector("textarea"), "the notebook");
    await click(button(container, "Run cell 1"));
    await until(() => container.textContent!.includes("Running"), "running");
    await act(async () =>
      button(container, /Stop \(clears variables\)/).click(),
    );
    await until(
      () => container.querySelector(".cell-status")!.textContent === "Stopped",
      "stopped",
    );
    expect(calls.some((c) => c.path.endsWith("/kernel/stop"))).toBe(true);
    expect(container.textContent).toContain("clears the notebook's variables");
  });

  it("warns about an imported notebook before anything runs", async () => {
    notebooks = [
      notebook("nb1", "From a colleague", [cell("code", "x = 1")], "import"),
    ];
    const container = page();
    await until(
      () => container.textContent!.includes("came from outside the app"),
      "the banner",
    );
    expect(execs()).toHaveLength(0);
  });

  it("adds chatbot code as a marked cell that is read before it runs", async () => {
    notebooks = [
      notebook("nb1", "Help", [cell("code", "import nlpsuite as nlp")]),
    ];
    lintWarnings = [
      {
        line: 2,
        message:
          "Imports os, which reaches files and programs on this computer.",
      },
    ];
    const container = page();
    await until(() => !!container.querySelector("textarea"), "the notebook");
    await click(button(container, /Get help from an AI chatbot/));
    await until(
      () => container.textContent!.includes("characters"),
      "the guide's size",
    );
    await act(async () => button(container, /Copy the guide/).click());
    expect(clipboard.mock.calls[0][0]).toContain(
      "/projects/p/script-guide?version=short&corpus=true",
    );
    type(
      container.querySelector(
        'textarea[aria-label="Code from the chatbot"]',
      ) as HTMLTextAreaElement,
      "```python\nimport nlpsuite as nlp\nimport os\n```",
    );
    await click(button(container, /Add as a new cell/));
    await until(
      () =>
        container.querySelectorAll('[aria-label^="Cell "][aria-label$=" code"]')
          .length === 2,
      "the new cell",
    );
    const added = container.querySelector(
      'textarea[aria-label="Cell 2 code"]',
    ) as HTMLTextAreaElement;
    expect(added.value).toBe("import nlpsuite as nlp\nimport os");
    expect(execs()).toHaveLength(0);
    await until(
      () => container.textContent!.includes("Line 2: Imports os"),
      "the lint warning",
    );
    expect(container.textContent).toContain("pasted in from outside the app");
    await click(button(container, "I have read it"));
    expect(container.textContent).not.toContain(
      "pasted in from outside the app",
    );
    // The edit is saved: the cell no longer carries the mark.
    await until(
      () =>
        calls.some(
          (c) => c.method === "POST" && c.path === "/projects/p/notebooks/nb1",
        ),
      "the save",
      4000,
    );
    const saved = calls
      .filter(
        (c) => c.method === "POST" && c.path === "/projects/p/notebooks/nb1",
      )
      .at(-1)!;
    const cells = (saved.body as { content: { cells: NotebookCell[] } }).content
      .cells;
    expect(cells[1].metadata).toEqual({});
  });

  it("opened from Learn with no notebook, pasted code gets a notebook of its own", async () => {
    notebooks = [];
    const opened = vi.fn();
    const container = render(
      <Scripts
        projectId="p"
        projectName="Speeches"
        parser="spacy"
        jobs={[]}
        onOpenJob={() => undefined}
        openGuide
        onGuideOpened={opened}
      />,
    );
    await until(
      () => container.textContent!.includes("Paste the code you got back"),
      "the dialog",
    );
    expect(opened).toHaveBeenCalled();
    type(
      container.querySelector(
        'textarea[aria-label="Code from the chatbot"]',
      ) as HTMLTextAreaElement,
      "import nlpsuite as nlp\nnlp.corpus().documents",
    );
    await click(button(container, /Add as a new cell/));
    await until(
      () =>
        calls.some(
          (c) => c.method === "POST" && c.path === "/projects/p/notebooks/nb1",
        ),
      "the pasted cell saved into the new notebook",
    );
    const created = calls.find(
      (c) => c.method === "POST" && c.path === "/projects/p/notebooks",
    )!;
    expect(created.body).toEqual({ name: "From a chatbot" });
    const saved = calls
      .filter(
        (c) => c.path === "/projects/p/notebooks/nb1" && c.method === "POST",
      )
      .at(-1)!;
    const cells = (saved.body as { content: { cells: NotebookCell[] } }).content
      .cells;
    expect(cells.map((c) => c.source)).toEqual([
      "import nlpsuite as nlp\nnlp.corpus().documents",
    ]);
    expect(cells[0].metadata).toEqual({ nlpsuite: { origin: "pasted" } });
  });

  it("starts a notebook from a template", async () => {
    notebooks = [];
    const container = page();
    await until(
      () => container.textContent!.includes("No notebooks in Speeches yet"),
      "the empty list",
    );
    await until(
      () => container.textContent!.includes("Library"),
      "the reference",
    );
    await click(button(container, /Start from a template/));
    const radio = container.querySelector(
      'input[value="word-group-over-time"]',
    ) as HTMLInputElement;
    await click(radio);
    await click(button(container, "Create"));
    await until(
      () =>
        calls.some(
          (c) => c.method === "POST" && c.path === "/projects/p/notebooks",
        ),
      "the create",
    );
    expect(
      calls.find(
        (c) => c.method === "POST" && c.path === "/projects/p/notebooks",
      )!.body,
    ).toEqual({
      name: "Word group over time",
      template: "word-group-over-time",
    });
  });

  it("shows the library and what the corpus's columns are called", async () => {
    notebooks = [notebook("nb1", "Any", [cell("code", "")])];
    const container = page();
    await until(
      () => container.textContent!.includes("Your corpus"),
      "the corpus summary",
    );
    expect(container.textContent).toContain(
      "4 documents, 1,200 words, 1946–2007",
    );
    await click(button(container, /term_rates/));
    expect(container.textContent).toContain(
      "nlp.term_rates(corpus, groups, *, per=1000)",
    );
  });
});
