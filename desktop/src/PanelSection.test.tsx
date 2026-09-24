// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { PanelParamField, panelDefaults, PanelSection } from "./PanelSection";
import { api, post } from "./api";
import type { PanelAnswer, PanelInfo } from "./api";

/**
 * The panels section, against a finished run.
 *
 * The control test here is the one this project has been bitten by before:
 * setting `input.value` directly does not trigger React's `onChange`, so a
 * test that does it "passes" while exercising nothing. The value has to go
 * through the native prototype's setter and the event dispatched for real,
 * and the test asserts its setup did what it claims before asserting the
 * behaviour (the typed number actually reached the drawn request).
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
function render(element: React.ReactElement) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  mounted.push({ root, container });
  return {
    container,
    rerender: (next: ReactElement) => act(() => root.render(next)),
  };
}

/** Set a controlled input's value the way a user does, through the native
 *  setter, so React's onChange actually fires. Setting `input.value` directly
 *  does not: React tracks the value it rendered, and a silent assignment is
 *  invisible to it — the trap behind a test that once passed by exercising
 *  nothing. */
function setInputValue(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )?.set;
  expect(setter, "the prototype value setter must exist").toBeDefined();
  setter!.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

const volcano: PanelInfo = {
  name: "keyness_volcano",
  title: "Keyness volcano",
  tool: "keyness",
  shape: "scatter_labelled",
  summary: "Effect size against strength of evidence.",
  requires: ["Word", "G2"],
  params: [
    {
      name: "label-top",
      label: "Words to label",
      type: "int",
      default: 18,
      required: false,
      help: "How many words to write beside their points.",
      choices: [],
      minimum: 0,
      maximum: 100,
    },
    {
      name: "label-by",
      label: "Label by",
      type: "choice",
      default: "evidence",
      required: false,
      help: "What picks the labelled words.",
      choices: ["evidence", "effect"],
      minimum: null,
      maximum: null,
    },
  ],
  notes: ["G2 measures evidence rather than effect size."],
};

const preparedAnswer: PanelAnswer = {
  ok: true,
  panel: "keyness_volcano",
  shape: "scatter_labelled",
  title: "Keyness volcano",
  subtitle: "",
  xLabel: "Log Ratio",
  yLabel: "G2",
  groups: [],
  caption: "caption",
  notes: [],
  annotations: [],
  diagnostics: [],
  table: { columns: ["Word"], rows: [], total: 0, truncated: false },
  marks: [
    {
      key: "freedom",
      label: "freedom",
      x: 2.3,
      y: 88,
      group: "",
      size: null,
      labelled: true,
      evidence: {
        scope: "terms",
        filters: [["Word", "freedom"]],
        count: 132,
        describe: "freedom: 120 in group A, 12 in group B",
      },
    },
  ],
};

describe("building controls from the declaration", () => {
  it("defaults come from the engine's declaration, not this file", () => {
    expect(panelDefaults(volcano)).toEqual({
      "label-top": 18,
      "label-by": "evidence",
    });
  });

  it("a numeric param renders as a number input with its declared bounds", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    act(() =>
      root.render(
        <PanelParamField
          param={volcano.params[0]}
          value={18}
          onChange={() => {}}
        />,
      ),
    );
    const input = container.querySelector("input") as HTMLInputElement;
    expect(input.type).toBe("number");
    expect(input.min).toBe("0");
    expect(input.max).toBe("100");
    act(() => root.unmount());
    container.remove();
  });

  it("a choice param offers exactly the declared choices", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    act(() =>
      root.render(
        <PanelParamField
          param={volcano.params[1]}
          value="evidence"
          onChange={() => {}}
        />,
      ),
    );
    const options = [...container.querySelectorAll("option")].map(
      (option) => option.value,
    );
    expect(options).toEqual(["evidence", "effect"]);
    act(() => root.unmount());
    container.remove();
  });

  it("typing a new number reaches onChange through React's own event path", () => {
    // The mandatory control test: the value goes through the prototype
    // setter, and the assertion is what React *did* with it — onChange fired
    // with the parsed number — rather than what the DOM briefly held. A
    // controlled input restores its rendered value; only the event is proof.
    const onChange = vi.fn();
    const container = document.createElement("div");
    const root = createRoot(container);
    act(() =>
      root.render(
        <PanelParamField
          param={volcano.params[0]}
          value={18}
          onChange={onChange}
        />,
      ),
    );
    const input = container.querySelector("input") as HTMLInputElement;
    act(() => setInputValue(input, "7"));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange).toHaveBeenCalledWith(7);
    act(() => root.unmount());
    container.remove();
  });
});

describe("the section on a run's record", () => {
  it("draws the declared primary figure when the run opens", async () => {
    vi.mocked(api).mockResolvedValue([volcano]);
    vi.mocked(post).mockResolvedValue(preparedAnswer);
    const { container } = render(<PanelSection projectId="p1" jobId="j1" />);
    await act(async () => {});
    expect(vi.mocked(post)).toHaveBeenCalledWith(
      "/projects/p1/jobs/j1/panels",
      { panel: "keyness_volcano", params: panelDefaults(volcano) },
    );
    expect(container.querySelector("svg[role=img]")).not.toBeNull();
  });

  it("opens the figure for the selected artifact when a run has several tables", async () => {
    const neighbour = {
      ...volcano,
      name: "word2vec_gensim_neighbours",
      title: "Nearest neighbours",
      requires: ["Word", "Neighbor", "Cosine"],
    };
    const projection = {
      ...volcano,
      name: "word2vec_gensim_tsne",
      title: "t-SNE projection",
      requires: ["Word", "X", "Y"],
    };
    vi.mocked(api).mockResolvedValue([neighbour, projection]);
    const { container } = render(
      <PanelSection
        projectId="p1"
        jobId="j1"
        selectedColumns={["Word", "X", "Y"]}
      />,
    );
    await act(async () => {});
    expect(
      container.querySelector('[role="tab"][aria-selected="true"]')
        ?.textContent,
    ).toBe("t-SNE projection");
  });

  it("uses the generic table view only after the server declares no panels", async () => {
    vi.mocked(api).mockResolvedValue([]);
    const { container } = render(
      <PanelSection
        projectId="p1"
        jobId="j1"
        fallback={<div data-testid="generic-table-view">Table chart</div>}
      />,
    );
    expect(
      container.querySelector('[data-testid="generic-table-view"]'),
    ).toBeNull();
    await act(async () => {});
    expect(
      container.querySelector('[data-testid="generic-table-view"]')
        ?.textContent,
    ).toBe("Table chart");
  });

  it("keeps the generic workbench opt-in when the run declares a panel", async () => {
    vi.mocked(api).mockResolvedValue([volcano]);
    const { container } = render(
      <PanelSection
        projectId="p1"
        jobId="j1"
        fallback={<div data-testid="generic-table-view">Table chart</div>}
      />,
    );
    await act(async () => {});
    expect(
      container.querySelector('[aria-label="Figures for this run"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="generic-table-view"]'),
    ).toBeNull();
    const expert = container.querySelector('button[aria-expanded="false"]');
    expect(expert?.textContent).toBe("Open generic chart workbench (expert)");
    act(() =>
      expert?.dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    expect(
      container.querySelector('[data-testid="generic-table-view"]')
        ?.textContent,
    ).toBe("Table chart");
  });

  it("offers the run's panels and draws one from the declared controls", async () => {
    vi.mocked(api).mockResolvedValue([volcano]);
    vi.mocked(post).mockResolvedValue(preparedAnswer);
    const { container } = render(<PanelSection projectId="p1" jobId="j1" />);
    await act(async () => {});
    // The picker is built from the declarations.
    const tabs = [...container.querySelectorAll('.panel-picker [role="tab"]')];
    expect(tabs.map((tab) => tab.textContent)).toEqual(["Keyness volcano"]);
    // Drawing posts exactly what the declaration defaulted to.
    const drawButton = container.querySelector(
      ".panel-draw",
    ) as HTMLButtonElement;
    act(() =>
      drawButton.dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    await act(async () => {});
    expect(vi.mocked(post)).toHaveBeenCalledWith(
      "/projects/p1/jobs/j1/panels",
      {
        panel: "keyness_volcano",
        params: { "label-top": 18, "label-by": "evidence" },
      },
    );
    // The figure rendered, with its caption.
    expect(container.querySelector("svg[role=img]")).not.toBeNull();
    expect(container.querySelector(".panel-caption")?.textContent).toBe(
      "caption",
    );
  });

  it("a figure drawn for one run never lands under another run", async () => {
    // The race: draw on run j1, switch to run j2 before the answer returns.
    // PanelSection is not remounted per run, and the draw is async, so an
    // unguarded answer from j1 would render under j2's heading -- with two
    // LDA runs open for comparison, the wrong run's topics, silently.
    vi.mocked(api).mockResolvedValue([volcano]);
    let answerJ1!: (answer: PanelAnswer) => void;
    vi.mocked(post).mockReturnValueOnce(
      new Promise<PanelAnswer>((resolve) => {
        answerJ1 = resolve;
      }) as ReturnType<typeof post>,
    );
    const view = render(<PanelSection projectId="p1" jobId="j1" />);
    await act(async () => {});
    const draw = view.container.querySelector(
      ".panel-draw",
    ) as HTMLButtonElement;
    act(() => draw.dispatchEvent(new MouseEvent("click", { bubbles: true })));

    view.rerender(<PanelSection projectId="p1" jobId="j2" />);
    await act(async () => {});
    // Control: j2's section is on screen, so an absent figure below means
    // the stale answer was refused -- not that nothing rendered at all.
    expect(view.container.querySelectorAll('[role="tab"]').length).toBe(1);

    await act(async () => {
      answerJ1({ ...preparedAnswer, caption: "from run j1" } as PanelAnswer);
    });
    expect(view.container.querySelector(".panel-caption")).toBeNull();
    expect(view.container.textContent).not.toContain("from run j1");
    // And j2 is not left stuck on "Drawing…" by j1's cancelled request.
    expect(
      (view.container.querySelector(".panel-draw") as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });

  it("a changed control value is what gets sent, not the default", async () => {
    vi.mocked(api).mockResolvedValue([volcano]);
    vi.mocked(post).mockResolvedValue(preparedAnswer);
    const { container } = render(<PanelSection projectId="p1" jobId="j1" />);
    await act(async () => {});
    const input = container.querySelector(
      'input[type="number"]',
    ) as HTMLInputElement;
    act(() => setInputValue(input, "5"));
    const drawButton = container.querySelector(
      ".panel-draw",
    ) as HTMLButtonElement;
    act(() =>
      drawButton.dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    await act(async () => {});
    // The control test's claim is what reached the engine: the typed value,
    // not the declaration's default.
    expect(vi.mocked(post).mock.calls.at(-1)![1]).toMatchObject({
      params: { "label-top": 5 },
    });
  });

  it("shows nothing at all when the run's tool has no panels", async () => {
    vi.mocked(api).mockResolvedValue([]);
    const { container } = render(<PanelSection projectId="p1" jobId="j1" />);
    await act(async () => {});
    expect(container.querySelector(".panel-section")).toBeNull();
    expect(container.textContent).toBe("");
  });

  it("a refusal renders as diagnostics, never a crash and never silence", async () => {
    vi.mocked(api).mockResolvedValue([volcano]);
    vi.mocked(post).mockResolvedValue({
      ok: false,
      diagnostics: [
        {
          severity: "ERROR",
          code: "PANEL_BAD_PARAM",
          message: "label-top must be at least 0 (got -3)",
        },
      ],
    });
    const { container } = render(<PanelSection projectId="p1" jobId="j1" />);
    await act(async () => {});
    const drawButton = container.querySelector(
      ".panel-draw",
    ) as HTMLButtonElement;
    act(() =>
      drawButton.dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    await act(async () => {});
    expect(container.textContent).toContain("PANEL_BAD_PARAM");
    expect(container.textContent).toContain("label-top");
    expect(container.querySelector("svg[role=img]")).toBeNull();
  });

  it("a network failure becomes a diagnostic too", async () => {
    vi.mocked(api).mockResolvedValue([volcano]);
    vi.mocked(post).mockRejectedValue(new Error("the engine did not answer"));
    const { container } = render(<PanelSection projectId="p1" jobId="j1" />);
    await act(async () => {});
    const drawButton = container.querySelector(
      ".panel-draw",
    ) as HTMLButtonElement;
    act(() =>
      drawButton.dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    await act(async () => {});
    expect(container.textContent).toContain("PANEL_REQUEST_FAILED");
  });

  it("clicking a mark shows the rows its evidence resolves to", async () => {
    vi.mocked(api).mockResolvedValue([volcano]);
    vi.mocked(post).mockResolvedValue({
      ...preparedAnswer,
      table: {
        columns: ["Word", "G2"],
        rows: [
          { Word: "freedom", G2: "88.4" },
          { Word: "the", G2: "41.2" },
        ],
        total: 2,
        truncated: false,
      },
    });
    const { container } = render(<PanelSection projectId="p1" jobId="j1" />);
    await act(async () => {});
    const drawButton = container.querySelector(
      ".panel-draw",
    ) as HTMLButtonElement;
    act(() =>
      drawButton.dispatchEvent(new MouseEvent("click", { bubbles: true })),
    );
    await act(async () => {});
    // Before the click: no filtered table.
    expect(container.querySelector(".panel-evidence")).toBeNull();
    const mark = container.querySelector('[role="button"]') as Element;
    act(() => mark.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    const evidence = container.querySelector(".panel-evidence");
    expect(evidence?.textContent).toContain("120 in group A");
    // The table is filtered to the rows the evidence's filters select.
    expect(evidence?.textContent).toContain("freedom");
    expect(evidence?.textContent).not.toContain("41.2");
  });
});
