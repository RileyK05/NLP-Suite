// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { PanelTabs } from "./PanelTabs";
import type { PanelAnswer, PanelInfo } from "./api";

/**
 * A live answer's figures as tabs.
 *
 * The draws here resolve to refusals whose message names what was asked
 * for, so each test can read which answer landed without drawing a figure.
 * The stale-answer test is the one that matters most: a tab switched while a
 * draw is out must not have the old figure land under the new tab.
 */

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
  return {
    container,
    rerender: (next: ReactElement) => act(() => root.render(next)),
  };
}

const panel = (name: string, title: string): PanelInfo => ({
  name,
  title,
  tool: "lda_gensim",
  shape: "ranked_bars",
  summary: `${title} summary`,
  requires: [],
  params: [
    {
      name: "top-n",
      label: "Top",
      type: "int",
      default: 10,
      required: false,
      help: "How many.",
      choices: [],
      minimum: 1,
      maximum: 50,
    },
  ],
  notes: [],
  question: `Question for ${title}?`,
});
const relevance = panel("lda_relevance", "Relevance");
const flow = panel("lda_flow", "Topic flow");

const answer = (text: string): PanelAnswer => ({
  ok: false,
  diagnostics: [{ severity: "INFO", code: "TEST", message: text }],
});

async function flush() {
  await act(async () => {
    await Promise.resolve();
  });
}

describe("PanelTabs", () => {
  it("offers one tab per figure and shows the first answer with its question", () => {
    const { container } = render(
      <PanelTabs
        panels={[relevance, flow]}
        first={answer("first drawn")}
        draw={vi.fn()}
        projectId="p1"
        resetKey="a1"
      />,
    );
    const tabs = [...container.querySelectorAll('[role="tab"]')];
    expect(tabs.map((tab) => tab.textContent)).toEqual([
      "Relevance",
      "Topic flow",
    ]);
    expect(container.textContent).toContain("first drawn");
    expect(container.textContent).toContain("Question for Relevance?");
  });

  it("draws another figure with its own defaults when its tab is chosen", async () => {
    const draw = vi.fn(
      async (info: PanelInfo, params: Record<string, unknown>) =>
        answer(`drew ${info.name} top ${params["top-n"]}`),
    );
    const { container } = render(
      <PanelTabs
        panels={[relevance, flow]}
        first={answer("first drawn")}
        draw={draw}
        projectId="p1"
        resetKey="a1"
      />,
    );
    const tab = [
      ...container.querySelectorAll('[role="tab"]'),
    ][1] as HTMLElement;
    act(() => tab.click());
    await flush();
    expect(draw).toHaveBeenCalledWith(flow, { "top-n": 10 });
    expect(container.textContent).toContain("drew lda_flow top 10");
    expect(tab.getAttribute("aria-selected")).toBe("true");
  });

  it("drops an answer that arrives after the reader moved to another tab", async () => {
    let release: (value: PanelAnswer) => void = () => {};
    const slow = new Promise<PanelAnswer>((resolve) => (release = resolve));
    const draw = vi.fn((info: PanelInfo) =>
      info.name === "lda_flow"
        ? slow
        : Promise.resolve(answer("relevance again")),
    );
    const { container } = render(
      <PanelTabs
        panels={[relevance, flow]}
        first={answer("first drawn")}
        draw={draw}
        projectId="p1"
        resetKey="a1"
      />,
    );
    const [first, second] = [
      ...container.querySelectorAll('[role="tab"]'),
    ] as HTMLElement[];
    act(() => second.click());
    act(() => first.click());
    await flush();
    await act(async () => release(answer("late flow answer")));
    expect(container.textContent).toContain("relevance again");
    expect(container.textContent).not.toContain("late flow answer");
  });

  it("opens the figure the reading asked for by its question", async () => {
    const draw = vi.fn(async (info: PanelInfo) => answer(`drew ${info.name}`));
    const view = render(
      <PanelTabs
        panels={[relevance, flow]}
        first={answer("first drawn")}
        draw={draw}
        projectId="p1"
        resetKey="a1"
        requested={null}
      />,
    );
    view.rerender(
      <PanelTabs
        panels={[relevance, flow]}
        first={answer("first drawn")}
        draw={draw}
        projectId="p1"
        resetKey="a1"
        requested={{ panel: "lda_flow", nonce: 1 }}
      />,
    );
    await flush();
    expect(view.container.textContent).toContain("drew lda_flow");
  });

  it("starts over on a new answer", async () => {
    const draw = vi.fn(async () => answer("second tab drawn"));
    const view = render(
      <PanelTabs
        panels={[relevance, flow]}
        first={answer("first drawn")}
        draw={draw}
        projectId="p1"
        resetKey="a1"
      />,
    );
    act(() =>
      (
        [...view.container.querySelectorAll('[role="tab"]')][1] as HTMLElement
      ).click(),
    );
    await flush();
    view.rerender(
      <PanelTabs
        panels={[relevance, flow]}
        first={answer("a fresh run")}
        draw={draw}
        projectId="p1"
        resetKey="a2"
      />,
    );
    expect(view.container.textContent).toContain("a fresh run");
    const tabs = [...view.container.querySelectorAll('[role="tab"]')];
    expect(tabs[0].getAttribute("aria-selected")).toBe("true");
  });
});
