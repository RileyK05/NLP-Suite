// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { PublicationFigure, type PublishFigure } from "./PublicationFigure";
import { PanelAnswerView } from "./PanelSection";
import type { PanelAnswer } from "./api";

/**
 * The publication view: the same figure, drawn by the engine with
 * matplotlib and seaborn, shown as an image and saved in three formats.
 *
 * What matters: the view asks for the figure that is on screen (the caller's
 * drawKey), shows a refusal the way every panel refusal is shown, and saves
 * at print resolution rather than the preview's.
 */

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
beforeEach(() => {
  URL.createObjectURL = vi.fn(() => "blob:figure");
  URL.revokeObjectURL = vi.fn();
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
const settle = () => act(async () => new Promise((r) => setTimeout(r, 0)));

const image: PublishFigure = async (format) => ({
  ok: true,
  blob: new Blob(["x"]),
  format,
  warnings: 0,
});

describe("PublicationFigure", () => {
  it("shows the engine's image and saves at print resolution", async () => {
    const publish = vi.fn(image);
    const container = render(
      <PublicationFigure publish={publish} drawKey="a" fileBase="fig" />,
    );
    await settle();
    expect(publish).toHaveBeenCalledWith("png", 130);
    expect(container.querySelector("img")?.getAttribute("src")).toBe(
      "blob:figure",
    );
    const svg = [...container.querySelectorAll("button")].find((button) =>
      button.textContent?.includes("SVG"),
    );
    await act(async () => svg?.click());
    expect(publish).toHaveBeenLastCalledWith("svg", 300);
  });

  it("shows a refusal as diagnostics, not a broken image", async () => {
    const container = render(
      <PublicationFigure
        publish={async () => ({
          ok: false,
          diagnostics: [
            {
              severity: "ERROR",
              code: "STATIC_UNAVAILABLE",
              message: "no seaborn",
            },
          ],
        })}
        drawKey="a"
        fileBase="fig"
      />,
    );
    await settle();
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("no seaborn");
  });

  it("redraws when the figure on screen changes", async () => {
    const publish = vi.fn(image);
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mounted.push({ root, container });
    act(() =>
      root.render(
        <PublicationFigure publish={publish} drawKey="a" fileBase="f" />,
      ),
    );
    await settle();
    act(() =>
      root.render(
        <PublicationFigure publish={publish} drawKey="b" fileBase="f" />,
      ),
    );
    await settle();
    expect(publish).toHaveBeenCalledTimes(2);
  });
});

const answer = {
  ok: true,
  panel: "p",
  shape: "ranked_bars",
  title: "T",
  subtitle: "",
  xLabel: "x",
  yLabel: "",
  groups: [],
  caption: "c",
  notes: [],
  marks: [
    {
      key: "a",
      label: "war",
      x: 3,
      y: 0,
      group: "",
      size: null,
      labelled: false,
      evidence: { scope: "rows", filters: [], count: 1, describe: "war: 3" },
    },
  ],
  annotations: [],
  table: { columns: [], rows: [], total: 0, truncated: false },
  diagnostics: [],
} as unknown as PanelAnswer;

describe("the figure view switch", () => {
  it("offers the publication view only when the caller can draw one", () => {
    const without = render(
      <PanelAnswerView
        projectId="p"
        answer={answer}
        selected={null}
        onSelect={() => {}}
      />,
    );
    expect(without.textContent).not.toContain("Publication figure");
    const withIt = render(
      <PanelAnswerView
        projectId="p"
        answer={answer}
        selected={null}
        onSelect={() => {}}
        publish={image}
        drawKey="k"
      />,
    );
    expect(withIt.textContent).toContain("Publication figure");
    expect(withIt.querySelector("svg")).not.toBeNull();
  });

  it("swaps the interactive canvas for the image", async () => {
    const container = render(
      <PanelAnswerView
        projectId="p"
        answer={answer}
        selected={null}
        onSelect={() => {}}
        publish={image}
        drawKey="k"
      />,
    );
    const tab = [...container.querySelectorAll('[role="tab"]')].find((t) =>
      t.textContent?.includes("Publication"),
    ) as HTMLElement;
    await act(async () => tab.click());
    await settle();
    expect(container.querySelector(".panel-canvas")).toBeNull();
    expect(container.querySelector(".publication-figure img")).not.toBeNull();
  });
});
