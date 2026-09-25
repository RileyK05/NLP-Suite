// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { Glance as GlanceAnswer, Job } from "./api";

/**
 * Corpus at a glance on the Corpus page. What matters: nothing runs until
 * the button is pressed; a running glance shows its stage; a finished one
 * shows its sentences and figures; a changed corpus keeps the old result on
 * screen but says so and offers a refresh.
 */

const glanceStatus = vi.fn<(projectId: string) => Promise<GlanceAnswer>>();
const startGlance = vi.fn<(projectId: string, parser: string) => Promise<Job>>();
vi.mock("./api", () => ({
  glanceStatus: (projectId: string) => glanceStatus(projectId),
  startGlance: (projectId: string, parser: string) => startGlance(projectId, parser),
  glanceFigureUrl: async () => "blob:figure",
}));

const { Glance } = await import("./Glance");

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
beforeEach(() => {
  glanceStatus.mockReset();
  startGlance.mockReset();
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
const button = (scope: HTMLElement, text: string) =>
  [...scope.querySelectorAll("button")].find((b) => b.textContent?.includes(text)) as HTMLButtonElement | undefined;

const ready: GlanceAnswer = {
  state: "ready",
  key: "k",
  summary: ["87 documents, 612,345 words.", "Easiest to read: 2013 Obama (Flesch 62)."],
  figures: [{ tool: "readability", label: "Readability scores", panel: "readability_by_decade", path: "runs/x/figures/a.png" }],
};

describe("Glance", () => {
  it("waits for the button, then runs over the whole corpus with the chosen parser", async () => {
    glanceStatus.mockResolvedValueOnce({ state: "none", key: "k" });
    glanceStatus.mockResolvedValue({ state: "running", key: "k", job: { stage: "Parsing English documents" } as Job });
    startGlance.mockResolvedValue({} as Job);
    const container = render(<Glance projectId="p1" parser="stanza" hasDocuments />);
    await settle();
    expect(startGlance).not.toHaveBeenCalled();
    await act(async () => button(container, "Take a look")!.click());
    await settle();
    expect(startGlance).toHaveBeenCalledWith("p1", "stanza");
    expect(container.textContent).toContain("Parsing English documents");
  });

  it("shows what it found", async () => {
    glanceStatus.mockResolvedValue(ready);
    const container = render(<Glance projectId="p1" parser="spacy" hasDocuments />);
    await settle();
    await settle();
    const items = [...container.querySelectorAll(".glance-summary li")].map((li) => li.textContent);
    expect(items).toEqual(ready.summary);
    const image = container.querySelector(".glance-figure img") as HTMLImageElement;
    expect(image.getAttribute("src")).toBe("blob:figure");
    expect(button(container, "Take a look")).toBeUndefined();
  });

  it("keeps a stale result on screen and offers a refresh", async () => {
    glanceStatus.mockResolvedValue({ ...ready, state: "stale" });
    const container = render(<Glance projectId="p1" parser="spacy" hasDocuments />);
    await settle();
    expect(container.textContent).toContain("The documents have changed");
    expect(container.querySelectorAll(".glance-summary li")).toHaveLength(2);
    expect(button(container, "Refresh")).toBeDefined();
  });

  it("cannot run without documents", async () => {
    glanceStatus.mockResolvedValue({ state: "none", key: "k" });
    const container = render(<Glance projectId="p1" parser="spacy" hasDocuments={false} />);
    await settle();
    expect(button(container, "Take a look")!.disabled).toBe(true);
  });
});
