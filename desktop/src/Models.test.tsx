// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ModelInfo, ModelListing } from "./api";

/**
 * The Models page: every registered model, whether it is here, and one
 * button that gets it. What matters: bundled models say so and cannot be
 * removed, a missing one offers Download, a running download shows progress
 * and Cancel, a failure shows the engine's sentence, and a download that
 * finishes tells the app (so the tools that needed it become available).
 */

const listModels = vi.fn<() => Promise<ModelListing>>();
const downloadModel = vi.fn<(id: string) => Promise<ModelInfo>>();
const cancelModelDownload = vi.fn<(id: string) => Promise<ModelInfo>>();
const deleteModel = vi.fn<(id: string) => Promise<ModelInfo>>();
vi.mock("./api", () => ({
  listModels: () => listModels(),
  downloadModel: (id: string) => downloadModel(id),
  cancelModelDownload: (id: string) => cancelModelDownload(id),
  deleteModel: (id: string) => deleteModel(id),
  title: (name: string) => ({ bert_topics: "Topic clusters (BERT)" })[name] ?? name,
}));

const { Models } = await import("./Models");

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
afterEach(() => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
  vi.useRealTimers();
});
beforeEach(() => {
  listModels.mockReset();
  downloadModel.mockReset();
  cancelModelDownload.mockReset();
  deleteModel.mockReset();
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

function model(overrides: Partial<ModelInfo>): ModelInfo {
  return {
    id: "bert-base-uncased",
    name: "BERT base (uncased)",
    kind: "token_embeddings",
    kindLabel: "Word vectors",
    description: "A vector for every word in its sentence.",
    sizeMb: 105,
    status: "ready",
    bundled: true,
    removable: false,
    license: "Apache-2.0",
    source: "https://huggingface.co/google-bert/bert-base-uncased",
    precision: "q8e4",
    usedBy: ["bert_topics"],
    download: null,
    ...overrides,
  };
}

const qwen = (overrides: Partial<ModelInfo> = {}) =>
  model({
    id: "qwen3-embedding-0.6b",
    name: "Qwen3 Embedding 0.6B",
    kind: "sentence_embeddings",
    kindLabel: "Sentence and document vectors",
    sizeMb: 569,
    status: "not_downloaded",
    bundled: false,
    usedBy: [],
    ...overrides,
  });

const card = (container: HTMLElement, name: string) =>
  container.querySelector(`[aria-label="${name}"]`) as HTMLElement;
const button = (scope: HTMLElement, text: string) =>
  [...scope.querySelectorAll("button")].find((b) => b.textContent?.includes(text)) as HTMLButtonElement | undefined;

describe("Models", () => {
  it("groups models by what they do and says which ship with the app", async () => {
    listModels.mockResolvedValue({ models: [model({}), qwen()], folder: "C:/data/models" });
    const container = render(<Models />);
    await settle();
    const headings = [...container.querySelectorAll("h2")].map((h) => h.textContent);
    expect(headings).toEqual(["Word vectors", "Sentence and document vectors"]);
    const bert = card(container, "BERT base (uncased)");
    expect(bert.textContent).toContain("Included with the app");
    expect(bert.textContent).toContain("Used by Topic clusters (BERT)");
    expect(button(bert, "Remove")).toBeUndefined();
    expect(button(bert, "Download")).toBeUndefined();
    expect(button(card(container, "Qwen3 Embedding 0.6B"), "Download")).toBeDefined();
    expect(container.textContent).toContain("569 MB");
    expect(container.textContent).toContain("C:/data/models");
  });

  it("downloads, shows progress, and tells the app when it finishes", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const onChanged = vi.fn();
    listModels
      .mockResolvedValueOnce({ models: [qwen()], folder: "" })
      .mockResolvedValueOnce({
        models: [qwen({ status: "downloading", download: { state: "downloading", done: 250, total: 1000, message: "" } })],
        folder: "",
      })
      .mockResolvedValue({ models: [qwen({ status: "ready", removable: true })], folder: "" });
    downloadModel.mockResolvedValue(qwen({ status: "downloading" }));
    const container = render(<Models onChanged={onChanged} />);
    await settle();
    await act(async () => button(card(container, "Qwen3 Embedding 0.6B"), "Download")!.click());
    await settle();
    expect(downloadModel).toHaveBeenCalledWith("qwen3-embedding-0.6b");
    const progress = container.querySelector("progress") as HTMLProgressElement;
    expect(progress.value).toBe(25);
    expect(button(card(container, "Qwen3 Embedding 0.6B"), "Cancel")).toBeDefined();
    await act(async () => {
      vi.advanceTimersByTime(1100);
    });
    await settle();
    expect(card(container, "Qwen3 Embedding 0.6B").textContent).toContain("Downloaded");
    expect(onChanged).toHaveBeenCalled();
    expect(button(card(container, "Qwen3 Embedding 0.6B"), "Remove")).toBeDefined();
  });

  it("shows why a download failed and offers to try again", async () => {
    listModels.mockResolvedValue({
      models: [
        qwen({
          download: {
            state: "failed",
            done: 0,
            total: 1,
            message: "Couldn't reach the download server. Check your internet connection (or proxy) and try again.",
          },
        }),
      ],
      folder: "",
    });
    const container = render(<Models />);
    await settle();
    const qwenCard = card(container, "Qwen3 Embedding 0.6B");
    expect(qwenCard.textContent).toContain("Couldn't reach the download server");
    expect(button(qwenCard, "Try again")).toBeDefined();
  });

  it("removes a downloaded model", async () => {
    const onChanged = vi.fn();
    listModels
      .mockResolvedValueOnce({ models: [qwen({ status: "ready", removable: true })], folder: "" })
      .mockResolvedValue({ models: [qwen()], folder: "" });
    deleteModel.mockResolvedValue(qwen({ freedMb: 569 }));
    const container = render(<Models onChanged={onChanged} />);
    await settle();
    await act(async () => button(card(container, "Qwen3 Embedding 0.6B"), "Remove")!.click());
    await settle();
    expect(deleteModel).toHaveBeenCalledWith("qwen3-embedding-0.6b");
    expect(onChanged).toHaveBeenCalled();
    expect(button(card(container, "Qwen3 Embedding 0.6B"), "Download")).toBeDefined();
  });
});
