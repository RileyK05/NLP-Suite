// @vitest-environment jsdom
/**
 * The in-app update offer: nothing happens until the reader asks, and the
 * engine is stopped before the installer replaces its files.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { relaunch } from "@tauri-apps/plugin-process";
import type { Update } from "@tauri-apps/plugin-updater";
import { UpdateBanner } from "./UpdateBanner";

vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn() }));
vi.mock("@tauri-apps/plugin-process", () => ({ relaunch: vi.fn() }));
vi.mock("@tauri-apps/plugin-updater", () => ({ check: vi.fn() }));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;
const order: string[] = [];

function fakeUpdate(overrides: Partial<Update> = {}): Update {
  return {
    version: "9.9.9",
    download: vi.fn(async (onEvent?: (event: unknown) => void) => {
      order.push("download");
      onEvent?.({ event: "Started", data: { contentLength: 200 } });
      onEvent?.({ event: "Progress", data: { chunkLength: 100 } });
    }),
    install: vi.fn(async () => {
      order.push("install");
    }),
    ...overrides,
  } as unknown as Update;
}

async function show(update: Update | null) {
  await act(async () => {
    root.render(<UpdateBanner find={() => Promise.resolve(update)} />);
  });
}

function button(label: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find(
    (b) => b.textContent === label || b.getAttribute("aria-label") === label,
  );
  if (!found) throw new Error(`no button ${label}`);
  return found;
}

beforeEach(() => {
  order.length = 0;
  vi.mocked(invoke).mockImplementation(async () => {
    order.push("stop_engine");
  });
  vi.mocked(relaunch).mockImplementation(async () => {
    order.push("relaunch");
  });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.clearAllMocks();
});

describe("UpdateBanner", () => {
  it("shows nothing when this is the newest version", async () => {
    await show(null);
    expect(container.textContent).toBe("");
  });

  it("offers the update and does nothing until asked", async () => {
    const update = fakeUpdate();
    await show(update);
    expect(container.textContent).toContain("NLP Suite 9.9.9 is available.");
    expect(update.download).not.toHaveBeenCalled();
  });

  it("downloads, stops the engine, installs, then restarts, in that order", async () => {
    await show(fakeUpdate());
    await act(async () => button("Install and restart").click());
    expect(order).toEqual(["download", "stop_engine", "install", "relaunch"]);
    expect(invoke).toHaveBeenCalledWith("stop_engine");
  });

  it("says so plainly when the install fails", async () => {
    await show(fakeUpdate({ install: vi.fn(async () => Promise.reject(new Error("locked"))) }));
    await act(async () => button("Install and restart").click());
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("could not be installed");
    expect(relaunch).not.toHaveBeenCalled();
  });

  it("can be dismissed", async () => {
    await show(fakeUpdate());
    await act(async () => button("Dismiss update").click());
    expect(container.textContent).toBe("");
  });
});
