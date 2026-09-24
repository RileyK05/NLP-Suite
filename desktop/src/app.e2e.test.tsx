// @vitest-environment jsdom
/**
 * The whole interactive stack, driven the way a researcher drives it.
 *
 * Every other layer has a test that proves it in isolation: the core module
 * against fixtures, the server through its own HTTP calls, the components
 * against fakes. What none of them establishes is the journey this file
 * walks -- the built interface mounted against the real server, the real
 * corpus, and the real parse, clicked through from "load documents" to "a
 * passage opens at the characters that produced a count."
 *
 * The server comes from the environment (scripts/interactive_stack.py prints
 * its handshake; scripts/run_interactive_frontend_test.py wires it in).
 * Nothing about the app is stubbed: the same connect() the Tauri shell runs,
 * the same /api routes, the same saved-question store. Only the browser
 * itself is simulated, because a full Tauri run cannot live in a test suite.
 *
 * These tests are slow by design and gated behind the real corpus, the same
 * way the backend's model_integration tests are: run them from the repo root
 * with python scripts/run_interactive_frontend_test.py.
 */
import { afterEach, describe, expect, it } from "vitest";
import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import App from "./App";

declare global {
  // eslint-disable-next-line no-var
  var IS_REACT_ACT_ENVIRONMENT: boolean;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

/**
 * The handshake, injected by scripts/run_interactive_frontend_test.py. In a
 * real browser connect() targets location.origin, which IS the server; jsdom
 * has no server there, so the base URL and token arrive as environment
 * variables and are applied to the connection module below.
 */
type Handshake = { baseUrl: string; token: string };
const environment = globalThis as {
  process?: { env: Record<string, string | undefined> };
};
const rawEnvironment = environment.process?.env ?? {};
const connection: Handshake | undefined =
  rawEnvironment.NLP_INTERACTIVE_BASE_URL &&
  rawEnvironment.NLP_INTERACTIVE_TOKEN
    ? {
        baseUrl: rawEnvironment.NLP_INTERACTIVE_BASE_URL,
        token: rawEnvironment.NLP_INTERACTIVE_TOKEN,
      }
    : undefined;

/**
 * Skipped unless the runner provides a server, so an ordinary `npm test` --
 * which has no server and cannot -- still reports green rather than a
 * failure that reads like a broken feature. The journey runs from the repo
 * root: python scripts/run_interactive_frontend_test.py
 */
const ddescribe = connection ? describe : describe.skip;

const settle = (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms));

/** Wait until the predicate holds, or fail with what the screen does say. */
async function waitFor(
  predicate: () => boolean,
  what: string,
  timeoutMs = 45_000,
) {
  const started = Date.now();
  for (;;) {
    let ok = false;
    try {
      ok = predicate();
    } catch {
      ok = false;
    }
    if (ok) return;
    if (Date.now() - started > timeoutMs) {
      throw new Error(
        `Timed out waiting for ${what}. The screen reads:\n${document.body.textContent}`,
      );
    }
    await settle(200);
  }
}

/**
 * A click that goes through act(), so React's state updates flush before the
 * promise resolves; then a settle for the network work the handler starts.
 */
async function click(element: Element, ms = 300) {
  await act(async () => {
    element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  await settle(ms);
}

function type(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    "value",
  )!.set!;
  act(() => {
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

function button(label: string | RegExp): HTMLButtonElement {
  const found = Array.from(
    document.body.querySelectorAll<HTMLButtonElement>("button"),
  ).find((element) =>
    typeof label === "string"
      ? (element.textContent ?? "").includes(label)
      : label.test(element.textContent ?? ""),
  );
  if (!found) {
    throw new Error(
      `no button ${String(label)} on:\n${document.body.textContent}`,
    );
  }
  return found;
}

function nav(label: string) {
  const found = Array.from(document.body.querySelectorAll("nav button")).find(
    (element) => (element.textContent ?? "").includes(label),
  );
  if (!found) throw new Error(`no nav item ${label}`);
  return found;
}

const mounted: Array<{ root: Root; container: HTMLElement }> = [];

function mountApp(): HTMLElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(<App />));
  mounted.push({ root, container });
  return container;
}

afterEach(async () => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
});

ddescribe("the interactive stack over the real corpus", () => {
  it(
    "connects, loads the corpus, tracks a phrase, brushes its positions, " +
      "opens the passage, and reopens the saved question",
    { timeout: 300_000 },
    async () => {
      if (!connection) {
        throw new Error(
          "No interactive server in the environment. " +
            "Run: python scripts/run_interactive_frontend_test.py",
        );
      }
      // The server is bound to port 3000 -- jsdom's own origin -- so the
      // app's connect() flow runs exactly as it would in a browser: token
      // from sessionStorage, base URL from location.origin, /health probe.
      sessionStorage.setItem("nlp-token", connection.token);
      mountApp();

      // The app lands on Overview once connected and has adopted the only
      // project. Waiting for the project's name is waiting for both.
      await waitFor(
        () =>
          document.body.textContent?.includes("State of the Union") ?? false,
        "the imported project to load",
      );
      await click(button("Corpus"));
      await waitFor(
        () => /87/.test(document.body.textContent ?? ""),
        "the corpus page to list the imported documents",
        20_000,
      );

      // ---- load documents (the real warm, over the cached parse) ----
      await click(nav("Interactive"));
      await click(button("Load documents"));
      // Warming costs a parquet read and a groupby over 600k tokens; the
      // note says so while it works.
      await waitFor(
        () =>
          (document.body.textContent ?? "").includes("documents are warm") ||
          /Reload documents/.test(document.body.textContent ?? ""),
        "the corpus to warm",
        120_000,
      );

      // ---- track the one phrase whose tokenization is the whole point ----
      const phrase = document.body.querySelector<HTMLInputElement>(
        ".phrase-query input",
      )!;
      type(phrase, "U.S.");
      await click(button("Track phrase"));
      await waitFor(
        () => document.body.textContent?.includes("Interpreted as") ?? false,
        "the tracked answer",
        120_000,
      );
      // Real tokenizer, real table: the count is a fact about the corpus,
      // and it has been stable across every session that produced one.
      expect(document.body.textContent).toContain("U.S.");
      expect(document.body.textContent).toContain("occurrences");

      // ---- the track of a document that contains the phrase ----
      expect(document.body.textContent).toContain("Within each document");
      // Tracks list every document, including empty ones; expand one that
      // has occurrences, which its row announces in its own meta line.
      const toggles = Array.from(
        document.body.querySelectorAll<HTMLButtonElement>(
          ".document-track .track-toggle",
        ),
      );
      expect(toggles.length).toBeGreaterThan(0);
      const loaded = toggles.find(
        (element) => !/· 0 ·/.test(element.textContent ?? ""),
      );
      expect(loaded).toBeDefined();
      await click(loaded!);
      const position = document.body.querySelector<HTMLButtonElement>(
        ".track-positions button",
      );
      expect(position).not.toBeNull();
      await click(position!);

      // ---- the source reader opens the exact characters ----
      await waitFor(
        () => document.body.textContent?.includes("Source reader") ?? false,
        "the source reader to open",
        60_000,
      );
      const reader = document.body.querySelector(".reader-passage")!;
      expect(reader.querySelector("mark")?.textContent).toContain("U.S.");

      // ---- widen the context: same place, more of the document ----
      await click(button("±1,000"));
      await waitFor(
        () =>
          (document.body.querySelector(".reader-passage")?.textContent ?? "")
            .length > reader.textContent!.length,
        "the reader to widen",
        60_000,
      );
      expect(
        document.body.querySelector(".reader-passage mark")!.textContent,
      ).toContain("U.S.");

      // ---- brush the position chart; the evidence follows, aggregates stay ----
      const grid = document.body.querySelector(".phrase-grid")!;
      const positionPanel = grid.querySelectorAll("section")[1];
      const bins = Array.from(
        positionPanel.querySelectorAll<HTMLButtonElement>(".research-bar"),
      );
      expect(bins.length).toBeGreaterThan(0);
      await click(bins[0]);
      await waitFor(
        () =>
          document.body.textContent?.includes("Evidence is filtered") ?? false,
        "the brushed filter to land",
        120_000,
      );
      // The aggregates keep describing the whole corpus while the evidence
      // describes the selection: brushing is not re-scoping.
      expect(document.body.textContent).toContain("Occurrences");
      const clearBrush = Array.from(
        document.body.querySelectorAll<HTMLButtonElement>("button"),
      ).find((element) =>
        (element.textContent ?? "").includes("Clear position filter"),
      );
      if (clearBrush) {
        await click(clearBrush);
        await waitFor(
          () => !document.body.textContent?.includes("Evidence is filtered"),
          "the brush to clear",
          120_000,
        );
      }

      // ---- keep the question, then reopen it from the shelf ----
      const name = document.body.querySelector<HTMLInputElement>(
        ".question-save input",
      )!;
      type(name, "Interactive U.S. check");
      await click(button("Save question"));
      await waitFor(
        () => document.body.textContent?.includes("saved") ?? false,
        "the save to land",
        60_000,
      );
      await click(button("Open"));
      await waitFor(
        () => document.body.textContent?.includes("revision") ?? false,
        "the reopened question",
        120_000,
      );
      expect(document.body.textContent).toContain("Interpreted as");

      // The whole journey never surfaced an error the reader would see.
      expect(document.body.textContent).not.toContain("could not be");
    },
  );
});
