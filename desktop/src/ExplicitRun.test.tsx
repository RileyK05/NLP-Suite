// @vitest-environment jsdom
/**
 * The bench starts work when the reader says so, and says so exactly once.
 *
 * Reported as a topic model that hung: the analysis was never requested, the
 * spinner ticked past two and a half minutes, and the server — asked
 * directly — answered in twelve seconds. The design asked *implicitly*, 250 ms
 * after the last control change, and the machinery deciding who was allowed
 * to ask disagreed with itself the moment a parameter changed while a request
 * was out: the superseded effect and the component each believed the other
 * owned the busy flag, so nothing ran, nothing was scheduled, and the panel
 * said "Working..." until it was reloaded.
 *
 * Contract R-C1 deletes the premise. These tests hold the line: editing is
 * free and silent, Run is the only trigger, one analysis is in flight at a
 * time, and a request that never answers frees the slot with an explicit
 * Retry rather than a silent second attempt.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { LiveBench } from "./LiveBench";
import { REQUEST_TIMEOUT_MS, type LiveAnswer, type LiveState } from "./live";
import type { Table, Tool } from "./api";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLElement }> = [];
beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
  vi.useRealTimers();
});

function render(element: ReactElement) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  mounted.push({ root, container });
  return container;
}

const table: Table = {
  columns: ["Topic", "Word", "Weight"],
  rows: [{ Topic: "0", Word: "freedom", Weight: "0.02" }],
  total: 1,
  truncated: false,
};

const answer: LiveAnswer = {
  request_id: "test-req-one",
  tool: "topic_model",
  ok: true,
  elapsed_ms: 12384,
  snapshot_id: "s1",
  diagnostics: [],
  path: "topics.csv",
  table,
  headline: "topic_model: 1 row(s) and 3 column(s).",
  observations: [],
  cautions: [],
  recommended_charts: [],
  figures: [],
  table_first: "",
  panels: [],
  answer_id: "",
};

const ready: LiveState = {
  state: "ready",
  stage: "",
  error: "",
  documents: 87,
  tokens: 237424,
  parsed: true,
  cached: true,
  source: "parsed",
  parse_ms: 40,
  selection: { documents: 87, words: 237424, names: [], ids: ["1"] },
  snapshot_id: "s1",
  parser: "spacy",
};

/** The real tool, with the parameter the report changed from 3 to 10. */
const tools: Tool[] = [
  {
    name: "topic_model",
    label: "Topic discovery (LDA)",
    description: "Recurring themes across the corpus.",
    requires_parse: true,
    category: "",
    params: [
      {
        name: "topics",
        type: "int",
        default: 3,
        required: false,
        help: "number of topics",
        label: "Number of topics",
        choices: [],
        minimum: 1,
        maximum: null,
      },
    ],
  } as unknown as Tool,
];

/** A promise whose settling this test decides. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const working = (container: HTMLElement): boolean =>
  (container.textContent ?? "").includes("Working");

function bench(
  onAnalyse: (...args: unknown[]) => Promise<unknown>,
  onAbort?: (requestId: string) => void,
) {
  return render(
    <LiveBench
      tools={tools}
      state={ready}
      contract={{ kinds: [], aggs: [], notes: {} } as never}
      onWarm={vi.fn()}
      onAnalyse={onAnalyse as never}
      onAbort={onAbort as never}
      onTrackPhrase={vi.fn().mockResolvedValue(null)}
      onReadSource={vi.fn().mockResolvedValue(null)}
      savedQuestions={[]}
      onSaveQuestion={vi.fn().mockResolvedValue(null)}
      onOpenQuestion={vi.fn().mockResolvedValue(null)}
      onDuplicateQuestion={vi.fn().mockResolvedValue(null)}
      onDeleteQuestion={vi.fn().mockResolvedValue(false)}
      onPublishQuestion={vi.fn()}
      busy={false}
      canPublish
      onPublish={vi.fn()}
    />,
  );
}

/** Pick the analysis. No request goes out — there is nothing to settle. */
async function choose(container: HTMLElement) {
  const select = container.querySelector(
    "select[aria-label='Analysis to run']",
  ) as HTMLSelectElement;
  await act(async () => {
    select.value = "topic_model";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

/** Retype the topic count, as the report did when moving 3 to 10. */
async function setTopics(container: HTMLElement, value: string) {
  const input = container.querySelector(
    "input[type='number']",
  ) as HTMLInputElement;
  if (!input) throw new Error("no topic-count field on the bench");
  // React tracks the DOM value it last wrote, so assigning `.value` directly
  // and firing an event is ignored as a no-op change. Going through the
  // prototype's setter is what makes React see a real edit -- without this the
  // field never changes and a test can "pass" having exercised nothing.
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    "value",
  )!.set!;
  await act(async () => {
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

/** The only act that starts an analysis (R-C1). */
async function pressRun(container: HTMLElement) {
  const button = [...container.querySelectorAll("button")].find((b) =>
    (b.textContent ?? "").includes("Run"),
  );
  if (!button) throw new Error("no Run button on the bench");
  await act(async () => {
    button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

describe("the test's own instrument", () => {
  it("really does edit the field", async () => {
    // If this fails, every result below is meaningless: the edit did nothing
    // and the component was never put in the state under test.
    const calls: unknown[][] = [];
    const container = bench(async (...args: unknown[]) => {
      calls.push(args);
      return answer;
    });
    await choose(container);
    await setTopics(container, "10");
    await pressRun(container);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(calls.length).toBe(1);
    expect(String((calls[0][1] as { topics: number }).topics)).toBe("10");
  });
});

describe("editing is free and silent (R-C1)", () => {
  it("choosing an analysis asks nothing", async () => {
    const onAnalyse = vi.fn();
    const container = bench(onAnalyse as never);
    await choose(container);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(onAnalyse).not.toHaveBeenCalled();
  });

  it("changing a setting asks nothing, however long the bench sits", async () => {
    // The old design asked 250 ms after the last change. A researcher
    // iterating on a setting paid a request per look and, if one landed
    // wrong, a wedged bench. Now: nothing, until Run.
    const onAnalyse = vi.fn();
    const container = bench(onAnalyse as never);
    await choose(container);
    await setTopics(container, "10");
    await setTopics(container, "4");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(120_000);
    });
    expect(onAnalyse).not.toHaveBeenCalled();
    expect(working(container)).toBe(false);
    expect(container.textContent).toContain("Settings changed");
  });
});

describe("Run starts exactly one analysis", () => {
  it("asks once, with a request id the engine can echo", async () => {
    const onAnalyse = vi.fn().mockResolvedValue(answer);
    const container = bench(onAnalyse as never);
    await choose(container);
    await pressRun(container);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(onAnalyse).toHaveBeenCalledTimes(1);
    const [tool, params, id] = onAnalyse.mock.calls[0] as [
      string,
      { topics: number },
      string,
    ];
    expect(tool).toBe("topic_model");
    expect(params.topics).toBe(3);
    expect(id.length).toBeGreaterThanOrEqual(8);
  });

  it("a setting changed while running costs nothing and leaves nothing wedged", async () => {
    // The incident in one sentence: edit a setting mid-request and the bench
    // believed work was happening forever. Now the edit is free, the first
    // answer is dropped as stale (it answers a question nobody is asking),
    // and the bench is idle and runnable — not "Working..." to nobody.
    const first = deferred<LiveAnswer>();
    const calls: unknown[][] = [];
    const onAnalyse = vi.fn((...args: unknown[]) => {
      calls.push(args);
      return calls.length === 1 ? first.promise : Promise.resolve(answer);
    });
    const container = bench(onAnalyse as never);
    await choose(container);
    await pressRun(container);
    expect(working(container)).toBe(true);

    await setTopics(container, "10");

    // The superseded answer lands, for a question nobody is asking.
    await act(async () => {
      first.resolve({ ...answer, request_id: "stale-req" });
      await first.promise;
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(working(container)).toBe(false);
    expect(calls.length).toBe(1);
    expect(container.textContent).toContain("Settings changed");
  });

  it("asks the question the controls ended up on when Run is pressed again", async () => {
    const first = deferred<LiveAnswer>();
    const calls: { tool: string; params: Record<string, unknown> }[] = [];
    const onAnalyse = vi.fn((tool: string, params: Record<string, unknown>) => {
      calls.push({ tool, params: { ...params } });
      return calls.length === 1 ? first.promise : Promise.resolve(answer);
    });
    const container = bench(onAnalyse as never);
    await choose(container);
    await pressRun(container);
    await setTopics(container, "10");
    await act(async () => {
      first.resolve(answer);
      await first.promise;
      await Promise.resolve();
      await Promise.resolve();
    });
    await pressRun(container);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(calls.length).toBe(2);
    expect(String(calls[calls.length - 1].params.topics)).toBe("10");
  });
});

describe("a request that never answers (R-C3)", () => {
  it("frees the slot and offers Run again, and retries nothing silently", async () => {
    const first = deferred<LiveAnswer>();
    const onAnalyse = vi.fn(() => first.promise);
    const container = bench(onAnalyse as never);
    await choose(container);
    await pressRun(container);
    expect(working(container)).toBe(true);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 2000);
    });

    // Not "Working..." to nobody: the slot is free and the reader is told,
    // with the one button that asks again. No hidden second request.
    expect(working(container)).toBe(false);
    expect(container.textContent).toContain("answer never");
    expect(onAnalyse).toHaveBeenCalledTimes(1);
    const again = [...container.querySelectorAll("button")].find((b) =>
      (b.textContent ?? "").includes("Run again"),
    );
    expect(again).toBeTruthy();
  });
});

describe("an abandoned question is abandoned on both sides (R-C8)", () => {
  it("tells the engine to stop waiting when the request is lost", async () => {
    const aborted: string[] = [];
    const never = deferred<LiveAnswer>();
    const container = bench(
      (() => never.promise) as never,
      (id) => aborted.push(id),
    );
    await choose(container);
    await pressRun(container);
    expect(aborted).toEqual([]);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 2000);
    });

    // The interface stopped waiting; the engine must be told, or a dead
    // question keeps an answer warm nobody will read.
    expect(aborted.length).toBe(1);
    expect(aborted[0]).toMatch(/^r-/);
    expect(container.textContent).toContain("Run again");
  });

  it("abandons an in-flight question when the bench goes away", async () => {
    const aborted: string[] = [];
    const never = deferred<LiveAnswer>();
    const container = bench(
      (() => never.promise) as never,
      (id) => aborted.push(id),
    );
    await choose(container);
    await pressRun(container);
    expect(aborted).toEqual([]);

    await act(async () => {
      const { root, container: box } = mounted[mounted.length - 1];
      root.unmount();
      box.remove();
    });

    expect(aborted.length).toBe(1);
    expect(aborted[0]).toMatch(/^r-/);
  });

  it("Ctrl+Enter runs, as the keyboard path to the same button", async () => {
    const calls: unknown[][] = [];
    const container = bench(async (...args: unknown[]) => {
      calls.push(args);
      return answer;
    });
    await choose(container);
    expect(calls.length).toBe(0);

    await act(async () => {
      document.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Enter",
          ctrlKey: true,
          bubbles: true,
        }),
      );
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(calls.length).toBe(1);
  });

  it("Ctrl+Enter alone does not run: nothing to run yet", async () => {
    const onAnalyse = vi.fn();
    const container = bench(onAnalyse as never);
    // No analysis chosen: the shortcut is the button's key, not a bypass.
    await act(async () => {
      document.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Enter",
          ctrlKey: true,
          bubbles: true,
        }),
      );
    });
    expect(onAnalyse).not.toHaveBeenCalled();
    expect(
      container.querySelector("select[aria-label='Analysis to run']"),
    ).toBeTruthy();
  });
});
