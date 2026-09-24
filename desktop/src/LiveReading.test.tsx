// @vitest-environment jsdom
/**
 * Does a live answer say anything, and does it open on a question?
 *
 * A sentiment analysis over 87 dated speeches used to arrive as a bar chart
 * of how many sentences each speech contained, longest first, with a row
 * count underneath. Nothing about it was wrong and nothing about it was an
 * answer. The engine could already read that table in words and name the
 * charts worth drawing from it; only the finished-run pane ever asked, so the
 * page people actually explore on had neither.
 *
 * These render the bench and read what is on screen, because the helpers were
 * right before and the component called them with the wrong argument.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, type ReactElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { LiveBench } from "./LiveBench";
import { chartSettings, openingSettings } from "./Workbench";
import type { LiveAnswer, LiveAnswerWithPanel, LiveState } from "./live";
import type { RecommendedChart, Table, Tool } from "./api";

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
  return container;
}
const click = (element: Element | null) => {
  if (!element) throw new Error("nothing to click");
  act(() => {
    element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
};

/** The document table a sentiment run over a dated corpus now produces. */
const sentimentTable: Table = {
  columns: ["Document ID", "Document", "Date", "Year", "Sentences", "Compound"],
  rows: [
    {
      "Document ID": "1",
      Document: "1934.txt",
      Date: "1934-01-03",
      Year: "1934",
      Sentences: "300",
      Compound: "0.2",
    },
    {
      "Document ID": "2",
      Document: "1935.txt",
      Date: "1935-01-04",
      Year: "1935",
      Sentences: "420",
      Compound: "0.1",
    },
    {
      "Document ID": "3",
      Document: "2024.txt",
      Date: "2024-03-07",
      Year: "2024",
      Sentences: "512",
      Compound: "-0.1",
    },
  ],
  total: 3,
  truncated: false,
};

const timeline: RecommendedChart = {
  kind: "line",
  x: "Date",
  y: "Compound",
  agg: "mean",
  top_n: null,
  question: "Has the tone of these documents changed over time?",
  why: "Document-level compound against the calendar.",
};
const ranking: RecommendedChart = {
  kind: "bar",
  x: "Document",
  y: "Compound",
  agg: "sum",
  top_n: 20,
  question: "Which documents read most positive or negative overall?",
  why: "Document-level compound is a mean over sentences.",
};

function answer(
  overrides: Partial<LiveAnswerWithPanel> = {},
): LiveAnswerWithPanel {
  return {
    request_id: "test-req-read1",
    tool: "sentiment_vader_anew",
    ok: true,
    elapsed_ms: 12,
    snapshot_id: "s1",
    diagnostics: [],
    path: "vader.csv",
    table: sentimentTable,
    headline: "sentiment_vader_anew: 3 row(s) and 6 column(s).",
    observations: ["These 3 dated row(s) run from 1934-01-03 to 2024-03-07."],
    cautions: [],
    recommended_charts: [timeline, ranking],
    figures: [],
    table_first: "",
    panels: [],
    answer_id: "",
    ...overrides,
  };
}

const ready: LiveState = {
  state: "ready",
  stage: "",
  error: "",
  documents: 3,
  tokens: 1200,
  parsed: true,
  cached: false,
  source: "parsed",
  parse_ms: 40,
  selection: { documents: 3, words: 1200, names: [], ids: ["1", "2", "3"] },
  snapshot_id: "s1",
  parser: "spacy",
};

const tools: Tool[] = [
  {
    name: "sentiment_vader_anew",
    label: "Sentiment (VADER / ANEW)",
    description: "Measure positive and negative tone.",
    params: [],
    requires_parse: true,
    category: "",
  } as unknown as Tool,
];

function bench(result: LiveAnswer | null) {
  const onAnalyse = vi.fn().mockResolvedValue(result);
  const container = render(
    <LiveBench
      tools={tools}
      state={ready}
      contract={{ kinds: [], aggs: [], notes: {} } as never}
      onWarm={vi.fn()}
      onAnalyse={onAnalyse}
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
  return { container, onAnalyse };
}

/** Choose the analysis, then press Run — the only way one starts (R-C1). */
async function run(container: HTMLElement) {
  const select = container.querySelector(
    "select[aria-label='Analysis to run']",
  ) as HTMLSelectElement;
  await act(async () => {
    select.value = "sentiment_vader_anew";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const button = [...container.querySelectorAll("button")].find((b) =>
    (b.textContent ?? "").includes("Run analysis"),
  );
  click(button ?? null);
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

const text = (container: HTMLElement) => container.textContent ?? "";

/**
 * A workbench control by the words above it.
 *
 * The workbench labels wrap their select rather than naming it, and the page
 * holds a dozen selects, so `querySelector("select")` finds whichever one is
 * first in the document -- which is how an earlier test in this suite came to
 * drive "Position detail" while believing it was driving the shelf.
 */
function workbenchControl(
  container: HTMLElement,
  label: string,
): HTMLSelectElement {
  const field = Array.from(
    container.querySelectorAll("label.field-label"),
  ).find(
    (element) => (element.childNodes[0]?.textContent ?? "").trim() === label,
  );
  const select = field?.querySelector("select");
  if (!select) throw new Error(`no control labelled ${label}`);
  return select as HTMLSelectElement;
}
const reading = (container: HTMLElement) =>
  container.querySelector("section.insight-panel");

describe("a live answer is read back in words", () => {
  it("shows a returned panel as the primary figure and reports panel refusal", async () => {
    vi.useFakeTimers();
    try {
      const { container } = bench(
        answer({
          panel: {
            ok: false,
            diagnostics: [
              {
                severity: "WARNING",
                code: "PANEL_UNAVAILABLE",
                message: "This result has no usable panel rows.",
              },
            ],
          },
        }),
      );
      await run(container);
      expect(
        container.querySelector('[aria-label="Live figure"]'),
      ).not.toBeNull();
      expect(text(container)).toContain(
        "This result has no usable panel rows.",
      );
      expect(
        container.querySelector('[aria-label="Explore this table"]'),
      ).toBeNull();
      click(
        [...container.querySelectorAll("button")].find((button) =>
          button.textContent?.includes("Open generic chart workbench (expert)"),
        ) ?? null,
      );
      expect(
        container.querySelector('[aria-label="Explore this table"]'),
      ).not.toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows the reading the engine produced, not just a row count", async () => {
    vi.useFakeTimers();
    try {
      const { container } = bench(answer());
      await run(container);
      const panel = reading(container);
      expect(panel).not.toBeNull();
      expect(text(panel as HTMLElement)).toContain("1934-01-03 to 2024-03-07");
    } finally {
      vi.useRealTimers();
    }
  });

  it("offers the recommended questions with the question in front of them", async () => {
    vi.useFakeTimers();
    try {
      const { container } = bench(answer());
      await run(container);
      expect(text(container)).toContain(
        "Has the tone of these documents changed over time?",
      );
      expect(text(container)).toContain(
        "Which documents read most positive or negative overall?",
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("draws the recommendation that is pressed", async () => {
    vi.useFakeTimers();
    try {
      const { container } = bench(answer());
      await run(container);
      // It opens on the timeline, so the ranking is the one that must move it.
      expect(workbenchControl(container, "Measure").value).toBe("Compound");
      expect(workbenchControl(container, "Group rows by").value).toBe("Date");

      const buttons = [...container.querySelectorAll(".insight-chart button")];
      const bar = buttons.find((b) => (b.textContent ?? "").includes("bar"));
      click(bar ?? null);

      // Pressing it has to move the controls, not just look like it did.
      expect(workbenchControl(container, "Measure").value).toBe("Compound");
      expect(workbenchControl(container, "Group rows by").value).toBe(
        "Document",
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it("says nothing rather than an empty panel when there is nothing to say", async () => {
    vi.useFakeTimers();
    try {
      const { container } = bench(
        answer({
          headline: "",
          observations: [],
          cautions: [],
          recommended_charts: [],
        }),
      );
      await run(container);
      expect(reading(container)).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("where the chart opens", () => {
  it("opens on the recommended chart rather than on column order", () => {
    // Column order would give the first label column against the first
    // measure: Document against Sentences, which is a chart of how long each
    // document is wearing a sentiment analysis's name.
    const settings = openingSettings(sentimentTable, [timeline, ranking]);
    expect(settings.kind).toBe("line");
    expect([settings.x, settings.y]).toEqual(["Date", "Compound"]);
  });

  it("falls back to column order when nothing is recommended", () => {
    const settings = openingSettings(sentimentTable, []);
    expect(settings.kind).toBe("bar");
    expect(settings.y).toBe("Sentences");
  });

  it("skips a recommendation naming a column this table does not have", () => {
    const stale: RecommendedChart = { ...timeline, x: "Publication Date" };
    const settings = openingSettings(sentimentTable, [stale, ranking]);
    expect([settings.x, settings.y]).toEqual(["Document", "Compound"]);
  });

  it("skips a kind the workbench cannot draw itself", () => {
    const engineOnly: RecommendedChart = { ...timeline, kind: "calendar" };
    const settings = openingSettings(sentimentTable, [engineOnly, ranking]);
    expect(settings.kind).toBe("bar");
  });

  it("keeps every category of a timeline and averages it", () => {
    // Both matter. Summed, two documents dated the same day become a spike
    // about how many documents that day had; cut to the top 20, the quiet
    // years vanish and the line is drawn straight through where they were.
    const settings = chartSettings(timeline);
    expect(settings?.topN).toBe(0);
    expect(settings?.agg).toBe("mean");
    expect(settings?.sort).toBe("table");
  });

  it("leaves a ranking ranked", () => {
    const settings = chartSettings(ranking);
    expect(settings?.topN).toBe(20);
    expect(settings?.sort).toBe("high");
  });
});
