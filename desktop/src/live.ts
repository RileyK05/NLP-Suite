import { visibleParams, type Params } from "./ParamFields";
import type { Reading, Table, Tool } from "./api";
import type { AnalyseResponse, LiveStateModel } from "./contract";
import type { PanelAnswer, PreparedPanel, PanelRefusal } from "./api";

/**
 * Talking to the live bench, and knowing when not to.
 *
 * The bargain is that parsing is slow and everything after it is fast. A
 * hundred thousand words take about thirty seconds to parse and about a fifth
 * of a second to compute readability over, so the corpus is parsed once and
 * then the controls are free to change.
 *
 * Changing a control does *nothing* (contract R-C1). Analyses start when the
 * reader presses Run. That is not a preference: the previous design asked
 * implicitly, 250 ms after the last change, and the machinery deciding who
 * was allowed to ask wedged itself the moment a setting changed while a
 * request was out. Explicit run deletes that whole class of bug, and makes
 * iterating on a setting free — edit, look, edit, then run when ready.
 */

/** How warm a project's corpus is. Generated from the engine's contract. */
export type LiveState = LiveStateModel;

/** One live answer. Generated from the engine's contract (R-C6). */
export type LiveAnswer = AnalyseResponse;

/** Server panel data is being added to the generated response contract. */
export type LiveAnswerWithPanel = Omit<LiveAnswer, "panel"> & {
  panel?: PanelAnswer | null;
};

/** Validate the optional panel at the generated contract's untyped boundary. */
export function livePanelAnswer(value: unknown): PanelAnswer | null {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Record<string, unknown>;
  if (!Array.isArray(candidate.diagnostics)) return null;
  if (candidate.ok === false) return candidate as unknown as PanelRefusal;
  if (
    candidate.ok !== true ||
    typeof candidate.panel !== "string" ||
    typeof candidate.shape !== "string" ||
    typeof candidate.title !== "string" ||
    typeof candidate.subtitle !== "string" ||
    typeof candidate.xLabel !== "string" ||
    typeof candidate.yLabel !== "string" ||
    typeof candidate.caption !== "string" ||
    !Array.isArray(candidate.groups) ||
    !Array.isArray(candidate.notes) ||
    !Array.isArray(candidate.marks) ||
    !Array.isArray(candidate.annotations) ||
    !candidate.table ||
    typeof candidate.table !== "object"
  ) {
    return null;
  }
  return candidate as unknown as PreparedPanel;
}

export const COLD: LiveState = {
  state: "cold",
  stage: "",
  error: "",
  documents: 0,
  tokens: 0,
  parsed: false,
  cached: false,
  source: "",
  parse_ms: 0,
  selection: null,
  snapshot_id: "",
  parser: "",
};

/**
 * How long the elapsed-seconds readout keeps ticking before it stops itself.
 *
 * Ten minutes is far beyond any live analysis observed -- the slowest over
 * the real corpus was a topic model at about half a minute. A timer that
 * never stops is a timer that keeps waking the tab long after anyone is
 * watching, and one that never ends cannot be advanced to completion by a
 * timer-based test. When it stops, the readout freezes at the bound, which
 * still says "this has been running for ten minutes" -- a different message
 * from "stuck", and an honest one.
 */
export const MAX_ELAPSED_TICKS = 600;

/** What a request is *about*, so a stale answer can be recognised as stale. */
export function questionKey(tool: string, params: Params): string {
  // Keys sorted, so the same question written in a different order is the
  // same question and does not cost a second request.
  const ordered = Object.keys(params)
    .sort()
    .map((name) => [name, params[name]]);
  return JSON.stringify([tool, ordered]);
}

/**
 * Whether an answer still answers the question on screen.
 *
 * Responses do not arrive in the order they were sent. Without this, letting
 * go of a slider at 30 can leave the chart showing 12 — because the request
 * for 12 was slower — and nothing on screen would say which number produced
 * the picture.
 */
export const isCurrent = (answer: string, asked: string): boolean =>
  answer === asked;

/**
 * Parameters a tool cannot run without, that nobody has filled in yet.
 *
 * Most analyses have a default for everything and can run the moment they are
 * chosen. Some cannot: a word-group analysis with no word groups, or a search
 * with no query, has nothing to compute.
 */
export function missingRequired(tool: Tool, params: Params): string[] {
  return visibleParams(tool, params)
    .filter((param) => param.required)
    .filter((param) => {
      const value = params[param.name];
      return (
        value === undefined || value === null || String(value).trim() === ""
      );
    })
    .map((param) => param.label);
}

/**
 * Whether the controls have moved on from the answer on show (R-C1).
 *
 * True only when a question exists and it is not the one already answered —
 * the cue to say "press Run", and the difference between iterating for free
 * and paying a request per keystroke.
 */
export function isDirty(shown: string, asked: string): boolean {
  return asked !== "" && asked !== shown;
}

/**
 * Whether a request that has been out this long is considered lost.
 *
 * A response should never take this long: the slowest live analysis over the
 * real corpus is the topic model at about half a minute, and the server
 * answers every request eventually, even if only to fail. Past the bound the
 * slot is released and the reader is offered an explicit Retry (R-C3): a
 * dropped fetch (tab sleep, a proxy reset, a dev-server reload) used to leave
 * the slot held forever, and the bench read as "Working…" to nobody.
 */
export const REQUEST_TIMEOUT_MS = 180_000;

/** Whether an out-for-this-long request should be treated as lost. */
export function requestIsLost(elapsedMs: number): boolean {
  return elapsedMs > REQUEST_TIMEOUT_MS;
}

/** A request identity the engine echoes back (R-C2). Short and sortable-ish. */
export function requestId(): string {
  return `r-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Tools that only run as recorded jobs, never on the live bench.
 *
 * lda_mallet stages an external binary's inputs and can run for minutes: a
 * job with a run directory is the right shape for it (R-C1's "explicit run"
 * includes knowing which kind of run you are asking for).
 */
const JOB_ONLY = new Set(["lda_mallet"]);

/** Tools the live bench can run. Visualisations are a publishing step, and
 *  table tools need a table rather than a corpus. */
export function liveTools(tools: Tool[]): Tool[] {
  return tools.filter(
    (tool) =>
      tool.category !== "visualization" &&
      !tool.name.startsWith("table_") &&
      !JOB_ONLY.has(tool.name) &&
      tool.availability?.state !== "needs_setup" &&
      // Its model is not installed yet: the Models page, not the bench, is
      // where that is fixed.
      tool.availability?.state !== "needs_model",
  );
}

/**
 * What to say while a corpus is warming.
 *
 * The wait is real and cannot be engineered away, so the interface says what
 * is happening and roughly how much there is to do rather than showing a
 * spinner and hoping.
 */
export function warmingNote(state: LiveState): string {
  if (state.state !== "warming") return "";
  const words = state.selection?.words ?? 0;
  if (!words) return state.stage || "Reading the documents…";
  // Two measurements on a laptop CPU, ten times apart in size: ~48,000 words
  // in 11.7 s and 516,987 words in 78.4 s. They fit a fixed cost plus a rate,
  // not a straight proportion -- loading the parser is several seconds whatever
  // the corpus. Assuming proportion predicted 141 s for the parse that took 78,
  // which makes the one slow step look twice as dear as it is.
  const seconds = Math.max(2, Math.round(5 + words / 7000));
  return `${state.stage || "Parsing"} — about ${seconds}s for ${words.toLocaleString()} words. This happens once.`;
}

/** Plain words for where a ready corpus's parse came from. */
export function readyNote(state: LiveState): string {
  if (state.state !== "ready") return "";
  const where =
    state.source === "parsed"
      ? `parsed in ${(state.parse_ms / 1000).toFixed(1)}s`
      : state.source === "disk"
        ? "recovered from an earlier parse"
        : "already in hand";
  return `${state.documents} document${state.documents === 1 ? "" : "s"}, ${state.tokens.toLocaleString()} words — ${where}.`;
}

/** The reading half of an answer, as the panels already draw it. */
export type LiveReading = Reading;
/** The table half of an answer, as the workbench already draws it. */
export type LiveTable = Table;
