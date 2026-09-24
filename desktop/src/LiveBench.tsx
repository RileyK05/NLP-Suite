import { useEffect, useRef, useState } from "react";
import { Flame, Play, Sparkles, TriangleAlert, Zap } from "lucide-react";
import { ParamFields, defaultParams, type Params } from "./ParamFields";
import { chartSettings, useChartSettings, Workbench } from "./Workbench";
import { ReadingPanel } from "./InsightPanel";
import { Diagnostics } from "./RunStatus";
import { PanelCanvas, type PanelSelection } from "./PanelCanvas";
import { publicationGaps, type ChartContract } from "./views";
import {
  MAX_ELAPSED_TICKS,
  isDirty,
  liveTools,
  missingRequired,
  questionKey,
  readyNote,
  requestId,
  requestIsLost,
  warmingNote,
  type LiveAnswer,
  type LiveAnswerWithPanel,
  type LiveState,
  livePanelAnswer,
} from "./live";
import type { RecommendedChart, Table, Tool } from "./api";
import { PhraseExplorer } from "./PhraseExplorer";
import type { Reading } from "./api";
import { PanelTabs } from "./PanelTabs";
import { post, publicationFigure } from "./api";
import type { FigureFormat, PanelAnswer, PanelInfo } from "./api";
import type {
  PhraseAnswer,
  PhraseRequest,
  PhraseWorkspaceAnswer,
  SavedPhraseQuestion,
  SourcePassage,
} from "./phrase";

/**
 * Analysis that follows the controls, instead of being ordered and waited for.
 *
 * Every analysis used to cost a submitted job: pick a tool, fill a form, wait,
 * open a file. Most of that wait is one thing — parsing. On a twenty-speech
 * corpus the parse takes thirty seconds and the analysis over it takes a fifth
 * of a second. Charging for the parse on every run meant that changing an
 * n-gram's `n` from 2 to 3 cost half a minute, so nobody changed it; people
 * ran the analysis they had already decided on, which is not exploring.
 *
 * Here the parse is paid once, when the documents are chosen. After that the
 * parameters drive the analysis directly — change one, and a new table arrives
 * in a fraction of a second and the chart redraws from it.
 *
 * What it does not do is publish. A live answer has no run directory, no
 * provenance and no artifact. It is the same `execute` a published run calls,
 * given the same corpus and the same parse, so the rows on screen are the rows
 * a publish would write — but until you publish, nothing has been recorded.
 */

export function LiveBench({
  projectId,
  tools,
  state,
  contract,
  onWarm,
  onAnalyse,
  onAbort,
  onTrackPhrase,
  onReadSource,
  savedQuestions,
  onSaveQuestion,
  onOpenQuestion,
  onDuplicateQuestion,
  onDeleteQuestion,
  onPublishQuestion,
  busy,
  canPublish,
  onPublish,
}: {
  /** Enables corpus evidence lookup from clickable purpose-built panel marks. */
  projectId?: string | null;
  tools: Tool[];
  state: LiveState;
  contract: ChartContract;
  /** Read and parse the chosen documents. Slow, and only happens once. */
  onWarm: () => void;
  onAnalyse: (
    tool: string,
    params: Params,
    requestId: string,
  ) => Promise<LiveAnswer | null>;
  /**
   * Tell the engine to stop waiting on a request the reader abandoned (R-C8).
   * Best effort: a failure here frees nothing that matters, because the
   * interface has already stopped waiting.
   */
  onAbort?: (requestId: string) => void;
  onTrackPhrase: (request: PhraseRequest) => Promise<PhraseAnswer | null>;
  /** Open the linked source reader at a verified offset of a loaded document. */
  onReadSource: (
    documentId: string,
    characterStart: number,
    characterEnd: number | null,
    radius?: number,
  ) => Promise<SourcePassage | null>;
  savedQuestions: SavedPhraseQuestion[];
  onSaveQuestion: (
    name: string,
    answer: PhraseAnswer,
    request: PhraseRequest,
    comparison: PhraseAnswer | null,
    replacing: string | null,
    expectedRevision: number | null,
  ) => Promise<SavedPhraseQuestion | null>;
  onOpenQuestion: (
    question: SavedPhraseQuestion,
  ) => Promise<PhraseWorkspaceAnswer | null>;
  onDuplicateQuestion: (
    question: SavedPhraseQuestion,
  ) => Promise<SavedPhraseQuestion | null>;
  onDeleteQuestion: (question: SavedPhraseQuestion) => Promise<boolean>;
  onPublishQuestion: (question: SavedPhraseQuestion) => void;
  busy: boolean;
  canPublish: boolean;
  onPublish: (tool: string, params: Params) => void;
}) {
  const available = liveTools(tools);
  const [toolName, setToolName] = useState("");
  const [params, setParams] = useState<Params>({});
  const [answer, setAnswer] = useState<LiveAnswerWithPanel | null>(null);
  /**
   * The question the answer on screen answered.
   *
   * Everything about iteration hangs off this: a control change with a
   * different key marks the bench dirty ("press Run") and costs nothing, and
   * a Run that repeats the shown key is the "run it twice, same topics?"
   * check the topic-modelling homework actually asks for.
   */
  const [shownKey, setShownKey] = useState("");
  const [asking, setAsking] = useState(false);
  const [lost, setLost] = useState(false);
  /**
   * How long the current request has been out.
   *
   * "Working..." said nothing about which request was running or for how
   * long, so a five-second n-gram count and a multi-minute topic model
   * looked identical -- and identical to hung. Ticking seconds make the
   * difference between working and stuck something the reader can see.
   */
  const [elapsed, setElapsed] = useState(0);

  const tool = available.find((item) => item.name === toolName) ?? null;
  // An analysis missing something it cannot run without is not a question yet.
  const needed = tool ? missingRequired(tool, params) : [];
  const asked = tool && !needed.length ? questionKey(tool.name, params) : "";
  const dirty = isDirty(shownKey, asked);

  // The question on screen, readable from inside a response handler without
  // making that handler a new closure on every keystroke.
  const current = useRef(asked);
  current.current = asked;
  // Whether a request is out. A ref: it gates Run, and gating must not draw.
  const inFlight = useRef(false);
  const startedAt = useRef(0);
  /** The request id of the question in flight, so it can be abandoned (R-C8). */
  const sentId = useRef("");

  /**
   * The only way an analysis starts: the reader pressing Run (contract R-C1).
   *
   * No effect asks on the user's behalf. That design meant a parameter edit
   * during a request could strand the bench on "Working..." forever -- a
   * setting change is now free and silent, and running is always an explicit
   * act. One in flight at a time (R-C3): bigrams over 600,000 tokens take
   * fifteen seconds, and two of those slow each other down to answer a
   * question nobody is waiting for any more.
   */
  const run = () => {
    if (!tool || needed.length || state.state !== "ready") return;
    if (inFlight.current) return;
    const mine = asked;
    const id = requestId();
    inFlight.current = true;
    sentId.current = id;
    setAsking(true);
    setLost(false);
    setElapsed(0);
    startedAt.current = Date.now();
    void onAnalyse(tool.name, params, id)
      .then((result) => {
        // Dropped unless it still answers the question on screen (R-C2).
        if (result && current.current === mine) {
          const { panel, ...answerFields } = result;
          setAnswer({ ...answerFields, panel: livePanelAnswer(panel) });
          setShownKey(mine);
        }
      })
      .finally(() => {
        inFlight.current = false;
        sentId.current = "";
        setAsking(false);
      });
  };

  // A tick state to re-render each second while a request is out. The
  // interval self-stops at a generous bound: a live answer has never taken
  // this long, and a timer that never stops outlives its reader's interest
  // (and every timer-based test). Elapsed derives from the start time, so a
  // suspended tab catches up instead of lying.
  useEffect(() => {
    if (!asking) return;
    const tick = setInterval(() => {
      const seconds = Math.round((Date.now() - startedAt.current) / 1000);
      setElapsed(seconds);
      // A request past the bound is treated as lost, not waited on (R-C3):
      // the single analysis slot is released and the reader is offered an
      // explicit Retry -- a dropped fetch (tab sleep, a proxy reset, a
      // dev-server reload) used to leave the slot held forever, and the
      // bench read as "Working..." to nobody.
      if (seconds >= MAX_ELAPSED_TICKS || requestIsLost(seconds * 1000)) {
        clearInterval(tick);
        if (inFlight.current) {
          // The reader stopped waiting: say so on the engine side too (R-C8),
          // so a dead question does not keep a slot warm nobody is watching.
          onAbort?.(sentId.current);
          inFlight.current = false;
          sentId.current = "";
          setAsking(false);
          setLost(true);
        }
      }
    }, 1000);
    return () => clearInterval(tick);
  }, [asking]);

  // Abandon an in-flight question when the bench goes away (R-C8): closing
  // the page mid-run must not leave the engine answering to nobody. The
  // callback rides in a ref so this effect runs once and cannot re-subscribe
  // on every render.
  const abortRef = useRef(onAbort);
  abortRef.current = onAbort;
  useEffect(() => {
    return () => {
      if (inFlight.current && sentId.current) {
        abortRef.current?.(sentId.current);
        inFlight.current = false;
        sentId.current = "";
      }
    };
  }, []);

  // Ctrl/Cmd+Enter runs, the way every "apply" control in an editor does.
  // A form that handled the keystroke first (the run dialog) marks it
  // defaultPrevented and is left alone.
  const runRef = useRef(run);
  runRef.current = run;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (!(event.ctrlKey || event.metaKey) || event.key !== "Enter") return;
      event.preventDefault();
      runRef.current();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const choose = (name: string) => {
    const picked = available.find((item) => item.name === name);
    setToolName(name);
    setParams(picked ? defaultParams(picked) : {});
    setAnswer(null);
    // Nothing has been answered for this analysis yet, whatever the last one
    // answered. The bench starts idle, never auto-running (R-C1).
    setShownKey("");
    setLost(false);
  };

  return (
    <>
      <div className="live-corpus">
        <span className="live-corpus-state">
          <Flame size={16} aria-hidden="true" />
          {state.state === "ready" ? (
            <span>{readyNote(state)}</span>
          ) : state.state === "warming" ? (
            <span>{warmingNote(state)}</span>
          ) : state.state === "failed" ? (
            <span className="live-failed">{state.error}</span>
          ) : (
            <span>
              Nothing is loaded yet. Reading and parsing your documents is the
              one slow step — after it, changing an analysis or a setting is
              immediate.
            </span>
          )}
        </span>
        <button
          className={state.state === "ready" ? "secondary" : "primary"}
          disabled={busy || state.state === "warming"}
          onClick={onWarm}
        >
          <Play size={15} />
          {state.state === "ready" ? "Reload documents" : "Load documents"}
        </button>
      </div>

      {/* Outside the ready gate on purpose. Reopening a saved question loads
          the documents it needs, so requiring documents to be loaded first
          made the shelf unreachable at exactly the moment it is most useful:
          opening the app to continue yesterday's work. */}
      <PhraseExplorer
        ready={state.state === "ready"}
        onTrack={onTrackPhrase}
        onReadSource={onReadSource}
        savedQuestions={savedQuestions}
        onSaveQuestion={onSaveQuestion}
        onOpenQuestion={onOpenQuestion}
        onDuplicateQuestion={onDuplicateQuestion}
        onDeleteQuestion={onDeleteQuestion}
        onPublishQuestion={onPublishQuestion}
        canPublish={canPublish && !busy}
      />

      {state.state === "ready" && (
        <>
          <div className="live-controls panel">
            <label className="field-label live-tool">
              Analysis
              <select
                aria-label="Analysis to run"
                value={toolName}
                onChange={(event) => choose(event.target.value)}
              >
                <option value="">Choose an analysis…</option>
                {available.map((item) => (
                  <option key={item.name} value={item.name}>
                    {item.label}
                  </option>
                ))}
              </select>
              <small>
                A general analysis over the loaded documents — separate from
                Track a phrase above.
              </small>
              {tool && <small>{tool.description}</small>}
            </label>

            {tool && (
              <ParamFields
                tool={tool}
                params={params}
                onChange={setParams}
                disabled={busy}
              />
            )}
          </div>

          {/* The one door out of the controls (R-C1). Disabled while a
              request is out, so one analysis at a time is a property of the
              button and not of a hidden scheduler. */}
          {tool && (
            <div className="live-run">
              <button
                type="button"
                className="primary"
                onClick={run}
                disabled={
                  !!busy ||
                  asking ||
                  needed.length > 0 ||
                  state.state !== "ready"
                }
              >
                <Zap size={15} aria-hidden="true" />
                {asking ? "Working..." : lost ? "Run again" : "Run analysis"}
              </button>
              {dirty && !asking && (
                <span className="live-dirty" role="status">
                  Settings changed — press Run to see them.
                </span>
              )}
            </div>
          )}

          {tool && needed.length > 0 && (
            <p className="live-status" role="status">
              <Zap size={14} aria-hidden="true" />
              Fill in {needed.join(" and ")} — then press Run.
            </p>
          )}

          {tool && !needed.length && (
            <p className="live-status" role="status">
              {asking ? (
                <>
                  <Zap size={14} aria-hidden="true" /> Working{" "}
                  <span className="live-elapsed">{elapsed}s</span>
                </>
              ) : answer?.ok ? (
                <>
                  <Zap size={14} aria-hidden="true" />
                  {answer.table?.total.toLocaleString()} row
                  {answer.table?.total === 1 ? "" : "s"} in {answer.elapsed_ms}
                  ms. Nothing has been saved — change anything above and press
                  Run when you want to see it.
                </>
              ) : null}
            </p>
          )}

          {lost && (
            <p className="alert warning" role="status">
              <TriangleAlert size={16} aria-hidden="true" />
              <span>
                This analysis was asked {elapsed}s ago and the answer never
                arrived (R-C3). The slot is free again — press Run to ask once
                more, or reload the documents if the wait repeats.
              </span>
            </p>
          )}

          {answer && !answer.ok && (
            <p className="alert warning" role="status">
              <TriangleAlert size={16} aria-hidden="true" />
              <span>
                {answer.diagnostics.map((d) => d.message).join(" ") ||
                  "That combination of settings produced no result."}
              </span>
            </p>
          )}

          {tool && answer?.ok && answer.table && (
            <LiveChart
              key={answer.tool}
              projectId={projectId}
              table={answer.table}
              path={answer.path}
              reading={answer}
              panel={answer.panel ?? undefined}
              panels={(answer.panels ?? []) as unknown as PanelInfo[]}
              answerId={answer.answer_id ?? ""}
              contract={contract}
              canPublish={canPublish && !busy}
              onPublish={() => onPublish(tool.name, params)}
            />
          )}
        </>
      )}
    </>
  );
}

/**
 * The chart over a live table.
 *
 * Its own component so the settings survive a new answer with the same shape:
 * changing `min-count` returns a table with the same columns, and re-picking
 * the axes on every keystroke would make the chart unusable. `useChartSettings`
 * resets only when the columns change, which is the moment the old axes stop
 * naming anything.
 */
function LiveChart({
  projectId,
  table,
  path,
  reading,
  panel,
  panels = [],
  answerId = "",
  contract,
  canPublish,
  onPublish,
}: {
  projectId?: string | null;
  table: Table;
  path: string;
  /** What this table says, and the questions worth putting to it. */
  reading: Reading;
  panel?: PanelAnswer;
  /** Every figure this answer can draw; the first is `panel`. */
  panels?: PanelInfo[];
  /** Cited to redraw a figure with other settings. */
  answerId?: string;
  contract: ChartContract;
  canPublish: boolean;
  onPublish: () => void;
}) {
  const recommended = reading.recommended_charts ?? [];
  const { settings, setSettings, selected, setSelected } = useChartSettings(
    table,
    recommended,
  );
  const [showGeneric, setShowGeneric] = useState(false);
  const [requested, setRequested] = useState<{
    panel: string;
    nonce: number;
  } | null>(null);
  const drawLive = (info: PanelInfo, params: Record<string, unknown>) =>
    post<PanelAnswer>(`/projects/${projectId}/live/panel`, {
      answer_id: answerId,
      panel: info.name,
      params,
    });
  const publishLive = (
    info: PanelInfo,
    params: Record<string, unknown>,
    format: FigureFormat,
    dpi: number,
  ) =>
    publicationFigure(`/projects/${projectId}/live/panel/static`, {
      answer_id: answerId,
      panel: info.name,
      params,
      format,
      dpi,
    });
  const draw = (chart: RecommendedChart) => {
    const next = chartSettings(chart);
    if (!next) return;
    setSettings(next);
    setShowGeneric(true);
    // The old drill-down belonged to the old chart's marks. Carrying it over
    // would filter the new chart's rows by a bar that is no longer drawn.
    setSelected(null);
  };
  return (
    <>
      {/* Above the chart, because it decides whether the chart is worth
          reading. A caution met after acting is not a caution. */}
      <ReadingPanel
        reading={reading}
        heading="What this says"
        note="Read from every row of this answer, not just the rows on this page."
        onChart={draw}
        onFigure={(name) =>
          setRequested((last) => ({
            panel: name,
            nonce: (last?.nonce ?? 0) + 1,
          }))
        }
      />
      {panel ? (
        <>
          <section className="panel panel-section" aria-label="Live figure">
            <div className="section-title">
              <div>
                <h2>
                  <Sparkles size={16} aria-hidden="true" /> Purpose-built live
                  figure
                </h2>
                <p>
                  Drawn from this answer's result shape, with every mark
                  clickable.
                </p>
              </div>
            </div>
            {projectId && answerId && panels.length ? (
              <PanelTabs
                panels={panels}
                first={panel}
                draw={drawLive}
                publish={publishLive}
                projectId={projectId}
                resetKey={answerId}
                requested={requested}
              />
            ) : (
              <PanelAnswerDisplay answer={panel} />
            )}
            <button
              type="button"
              className="text-button"
              aria-expanded={showGeneric}
              onClick={() => setShowGeneric((shown) => !shown)}
            >
              {showGeneric
                ? "Hide generic chart workbench"
                : "Open generic chart workbench (expert)"}
            </button>
          </section>
          {showGeneric && (
            <Workbench
              table={table}
              artifactPath={path || "this analysis"}
              settings={settings}
              onSettings={setSettings}
              selected={selected}
              onSelect={setSelected}
              canPublish={canPublish}
              onPublish={onPublish}
              gaps={publicationGaps(contract, settings)}
              headingLevel={2}
            />
          )}
        </>
      ) : (
        <Workbench
          table={table}
          artifactPath={path || "this analysis"}
          settings={settings}
          onSettings={setSettings}
          selected={selected}
          onSelect={setSelected}
          canPublish={canPublish}
          onPublish={onPublish}
          gaps={publicationGaps(contract, settings)}
          headingLevel={2}
        />
      )}
      <p className="muted live-provenance">
        <Sparkles size={14} aria-hidden="true" /> These are the rows a published
        run would write — the same engine, over the same documents. Publishing
        records them with their inputs, settings and diagnostics; until then
        nothing here is saved.
      </p>
    </>
  );
}

/** A graceful rendering path for callers that have no project evidence API. */
function PanelAnswerDisplay({ answer }: { answer: PanelAnswer }) {
  const [selected, setSelected] = useState<PanelSelection>(null);
  if (!answer.ok) {
    return <Diagnostics items={answer.diagnostics} />;
  }
  return (
    <>
      <Diagnostics items={answer.diagnostics} />
      <PanelCanvas
        prepared={answer}
        selected={selected}
        onSelect={setSelected}
      />
      {selected && (
        <p className="muted">
          {answer.marks.find((mark) => mark.key === selected.key)?.evidence
            .describe ?? "This mark's evidence is no longer available."}
        </p>
      )}
      <p className="muted">
        Connect this view to its project to open source passages for a selected
        mark.
      </p>
    </>
  );
}
