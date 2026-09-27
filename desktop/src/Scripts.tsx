import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  ArrowDown,
  ArrowUp,
  BookOpen,
  Bot,
  ChevronDown,
  ChevronRight,
  Code2,
  Copy,
  CopyPlus,
  Download,
  FileUp,
  LoaderCircle,
  Play,
  Plus,
  RotateCcw,
  Save,
  Search,
  Square,
  SquareCode,
  Trash2,
  TriangleAlert,
  Type,
  X,
} from "lucide-react";

import { artifactBlobUrl, artifactText, download, when, type Job } from "./api";
import { ChartCanvas } from "./ChartCanvas";
import { buildLayout } from "./chartLayout";
import { CodeEditor } from "./CodeEditor";
import { Markdown } from "./Markdown";
import { ResultTable } from "./ResultTable";
import {
  applyEvents,
  cameFromOutside,
  createNotebook,
  deleteNotebook,
  duplicateNotebook,
  endUnfinished,
  executeCell,
  exportPath,
  getNotebook,
  guidePath,
  kernelEvents,
  kernelFilePath,
  lintCode,
  listNotebooks,
  matplotlibCode,
  newCell,
  parseIpynb,
  runAndSave,
  saveNotebook,
  scriptCorpus,
  scriptReference,
  shownChartSettings,
  stopKernel,
  type CellOutput,
  type CellRun,
  type CorpusSummary,
  type LintWarning,
  type Notebook,
  type NotebookCell,
  type ScriptReference,
} from "./notebooks";

/** How often a running cell's output is read back. */
const POLL_MS = 350;
/** How long after the last keystroke a notebook is saved. */
const SAVE_DELAY_MS = 1200;

const errorText = (caught: unknown) =>
  caught instanceof Error ? caught.message : String(caught);

/** A kernel figure, fetched with the app's credentials (an <img src> has none). */
function KernelImage({ path, alt }: { path: string; alt: string }) {
  const [url, setUrl] = useState("");
  const [failed, setFailed] = useState("");
  useEffect(() => {
    let alive = true;
    let made = "";
    artifactBlobUrl(path)
      .then((value) => {
        made = value;
        if (alive) setUrl(value);
        else URL.revokeObjectURL(value);
      })
      .catch((caught) => alive && setFailed(errorText(caught)));
    return () => {
      alive = false;
      if (made) URL.revokeObjectURL(made);
    };
  }, [path]);
  if (failed)
    return <p className="muted">The figure could not be shown: {failed}</p>;
  if (!url) return <p className="muted">Loading the figure…</p>;
  return <img className="cell-figure" src={url} alt={alt} />;
}

/** One shown table: the first rows, and the chart beside it. */
function TableOutput({
  output,
  filePath,
}: {
  output: Extract<CellOutput, { kind: "table" }>;
  filePath: (name: string) => string;
}) {
  const [copied, setCopied] = useState(false);
  const [view, setView] = useState<"chart" | "table">(
    output.chart ? "chart" : "table",
  );
  const settings = output.chart ? shownChartSettings(output.chart) : null;
  const rows = output.chart_rows?.rows ?? output.preview.rows;
  const layout = settings ? buildLayout(rows, settings) : null;
  const heading = output.title || output.name.replaceAll("_", " ");
  const csv = `${output.name}.csv`;
  const copyCode = async () => {
    if (!output.chart) return;
    await navigator.clipboard.writeText(
      matplotlibCode(output.chart, csv, output.title),
    );
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };
  const { total, truncated } = output.preview;
  return (
    <div className="cell-table">
      <div className="cell-table-head">
        <strong>{heading}</strong>
        <span className="muted">
          {total.toLocaleString()} row{total === 1 ? "" : "s"}
          {output.saved ? " · kept with the results" : ""}
        </span>
        <span className="cell-table-actions">
          {settings && (
            <span
              className="segmented"
              role="group"
              aria-label={`Show ${heading} as`}
            >
              <button
                className={view === "chart" ? "selected" : ""}
                onClick={() => setView("chart")}
              >
                Chart
              </button>
              <button
                className={view === "table" ? "selected" : ""}
                onClick={() => setView("table")}
              >
                Table
              </button>
            </span>
          )}
          {output.chart && (
            <button
              className="secondary compact"
              onClick={() => void copyCode()}
              title="About a dozen lines of matplotlib that draw this chart from the CSV"
            >
              <Code2 size={14} /> {copied ? "Copied" : "Copy matplotlib code"}
            </button>
          )}
          <button
            className="secondary compact"
            onClick={() => void download(filePath(output.file), csv)}
          >
            <Download size={14} /> CSV
          </button>
        </span>
      </div>
      {output.chart?.question && view === "chart" && (
        <p className="cell-chart-question">{output.chart.question}</p>
      )}
      {view === "chart" && layout?.ok && (
        <div className="cell-chart">
          <ChartCanvas
            layout={layout.layout}
            selected={null}
            onSelect={() => undefined}
            title={heading}
            name={output.name}
          />
        </div>
      )}
      {view === "chart" && layout && !layout.ok && (
        <p className="muted">No chart: {layout.reason}</p>
      )}
      {output.chart_note && (
        <p className="muted cell-note">{output.chart_note}</p>
      )}
      {view === "table" && (
        <ResultTable
          columns={output.preview.columns}
          rows={output.preview.rows}
          copyable
          footerNote={
            truncated
              ? `The first ${output.preview.rows.length.toLocaleString()} rows. The CSV has all ${total.toLocaleString()}.`
              : undefined
          }
        />
      )}
    </div>
  );
}

function Outputs({
  run,
  filePath,
}: {
  run: CellRun;
  filePath: (name: string) => string;
}) {
  const [traceOpen, setTraceOpen] = useState(false);
  return (
    <div className="cell-outputs" aria-live="polite">
      {run.outputs.map((output, index) => {
        if (output.kind === "stream")
          return (
            <pre key={index} className={`cell-stream ${output.name}`}>
              {output.text}
            </pre>
          );
        if (output.kind === "text")
          return (
            <pre key={index} className="cell-text">
              {output.text}
            </pre>
          );
        if (output.kind === "figure")
          return (
            <div key={index} className="cell-figure-wrap">
              <KernelImage path={filePath(output.png)} alt={output.name} />
              <span className="cell-figure-actions">
                <button
                  className="secondary compact"
                  onClick={() =>
                    void download(filePath(output.png), `${output.name}.png`)
                  }
                >
                  <Download size={14} /> PNG
                </button>
                <button
                  className="secondary compact"
                  onClick={() =>
                    void download(filePath(output.svg), `${output.name}.svg`)
                  }
                >
                  <Download size={14} /> SVG
                </button>
              </span>
            </div>
          );
        return <TableOutput key={index} output={output} filePath={filePath} />;
      })}
      {run.error && (
        <div className="cell-error" role="alert">
          <strong>{run.error.type}</strong> {run.error.message}
          {run.error.trace.length > 0 && (
            <>
              <button
                className="link-button"
                onClick={() => setTraceOpen((open) => !open)}
                aria-expanded={traceOpen}
              >
                {traceOpen ? "Hide where" : "Where"}
              </button>
              {traceOpen && <pre>{run.error.trace.join("\n")}</pre>}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function RunStatus({ run, now }: { run: CellRun | undefined; now: number }) {
  if (!run) return <span className="cell-status">Not run</span>;
  if (run.state === "queued")
    return <span className="cell-status">Waiting</span>;
  if (run.state === "running")
    return (
      <span className="cell-status running">
        <LoaderCircle size={13} className="spin" /> Running{" "}
        {Math.max(0, Math.round((now - run.started) / 1000))}s
      </span>
    );
  const seconds = run.seconds == null ? "" : ` in ${run.seconds.toFixed(1)}s`;
  if (run.state === "error" && run.error?.type === "Stopped")
    return <span className="cell-status failed">Stopped</span>;
  return run.state === "error" ? (
    <span className="cell-status failed">Stopped with an error{seconds}</span>
  ) : (
    <span className="cell-status done">Done{seconds}</span>
  );
}

/** The "read this before you run it" note on code that came from outside the app. */
function OutsideBanner({
  children,
  onDismiss,
  warnings,
}: {
  children: ReactNode;
  onDismiss?: () => void;
  warnings: LintWarning[];
}) {
  return (
    <div className="outside-banner" role="note">
      <TriangleAlert size={16} />
      <div>
        <p>{children}</p>
        {warnings.length > 0 && (
          <ul>
            {warnings.map((warning, index) => (
              <li key={index}>
                {warning.line ? `Line ${warning.line}: ` : ""}
                {warning.message}
              </li>
            ))}
          </ul>
        )}
      </div>
      {onDismiss && (
        <button className="secondary compact" onClick={onDismiss}>
          I have read it
        </button>
      )}
    </div>
  );
}

function Cell({
  cell,
  index,
  count,
  run,
  now,
  busy,
  editing,
  reference,
  columns,
  insert,
  focusSignal,
  warnings,
  filePath,
  onChange,
  onRun,
  onKind,
  onMove,
  onDelete,
  onAddBelow,
  onEdit,
  onFocus,
  onTrust,
}: {
  cell: NotebookCell;
  index: number;
  count: number;
  run: CellRun | undefined;
  now: number;
  busy: boolean;
  editing: boolean;
  reference: ScriptReference | null;
  columns: string[];
  insert: { text: string } | null;
  focusSignal: number;
  warnings: LintWarning[];
  filePath: (name: string) => string;
  onChange: (source: string) => void;
  onRun: (advance: boolean) => void;
  onKind: (kind: "code" | "markdown") => void;
  onMove: (by: -1 | 1) => void;
  onDelete: () => void;
  onAddBelow: (kind: "code" | "markdown") => void;
  onEdit: (editing: boolean) => void;
  onFocus: () => void;
  onTrust: () => void;
}) {
  const code = cell.cell_type === "code";
  const position = index + 1;
  const pasted = cell.metadata.nlpsuite?.origin === "pasted";
  return (
    <section
      className={`notebook-cell ${code ? "code" : "markdown"}`}
      aria-label={`Cell ${position}`}
      onFocusCapture={onFocus}
    >
      <div className="cell-bar">
        <span className="cell-number">{position}</span>
        <span
          className="segmented"
          role="group"
          aria-label={`Cell ${position} type`}
        >
          <button
            className={code ? "selected" : ""}
            onClick={() => onKind("code")}
          >
            Python
          </button>
          <button
            className={!code ? "selected" : ""}
            onClick={() => onKind("markdown")}
          >
            Markdown
          </button>
        </span>
        {code && <RunStatus run={run} now={now} />}
        <span className="cell-tools">
          {code ? (
            <button
              className="icon-button"
              aria-label={`Run cell ${position}`}
              title="Run (Shift+Enter runs and moves on; Ctrl+Enter stays)"
              disabled={busy}
              onClick={() => onRun(false)}
            >
              <Play size={15} />
            </button>
          ) : (
            <button
              className="icon-button"
              aria-label={
                editing
                  ? `Finish editing cell ${position}`
                  : `Edit cell ${position}`
              }
              onClick={() => onEdit(!editing)}
            >
              {editing ? <BookOpen size={15} /> : <Type size={15} />}
            </button>
          )}
          <button
            className="icon-button"
            aria-label={`Move cell ${position} up`}
            disabled={index === 0}
            onClick={() => onMove(-1)}
          >
            <ArrowUp size={15} />
          </button>
          <button
            className="icon-button"
            aria-label={`Move cell ${position} down`}
            disabled={index === count - 1}
            onClick={() => onMove(1)}
          >
            <ArrowDown size={15} />
          </button>
          <button
            className="icon-button"
            aria-label={`Delete cell ${position}`}
            onClick={onDelete}
          >
            <Trash2 size={15} />
          </button>
        </span>
      </div>
      {pasted && (
        <OutsideBanner onDismiss={onTrust} warnings={warnings}>
          This code was pasted in from outside the app. Read it before running
          it: it runs with the same rights as the app.
        </OutsideBanner>
      )}
      {code || editing ? (
        <CodeEditor
          value={cell.source}
          onChange={onChange}
          language={code ? "python" : "markdown"}
          onRun={code ? onRun : () => onEdit(false)}
          functions={reference?.functions ?? []}
          columns={columns}
          label={`Cell ${position} ${code ? "code" : "text"}`}
          placeholder={
            code ? "nlp.  (Shift+Enter to run)" : "Notes, in Markdown"
          }
          insert={insert}
          focusSignal={focusSignal}
        />
      ) : (
        <div
          className="cell-markdown"
          onDoubleClick={() => onEdit(true)}
          title="Double-click to edit"
        >
          {cell.source.trim() ? (
            <Markdown source={cell.source} />
          ) : (
            <p className="muted">
              Empty text cell. Double-click to write in it.
            </p>
          )}
        </div>
      )}
      {code && run && (run.outputs.length > 0 || run.error) && (
        <Outputs run={run} filePath={filePath} />
      )}
      <div className="cell-add">
        <button className="link-button" onClick={() => onAddBelow("code")}>
          <Plus size={13} /> Code
        </button>
        <button className="link-button" onClick={() => onAddBelow("markdown")}>
          <Plus size={13} /> Text
        </button>
      </div>
    </section>
  );
}

/** The library reference beside the notebook, and what the corpus has in it. */
function ReferencePanel({
  reference,
  corpus,
  onInsert,
  onClose,
}: {
  reference: ScriptReference | null;
  corpus: CorpusSummary | null;
  onInsert: (text: string) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const needle = query.trim().toLowerCase();
  const functions = (reference?.functions ?? []).filter(
    (item) =>
      !needle ||
      item.name.toLowerCase().includes(needle) ||
      item.doc.toLowerCase().includes(needle),
  );
  return (
    <aside className="script-reference" aria-label="Library reference">
      <div className="script-reference-head">
        <h2>Library</h2>
        <button
          className="icon-button"
          aria-label="Hide the library reference"
          onClick={onClose}
        >
          <X size={17} />
        </button>
      </div>
      <label className="reference-search">
        <Search size={14} />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search functions"
          aria-label="Search the library"
        />
      </label>
      {!reference && <p className="muted">Loading the reference…</p>}
      <ul className="reference-list">
        {functions.map((item) => {
          const expanded = open === item.name;
          const [summary, ...rest] = item.doc.split("\n\n");
          return (
            <li key={item.name}>
              <button
                className="reference-name"
                onClick={() => setOpen(expanded ? null : item.name)}
                aria-expanded={expanded}
              >
                {expanded ? (
                  <ChevronDown size={13} />
                ) : (
                  <ChevronRight size={13} />
                )}
                <code>{item.name}</code>
              </button>
              <p className="reference-summary">{summary}</p>
              {expanded && (
                <div className="reference-detail">
                  <pre>{item.signature}</pre>
                  {rest.length > 0 && (
                    <pre className="reference-doc">{rest.join("\n\n")}</pre>
                  )}
                  {item.members && (
                    <pre className="reference-doc">{item.members}</pre>
                  )}
                  {item.insert && (
                    <button
                      className="secondary compact"
                      onClick={() => onInsert(item.insert!)}
                    >
                      <Plus size={13} /> Insert
                    </button>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
      {corpus && (
        <section className="reference-corpus">
          <h3>Your corpus</h3>
          <p>
            {corpus.documents.toLocaleString()} document
            {corpus.documents === 1 ? "" : "s"}, {corpus.words.toLocaleString()}{" "}
            words
            {corpus.years ? `, ${corpus.years[0]}–${corpus.years[1]}` : ""}.
          </p>
          <p className="muted">
            <code>corpus.documents</code> has these columns;{" "}
            <code>filter()</code> and <code>x=</code> take their names:
          </p>
          <ul className="reference-columns">
            {corpus.columns.map((column) => (
              <li key={column}>
                <code>{column}</code>
                {corpus.samples[column]?.length ? (
                  <span className="muted">
                    {" "}
                    e.g. {corpus.samples[column].slice(0, 3).join(", ")}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      )}
    </aside>
  );
}

/** "Get help from an AI chatbot": the guide to copy, and a box for the code that comes back. */
function GuideDialog({
  projectId,
  onClose,
  onAddCell,
}: {
  projectId: string;
  onClose: () => void;
  onAddCell: (code: string) => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [version, setVersion] = useState<"short" | "full">("short");
  const [includeCorpus, setIncludeCorpus] = useState(true);
  const [size, setSize] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const [pasted, setPasted] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    const el = ref.current;
    if (el && typeof el.showModal === "function" && !el.open) el.showModal();
    return () => el?.close?.();
  }, []);
  const path = guidePath(projectId, version, includeCorpus);
  useEffect(() => {
    let alive = true;
    setSize(null);
    artifactText(path)
      .then((text) => alive && setSize(text.length))
      .catch((caught) => alive && setError(errorText(caught)));
    return () => {
      alive = false;
    };
  }, [path]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(await artifactText(path));
      setCopied(true);
    } catch (caught) {
      setError(errorText(caught));
    }
  };
  return (
    <dialog
      ref={ref}
      className="guide-dialog"
      aria-label="Get help from an AI chatbot"
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
    >
      <div className="dialog-heading">
        <h2>Get help from an AI chatbot</h2>
        <button
          className="icon-button"
          aria-label="Close dialog"
          onClick={onClose}
        >
          <X size={20} />
        </button>
      </div>
      <p className="muted">
        The app has no AI in it and sends nothing anywhere. Instead, it writes a
        guide to its own library that you give to the chatbot you already use,
        so the code it writes calls the suite rather than guessing.
      </p>
      {error && (
        <div className="alert error" role="alert">
          <TriangleAlert size={17} />
          <span>{error}</span>
        </div>
      )}
      <ol className="guide-steps">
        <li>
          <strong>Copy the guide.</strong>
          <div className="guide-options">
            <span className="segmented" role="group" aria-label="Guide length">
              <button
                className={version === "short" ? "selected" : ""}
                onClick={() => setVersion("short")}
              >
                Short
              </button>
              <button
                className={version === "full" ? "selected" : ""}
                onClick={() => setVersion("full")}
              >
                Full, with examples
              </button>
            </span>
            <label className="check">
              <input
                type="checkbox"
                checked={includeCorpus}
                onChange={(e) => setIncludeCorpus(e.target.checked)}
              />
              Include my corpus's details (names, dates and counts; never its
              text)
            </label>
          </div>
          <div className="guide-actions">
            <button className="primary" onClick={() => void copy()}>
              <Copy size={15} /> {copied ? "Copied" : "Copy the guide"}
            </button>
            <button
              className="secondary"
              onClick={() =>
                void download(path, "NLP Suite guide for AI.md").catch((c) =>
                  setError(errorText(c)),
                )
              }
            >
              <Download size={15} /> Save as a file
            </button>
            <span className="muted">
              {size == null
                ? "Measuring…"
                : `${size.toLocaleString()} characters`}
              {version === "full"
                ? " · better attached as a file than pasted"
                : ""}
            </span>
          </div>
        </li>
        <li>
          <strong>Paste it into your chatbot, then say what you want.</strong>{" "}
          For example:{" "}
          <em>
            Graph how often immigration is mentioned in each speech over time.
          </em>
        </li>
        <li>
          <strong>Paste the code you got back.</strong> It becomes a new cell,
          marked as coming from outside the app, and it is not run until you run
          it.
          <textarea
            className="guide-paste"
            value={pasted}
            onChange={(e) => setPasted(e.target.value)}
            aria-label="Code from the chatbot"
            placeholder="import nlpsuite as nlp&#10;…"
            rows={7}
          />
          <div className="guide-actions">
            <button
              className="primary"
              disabled={!pasted.trim()}
              onClick={() => {
                onAddCell(pasted.replace(/^```(?:python)?\n?|\n?```\s*$/g, ""));
                onClose();
              }}
            >
              <Plus size={15} /> Add as a new cell
            </button>
          </div>
        </li>
      </ol>
    </dialog>
  );
}

type Working = Pick<Notebook, "id" | "name" | "content" | "revision">;

/**
 * The Scripts page: notebooks whose cells call the suite's tools.
 *
 * Cells run in a kernel -- a separate Python process per open notebook --
 * and what they show stays on this page, like the live bench. "Run and save"
 * runs the saved notebook top to bottom as a job and publishes what it
 * showed as a run, which is how a notebook's figure becomes citable.
 */
export function Scripts({
  projectId,
  projectName,
  parser,
  jobs,
  onOpenJob,
  openGuide = false,
  onGuideOpened,
}: {
  projectId: string;
  projectName: string;
  parser: string;
  jobs: Job[];
  onOpenJob: (job: Job) => void;
  /** Open with the chatbot dialog showing (the Learn page's way in). */
  openGuide?: boolean;
  onGuideOpened?: () => void;
}) {
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [working, setWorking] = useState<Working | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [runs, setRuns] = useState<Record<string, CellRun>>({});
  const [execToCell, setExecToCell] = useState<Record<number, string>>({});
  const [queue, setQueue] = useState<string[]>([]);
  const [reference, setReference] = useState<ScriptReference | null>(null);
  const [corpus, setCorpus] = useState<CorpusSummary | null>(null);
  const [showReference, setShowReference] = useState(true);
  const [guideOpen, setGuideOpen] = useState(openGuide);
  // Code pasted from the dialog before the project had a notebook.
  const pendingCell = useRef<NotebookCell | null>(null);
  useEffect(() => {
    if (!openGuide) return;
    setGuideOpen(true);
    onGuideOpened?.();
  }, [openGuide, onGuideOpened]);
  const [editingText, setEditingText] = useState<Record<string, boolean>>({});
  const [activeCell, setActiveCell] = useState<string | null>(null);
  const [focus, setFocus] = useState<{ id: string; n: number } | null>(null);
  const [insert, setInsert] = useState<{ id: string; text: string } | null>(
    null,
  );
  const [warnings, setWarnings] = useState<Record<string, LintWarning[]>>({});
  const [savedJobId, setSavedJobId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [creating, setCreating] = useState<"" | "new" | "template">("");
  const [newName, setNewName] = useState("");
  const [template, setTemplate] = useState("");
  const [now, setNow] = useState(() => Date.now());
  const cursor = useRef(0);
  const importInput = useRef<HTMLInputElement>(null);
  // The kernel ran something for this notebook, so a fresh kernel means lost variables.
  const hadKernel = useRef(false);
  // The cell Run all started last, so a failure can end the queue.
  const lastStarted = useRef<string | null>(null);

  const running = Object.values(runs).some((run) => run.state === "running");
  // Polled only once the kernel has said which exec number the cell is, so
  // no event arrives for a cell the page cannot yet place.
  const placed = Object.values(runs).some(
    (run) => run.state === "running" && run.exec != null,
  );
  const cells = working?.content.cells ?? [];
  const columns = corpus?.columns ?? [];

  // ---------------------------------------------------------------- load --

  const refreshList = useCallback(async () => {
    const listing = await listNotebooks(projectId);
    setNotebooks(listing);
    return listing;
  }, [projectId]);

  useEffect(() => {
    setNotebooks([]);
    setActiveId(null);
    setWorking(null);
    setRuns({});
    setExecToCell({});
    setQueue([]);
    setSavedJobId(null);
    setCorpus(null);
    setError("");
    setNotice("");
    let alive = true;
    refreshList()
      .then((listing) => {
        if (!alive) return;
        const remembered = localStorageGet(`nlp-notebook-${projectId}`);
        const first = listing.find((n) => n.id === remembered) ?? listing[0];
        if (first) setActiveId(first.id);
      })
      .catch((caught) => alive && setError(errorText(caught)));
    scriptCorpus(projectId)
      .then((summary) => alive && setCorpus(summary))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [projectId, refreshList]);

  useEffect(() => {
    scriptReference()
      .then(setReference)
      .catch((caught) => setError(errorText(caught)));
  }, []);

  useEffect(() => {
    if (!activeId) return;
    let alive = true;
    localStorageSet(`nlp-notebook-${projectId}`, activeId);
    setRuns({});
    setExecToCell({});
    setQueue([]);
    setSavedJobId(null);
    setDirty(false);
    hadKernel.current = false;
    getNotebook(projectId, activeId)
      .then((notebook) => {
        if (!alive) return;
        const pending = pendingCell.current;
        pendingCell.current = null;
        if (pending) {
          // Code pasted before there was a notebook: it replaces the new
          // notebook's starter cell, and is saved like any other edit.
          setWorking({
            ...notebook,
            content: { ...notebook.content, cells: [pending] },
          });
          setDirty(true);
        } else setWorking(notebook);
        setEditingText({});
      })
      .catch((caught) => alive && setError(errorText(caught)));
    return () => {
      alive = false;
    };
  }, [projectId, activeId]);

  // ---------------------------------------------------------------- save --

  const save = useCallback(
    async (current: Working) => {
      setSaving(true);
      try {
        const saved = await saveNotebook(projectId, current);
        setWorking((latest) =>
          latest && latest.id === saved.id
            ? { ...latest, revision: saved.revision }
            : latest,
        );
        setDirty(false);
        setNotebooks((list) =>
          list.map((n) => (n.id === saved.id ? { ...n, ...saved } : n)),
        );
        return saved;
      } finally {
        setSaving(false);
      }
    },
    [projectId],
  );

  useEffect(() => {
    if (!dirty || !working || saving) return;
    const timer = window.setTimeout(() => {
      save(working).catch((caught) => setError(errorText(caught)));
    }, SAVE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [dirty, working, saving, save]);

  const edit = (change: (cells: NotebookCell[]) => NotebookCell[]) => {
    setWorking((current) =>
      current
        ? {
            ...current,
            content: {
              ...current.content,
              cells: change(current.content.cells),
            },
          }
        : current,
    );
    setDirty(true);
  };

  // ------------------------------------------------------------- running --

  // Outputs belong to the notebook on screen (`working`), not the one just
  // chosen (`activeId`): for the render between the two, an output built from
  // the new id asked the new notebook's kernel -- which has none -- for the old
  // notebook's figure.
  const shownId = working?.id ?? null;
  const filePath = useCallback(
    (name: string) => (shownId ? kernelFilePath(projectId, shownId, name) : ""),
    [projectId, shownId],
  );

  const start = useCallback(
    async (cellId: string) => {
      if (!activeId || !working) return;
      const index = working.content.cells.findIndex(
        (cell) => cell.id === cellId,
      );
      const cell = working.content.cells[index];
      if (!cell || cell.cell_type !== "code") return;
      setRuns((current) => ({
        ...current,
        [cellId]: {
          exec: null,
          state: "running",
          outputs: [],
          seconds: null,
          error: null,
          started: Date.now(),
        },
      }));
      try {
        const started = await executeCell(
          projectId,
          activeId,
          String(index + 1),
          cell.source,
          parser,
        );
        if (started.exec === 1 && hadKernel.current)
          setNotice(
            "This notebook's Python had stopped (after 15 idle minutes, or to make room for another notebook), " +
              "so variables from earlier cells are gone. Run all rebuilds them.",
          );
        hadKernel.current = true;
        cursor.current = started.after;
        setExecToCell((current) => ({ ...current, [started.exec]: cellId }));
        setRuns((current) =>
          current[cellId]
            ? {
                ...current,
                [cellId]: { ...current[cellId], exec: started.exec },
              }
            : current,
        );
      } catch (caught) {
        setRuns((current) => ({
          ...current,
          [cellId]: {
            ...current[cellId],
            state: "error",
            error: { type: "Not run", message: errorText(caught), trace: [] },
          },
        }));
        setQueue([]);
      }
    },
    [activeId, working, projectId, parser],
  );

  // Poll while a cell runs.
  useEffect(() => {
    if (!placed || !activeId) return;
    let alive = true;
    const timer = window.setInterval(() => {
      setNow(Date.now());
      kernelEvents(projectId, activeId, cursor.current)
        .then((state) => {
          if (!alive) return;
          cursor.current = state.next;
          if (state.events.length)
            setRuns((current) =>
              applyEvents(current, execToCell, state.events),
            );
          else if (!state.running)
            setRuns((current) =>
              endUnfinished(
                current,
                "The notebook's Python stopped, and its variables went with it.",
              ),
            );
        })
        .catch(() => undefined);
    }, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [placed, activeId, projectId, execToCell]);

  // Run all: the next queued cell starts when the one before it finishes
  // cleanly. One that fails ends the queue -- the cells below would run
  // without what it was meant to make -- and they go back to "not run".
  useEffect(() => {
    if (running || !queue.length) return;
    const last = lastStarted.current;
    if (last && runs[last]?.state === "error") {
      setQueue([]);
      setRuns((current) => endUnfinished(current, ""));
      return;
    }
    const [head, ...rest] = queue;
    lastStarted.current = head;
    setQueue(rest);
    void start(head);
  }, [running, queue, runs, start]);

  const runCell = (cellId: string, advance: boolean) => {
    if (running) return;
    void start(cellId);
    if (advance) {
      const index = cells.findIndex((cell) => cell.id === cellId);
      const next = cells[index + 1];
      if (next) setFocus({ id: next.id, n: Date.now() });
      else {
        const added = newCell("code");
        edit((current) => [...current, added]);
        setFocus({ id: added.id, n: Date.now() });
      }
    }
  };

  const runAll = () => {
    if (running) return;
    const all = cells.filter((cell) => cell.cell_type === "code");
    // Pasted code runs when its reader runs it, not swept up by Run all: the
    // run stops above the first pasted cell nobody has marked as read.
    const unread = all.findIndex(
      (cell) => cell.metadata.nlpsuite?.origin === "pasted",
    );
    const code = (unread === -1 ? all : all.slice(0, unread)).map(
      (cell) => cell.id,
    );
    if (unread !== -1)
      setNotice(
        `Run all stops above cell ${cells.indexOf(all[unread]) + 1}: its code was pasted in from outside the app. ` +
          "Read it, then press “I have read it” or run it by itself.",
      );
    if (!code.length) return;
    setRuns((current) => {
      const next = { ...current };
      for (const id of code)
        next[id] = {
          exec: null,
          state: "queued",
          outputs: [],
          seconds: null,
          error: null,
          started: 0,
        };
      return next;
    });
    lastStarted.current = null;
    setQueue(code);
  };

  const stop = async (reset: boolean) => {
    if (!activeId) return;
    setQueue([]);
    try {
      await stopKernel(projectId, activeId);
    } catch (caught) {
      setError(errorText(caught));
    }
    hadKernel.current = false;
    setRuns((current) =>
      reset
        ? {}
        : endUnfinished(
            current,
            "Stopped. The notebook's variables were cleared; run the cells above it again.",
          ),
    );
    setExecToCell({});
    setNotice(
      reset
        ? "Restarted: every variable is gone and the outputs are cleared."
        : "Stopped. Python cannot be interrupted mid-cell on Windows, so stopping clears the notebook's variables.",
    );
  };

  const saveAsRun = async () => {
    if (!working || !activeId) return;
    try {
      if (dirty) await save(working);
      const job = await runAndSave(projectId, activeId, parser);
      setSavedJobId(job.id);
      setNotice("");
    } catch (caught) {
      setError(errorText(caught));
    }
  };
  const savedJob = jobs.find((job) => job.id === savedJobId) ?? null;

  // Lint what came from outside, so the banner can point at lines worth reading.
  const outsideCode = cells
    .filter(
      (cell) =>
        cell.cell_type === "code" &&
        (cell.metadata.nlpsuite?.origin ||
          (working && cameFromOutside(working.content))),
    )
    .map((cell) => `${cell.id}\u0000${cell.source}`)
    .join("\u0001");
  useEffect(() => {
    if (!outsideCode) {
      setWarnings({});
      return;
    }
    let alive = true;
    const timer = window.setTimeout(() => {
      const pairs = outsideCode
        .split("\u0001")
        .map((pair) => pair.split("\u0000") as [string, string]);
      Promise.all(
        pairs.map(([id, code]) =>
          lintCode(code).then((result) => [id, result.warnings] as const),
        ),
      )
        .then((found) => alive && setWarnings(Object.fromEntries(found)))
        .catch(() => undefined);
    }, 500);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [outsideCode]);

  // ---------------------------------------------------------- notebooks --

  const create = async (body: {
    name: string;
    template?: string;
    imported?: unknown;
  }) => {
    try {
      const made = await createNotebook(projectId, body);
      await refreshList();
      setActiveId(made.id);
      setCreating("");
      setNewName("");
      setTemplate("");
    } catch (caught) {
      setError(errorText(caught));
    }
  };

  const importFile = async (file: File) => {
    try {
      const imported = parseIpynb(await file.text());
      await create({ name: file.name.replace(/\.ipynb$/i, ""), imported });
    } catch (caught) {
      setError(errorText(caught));
    }
  };

  const remove = async () => {
    if (!working) return;
    if (
      !window.confirm(
        `Delete the notebook “${working.name}”? Runs it saved are kept.`,
      )
    )
      return;
    try {
      await deleteNotebook(projectId, working.id);
      const listing = await refreshList();
      setWorking(null);
      setActiveId(listing[0]?.id ?? null);
    } catch (caught) {
      setError(errorText(caught));
    }
  };

  const duplicate = async () => {
    if (!working) return;
    try {
      if (dirty) await save(working);
      const copy = await duplicateNotebook(projectId, working.id);
      await refreshList();
      setActiveId(copy.id);
    } catch (caught) {
      setError(errorText(caught));
    }
  };

  const templates = reference?.templates ?? [];
  const outside = working ? cameFromOutside(working.content) : false;

  const nameTaken = (name: string) =>
    notebooks.some((n) => n.name.toLowerCase() === name.trim().toLowerCase());

  return (
    <div>
      <div className="page-heading">
        <div>
          <span className="eyebrow">WORKSHOP</span>
          <h1>Scripts</h1>
          <p>
            Notebooks whose cells call the suite's tools. What a cell shows
            stays here; <em>Run and save</em> keeps it as a run you can cite.
          </p>
        </div>
      </div>
      {error && (
        <div className="alert error" role="alert">
          <TriangleAlert size={18} />
          <span>{error}</span>
          <button aria-label="Dismiss error" onClick={() => setError("")}>
            <X size={16} />
          </button>
        </div>
      )}
      {notice && (
        <div className="alert notice" role="status">
          <span>{notice}</span>
          <button aria-label="Dismiss notice" onClick={() => setNotice("")}>
            <X size={16} />
          </button>
        </div>
      )}
      <div
        className={`scripts-layout ${showReference ? "with-reference" : ""}`}
      >
        <nav className="notebook-list" aria-label="Notebooks">
          <div className="notebook-list-actions">
            <button
              className="primary"
              onClick={() => setCreating(creating === "new" ? "" : "new")}
            >
              <Plus size={15} /> New notebook
            </button>
            <button
              className="secondary"
              onClick={() =>
                setCreating(creating === "template" ? "" : "template")
              }
            >
              <SquareCode size={15} /> Start from a template
            </button>
            <button
              className="secondary"
              onClick={() => importInput.current?.click()}
            >
              <FileUp size={15} /> Import .ipynb
            </button>
            <input
              ref={importInput}
              type="file"
              accept=".ipynb,application/x-ipynb+json,application/json"
              hidden
              aria-label="Import a Jupyter notebook"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void importFile(file);
                e.target.value = "";
              }}
            />
          </div>
          {creating && (
            <form
              className="notebook-create"
              onSubmit={(e) => {
                e.preventDefault();
                const chosen = templates.find((t) => t.id === template);
                const name = newName.trim() || chosen?.name || "";
                if (!name || (creating === "template" && !chosen)) return;
                void create({
                  name,
                  template: creating === "template" ? template : undefined,
                });
              }}
            >
              {creating === "template" && (
                <div
                  className="template-choices"
                  role="radiogroup"
                  aria-label="Templates"
                >
                  {templates.map((t) => (
                    <label
                      key={t.id}
                      className={template === t.id ? "selected" : ""}
                    >
                      <input
                        type="radio"
                        name="template"
                        value={t.id}
                        checked={template === t.id}
                        onChange={() => {
                          setTemplate(t.id);
                          if (
                            !newName.trim() ||
                            templates.some((o) => o.name === newName)
                          )
                            setNewName(t.name);
                        }}
                      />
                      <strong>{t.name}</strong>
                      <small>{t.description}</small>
                    </label>
                  ))}
                </div>
              )}
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Notebook name"
                aria-label="Notebook name"
                autoFocus
              />
              {nameTaken(newName) && (
                <small className="muted">
                  A notebook with this name exists.
                </small>
              )}
              <div className="notebook-create-actions">
                <button
                  className="primary"
                  type="submit"
                  disabled={
                    nameTaken(newName) ||
                    (creating === "template" ? !template : !newName.trim())
                  }
                >
                  Create
                </button>
                <button
                  className="secondary"
                  type="button"
                  onClick={() => setCreating("")}
                >
                  Cancel
                </button>
              </div>
            </form>
          )}
          <ul>
            {notebooks.map((notebook) => (
              <li key={notebook.id}>
                <button
                  className={
                    notebook.id === activeId
                      ? "notebook-link selected"
                      : "notebook-link"
                  }
                  aria-current={notebook.id === activeId ? "page" : undefined}
                  onClick={() => setActiveId(notebook.id)}
                >
                  <span>{notebook.name}</span>
                  <small>Edited {when(notebook.updated)}</small>
                </button>
              </li>
            ))}
          </ul>
          {!notebooks.length && (
            <p className="muted notebook-empty">
              No notebooks in {projectName || "this project"} yet. A template is
              the quickest start: “Word group over time” graphs any list of
              words across the corpus.
            </p>
          )}
        </nav>

        <div className="notebook-main">
          {working ? (
            <>
              <div className="notebook-toolbar">
                <input
                  className="notebook-name"
                  value={working.name}
                  aria-label="Notebook name"
                  onChange={(e) => {
                    const name = e.target.value;
                    setWorking((current) =>
                      current ? { ...current, name } : current,
                    );
                    if (name.trim()) setDirty(true);
                  }}
                />
                <span className="muted notebook-save-state">
                  {saving
                    ? "Saving…"
                    : dirty
                      ? "Unsaved changes"
                      : `Saved · revision ${working.revision}`}
                </span>
                <div className="notebook-actions">
                  <button
                    className="primary"
                    onClick={runAll}
                    disabled={running || !cells.length}
                  >
                    <Play size={15} /> Run all
                  </button>
                  <button
                    className="secondary"
                    onClick={() => void stop(false)}
                    disabled={!running && !queue.length}
                    title="Ends the notebook's Python: the running cell stops and every variable is cleared"
                  >
                    <Square size={14} /> Stop (clears variables)
                  </button>
                  <button
                    className="secondary"
                    onClick={() => void stop(true)}
                    title="A fresh start: no variables, no outputs"
                  >
                    <RotateCcw size={14} /> Restart
                  </button>
                  <button
                    className="secondary"
                    onClick={() => void saveAsRun()}
                    disabled={
                      running ||
                      (savedJob != null &&
                        ["QUEUED", "RUNNING"].includes(savedJob.state))
                    }
                    title="Runs the saved notebook top to bottom in a fresh Python and keeps what it shows as a run"
                  >
                    <Save size={14} /> Run and save
                  </button>
                  <button
                    className="secondary"
                    onClick={() =>
                      void download(
                        exportPath(projectId, working.id),
                        `${working.name}.zip`,
                      ).catch((c) => setError(errorText(c)))
                    }
                    title="A zip with the notebook, its data and figures, and how to run it outside the app"
                  >
                    <Download size={14} /> Export
                  </button>
                  <button
                    className="secondary"
                    onClick={() => setGuideOpen(true)}
                  >
                    <Bot size={15} /> Get help from an AI chatbot
                  </button>
                  <button
                    className="icon-button"
                    aria-label="Duplicate this notebook"
                    title="Duplicate"
                    onClick={() => void duplicate()}
                  >
                    <CopyPlus size={16} />
                  </button>
                  <button
                    className="icon-button"
                    aria-label="Delete this notebook"
                    title="Delete"
                    onClick={() => void remove()}
                  >
                    <Trash2 size={16} />
                  </button>
                  {!showReference && (
                    <button
                      className="secondary"
                      onClick={() => setShowReference(true)}
                    >
                      <BookOpen size={15} /> Library
                    </button>
                  )}
                </div>
              </div>
              {savedJob && (
                <div
                  className={
                    ["FAILED", "PARTIAL", "CANCELLED"].includes(savedJob.state)
                      ? "notebook-run-status warning"
                      : "notebook-run-status"
                  }
                  role="status"
                >
                  {["QUEUED", "RUNNING"].includes(savedJob.state) ? (
                    <>
                      <LoaderCircle size={15} className="spin" /> Running and
                      saving
                      {savedJob.stage ? `: ${savedJob.stage}` : "…"}
                    </>
                  ) : (
                    <>
                      {savedJob.state === "DONE"
                        ? "Saved as a run."
                        : savedJob.state === "PARTIAL"
                          ? `Saved as a run, but it stopped early: ${savedJob.diagnostics?.[0]?.message ?? "a cell failed"}`
                          : `The run failed: ${savedJob.diagnostics?.[0]?.message ?? savedJob.state}`}
                      {savedJob.run_dir && (
                        <button
                          className="secondary compact"
                          onClick={() => onOpenJob(savedJob)}
                        >
                          Open in Past runs
                        </button>
                      )}
                    </>
                  )}
                </div>
              )}
              {outside && (
                <OutsideBanner warnings={[]}>
                  This notebook came from outside the app (
                  {working.content.metadata.nlpsuite?.created_by === "pasted"
                    ? "pasted"
                    : "imported"}
                  ). Read its code before running it: it runs with the same
                  rights as the app.
                </OutsideBanner>
              )}
              <div className="notebook-cells">
                {cells.map((cell, index) => (
                  <Cell
                    key={cell.id}
                    cell={cell}
                    index={index}
                    count={cells.length}
                    run={runs[cell.id]}
                    now={now}
                    busy={running}
                    editing={!!editingText[cell.id] || !cell.source.trim()}
                    reference={reference}
                    columns={columns}
                    insert={insert?.id === cell.id ? insert : null}
                    focusSignal={focus?.id === cell.id ? focus.n : 0}
                    warnings={warnings[cell.id] ?? []}
                    filePath={filePath}
                    onFocus={() => setActiveCell(cell.id)}
                    onChange={(source) =>
                      edit((current) =>
                        current.map((c) =>
                          c.id === cell.id ? { ...c, source } : c,
                        ),
                      )
                    }
                    onRun={(advance) => runCell(cell.id, advance)}
                    onKind={(kind) =>
                      edit((current) =>
                        current.map((c) =>
                          c.id === cell.id && c.cell_type !== kind
                            ? {
                                ...newCell(kind, c.source),
                                id: c.id,
                                metadata: c.metadata,
                              }
                            : c,
                        ),
                      )
                    }
                    onMove={(by) =>
                      edit((current) => {
                        const next = [...current];
                        const [moved] = next.splice(index, 1);
                        next.splice(index + by, 0, moved);
                        return next;
                      })
                    }
                    onDelete={() =>
                      edit((current) => current.filter((c) => c.id !== cell.id))
                    }
                    onAddBelow={(kind) => {
                      const added = newCell(kind);
                      edit((current) => [
                        ...current.slice(0, index + 1),
                        added,
                        ...current.slice(index + 1),
                      ]);
                      setFocus({ id: added.id, n: Date.now() });
                    }}
                    onEdit={(on) =>
                      setEditingText((current) => ({
                        ...current,
                        [cell.id]: on,
                      }))
                    }
                    onTrust={() =>
                      edit((current) =>
                        current.map((c) =>
                          c.id === cell.id ? { ...c, metadata: {} } : c,
                        ),
                      )
                    }
                  />
                ))}
                <div className="cell-add end">
                  <button
                    className="secondary"
                    onClick={() => {
                      const added = newCell("code");
                      edit((current) => [...current, added]);
                      setFocus({ id: added.id, n: Date.now() });
                    }}
                  >
                    <Plus size={14} /> Code cell
                  </button>
                  <button
                    className="secondary"
                    onClick={() =>
                      edit((current) => [...current, newCell("markdown")])
                    }
                  >
                    <Plus size={14} /> Text cell
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="notebook-placeholder">
              <SquareCode size={34} strokeWidth={1.3} />
              <p>Choose a notebook, or start one from a template.</p>
            </div>
          )}
        </div>

        {showReference && (
          <ReferencePanel
            reference={reference}
            corpus={corpus}
            onClose={() => setShowReference(false)}
            onInsert={(text) => {
              const target =
                cells.find(
                  (cell) => cell.id === activeCell && cell.cell_type === "code",
                ) ??
                [...cells].reverse().find((cell) => cell.cell_type === "code");
              if (target) setInsert({ id: target.id, text });
              else {
                const added = newCell("code", text);
                edit((current) => [...current, added]);
              }
            }}
          />
        )}
      </div>
      {guideOpen && (
        <GuideDialog
          projectId={projectId}
          onClose={() => setGuideOpen(false)}
          onAddCell={(code) => {
            const added = newCell("code", code, "pasted");
            if (working) {
              edit((current) => [...current, added]);
              setFocus({ id: added.id, n: Date.now() });
              return;
            }
            // Opened from Learn with no notebook yet: the code gets one.
            pendingCell.current = added;
            const taken = new Set(notebooks.map((n) => n.name.toLowerCase()));
            let name = "From a chatbot";
            for (let n = 2; taken.has(name.toLowerCase()); n++)
              name = `From a chatbot ${n}`;
            void create({ name });
          }}
        />
      )}
    </div>
  );
}

function localStorageGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function localStorageSet(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // A remembered choice is a convenience; losing it costs one click.
  }
}
