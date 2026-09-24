import { Notices } from "./Notices";
import { Learn } from "./Learn";
import { ProjectManager } from "./ProjectManager";
import { useChartSettings, Workbench } from "./Workbench";
import { CorpusScope } from "./CorpusScope";
import {
  allDocuments,
  selectedDocuments,
  selectionError,
} from "./corpusSelection";
import { Explore, firstWorthOpening, sourceKey } from "./Explore";
import { publishParams, type Settings as ChartSettings } from "./chartLayout";
import {
  publicationGaps,
  toViewSettings,
  type ChartContract,
  type RowFilter,
  type View,
} from "./views";
import { ParamFields, type Params } from "./ParamFields";
import { ResultTable } from "./ResultTable";
import { LiveBench } from "./LiveBench";
import { COLD, type LiveAnswer, type LiveState } from "./live";
import type {
  PhraseAnswer,
  PhraseRequest,
  PhraseWorkspaceAnswer,
  SavedPhraseQuestion,
  SourcePassage,
} from "./phrase";
import { VisualizeDialog, type ChartSeed } from "./VisualizeDialog";
import { ArtifactViewer, viewableArtifact } from "./ArtifactViewer";
import { InsightPanel } from "./InsightPanel";
import { PanelSection } from "./PanelSection";
import { FAMILY_ORDER, ToolCard } from "./ToolCard";
import { Badge, Diagnostics } from "./RunStatus";
import {
  createContext,
  Fragment,
  useContext,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  CheckCircle2,
  Compass,
  ChartColumnBig,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Eye,
  FileText,
  Files,
  FlaskConical,
  FolderOpen,
  Layers3,
  LayoutDashboard,
  LoaderCircle,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
  TriangleAlert,
} from "lucide-react";
import {
  api,
  apiWithTimeout,
  connect,
  download,
  desktopParams,
  initialParams,
  post,
  compareRuns,
  rememberLabels,
  title,
  del,
  upload,
  when,
  sourcePassage,
  type Document,
  type Envelope,
  type CompareAnswer,
  type ImportReport,
  type Job,
  type Project,
  type Setup,
  type Table,
  type TableSource,
  type Tool,
} from "./api";

type Page =
  "overview" | "corpus" | "interactive" | "studio" | "runs" | "learn" | "setup";

/**
 * The sidebar, in three groups.
 *
 * The order inside the middle group is the order the work happens in: ask
 * something and watch it answer, then run the heavier version over everything,
 * then come back to what you ran. It used to be seven flat entries in which
 * "Analyses" and "Visualize" were two doors onto one action -- choose a tool,
 * fill its parameters, queue a job -- while "Explore" held both the live bench
 * and the archive of finished tables, which are not the same activity at all.
 *
 * `section` is only a heading. Reachability is still one click per page, which
 * is what `tests/test_desktop_navigation.py` checks.
 */
const navigation = [
  {
    key: "overview",
    label: "Overview",
    icon: LayoutDashboard,
    section: "WORKSPACE",
  },
  { key: "corpus", label: "Corpus", icon: Files, section: "WORKSPACE" },
  {
    key: "interactive",
    label: "Interactive",
    icon: Compass,
    section: "RESEARCH",
    hint: "Ask, and see the answer change",
  },
  {
    key: "studio",
    label: "Analyze & visualize",
    icon: FlaskConical,
    section: "RESEARCH",
    hint: "Run it over the whole corpus",
  },
  {
    key: "runs",
    label: "Past runs",
    icon: Layers3,
    section: "RESEARCH",
    hint: "Everything you have already produced",
  },
  { key: "learn", label: "Learn", icon: BookOpen, section: "REFERENCE" },
] as const;
/**
 * Where published tables and saved views are shown.
 *
 * Read by the section that draws them and by the two fetches that fill it, so
 * that moving the archive moves its data with it. It used to live on Explore;
 * when it moved to Past runs the fetches stayed behind, and the section drew
 * itself on the right page with nothing in it and no error to explain why.
 */
const ARCHIVE_PAGE = "runs" as const;

const fmt = (value: number) => value.toLocaleString();
const active = (job: Job) => ["QUEUED", "RUNNING"].includes(job.state);

/**
 * The English parsers a run can use.
 *
 * The desktop sent `parser: "spacy"` with every job, so a machine with a
 * working Stanza install had no way to use it, and nothing on screen said
 * which parser a result came from. The engine has accepted both all along.
 */
const PARSERS: { id: string; label: string; model: string; hint: string }[] = [
  {
    id: "spacy",
    label: "spaCy",
    model: "en_core_web_sm",
    hint: "Faster, and enough for most analyses.",
  },
  {
    id: "stanza",
    label: "Stanza",
    model: "stanza_model_en",
    hint: "Slower, with more detailed grammatical annotation.",
  },
];

/** Whether a parser's language model is actually on this machine. */
const parserReady = (setup: Setup | null, id: string): boolean => {
  const parser = PARSERS.find((option) => option.id === id);
  if (!parser || !setup?.components.length) return true;
  return (
    (setup.components.find((c) => c.name === parser.id)?.present ?? true) &&
    (setup.components.find((c) => c.name === parser.model)?.present ?? true)
  );
};

/** A missing component named the way Settings names it, not as an import name. */
const componentLabel = (setup: Setup | null, name: string): string =>
  setup?.components.find((c) => c.name === name)?.label ?? name;

const DialogError = createContext("");
function Dialog({
  label,
  close,
  children,
}: {
  label: string;
  close: () => void;
  children: ReactNode;
}) {
  const error = useContext(DialogError);
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current!;
    el.showModal();
    el.querySelector<HTMLElement>("[autofocus]")?.focus();
    return () => el.close();
  }, []);
  return (
    <dialog
      ref={ref}
      aria-label={label}
      onCancel={(e) => {
        e.preventDefault();
        close();
      }}
    >
      <div className="dialog-heading">
        <h2>{label}</h2>
        <button
          className="icon-button"
          aria-label="Close dialog"
          onClick={close}
        >
          <X size={20} />
        </button>
      </div>
      {error && (
        <div className="alert error" role="alert">
          <TriangleAlert size={17} />
          <span>{error}</span>
        </div>
      )}
      {children}
    </dialog>
  );
}
/**
 * The workbench inside a run's record, which has nothing to save to.
 *
 * Its settings are the component's own: this is the "look at what just
 * finished" reading of a result, not an arrangement anybody is keeping. Explore
 * owns its settings instead, because a saved view has to be able to set them.
 * The hook is the shared half, so neither place invents its own reset rule.
 */
function RunWorkbench({
  table,
  artifactPath,
  contract,
  canPublish,
  onPublish,
  onHandOff,
}: {
  table: Table;
  artifactPath: string;
  contract: ChartContract;
  canPublish: boolean;
  onPublish: (settings: ChartSettings) => void;
  onHandOff?: (seed: ChartSeed) => void;
}) {
  const { settings, setSettings, selected, setSelected } =
    useChartSettings(table);
  return (
    <Workbench
      table={table}
      artifactPath={artifactPath}
      settings={settings}
      onSettings={setSettings}
      selected={selected}
      onSelect={setSelected}
      canPublish={canPublish}
      onPublish={onPublish}
      gaps={publicationGaps(contract, settings)}
      onHandOff={onHandOff}
    />
  );
}

export default function App() {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState("");
  const [resourceUploads, setResourceUploads] = useState(0);
  const [page, setPage] = useState<Page>("overview");
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState("");
  const [documents, setDocuments] = useState<Document[]>([]);
  const [trashedDocuments, setTrashedDocuments] = useState<Document[]>([]);
  const [showTrash, setShowTrash] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [trashedJobs, setTrashedJobs] = useState<Job[]>([]);
  const [showRunTrash, setShowRunTrash] = useState(false);
  const [tools, setToolCatalog] = useState<Tool[]>([]);
  // Accepting a catalog always records its labels, so a run row can name its
  // tool in words even though a job only carries the identifier.
  const setTools = useCallback((catalog: Tool[]) => {
    rememberLabels(catalog);
    setToolCatalog(catalog);
  }, []);
  const [setup, setSetup] = useState<Setup | null>(null);
  const [newProject, setNewProject] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [preview, setPreview] = useState<Document | null>(null);
  const [query, setQuery] = useState("");
  const [analysis, setAnalysis] = useState<Tool | null>(null);
  const [corpusSelection, setCorpusSelection] = useState(allDocuments);
  const [visualizing, setVisualizing] = useState(false);
  // Set when a recommended chart is accepted, so the dialog opens on that
  // chart instead of on its defaults. Cleared whenever the dialog is opened
  // from the toolbar, which is a request to choose freely.
  const [chartSeed, setChartSeed] = useState<ChartSeed | null>(null);
  const [figureRequest, setFigureRequest] = useState<{
    panel: string;
    nonce: number;
  } | null>(null);
  const [viewing, setViewing] = useState(false);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [parser, setParser] = useState("spacy");

  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  /** The run a "Compare with…" click armed; null when no comparison is open. */
  const [compareArmed, setCompareArmed] = useState<Job | null>(null);
  const [compareView, setCompareView] = useState<CompareAnswer | null>(null);
  const [compareBusy, setCompareBusy] = useState(false);
  const [envelope, setEnvelope] = useState<Envelope | null>(null);
  const [table, setTable] = useState<Table | null>(null);
  const [tableLoading, setTableLoading] = useState(false);
  const [artifactIndex, setArtifactIndex] = useState(0);
  const [tableQuery, setTableQuery] = useState("");
  const [tableOffset, setTableOffset] = useState(0);
  // The interaction lab's own selection, deliberately separate from the Runs
  // pane's artifact: looking at a readability table on Explore should not
  // change which run is open under Runs & results, or the reverse.
  const [sources, setSources] = useState<TableSource[]>([]);
  const [sourcesLoading, setSourcesLoading] = useState(false);
  const [source, setSource] = useState<TableSource | null>(null);
  const [sourceTable, setSourceTable] = useState<Table | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [compareSource, setCompareSource] = useState<TableSource | null>(null);
  const [comparison, setComparison] = useState<{
    key: string;
    table: Table | null;
  } | null>(null);
  const comparisonKey = compareSource
    ? `${projectId}:${sourceKey(compareSource)}`
    : "";
  // Saved views, and the words for what publishing will not reproduce. The
  // contract is fetched once per session: it describes the build, not the
  // project, so refetching it per project would be asking the same question
  // repeatedly and getting the same answer.
  const [views, setViews] = useState<View[]>([]);
  const [savedQuestions, setSavedQuestions] = useState<SavedPhraseQuestion[]>(
    [],
  );
  // How warm this project's corpus is. Parsing is the one slow step in a live
  // analysis, so its progress is app state rather than something hidden inside
  // the bench: the reader is waiting on it.
  const [live, setLive] = useState<LiveState>(COLD);
  const [contract, setContract] = useState<ChartContract>({});
  const [importReport, setImportReport] = useState<ImportReport | null>(null);
  const [dragging, setDragging] = useState(false);
  const filesRef = useRef<HTMLInputElement>(null);
  const restoreRef = useRef<HTMLInputElement>(null);
  const [toolQuery, setToolQuery] = useState("");
  const folderRef = useRef<HTMLInputElement>(null);
  const project = projects.find((p) => p.id === projectId);
  const running = jobs.filter(active).length;
  const completed = jobs.filter((j) => j.run_dir).length;

  const perform = useCallback(
    async (label: string, work: () => Promise<void>) => {
      setBusy(label);
      setError("");
      setNotice("");
      try {
        await work();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy("");
      }
    },
    [],
  );

  useEffect(() => {
    let alive = true;
    connect()
      .then(async () => {
        const [p, t, s] = await Promise.all([
          apiWithTimeout<Project[]>("/projects", 30),
          apiWithTimeout<Tool[]>("/tools", 30),
          apiWithTimeout<Setup>("/setup", 30),
        ]);
        if (!alive) return;
        setProjects(p);
        setTools(t);
        setSetup(s);
        const saved = localStorage.getItem("nlp-project");
        setProjectId(
          p.some((item) => item.id === saved) ? saved! : p[0]?.id || "",
        );
        setReady(true);
      })
      .catch((e) => {
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    setDocuments([]);
    setCorpusSelection(allDocuments());
    setJobs([]);
    setSelectedJob(null);
    setEnvelope(null);
    setTable(null);
    setQuery("");
    setPreview(null);
    // Explore's selection belongs to the project it was chosen in. Left
    // behind, it would ask the new project for another project's run id.
    setSources([]);
    setSource(null);
    setSourceTable(null);
    setCompareSource(null);
    setComparison(null);
    setViews([]);
    setSavedQuestions([]);
    // A warmed corpus belongs to the project it was read from.
    setLive(COLD);
    if (!projectId) return;
    localStorage.setItem("nlp-project", projectId);
    let alive = true;
    const refresh = () =>
      Promise.all([
        api<Document[]>(`/projects/${projectId}/documents`),
        api<Document[]>(`/projects/${projectId}/documents?trashed=true`),
        api<Job[]>(`/projects/${projectId}/jobs`),
        api<Job[]>(`/projects/${projectId}/jobs/trash`),
        api<Project[]>("/projects"),
      ])
        .then(([d, td, j, tj, p]) => {
          if (alive) {
            setDocuments(d);
            setTrashedDocuments(td);
            setJobs(j);
            setTrashedJobs(tj);
            setProjects(p);
          }
        })
        .catch((e) => {
          if (alive) setError(e.message);
        });
    void refresh();
    const timer = setInterval(refresh, 2000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [projectId]);

  const refreshCorpus = async () => {
    const [p, d, td] = await Promise.all([
      api<Project[]>("/projects"),
      api<Document[]>(`/projects/${projectId}/documents`),
      api<Document[]>(`/projects/${projectId}/documents?trashed=true`),
    ]);
    setProjects(p);
    setDocuments(d);
    setTrashedDocuments(td);
  };
  const moveDocument = async (doc: Document, restore: boolean) =>
    perform(restore ? "Restoring document…" : "Moving document to Trash…", async () => {
      await post(`/projects/${projectId}/documents/${doc.id}/${restore ? "restore" : "trash"}`, {});
      await refreshCorpus();
      setCorpusSelection(allDocuments());
      setNotice(restore ? `${doc.name} restored.` : `${doc.name} moved to Trash.`);
    });
  const purgeDocument = async (doc: Document) => {
    if (!window.confirm(`Permanently delete ${doc.name}? This cannot be undone.`)) return;
    await perform("Permanently deleting document…", async () => {
      await post(`/projects/${projectId}/documents/${doc.id}/purge`, {});
      await refreshCorpus();
      setCorpusSelection(allDocuments());
      setNotice(`${doc.name} permanently deleted.`);
    });
  };
  const finishImport = async (report: ImportReport) => {
    setImportReport(report);
    await refreshCorpus();
    setPage("corpus");
    setNotice(
      `${report.imported} document${report.imported === 1 ? "" : "s"} imported. ${report.duplicates ? `${report.duplicates} unchanged duplicates skipped.` : "Original files were not changed."}`,
    );
  };
  const importFiles = (files: File[]) => {
    if (!projectId) {
      setNewProject(true);
      setNotice("Create a project first, then add your documents.");
      return;
    }
    if (busy) return;
    void perform(`Importing ${files.length} documents…`, async () =>
      finishImport(await upload(projectId, files)),
    );
  };
  const importPaths = useCallback(
    (paths: string[]) => {
      if (!projectId || busy) return;
      void perform("Importing documents…", async () => {
        const report = await post<ImportReport>(
          `/projects/${projectId}/import-paths`,
          { paths },
        );
        await finishImport(report);
      });
    },
    [projectId, busy, perform],
  );
  useEffect(() => {
    if (!isTauri()) return;
    let disposed = false;
    let unlisten: (() => void) | undefined;
    getCurrentWebviewWindow()
      .onDragDropEvent((event) => {
        setDragging(event.payload.type === "over");
        if (event.payload.type === "drop") importPaths(event.payload.paths);
      })
      .then((stop) => {
        if (disposed) stop();
        else unlisten = stop;
      })
      .catch((e) => setError(String(e)));
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [importPaths]);

  const pickFiles = async (folder = false) => {
    if (!projectId) {
      setNewProject(true);
      return;
    }
    if (isTauri()) {
      try {
        const selected = await open({
          directory: folder,
          multiple: !folder,
          filters: folder
            ? undefined
            : [
                {
                  name: "Text documents",
                  extensions: [
                    "txt",
                    "csv",
                    "tsv",
                    "html",
                    "htm",
                    "pdf",
                    "docx",
                    "rtf",
                  ],
                },
              ],
        });
        if (selected)
          importPaths(Array.isArray(selected) ? selected : [selected]);
      } catch (e) {
        setError(String(e));
      }
    } else {
      (folder ? folderRef : filesRef).current?.click();
    }
  };
  const inspectJob = (job: Job) =>
    void perform("Opening results…", async () => {
      setSelectedJob(job);
      setTable(null);
      setEnvelope(null);
      setTableQuery("");
      setTableOffset(0);
      setArtifactIndex(0);
      setPage("runs");
      if (!job.run_dir) return;
      const env = await api<Envelope>(
        `/projects/${projectId}/jobs/${job.id}/results`,
      );
      setEnvelope(env);
    });
  const changeArtifact = (index: number) => {
    setArtifactIndex(index);
    setTable(null);
    setTableQuery("");
    setTableOffset(0);
    setViewing(false);
  };
  /** Draw the workbench's current view over the whole file, as a recorded run.
   *
   * The interactive chart is a way of looking at the page on screen; this is
   * the same settings handed to the engine, which reads the complete artifact
   * and publishes a run directory with its provenance. */
  /**
   * Hand a live chart's settings to the engine, which draws the whole file.
   *
   * One function for both places that can publish — the Runs pane and the
   * Explore page — because the translation from workbench settings to engine
   * parameters is the sort of thing that goes wrong once a second copy of it
   * exists.
   */
  const publishChart = (
    jobId: string,
    index: number,
    path: string,
    settings: ChartSettings,
  ) => {
    void perform("Drawing the full chart…", async () => {
      const job = await post<Job>(
        `/projects/${projectId}/jobs/${jobId}/visualize`,
        {
          tool: "table_charts",
          index,
          params: publishParams(settings, path.replace(/\.csv$/i, "")),
        },
      );
      setJobs(await api<Job[]>(`/projects/${projectId}/jobs`));
      setSelectedJob(job);
      setEnvelope(null);
      setTable(null);
      // The published chart lands in the run record, so go where it will
      // appear rather than leaving the reader watching a page it will not
      // show up on.
      setPage("runs");
      setNotice(
        "Chart submitted over the whole file. It appears here when it finishes.",
      );
    });
  };

  /**
   * Save a view, or write over an existing one.
   *
   * Returns the stored record rather than nothing, because Explore has to
   * follow it: after saving, the view that is open is the one that was just
   * written, and after a failure nothing is open that was not open before. A
   * handler that reported success by staying silent would leave the page
   * showing an arrangement the workspace does not have.
   */
  const saveView = async (
    existing: View | null,
    name: string,
    chosen: TableSource,
    settings: ChartSettings,
    filters: RowFilter[],
  ): Promise<View | null> => {
    setError("");
    setNotice("");
    const payload = {
      name,
      job: chosen.job,
      index: chosen.index,
      settings: toViewSettings(settings),
      filters,
    };
    try {
      const saved = await post<View>(
        existing
          ? `/projects/${projectId}/views/${existing.id}`
          : `/projects/${projectId}/views`,
        payload,
      );
      setViews(await api<View[]>(`/projects/${projectId}/views`));
      setNotice(
        existing
          ? `Saved “${saved.name}”. Nothing was run.`
          : `“${saved.name}” saved. Reopen it from the Saved view list.`,
      );
      return saved;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  };

  const duplicateView = async (view: View): Promise<View | null> => {
    setError("");
    setNotice("");
    try {
      const copy = await post<View>(
        `/projects/${projectId}/views/${view.id}/duplicate`,
        {},
      );
      setViews(await api<View[]>(`/projects/${projectId}/views`));
      setNotice(`Copied to “${copy.name}”. “${view.name}” is unchanged.`);
      return copy;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  };

  const deleteView = async (view: View): Promise<boolean> => {
    setError("");
    setNotice("");
    try {
      await post(`/projects/${projectId}/views/${view.id}/delete`, {});
      setViews(await api<View[]>(`/projects/${projectId}/views`));
      setNotice(`Deleted “${view.name}”. The result it charted is untouched.`);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    }
  };

  /**
   * Read and parse the chosen documents so analyses become interactive.
   *
   * The slow step, and the only one. It uses the same corpus selection the
   * analysis forms use, so "the documents I am exploring" and "the documents I
   * would run on" are one choice rather than two that can disagree.
   */
  const warmCorpus = () => {
    void perform("Reading and parsing your documents…", async () => {
      const state = await post<LiveState>(`/projects/${projectId}/live/warm`, {
        selection: corpusSelection,
        parser: "spacy",
      });
      setLive(state);
      if (state.state === "failed") setError(state.error);
    });
  };

  /**
   * One live answer. Returns it rather than storing it, because the bench has
   * to decide whether it is still the question on screen — answers do not come
   * back in the order they were asked.
   *
   * The request carries its own identity and the snapshot it asks about
   * (contract R-C2/R-C4): a superseded answer is recognisable, and a corpus
   * that changed underneath is refused rather than answered.
   */
  const askLive = async (
    tool: string,
    params: Params,
    requestId: string,
  ): Promise<LiveAnswer | null> => {
    try {
      return await post<LiveAnswer>(`/projects/${projectId}/live/analyse`, {
        request_id: requestId,
        tool,
        params,
        snapshot_id: live.snapshot_id,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  };

  /**
   * Tell the engine to stop waiting on a question the reader abandoned (R-C8).
   * Best effort and silent: the interface has already stopped waiting, so a
   * failure here changes nothing the reader can see.
   */
  const abortLive = (requestId: string) => {
    void del(`/projects/${projectId}/live/analyse/${encodeURIComponent(requestId)}`).catch(
      () => undefined,
    );
  };

  /**
   * Put two finished runs side by side (the graded "which backend is better"
   * questions). The engine names every disagreement; this only asks.
   */
  const runCompare = async (expected: Job, actual: Job) => {
    setCompareBusy(true);
    try {
      setCompareView(await compareRuns(projectId, expected.id, actual.id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCompareBusy(false);
    }
  };

  const trackPhrase = async (
    request: PhraseRequest,
  ): Promise<PhraseAnswer | null> => {
    try {
      return await post<PhraseAnswer>(
        `/projects/${projectId}/live/phrase`,
        request,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  };

  const readSourcePassage = useCallback(
    async (
      documentId: string,
      characterStart: number,
      characterEnd: number | null,
      radius = 100,
    ): Promise<SourcePassage | null> => {
      try {
        return await sourcePassage(
          projectId,
          documentId,
          characterStart,
          characterEnd,
          radius,
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        return null;
      }
    },
    [projectId],
  );

  /**
   * Keep a question, either as a new record or as a new revision of one.
   *
   * *replacing* names the saved question this supersedes. Without it every
   * save created a record, so reopening a saved question, changing it and
   * saving refused the name it already had and demanded a new one — the one
   * path a researcher takes most.
   */
  const savePhraseQuestion = async (
    name: string,
    answer: PhraseAnswer,
    request: PhraseRequest,
    comparison: PhraseAnswer | null,
    replacing: string | null = null,
    expectedRevision: number | null = null,
  ): Promise<SavedPhraseQuestion | null> => {
    if (!live.selection?.ids.length || !live.parser) {
      setError("Reload the documents before saving this question.");
      return null;
    }
    // The scope is read from the live state but the evidence came from the
    // answer. If a reload happened in between they describe different corpora,
    // and the record would name documents the numbers never came from.
    if (live.snapshot_id !== answer.snapshot_id) {
      setError(
        "These documents are not the ones that answer came from. Track the phrase again before saving it.",
      );
      return null;
    }
    if (comparison && comparison.snapshot_id !== answer.snapshot_id) {
      setError(
        "The two phrases were answered from different document sets. Track them again before saving.",
      );
      return null;
    }
    try {
      const saved = await post<SavedPhraseQuestion>(
        replacing
          ? `/projects/${projectId}/questions/${replacing}`
          : `/projects/${projectId}/questions`,
        {
          name,
          snapshot_id: answer.snapshot_id,
          parser: live.parser,
          document_ids: live.selection.ids,
          specification: {
            schema_version: 1,
            kind: "phrase_distribution",
            text: request.text,
            comparison_text: comparison?.question.subject.text ?? null,
            case_sensitive: request.case_sensitive,
            normalize: request.normalize,
            match_lemma: request.match_lemma,
            match_nominalization: request.match_nominalization,
            position_bins: request.position_bins,
            evidence_year: request.evidence_year ?? null,
            evidence_document_id: request.evidence_document_id ?? null,
          },
          // How this answer read the phrase, kept with it. Publishing the
          // question later asks it with these tokens, so a parser upgrade
          // between saving and publishing changes the answer visibly or not
          // at all -- never quietly.
          matching: {
            version: answer.question.matching_profile.version,
            tokenizer: answer.question.matching_profile.tokenizer,
            source: answer.question.matching_profile.source,
            tokens: answer.question.subject.tokens,
            comparison_tokens: comparison?.question.subject.tokens ?? [],
          },
          ...(replacing && expectedRevision !== null
            ? { expected_revision: expectedRevision }
            : {}),
        },
      );
      setSavedQuestions(
        await api<SavedPhraseQuestion[]>(`/projects/${projectId}/questions`),
      );
      setNotice(
        replacing
          ? `“${saved.name}” updated to revision ${saved.revision}. Nothing was rerun.`
          : `“${saved.name}” saved. Nothing was rerun.`,
      );
      return saved;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  };

  const openPhraseQuestion = async (
    question: SavedPhraseQuestion,
  ): Promise<PhraseWorkspaceAnswer | null> => {
    try {
      if (live.snapshot_id !== question.snapshot_id) {
        const state = await post<LiveState>(
          `/projects/${projectId}/live/warm`,
          {
            selection: { document_ids: question.document_ids },
            parser: question.parser,
          },
        );
        setLive(state);
        if (state.state !== "ready") {
          throw new Error(
            state.error || "The saved corpus could not be loaded.",
          );
        }
        if (state.snapshot_id !== question.snapshot_id) {
          throw new Error(
            "The saved question belongs to a different text snapshot. Its evidence was not silently replaced.",
          );
        }
      }
      const spec = question.specification;
      const request: PhraseRequest = {
        text: spec.text,
        snapshot_id: question.snapshot_id,
        case_sensitive: spec.case_sensitive,
        // A question saved before these existed records none of them, and
        // the question it was saved as is the literal one.
        normalize: spec.normalize ?? false,
        match_lemma: spec.match_lemma ?? false,
        match_nominalization: spec.match_nominalization ?? false,
        position_bins: spec.position_bins,
        evidence_offset: 0,
        evidence_limit: 50,
        evidence_year: spec.evidence_year,
        evidence_document_id: spec.evidence_document_id,
      };
      const [primary, comparison] = await Promise.all([
        post<PhraseAnswer>(`/projects/${projectId}/live/phrase`, request),
        spec.comparison_text
          ? post<PhraseAnswer>(`/projects/${projectId}/live/phrase`, {
              ...request,
              text: spec.comparison_text,
            })
          : Promise.resolve(null),
      ]);
      return { primary, comparison };
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  };

  const duplicatePhraseQuestion = async (
    question: SavedPhraseQuestion,
  ): Promise<SavedPhraseQuestion | null> => {
    try {
      const copy = await post<SavedPhraseQuestion>(
        `/projects/${projectId}/questions/${question.id}/duplicate`,
        {},
      );
      setSavedQuestions(
        await api<SavedPhraseQuestion[]>(`/projects/${projectId}/questions`),
      );
      setNotice(`Copied to “${copy.name}”. Nothing was rerun.`);
      return copy;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  };

  const deletePhraseQuestion = async (
    question: SavedPhraseQuestion,
  ): Promise<boolean> => {
    try {
      await post(`/projects/${projectId}/questions/${question.id}/delete`, {});
      setSavedQuestions(
        await api<SavedPhraseQuestion[]>(`/projects/${projectId}/questions`),
      );
      setNotice(
        `Deleted “${question.name}”. Its source documents are untouched.`,
      );
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    }
  };

  const publishPhraseQuestion = (question: SavedPhraseQuestion) => {
    void perform("Publishing the saved question...", async () => {
      const job = await post<Job>(
        `/projects/${projectId}/questions/${question.id}/publish`,
        {},
      );
      setJobs(await api<Job[]>(`/projects/${projectId}/jobs`));
      setSelectedJob(job);
      setEnvelope(null);
      setTable(null);
      setPage("runs");
      setNotice(
        `"${question.name}" is being published with its saved documents and settings.`,
      );
    });
  };

  /**
   * Turn what is on screen into a recorded run.
   *
   * The live answer and the published one come from the same engine over the
   * same documents, so this submits the ordinary job with the same tool,
   * parameters and selection — and the result carries the provenance the live
   * view has none of.
   */
  const publishLive = (tool: string, params: Params) => {
    void perform("Recording this analysis as a run…", async () => {
      const job = await post<Job>(`/projects/${projectId}/jobs`, {
        tool,
        params,
        parser: "spacy",
        selection: corpusSelection,
      });
      setJobs(await api<Job[]>(`/projects/${projectId}/jobs`));
      setSelectedJob(job);
      setEnvelope(null);
      setTable(null);
      setPage("runs");
      setNotice(
        "Analysis submitted over your documents. It appears here when it finishes.",
      );
    });
  };

  const openAnalysis = (tool: Tool) => {
    setCorpusSelection(allDocuments());
    setAnalysis(tool);
    setParams(initialParams(tool));
    // Open on a parser this machine can actually run, so the form does not
    // start on a disabled option that has to be changed before Run works.
    setParser((current) =>
      parserReady(setup, current)
        ? current
        : (PARSERS.find((option) => parserReady(setup, option.id))?.id ??
          current),
    );
  };
  useEffect(() => {
    if (!selectedJob || busy) return;
    const latest = jobs.find((job) => job.id === selectedJob.id);
    if (
      !latest ||
      (latest.state === selectedJob.state && latest.stage === selectedJob.stage)
    )
      return;
    const finished = active(selectedJob) && !active(latest);
    setSelectedJob(latest);
    if (finished && latest.run_dir && !busy) inspectJob(latest);
  }, [jobs, selectedJob?.id, busy]);
  useEffect(() => {
    setTable(null);
    setTableLoading(false);
    if (
      !selectedJob ||
      !envelope?.artifacts[artifactIndex]?.path.endsWith(".csv")
    )
      return;
    let alive = true;
    const controller = new AbortController();
    setTableLoading(true);
    const timer = setTimeout(() => {
      api<Table>(
        `/projects/${projectId}/jobs/${selectedJob.id}/artifacts/${artifactIndex}?q=${encodeURIComponent(tableQuery)}&offset=${tableOffset}`,
        { signal: controller.signal },
      )
        .then((value) => {
          if (alive) setTable(value);
        })
        .catch((error) => {
          if (alive) setError(String(error));
        })
        .finally(() => {
          if (alive) setTableLoading(false);
        });
    }, 250);
    return () => {
      alive = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [
    projectId,
    selectedJob?.id,
    envelope?.artifacts[artifactIndex]?.path,
    artifactIndex,
    tableQuery,
    tableOffset,
  ]);

  // The tables the archive can offer. Refetched when a run publishes, so a
  // table that has just finished is in the list without a reload.
  useEffect(() => {
    if (!projectId || page !== ARCHIVE_PAGE) return;
    let alive = true;
    setSourcesLoading(true);
    api<TableSource[]>(`/projects/${projectId}/tables`)
      .then((found) => {
        if (!alive) return;
        setSources(found);
        // Keep the reader where they were if their table is still published;
        // a refresh that silently jumped to a different table would make any
        // conclusion drawn from the chart belong to the wrong file.
        setSource((current) => {
          const same =
            current &&
            found.find((item) => sourceKey(item) === sourceKey(current));
          return same ?? firstWorthOpening(found);
        });
      })
      .catch((error) => {
        if (alive) setError(String(error));
      })
      .finally(() => {
        if (alive) setSourcesLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [projectId, page, completed]);

  useEffect(() => {
    if (!projectId || page !== "interactive") return;
    let alive = true;
    api<SavedPhraseQuestion[]>(`/projects/${projectId}/questions`)
      .then((found) => alive && setSavedQuestions(found))
      .catch((error) => alive && setError(String(error)));
    return () => {
      alive = false;
    };
  }, [projectId, page]);

  // Saved views for this project. Refetched with the table list, so a view
  // saved against a run that has just finished can be opened without a reload.
  useEffect(() => {
    if (!projectId || page !== ARCHIVE_PAGE) return;
    let alive = true;
    api<View[]>(`/projects/${projectId}/views`)
      .then((found) => alive && setViews(found))
      .catch((error) => alive && setError(String(error)));
    return () => {
      alive = false;
    };
  }, [projectId, page, completed]);

  // Whether this project already has a warm corpus -- it may, from earlier in
  // the session, and offering to load documents that are already loaded would
  // make the one slow step look compulsory.
  useEffect(() => {
    if (!projectId || page !== "interactive") return;
    let alive = true;
    api<LiveState>(`/projects/${projectId}/live`)
      .then((state) => alive && setLive(state))
      .catch(() => {
        // An unreachable bench is not a reason to blank the page; the Load
        // button will report the real failure when it is pressed.
      });
    return () => {
      alive = false;
    };
  }, [projectId, page]);

  // What the engine cannot reproduce from a preview, in its own words rather
  // than a second copy of the rule written here.
  useEffect(() => {
    if (!ready) return;
    let alive = true;
    api<ChartContract>("/chart-contract")
      .then((served) => alive && setContract(served))
      .catch(() => {
        // A missing contract must not stop anybody charting. The worst case is
        // that a caution goes unshown, which is where this started.
      });
    return () => {
      alive = false;
    };
  }, [ready]);

  // The chosen table itself. Same bounded preview the Runs pane reads, so the
  // workbench's "rows on this page" means the same thing in both places.
  useEffect(() => {
    setSourceTable(null);
    if (!projectId || !source) return;
    let alive = true;
    const controller = new AbortController();
    setSourceLoading(true);
    api<Table>(
      `/projects/${projectId}/jobs/${source.job}/artifacts/${source.index}`,
      { signal: controller.signal },
    )
      .then((value) => {
        if (alive) setSourceTable(value);
      })
      .catch((error) => {
        if (alive && !controller.signal.aborted) setError(String(error));
      })
      .finally(() => {
        if (alive) setSourceLoading(false);
      });
    return () => {
      alive = false;
      controller.abort();
    };
  }, [projectId, source && sourceKey(source)]);
  useEffect(() => {
    if (!projectId || !compareSource) return;
    const controller = new AbortController();
    let alive = true;
    const key = comparisonKey;
    setComparison(null);
    api<Table>(
      `/projects/${projectId}/jobs/${compareSource.job}/artifacts/${compareSource.index}`,
      { signal: controller.signal },
    )
      .then((value) => {
        if (alive) setComparison({ key, table: value });
      })
      .catch(() => {
        if (alive && !controller.signal.aborted)
          setComparison({ key, table: null });
      });
    return () => {
      alive = false;
      controller.abort();
    };
  }, [projectId, comparisonKey]);
  const filtered = (showTrash ? trashedDocuments : documents).filter((d) =>
    d.name.toLowerCase().includes(query.toLowerCase()),
  );
  const runList = (list: Job[], trashed = false) =>
    list.length ? (
      <div className="run-list">
        {list.map((job) => (
          <div
            key={job.id}
            className="run-entry"
          >
            <button className="run-row" onClick={() => inspectJob(job)} disabled={!!busy || trashed} title={trashed ? "Restore this run to inspect its results" : undefined}>
            <span className="run-icon">
              <FlaskConical size={18} />
            </span>
            <span className="run-name">
              <strong>{title(job.tool)}</strong>
              <small>
                {when(job.created)} · {job.stage}
              </small>
            </span>
            <Badge state={job.state} />
            <ChevronRight size={16} />
            </button>
            {trashed ? <>
              <span className="muted">Restore to inspect</span>
              <button className="text-button" disabled={!!busy} onClick={() => void perform("Restoring run…", async () => { await post(`/projects/${projectId}/jobs/${job.id}/restore`, {}); setJobs(await api<Job[]>(`/projects/${projectId}/jobs`)); setTrashedJobs(await api<Job[]>(`/projects/${projectId}/jobs/trash`)); })}>Restore</button>
              <button className="text-button" disabled={!!busy} onClick={() => { if (window.confirm(`Permanently delete this ${title(job.tool)} run? Saved views that use it will remain marked as missing their source.`)) void perform("Permanently deleting run…", async () => { await post(`/projects/${projectId}/jobs/${job.id}/purge`, {}); setTrashedJobs(await api<Job[]>(`/projects/${projectId}/jobs/trash`)); }); }}>Delete permanently</button>
            </> : <button className="text-button" disabled={!!busy || active(job)} onClick={() => { if (window.confirm(`Move this ${title(job.tool)} run to Trash? Saved views that use it will be unavailable until restored.`)) void perform("Moving run to Trash…", async () => { await post(`/projects/${projectId}/jobs/${job.id}/trash`, {}); setJobs(await api<Job[]>(`/projects/${projectId}/jobs`)); setTrashedJobs(await api<Job[]>(`/projects/${projectId}/jobs/trash`)); if (selectedJob?.id === job.id) { setSelectedJob(null); setEnvelope(null); setTable(null); } }); }}>Move to Trash</button>}
          </div>
        ))}
      </div>
    ) : (
      <div className="empty-inline">
        <FlaskConical size={26} />
        <p>No analyses yet. Your first discovery starts here.</p>
        <button className="text-button" onClick={() => setPage("studio")}>
          Browse analyses <ArrowRight size={15} />
        </button>
      </div>
    );

  /**
   * Explore, as the half of it the current page names.
   *
   * The live bench and the archive of published tables were two tabs inside
   * one page. The sidebar names them separately now, so each page asks for the
   * half it is, and the tab pair is not drawn at all. Built here rather than
   * written out at both call sites, because the prop list is long enough that
   * two copies would drift.
   */
  const exploreView = (locked: "live" | "results") => (
    <Explore
      projectId={projectId}
      lockedMode={locked}
      embedded={locked === "results"}
      sources={sources}
      loading={sourcesLoading}
      selected={source}
      onSelect={(chosen) => {
        setSource(chosen);
        setCompareSource(null);
      }}
      table={sourceTable}
      tableLoading={sourceLoading}
      compareSource={compareSource}
      onCompareSource={setCompareSource}
      compare={comparison?.key === comparisonKey ? comparison.table : null}
      compareLoading={!!compareSource && comparison?.key !== comparisonKey}
      canPublish={!busy && !!projectId}
      onPublish={(chosen, settings) =>
        publishChart(chosen.job, chosen.index, chosen.path, settings)
      }
      onBrowseAnalyses={() => setPage("studio")}
      liveBench={
        <LiveBench
          projectId={projectId}
          tools={tools}
          state={live}
          contract={contract}
          onWarm={warmCorpus}
          onAnalyse={askLive}
          onAbort={abortLive}
          onTrackPhrase={trackPhrase}
          onReadSource={readSourcePassage}
          savedQuestions={savedQuestions}
          onSaveQuestion={savePhraseQuestion}
          onOpenQuestion={openPhraseQuestion}
          onDuplicateQuestion={duplicatePhraseQuestion}
          onDeleteQuestion={deletePhraseQuestion}
          onPublishQuestion={publishPhraseQuestion}
          busy={!!busy}
          canPublish={!busy && !!projectId}
          onPublish={publishLive}
        />
      }
      views={views}
      contract={contract}
      onSaveView={(name, chosen, settings, filters) =>
        saveView(null, name, chosen, settings, filters)
      }
      onUpdateView={(view, chosen, settings, filters) =>
        saveView(view, view.name, chosen, settings, filters)
      }
      onDuplicateView={duplicateView}
      onDeleteView={deleteView}
    />
  );

  return (
    <DialogError.Provider value={error}>
      <div
        className="app-shell"
        onDragOver={(e) => {
          e.preventDefault();
          if (projectId) setDragging(true);
        }}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node))
            setDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!isTauri()) importFiles(Array.from(e.dataTransfer.files));
        }}
      >
        <input
          aria-label="Upload text documents"
          ref={filesRef}
          type="file"
          accept=".txt,.csv,.tsv,.html,.htm,.pdf,.docx,.rtf"
          multiple
          hidden
          onChange={(e) => {
            importFiles(Array.from(e.target.files || []));
            e.target.value = "";
          }}
        />
        <input
          aria-label="Upload a corpus folder"
          ref={folderRef}
          type="file"
          multiple
          hidden
          {...{ webkitdirectory: "" }}
          onChange={(e) => {
            importFiles(
              Array.from(e.target.files || []).filter((f) =>
                /\.(txt|csv|tsv|html?|pdf|docx|rtf)$/i.test(f.name),
              ),
            );
            e.target.value = "";
          }}
        />
        <aside className="sidebar">
          <a
            className="brand"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setPage("overview");
            }}
            aria-label="NLP Suite home"
          >
            <span className="brand-mark">
              <span />
              <span />
              <span />
              <span />
            </span>
            <span>
              NLP Suite<small>RESEARCH WORKSPACE</small>
            </span>
          </a>
          <div className="workspace-label">
            WORKSPACE{" "}
            <button
              aria-label="Create a project"
              className="icon-button"
              disabled={!ready || !!busy}
              onClick={() => setNewProject(true)}
            >
              <Plus size={16} />
            </button>
          </div>
          <div className="project-switch">
            <FolderOpen size={19} />
            <select
              aria-label="Current project"
              value={projectId}
              disabled={!!busy || !projects.length}
              onChange={(e) => setProjectId(e.target.value)}
            >
              {!projects.length && <option value="">No project yet</option>}
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
            <ChevronDown size={14} />
          </div>
          <nav aria-label="Workspace navigation">
            {navigation.map((item, index) => (
              <Fragment key={item.key}>
                {item.section !== navigation[index - 1]?.section && (
                  <span className="nav-section">{item.section}</span>
                )}
                <button
                  className={
                    page === item.key ? "nav-link selected" : "nav-link"
                  }
                  onClick={() => setPage(item.key)}
                  aria-current={page === item.key ? "page" : undefined}
                >
                  <item.icon size={19} />
                  <span className="nav-text">
                    {item.label}
                    {"hint" in item && <small>{item.hint}</small>}
                  </span>
                  {item.key === "corpus" && !!documents.length && (
                    <span className="nav-count">{documents.length}</span>
                  )}
                  {item.key === "runs" && running > 0 && (
                    <span className="nav-count">{running}</span>
                  )}
                </button>
              </Fragment>
            ))}
          </nav>
          <div className="sidebar-note">
            <span className="tiny-eyebrow">BUILT FOR CLOSE READING</span>
            <p>
              From a collection of texts
              <br />
              to a clearer understanding.
            </p>
            <BookOpen size={27} strokeWidth={1.2} />
          </div>
          <div className="sidebar-bottom">
            <button
              className={`nav-link ${page === "setup" ? "selected" : ""}`}
              onClick={() => setPage("setup")}
            >
              <Settings2 size={18} />
              Settings & backups
            </button>
            <div className="local-status">
              <span />
              Local workspace <ShieldCheck size={14} />
            </div>
            <small>Desktop beta · 0.3.1</small>
          </div>
        </aside>
        <div className="main-shell">
          <header className="topbar">
            <div className="breadcrumb">
              Workspace <ChevronRight size={13} />
              <span>{project?.name || "Getting started"}</span>
            </div>
            <div className="topbar-right">
              <span className="private-label">
                <ShieldCheck size={14} />
                Private by design
              </span>
              <button
                className="icon-button"
                aria-label="Settings and backups"
                title="Settings and backups"
                onClick={() => setPage("setup")}
              >
                <CircleHelp size={19} />
              </button>
            </div>
          </header>
          <main>
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
                <CheckCircle2 size={18} />
                <span>{notice}</span>
                <button
                  aria-label="Dismiss notice"
                  onClick={() => setNotice("")}
                >
                  <X size={16} />
                </button>
              </div>
            )}
            {busy && (
              <div className="working" role="status">
                <LoaderCircle size={15} className="spin" />
                {busy}
              </div>
            )}
            {!ready ? (
              <section className="connection-state">
                <span className="eyebrow">NLP SUITE DESKTOP</span>
                <h1>
                  {error
                    ? "Let’s reconnect your workspace."
                    : "Opening your workspace…"}
                </h1>
                <p>
                  The interface connects only to the analysis engine on this
                  computer.
                </p>
                {error && (
                  <button className="primary" onClick={() => location.reload()}>
                    Try again
                  </button>
                )}
              </section>
            ) : (
              <>
                {page === "overview" && (
                  <>
                    <div className="page-heading">
                      <div>
                        <span className="eyebrow">YOUR RESEARCH, IN FOCUS</span>
                        <h1>
                          {project
                            ? "Workspace overview"
                            : "A new chapter for your research."}
                        </h1>
                        <p>
                          {project
                            ? "A home for your documents, analyses, and discoveries."
                            : "Bring your texts together. Ask better questions. Keep the evidence."}
                        </p>
                      </div>
                      <button
                        className="primary"
                        disabled={!!busy}
                        onClick={() =>
                          project ? void pickFiles() : setNewProject(true)
                        }
                      >
                        <Plus size={17} />
                        {project ? "Add documents" : "Create a project"}
                      </button>
                    </div>
                    <section className="hero">
                      <div className="hero-copy">
                        <span className="eyebrow">
                          FROM TEXT TO UNDERSTANDING
                        </span>
                        <h2>
                          A closer reading.
                          <br />
                          <em>A bigger picture.</em>
                        </h2>
                        <p>
                          Your corpus is more than a collection of files.
                          Explore the language, patterns, and ideas that connect
                          it.
                        </p>
                        {/* The button names where it goes, and it goes as
                            soon as there is a corpus to go with. Exploring
                            needs documents, not a finished run: gating this
                            on a completed job sent people to the queue to
                            wait half a minute for an answer the bench gives
                            in a fifth of a second. */}
                        <button
                          className="hero-button"
                          onClick={() => {
                            if (!project) return setNewProject(true);
                            if (!documents.length) return setPage("corpus");
                            return setPage("interactive");
                          }}
                        >
                          {!project
                            ? "Create a project"
                            : !documents.length
                              ? "Build your first corpus"
                              : "Run an analysis live"}
                          <ArrowRight size={17} />
                        </button>
                      </div>
                      <div className="hero-art" aria-hidden="true">
                        <div className="paper paper-back">
                          <div className="paper-lines" />
                        </div>
                        <div className="paper paper-front">
                          <span className="paper-kicker">
                            THE LANGUAGE OF IDEAS
                          </span>
                          <div className="paper-title">
                            Words become
                            <br />
                            understanding.
                          </div>
                          <div className="paper-lines" />
                          <div className="word-tags">
                            <span>language</span>
                            <span>patterns</span>
                            <span>meaning</span>
                          </div>
                        </div>
                        <div className="insight-seal">
                          <Sparkles size={20} />
                          <span>
                            A little more
                            <br />
                            perspective.
                          </span>
                        </div>
                      </div>
                    </section>
                    <div className="metrics">
                      <div>
                        <span className="metric-label">
                          <Files size={17} />
                          Documents
                        </span>
                        <strong>{fmt(documents.length)}</strong>
                        <small>
                          {documents.length
                            ? "Preserved in your project"
                            : "A blank page. A good beginning."}
                        </small>
                      </div>
                      <div>
                        <span className="metric-label">
                          <BookOpen size={17} />
                          Words in corpus
                        </span>
                        <strong>{fmt(project?.words || 0)}</strong>
                        <small>Whitespace-delimited intake count</small>
                      </div>
                      <div>
                        <span className="metric-label">
                          <FlaskConical size={17} />
                          Results available
                        </span>
                        <strong>{fmt(completed)}</strong>
                        <small>
                          {running
                            ? `${running} analysis in progress`
                            : completed
                              ? "Recorded with their inputs and settings"
                              : "Explore runs analyses without recording them"}
                        </small>
                      </div>
                    </div>
                    <div className="overview-grid">
                      <section className="panel">
                        <div className="section-title">
                          <div>
                            <h2>Recent analyses</h2>
                            <p>Your work, with a trail back to the source.</p>
                          </div>
                          <button
                            className="text-button"
                            onClick={() => setPage("runs")}
                          >
                            View all <ArrowUpRight size={15} />
                          </button>
                        </div>
                        {runList(jobs.slice(0, 4))}
                      </section>
                      <section className="panel collection-panel">
                        <div className="section-title">
                          <h2>Your collection</h2>
                          <span className="subtle-count">
                            {documents.length} texts
                          </span>
                        </div>
                        {documents.slice(0, 3).map((doc) => (
                          <button
                            className="document-mini"
                            key={doc.id}
                            onClick={() =>
                              void perform("Opening document…", async () =>
                                setPreview(
                                  await api<Document>(
                                    `/projects/${projectId}/documents/${doc.id}`,
                                  ),
                                ),
                              )
                            }
                          >
                            <FileText size={18} />
                            <span>
                              {doc.name}
                              <small>{fmt(doc.words)} words</small>
                            </span>
                          </button>
                        ))}
                        {!documents.length && (
                          <p className="empty-copy">
                            Add a few documents or an entire folder. We’ll keep
                            the originals untouched.
                          </p>
                        )}
                        <button
                          className="collection-link"
                          onClick={() => setPage("corpus")}
                        >
                          Open corpus <ArrowRight size={16} />
                        </button>
                      </section>
                    </div>
                    <div className="quiet-footer">
                      <ShieldCheck size={15} />
                      <span>
                        Your documents stay on this computer. No account. No
                        cloud upload.
                      </span>
                      <span className="footer-end">
                        Curiosity, backed by evidence.
                      </span>
                    </div>
                  </>
                )}
                {page === "corpus" && (
                  <>
                    <div className="page-heading">
                      <div>
                        <span className="eyebrow">THE SOURCE MATERIAL</span>
                        <h1>Your corpus</h1>
                        <p>
                          Bring your documents together. Keep their identity
                          intact.
                        </p>
                      </div>
                      <button
                        className="primary"
                        disabled={!!busy}
                        onClick={() => void pickFiles()}
                      >
                        <Plus size={17} />
                        Add documents
                      </button>
                    </div>
                    <section className="dropzone">
                      <span className="upload-icon">
                        <Upload size={26} />
                      </span>
                      <div>
                        <h2>Drop your texts here.</h2>
                        <p>
                          Text, tables, HTML, PDF, DOCX or RTF. Up to 20 MB per
                          file. Scanned PDFs need OCR.
                        </p>
                        <div className="drop-actions">
                          <button
                            className="secondary"
                            disabled={!!busy}
                            onClick={() => void pickFiles()}
                          >
                            Choose files
                          </button>
                          <button
                            className="text-button"
                            disabled={!!busy}
                            onClick={() => void pickFiles(true)}
                          >
                            <FolderOpen size={16} />
                            Choose a folder
                          </button>
                        </div>
                      </div>
                      <span className="file-tag">TEXT + DOCS</span>
                    </section>
                    {setup?.sample_available &&
                      !!project &&
                      !documents.length && (
                        <div className="sample-callout">
                          <BookOpen size={23} />
                          <div>
                            <strong>Try the included example documents.</strong>
                            <p>
                              Three short fictional texts help you learn the
                              tools before importing your own collection.
                            </p>
                          </div>
                          <button
                            className="secondary"
                            disabled={!!busy}
                            onClick={() =>
                              void perform(
                                "Importing example documents…",
                                async () =>
                                  finishImport(
                                    await post<ImportReport>(
                                      `/projects/${projectId}/import-sample`,
                                      {},
                                    ),
                                  ),
                              )
                            }
                          >
                            Try example documents <ArrowRight size={15} />
                          </button>
                        </div>
                      )}
                    {importReport && importReport.errors.length > 0 && (
                      <details className="import-errors" open>
                        <summary>
                          {importReport.errors.length} document(s) need
                          attention
                        </summary>
                        {importReport.errors.map((e, i) => (
                          <p key={i}>
                            <strong>{e.name}:</strong> {e.message}
                          </p>
                        ))}
                      </details>
                    )}
                    <section className="panel">
                      <div className="section-title">
                        <h2>
                          {showTrash ? "Trash" : "Documents"}{" "}
                          <span className="subtle-count">
                            {showTrash ? trashedDocuments.length : documents.length}
                          </span>
                        </h2>
                        <button className="text-button" onClick={() => { setShowTrash(!showTrash); setQuery(""); }}>
                          {showTrash ? "Back to documents" : `Trash (${trashedDocuments.length})`}
                        </button>
                        <label className="search-box">
                          <Search size={16} />
                          <input
                            placeholder="Find a document…"
                            aria-label="Find a document"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                          />
                        </label>
                      </div>
                      <div className="table-scroll">
                        <table className="document-table">
                          <thead>
                            <tr>
                              <th>DOCUMENT</th>
                              <th>WORDS</th>
                              <th>SIZE</th>
                              <th>STATUS</th>
                              <th>ACTION</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filtered.map((doc) => (
                              <tr key={doc.id}>
                                <td>
                                  <button
                                    className="document-name"
                                    disabled={showTrash}
                                    onClick={() =>
                                      void perform(
                                        "Opening document…",
                                        async () =>
                                          setPreview(
                                            await api<Document>(
                                              `/projects/${projectId}/documents/${doc.id}`,
                                            ),
                                          ),
                                      )
                                    }
                                  >
                                    <FileText size={18} />
                                    {doc.name}
                                  </button>
                                </td>
                                <td>{fmt(doc.words)}</td>
                                <td>{(doc.bytes / 1024).toFixed(1)} KB</td>
                                <td>
                                  <span className="ready-label">
                                    <CheckCircle2 size={14} />
                                    {showTrash ? "In Trash" : "Imported"}
                                  </span>
                                </td>
                                <td>
                                  {showTrash ? (
                                    <>
                                      <button className="text-button" disabled={!!busy} onClick={() => void moveDocument(doc, true)}>Restore</button>{" "}
                                      <button className="text-button" disabled={!!busy} onClick={() => void purgeDocument(doc)}>Delete permanently</button>
                                    </>
                                  ) : (
                                    <button className="text-button" disabled={!!busy || running > 0} onClick={() => void moveDocument(doc, false)}>Move to Trash</button>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {!filtered.length && (
                        <p className="empty-copy">
                          {query
                            ? "No documents match your search."
                            : "Your imported documents will appear here."}
                        </p>
                      )}
                    </section>
                  </>
                )}
                {page === "studio" && (
                  <>
                    <div className="page-heading">
                      <div>
                        <span className="eyebrow">
                          ASK SOMETHING OF YOUR TEXTS
                        </span>
                        <h1>Analyze &amp; visualize</h1>
                        <p>
                          Run a workflow over the whole corpus. Every result
                          stays connected to its source, and every figure keeps
                          its data attached.
                        </p>
                      </div>
                      <span className="corpus-pill">
                        <Files size={16} />
                        {documents.length} documents in scope
                      </span>
                    </div>
                    {!documents.length && (
                      <div className="sample-callout">
                        <FolderOpen size={22} />
                        <div>
                          <strong>Choose documents or a CSV table.</strong>
                          <p>
                            Text analyses need imported documents. The CSV
                            statistics and charting workflows only need a
                            project and a table you pick at run time.
                          </p>
                        </div>
                        <button
                          className="secondary"
                          onClick={() => setPage("corpus")}
                        >
                          Add documents
                        </button>
                      </div>
                    )}
                    <label className="search-box">
                      <Search size={16} />
                      <input
                        aria-label="Find a workflow"
                        placeholder="Find a workflow…"
                        value={toolQuery}
                        onChange={(e) => setToolQuery(e.target.value)}
                      />
                    </label>
                    {/* Shelves in the order the course meets the work
                        (taxonomy in core/profiler/registry.py). One card per
                        syllabus tool, never merged: the rubric compares
                        MALLET against Gensim and CBOW against skip-gram, so
                        a combined card would hide the comparison. */}
                    {FAMILY_ORDER.map((family) => {
                      const shown = tools.filter(
                        (tool) =>
                          (tool.family ?? "") === family &&
                          `${tool.label} ${tool.description}`
                            .toLowerCase()
                            .includes(toolQuery.toLowerCase()),
                      );
                      if (!shown.length) return null;
                      return (
                        <section
                          key={family}
                          aria-label={shown[0].family_label ?? family}
                        >
                          <div className="studio-label">
                            <span>
                              {(shown[0].family_label ?? family).toUpperCase()}
                            </span>
                            <span>
                              {shown.length}{" "}
                              {shown.length === 1 ? "workflow" : "workflows"}
                            </span>
                          </div>
                          <div className="analysis-grid">
                            {shown.map((tool) => (
                              <ToolCard
                                key={tool.name}
                                tool={tool}
                                onOpen={() => openAnalysis(tool)}
                                disabled={!!busy}
                              />
                            ))}
                          </div>
                        </section>
                      );
                    })}
                    <p className="scope-note">
                      <CircleHelp size={16} />
                      Choose a workflow to see its inputs and settings. Chart
                      data CSVs sit next to every figure, and GEXF networks open
                      directly in Gephi.{" "}
                      {setup?.components.length ? (
                        <>
                          This installation{" "}
                          {setup.components.find((c) => c.name === "kaleido")
                            ?.present
                            ? "can save charts as PNG, SVG and PDF"
                            : "cannot save charts as PNG, SVG or PDF — Kaleido is not installed"}
                          , and{" "}
                          {setup.components.find((c) => c.name === "openpyxl")
                            ?.present
                            ? "writes native Excel workbooks."
                            : "cannot write Excel workbooks — the Excel writer is not installed."}
                        </>
                      ) : (
                        "Interactive HTML always works; static images and Excel workbooks depend on what is installed — see Settings & backups."
                      )}
                    </p>
                  </>
                )}
                {page === "interactive" && exploreView("live")}
                {page === "learn" && (
                  <Learn tools={tools} onChoose={openAnalysis} canRun={!busy} />
                )}
                {page === "runs" && (
                  <>
                    <div className="page-heading">
                      <div>
                        <span className="eyebrow">THE RESEARCH RECORD</span>
                        <h1>Runs & results</h1>
                        <p>Every run is a snapshot—not a moving target.</p>
                      </div>
                      <button
                        className="secondary"
                        onClick={() => setPage("studio")}
                      >
                        <Plus size={16} />
                        New analysis
                      </button>
                    </div>
                    <section className="panel">
                      <div className="section-title">
                        <h2>{showRunTrash ? "Run Trash" : "Past runs"}</h2>
                        <button className="text-button" onClick={() => setShowRunTrash(!showRunTrash)}>{showRunTrash ? "Back to runs" : `Trash (${trashedJobs.length})`}</button>
                      </div>
                      {runList(showRunTrash ? trashedJobs : jobs, showRunTrash)}
                    </section>
                    {/* Drawn on ARCHIVE_PAGE, which is also what the table
                        and saved-view fetches are gated on. */}
                    <section className="panel">
                      {exploreView("results")}
                    </section>
                    {selectedJob && (
                      <section className="panel result-panel">
                        <div className="section-title">
                          <div>
                            <h2>{title(selectedJob.tool)}</h2>
                            <p>
                              {when(selectedJob.created)} ·{" "}
                              {selectedJob.id.slice(0, 8)}
                            </p>
                          </div>
                          {envelope && (
                            <button
                              className="primary"
                              disabled={!!busy}
                              onClick={() =>
                                void perform("Preparing export…", () =>
                                  download(
                                    `/projects/${projectId}/jobs/${selectedJob.id}/export`,
                                    `nlp-suite-${selectedJob.id.slice(0, 8)}.zip`,
                                  ),
                                )
                              }
                            >
                              <ArrowDownToLine size={16} />
                              Export run
                            </button>
                          )}
                        </div>
                        {active(selectedJob) && (
                          <button
                            className="secondary"
                            disabled={!!busy}
                            onClick={() =>
                              void perform("Cancelling analysis…", async () => {
                                await post(
                                  `/projects/${projectId}/jobs/${selectedJob.id}/cancel`,
                                  {},
                                );
                                setJobs(
                                  await api<Job[]>(
                                    `/projects/${projectId}/jobs`,
                                  ),
                                );
                              })
                            }
                          >
                            Cancel analysis
                          </button>
                        )}
                        {!envelope && (
                          <p className="empty-copy">
                            {active(selectedJob)
                              ? "This run is still working. Open it again when it completes."
                              : "This run did not publish an artifact. Review the messages below."}
                          </p>
                        )}
                        <Diagnostics
                          items={
                            envelope?.diagnostics || selectedJob.diagnostics
                          }
                        />
                        {envelope && (
                          <>
                            <InsightPanel
                              projectId={projectId}
                              jobId={selectedJob.id}
                              onChart={(kind, x, y) => {
                                setChartSeed({ kind, x, y });
                                setVisualizing(true);
                              }}
                              onFigure={(panel) =>
                                setFigureRequest((last) => ({
                                  panel,
                                  nonce: (last?.nonce ?? 0) + 1,
                                }))
                              }
                            />
                            {/* Purpose-built figures over this run's result
                                shape. Nothing renders unless the run's tool
                                actually has panels — an empty section would
                                read as a bug (see ARCHIVE_PAGE). */}
                            <div className="result-toolbar">
                              <select
                                aria-label="Result artifact"
                                value={artifactIndex}
                                onChange={(e) =>
                                  changeArtifact(Number(e.target.value))
                                }
                              >
                                {/* Tools already describe what each file is
                                    ("bigram PMI", "PMI collocation network").
                                    The list used to show only the filename and
                                    throw the description away, which left runs
                                    that publish six files unreadable. */}
                                {envelope.artifacts.map((a, i) => (
                                  <option key={a.path} value={i}>
                                    {a.description
                                      ? `${a.description} — ${a.path}`
                                      : a.path}
                                  </option>
                                ))}
                              </select>
                              <label className="search-box">
                                <Search size={15} />
                                <input
                                  aria-label="Search result table"
                                  placeholder="Search all result rows…"
                                  value={tableQuery}
                                  onChange={(e) => {
                                    setTableQuery(e.target.value);
                                    setTableOffset(0);
                                  }}
                                />
                              </label>
                              <button
                                className="icon-button"
                                title="View this artifact in the app"
                                aria-label="View this artifact in the app"
                                disabled={
                                  !!busy ||
                                  !viewableArtifact(
                                    envelope.artifacts[artifactIndex].path,
                                  )
                                }
                                onClick={() => setViewing(!viewing)}
                              >
                                <Eye size={18} />
                              </button>
                              <button
                                className="icon-button"
                                title="Visualize this CSV table"
                                aria-label="Visualize this CSV table"
                                disabled={
                                  !!busy ||
                                  !envelope.artifacts.length ||
                                  !envelope.artifacts[artifactIndex].path
                                    .toLowerCase()
                                    .endsWith(".csv")
                                }
                                onClick={() => {
                                  setChartSeed(null);
                                  setVisualizing(true);
                                }}
                              >
                                <ChartColumnBig size={18} />
                              </button>
                              <button
                                className="icon-button"
                                title="Download selected artifact"
                                aria-label="Download selected artifact"
                                disabled={!!busy || !envelope.artifacts.length}
                                onClick={() =>
                                  void perform("Preparing download…", () =>
                                    download(
                                      `/projects/${projectId}/jobs/${selectedJob.id}/artifacts/${artifactIndex}?download=true`,
                                      envelope.artifacts[artifactIndex].path,
                                    ),
                                  )
                                }
                              >
                                <ArrowDownToLine size={18} />
                              </button>
                            </div>
                            {viewing &&
                              viewableArtifact(
                                envelope.artifacts[artifactIndex].path,
                              ) && (
                                <ArtifactViewer
                                  path={envelope.artifacts[artifactIndex].path}
                                  fetchUrl={`/projects/${projectId}/jobs/${selectedJob.id}/artifacts/${artifactIndex}`}
                                />
                              )}
                            {tableLoading && (
                              <p role="status">Loading result table…</p>
                            )}
                            {projectId && (
                              <PanelSection
                                projectId={projectId}
                                jobId={selectedJob.id}
                                selectedColumns={table?.columns}
                                requested={figureRequest}
                                fallback={table ? (
                                  <RunWorkbench
                                    key={`${selectedJob.id}-${artifactIndex}`}
                                    table={table}
                                    contract={contract}
                                    artifactPath={envelope.artifacts[artifactIndex]?.path ?? ""}
                                    canPublish={!busy && !!projectId}
                                    onPublish={(settings) =>
                                      publishChart(
                                        selectedJob.id,
                                        artifactIndex,
                                        envelope.artifacts[artifactIndex]?.path ?? "",
                                        settings,
                                      )
                                    }
                                    onHandOff={(seed) => {
                                      setChartSeed(seed);
                                      setVisualizing(true);
                                    }}
                                  />
                                ) : undefined}
                              />
                            )}
                            {table && (
                              <>
                                <ResultTable
                                  columns={table.columns}
                                  rows={table.rows}
                                  copyable
                                  footerNote={
                                    <>
                                      Rows{" "}
                                      {table.rows.length ? tableOffset + 1 : 0}–
                                      {tableOffset + table.rows.length} of{" "}
                                      {fmt(table.filtered_total ?? table.total)}{" "}
                                      matching rows ({fmt(table.total)} total)
                                      <div className="drop-actions">
                                        <button
                                          className="text-button"
                                          disabled={tableOffset === 0 || !!busy}
                                          onClick={() =>
                                            setTableOffset(
                                              Math.max(0, tableOffset - 500),
                                            )
                                          }
                                        >
                                          Previous page
                                        </button>
                                        <button
                                          className="text-button"
                                          disabled={!table.truncated || !!busy}
                                          onClick={() =>
                                            setTableOffset(tableOffset + 500)
                                          }
                                        >
                                          Next page
                                        </button>
                                      </div>
                                    </>
                                  }
                                />
                              </>
                            )}
                            {selectedJob &&
                              (selectedJob.state === "DONE" ||
                                selectedJob.state === "PARTIAL") && (
                                <div className="drop-actions">
                                  <button
                                    className="text-button"
                                    disabled={!!busy}
                                    onClick={() => {
                                      setCompareArmed(selectedJob);
                                      setCompareView(null);
                                    }}
                                  >
                                    Compare with another run…
                                  </button>
                                </div>
                              )}
                            <details className="provenance">
                              <summary>
                                <ShieldCheck size={16} />
                                Source & reproducibility record
                              </summary>
                              <p>
                                {envelope.inputs.length} input documents ·
                                corpus fingerprint
                              </p>
                              <code>{envelope.corpus_sha256}</code>
                              {selectedJob.scope && (
                                <p>
                                  Scope: {selectedJob.scope.document_count}{" "}
                                  documents;
                                  {selectedJob.scope.explicit_documents
                                    ? " chosen individually"
                                    : " all matching documents"}
                                  .
                                  {(selectedJob.scope.date_from ||
                                    selectedJob.scope.date_to) && (
                                    <>
                                      {" "}
                                      From{" "}
                                      {selectedJob.scope.date_from ??
                                        "the earliest date"}{" "}
                                      through{" "}
                                      {selectedJob.scope.date_to ??
                                        "the latest date"}
                                      , inclusive.
                                      {selectedJob.scope.include_undated
                                        ? " Undated documents allowed by the date filter."
                                        : " Undated documents excluded by the date filter."}
                                    </>
                                  )}{" "}
                                  The input manifest records the exact documents
                                  and date provenance.
                                </p>
                              )}
                              <h3>Parameters</h3>
                              <pre>
                                {JSON.stringify(envelope.params, null, 2)}
                              </pre>
                              <p>
                                The exported result.json includes input hashes,
                                settings, and diagnostics.
                              </p>
                            </details>
                          </>
                        )}
                      </section>
                    )}
                  </>
                )}
                {page === "setup" && (
                  <>
                    <div className="page-heading">
                      <div>
                        <span className="eyebrow">
                          A HEALTHY RESEARCH ENVIRONMENT
                        </span>
                        <h1>Settings & backups</h1>
                        <p>Know what’s available before you press Run.</p>
                      </div>
                      <button
                        className="secondary"
                        disabled={!!busy}
                        onClick={() =>
                          void perform("Checking environment…", async () => {
                            const [environment, catalog] = await Promise.all([
                              api<Setup>("/setup"),
                              api<Tool[]>("/tools"),
                            ]);
                            setSetup(environment);
                            setTools(catalog);
                          })
                        }
                      >
                        Check again
                      </button>
                    </div>
                    <section className="panel settings-panel">
                      <div className="section-title">
                        <h2>Local analysis engine</h2>
                        <span className="ready-label">
                          <CheckCircle2 size={15} />
                          Connected
                        </span>
                      </div>
                      <dl>
                        <dt>Workspace storage</dt>
                        <dd>
                          <code>{setup?.workspace}</code>
                        </dd>
                        <dt>Document privacy</dt>
                        <dd>
                          Local storage and processing. No telemetry or cloud
                          account.
                        </dd>
                      </dl>
                    </section>
                    <section className="panel settings-panel">
                      <h2>Analysis readiness</h2>
                      <p className="muted">
                        The installer includes the English language tools. If a
                        component is unavailable, reinstall the complete app.
                        Your projects are stored separately from the
                        installation. Some analyses also need a resource file
                        you choose.
                      </p>
                      {/* The engine says which components are missing. Showing
                          only "Unavailable in this installation" threw that
                          away and left nothing to act on. */}
                      {tools
                        .filter(
                          (tool) => tool.availability?.state === "needs_setup",
                        )
                        .map((tool) => (
                          <div className="dependency-row" key={tool.name}>
                            <strong>{tool.label}</strong>
                            <span>
                              Needs{" "}
                              {(tool.availability?.missing ?? [])
                                .map((name) => componentLabel(setup, name))
                                .join(", ") || "a component that is missing"}
                            </span>
                          </div>
                        ))}
                      {tools.every(
                        (tool) => tool.availability?.state !== "needs_setup",
                      ) && (
                        <p className="ready-label">
                          Included analysis components are ready.
                        </p>
                      )}
                    </section>
                    <section className="panel settings-panel">
                      <h2>Installed components</h2>
                      <p className="muted">
                        What this installation can do, and what each part is
                        for. Nothing here is checked over the network.
                      </p>
                      {(setup?.components ?? []).map((component) => (
                        <div className="dependency-row" key={component.name}>
                          <strong>
                            {component.present ? (
                              <CheckCircle2 size={14} aria-hidden="true" />
                            ) : (
                              <TriangleAlert size={14} aria-hidden="true" />
                            )}{" "}
                            {component.label}
                          </strong>
                          <span>
                            {component.purpose}{" "}
                            {!component.present && (
                              <em>
                                Not found — the analyses above are affected.
                              </em>
                            )}
                          </span>
                        </div>
                      ))}
                    </section>
                    <Notices />
                    <ProjectManager
                      project={project}
                      busy={!!busy}
                      perform={perform}
                      refresh={async (selectProjectId) => {
                        const available = await api<Project[]>("/projects");
                        setProjects(available);
                        setProjectId((current) =>
                          selectProjectId && available.some((p) => p.id === selectProjectId)
                            ? selectProjectId
                            : available.some((p) => p.id === current)
                            ? current
                            : available[0]?.id || "",
                        );
                      }}
                    />
                    <section className="panel settings-panel">
                      <h2>Project backup & restore</h2>
                      <p>
                        Save documents, originals, completed runs, and
                        provenance together. Restore creates a separate project.
                        Wait for or cancel active runs before backing up.
                      </p>
                      <div className="drop-actions">
                        <button
                          className="secondary"
                          disabled={!!busy || !projectId || running > 0}
                          onClick={() =>
                            void perform("Backing up project…", () =>
                              download(
                                `/projects/${projectId}/backup`,
                                `project-${projectId.slice(0, 8)}.nlpsuite`,
                              ),
                            )
                          }
                        >
                          Back up project
                        </button>
                        <button
                          className="secondary"
                          disabled={!!busy}
                          onClick={() => restoreRef.current?.click()}
                        >
                          Restore project
                        </button>
                      </div>
                      <input
                        ref={restoreRef}
                        aria-label="Restore a project archive"
                        type="file"
                        accept=".nlpsuite,.zip"
                        hidden
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          e.target.value = "";
                          if (file)
                            void perform("Restoring project…", async () => {
                              const restored = await api<Project>(
                                "/projects/restore",
                                { method: "POST", body: file },
                              );
                              setProjects(await api<Project[]>("/projects"));
                              setProjectId(restored.id);
                              setPage("overview");
                              setNotice("Project restored as a separate copy.");
                            });
                        }}
                      />
                    </section>
                    <section className="panel settings-panel">
                      <h2>A few things worth knowing</h2>
                      <p>
                        Imports preserve original documents and extract analysis
                        text; original files are never renamed or edited.
                        Reimporting an unchanged file skips it. Runs retain the
                        document selection from the moment they were submitted.
                      </p>
                      <p>
                        Jobs run one at a time to keep resource use predictable.
                        Parser-assisted workflows currently target English. A
                        successful run is evidence that the tool completed—not
                        proof of scientific validity or full legacy parity.
                      </p>
                    </section>
                  </>
                )}
              </>
            )}
          </main>
        </div>
        {dragging && (
          <div className="drag-overlay">
            <Upload size={48} />
            <h2>Drop to add to your corpus</h2>
            <p>We’ll preserve a copy. Your originals stay untouched.</p>
          </div>
        )}
        {newProject && (
          <Dialog
            label="Create a research project"
            close={() => !busy && setNewProject(false)}
          >
            <p className="muted">
              A dedicated home for a corpus, its analyses, and the evidence
              behind them.
            </p>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void perform("Creating project…", async () => {
                  const p = await post<Project>("/projects", {
                    name: projectName,
                  });
                  setProjects(await api<Project[]>("/projects"));
                  setProjectId(p.id);
                  setNewProject(false);
                  setProjectName("");
                  setPage("corpus");
                });
              }}
            >
              <label className="field-label">
                Project name
                <input
                  autoFocus
                  required
                  maxLength={120}
                  placeholder="e.g. Community interviews"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                />
              </label>
              <div className="dialog-actions">
                <button
                  type="button"
                  className="secondary"
                  disabled={!!busy}
                  onClick={() => setNewProject(false)}
                >
                  Cancel
                </button>
                <button
                  className="primary"
                  disabled={!!busy || !projectName.trim()}
                >
                  Create project <ArrowRight size={16} />
                </button>
              </div>
            </form>
          </Dialog>
        )}
        {preview && (
          <Dialog label={preview.name} close={() => setPreview(null)}>
            <div className="preview-meta">
              <span>{fmt(preview.words)} words</span>
              <span>{(preview.bytes / 1024).toFixed(1)} KB</span>
              <span>
                <ShieldCheck size={14} />
                Source copy preserved
              </span>
            </div>
            <pre className="document-preview">{preview.text}</pre>
            {preview.truncated && (
              <p className="muted">
                Preview limited to the first 30,000 characters. The full
                document is used for analysis.
              </p>
            )}
            <details className="provenance">
              <summary>Document fingerprint</summary>
              <code>{preview.sha256}</code>
            </details>
          </Dialog>
        )}
        {analysis && (
          <Dialog
            label={analysis.label}
            close={() => !busy && !resourceUploads && setAnalysis(null)}
          >
            <p className="muted">{analysis.description}</p>
            {analysis.availability && (
              <p role="status" className="muted">
                {analysis.availability.message}
              </p>
            )}
            <div className="analysis-scope">
              <Files size={19} />
              <span>
                <strong>
                  {analysis.input_kind === "csv"
                    ? "Selected CSV table"
                    : `${selectedDocuments(documents, corpusSelection).length} documents`}
                </strong>{" "}
                in {project?.name || "no project selected"}
                <small>A fixed snapshot is captured when you start.</small>
              </span>
            </div>
            <form
              onKeyDown={(e) => {
                // Ctrl/Cmd+Enter submits here first and marks the keystroke
                // handled, so the interactive bench behind this dialog does
                // not also treat it as its own Run.
                if (
                  (e.ctrlKey || e.metaKey) &&
                  e.key === "Enter" &&
                  !e.defaultPrevented
                ) {
                  e.preventDefault();
                  e.currentTarget.requestSubmit();
                }
              }}
              onSubmit={(e) => {
                e.preventDefault();
                if (
                  resourceUploads ||
                  busy ||
                  (analysis.input_kind !== "csv" &&
                    selectionError(documents, corpusSelection))
                )
                  return;
                void perform("Submitting analysis…", async () => {
                  const cleaned = Object.fromEntries(
                    Object.entries(params).map(([k, v]) => [
                      k,
                      v === "" &&
                      analysis.params.find((p) => p.name === k)?.type !== "str"
                        ? null
                        : v,
                    ]),
                  );
                  const job = await post<Job>(`/projects/${projectId}/jobs`, {
                    tool: analysis.name,
                    params: cleaned,
                    parser,
                    ...(analysis.input_kind !== "csv"
                      ? { selection: corpusSelection }
                      : {}),
                  });
                  setJobs(await api<Job[]>(`/projects/${projectId}/jobs`));
                  setAnalysis(null);
                  setPage("runs");
                  setSelectedJob(job);
                  setEnvelope(null);
                  setTable(null);
                  setNotice(
                    "Analysis submitted. You can keep exploring while it runs.",
                  );
                });
              }}
            >
              {/* The parser was fixed at "spacy" in this request body, so a
                  machine with a working Stanza install could not use it and
                  nobody could see which parser had run. Both backends are
                  offered; one whose model is not installed is shown as such
                  rather than hidden, because that is the thing to go and fix. */}
              {analysis.input_kind !== "csv" && (
                <CorpusScope
                  documents={documents}
                  value={corpusSelection}
                  onChange={setCorpusSelection}
                  disabled={!!busy}
                />
              )}
              {analysis.requires_parse && (
                <label className="field-label">
                  English parser
                  <select
                    value={parser}
                    onChange={(e) => setParser(e.target.value)}
                  >
                    {PARSERS.map((option) => {
                      const ready = parserReady(setup, option.id);
                      return (
                        <option
                          key={option.id}
                          value={option.id}
                          disabled={!ready}
                        >
                          {option.label}
                          {ready ? "" : " — model not installed"}
                        </option>
                      );
                    })}
                  </select>
                  <small>
                    {PARSERS.find((option) => option.id === parser)?.hint} If
                    its model turns out to be unusable, the run says so and uses
                    the other one rather than failing.
                  </small>
                </label>
              )}
              <ParamFields
                tool={analysis}
                params={params}
                onChange={setParams}
                onBusyChange={(uploading) =>
                  setResourceUploads((count) => count + (uploading ? 1 : -1))
                }
              />
              <div className="dialog-actions">
                <span className="muted">
                  <Clock3 size={14} />
                  Runs in the background
                </span>
                <button
                  className="primary"
                  disabled={
                    !!busy ||
                    resourceUploads > 0 ||
                    !projectId ||
                    analysis.availability?.state === "needs_setup" ||
                    desktopParams(analysis).some(
                      (param) =>
                        param.required &&
                        (params[param.name] == null ||
                          String(params[param.name]).trim() === ""),
                    ) ||
                    (analysis.input_kind !== "csv" &&
                      !!selectionError(documents, corpusSelection))
                  }
                >
                  Run analysis <ArrowRight size={16} />
                </button>
              </div>
            </form>
          </Dialog>
        )}
        {compareArmed && (
          <Dialog
            label={
              compareView
                ? "Run comparison"
                : `Compare “${title(compareArmed.tool)}” with…`
            }
            close={() => {
              setCompareArmed(null);
              setCompareView(null);
            }}
          >
            {!compareView ? (
              <div className="run-list">
                {jobs
                  .filter(
                    (job) =>
                      job.id !== compareArmed.id &&
                      (job.state === "DONE" || job.state === "PARTIAL"),
                  )
                  .map((job) => (
                    <button
                      key={job.id}
                      className="run-row"
                      disabled={compareBusy || !!busy}
                      onClick={() => void runCompare(compareArmed, job)}
                    >
                      <span className="run-icon">
                        <FlaskConical size={18} />
                      </span>
                      <span className="run-name">
                        <strong>{title(job.tool)}</strong>
                        <small>
                          {when(job.created)} · {job.state}
                        </small>
                      </span>
                      <ChevronRight size={16} />
                    </button>
                  ))}
                {!jobs.some(
                  (job) =>
                    job.id !== compareArmed.id &&
                    (job.state === "DONE" || job.state === "PARTIAL"),
                ) && (
                  <div className="empty-inline">
                    <p>
                      No other finished run to compare against yet. Run the
                      same analysis with the other backend first.
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div>
                <p>
                  {!compareView.ok
                    ? "The comparison could not run."
                    : compareView.passed
                      ? "These two runs agree on every row, column and parameter the comparison checks."
                      : `These two runs disagree — every difference is listed below.`}
                </p>
                {compareView.ok &&
                  !compareView.passed &&
                  compareView.diff && (
                    <ResultTable
                      columns={compareView.diff.columns}
                      rows={compareView.diff.rows}
                    />
                  )}
                {!!compareView.diagnostics?.length && (
                  <ul className="muted">
                    {compareView.diagnostics.map((entry) => (
                      <li key={`${entry.code}-${entry.message}`}>
                        {entry.message}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </Dialog>
        )}
        {visualizing && selectedJob && envelope && (
          <Dialog
            label="Visualize this result"
            close={() => !busy && setVisualizing(false)}
          >
            <VisualizeDialog
              tools={tools}
              table={table}
              artifactPath={envelope.artifacts[artifactIndex]?.path ?? ""}
              canRun={!busy && !!projectId}
              components={setup?.components ?? []}
              seed={chartSeed}
              onSubmit={(tool, vizParams) => {
                void perform("Building visualization…", async () => {
                  const job = await post<Job>(
                    `/projects/${projectId}/jobs/${selectedJob.id}/visualize`,
                    { tool, index: artifactIndex, params: vizParams },
                  );
                  setJobs(await api<Job[]>(`/projects/${projectId}/jobs`));
                  setVisualizing(false);
                  setPage("runs");
                  setSelectedJob(job);
                  setEnvelope(null);
                  setTable(null);
                  setNotice("Visualization submitted. Watch Runs & results.");
                });
              }}
            />
          </Dialog>
        )}
      </div>
    </DialogError.Provider>
  );
}
