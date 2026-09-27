import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CopyPlus,
  GitCompareArrows,
  LoaderCircle,
  Play,
  Plus,
  Save,
  Trash2,
  TriangleAlert,
  X,
} from "lucide-react";

import { post, when, type Job, type Project } from "./api";
import { PanelSection } from "./PanelSection";
import {
  METHODS,
  comparisonRuns,
  createComparison,
  definitionProblems,
  deleteComparison,
  detailValues,
  duplicateComparison,
  formatFocus,
  listComparisons,
  newDefinition,
  parseFocus,
  previewComparison,
  projectFields,
  runComparison,
  runEnvelope,
  runTable,
  saveComparison,
  type Comparison,
  type ComparisonRun,
  type Definition,
  type Method,
  type Preview,
  type Selection,
  type Side,
  type SidePreview,
} from "./comparisons";

const errorText = (caught: unknown) =>
  caught instanceof Error ? caught.message : String(caught);
const MAX_SIDES = 6;
const PREVIEW_DELAY_MS = 450;
const COMPARING =
  "Comparing. The first run over new documents parses them; later runs read the parse back.";
const RUN_POLL_MS = 1500;

type CarryoverRow = Record<string, string>;
type SourceContext = { name: string; excerpt: string; found: boolean };
type OpenCarryover = {
  row: CarryoverRow;
  sourceA: SourceContext;
  sourceB: SourceContext;
};

type Working = Pick<Comparison, "id" | "name" | "definition" | "revision">;

/** One side's documents: which project, and which of its documents by their details and dates. */
function SideCard({
  side,
  index,
  projects,
  values,
  summary,
  removable,
  onChange,
  onRemove,
}: {
  side: Side;
  index: number;
  projects: Project[];
  values: Record<string, string[]> | undefined;
  summary: Preview["sides"][number] | undefined;
  removable: boolean;
  onChange: (side: Side) => void;
  onRemove: () => void;
}) {
  const selection: Selection = side.selection ?? {};
  const filters = selection.fields ?? {};
  const setSelection = (next: Selection) => {
    const empty =
      !Object.keys(next.fields ?? {}).length &&
      !next.date_from &&
      !next.date_to;
    onChange({ ...side, selection: empty ? null : next });
  };
  const setFilter = (name: string, chosen: string[] | null) => {
    const fields = { ...filters };
    if (chosen === null) delete fields[name];
    else fields[name] = chosen;
    setSelection({
      ...selection,
      fields: Object.keys(fields).length ? fields : null,
    });
  };
  const unused = Object.keys(values ?? {}).filter((name) => !(name in filters));
  return (
    <section className="side-card" aria-label={`Side ${index + 1}`}>
      <div className="side-card-head">
        <input
          className="side-name"
          value={side.name}
          onChange={(e) => onChange({ ...side, name: e.target.value })}
          aria-label={`Side ${index + 1} name`}
        />
        {removable && (
          <button
            className="icon-button"
            aria-label={`Remove side ${index + 1}`}
            onClick={onRemove}
          >
            <X size={16} />
          </button>
        )}
      </div>
      <label className="side-field">
        <span>Project</span>
        <select
          value={side.project_id}
          aria-label={`Side ${index + 1} project`}
          onChange={(e) =>
            onChange({ ...side, project_id: e.target.value, selection: null })
          }
        >
          {!projects.some((p) => p.id === side.project_id) && (
            <option value={side.project_id}>(not in this workspace)</option>
          )}
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name}
            </option>
          ))}
        </select>
      </label>
      {Object.entries(filters).map(([name, chosen]) => (
        <fieldset key={name} className="side-filter">
          <legend>
            {name}
            <button
              className="link-button"
              onClick={() => setFilter(name, null)}
              aria-label={`Remove the ${name} filter`}
            >
              <X size={12} />
            </button>
          </legend>
          <div className="side-filter-values">
            {(values?.[name] ?? chosen).map((value) => (
              <label key={value}>
                <input
                  type="checkbox"
                  checked={chosen.some(
                    (c) => c.toLowerCase() === value.toLowerCase(),
                  )}
                  onChange={(e) => {
                    const next = e.target.checked
                      ? [...chosen, value]
                      : chosen.filter(
                          (c) => c.toLowerCase() !== value.toLowerCase(),
                        );
                    setFilter(name, next.length ? next : null);
                  }}
                />
                {value}
              </label>
            ))}
          </div>
        </fieldset>
      ))}
      {unused.length > 0 && (
        <select
          className="side-add-filter"
          value=""
          aria-label={`Side ${index + 1}: keep only documents whose detail is`}
          onChange={(e) => {
            const name = e.target.value;
            if (name) setFilter(name, values?.[name]?.slice(0, 1) ?? []);
          }}
        >
          <option value="">Keep only documents whose…</option>
          {unused.map((name) => (
            <option key={name} value={name}>
              {name} is…
            </option>
          ))}
        </select>
      )}
      <div className="side-dates">
        <label>
          <span>From</span>
          <input
            type="date"
            value={selection.date_from ?? ""}
            aria-label={`Side ${index + 1} from date`}
            onChange={(e) =>
              setSelection({ ...selection, date_from: e.target.value || null })
            }
          />
        </label>
        <label>
          <span>To</span>
          <input
            type="date"
            value={selection.date_to ?? ""}
            aria-label={`Side ${index + 1} to date`}
            onChange={(e) =>
              setSelection({ ...selection, date_to: e.target.value || null })
            }
          />
        </label>
      </div>
      <p
        className={summary?.error ? "side-summary error" : "side-summary"}
        role="status"
      >
        {!summary ? "Counting…" : describeSide(summary)}
      </p>
    </section>
  );
}

/** One side's preview in a line: its size, its years, how many values each detail takes. */
function describeSide(summary: SidePreview): string {
  if (summary.error !== undefined) return summary.error;
  const years = summary.first_date
    ? `, ${summary.first_date.slice(0, 4)}–${(summary.last_date ?? "").slice(0, 4)}`
    : "";
  const details = Object.entries(summary.details)
    .map(([name, { count, examples }]) =>
      count === 1 ? ` · ${name}: ${examples[0]}` : ` · ${count} ${name} values`,
    )
    .join("");
  return (
    `${summary.documents.toLocaleString()} document${summary.documents === 1 ? "" : "s"}, ` +
    `${summary.words.toLocaleString()} words${years}${details}`
  );
}

/** What a finished comparison found, in sentences, above its figures. */
function Findings({
  projectId,
  run,
}: {
  projectId: string;
  run: ComparisonRun;
}) {
  const [readings, setReadings] = useState<string[]>([]);
  const [carryovers, setCarryovers] = useState<CarryoverRow[]>([]);
  const [opening, setOpening] = useState<number | null>(null);
  const [openError, setOpenError] = useState("");
  const [opened, setOpened] = useState<OpenCarryover | null>(null);
  useEffect(() => {
    let alive = true;
    setReadings([]);
    setCarryovers([]);
    setOpened(null);
    setOpenError("");
    if (!run.run_dir) return;
    runEnvelope(projectId, run.id)
      .then(async (envelope) => {
        const found: string[] = [];
        let passages: CarryoverRow[] = [];
        for (const [index, artifact] of envelope.artifacts.entries()) {
          if (/contrast_(measure|tone)_tests\.csv$/.test(artifact.path)) {
            const table = await runTable(projectId, run.id, index);
            for (const row of table.rows)
              if (
                ["All groups", "All documents"].includes(row.Group) &&
                row.Reading
              )
                found.push(row.Reading);
          } else if (/contrast_carryover\.csv$/.test(artifact.path)) {
            passages = (await runTable(projectId, run.id, index)).rows;
          }
        }
        if (alive) {
          setReadings(found);
          setCarryovers(passages);
        }
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [projectId, run.id, run.run_dir]);
  const readCarryover = async (row: CarryoverRow, index: number) => {
    setOpening(index);
    setOpenError("");
    setOpened(null);
    try {
      const [sourceA, sourceB] = await Promise.all([
        post<SourceContext>(
          `/projects/${encodeURIComponent(row["Project ID A"])}/documents/${encodeURIComponent(row["Document ID A"])}/context`,
          { passage: row["Passage A"] },
        ),
        post<SourceContext>(
          `/projects/${encodeURIComponent(row["Project ID B"])}/documents/${encodeURIComponent(row["Document ID B"])}/context`,
          { passage: row["Passage B"] },
        ),
      ]);
      setOpened({ row, sourceA, sourceB });
    } catch (caught) {
      setOpenError(errorText(caught));
    } finally {
      setOpening(null);
    }
  };
  const notes = (run.diagnostics ?? [])
    .filter((d) => d.code === "CONTRAST_NOTE")
    .map((d) => d.message);
  const warnings = (run.diagnostics ?? [])
    .filter((d) => d.severity !== "INFO")
    .map((d) => d.message);
  return (
    <div className="compare-findings">
      {notes.length > 0 && (
        <div className="compare-notes" role="note">
          <strong>Read these first</strong>
          <ul>
            {notes.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        </div>
      )}
      {readings.length > 0 && (
        <ul className="compare-readings">
          {readings.map((reading, index) => (
            <li key={index}>{reading}</li>
          ))}
        </ul>
      )}
      {carryovers.length > 0 && (
        <section
          className="compare-carryover"
          aria-label="Carried-over passages"
        >
          <h3>Carried-over passages</h3>
          <ol>
            {carryovers.map((row, index) => (
              <li
                key={`${row["Document ID A"]}-${row["Sentence ID A"]}-${row["Document ID B"]}`}
              >
                <p>
                  <strong>{row["Document A"]}</strong>: “{row["Passage A"]}”
                  <br />
                  <strong>{row["Document B"]}</strong>: “{row["Passage B"]}”
                  <small>
                    {" "}
                    {row.Similarity} similarity · {row.Group}
                  </small>
                </p>
                <button
                  type="button"
                  className="text-button"
                  disabled={opening !== null}
                  onClick={() => void readCarryover(row, index)}
                >
                  {opening === index ? "Opening sources…" : "Read in context"}
                </button>
              </li>
            ))}
          </ol>
          {openError && (
            <p className="alert warning" role="status">
              {openError}
            </p>
          )}
          {opened && (
            <div
              className="compare-carryover-context"
              role="region"
              aria-label="Source context"
            >
              <button
                type="button"
                className="text-button"
                onClick={() => setOpened(null)}
              >
                Close source context
              </button>
              <div className="compare-context-pair">
                <article>
                  <h4>{opened.sourceA.name}</h4>
                  <p>{opened.row["Passage A"]}</p>
                  <pre>
                    {opened.sourceA.found
                      ? opened.sourceA.excerpt
                      : "This sentence could not be located in the stored source."}
                  </pre>
                </article>
                <article>
                  <h4>{opened.sourceB.name}</h4>
                  <p>{opened.row["Passage B"]}</p>
                  <pre>
                    {opened.sourceB.found
                      ? opened.sourceB.excerpt
                      : "This sentence could not be located in the stored source."}
                  </pre>
                </article>
              </div>
              <small>
                Sentence IDs {opened.row["Sentence ID A"]} and{" "}
                {opened.row["Sentence ID B"]} in their imported text.
              </small>
            </div>
          )}
        </section>
      )}
      {warnings.length > 0 && (
        <ul className="compare-warnings">
          {warnings.map((warning, index) => (
            <li key={index}>
              <TriangleAlert size={13} /> {warning}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * The Compare page: two to six sets of documents, from one project or
 * several, compared on the same measures (docs/PLAN_0.5.0.md section 3).
 *
 * A comparison lives in the project that is open; its sides may name any
 * project. Running it queues an ordinary run of the `contrast` tool here, so
 * its tables open in Past runs and its figures are the panels every run has.
 */
export function Compare({
  projectId,
  projects,
  parser,
  jobs,
  onOpenJob,
}: {
  projectId: string;
  projects: Project[];
  parser: string;
  jobs: Job[];
  onOpenJob: (job: Job) => void;
}) {
  const [saved, setSaved] = useState<Comparison[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [working, setWorking] = useState<Working | null>(null);
  const [dirty, setDirty] = useState(false);
  const [focusText, setFocusText] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  // Why the sides could not be counted (an invalid definition), shown by the Compare button.
  const [previewError, setPreviewError] = useState("");
  const [fieldsByProject, setFieldsByProject] = useState<
    Record<string, Record<string, string[]>>
  >({});
  const [runs, setRuns] = useState<ComparisonRun[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    const list = await listComparisons(projectId);
    setSaved(list);
    return list;
  }, [projectId]);

  useEffect(() => {
    setSaved([]);
    setActiveId(null);
    setWorking(null);
    setRuns([]);
    setError("");
    let alive = true;
    refresh()
      .then((list) => alive && list[0] && setActiveId(list[0].id))
      .catch((caught) => alive && setError(errorText(caught)));
    comparisonRuns(projectId)
      .then((list) => alive && setRuns(list))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [projectId, refresh]);

  useEffect(() => {
    const found = saved.find((item) => item.id === activeId);
    if (!found) return;
    setWorking({
      id: found.id,
      name: found.name,
      definition: found.definition,
      revision: found.revision,
    });
    setFocusText(formatFocus(found.definition.focus));
    setDirty(false);
    // Only when a different comparison is chosen; saving refreshes `saved` too.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  // Each side's project's details, for its filter menus (fetched once per project).
  const projectsInUse = useMemo(
    () => [
      ...new Set(
        (working?.definition.sides ?? []).map((side) => side.project_id),
      ),
    ],
    [working?.definition.sides],
  );
  useEffect(() => {
    for (const id of projectsInUse) {
      if (id in fieldsByProject || !projects.some((p) => p.id === id)) continue;
      setFieldsByProject((current) => ({ ...current, [id]: {} }));
      projectFields(id)
        .then((fields) =>
          setFieldsByProject((current) => ({
            ...current,
            [id]: detailValues(fields),
          })),
        )
        .catch(() => undefined);
    }
  }, [projectsInUse, fieldsByProject, projects]);

  // What each side holds, and what the sides can be lined up by.
  const definitionKey = JSON.stringify(working?.definition ?? null);
  useEffect(() => {
    if (!working) return;
    let alive = true;
    const timer = window.setTimeout(() => {
      previewComparison(projectId, working.definition)
        .then((found) => {
          if (!alive) return;
          setPreview(found);
          setPreviewError("");
        })
        .catch((caught) => alive && setPreviewError(errorText(caught)));
    }, PREVIEW_DELAY_MS);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [definitionKey, projectId]);

  const latest = runs.find((run) => run.comparison_id === working?.id) ?? null;
  const live = latest
    ? (jobs.find((job) => job.id === latest.id) ?? latest)
    : null;
  const running = live != null && ["QUEUED", "RUNNING"].includes(live.state);

  // Refresh the runs list while one runs, so its diagnostics and run_dir arrive.
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      comparisonRuns(projectId)
        .then(setRuns)
        .catch(() => undefined);
    }, RUN_POLL_MS);
    return () => window.clearInterval(timer);
  }, [running, projectId]);
  // "Comparing…" describes a run in progress; the status line takes over once it ends.
  useEffect(() => {
    if (!running)
      setNotice((current) => (current === COMPARING ? "" : current));
  }, [running]);
  useEffect(() => {
    if (live && latest && live.state !== latest.state)
      comparisonRuns(projectId)
        .then(setRuns)
        .catch(() => undefined);
  }, [live?.state, latest?.state, projectId, live, latest]);

  const edit = (change: (definition: Definition) => Definition) => {
    setWorking((current) =>
      current
        ? { ...current, definition: change(current.definition) }
        : current,
    );
    setDirty(true);
  };

  const create = async () => {
    setBusy(true);
    try {
      const taken = new Set(saved.map((item) => item.name.toLowerCase()));
      let name = "New comparison";
      for (let n = 2; taken.has(name.toLowerCase()); n++)
        name = `New comparison ${n}`;
      const made = await createComparison(
        projectId,
        name,
        newDefinition(projectId),
      );
      await refresh();
      setActiveId(made.id);
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  };

  const save = async (): Promise<Comparison | null> => {
    if (!working) return null;
    try {
      const stored = await saveComparison(projectId, working);
      setWorking({ ...working, revision: stored.revision });
      setDirty(false);
      await refresh();
      return stored;
    } catch (caught) {
      setError(errorText(caught));
      return null;
    }
  };

  const run = async () => {
    if (!working) return;
    setBusy(true);
    setNotice("");
    try {
      if (dirty && !(await save())) return;
      const job = await runComparison(projectId, working.id, parser);
      setRuns((current) => [
        { ...job, comparison_id: working.id, comparison_name: working.name },
        ...current,
      ]);
      setNotice(COMPARING);
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  };

  const problems = working ? definitionProblems(working.definition) : [];
  const sideErrors = (preview?.sides ?? []).filter((side) => side.error);

  return (
    <div>
      <div className="page-heading">
        <div>
          <span className="eyebrow">WORKSHOP</span>
          <h1>Compare</h1>
          <p>
            Two collections side by side, or two kinds of document within one:
            the same measures, the words that set each apart, and how often each
            uses the words you care about.
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
      <div className="compare-layout">
        <nav className="notebook-list" aria-label="Comparisons">
          <div className="notebook-list-actions">
            <button
              className="primary"
              disabled={busy}
              onClick={() => void create()}
            >
              <Plus size={15} /> New comparison
            </button>
          </div>
          <ul>
            {saved.map((item) => (
              <li key={item.id}>
                <button
                  className={
                    item.id === activeId
                      ? "notebook-link selected"
                      : "notebook-link"
                  }
                  aria-current={item.id === activeId ? "page" : undefined}
                  onClick={() => setActiveId(item.id)}
                >
                  <span>{item.name}</span>
                  <small>
                    {item.definition.sides.map((side) => side.name).join(" · ")}
                  </small>
                </button>
              </li>
            ))}
          </ul>
          {!saved.length && (
            <p className="muted notebook-empty">
              No comparisons yet. Start one, then choose what each side holds:
              another project, or documents of one kind (Kind = ina) in this
              one.
            </p>
          )}
        </nav>

        <div className="compare-main">
          {working ? (
            <>
              <div className="notebook-toolbar">
                <input
                  className="notebook-name"
                  value={working.name}
                  aria-label="Comparison name"
                  onChange={(e) => {
                    setWorking({ ...working, name: e.target.value });
                    setDirty(true);
                  }}
                />
                <span className="muted notebook-save-state">
                  {dirty
                    ? "Unsaved changes"
                    : `Saved · revision ${working.revision}`}
                </span>
                <div className="notebook-actions">
                  <button
                    className="primary"
                    disabled={
                      busy ||
                      running ||
                      problems.length > 0 ||
                      sideErrors.length > 0 ||
                      !!previewError
                    }
                    onClick={() => void run()}
                    title={
                      problems[0] ??
                      sideErrors[0]?.error ??
                      (previewError || "Run the comparison as a job")
                    }
                  >
                    <Play size={15} /> Compare
                  </button>
                  <button
                    className="secondary"
                    disabled={!dirty}
                    onClick={() => void save()}
                  >
                    <Save size={14} /> Save
                  </button>
                  <button
                    className="icon-button"
                    aria-label="Duplicate this comparison"
                    title="Duplicate"
                    onClick={() =>
                      void duplicateComparison(projectId, working.id)
                        .then(async (copy) => {
                          await refresh();
                          setActiveId(copy.id);
                        })
                        .catch((caught) => setError(errorText(caught)))
                    }
                  >
                    <CopyPlus size={16} />
                  </button>
                  <button
                    className="icon-button"
                    aria-label="Delete this comparison"
                    title="Delete"
                    onClick={() => {
                      if (
                        !window.confirm(
                          `Delete the comparison “${working.name}”? Its runs are kept.`,
                        )
                      )
                        return;
                      void deleteComparison(projectId, working.id)
                        .then(async () => {
                          const list = await refresh();
                          setWorking(null);
                          setActiveId(list[0]?.id ?? null);
                        })
                        .catch((caught) => setError(errorText(caught)));
                    }}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>

              <div className="side-cards">
                {working.definition.sides.map((side, index) => (
                  <SideCard
                    key={index}
                    side={side}
                    index={index}
                    projects={projects}
                    values={fieldsByProject[side.project_id]}
                    summary={preview?.sides[index]}
                    removable={working.definition.sides.length > 2}
                    onChange={(next) =>
                      edit((definition) => ({
                        ...definition,
                        sides: definition.sides.map((old, i) =>
                          i === index ? next : old,
                        ),
                      }))
                    }
                    onRemove={() =>
                      edit((definition) => ({
                        ...definition,
                        sides: definition.sides.filter((_, i) => i !== index),
                        methods: definition.methods,
                      }))
                    }
                  />
                ))}
                {working.definition.sides.length < MAX_SIDES && (
                  <button
                    className="side-add"
                    onClick={() =>
                      edit((definition) => ({
                        ...definition,
                        sides: [
                          ...definition.sides,
                          {
                            name: `Side ${String.fromCharCode(65 + definition.sides.length)}`,
                            project_id: projectId,
                            selection: null,
                          },
                        ],
                        methods: definition.methods.filter(
                          (m) => m !== "carryover",
                        ),
                      }))
                    }
                  >
                    <Plus size={15} /> Add a side
                  </button>
                )}
              </div>

              <div className="compare-options">
                <label className="field-label">
                  Line them up by
                  <select
                    value={working.definition.alignment}
                    aria-label="Line the sides up by"
                    onChange={(e) =>
                      edit((definition) => ({
                        ...definition,
                        alignment: e.target.value,
                      }))
                    }
                  >
                    <option value="none">Nothing: each side as a whole</option>
                    {(preview?.alignments ?? []).map((item) => (
                      <option
                        key={item.name}
                        value={`field:${item.name}`}
                        disabled={!item.shared_values}
                      >
                        {item.name}:{" "}
                        {item.shared_values
                          ? `${item.shared_values} on every side`
                          : "no value is on every side"}
                        {item.examples.length
                          ? ` (${item.examples.slice(0, 3).join(", ")}…)`
                          : ""}
                      </option>
                    ))}
                    {preview?.dated && <option value="period">Decade</option>}
                    {working.definition.alignment !== "none" &&
                      !(preview?.alignments ?? []).some(
                        (item) =>
                          `field:${item.name}` === working.definition.alignment,
                      ) &&
                      working.definition.alignment !== "period" && (
                        <option value={working.definition.alignment}>
                          {working.definition.alignment.replace("field:", "")}
                        </option>
                      )}
                  </select>
                  <small>
                    Compares like with like: the same speaker (or decade) on
                    each side, leaving out values only one side has.
                  </small>
                </label>
                <div className="field-label">
                  <span>Focus words (optional)</span>
                  <textarea
                    className="focus-words"
                    rows={3}
                    value={focusText}
                    aria-label="Focus words"
                    placeholder="immigration: immigration, immigrant, border, asylum"
                    onChange={(e) => {
                      setFocusText(e.target.value);
                      const focus = parseFocus(e.target.value);
                      edit((definition) => ({
                        ...definition,
                        focus: Object.keys(focus).length ? focus : null,
                      }));
                    }}
                  />
                  <small>
                    A name, a colon, then words; one group per line. Counted as
                    <select
                      className="inline-select"
                      value={working.definition.match}
                      aria-label="How focus words are counted"
                      onChange={(e) =>
                        edit((definition) => ({
                          ...definition,
                          match: e.target.value as Definition["match"],
                        }))
                      }
                    >
                      <option value="lemma">every form of each word</option>
                      <option value="form">the exact word forms</option>
                      <option value="exact-lowercase">
                        exact words in the raw text
                      </option>
                    </select>
                  </small>
                </div>
              </div>

              <fieldset className="compare-methods">
                <legend>Methods</legend>
                {METHODS.map((method) => {
                  const chosen = working.definition.methods.includes(method.id);
                  const needsTwo =
                    method.id === "carryover" &&
                    working.definition.sides.length > 2;
                  return (
                    <label
                      key={method.id}
                      className={
                        method.available ? "method" : "method unavailable"
                      }
                      title={method.help}
                    >
                      <input
                        type="checkbox"
                        checked={chosen}
                        disabled={!method.available || needsTwo}
                        onChange={(e) =>
                          edit((definition) => ({
                            ...definition,
                            methods: e.target.checked
                              ? [...definition.methods, method.id as Method]
                              : definition.methods.filter(
                                  (m) => m !== method.id,
                                ),
                          }))
                        }
                      />
                      <span>
                        <strong>{method.label}</strong>{" "}
                        <small className="muted">({method.cost})</small>
                        <small className="method-help">{method.help}</small>
                      </span>
                    </label>
                  );
                })}
              </fieldset>
              {(problems.length > 0 || previewError) && (
                <p className="muted compare-problems">
                  {[...problems, previewError].filter(Boolean).join(" ")}
                </p>
              )}

              {live && (
                <section className="compare-results" aria-label="Results">
                  <div className="notebook-run-status" role="status">
                    {running ? (
                      <>
                        <LoaderCircle size={15} className="spin" />{" "}
                        {live.stage || "Waiting"}…
                      </>
                    ) : (
                      <>
                        {live.state === "DONE"
                          ? `Compared ${when(live.finished ?? live.created)}.`
                          : `The comparison ${live.state === "PARTIAL" ? "finished with problems" : "failed"}: ${live.diagnostics?.find((d) => d.severity === "ERROR")?.message ?? live.state}`}
                        {live.run_dir && (
                          <button
                            className="secondary compact"
                            onClick={() => onOpenJob(live)}
                          >
                            Open in Past runs
                          </button>
                        )}
                      </>
                    )}
                  </div>
                  {!running && latest && (
                    <Findings projectId={projectId} run={latest} />
                  )}
                  {!running && live.run_dir && (
                    <PanelSection projectId={projectId} jobId={live.id} />
                  )}
                </section>
              )}
            </>
          ) : (
            <div className="notebook-placeholder">
              <GitCompareArrows size={34} strokeWidth={1.3} />
              <p>Choose a comparison, or start a new one.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
