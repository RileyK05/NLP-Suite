import { useMemo, useState, type ReactNode } from "react";
import {
  Bookmark,
  Copy,
  FlaskConical,
  GitCompareArrows,
  Save,
  TableProperties,
  Trash2,
  TriangleAlert,
  Undo2,
} from "lucide-react";
import { initialSettings, Workbench } from "./Workbench";
import { PanelSection } from "./PanelSection";
import type { Selection } from "./ChartCanvas";
import type { Settings } from "./chartLayout";
import { when, type Table, type TableSource } from "./api";
import {
  filtersFor,
  fromViewSettings,
  isDirty,
  publicationGaps,
  selectionFor,
  type ChartContract,
  type RowFilter,
  type View,
} from "./views";

/**
 * The interaction lab, as a place rather than a panel.
 *
 * The workbench existed before this page did, but only inside the record of
 * a finished run: you had to know which run had produced a table, find it,
 * expand it, and pick the right artifact before a chart appeared. That is a
 * filing cabinet with a chart in one of the drawers. Someone opening the app
 * to look at their data found nothing to look at.
 *
 * So this page asks one question — which table? — and then gets out of the
 * way. Everything a project has ever published is in the list, newest first,
 * and choosing one puts it straight under the controls.
 *
 * It also remembers. The arrangement someone works out — this measure against
 * that axis, grouped this way, drilled into that bar — used to live only as
 * long as the panel was open, so the comparison you wanted to show a colleague
 * next week was one you had to rebuild from memory. A saved view is that
 * arrangement written down, and reopening one is not a run: nothing is
 * computed, nothing is published, and the result it draws over is untouched.
 */

/**
 * The two ways to arrive at a chart.
 *
 * "live" runs the analysis now, over documents parsed once, and redraws as the
 * settings change. "results" charts a table some earlier run published. Live
 * is first and is the default because it is the thing people come here to do;
 * a finished result is what you go back to, not what you start from.
 */
export type Mode = "live" | "results";

/** Results first: a run's input manifest is a record, not a finding. */
export function orderSources(sources: TableSource[]): TableSource[] {
  return [
    ...sources.filter((source) => !source.manifest),
    ...sources.filter((source) => source.manifest),
  ];
}

/**
 * What to call a table in the picker.
 *
 * Several runs of the same tool publish files with the same name, so the
 * name alone cannot tell them apart — four rows reading "chart_data.csv" are
 * four rows the reader has to guess between.
 */
export function sourceLabel(source: TableSource): string {
  return `${source.label} — ${source.path} · ${when(source.created)}`;
}

/** The table to land on when the page opens with nothing chosen yet. */
export function firstWorthOpening(sources: TableSource[]): TableSource | null {
  return orderSources(sources)[0] ?? null;
}

export const sourceKey = (source: TableSource): string =>
  `${source.job}:${source.index}`;

/**
 * The tables worth comparing a choice against, in picking order.
 *
 * Two runs of the *same* analysis are the natural comparison — readability
 * before an edit against readability after it — so they come first; tables
 * from other tools follow, because "the same shape over a different count" is
 * still a legitimate thing to look at.
 */
export function compareCandidates(
  sources: TableSource[],
  selected: TableSource,
): TableSource[] {
  const others = orderSources(sources).filter(
    (source) => sourceKey(source) !== sourceKey(selected),
  );
  return [
    ...others.filter((source) => source.tool === selected.tool),
    ...others.filter((source) => source.tool !== selected.tool),
  ];
}

/** Where the compare picker lands when it opens: the best candidate, if any. */
export function defaultCompareSource(
  sources: TableSource[],
  selected: TableSource,
): TableSource | null {
  return compareCandidates(sources, selected)[0] ?? null;
}

/**
 * The arrangement's columns that a comparison table is missing.
 *
 * An arrangement is only meaningful over a table that has the columns it
 * names; listing what is absent beats drawing a chart of blank categories and
 * letting the reader conclude the other table is empty.
 */
export function missingCompareColumns(
  settings: Settings,
  columns: string[],
): string[] {
  const needed = [settings.y];
  // A histogram bins its measure; the x column is an ignored selector there,
  // so its absence in the other table says nothing about the comparison.
  if (settings.kind !== "histogram") needed.push(settings.x);
  if (settings.group) needed.push(settings.group);
  return needed.filter(
    (column): column is string => !!column && !columns.includes(column),
  );
}

/** The table a saved view was saved from, if this project still has it. */
export function sourceOfView(
  sources: TableSource[],
  view: View,
): TableSource | null {
  return (
    sources.find(
      (source) => source.job === view.job && source.index === view.index,
    ) ?? null
  );
}

/**
 * Whether an open view and the table on screen are the same file.
 *
 * Opening a view of another table asks the page above for that table, which
 * arrives a render later. Until it does these two disagree, and drawing that
 * pairing would flash a complaint about a column nobody chose.
 */
export function viewMatchesSource(
  view: View | null,
  source: TableSource | null,
): boolean {
  if (!view) return true;
  return !!source && view.job === source.job && view.index === source.index;
}

/** Views are listed under the table they belong to, most recent first. */
export function viewsOfSource(
  views: View[],
  source: TableSource | null,
): View[] {
  if (!source) return [];
  return views.filter(
    (view) => view.job === source.job && view.index === source.index,
  );
}

/** What to call a saved view in the list: its name, and how far it has moved. */
export function viewLabel(view: View): string {
  const revised = view.revision > 1 ? ` · revision ${view.revision}` : "";
  return `${view.name}${revised} · ${when(view.updated)}`;
}

/**
 * Managing the saved view, kept away from managing the chart.
 *
 * These four controls first lived in the workbench's action row, beside "Show
 * all rows" and "Publish this chart". That put five buttons in one line and,
 * worse, put Delete next to the primary action — a destructive thing one slip
 * away from the thing people came to press. Here the arrangement's own
 * controls sit above the chart, and Delete asks twice.
 */
export function ViewBar({
  view,
  dirty,
  onSave,
  onRevert,
  onDuplicate,
  onDelete,
}: {
  view: View;
  dirty: boolean;
  onSave: () => void;
  onRevert: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  return (
    <div className="view-bar">
      <span className="view-bar-name">
        <Bookmark size={15} aria-hidden="true" />
        <strong>{view.name}</strong>
        <small>
          {view.revision > 1 ? `revision ${view.revision}` : "saved"}
          {dirty ? " · unsaved changes" : ""}
        </small>
      </span>
      <span className="view-bar-actions">
        <button
          className="secondary"
          disabled={!dirty}
          title={
            dirty
              ? `Save these changes to “${view.name}”`
              : "No changes to save"
          }
          onClick={onSave}
        >
          <Save size={15} /> Save
        </button>
        {dirty && (
          <button
            className="secondary"
            title="Go back to the saved arrangement"
            onClick={onRevert}
          >
            <Undo2 size={15} /> Revert
          </button>
        )}
        <button
          className="secondary"
          title="Copy this view, so the original survives the next edit"
          onClick={onDuplicate}
        >
          <Copy size={15} /> Duplicate
        </button>
        {confirming ? (
          <>
            <button
              className="secondary danger"
              title={`Delete “${view.name}”. The result it charted is untouched.`}
              onClick={() => {
                setConfirming(false);
                onDelete();
              }}
            >
              <Trash2 size={15} /> Really delete
            </button>
            <button
              className="text-button"
              onClick={() => setConfirming(false)}
            >
              Keep it
            </button>
          </>
        ) : (
          <button
            className="secondary danger"
            title="Delete this view. The result it charted is untouched."
            onClick={() => setConfirming(true)}
          >
            <Trash2 size={15} /> Delete
          </button>
        )}
      </span>
    </div>
  );
}

/**
 * Whether this name is already taken, ignoring case and surrounding space.
 *
 * The server refuses a duplicate, and does it case-sensitively; this is
 * stricter on purpose. "Trend" and "trend" sitting next to each other in a
 * picker is a list nobody can read, and finding out after pressing the button
 * is later than finding out while typing.
 */
export function nameIsTaken(taken: string[], name: string): boolean {
  const wanted = name.trim().toLowerCase();
  if (!wanted) return false;
  return taken.some((existing) => existing.trim().toLowerCase() === wanted);
}

/** Naming an arrangement that has not been saved yet. */
export function SaveBar({
  taken,
  onSave,
}: {
  taken: string[];
  onSave: (name: string) => void;
}) {
  const [name, setName] = useState("");
  const trimmed = name.trim();
  const clash = nameIsTaken(taken, trimmed);
  return (
    <div className="view-bar">
      <span className="view-bar-name">
        <Bookmark size={15} aria-hidden="true" />
        <input
          aria-label="Name for this view"
          placeholder="Name this arrangement to come back to it…"
          maxLength={120}
          value={name}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && trimmed && !clash) {
              onSave(trimmed);
              setName("");
            }
          }}
        />
        {clash && (
          <small className="view-bar-clash">
            This project already has a view with that name.
          </small>
        )}
      </span>
      <span className="view-bar-actions">
        <button
          className="secondary"
          disabled={!trimmed || clash}
          title="Keep this arrangement. Nothing is run and nothing is published."
          onClick={() => {
            onSave(trimmed);
            setName("");
          }}
        >
          <Save size={15} /> Save view
        </button>
      </span>
    </div>
  );
}

export function Explore({
  projectId,
  sources,
  loading,
  selected,
  onSelect,
  table,
  tableLoading,
  canPublish,
  onPublish,
  onBrowseAnalyses,
  liveBench,
  defaultMode = "live",
  lockedMode,
  embedded = false,
  views,
  contract,
  onSaveView,
  onUpdateView,
  onDuplicateView,
  onDeleteView,
  compareSource = null,
  onCompareSource,
  compare = null,
  compareLoading = false,
}: {
  /** Needed to ask the run which purpose-built panels it declares. */
  projectId?: string | null;
  sources: TableSource[];
  loading: boolean;
  selected: TableSource | null;
  onSelect: (source: TableSource) => void;
  table: Table | null;
  tableLoading: boolean;
  canPublish: boolean;
  onPublish: (source: TableSource, settings: Settings) => void;
  onBrowseAnalyses: () => void;
  /** The live bench, rendered by the page above so Explore stays a layout. */
  liveBench: ReactNode;
  /**
   * Which side to open on. Live, unless a caller says otherwise: running an
   * analysis is what this page is for, and a finished result is what you come
   * back to rather than what you start from.
   */
  defaultMode?: Mode;
  /**
   * The mode this page *is*, rather than the one it opens on.
   *
   * Set by the page, which already said so in the sidebar. When set, the tab
   * pair is not drawn: it would offer a second route to a page that is one
   * click away, and the two routes would disagree about which is selected.
   */
  lockedMode?: Mode;
  /**
   * Whether this is the page or a section inside one.
   *
   * Embedded, it sits under the host page's `h1`, so it contributes an `h2`
   * and its workbench an `h3` -- the depths that panel already takes under a
   * run's own heading. Two `h1`s on one page is not a cosmetic problem: the
   * heading list is how a long page gets navigated.
   */
  embedded?: boolean;
  views: View[];
  contract: ChartContract;
  onSaveView: (
    name: string,
    source: TableSource,
    settings: Settings,
    filters: RowFilter[],
  ) => Promise<View | null>;
  onUpdateView: (
    view: View,
    source: TableSource,
    settings: Settings,
    filters: RowFilter[],
  ) => Promise<View | null>;
  onDuplicateView: (view: View) => Promise<View | null>;
  onDeleteView: (view: View) => Promise<boolean>;
  /** The table on the comparison side, chosen and fetched by the page above. */
  compareSource?: TableSource | null;
  onCompareSource?: (source: TableSource | null) => void;
  /** The comparison table's loaded rows; null until its fetch lands. */
  compare?: Table | null;
  compareLoading?: boolean;
}) {
  const [ownMode, setMode] = useState<Mode>(defaultMode);
  const mode = lockedMode ?? ownMode;
  const ordered = orderSources(sources);
  const results = ordered.filter((source) => !source.manifest);
  const manifests = ordered.filter((source) => source.manifest);

  const [openId, setOpenId] = useState<string | null>(null);

  const open = views.find((view) => view.id === openId) ?? null;
  const here = viewsOfSource(views, selected);
  const tableKey = selected ? sourceKey(selected) : "";

  /**
   * What is on screen: the saved arrangement, plus whatever has been changed
   * since.
   *
   * Derived rather than copied into state by an effect. An effect does not run
   * while the page is rendered to markup, so the first paint had no settings
   * at all and drew no chart; and an effect that syncs state to props has to
   * win a race with every other effect that touches the same props. Here the
   * base is a pure function of (table, open view), and an edit is tagged with
   * the pair it was made against — so changing table or view drops the edits
   * that belonged to the previous one without anything having to remember to.
   */
  const stateKey = `${tableKey}:${openId ?? ""}`;
  const [edit, setEdit] = useState<{
    key: string;
    settings: Settings;
    mark: Selection;
  } | null>(null);

  const base = useMemo(() => {
    if (!table) return null;
    const start = initialSettings(table);
    return {
      settings: open ? fromViewSettings(open.settings, start) : start,
      mark: open ? selectionFor(open.settings, open.filters) : null,
    };
  }, [table, open]);

  const settled = viewMatchesSource(open, selected);

  const live = edit && edit.key === stateKey ? edit : base;
  const settings = live?.settings ?? null;
  const mark = live?.mark ?? null;

  const setSettings = (next: Settings) => {
    setEdit({ key: stateKey, settings: next, mark });
    setComparisonMark(null);
  };
  const setMark = (next: Selection) => {
    if (settings) setEdit({ key: stateKey, settings, mark: next });
  };

  const filters = settings ? filtersFor(settings, mark) : [];
  const dirty = settings ? isDirty(open, settings, filters) : false;
  const gaps = settings ? publicationGaps(contract, settings) : [];
  const [comparisonMark, setComparisonMark] = useState<Selection>(null);
  const candidates = selected ? compareCandidates(sources, selected) : [];
  const missing =
    settings && compare ? missingCompareColumns(settings, compare.columns) : [];

  const reopen = (view: View) => {
    const source = sourceOfView(sources, view);
    if (source && (!selected || sourceKey(source) !== sourceKey(selected))) {
      onSelect(source);
    }
    setOpenId(view.id);
  };

  // Dropping the edits is the whole of it: what is drawn falls back to the
  // saved arrangement, which is where `base` came from.
  const revert = () => setEdit(null);

  return (
    <>
      {embedded ? (
        <div className="section-title">
          <div>
            <h2>Published tables &amp; saved views</h2>
            <p>
              Every table a finished run published, with the arrangements you
              saved over them. Charting one here changes nothing on disk.
            </p>
          </div>
          <span className="corpus-pill">
            <TableProperties size={16} />
            {results.length} table{results.length === 1 ? "" : "s"}
          </span>
        </div>
      ) : (
        <div className="page-heading">
          <div>
            <span className="eyebrow">
              {mode === "live"
                ? "LOOK BEFORE YOU PUBLISH"
                : "THE RESEARCH RECORD"}
            </span>
            <h1>
              {lockedMode === "live"
                ? "Interactive"
                : lockedMode === "results"
                  ? "Published tables"
                  : "Explore"}
            </h1>
            <p>
              {mode === "live" ? (
                <>
                  Ask questions of your documents and watch the answers redraw
                  as you change them. Parsing happens once; after that, changing
                  the question, a parameter or the chart is immediate. Nothing
                  is recorded until you publish.
                </>
              ) : (
                <>
                  Every table a finished run published, with the arrangements
                  you saved over them. Charting one here changes nothing on
                  disk.
                </>
              )}
            </p>
          </div>
          {/* Counts what the results side offers, so it belongs to the
            results side. Live mode reports its own corpus in the bench
            below, and a "0 tables" pill beside a page promising live
            analysis reads as an empty page. */}
          {mode === "results" && (
            <span className="corpus-pill">
              <TableProperties size={16} />
              {results.length} table{results.length === 1 ? "" : "s"} to explore
            </span>
          )}
        </div>
      )}

      {!lockedMode && (
        <div
          className="explore-modes"
          role="tablist"
          aria-label="What to explore"
        >
          <button
            role="tab"
            aria-selected={mode === "live"}
            className={mode === "live" ? "viz-kind selected" : "viz-kind"}
            onClick={() => setMode("live")}
          >
            Run an analysis live
          </button>
          <button
            role="tab"
            aria-selected={mode === "results"}
            className={mode === "results" ? "viz-kind selected" : "viz-kind"}
            onClick={() => setMode("results")}
          >
            Chart a finished result
          </button>
        </div>
      )}

      {mode === "live" && (
        <p className="mode-note muted">
          Two ways to question the loaded documents, kept apart on purpose:{" "}
          <strong>Track a phrase</strong> follows one phrase through time,
          documents, positions and passages, and its saved questions are
          published as evidence-bearing runs. <strong>Analysis</strong> below
          runs any of the engine's analyses over the same loaded corpus and
          charts the result.
        </p>
      )}

      {mode === "live" && liveBench}

      {mode === "results" && loading && (
        <p role="status">Looking for tables to explore…</p>
      )}

      {mode === "results" && !loading && !sources.length && (
        <div className="sample-callout">
          <FlaskConical size={22} />
          <div>
            <strong>Nothing to explore yet.</strong>
            <p>
              This page charts tables that runs have already published. Run an
              analysis — readability, lexical diversity, word frequencies — and
              its table appears here the moment it finishes.
            </p>
          </div>
          <button className="secondary" onClick={onBrowseAnalyses}>
            Browse analyses
          </button>
        </div>
      )}

      {mode === "results" && !loading && sources.length > 0 && (
        <>
          <label className="explore-picker">
            <TableProperties size={16} />
            <span>Table</span>
            <select
              aria-label="Table to explore"
              value={selected ? sourceKey(selected) : ""}
              onChange={(event) => {
                const match = sources.find(
                  (source) => sourceKey(source) === event.target.value,
                );
                // A view belongs to one table. Choosing another table by hand
                // closes it rather than quietly repointing it at rows it was
                // never arranged over.
                if (match) {
                  setOpenId(null);
                  onSelect(match);
                }
              }}
            >
              {results.length > 0 && (
                <optgroup label="Results">
                  {results.map((source) => (
                    <option key={sourceKey(source)} value={sourceKey(source)}>
                      {sourceLabel(source)}
                    </option>
                  ))}
                </optgroup>
              )}
              {manifests.length > 0 && (
                <optgroup label="What each run read">
                  {manifests.map((source) => (
                    <option key={sourceKey(source)} value={sourceKey(source)}>
                      {sourceLabel(source)}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </label>

          {views.length > 0 && (
            <label className="explore-picker">
              <Bookmark size={16} />
              <span>Saved view</span>
              <select
                aria-label="Saved view to open"
                value={openId ?? ""}
                onChange={(event) => {
                  const match = views.find(
                    (view) => view.id === event.target.value,
                  );
                  if (match) reopen(match);
                  else setOpenId(null);
                }}
              >
                <option value="">
                  {here.length
                    ? "Start from scratch"
                    : "Start from scratch — no saved views of this table"}
                </option>
                {here.length > 0 && (
                  <optgroup label="Of this table">
                    {here.map((view) => (
                      <option key={view.id} value={view.id}>
                        {viewLabel(view)}
                      </option>
                    ))}
                  </optgroup>
                )}
                {views.length > here.length && (
                  <optgroup label="Of other tables">
                    {views
                      .filter((view) => !here.includes(view))
                      .map((view) => (
                        <option key={view.id} value={view.id}>
                          {viewLabel(view)} · {view.artifact_path}
                        </option>
                      ))}
                  </optgroup>
                )}
              </select>
            </label>
          )}

          {open && open.source.state !== "ok" && (
            <p className="alert warning" role="status">
              <TriangleAlert size={16} aria-hidden="true" />
              <span>{open.source.detail}</span>
            </p>
          )}

          {(tableLoading || !settled) && (
            <p role="status">
              {settled ? "Loading that table…" : `Opening “${open?.name}”…`}
            </p>
          )}

          {!tableLoading && settled && table && selected && settings && (
            <>
              {open ? (
                <ViewBar
                  key={open.id}
                  view={open}
                  dirty={dirty}
                  onSave={() =>
                    void onUpdateView(open, selected, settings, filters)
                  }
                  onRevert={revert}
                  onDuplicate={() =>
                    void onDuplicateView(open).then(
                      (copy) => copy && setOpenId(copy.id),
                    )
                  }
                  onDelete={() =>
                    void onDeleteView(open).then(
                      (gone) => gone && setOpenId(null),
                    )
                  }
                />
              ) : (
                <SaveBar
                  taken={views.map((item) => item.name)}
                  onSave={(chosen) =>
                    void onSaveView(chosen, selected, settings, filters).then(
                      (saved) => saved && setOpenId(saved.id),
                    )
                  }
                />
              )}

              {projectId && !selected.manifest ? (
                <PanelSection
                  key={`${projectId}:${sourceKey(selected)}`}
                  projectId={projectId}
                  jobId={selected.job}
                  selectedColumns={table.columns}
                  fallback={
                    <Workbench
                      table={table}
                      artifactPath={selected.path}
                      settings={settings}
                      onSettings={setSettings}
                      selected={mark}
                      onSelect={setMark}
                      canPublish={canPublish}
                      onPublish={() => onPublish(selected, settings)}
                      gaps={gaps}
                      headingLevel={embedded ? 3 : 2}
                    />
                  }
                />
              ) : (
                <Workbench
                  key={sourceKey(selected)}
                  table={table}
                  artifactPath={selected.path}
                  settings={settings}
                  onSettings={setSettings}
                  selected={mark}
                  onSelect={setMark}
                  canPublish={canPublish}
                  onPublish={() => onPublish(selected, settings)}
                  gaps={gaps}
                  headingLevel={embedded ? 3 : 2}
                />
              )}
              {onCompareSource && candidates.length > 0 && (
                <section
                  className="panel"
                  aria-label="Compare finished results"
                >
                  <label className="explore-picker">
                    <GitCompareArrows size={16} />
                    <span>Compare with</span>
                    <select
                      aria-label="Table to compare"
                      value={compareSource ? sourceKey(compareSource) : ""}
                      onChange={(event) => {
                        setComparisonMark(null);
                        onCompareSource(
                          candidates.find(
                            (candidate) =>
                              sourceKey(candidate) === event.target.value,
                          ) ?? null,
                        );
                      }}
                    >
                      <option value="">No comparison</option>
                      {candidates.map((candidate) => (
                        <option
                          key={sourceKey(candidate)}
                          value={sourceKey(candidate)}
                        >
                          {sourceLabel(candidate)}
                        </option>
                      ))}
                    </select>
                  </label>
                  {compareSource && <p>{sourceLabel(compareSource)}</p>}
                  {compareSource && compareLoading && (
                    <p role="status">Loading comparison…</p>
                  )}
                  {compareSource && !compareLoading && !compare && (
                    <p role="alert">
                      The comparison table could not be read. Choose another
                      table or try again.
                    </p>
                  )}
                  {compareSource &&
                    !compareLoading &&
                    compare &&
                    missing.length > 0 && (
                      <p role="status">
                        This table is missing columns used by the current chart:{" "}
                        {missing.join(", ")}. Choose another table or change the
                        chart columns above.
                      </p>
                    )}
                  {compareSource &&
                    !compareLoading &&
                    compare &&
                    !missing.length && (
                      <>
                        <p className="muted">
                          Both charts use the same settings. Each chart fits its
                          own axes and uses its own loaded rows; compare axis
                          values, not bar heights. Saved views above belong to
                          the first table.
                        </p>
                        <Workbench
                          key={sourceKey(compareSource)}
                          table={compare}
                          artifactPath={compareSource.path}
                          settings={settings}
                          onSettings={(next) => {
                            setSettings(next);
                            setComparisonMark(null);
                          }}
                          selected={comparisonMark}
                          onSelect={setComparisonMark}
                          canPublish={canPublish}
                          onPublish={() => onPublish(compareSource, settings)}
                          gaps={gaps}
                          headingLevel={3}
                        />
                      </>
                    )}
                </section>
              )}
            </>
          )}

          {!tableLoading && settled && !table && selected && (
            <p className="alert warning" role="status">
              <span>
                {selected.path} could not be read. Its run directory may have
                been moved or deleted.
              </span>
            </p>
          )}
        </>
      )}
    </>
  );
}
