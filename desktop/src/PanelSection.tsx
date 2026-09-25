import { useEffect, useRef, useState, type ReactNode } from "react";
import { Sparkles } from "lucide-react";
import { api, post, publicationFigure } from "./api";
import type {
  Diagnostic,
  PanelAnswer,
  PanelInfo,
  Param,
  PreparedPanel,
} from "./api";
import { Diagnostics } from "./RunStatus";
import { PanelCanvas, type PanelSelection } from "./PanelCanvas";
import { BundleGallery } from "./BundleGallery";
import { PanelPassages } from "./PanelPassages";
import { PublicationFigure, type PublishFigure } from "./PublicationFigure";
import { ResultTable } from "./ResultTable";
import { rowsBehind } from "./panelLayout";

/**
 * Purpose-built panels over one finished run, offered where the run is read.
 *
 * The controls are built from the panel's own declaration (`PanelInfo.params`
 * — name, type, default, choices, bounds, help), never hard-coded per panel:
 * that is the whole point of shipping declarations, and the reason
 * `/api/panels` exists. A new panel needs no edit here.
 *
 * The section exists only when the run's tool has panels to offer. A run with
 * no panels shows nothing rather than an empty heading — the same lesson the
 * `ARCHIVE_PAGE` bug taught: a permanently-empty section is worse than no
 * section, because it looks broken without saying why.
 *
 * Failures come back as `ok: false` plus diagnostics, and render the way the
 * rest of the app renders them — never as a crash, never silently.
 */
export function PanelSection({
  projectId,
  jobId,
  fallback,
  selectedColumns,
  requested,
}: {
  projectId: string;
  jobId: string;
  /** Shown only after the server confirms this run has no declared panels. */
  fallback?: ReactNode;
  /** Prefer this artifact's figure when one run publishes several tables. */
  selectedColumns?: string[];
  /** A figure the reader asked for by its question in the run's reading. */
  requested?: { panel: string; nonce: number } | null;
}) {
  const [panels, setPanels] = useState<PanelInfo[] | null>(null);
  const [checked, setChecked] = useState(false);
  const [active, setActive] = useState<PanelInfo | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [answer, setAnswer] = useState<PanelAnswer | null>(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<PanelSelection>(null);
  const [showGeneric, setShowGeneric] = useState(false);
  // What was last *drawn* -- not the controls' unapplied edits -- so the
  // publication figure is always the figure on screen.
  const [drawn, setDrawn] = useState<{
    panel: string;
    params: Record<string, unknown>;
  } | null>(null);

  // The run on screen *now*, updated during render rather than in an effect:
  // a draw's answer can resolve in the gap between a re-render and its
  // effects, and it must be compared against the run actually showing. This
  // component is not remounted when the reader switches runs, so without it
  // an answer for the previous run renders under the new run's heading.
  const run = `${projectId}/${jobId}`;
  const showing = useRef(run);
  showing.current = run;
  const drawSequence = useRef(0);

  /** Draws one figure; true when it drew (false when refused, failed or superseded). */
  const draw = async (
    panel: PanelInfo,
    values: Record<string, unknown>,
  ): Promise<boolean> => {
    const mine = run;
    const sequence = ++drawSequence.current;
    setBusy(true);
    setAnswer(null);
    setSelected(null);
    setDrawn({ panel: panel.name, params: values });
    try {
      const result = await post<PanelAnswer>(
        `/projects/${projectId}/jobs/${jobId}/panels`,
        { panel: panel.name, params: values },
      );
      if (showing.current !== mine || drawSequence.current !== sequence)
        return false;
      setAnswer(result);
      return result.ok;
    } catch (error) {
      if (showing.current !== mine || drawSequence.current !== sequence)
        return false;
      setAnswer({
        ok: false,
        diagnostics: [
          {
            severity: "ERROR",
            code: "PANEL_REQUEST_FAILED",
            message: error instanceof Error ? error.message : String(error),
          },
        ],
      });
      return false;
    } finally {
      if (showing.current === mine && drawSequence.current === sequence)
        setBusy(false);
    }
  };

  useEffect(() => {
    let alive = true;
    drawSequence.current += 1;
    setPanels(null);
    setChecked(false);
    setActive(null);
    setAnswer(null);
    setSelected(null);
    setShowGeneric(false);
    // A draw still out for the previous run will never clear this itself --
    // its answer is dropped below -- so the new run starts ready, not stuck
    // on "Drawing…" (the same stranded-flag shape as the live bench's old bug).
    setBusy(false);
    api<PanelInfo[]>(`/projects/${projectId}/jobs/${jobId}/panels`)
      .then((list) => {
        if (!alive) return;
        setChecked(true);
        // A run with no panels shows no section at all, not an empty one.
        if (!list.length) return;
        const matching = list.filter((panel) =>
          panel.requires.every((column) => selectedColumns?.includes(column)),
        );
        const ordered = selectedColumns?.length
          ? [...matching, ...list.filter((panel) => !matching.includes(panel))]
          : list;
        setPanels(ordered);
        // The first figure can refuse a run another reads well (eight
        // meaning groups need sixteen words): open the first that draws, and
        // show the first one's reasons only when every figure refuses.
        void (async () => {
          for (const panel of ordered) {
            if (!alive) return;
            setActive(panel);
            setParams(panelDefaults(panel));
            if (await draw(panel, panelDefaults(panel))) return;
          }
          if (!alive || ordered.length < 2) return;
          setActive(ordered[0]);
          setParams(panelDefaults(ordered[0]));
          void draw(ordered[0], panelDefaults(ordered[0]));
        })();
      })
      .catch(() => {
        // Panels are an offer on top of a finished run; if they cannot be
        // listed, the run record itself still stands.
        if (alive) setChecked(true);
      });
    return () => {
      alive = false;
    };
  }, [projectId, jobId, selectedColumns?.join("\u0000")]);

  // A question in the reading names a figure: open its tab and draw it.
  useEffect(() => {
    if (!requested || !panels) return;
    const panel = panels.find(
      (candidate) => candidate.name === requested.panel,
    );
    if (!panel) return;
    setActive(panel);
    setParams(panelDefaults(panel));
    void draw(panel, panelDefaults(panel));
  }, [requested?.nonce]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!panels || !active) return checked ? <>{fallback}</> : null;

  return (
    <>
      <section
        className="panel panel-section"
        aria-label="Figures for this run"
      >
        <div className="section-title">
          <div>
            <h2>
              <Sparkles size={16} aria-hidden="true" /> Purpose-built figures
            </h2>
            <p>
              Drawn from this run's own result shape, with every mark clickable.
            </p>
          </div>
        </div>
        {fallback && (
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
        )}
        <div
          className="panel-picker"
          role="tablist"
          aria-label="Choose a figure"
        >
          {panels.map((panel) => (
            <button
              key={panel.name}
              type="button"
              role="tab"
              aria-selected={panel.name === active.name}
              className={
                panel.name === active.name ? "viz-kind selected" : "viz-kind"
              }
              title={panel.summary}
              onClick={() => {
                setActive(panel);
                setParams(panelDefaults(panel));
                void draw(panel, panelDefaults(panel));
              }}
            >
              {panel.title}
            </button>
          ))}
        </div>
        <div className="panel-controls">
          {active.params.map((param) => (
            <PanelParamField
              key={param.name}
              param={param}
              value={params[param.name]}
              onChange={(value) =>
                setParams((current) => ({ ...current, [param.name]: value }))
              }
            />
          ))}
          <button
            type="button"
            className="secondary panel-draw"
            disabled={busy}
            onClick={() => void draw(active, params)}
          >
            {busy ? "Drawing…" : "Draw figure"}
          </button>
        </div>
        {active.question && <p className="panel-question">{active.question}</p>}
        <p className="panel-summary">{active.summary}</p>
        {answer && (
          <PanelAnswerView
            projectId={projectId}
            answer={answer}
            selected={selected}
            onSelect={setSelected}
            publish={
              drawn
                ? (format, dpi) =>
                    publicationFigure(
                      `/projects/${projectId}/jobs/${jobId}/panels/static`,
                      { panel: drawn.panel, params: drawn.params, format, dpi },
                    )
                : undefined
            }
            drawKey={`${run}:${JSON.stringify(drawn)}`}
          />
        )}
      </section>
      <BundleGallery projectId={projectId} jobId={jobId} />
      {showGeneric && fallback}
    </>
  );
}

/** The defaults the engine declares, as a starting set of parameters. */
export function panelDefaults(panel: PanelInfo): Record<string, unknown> {
  return Object.fromEntries(
    panel.params.map((param) => [param.name, param.default]),
  );
}

/** One declared parameter as a control. Labels, choices, bounds and help all
 *  come from the engine — the desktop invents none of them. */
export function PanelParamField({
  param,
  value,
  onChange,
}: {
  param: Param;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const NUMERIC = param.type === "int" || param.type === "float";
  if (param.type === "bool") {
    return (
      <label className="field-label">
        {param.label}
        <input
          type="checkbox"
          checked={!!value}
          onChange={(event) => onChange(event.target.checked)}
        />
        <small>{param.help}</small>
      </label>
    );
  }
  if (param.type === "choice") {
    return (
      <label className="field-label">
        {param.label}
        <select
          value={String(value ?? "")}
          onChange={(event) => onChange(event.target.value)}
        >
          {param.choices.map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
        <small>{param.help}</small>
      </label>
    );
  }
  return (
    <label className="field-label">
      {param.label}
      <input
        type={NUMERIC ? "number" : "text"}
        step={param.type === "int" ? "1" : "any"}
        min={param.minimum ?? undefined}
        max={param.maximum ?? undefined}
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(event) =>
          onChange(
            NUMERIC && event.target.value !== ""
              ? Number(event.target.value)
              : event.target.value,
          )
        }
      />
      <small>{param.help}</small>
    </label>
  );
}

/**
 * A panel answer, whichever way it went.
 *
 * A refusal renders its diagnostics exactly as a run's are rendered. A
 * success renders the figure, then the rows behind a clicked mark.
 */
export function PanelAnswerView({
  projectId,
  answer,
  selected,
  onSelect,
  publish,
  drawKey = "",
}: {
  projectId: string;
  answer: PanelAnswer;
  selected: PanelSelection;
  onSelect: (selection: PanelSelection) => void;
  /** Draws this figure for publication; absent, only the interactive view. */
  publish?: PublishFigure;
  /** Identifies the drawn figure, so the publication view redraws with it. */
  drawKey?: string;
}) {
  const [view, setView] = useState<"interactive" | "publication">(
    "interactive",
  );
  if (!answer.ok) {
    return <Diagnostics items={answer.diagnostics as Diagnostic[]} />;
  }
  const prepared: PreparedPanel = answer;
  // A network edge is selectable like a mark (its link is the finding) and
  // carries its own evidence, so it opens its passages and rows the same way.
  const selectedMark = selected
    ? (prepared.marks.find((mark) => mark.key === selected.key) ??
      prepared.edges?.find((edge) => edge.key === selected.key) ??
      null)
    : null;
  const publication = publish && view === "publication";
  return (
    <>
      <Diagnostics items={prepared.diagnostics} />
      {publish && (
        <div className="figure-view" role="tablist" aria-label="Figure view">
          {(
            [
              ["interactive", "Interactive"],
              ["publication", "Publication figure"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={view === value}
              className={view === value ? "viz-kind selected" : "viz-kind"}
              onClick={() => setView(value)}
            >
              {label}
            </button>
          ))}
          <span className="muted">
            {publication
              ? "Drawn with matplotlib and seaborn: spread, violins, clustering and full labels, ready to save."
              : "Point at a mark to read it; click to filter the rows behind it."}
          </span>
        </div>
      )}
      {publication ? (
        <PublicationFigure
          publish={publish}
          drawKey={drawKey}
          fileBase={prepared.panel}
        />
      ) : (
        <PanelCanvas
          prepared={prepared}
          selected={selected}
          onSelect={onSelect}
        />
      )}
      {!publication && selectedMark && (
        <div className="panel-evidence">
          <p className="insight-note">{selectedMark.evidence.describe}</p>
          {/* Keyed by mark: choosing another point starts a fresh reading,
              and a search still out for the previous one is dropped with
              the component that asked for it. */}
          <PanelPassages
            key={selectedMark.key}
            projectId={projectId}
            evidence={selectedMark.evidence}
          />
          <ResultTable
            columns={prepared.table.columns}
            rows={rowsBehind(prepared.table, selectedMark.evidence)}
            footerNote={
              <span>
                {rowsBehind(prepared.table, selectedMark.evidence).length} of{" "}
                {prepared.table.rows.length} shown rows carry this mark's
                evidence (of {prepared.table.total} in the run).
              </span>
            }
          />
        </div>
      )}
    </>
  );
}
