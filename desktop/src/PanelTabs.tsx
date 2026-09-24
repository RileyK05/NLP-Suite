import { useEffect, useRef, useState } from "react";
import type { FigureFormat, FigureImage, PanelAnswer, PanelInfo } from "./api";
import type { PanelSelection } from "./PanelCanvas";
import {
  PanelAnswerView,
  PanelParamField,
  panelDefaults,
} from "./PanelSection";

/**
 * A tool's figures as tabs, each with the controls its declaration names.
 *
 * The live bench used to draw only the first figure a tool had, with its
 * defaults and no way to change them — so an LDA answer showed topic-term
 * bars and nothing else, and a figure needing a word to look up could not be
 * given one. This is the same arrangement a finished run offers, fed by
 * whatever `draw` the caller has: a finished run draws from its artifact, a
 * live answer from the rows the server kept for it.
 *
 * Answers are dropped unless they still belong to the figure and the answer
 * on screen: switching tabs, or pressing Run again, while a draw is out must
 * not let the old figure land under the new heading.
 */
export function PanelTabs({
  panels,
  first,
  draw,
  projectId,
  resetKey,
  requested,
  publish,
}: {
  panels: PanelInfo[];
  /** The first figure, already drawn with its defaults by the caller. */
  first?: PanelAnswer | null;
  draw: (
    panel: PanelInfo,
    params: Record<string, unknown>,
  ) => Promise<PanelAnswer>;
  /** The project whose documents a selected mark's passages come from. */
  projectId: string;
  /** Changes when the underlying answer changes; everything resets. */
  resetKey: string;
  /** A figure the reader asked for by its question, elsewhere on the page. */
  requested?: { panel: string; nonce: number } | null;
  /** Draws a figure for publication (matplotlib + seaborn). */
  publish?: (
    panel: PanelInfo,
    params: Record<string, unknown>,
    format: FigureFormat,
    dpi: number,
  ) => Promise<FigureImage>;
}) {
  const [active, setActive] = useState<PanelInfo | null>(panels[0] ?? null);
  const [params, setParams] = useState<Record<string, unknown>>(
    panels[0] ? panelDefaults(panels[0]) : {},
  );
  const [answer, setAnswer] = useState<PanelAnswer | null>(first ?? null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<PanelSelection>(null);
  // The parameters last drawn, so the publication view matches the screen.
  const [drawnParams, setDrawnParams] = useState<Record<string, unknown>>(
    panels[0] ? panelDefaults(panels[0]) : {},
  );
  const sequence = useRef(0);

  useEffect(() => {
    sequence.current += 1;
    setActive(panels[0] ?? null);
    setParams(panels[0] ? panelDefaults(panels[0]) : {});
    setAnswer(first ?? null);
    setSelected(null);
    setBusy(false);
    setDrawnParams(panels[0] ? panelDefaults(panels[0]) : {});
    // A new answer is a new set of figures; the old draw belongs to rows
    // that are no longer on screen.
  }, [resetKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const show = async (panel: PanelInfo, values: Record<string, unknown>) => {
    const mine = ++sequence.current;
    setActive(panel);
    setBusy(true);
    setAnswer(null);
    setSelected(null);
    setDrawnParams(values);
    try {
      const result = await draw(panel, values);
      if (sequence.current === mine) setAnswer(result);
    } catch (error) {
      if (sequence.current === mine) {
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
      }
    } finally {
      if (sequence.current === mine) setBusy(false);
    }
  };

  // A question in the reading names a figure: open its tab.
  useEffect(() => {
    if (!requested) return;
    const panel = panels.find(
      (candidate) => candidate.name === requested.panel,
    );
    if (!panel) return;
    const defaults = panelDefaults(panel);
    setParams(defaults);
    void show(panel, defaults);
  }, [requested?.nonce]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!active) return null;
  return (
    <>
      {panels.length > 1 && (
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
              title={panel.question || panel.summary}
              onClick={() => {
                const defaults = panelDefaults(panel);
                setParams(defaults);
                void show(panel, defaults);
              }}
            >
              {panel.title}
            </button>
          ))}
        </div>
      )}
      {active.question && <p className="panel-question">{active.question}</p>}
      {active.params.length > 0 && (
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
            onClick={() => void show(active, params)}
          >
            {busy ? "Drawing…" : "Draw figure"}
          </button>
        </div>
      )}
      <p className="panel-summary">{active.summary}</p>
      {busy && !answer && <p className="muted">Drawing…</p>}
      {answer && (
        <PanelAnswerView
          projectId={projectId}
          answer={answer}
          selected={selected}
          onSelect={setSelected}
          publish={
            publish
              ? (format, dpi) => publish(active, drawnParams, format, dpi)
              : undefined
          }
          drawKey={`${resetKey}:${active.name}:${JSON.stringify(drawnParams)}`}
        />
      )}
    </>
  );
}
