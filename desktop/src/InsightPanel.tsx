import { useEffect, useState } from "react";
import {
  BadgeCheck,
  BarChart3,
  Lightbulb,
  Sparkles,
  Table2,
  TriangleAlert,
} from "lucide-react";
import { api } from "./api";
import type { Insight, Reading, RecommendedChart } from "./api";

/**
 * What a finished run actually says, and which charts are worth drawing.
 *
 * The reading is produced by the Python engine from the published CSV, not
 * recomputed here, so what the desktop shows is a reading of the same artifact
 * the user can open and export.
 *
 * Cautions are shown before recommendations on purpose: knowing that a result
 * has tripped a known trap changes whether any chart of it is worth making.
 */
export function InsightPanel({
  projectId,
  jobId,
  onChart,
  onFigure,
}: {
  projectId: string;
  jobId: string;
  onChart?: (kind: string, x: string, y: string) => void;
  /** Open one of the run's own figures, by panel name. */
  onFigure?: (panel: string) => void;
}) {
  const [insight, setInsight] = useState<Insight | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    setInsight(null);
    setFailed(false);
    api<Insight>(`/projects/${projectId}/jobs/${jobId}/insight`)
      .then((value) => {
        if (alive) setInsight(value);
      })
      .catch(() => {
        // A reading is a convenience on top of the result. If it cannot be
        // produced the run is still complete and its artifacts still open.
        if (alive) setFailed(true);
      });
    return () => {
      alive = false;
    };
  }, [projectId, jobId]);

  if (failed || !insight) return null;
  if (!insight.available) return null;
  return (
    <ReadingPanel
      reading={insight}
      note={insight.table ? `Reading the table ${insight.table}.` : ""}
      onChart={
        onChart ? (chart) => onChart(chart.kind, chart.x, chart.y) : undefined
      }
      onFigure={onFigure}
    />
  );
}

/**
 * A reading, rendered from a reading — with no idea where it came from.
 *
 * Split out because the live bench needs exactly this and had nothing: it
 * drew a chart and said how many rows were in it, so a sentiment analysis
 * over 87 speeches arrived as 87 bars and a row count. The engine had been
 * able to read that table in words the whole time; only the finished-run
 * pane ever asked.
 */
export function ReadingPanel({
  reading,
  heading = "What this result says",
  note = "",
  onChart,
  onFigure,
}: {
  reading: Reading;
  heading?: string;
  /** One line about where these rows are from, when that is not obvious. */
  note?: string;
  /** Draw this recommendation. Absent, the recommendations are read-only. */
  onChart?: (chart: RecommendedChart) => void;
  /** Open one of the tool's own figures, by panel name. */
  onFigure?: (panel: string) => void;
}) {
  const charts = reading.recommended_charts ?? [];
  const figures = reading.figures ?? [];
  const tableFirst = reading.table_first ?? "";
  const observations = reading.observations ?? [];
  const cautions = reading.cautions ?? [];
  if (
    !reading.headline &&
    !observations.length &&
    !cautions.length &&
    !charts.length &&
    !figures.length &&
    !tableFirst
  ) {
    return null;
  }
  return (
    <section className="insight-panel" aria-label={heading}>
      <h3>
        <Lightbulb size={16} aria-hidden="true" /> {heading}
      </h3>
      {reading.headline && (
        <p className="insight-headline">{reading.headline}</p>
      )}
      {note && <p className="insight-note">{note}</p>}

      {observations.length > 0 && (
        <ul className="insight-list">
          {observations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}

      {cautions.length > 0 && (
        <div className="insight-cautions">
          <h4>
            <TriangleAlert size={15} aria-hidden="true" /> Read with care
          </h4>
          <ul className="insight-list">
            {cautions.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {tableFirst && (
        <p className="insight-note">
          <Table2 size={14} aria-hidden="true" /> {tableFirst}
        </p>
      )}

      {figures.length > 0 && (
        <div className="insight-charts">
          <h4>
            <Sparkles size={15} aria-hidden="true" /> Questions this result
            answers
          </h4>
          {figures.map((figure) => (
            <div className="insight-chart" key={figure.panel}>
              <strong>{figure.question || figure.title}</strong>
              {onFigure && (
                <button
                  type="button"
                  className="secondary"
                  onClick={() => onFigure(figure.panel)}
                >
                  Show {figure.title}
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {charts.length > 0 && (
        <div className="insight-charts">
          <h4>
            <BarChart3 size={15} aria-hidden="true" /> Questions worth asking of
            this table
          </h4>
          <p className="insight-note">
            <BadgeCheck size={14} aria-hidden="true" /> Pairings that could only
            restate the data — a column against itself, or two columns that are
            the same quantity — are left out rather than offered.
          </p>
          {charts.map((chart) => (
            <div
              className="insight-chart"
              key={`${chart.kind}:${chart.x}:${chart.y}`}
            >
              <strong>{chart.question}</strong>
              <span>{chart.why}</span>
              {onChart && (
                <button
                  type="button"
                  className="secondary"
                  onClick={() => onChart(chart)}
                >
                  {chart.kind === "histogram"
                    ? `Draw histogram of ${chart.x}`
                    : `Draw ${chart.kind}: ${chart.y} by ${chart.x}`}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
