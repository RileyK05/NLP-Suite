import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, Search } from "lucide-react";
import { api, title, type PanelInfo, type Tool } from "./api";
import { ChartCanvas, type Selection } from "./ChartCanvas";
import { CHART_EXAMPLES, CHART_TASKS } from "./chartExamples";
import {
  buildLayout,
  rowsBehind,
  type LiveKind,
  type Settings,
} from "./chartLayout";
import { ResultTable } from "./ResultTable";
import { CHART_KINDS } from "./VisualizeDialog";

const SHAPE_LABELS: Record<string, string> = {
  scatter_labelled: "Labelled points",
  ranked_bars: "Ranked bars",
  stream: "Stream",
  line_series: "Trend lines",
  ribbon: "Ribbon",
  small_multiples: "Small multiples",
  heatmap: "Heatmap",
  distribution: "Distribution",
  network: "Network",
  positions: "Positions",
  geographic: "Geographic map",
};

const shapeLabel = (shape: string) =>
  SHAPE_LABELS[shape] ?? shape.replaceAll("_", " ");

function ChartGallery() {
  const [kind, setKind] = useState<LiveKind>("bar");
  const [selected, setSelected] = useState<Selection>(null);
  const sample = CHART_EXAMPLES[kind];
  const settings = useMemo<Settings>(
    () => ({
      ...sample.settings,
      kind,
      agg: "",
      bins: 8,
      topN: 0,
      sort: kind === "bar" ? "high" : "table",
    }),
    [kind, sample],
  );
  const result = useMemo(
    () => buildLayout(sample.rows, settings),
    [sample, settings],
  );
  const columns = Object.keys(sample.rows[0]);
  const evidence = selected
    ? rowsBehind(sample.rows, settings, selected)
    : null;
  const chooseChart = (next: LiveKind) => {
    setKind(next);
    setSelected(null);
  };

  return (
    <div className="viz-studio">
      <label className="viz-compact-picker">
        Chart type
        <select
          value={kind}
          onChange={(event) => chooseChart(event.target.value as LiveKind)}
        >
          {CHART_TASKS.map((task) => (
            <optgroup key={task.label} label={task.label}>
              {task.kinds.map((id) => (
                <option key={id} value={id}>
                  {CHART_KINDS.find((entry) => entry.kind === id)?.label}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      <div role="group" aria-label="Chart types" className="viz-picker">
        {CHART_TASKS.map((task) => (
          <div key={task.label} className="viz-task">
            <h2>{task.label}</h2>
            {task.kinds.map((id) => {
              const chart = CHART_KINDS.find((entry) => entry.kind === id)!;
              return (
                <button
                  key={id}
                  type="button"
                  aria-pressed={kind === id}
                  className="viz-picker-button"
                  onClick={() => chooseChart(id)}
                >
                  <strong>{chart.label}</strong>
                  <span>{chart.hint}</span>
                </button>
              );
            })}
          </div>
        ))}
      </div>
      <div className="viz-stage">
        {result.ok ? (
          <ChartCanvas
            key={kind}
            layout={result.layout}
            title={sample.title}
            subtitle={`Example data · ${sample.rows.length} rows`}
            selected={selected}
            onSelect={setSelected}
            name={`nlp-suite-example-${kind}`}
          />
        ) : (
          <p role="status">{result.reason}</p>
        )}
        {evidence ? (
          <section className="viz-source">
            <div className="viz-evidence-head">
              <h3>
                {evidence.length} {evidence.length === 1 ? "row" : "rows"}{" "}
                behind the selection
              </h3>
              <button
                type="button"
                className="text-button"
                onClick={() => setSelected(null)}
              >
                Clear selection
              </button>
            </div>
            <ResultTable columns={columns} rows={evidence} copyable />
          </section>
        ) : (
          <details key={`data-${kind}`} className="viz-source">
            <summary>View example data</summary>
            <ResultTable columns={columns} rows={sample.rows} copyable />
          </details>
        )}
      </div>
    </div>
  );
}

function FigureDirectory({
  panels,
  loading,
  retry,
  tools,
  onAnalysis,
}: {
  panels: PanelInfo[] | null;
  loading: boolean;
  retry: () => void;
  tools: Tool[];
  onAnalysis?: (tool: Tool) => void;
}) {
  const [query, setQuery] = useState("");
  const [shape, setShape] = useState("");
  const [analysis, setAnalysis] = useState("");
  if (loading) return <p role="status">Loading analysis figures…</p>;
  if (!panels)
    return (
      <div className="viz-empty">
        <p role="status">The analysis figure list could not be loaded.</p>
        <button type="button" className="secondary" onClick={retry}>
          Try again
        </button>
      </div>
    );

  const toolsById = new Map(tools.map((tool) => [tool.name, tool]));
  const toolLabel = (id: string) => toolsById.get(id)?.label ?? title(id);
  const analysisIds = [...new Set(panels.map((panel) => panel.tool))].sort(
    (a, b) => toolLabel(a).localeCompare(toolLabel(b)),
  );
  const matchingAnalysis = analysis
    ? panels.filter((panel) => panel.tool === analysis)
    : panels;
  const shapes = [
    ...new Set(matchingAnalysis.map((panel) => panel.shape)),
  ].sort((a, b) => shapeLabel(a).localeCompare(shapeLabel(b)));
  const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  const shown = matchingAnalysis.filter((panel) => {
    if (shape && panel.shape !== shape) return false;
    const tool = toolsById.get(panel.tool);
    const fields = [
      panel.title,
      panel.question,
      panel.summary,
      panel.tool.replaceAll("_", " "),
      tool?.label,
      shapeLabel(panel.shape),
    ]
      .join(" ")
      .toLocaleLowerCase();
    return terms.every((term) => fields.includes(term));
  });

  return (
    <>
      <div className="viz-directory-controls">
        <label>
          Search analysis figures
          <span className="viz-search">
            <Search size={16} aria-hidden="true" />
            <input
              type="search"
              value={query}
              placeholder="e.g. topic, sentiment, network"
              onChange={(event) => setQuery(event.target.value)}
            />
          </span>
        </label>
        <label>
          Analysis tool
          <select
            value={analysis}
            onChange={(event) => {
              setAnalysis(event.target.value);
              setShape("");
            }}
          >
            <option value="">All analyses</option>
            {analysisIds.map((id) => (
              <option key={id} value={id}>
                {toolLabel(id)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Figure shape
          <select
            value={shape}
            onChange={(event) => setShape(event.target.value)}
          >
            <option value="">All shapes</option>
            {shapes.map((key) => (
              <option key={key} value={key}>
                {shapeLabel(key)}
              </option>
            ))}
          </select>
        </label>
        <p role="status">
          {shown.length} of {matchingAnalysis.length} figures
        </p>
      </div>
      {shown.length ? (
        <ul className="viz-figure-list">
          {shown.map((panel) => {
            const tool = toolsById.get(panel.tool);
            return (
              <li key={panel.name}>
                <div className="viz-figure-content">
                  <p className="viz-figure-shape">{shapeLabel(panel.shape)}</p>
                  <h3>{panel.title}</h3>
                  <p>{panel.question || panel.summary}</p>
                </div>
                {tool && onAnalysis ? (
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => onAnalysis(tool)}
                  >
                    {tool.label} <ArrowUpRight size={14} aria-hidden="true" />
                  </button>
                ) : (
                  <p className="viz-figure-tool">{title(panel.tool)}</p>
                )}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="viz-empty">No figures match these filters.</p>
      )}
    </>
  );
}

export function Visualizations({
  onExplore,
  onRuns,
  tools = [],
  onAnalysis,
}: {
  onExplore: () => void;
  onRuns: () => void;
  tools?: Tool[];
  onAnalysis?: (tool: Tool) => void;
}) {
  const [panels, setPanels] = useState<PanelInfo[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const [view, setView] = useState<"charts" | "figures">("charts");

  useEffect(() => {
    let alive = true;
    api<PanelInfo[]>("/panels")
      .then((list) => {
        if (alive) setPanels(list);
      })
      .catch(() => {
        if (alive) setPanels(null);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [attempt]);

  return (
    <section className="panel visualizations" aria-label="Visualizations">
      <header className="viz-hero">
        <div>
          <h1>Visualizations</h1>
          <p>
            {view === "charts"
              ? "Try charts with example data, then open Interactive to explore your corpus."
              : "Find figures that answer a corpus question and open the analysis that produces them."}
          </p>
        </div>
        {view === "charts" ? (
          <button type="button" className="secondary" onClick={onExplore}>
            Open Interactive <ArrowUpRight size={16} aria-hidden="true" />
          </button>
        ) : (
          <button type="button" className="secondary" onClick={onRuns}>
            Open Past runs <ArrowUpRight size={16} aria-hidden="true" />
          </button>
        )}
      </header>
      <div
        className="viz-filters"
        role="group"
        aria-label="Visualization collection"
      >
        <button
          type="button"
          aria-pressed={view === "charts"}
          onClick={() => setView("charts")}
        >
          Chart examples <span>{CHART_KINDS.length}</span>
        </button>
        <button
          type="button"
          aria-pressed={view === "figures"}
          onClick={() => setView("figures")}
        >
          Analysis figures {panels && <span>{panels.length}</span>}
        </button>
      </div>
      {view === "charts" ? (
        <ChartGallery />
      ) : (
        <FigureDirectory
          panels={panels}
          loading={loading}
          retry={() => {
            setLoading(true);
            setAttempt(attempt + 1);
          }}
          tools={tools}
          onAnalysis={onAnalysis}
        />
      )}
    </section>
  );
}
