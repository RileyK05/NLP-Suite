import {
  createContext,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
} from "react";
import {
  buildPanelLayout,
  groupColors,
  type PlacedEdge,
  type PlacedMark,
  type PanelLayout,
} from "./panelLayout";
import { formatNumber } from "./chartLayout";
import {
  axisTicks,
  fitText,
  labelMargin,
  logTicks,
  niceDomain,
  placeLabels,
  textWidth,
  tickLabels,
} from "./panelTicks";
import type { PanelAnnotation, PanelEvidence, PreparedPanel } from "./api";

/**
 * A prepared panel, drawn.
 *
 * Plain SVG, the same choice `ChartCanvas` makes, and for the same reason:
 * every mark has to stay clickable so pointing at it can say what it stands
 * for (`mark.evidence.describe`, written once engine-side) and filter the
 * table beneath the figure through `evidence.filters`. Nothing is computed
 * here — geometry comes from `panelLayout`, which is pure and tested — so
 * what is drawn and what is claimed cannot drift apart.
 *
 * Reading a panel is possible without a pointer, exactly as in `ChartCanvas`:
 * one tab stop per panel, arrows move between marks, Enter picks, and the
 * name a screen reader hears is the same sentence a hover shows.
 *
 * Reference lines carry their `note` visibly. A line at G2 = 3.84 means
 * nothing; "p < 0.05 at 1 degree of freedom" means something — that is the
 * whole reason `Annotation.note` exists, and a line that does not say it is
 * decoration.
 */

const DEFAULT_WIDTH = 900;
const DEFAULT_HEIGHT = 420;
/** The export's height includes its title, subtitle and caption margins;
 *  the app draws those as HTML around the SVG, so the SVG needs this much
 *  less for the same plot. It also maps the engine's default 500 onto the
 *  420 every existing panel was designed at. */
const EXPORT_CHROME = 80;
const MIN_HEIGHT = 200;
const PAD = { top: 22, right: 26, bottom: 66, left: 84 };
const AXIS = "#b3bdb2";
const GRID = "#e8ece5";
const TEXT = "#5d6a5c";
const DOT = "#8a938a";
const EDGE = "#9aa39a";
const MARK_RADIUS = 5;
const MAX_LABELS = 24;
/** Row-labelled shapes (heatmap, distribution, positions, ribbon, ranked
 *  bars) size their left margin to their labels (`labelMargin`), from this
 *  floor up to 38% of the width. */
const ROW_LABEL_FLOOR = 100;
/** The most a category label may take before it is cut with "…". */
const LABEL_MAX_PX = 300;
/** Below this many px per row or column, only every k-th label is drawn. */
const LABEL_PX = 11;
/** How far past the plot area a mark may be drawn: the largest marker's
 *  radius, so the clip never cuts a point in half. */
const CLIP_REACH = 14;
/** A non-negative series keeps zero on its axis when its lowest value is
 *  within this share of its highest (`core/viz/static/shapes._ZERO_NEAR`). */
const ZERO_NEAR = 0.35;
/** Shapes drawn as dots, where hovering finds the nearest point. */
const NEAREST_SHAPES = new Set([
  "scatter_labelled",
  "line_series",
  "small_multiples",
  "distribution",
  "network",
]);
/** How far (in drawing units) the pointer may be from a point to read it. */
const NEAREST_REACH = 22;
const TOOLTIP_WIDTH = 280;
/** Shapes a reader can zoom by dragging a box over the plot. */
const ZOOM_SHAPES = new Set(["scatter_labelled", "line_series"]);
/** A drag shorter than this (drawing units) is a click, not a zoom box. */
const MIN_DRAG = 8;

/** A zoomed view, as fractions of the unzoomed plot: x0..x1 left to right,
 *  y0..y1 bottom to top. Nested zooms compose, so each is a fraction of the
 *  view it was drawn on. */
export type Zoom = { x0: number; x1: number; y0: number; y1: number };

/** A scale narrowed to a zoom's share of it (no further padding). */
function zoomScale(
  scale: ReturnType<typeof linearScale>,
  from: number,
  to: number,
  plain = false,
) {
  const span = scale.max - scale.min;
  return linearScale(scale.min + from * span, scale.min + to * span, 0, 5, {
    plain,
  });
}

export type PanelSelection = { key: string } | null;

/** What the canvas can select, hover and describe: a mark, or a network
 *  edge, which carries its own evidence for the same reason a mark does. */
type Selectable = { key: string; evidence: PanelEvidence };

type Canvas = {
  WIDTH: number;
  HEIGHT: number;
  PLOT: { width: number; height: number };
};

/** The SVG's size: the builder's, when it asked (a heatmap of 87 documents
 *  needs more height than a ranking of twenty words), else the default. */
export function canvasSize(prepared: {
  width?: number;
  height?: number;
}): Canvas {
  const WIDTH = prepared.width ?? DEFAULT_WIDTH;
  const HEIGHT =
    prepared.height === undefined
      ? DEFAULT_HEIGHT
      : Math.max(MIN_HEIGHT, prepared.height - EXPORT_CHROME);
  return {
    WIDTH,
    HEIGHT,
    PLOT: {
      width: WIDTH - PAD.left - PAD.right,
      height: HEIGHT - PAD.top - PAD.bottom,
    },
  };
}

const CanvasSize = createContext<Canvas>(canvasSize({}));
const useCanvas = () => useContext(CanvasSize);

/**
 * Linear scale with padding at both ends, so a point at the maximum is drawn
 * whole rather than cut in half by the plot's edge (the 2018 point of the
 * per-million timeline was). `padFraction = 0` pins the ends (a stacked
 * stream is read against 0 and its total); `floor` keeps a zero baseline
 * when every value is at or above it, so bars and rates start at zero.
 */
function linearScale(
  min: number,
  max: number,
  padFraction = 0.05,
  count = 5,
  { floor, plain = false }: { floor?: number; plain?: boolean } = {},
) {
  const span = max - min;
  const pad = span > 0 ? span * padFraction : Math.abs(max) * 0.1 || 1;
  const low = floor !== undefined && min >= floor ? floor : min - pad;
  const high = max + pad;
  const ticks = axisTicks(low, high, count, { plain });
  return {
    min: low,
    max: high,
    ticks: ticks.map((tick) => tick.value),
    labelled: ticks,
    at: (value: number) => (value - low) / (high - low || 1),
  };
}

/** Whether an axis is years: whole numbers in a calendar range print
 *  ungrouped (1990, never "1,990"). */
function isYearAxis(min: number, max: number): boolean {
  return min >= 1000 && max <= 2500;
}

/** A colour-scale end value: the scale's ends are data extremes, not round
 *  ticks, so they get a few significant figures, and never "-0". */
function scaleNumber(value: number): string {
  return Math.abs(value) < 1e-12 ? "0" : formatNumber(value);
}

/** A category label cut to fit `maxPx`, the full text kept in the mark's
 *  evidence for the tooltip. */
function truncate(label: string, maxPx = LABEL_MAX_PX): string {
  return fitText(label, maxPx);
}

/** Every mark the layout placed, in no particular order: for counting tab
 *  stops and finding a group's colour for the legend. */
function placedMarks(layout: PanelLayout): PlacedMark[] {
  if (layout.shape === "scatter_labelled" || layout.shape === "line_series")
    return layout.marks;
  if (layout.shape === "ranked_bars")
    return layout.rows.flatMap((row) => row.marks);
  if (layout.shape === "ribbon")
    return layout.bands.flatMap((band) => band.segments);
  if (layout.shape === "stream")
    return layout.bands.flatMap((band) =>
      band.marks.filter((mark): mark is PlacedMark => mark !== null),
    );
  if (layout.shape === "heatmap") return layout.cells;
  if (layout.shape === "distribution")
    return layout.rows.flatMap((row) => row.points);
  if (layout.shape === "positions")
    return layout.rows.flatMap((row) => row.marks);
  if (layout.shape === "network") return layout.nodes;
  return layout.facets.flatMap((facet) => facet.marks);
}

export function PanelCanvas({
  prepared,
  selected,
  onSelect,
}: {
  prepared: PreparedPanel;
  selected: PanelSelection;
  onSelect: (selection: PanelSelection) => void;
}) {
  const [hovered, setHovered] = useState<Selectable | null>(null);
  const [focusIndex, setFocusIndex] = useState(0);
  // Groups the reader has switched off in the legend. Colours come from the
  // declared groups, not the shown ones, so a group keeps its colour.
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  // Where the pointer is, in the wrapper's own pixels, for the tooltip.
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);
  const [zoom, setZoom] = useState<Zoom | null>(null);
  const [drag, setDrag] = useState<{
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null>(null);
  // A drag ends in a click event too; this keeps it from also picking.
  const dragged = useRef(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const clipId = useId().replaceAll(":", "");
  const shown = hidden.size
    ? {
        ...prepared,
        marks: prepared.marks.filter(
          (mark) => !hidden.has(mark.group || "(all)"),
        ),
      }
    : prepared;
  const result = buildPanelLayout(shown);
  const canvas = canvasSize(prepared);
  const { WIDTH, HEIGHT, PLOT } = canvas;

  // A new panel is a new picture: a stale tab index, hover or hidden group
  // would describe marks that are no longer on the page.
  useEffect(() => {
    setFocusIndex(0);
    setHovered(null);
    setHidden(new Set());
    setPointer(null);
    setZoom(null);
    setDrag(null);
  }, [prepared]);

  if (!result.ok) {
    return <p className="empty-copy">{result.reason}</p>;
  }
  const layout = result.layout;

  // Network edges are tab stops too: in a co-occurrence network the link is
  // the finding, and a keyboard reader must be able to reach it.
  const markTotal =
    placedMarks(layout).length +
    (layout.shape === "network" ? layout.edges.length : 0);
  const activeMark = Math.min(focusIndex, Math.max(0, markTotal - 1));

  let ordinal = 0;
  const markProps = (mark: Selectable, activate: () => void) => {
    const index = ordinal++;
    return {
      tabIndex: index === activeMark ? 0 : -1,
      role: "button" as const,
      "aria-label": mark.evidence.describe,
      "data-mark": index,
      "data-key": mark.key,
      onFocus: () => {
        setFocusIndex(index);
        setHovered(mark);
      },
      onKeyDown: (event: KeyboardEvent) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        } else if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
          event.preventDefault();
          const marks = svgRef.current?.querySelectorAll("[data-mark]");
          if (!marks?.length) return;
          const step = event.key === "ArrowRight" ? 1 : -1;
          const next = (index + step + marks.length) % marks.length;
          (marks[next] as HTMLElement).focus();
        }
      },
      onClick: activate,
      onMouseEnter: () => setHovered(mark),
      style: { cursor: "pointer" },
    };
  };

  const pick = (mark: Selectable) => {
    onSelect(selected?.key === mark.key ? null : { key: mark.key });
  };
  const isDim = (mark: Selectable) =>
    !selected || selected.key === mark.key ? 1 : 0.25;

  const readout = hovered
    ? hovered.evidence.describe
    : "Point to read a mark, click to filter the table beside it.";

  // The legend lists every declared group, shown or not, so a hidden group
  // can be brought back; each keeps its declared colour.
  const legend = prepared.groups.filter((group) => group !== "(all)");
  const legendColors = groupColors(prepared.groups);
  const everyMark = placedMarks(layout);
  const selectables = new Map<string, Selectable>(
    [...everyMark, ...(layout.shape === "network" ? layout.edges : [])].map(
      (mark) => [mark.key, mark],
    ),
  );

  const toggle = (group: string) =>
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(group)) next.delete(group);
      // Never hide the last group: an empty figure reads as a broken one.
      else if (legend.length - next.size > 1) next.add(group);
      return next;
    });

  /** The point nearest the pointer, within reach, for the shapes drawn as
   *  dots: small points should not need pixel-exact aim. */
  const nearest = (event: MouseEvent<SVGSVGElement>): Selectable | null => {
    const svg = svgRef.current;
    const matrix = svg?.getScreenCTM?.();
    if (!svg || !matrix || !NEAREST_SHAPES.has(layout.shape)) return null;
    const at = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      matrix.inverse(),
    );
    let best: Selectable | null = null;
    let bestDistance = NEAREST_REACH;
    for (const circle of svg.querySelectorAll("circle[data-key]")) {
      const dx = Number(circle.getAttribute("cx")) - at.x;
      const dy = Number(circle.getAttribute("cy")) - at.y;
      const distance = Math.hypot(dx, dy);
      if (distance < bestDistance) {
        const mark = selectables.get(circle.getAttribute("data-key") ?? "");
        if (mark) {
          best = mark;
          bestDistance = distance;
        }
      }
    }
    return best;
  };

  const inDrawing = (event: MouseEvent<SVGSVGElement>) => {
    const matrix = svgRef.current?.getScreenCTM?.();
    if (!matrix) return null;
    const at = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      matrix.inverse(),
    );
    return { x: at.x, y: at.y };
  };
  const zoomable = ZOOM_SHAPES.has(layout.shape);

  const onDown = (event: MouseEvent<SVGSVGElement>) => {
    if (!zoomable || event.button !== 0) return;
    if ((event.target as Element).closest?.("[data-mark]")) return;
    const at = inDrawing(event);
    if (at) setDrag({ x0: at.x, y0: at.y, x1: at.x, y1: at.y });
  };

  const onUp = () => {
    if (!drag) return;
    const wide = Math.abs(drag.x1 - drag.x0) >= MIN_DRAG;
    const tall = Math.abs(drag.y1 - drag.y0) >= MIN_DRAG;
    setDrag(null);
    if (!wide || !tall) return;
    dragged.current = true;
    const fx = (px: number) =>
      Math.min(1, Math.max(0, (px - PAD.left) / PLOT.width));
    const fy = (py: number) =>
      Math.min(1, Math.max(0, (PAD.top + PLOT.height - py) / PLOT.height));
    const box = {
      x0: fx(Math.min(drag.x0, drag.x1)),
      x1: fx(Math.max(drag.x0, drag.x1)),
      y0: fy(Math.max(drag.y0, drag.y1)),
      y1: fy(Math.min(drag.y0, drag.y1)),
    };
    // A fraction of the view on screen, composed with any zoom before it.
    const base = zoom ?? { x0: 0, x1: 1, y0: 0, y1: 1 };
    const across = base.x1 - base.x0;
    const up = base.y1 - base.y0;
    setZoom({
      x0: base.x0 + box.x0 * across,
      x1: base.x0 + box.x1 * across,
      y0: base.y0 + box.y0 * up,
      y1: base.y0 + box.y1 * up,
    });
  };

  const onMove = (event: MouseEvent<SVGSVGElement>) => {
    const box = wrapRef.current?.getBoundingClientRect();
    if (box)
      setPointer({ x: event.clientX - box.left, y: event.clientY - box.top });
    if (drag) {
      const at = inDrawing(event);
      if (at) setDrag({ ...drag, x1: at.x, y1: at.y });
      return;
    }
    if ((event.target as Element).closest?.("[data-mark]")) return;
    const close = nearest(event);
    if (close) setHovered(close);
  };

  const onBackgroundClick = (event: MouseEvent<SVGSVGElement>) => {
    if (dragged.current) {
      dragged.current = false;
      return;
    }
    // A click on a mark is the mark's own; only a miss looks for the nearest.
    if ((event.target as Element).closest?.("[data-mark]")) return;
    const close = nearest(event);
    if (close) pick(close);
  };

  const wrapWidth = wrapRef.current?.clientWidth ?? WIDTH;
  const tooltipLeft =
    pointer && pointer.x > wrapWidth - TOOLTIP_WIDTH - 24
      ? pointer.x - TOOLTIP_WIDTH - 14
      : (pointer?.x ?? 0) + 14;

  return (
    <div className="panel-canvas" ref={wrapRef}>
      <p className="chart-readout" role="status">
        {readout}
      </p>
      {hovered && pointer && (
        <div
          className="panel-tooltip"
          aria-hidden="true"
          style={{ left: tooltipLeft, top: pointer.y + 14 }}
        >
          {hovered.evidence.describe}
        </div>
      )}
      <CanvasSize.Provider value={canvas}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={prepared.title}
          onMouseMove={onMove}
          onMouseDown={onDown}
          onMouseUp={onUp}
          onDoubleClick={() => setZoom(null)}
          onClick={onBackgroundClick}
          onMouseLeave={() => {
            setHovered(null);
            setPointer(null);
            setDrag(null);
          }}
        >
          <title>{prepared.title}</title>
          <defs>
            {/* The plot area plus a marker's reach: a point on the frame is
                drawn whole, not halved by the clip. */}
            <clipPath id={clipId}>
              <rect
                x={PAD.left - CLIP_REACH}
                y={PAD.top - CLIP_REACH}
                width={PLOT.width + CLIP_REACH * 2}
                height={PLOT.height + CLIP_REACH * 2}
              />
            </clipPath>
          </defs>
          {layout.shape === "scatter_labelled" && (
            <ScatterFigure
              layout={layout}
              prepared={prepared}
              zoom={zoom}
              clipId={clipId}
              annotations={result.annotations}
              markProps={markProps}
              pick={pick}
              isDim={isDim}
            />
          )}
          {layout.shape === "line_series" && (
            <LineSeriesFigure
              layout={layout}
              prepared={prepared}
              zoom={zoom}
              clipId={clipId}
              markProps={markProps}
              pick={pick}
              isDim={isDim}
              hovered={hovered}
              selected={selected}
            />
          )}
          {layout.shape === "ranked_bars" && (
            <RankedBarsFigure
              layout={layout}
              prepared={prepared}
              markProps={markProps}
              pick={pick}
              isDim={isDim}
            />
          )}
          {layout.shape === "stream" && (
            <StreamFigure
              layout={layout}
              prepared={prepared}
              clipId={clipId}
              markProps={markProps}
              pick={pick}
            />
          )}
          {layout.shape === "ribbon" && (
            <RibbonFigure
              layout={layout}
              prepared={prepared}
              clipId={clipId}
              markProps={markProps}
              pick={pick}
              isDim={isDim}
            />
          )}
          {layout.shape === "heatmap" && (
            <HeatmapFigure
              layout={layout}
              prepared={prepared}
              gradientId={`${clipId}-scale`}
              markProps={markProps}
              pick={pick}
              isDim={isDim}
            />
          )}
          {layout.shape === "distribution" && (
            <DistributionFigure
              layout={layout}
              prepared={prepared}
              markProps={markProps}
              pick={pick}
              isDim={isDim}
            />
          )}
          {layout.shape === "positions" && (
            <PositionsFigure
              layout={layout}
              prepared={prepared}
              markProps={markProps}
              pick={pick}
              isDim={isDim}
            />
          )}
          {layout.shape === "network" && (
            <NetworkFigure
              layout={layout}
              prepared={prepared}
              markProps={markProps}
              pick={pick}
              isDim={isDim}
            />
          )}
          {layout.shape === "small_multiples" && (
            <SmallMultiplesFigure
              layout={layout}
              prepared={prepared}
              markProps={markProps}
              pick={pick}
              isDim={isDim}
            />
          )}
          {drag && (
            <rect
              className="panel-zoom-box"
              x={Math.min(drag.x0, drag.x1)}
              y={Math.min(drag.y0, drag.y1)}
              width={Math.abs(drag.x1 - drag.x0)}
              height={Math.abs(drag.y1 - drag.y0)}
              pointerEvents="none"
            />
          )}
        </svg>
      </CanvasSize.Provider>
      {zoomable && (
        <p className="panel-zoom-hint">
          {zoom ? (
            <button
              type="button"
              className="text-button"
              onClick={() => setZoom(null)}
            >
              Reset zoom
            </button>
          ) : (
            "Drag a box over the plot to zoom in; double-click to reset."
          )}
        </p>
      )}
      {legend.length > 1 && (
        <ul className="chart-legend panel-legend">
          {legend.map((group) => (
            <li key={group}>
              <button
                type="button"
                aria-pressed={!hidden.has(group)}
                className={hidden.has(group) ? "legend-off" : undefined}
                title={
                  hidden.has(group)
                    ? `Show ${group}`
                    : `Hide ${group} (the other groups stay)`
                }
                onClick={() => toggle(group)}
              >
                <span
                  style={{ background: legendColors.get(group) }}
                  aria-hidden="true"
                />
                {group}
              </button>
            </li>
          ))}
        </ul>
      )}
      <p className="panel-caption">{prepared.caption}</p>
      {prepared.notes.length > 0 && (
        <details className="panel-notes">
          <summary>What this can and cannot show</summary>
          <ul>
            {prepared.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

type MarkPropsFn = (
  mark: Selectable,
  activate: () => void,
) => Record<string, unknown>;

type FigureProps = {
  prepared: PreparedPanel;
  markProps: MarkPropsFn;
  pick: (mark: Selectable) => void;
  isDim: (mark: Selectable) => number;
};

/** A plot rectangle, for the shapes that do not use the default one. */
type Area = { left: number; top: number; width: number; height: number };

/** Axes and reference lines shared by every shape's figure. */
function Frame({
  prepared,
  xTicks,
  yTicks,
  yLabelX,
  yTitle = true,
  area,
}: {
  prepared: PreparedPanel;
  xTicks: { at: number; label: string }[];
  yTicks: { at: number; label: string }[];
  yLabelX?: number;
  /** Ranked bars label their own rows where the rotated title would sit. */
  yTitle?: boolean;
  /** The plot rectangle; the default one unless a shape needs its own. */
  area?: Area;
}) {
  const { WIDTH, HEIGHT, PLOT } = useCanvas();
  const box = area ?? {
    left: PAD.left,
    top: PAD.top,
    width: PLOT.width,
    height: PLOT.height,
  };
  const labelX = yLabelX ?? box.left - 8;
  // Thin by width, not by count: every k-th label, with k the smallest that
  // leaves each label its own room, so no two tick labels print over each
  // other at any number of ticks.
  const widest = Math.max(0, ...xTicks.map((tick) => textWidth(tick.label)));
  const spacing =
    xTicks.length > 1
      ? Math.abs(xTicks[xTicks.length - 1].at - xTicks[0].at) /
        (xTicks.length - 1)
      : Infinity;
  const every = Math.max(
    Math.ceil(xTicks.length / MAX_LABELS),
    Math.ceil((widest + 8) / spacing),
  );
  const shown =
    every <= 1 ? xTicks : xTicks.filter((_, index) => index % every === 0);
  return (
    <>
      {shown.map((tick) => (
        <text
          key={`x-${tick.label}`}
          x={tick.at}
          y={box.top + box.height + 16}
          fontSize="10"
          textAnchor="middle"
          fill={TEXT}
        >
          {tick.label}
        </text>
      ))}
      {yTicks.map((tick) => (
        <g key={`y-${tick.label}`}>
          <line
            x1={box.left}
            x2={box.left + box.width}
            y1={tick.at}
            y2={tick.at}
            stroke={GRID}
          />
          <text
            x={labelX}
            y={tick.at + 4}
            fontSize="10"
            textAnchor="end"
            fill={TEXT}
          >
            {tick.label}
          </text>
        </g>
      ))}
      <line
        x1={box.left}
        x2={box.left + box.width}
        y1={box.top + box.height}
        y2={box.top + box.height}
        stroke={AXIS}
      />
      <line
        x1={box.left}
        x2={box.left}
        y1={box.top}
        y2={box.top + box.height}
        stroke={AXIS}
      />
      <text
        x={WIDTH / 2}
        y={HEIGHT - 8}
        fontSize="11"
        textAnchor="middle"
        fill={TEXT}
      >
        {prepared.xLabel}
      </text>
      {yTitle && (
        <text
          x={14}
          y={box.top + box.height / 2}
          fontSize="11"
          textAnchor="middle"
          fill={TEXT}
          transform={`rotate(-90 14 ${box.top + box.height / 2})`}
        >
          {prepared.yLabel}
        </text>
      )}
    </>
  );
}

/** A reference line, drawn with the reason it is there on the page. */
function ReferenceLine({
  annotation,
  xAt,
  yAt,
}: {
  annotation: PanelAnnotation;
  xAt: (value: number) => number;
  yAt: (value: number) => number;
}) {
  const { WIDTH, PLOT } = useCanvas();
  if (annotation.kind === "note") return null;
  const vertical = annotation.kind === "vline";
  const label = `${annotation.label}${annotation.note ? ` — ${annotation.note}` : ""}`;
  return (
    <g className="panel-annotation">
      {vertical ? (
        <line
          x1={xAt(annotation.value)}
          x2={xAt(annotation.value)}
          y1={PAD.top}
          y2={PAD.top + PLOT.height}
          stroke={DOT}
          strokeDasharray="4 3"
        />
      ) : (
        <line
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={yAt(annotation.value)}
          y2={yAt(annotation.value)}
          stroke={DOT}
          strokeDasharray="4 3"
        />
      )}
      <text
        x={vertical ? xAt(annotation.value) + 5 : WIDTH - PAD.right - 5}
        y={vertical ? PAD.top + 12 : yAt(annotation.value) - 5}
        fontSize="10"
        fill={TEXT}
        textAnchor={vertical ? "start" : "end"}
      >
        {label}
      </text>
    </g>
  );
}

/** Row labels at the left of a row-banded plot, cut like the export's, and
 *  thinned to every k-th row when rows are too short to hold text. */
function RowLabels({ labels, area }: { labels: string[]; area: Area }) {
  const band = area.height / Math.max(1, labels.length);
  const every = Math.max(1, Math.ceil(LABEL_PX / band));
  return (
    <>
      {labels.map((label, row) =>
        row % every === 0 ? (
          <text
            key={`row-${row}`}
            x={area.left - 8}
            y={area.top + band * (row + 0.5) + 3.5}
            fontSize="10"
            textAnchor="end"
            fill={TEXT}
          >
            {truncate(label, area.left - 14)}
          </text>
        ) : null,
      )}
    </>
  );
}

/** The left edge for a row-labelled plot: as wide as its labels need. */
function rowLabelLeft(labels: string[], width: number): number {
  return labelMargin(labels, width, { floor: ROW_LABEL_FLOOR });
}

/** Ticks as the Frame takes them, from a scale and its position function. */
function frameTicks(
  scale: { labelled: { value: number; label: string }[] },
  at: (value: number) => number,
): { at: number; label: string }[] {
  return scale.labelled.map((tick) => ({
    at: at(tick.value),
    label: tick.label,
  }));
}

function ScatterFigure({
  layout,
  prepared,
  zoom,
  clipId,
  annotations,
  markProps,
  pick,
  isDim,
}: {
  layout: Extract<PanelLayout, { shape: "scatter_labelled" }>;
  prepared: PreparedPanel;
  zoom: Zoom | null;
  clipId: string;
  annotations: PanelAnnotation[];
  markProps: MarkPropsFn;
  pick: (mark: PlacedMark) => void;
  isDim: (mark: PlacedMark) => number;
}) {
  const { PLOT } = useCanvas();
  // The domain covers the reference lines as well as the marks. Scaled to
  // the marks alone, a line outside their range lands off the plot and is
  // clipped away -- and that is exactly when it matters: a keyness table
  // already cut to its strongest words sits wholly above p < 0.05, and every
  // real PMI score sits far above independence. Plotly's autorange includes
  // shapes, so the exported figure shows these lines; the app must too.
  const xs = [
    ...prepared.marks.map((m) => m.x),
    ...annotations.filter((a) => a.kind === "vline").map((a) => a.value),
  ];
  const ys = [
    ...prepared.marks.map((m) => m.y),
    ...annotations.filter((a) => a.kind === "hline").map((a) => a.value),
  ];
  const baseX = linearScale(Math.min(...xs), Math.max(...xs), 0.06);
  const baseY = linearScale(Math.min(...ys), Math.max(...ys), 0.08);
  const x = zoom ? zoomScale(baseX, zoom.x0, zoom.x1) : baseX;
  const y = zoom ? zoomScale(baseY, zoom.y0, zoom.y1) : baseY;
  const inView = (mark: PlacedMark) =>
    mark.x >= x.min && mark.x <= x.max && mark.y >= y.min && mark.y <= y.max;
  const xAt = (value: number) => PAD.left + x.at(value) * PLOT.width;
  const yAt = (value: number) =>
    PAD.top + PLOT.height - y.at(value) * PLOT.height;
  const largest = Math.max(...prepared.marks.map((m) => m.size ?? 0), 0);
  // Many points share the area: past 300, the biggest bubble shrinks with
  // the count, as the publication figure's does (`shapes.draw_scatter`),
  // so 1,500 pairs are points rather than one blue mass.
  const dense = prepared.marks.length > 300;
  const biggest = dense
    ? Math.max(5, 13 * Math.sqrt(300 / prepared.marks.length))
    : 13;
  const radius = (mark: PlacedMark) =>
    mark.size === null || largest === 0
      ? dense
        ? 2.5
        : MARK_RADIUS
      : Math.max(dense ? 1.8 : 3, Math.sqrt(mark.size / largest) * biggest);
  // Labels the engine chose, placed biggest mark first so the most
  // important label gets its first-choice position; one that cannot be
  // placed clear of the others is left to the hover readout.
  // Zoomed in, every point in view is worth its label: that is what a
  // reader zooms for. Otherwise, the ones the engine chose.
  const labelled = layout.marks
    .filter((mark) => (zoom ? inView(mark) : mark.labelled))
    .sort((a, b) => (b.size ?? 0) - (a.size ?? 0))
    .slice(0, zoom ? 60 : undefined);
  const labels = placeLabels(
    labelled.map((mark) => ({
      x: xAt(mark.x),
      y: yAt(mark.y),
      radius: radius(mark),
      text: mark.label,
    })),
    {
      left: PAD.left,
      top: PAD.top - 4,
      right: PAD.left + PLOT.width,
      bottom: PAD.top + PLOT.height,
    },
  );
  return (
    <>
      <Frame
        prepared={prepared}
        xTicks={
          prepared.xLog10
            ? logTicks(x.min, x.max).map((tick) => ({
                at: xAt(tick.value),
                label: tick.label,
              }))
            : frameTicks(x, xAt)
        }
        yTicks={frameTicks(y, yAt)}
      />
      <g clipPath={`url(#${clipId})`}>
        {annotations.map((annotation, index) => (
          <ReferenceLine
            key={`${annotation.kind}-${index}`}
            annotation={annotation}
            xAt={xAt}
            yAt={yAt}
          />
        ))}
        {layout.marks.map((mark) => (
          <circle
            key={mark.key}
            cx={xAt(mark.x)}
            cy={yAt(mark.y)}
            r={radius(mark)}
            fill={mark.color}
            fillOpacity={0.78}
            opacity={isDim(mark)}
            {...(markProps(mark, () => pick(mark)) as object)}
          />
        ))}
        {labels.map((label) => (
          <text
            key={`label-${labelled[label.index].key}`}
            x={label.x}
            y={label.y}
            fontSize="10"
            textAnchor={label.anchor}
            fill={TEXT}
            pointerEvents="none"
            className="panel-label"
          >
            {label.text}
          </text>
        ))}
      </g>
    </>
  );
}

function LineSeriesFigure({
  layout,
  prepared,
  clipId,
  markProps,
  pick,
  isDim,
  hovered,
  selected,
  zoom,
}: {
  layout: Extract<PanelLayout, { shape: "line_series" }>;
  prepared: PreparedPanel;
  zoom: Zoom | null;
  clipId: string;
  markProps: MarkPropsFn;
  pick: (mark: PlacedMark) => void;
  isDim: (mark: PlacedMark) => number;
  hovered: Selectable | null;
  selected: PanelSelection;
}) {
  const { PLOT } = useCanvas();
  const years = [...new Set(layout.marks.map((mark) => mark.x))].sort(
    (a, b) => a - b,
  );
  const first = years[0];
  const last = years[years.length - 1];
  // Half a step of room at each end of x and 8% above the highest point, so
  // the first, last and highest marks are drawn whole. A series that never
  // goes below zero keeps its zero baseline.
  const baseX = linearScale(first, last, 0.015, 5, {
    plain: isYearAxis(first, last),
  });
  const x = zoom
    ? zoomScale(baseX, zoom.x0, zoom.x1, isYearAxis(first, last))
    : baseX;
  const minY = Math.min(...layout.marks.map((mark) => mark.y));
  const maxY = Math.max(...layout.marks.map((mark) => mark.y));
  // Zero stays on the axis only when the data comes near it (lowest value
  // within 35% of the highest): TTR between 0.12 and 0.37 keeps its zero,
  // MTLD between 55 and 120 does not spend half the plot on empty space.
  // The publication renderer applies the same rule (`shapes._limits`).
  const nearZero = minY >= 0 && minY <= maxY * ZERO_NEAR;
  const baseY = linearScale(minY, maxY > minY ? maxY : minY + 1, 0.08, 5, {
    floor: nearZero ? 0 : undefined,
  });
  const y = zoom ? zoomScale(baseY, zoom.y0, zoom.y1) : baseY;
  const xAt = (value: number) => PAD.left + x.at(value) * PLOT.width;
  const yAt = (value: number) =>
    PAD.top + PLOT.height - y.at(value) * PLOT.height;
  // An annual series labels every year it has. Documents at fractional
  // dates (87 speeches, each at its own day of the year) would print 87
  // overlapping "1990.4520547…" labels, so they get round-number ticks.
  const annual =
    !zoom &&
    years.length <= MAX_LABELS &&
    years.every((year) => Number.isInteger(year));
  const xTicks = annual
    ? years.map((year) => ({ at: xAt(year), label: String(year) }))
    : frameTicks(x, xAt);
  const pointsOnly = new Set(prepared.pointsOnly ?? []);
  // Beside points-only series (each a single document), a joined series is
  // their summary -- a rolling median -- and is drawn as a line, not as a
  // second set of data-sized dots. Its marks stay clickable, drawn only on
  // hover or selection.
  const summarised = pointsOnly.size > 0;
  const markCount = (group: string) =>
    layout.marks.filter((mark) => mark.group === group).length;
  const radiusOf = (mark: PlacedMark) =>
    pointsOnly.has(mark.group) ? 3 : markCount(mark.group) > 40 ? 3 : 4.5;
  return (
    <>
      <Frame prepared={prepared} xTicks={xTicks} yTicks={frameTicks(y, yAt)} />
      <g clipPath={`url(#${clipId})`}>
        {layout.series.flatMap((series) =>
          series.segments.map((segment, index) => (
            <polyline
              key={`${series.group}-${index}`}
              points={segment
                .map((mark) => `${xAt(mark.x)},${yAt(mark.y)}`)
                .join(" ")}
              fill="none"
              stroke={series.color}
              strokeWidth={summarised ? 2.75 : 1.75}
              strokeLinejoin="round"
              aria-hidden="true"
            />
          )),
        )}
        {layout.marks.map((mark) => {
          const summary = summarised && !pointsOnly.has(mark.group);
          return (
            <circle
              key={mark.key}
              cx={xAt(mark.x)}
              cy={yAt(mark.y)}
              r={summary ? 5 : radiusOf(mark)}
              fill={mark.color}
              fillOpacity={
                summary
                  ? hovered?.key === mark.key || selected?.key === mark.key
                    ? 1
                    : 0
                  : pointsOnly.has(mark.group)
                    ? 0.7
                    : 1
              }
              opacity={isDim(mark)}
              {...(markProps(mark, () => pick(mark)) as object)}
            />
          );
        })}
      </g>
    </>
  );
}

function RankedBarsFigure({
  layout,
  prepared,
  markProps,
  pick,
  isDim,
}: {
  layout: Extract<PanelLayout, { shape: "ranked_bars" }>;
  prepared: PreparedPanel;
  markProps: MarkPropsFn;
  pick: (mark: PlacedMark) => void;
  isDim: (mark: PlacedMark) => number;
}) {
  const { WIDTH, PLOT } = useCanvas();
  // Room at the left for the labels themselves: an entity name or an n-gram
  // is the finding, and a 15-character cut turned most of them into "…".
  const left = labelMargin(
    layout.rows.map((row) => row.label),
    WIDTH,
    { floor: PAD.left },
  );
  const area: Area = {
    left,
    top: PAD.top,
    width: WIDTH - left - PAD.right,
    height: PLOT.height,
  };
  const rowHeight = area.height / Math.max(1, layout.rows.length);
  const barHeight = Math.max(4, Math.min(rowHeight * 0.62, 22));
  // The value axis includes zero and ends on round ticks, so the longest bar
  // is read against a labelled value; one precision throughout.
  const [valueMin, valueMax] = niceDomain(
    Math.min(0, layout.valueMin),
    Math.max(0, layout.valueMax),
  );
  const span = valueMax - valueMin || 1;
  const xAt = (value: number) =>
    area.left + ((value - valueMin) / span) * area.width;
  // A value axis and its title, as the exported figure has. Without them a
  // bar's length could be compared with its neighbour's but not read, and the
  // title is where a panel says what a length *is* -- the relevance panel's
  // says its bars are occurrences, not the log-scale scores that order them.
  const xTicks = axisTicks(valueMin, valueMax)
    .filter((tick) => tick.value >= valueMin && tick.value <= valueMax)
    .map((tick) => ({ at: xAt(tick.value), label: tick.label }));
  // Rank 0 is the first row: SVG y grows downward, so draw order is rank
  // order — no reversal, unlike the Plotly renderer, whose category axis
  // grows upward.
  return (
    <>
      {valueMin < 0 && (
        <line
          x1={xAt(0)}
          x2={xAt(0)}
          y1={area.top}
          y2={area.top + area.height}
          stroke={AXIS}
          aria-hidden="true"
        />
      )}
      {layout.rows.map((row, index) => {
        const top = area.top + index * rowHeight + (rowHeight - barHeight) / 2;
        return (
          <g key={`row-${row.label}`}>
            <text
              x={area.left - 8}
              y={top + barHeight / 2 + 4}
              fontSize="10"
              textAnchor="end"
              fill={TEXT}
            >
              {truncate(row.label, area.left - 14)}
            </text>
            {row.marks.map((mark, markIndex) => {
              // Bars grow from the common baseline (x = 0) in the value's
              // direction, so a negative reading still reads as a length.
              const left = xAt(Math.min(0, mark.x));
              const span = Math.abs(xAt(mark.x) - xAt(0));
              return (
                <rect
                  key={mark.key}
                  x={left + 1}
                  y={top + markIndex * 2}
                  width={Math.max(1, span - 2)}
                  height={Math.max(1, barHeight - (row.marks.length - 1) * 2)}
                  fill={mark.color}
                  fillOpacity={markIndex === 0 ? 0.95 : 0.5}
                  opacity={isDim(mark)}
                  {...(markProps(mark, () => pick(mark)) as object)}
                />
              );
            })}
          </g>
        );
      })}
      <Frame
        prepared={prepared}
        xTicks={xTicks}
        yTicks={[]}
        yTitle={false}
        area={area}
      />
    </>
  );
}

function StreamFigure({
  layout,
  prepared,
  clipId,
  markProps,
  pick,
}: {
  layout: Extract<PanelLayout, { shape: "stream" }>;
  prepared: PreparedPanel;
  clipId: string;
  markProps: MarkPropsFn;
  pick: (mark: PlacedMark) => void;
}) {
  const { PLOT } = useCanvas();
  const axis = layout.axis;
  const totalMax = Math.max(...layout.totals, 0) || 1;
  const xAt = (index: number) =>
    PAD.left +
    (axis.length === 1 ? 0.5 : index / (axis.length - 1)) * PLOT.width;
  const yAt = (value: number) =>
    PAD.top + PLOT.height - (value / totalMax) * PLOT.height;
  return (
    <>
      <Frame
        prepared={prepared}
        xTicks={(() => {
          const labels = tickLabels(axis, {
            plain: isYearAxis(axis[0], axis[axis.length - 1]),
          });
          return axis.map((_, index) => ({
            at: xAt(index),
            label: labels[index],
          }));
        })()}
        yTicks={axisTicks(0, totalMax).map((tick) => ({
          at: yAt(tick.value),
          label: tick.label,
        }))}
      />
      <g clipPath={`url(#${clipId})`}>
        {layout.bands.map((band) => {
          // Upper edge left to right, then lower edge right to left: one
          // polygon per band, filled down to the band beneath it.
          const upper = band.upper.map(
            (value, index) => `${xAt(index)},${yAt(value)}`,
          );
          const lower = [...band.lower]
            .map((value, index) => `${xAt(index)},${yAt(value)}`)
            .reverse();
          return (
            <polygon
              key={`band-${band.group}`}
              points={`${upper.join(" ")} ${lower.join(" ")}`}
              fill={band.color}
              fillOpacity={0.82}
              stroke={band.color}
              strokeWidth={0.5}
            />
          );
        })}
        {layout.bands.map((band) =>
          band.marks.map((mark, index) => {
            if (!mark) return null;
            // The band already placed the mark; only its drawn height (upper
            // minus lower) changes, so a click still carries the evidence.
            const clickable: PlacedMark = {
              ...mark,
              y: band.upper[index] - band.lower[index],
            };
            return (
              <circle
                key={mark.key}
                cx={xAt(index)}
                cy={yAt((band.lower[index] + band.upper[index]) / 2)}
                r={4}
                fill={band.color}
                opacity={0.9}
                {...(markProps(clickable, () => pick(clickable)) as object)}
              />
            );
          }),
        )}
      </g>
    </>
  );
}

function RibbonFigure({
  layout,
  prepared,
  clipId,
  markProps,
  pick,
  isDim,
}: {
  layout: Extract<PanelLayout, { shape: "ribbon" }>;
  prepared: PreparedPanel;
  clipId: string;
  markProps: MarkPropsFn;
  pick: (mark: PlacedMark) => void;
  isDim: (mark: PlacedMark) => number;
}) {
  const { WIDTH, PLOT } = useCanvas();
  const max = layout.positionMax || 1;
  const left = rowLabelLeft(
    layout.bands.map((band) => band.label),
    WIDTH,
  );
  const plotWidth = WIDTH - left - PAD.right;
  const xAt = (value: number) => left + (value / max) * plotWidth;
  // Row 0 at the top: SVG y grows downward, so draw order is row order —
  // no reversal, unlike the Plotly renderer, whose category axis counts up.
  const bandHeight = PLOT.height / Math.max(1, layout.bands.length);
  const bandTop = (row: number) =>
    PAD.top +
    Math.max(
      0,
      layout.bands.findIndex((b) => b.row === row),
    ) *
      bandHeight;
  const ticks = axisTicks(0, max);
  return (
    <>
      <Frame
        prepared={prepared}
        xTicks={ticks.map((tick) => ({
          at: xAt(tick.value),
          label: tick.label,
        }))}
        yTicks={[]}
        yTitle={false}
        area={{ left, top: PAD.top, width: plotWidth, height: PLOT.height }}
      />
      {/* Row labels sit left of the plot, so outside the clip: inside it,
          they were clipped away with everything else beyond PAD.left. */}
      {layout.bands.map((band) => (
        <text
          key={`label-${band.label}`}
          x={left - 8}
          y={bandTop(band.row) + bandHeight / 2 + 4}
          fontSize="10"
          textAnchor="end"
          fill={TEXT}
        >
          {truncate(band.label, left - 14)}
        </text>
      ))}
      <g clipPath={`url(#${clipId})`}>
        {layout.bands.map((band) => {
          const top = bandTop(band.row);
          return (
            <g key={`band-${band.label}`}>
              {band.segments.map((segment) => {
                const left = xAt(segment.x);
                const width = Math.max(
                  1,
                  ((segment.size ?? 0) / max) * PLOT.width,
                );
                return (
                  <rect
                    key={segment.key}
                    x={left}
                    y={top + bandHeight * 0.12}
                    width={width}
                    height={bandHeight * 0.76}
                    fill={segment.color}
                    opacity={isDim(segment)}
                    {...(markProps(segment, () => pick(segment)) as object)}
                  />
                );
              })}
            </g>
          );
        })}
      </g>
      <line
        x1={left}
        x2={left}
        y1={PAD.top}
        y2={PAD.top + PLOT.height}
        stroke={AXIS}
      />
    </>
  );
}

/**
 * Heatmap: cell (x, y) is column x and row y, row 0 at the top — SVG y grows
 * downward, so no reversal (the Plotly renderer reverses its axis). The
 * colour was decided in `panelLayout.heatColor`; this draws it, and a scale
 * beside the matrix saying which values the ends of the colour stand for.
 */
function HeatmapFigure({
  layout,
  prepared,
  gradientId,
  markProps,
  pick,
  isDim,
}: FigureProps & {
  layout: Extract<PanelLayout, { shape: "heatmap" }>;
  gradientId: string;
}) {
  const { WIDTH, HEIGHT } = useCanvas();
  const columns = Math.max(1, layout.xCategories.length);
  const rowCount = Math.max(1, layout.yCategories.length);
  // Room below for x labels on a 45-degree slant, and at the right for the
  // colour scale.
  const left = rowLabelLeft(layout.yCategories, WIDTH);
  const area: Area = {
    left,
    top: PAD.top,
    width: WIDTH - left - 90,
    height: HEIGHT - PAD.top - 110,
  };
  const cellWidth = area.width / columns;
  const cellHeight = area.height / rowCount;
  // A hairline between cells while they are big enough to spare it.
  const gap = cellWidth > 4 && cellHeight > 4 ? 1 : 0;
  const everyColumn = Math.max(1, Math.ceil(LABEL_PX / cellWidth));
  // A slanted column label has the 110px below the matrix, less the
  // x-axis title: about 125px along a 45-degree line.
  const columnLabelPx = 125;
  const [low, high] = layout.domain;
  const scaleTop = area.top;
  const scaleHeight = Math.min(area.height, 160);
  const scaleLeft = area.left + area.width + 18;
  return (
    <>
      <defs>
        {/* Low at the bottom, high at the top; evenly spaced stops, which an
            SVG gradient interpolates linearly in RGB -- the same rule. */}
        <linearGradient id={gradientId} x1="0" y1="1" x2="0" y2="0">
          {layout.stops.map((stop, index) => (
            <stop
              key={stop}
              offset={index / (layout.stops.length - 1)}
              stopColor={stop}
            />
          ))}
        </linearGradient>
      </defs>
      <RowLabels labels={layout.yCategories} area={area} />
      {layout.xCategories.map((label, column) => {
        if (column % everyColumn !== 0) return null;
        const x = area.left + cellWidth * (column + 0.5);
        const y = area.top + area.height + 10;
        return (
          <text
            key={`col-${column}`}
            x={x}
            y={y}
            fontSize="10"
            textAnchor="end"
            fill={TEXT}
            transform={`rotate(-45 ${x} ${y})`}
          >
            {truncate(label, columnLabelPx)}
          </text>
        );
      })}
      {layout.cells.map((cell) => (
        <rect
          key={cell.key}
          x={area.left + cell.x * cellWidth}
          y={area.top + cell.y * cellHeight}
          width={Math.max(0.5, cellWidth - gap)}
          height={Math.max(0.5, cellHeight - gap)}
          fill={cell.color}
          opacity={isDim(cell)}
          {...(markProps(cell, () => pick(cell)) as object)}
        />
      ))}
      <g className="panel-scale" aria-hidden="true">
        {prepared.valueLabel && (
          <text
            x={scaleLeft}
            y={scaleTop + scaleHeight + 16}
            fontSize="9"
            fill={TEXT}
          >
            {fitText(prepared.valueLabel, 80, 9)}
          </text>
        )}
        <rect
          x={scaleLeft}
          y={scaleTop}
          width={12}
          height={scaleHeight}
          fill={`url(#${gradientId})`}
          stroke={AXIS}
        />
        {[high, (low + high) / 2, low].map((value, index) => (
          <text
            key={`scale-${index}`}
            x={scaleLeft + 16}
            y={scaleTop + (scaleHeight * index) / 2 + 4}
            fontSize="10"
            fill={TEXT}
          >
            {scaleNumber(value)}
          </text>
        ))}
      </g>
      <text
        x={area.left + area.width / 2}
        y={HEIGHT - 8}
        fontSize="11"
        textAnchor="middle"
        fill={TEXT}
      >
        {prepared.xLabel}
      </text>
    </>
  );
}

/**
 * Distribution: per row, a box from Q1 to Q3 with the median marked and
 * whiskers to the row's extremes, then every observation over it at its
 * jittered height. Row 0 at the top, no reversal.
 */
function DistributionFigure({
  layout,
  prepared,
  markProps,
  pick,
  isDim,
}: FigureProps & {
  layout: Extract<PanelLayout, { shape: "distribution" }>;
}) {
  const { WIDTH, PLOT } = useCanvas();
  const left = rowLabelLeft(
    layout.rows.map((row) => row.label),
    WIDTH,
  );
  const area: Area = {
    left,
    top: PAD.top,
    width: WIDTH - left - PAD.right,
    height: PLOT.height,
  };
  const x = linearScale(layout.valueMin, layout.valueMax);
  const xAt = (value: number) => area.left + x.at(value) * area.width;
  const band = area.height / Math.max(1, layout.rows.length);
  const centre = (row: number) => area.top + band * (row + 0.5);
  const radius = Math.min(4, Math.max(1.5, band * 0.18));
  return (
    <>
      <Frame
        prepared={prepared}
        area={area}
        xTicks={frameTicks(x, xAt)}
        yTicks={[]}
        yTitle={false}
      />
      <RowLabels labels={layout.rows.map((row) => row.label)} area={area} />
      {layout.rows.map((row) => {
        if (!row.stats) return null;
        const { min, q1, median, q3, max } = row.stats;
        const mid = centre(row.row);
        const half = band * 0.3;
        return (
          <g key={`box-${row.row}`} aria-hidden="true" stroke={DOT}>
            <line x1={xAt(min)} x2={xAt(q1)} y1={mid} y2={mid} />
            <line x1={xAt(q3)} x2={xAt(max)} y1={mid} y2={mid} />
            <line
              x1={xAt(min)}
              x2={xAt(min)}
              y1={mid - half / 2}
              y2={mid + half / 2}
            />
            <line
              x1={xAt(max)}
              x2={xAt(max)}
              y1={mid - half / 2}
              y2={mid + half / 2}
            />
            <rect
              x={xAt(q1)}
              y={mid - half}
              width={Math.max(1, xAt(q3) - xAt(q1))}
              height={half * 2}
              fill={DOT}
              fillOpacity={0.12}
            />
            <line
              x1={xAt(median)}
              x2={xAt(median)}
              y1={mid - half}
              y2={mid + half}
              strokeWidth={2}
            />
          </g>
        );
      })}
      {layout.rows.flatMap((row) =>
        row.points.map((point) => (
          <circle
            key={point.key}
            cx={xAt(point.x)}
            cy={centre(row.row) + point.offset * band}
            r={radius}
            fill={point.color}
            fillOpacity={0.8}
            opacity={isDim(point)}
            {...(markProps(point, () => pick(point)) as object)}
          />
        )),
      )}
    </>
  );
}

/**
 * Positions: each row is a document from start to end, each mark a tick
 * where something falls in it. Thin ticks are hard to hit, so each carries
 * a wider transparent stroke that takes the pointer.
 */
function PositionsFigure({
  layout,
  prepared,
  markProps,
  pick,
  isDim,
}: FigureProps & {
  layout: Extract<PanelLayout, { shape: "positions" }>;
}) {
  const { WIDTH, PLOT } = useCanvas();
  const left = rowLabelLeft(
    layout.rows.map((row) => row.label),
    WIDTH,
  );
  const area: Area = {
    left,
    top: PAD.top,
    width: WIDTH - left - PAD.right,
    height: PLOT.height,
  };
  const low = layout.positionMin;
  const high = layout.positionMax > low ? layout.positionMax : low + 1;
  const xAt = (value: number) =>
    area.left + ((value - low) / (high - low)) * area.width;
  const band = area.height / Math.max(1, layout.rows.length);
  return (
    <>
      <Frame
        prepared={prepared}
        area={area}
        xTicks={axisTicks(low, high).map((tick) => ({
          at: xAt(tick.value),
          label: tick.label,
        }))}
        yTicks={[]}
        yTitle={false}
      />
      <RowLabels labels={layout.rows.map((row) => row.label)} area={area} />
      {layout.rows.map((row) => {
        const top = area.top + band * row.row;
        return (
          <g key={`row-${row.row}`}>
            <line
              x1={area.left}
              x2={area.left + area.width}
              y1={top + band / 2}
              y2={top + band / 2}
              stroke={GRID}
              aria-hidden="true"
            />
            {row.marks.map((mark) => (
              <g
                key={mark.key}
                opacity={isDim(mark)}
                {...(markProps(mark, () => pick(mark)) as object)}
              >
                <line
                  x1={xAt(mark.x)}
                  x2={xAt(mark.x)}
                  y1={top + band * 0.15}
                  y2={top + band * 0.85}
                  stroke={mark.color}
                  strokeWidth={1.5}
                />
                <line
                  x1={xAt(mark.x)}
                  x2={xAt(mark.x)}
                  y1={top}
                  y2={top + band}
                  stroke="transparent"
                  strokeWidth={7}
                />
              </g>
            ))}
          </g>
        );
      })}
    </>
  );
}

/**
 * Network: the builder's [0, 1] layout scaled into the plot with room for
 * the largest node at the edges; y = 0 at the top, no reversal. Edges first,
 * so nodes sit on top; each edge is a button like a mark, with a wide
 * transparent stroke under the visible one so a 1px link can be clicked.
 */
function NetworkFigure({
  layout,
  prepared,
  markProps,
  pick,
  isDim,
}: FigureProps & {
  layout: Extract<PanelLayout, { shape: "network" }>;
}) {
  const { WIDTH, HEIGHT } = useCanvas();
  const inset = 24;
  // A dendrogram's leaves stand in one column a row apart, so their labels
  // go to the left, right-aligned, in a margin kept for them; above each
  // leaf they would sit on the one above. The engine does the same.
  const elbow = prepared.edgeStyle === "elbow";
  const labelRoom = elbow ? 150 : 0;
  const area: Area = {
    left: 20 + inset + labelRoom,
    top: PAD.top + inset,
    width: WIDTH - 40 - inset * 2 - labelRoom,
    height: HEIGHT - PAD.top - 36 - inset * 2,
  };
  const xAt = (value: number) => area.left + value * area.width;
  const yAt = (value: number) => area.top + value * area.height;
  const largest = Math.max(...layout.nodes.map((m) => m.size ?? 0), 0);
  // The same area-proportional sizing as the scatter.
  const radius = (mark: PlacedMark) =>
    mark.size === null || largest === 0
      ? MARK_RADIUS + 1
      : Math.max(3, Math.sqrt(mark.size / largest) * 13);
  const path = (edge: PlacedEdge) =>
    edge.points.map((point) => `${xAt(point.x)},${yAt(point.y)}`).join(" ");
  // Hub words first: the biggest node's label gets its first-choice place.
  const labelledNodes = layout.nodes
    .filter((node) => node.labelled)
    .sort((a, b) => (b.size ?? 0) - (a.size ?? 0));
  const networkLabels = placeLabels(
    labelledNodes.map((node) => ({
      x: xAt(node.x),
      y: yAt(node.y),
      radius: radius(node),
      text: node.label,
    })),
    { left: 4, top: 4, right: WIDTH - 4, bottom: HEIGHT - 24 },
  );
  return (
    <>
      {layout.edges.map((edge) => (
        <g
          key={edge.key}
          opacity={isDim(edge)}
          {...(markProps(edge, () => pick(edge)) as object)}
        >
          <polyline
            points={path(edge)}
            fill="none"
            stroke={EDGE}
            strokeWidth={edge.width}
            strokeOpacity={0.8}
          />
          <polyline
            points={path(edge)}
            fill="none"
            stroke="transparent"
            strokeWidth={Math.max(10, edge.width + 6)}
          />
        </g>
      ))}
      {layout.nodes.map((node) => (
        <circle
          key={node.key}
          cx={xAt(node.x)}
          cy={yAt(node.y)}
          r={radius(node)}
          fill={node.color}
          stroke="#ffffff"
          strokeWidth={0.8}
          opacity={isDim(node)}
          {...(markProps(node, () => pick(node)) as object)}
        />
      ))}
      {elbow
        ? layout.nodes
            .filter((node) => node.labelled)
            .map((node) => (
              <text
                key={`label-${node.key}`}
                x={xAt(node.x) - radius(node) - 4}
                y={yAt(node.y) + 3}
                fontSize="10"
                textAnchor="end"
                fill={TEXT}
                pointerEvents="none"
              >
                {truncate(node.label, labelRoom)}
              </text>
            ))
        : networkLabels.map((label) => (
            <text
              key={`label-${labelledNodes[label.index].key}`}
              x={label.x}
              y={label.y}
              fontSize="10"
              textAnchor={label.anchor}
              fill={TEXT}
              pointerEvents="none"
              className="panel-label"
            >
              {label.text}
            </text>
          ))}
      {prepared.xLabel && (
        <text
          x={WIDTH / 2}
          y={HEIGHT - 8}
          fontSize="11"
          textAnchor="middle"
          fill={TEXT}
        >
          {prepared.xLabel}
        </text>
      )}
    </>
  );
}

/**
 * Small multiples: a grid of little line plots, at most three across, one
 * per facet in declared order. The x scale is shared so a year sits in the
 * same place in every cell; each y scale is the facet's own, which is the
 * point of the shape — measures in different units side by side, none
 * flattened by another's range.
 */
function SmallMultiplesFigure({
  layout,
  prepared,
  markProps,
  pick,
  isDim,
}: FigureProps & {
  layout: Extract<PanelLayout, { shape: "small_multiples" }>;
}) {
  const { WIDTH, HEIGHT } = useCanvas();
  const grid: Area = {
    left: 16,
    top: 8,
    width: WIDTH - 16 - PAD.right,
    height: HEIGHT - 8 - 30,
  };
  const cellWidth = grid.width / layout.columns;
  const cellHeight = grid.height / Math.max(1, layout.rows);
  const x = linearScale(layout.xMin, layout.xMax, 0.03, 4, {
    plain: isYearAxis(layout.xMin, layout.xMax),
  });
  const pointsOnly = new Set(prepared.pointsOnly ?? []);
  return (
    <>
      {layout.facets.map((facet, index) => {
        const cellLeft = grid.left + (index % layout.columns) * cellWidth;
        const cellTop =
          grid.top + Math.floor(index / layout.columns) * cellHeight;
        const plot: Area = {
          left: cellLeft + 48,
          top: cellTop + 20,
          width: cellWidth - 48 - 14,
          height: cellHeight - 20 - 26,
        };
        const y = linearScale(facet.yMin, facet.yMax, 0.08, 3);
        const xAt = (value: number) => plot.left + x.at(value) * plot.width;
        const yAt = (value: number) =>
          plot.top + plot.height - y.at(value) * plot.height;
        return (
          <g key={`facet-${facet.name}`} className="panel-facet">
            <text
              x={plot.left}
              y={cellTop + 13}
              fontSize="11"
              fontWeight={600}
              fill={TEXT}
            >
              {facet.name}
            </text>
            {y.labelled.map((tick) => (
              <g key={`y-${tick.value}`} aria-hidden="true">
                <line
                  x1={plot.left}
                  x2={plot.left + plot.width}
                  y1={yAt(tick.value)}
                  y2={yAt(tick.value)}
                  stroke={GRID}
                />
                <text
                  x={plot.left - 6}
                  y={yAt(tick.value) + 3.5}
                  fontSize="9"
                  textAnchor="end"
                  fill={TEXT}
                >
                  {tick.label}
                </text>
              </g>
            ))}
            {x.labelled.map((tick) => (
              <text
                key={`x-${tick.value}`}
                x={xAt(tick.value)}
                y={plot.top + plot.height + 13}
                fontSize="9"
                textAnchor="middle"
                fill={TEXT}
                aria-hidden="true"
              >
                {tick.label}
              </text>
            ))}
            <line
              x1={plot.left}
              x2={plot.left + plot.width}
              y1={plot.top + plot.height}
              y2={plot.top + plot.height}
              stroke={AXIS}
            />
            {facet.series.flatMap((series) =>
              series.segments.map((segment, segmentIndex) => (
                <polyline
                  key={`${series.group}-${segmentIndex}`}
                  points={segment
                    .map((mark) => `${xAt(mark.x)},${yAt(mark.y)}`)
                    .join(" ")}
                  fill="none"
                  stroke={series.color}
                  strokeWidth={1.6}
                  aria-hidden="true"
                />
              )),
            )}
            {facet.marks.map((mark) => (
              <circle
                key={mark.key}
                cx={xAt(mark.x)}
                cy={yAt(mark.y)}
                r={pointsOnly.has(mark.group) ? 2.5 : 3}
                fill={mark.color}
                fillOpacity={pointsOnly.has(mark.group) ? 0.6 : 1}
                opacity={isDim(mark)}
                {...(markProps(mark, () => pick(mark)) as object)}
              />
            ))}
          </g>
        );
      })}
      <text
        x={WIDTH / 2}
        y={HEIGHT - 8}
        fontSize="11"
        textAnchor="middle"
        fill={TEXT}
      >
        {prepared.xLabel}
      </text>
    </>
  );
}
