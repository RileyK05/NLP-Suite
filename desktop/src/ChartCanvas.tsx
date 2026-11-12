import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { ArrowDownToLine, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import type {
  Box,
  CalendarCell,
  ChartSelection,
  Cell,
  FlowLink,
  Layout,
  Mark,
  PieSlice,
  SunburstArc,
  TreemapTile,
  ViolinGroup,
  WaffleSquare,
} from "./chartLayout";
import { formatNumber } from "./chartLayout";
import { fitText, textWidth } from "./panelTicks";
import {
  IDENTITY,
  isIdentity,
  panBy,
  zoomAt,
  type ViewTransform,
} from "./chartTransform";

/**
 * A layout, drawn.
 *
 * Plain SVG rather than a charting library: the app is offline and packaged,
 * and every mark here has to be clickable and identifiable so that pointing at
 * a bar can say what it stands for and filter the table beneath it. Nothing is
 * computed in this file — geometry comes from `chartLayout`, which is pure and
 * tested, and the view (zoom/pan) comes from `chartTransform`, equally pure —
 * so what is drawn and what is claimed cannot drift apart.
 *
 * Reading a chart is also possible without a pointer: every mark is a button
 * in the tab order (one stop per chart; arrows move between marks), with the
 * same description a hover shows. A chart that only answers a mouse tells part
 * of the audience nothing.
 */

const WIDTH = 900;
const HEIGHT = 430;
const PAD = { top: 24, right: 24, bottom: 70, left: 78 };
const PLOT = {
  width: WIDTH - PAD.left - PAD.right,
  height: HEIGHT - PAD.top - PAD.bottom,
};
const AXIS = "#b3bdb2";
const GRID = "#e8ece5";
const TEXT = "#5d6a5c";
const BUBBLE_MAX = 22;
const SVG_NS = "http://www.w3.org/2000/svg";

export type Selection = ChartSelection | null;

/** Everything the pointer can read in a chart. The non-Cartesian kinds carry
 *  their own geometry, but every one of them still answers with a label and a
 *  group, because that is all a click needs to filter the table. */
type HoverItem =
  | Mark
  | Box
  | Cell
  | PieSlice
  | SunburstArc
  | TreemapTile
  | ViolinGroup
  | WaffleSquare
  | CalendarCell
  | FlowLink;

const TAU = Math.PI * 2;
const TWELVE = -Math.PI / 2;

/** A point on a circle, twelve o'clock clockwise. */
function polar(cx: number, cy: number, radius: number, angle: number) {
  return {
    x: cx + Math.cos(angle) * radius,
    y: cy + Math.sin(angle) * radius,
  };
}

/** A pie wedge, including the centre. */
function wedgePath(
  cx: number,
  cy: number,
  radius: number,
  start: number,
  end: number,
): string {
  const from = polar(cx, cy, radius, start);
  const to = polar(cx, cy, radius, end);
  const large = end - start > Math.PI ? 1 : 0;
  return `M ${cx} ${cy} L ${from.x} ${from.y} A ${radius} ${radius} 0 ${large} 1 ${to.x} ${to.y} Z`;
}

/** A ring segment of a sunburst. */
function ringPath(
  cx: number,
  cy: number,
  inner: number,
  outer: number,
  start: number,
  end: number,
): string {
  const a = polar(cx, cy, outer, start);
  const b = polar(cx, cy, outer, end);
  const c = polar(cx, cy, inner, end);
  const d = polar(cx, cy, inner, start);
  const large = end - start > Math.PI ? 1 : 0;
  return `M ${a.x} ${a.y} A ${outer} ${outer} 0 ${large} 1 ${b.x} ${b.y} L ${c.x} ${c.y} A ${inner} ${inner} 0 ${large} 0 ${d.x} ${d.y} Z`;
}

function cellColor(value: number, min: number, max: number) {
  const low = [237, 242, 247];
  const signed = min < 0 && max > 0;
  const high = signed && value < 0 ? [154, 59, 24] : [18, 78, 123];
  const span = max - min;
  const share = signed
    ? Math.abs(value) / max
    : max === min
      ? 0.55
      : Number.isFinite(span)
        ? (value - min) / span
        : (value / 2 - min / 2) / (max / 2 - min / 2);
  const channels = low.map((start, index) =>
    Math.round(start + (high[index] - start) * share),
  );
  return {
    fill: `rgb(${channels[0]} ${channels[1]} ${channels[2]})`,
    ink: share >= 0.71 ? "#fff" : "#263d34",
  };
}

function ColorKey({
  x,
  y,
  label,
  min,
  max,
}: {
  x: number;
  y: number;
  label: string;
  min: number;
  max: number;
}) {
  const id = useId();
  const signed = min < 0 && max > 0;
  const width = 112;
  return (
    <g>
      <defs>
        <linearGradient id={id}>
          <stop offset="0" stopColor={cellColor(min, min, max).fill} />
          {signed && (
            <stop offset="0.5" stopColor={cellColor(0, min, max).fill} />
          )}
          <stop offset="1" stopColor={cellColor(max, min, max).fill} />
        </linearGradient>
      </defs>
      <text x={x - 10} y={y + 4} textAnchor="end" fontSize="12" fill={TEXT}>
        {label}
      </text>
      <rect
        x={x}
        y={y - 5}
        width={min === max ? 12 : width}
        height={10}
        fill={`url(#${id})`}
        stroke="#d7e1e9"
        strokeWidth="0.6"
      />
      {[min, ...(signed ? [0] : []), ...(min === max ? [] : [max])].map(
        (value, index, ticks) => (
          <text
            key={value}
            x={x + (index / Math.max(1, ticks.length - 1)) * width}
            y={y + 19}
            fontSize="11"
            fill={TEXT}
            textAnchor={
              index === 0 ? "start" : value === max ? "end" : "middle"
            }
          >
            {formatNumber(value)}
          </text>
        ),
      )}
    </g>
  );
}

/** Fractional part of a number, for deterministic jitter. */
function frac(value: number): number {
  return value - Math.floor(value);
}

/** A short share for a label inside a slice ("12%"). */
function percent(share: number): string {
  if (share >= 0.1) return `${Math.round(share * 100)}%`;
  if (share >= 0.01) return `${(share * 100).toFixed(0)}%`;
  return "";
}

function ticksToShow(layout: Layout): { at: number; label: string }[] {
  const ticks = layout.x.ticks;
  if (layout.x.kind !== "category") return ticks;
  // Category labels overlap once there are more than a couple of dozen; thin
  // them evenly rather than letting them collide into an unreadable band.
  const every = Math.ceil(ticks.length / 24);
  return every <= 1 ? ticks : ticks.filter((_, index) => index % every === 0);
}

/** Split long titles and legend entries within the available SVG width. */
function wrapText(text: string, max: number): string[] {
  const lines: string[] = [];
  let rest = text.trim();
  while (rest.length > max) {
    const space = rest.lastIndexOf(" ", max);
    const split = space > 0 ? space : max;
    lines.push(rest.slice(0, split));
    rest = rest.slice(split).trimStart();
  }
  if (rest) lines.push(rest);
  return lines;
}

function chartImage(
  svg: SVGSVGElement,
  title: string,
  subtitle: string | undefined,
  layout: Layout,
) {
  const root = document.createElementNS(SVG_NS, "svg");
  const append = (
    name: string,
    attributes: Record<string, string | number>,
    text?: string,
  ) => {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attributes))
      node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    root.append(node);
    return node;
  };
  append("title", {}, title);
  root.setAttribute("role", "img");
  root.setAttribute("aria-label", title);
  root.setAttribute("font-family", "Arial, sans-serif");
  const background = append("rect", { width: WIDTH, fill: "#fff" });
  let y = 34;
  for (const line of wrapText(title, 76)) {
    append(
      "text",
      { x: PAD.left, y, "font-size": 22, "font-weight": 700, fill: "#263d34" },
      line,
    );
    y += 28;
  }
  for (const line of wrapText(subtitle ?? "", 112)) {
    append("text", { x: PAD.left, y, "font-size": 14, fill: TEXT }, line);
    y += 18;
  }
  const plot = svg.cloneNode(true) as SVGSVGElement;
  for (const node of plot.querySelectorAll("[data-mark]")) {
    for (const attribute of ["tabindex", "role", "aria-pressed", "data-mark"])
      node.removeAttribute(attribute);
  }
  plot.removeAttribute("role");
  plot.removeAttribute("aria-label");
  plot.setAttribute("width", String(WIDTH));
  plot.setAttribute("height", String(HEIGHT));
  plot.setAttribute("y", String(y - 12));
  root.append(plot);
  y += HEIGHT;
  if (layout.series.length > 1) {
    const width = (WIDTH - PAD.left - PAD.right) / 3;
    for (let start = 0; start < layout.series.length; start += 3) {
      let rowHeight = 18;
      layout.series.slice(start, start + 3).forEach((series, column) => {
        const x = PAD.left + width * column;
        const suffix =
          layout.shape === "pie" && series.marks[0]
            ? ` · ${formatNumber(series.marks[0].y)}`
            : "";
        const lines = wrapText(`${series.group}${suffix}`, 34);
        append("rect", {
          x,
          y: y - 9,
          width: 10,
          height: 10,
          rx: 2,
          fill: series.color,
        });
        lines.forEach((line, index) =>
          append(
            "text",
            {
              x: x + 16,
              y: y + 16 * index,
              "font-size": 13,
              fill: TEXT,
            },
            line,
          ),
        );
        rowHeight = Math.max(rowHeight, lines.length * 16 + 6);
      });
      y += rowHeight;
    }
  }
  const height = y + 12;
  background.setAttribute("height", String(height));
  root.setAttribute("width", String(WIDTH));
  root.setAttribute("height", String(height));
  root.setAttribute("viewBox", `0 0 ${WIDTH} ${height}`);
  return { markup: new XMLSerializer().serializeToString(root), height };
}

export function ChartCanvas({
  layout,
  selected,
  onSelect,
  title,
  subtitle,
  name,
}: {
  layout: Layout;
  selected: Selection;
  onSelect: (selection: Selection) => void;
  title: string;
  /** A short line under the title: what the picture covers. */
  subtitle?: string;
  /** Filename base for downloads; defaults to the title. */
  name?: string;
}) {
  const [hovered, setHovered] = useState<HoverItem | null>(null);
  const [view, setView] = useState<ViewTransform>(IDENTITY);
  const [panning, setPanning] = useState(false);
  const [focusIndex, setFocusIndex] = useState(0);
  const [exportError, setExportError] = useState("");
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<HTMLDivElement>(null);
  // Set when a drag ends so the click that follows it (fired by the browser
  // on release) does not also select whatever mark the pointer lands on.
  const suppressClick = useRef(false);
  const drag = useRef<{
    pointerId: number;
    startClientX: number;
    startPan: number;
    moved: boolean;
  } | null>(null);
  const clipId = useId();
  const [colorMin, colorMax] = useMemo(() => {
    const values = layout.cells.map((cell) => cell.value);
    const min = layout.calendar?.min ?? Math.min(...values);
    const max = layout.calendar?.max ?? Math.max(...values);
    const extent = Math.max(-min, max);
    return min < 0 && max > 0 ? [-extent, extent] : [min, max];
  }, [layout]);

  // A new arrangement is a new picture: keeping an old window onto different
  // axes would show a slice of categories the reader never chose.
  useEffect(() => {
    setView(IDENTITY);
    setFocusIndex(0);
    setHovered(null);
  }, [layout]);

  // Wheel zoom needs a non-passive listener: React registers wheel as passive
  // at the root, so preventDefault inside onWheel cannot stop the page from
  // scrolling while the pointer is over the plot.
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      // Only the Cartesian kinds have an x window to zoom; a pie or a waffle
      // has no axis, and swallowing the page's scroll over one is a theft.
      if (
        !["bars", "line", "points"].includes(layout.shape) ||
        layout.horizontal ||
        !(event.ctrlKey || event.metaKey)
      )
        return;
      const rect = el.getBoundingClientRect();
      if (!rect.width) return;
      const x = ((event.clientX - rect.left) / rect.width) * WIDTH;
      const y = ((event.clientY - rect.top) / rect.height) * HEIGHT;
      // Wheeling over the axis labels keeps its ordinary meaning (scrolling
      // the page); only the plot itself swallows it.
      if (
        x < PAD.left ||
        x > WIDTH - PAD.right ||
        y < PAD.top ||
        y > HEIGHT - PAD.bottom
      )
        return;
      event.preventDefault();
      const anchor = (x - PAD.left) / PLOT.width;
      const factor = event.deltaY < 0 ? 1.25 : 0.8;
      setView((current) => zoomAt(current, factor, anchor));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [layout.shape, layout.horizontal]);

  const clientToViewX = (clientX: number) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || !rect.width) return 0;
    return ((clientX - rect.left) / rect.width) * WIDTH;
  };

  const xAt = (value: number) =>
    PAD.left +
    ((value - layout.x.min) / (layout.x.max - layout.x.min || 1)) * PLOT.width;
  const yAt = (value: number) =>
    PAD.top +
    PLOT.height -
    ((value - layout.y.min) / (layout.y.max - layout.y.min || 1)) * PLOT.height;
  const horizontalBars = layout.shape === "bars" && !!layout.horizontal;
  // A horizontal ranking spends its left margin on words, and words deserve
  // the room. The margin follows the longest shown label, capped so the plot
  // keeps most of the width.
  const leftPad = horizontalBars
    ? Math.min(
        214,
        Math.max(
          78,
          Math.max(0, ...layout.x.categories.map((name) => name.length)) * 6.2 +
            18,
        ),
      )
    : PAD.left;
  const plotWidth = WIDTH - leftPad - PAD.right;
  const hx = (value: number) =>
    leftPad +
    ((value - layout.y.min) / (layout.y.max - layout.y.min || 1)) * plotWidth;
  const rowHeight = PLOT.height / Math.max(1, layout.x.categories.length);

  const isSelected = ({ label, group, key }: ChartSelection) =>
    selected?.key !== undefined
      ? selected.key === key
      : selected?.label === label && selected.group === group;
  const dim = (label: string, group: string, key?: string) =>
    !selected || isSelected({ label, group, key }) ? 1 : 0.25;

  const bandWidth =
    layout.x.kind === "category"
      ? PLOT.width / Math.max(1, layout.x.categories.length)
      : 0;
  const groupCount = Math.max(1, layout.series.length);
  const bandInset = bandWidth * 0.09;
  /** Values above every bar are a gift at twelve categories and a smear at
   *  forty. The threshold is about the reader, not the data. */
  const showBarValues =
    layout.shape === "bars" &&
    layout.series.length <= 2 &&
    layout.x.categories.length <= 14;

  const pick = ({ label, group, rowIndices, parent, key }: ChartSelection) => {
    if (suppressClick.current) return;
    // Waffle squares share one selection per category.
    const selection = {
      label,
      group,
      rowIndices,
      parent,
      key: layout.shape === "waffle" ? undefined : key,
    };
    onSelect(isSelected(selection) ? null : selection);
  };

  // Marks in draw order, counted before rendering so the roving tab stop can
  // survive the mark list shrinking (a re-layout with fewer marks must not
  // leave the tab index pointing past the end of the list).
  const markTotal =
    layout.shape === "box"
      ? layout.boxes.length
      : layout.shape === "heatmap"
        ? layout.cells.length
        : layout.shape === "pie"
          ? (layout.slices?.length ?? 0)
          : layout.shape === "sunburst"
            ? (layout.arcs?.length ?? 0)
            : layout.shape === "treemap"
              ? (layout.tiles?.length ?? 0)
              : layout.shape === "violin"
                ? (layout.violins?.length ?? 0)
                : layout.shape === "waffle"
                  ? (layout.waffle?.squares.length ?? 0)
                  : layout.shape === "calendar"
                    ? (layout.calendar?.cells.length ?? 0)
                    : layout.shape === "sankey"
                      ? (layout.flows?.links.length ?? 0)
                      : layout.series.reduce(
                          (total, series) => total + series.marks.length,
                          0,
                        );
  const activeMark = Math.min(focusIndex, Math.max(0, markTotal - 1));
  const isHovered = (item: HoverItem) => hovered?.key === item.key;

  /**
   * The keyboard half of a mark: roving tab stop, arrows to move, Enter to
   * pick — the same selection a click produces, so pointing and tabbing never
   * disagree about what a mark means.
   */
  let markOrdinal = 0;
  const markKeys = (item: HoverItem, activate: () => void) => {
    const index = markOrdinal++;
    return {
      tabIndex: index === activeMark ? 0 : -1,
      role: "button" as const,
      "aria-label": describe(item, layout) ?? "",
      "aria-pressed":
        "label" in item
          ? isSelected({
              ...item,
              key: layout.shape === "waffle" ? undefined : item.key,
            })
          : isSelected({ label: item.column, group: item.row }),
      "data-mark": index,
      onFocus: () => {
        setFocusIndex(index);
        setHovered(item);
      },
      onKeyDown: (event: KeyboardEvent) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          if (!suppressClick.current) activate();
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
    };
  };

  const readout = describe(hovered, layout);

  // The view maps data x-positions to screen: screen = panPx + scale * data.
  // Applied as an SVG transform about the plot's left edge, so a point at
  // PAD.left stays at PAD.left when only the scale changes.
  const panPx = view.pan * PLOT.width;
  const plotShift = `translate(${PAD.left + panPx - view.scale * PAD.left} 0) scale(${view.scale} 1)`;

  const fileBase = (name ?? title).replace(/[^\w.-]+/g, "_") || "chart";

  const exportImage = () => {
    const svg = svgRef.current;
    if (!svg) throw new Error("The chart is not on the page yet.");
    return chartImage(svg, title, subtitle, layout);
  };

  const saveBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const downloadSvg = () => {
    try {
      setExportError("");
      saveBlob(
        new Blob([exportImage().markup], { type: "image/svg+xml" }),
        `${fileBase}.svg`,
      );
    } catch (error) {
      setExportError(error instanceof Error ? error.message : String(error));
    }
  };

  const downloadPng = () => {
    try {
      setExportError("");
      const { markup, height } = exportImage();
      const url = URL.createObjectURL(
        new Blob([markup], { type: "image/svg+xml" }),
      );
      const image = new Image();
      image.onload = () => {
        // 2x the drawing size so the PNG stays crisp on high-density screens.
        const canvas = document.createElement("canvas");
        canvas.width = WIDTH * 2;
        canvas.height = height * 2;
        const context = canvas.getContext("2d");
        URL.revokeObjectURL(url);
        if (!context) {
          setExportError("This environment cannot rasterize images.");
          return;
        }
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((png) => {
          if (png) saveBlob(png, `${fileBase}.png`);
          else setExportError("The PNG could not be encoded on this machine.");
        }, "image/png");
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        setExportError("The chart could not be rendered as a PNG here.");
      };
      image.src = url;
    } catch (error) {
      setExportError(error instanceof Error ? error.message : String(error));
    }
  };

  const zoomable =
    ["bars", "line", "points"].includes(layout.shape) && !horizontalBars;
  const cartesian = ![
    "pie",
    "sunburst",
    "treemap",
    "radar",
    "waffle",
    "calendar",
    "sankey",
  ].includes(layout.shape);
  const tooltipLeft =
    pointer && plotRef.current
      ? pointer.x > plotRef.current.clientWidth - 300
        ? pointer.x - 286
        : pointer.x + 16
      : 0;
  const sliceShape = ["pie", "waffle", "sunburst", "treemap"].includes(
    layout.shape,
  );
  const legendItems =
    sliceShape && layout.series.length > 12
      ? layout.series.slice(0, 12)
      : layout.series;
  const legendRest = layout.series.length - legendItems.length;

  return (
    <div className="chart-canvas" ref={wrapRef}>
      <div className="chart-head">
        <div>
          <span className="chart-kicker">
            {layout.shape === "pie"
              ? "Share of a whole"
              : layout.shape === "sunburst"
                ? "Two-level hierarchy"
                : layout.shape === "treemap"
                  ? "Area by share"
                  : layout.shape === "violin"
                    ? "Distribution"
                    : layout.shape === "radar"
                      ? "Profile on spokes"
                      : layout.shape === "waffle"
                        ? "100 squares, one per percent"
                        : layout.shape === "calendar"
                          ? "Calendar heatmap"
                          : layout.shape === "sankey"
                            ? "Flow"
                            : layout.shape === "heatmap"
                              ? "Matrix"
                              : layout.shape === "line"
                                ? layout.step
                                  ? "Cumulative distribution"
                                  : "Trend"
                                : layout.shape === "points"
                                  ? "Relationship"
                                  : layout.shape === "box"
                                    ? "Distribution"
                                    : "Comparison"}
          </span>
          <div className="chart-title">{title}</div>
          {subtitle && <p>{subtitle}</p>}
        </div>
        <div className="chart-tools">
          {zoomable && (
            <div className="chart-zoom" role="group" aria-label="Chart zoom">
              <button
                type="button"
                className="text-button"
                aria-label="Zoom out"
                disabled={isIdentity(view)}
                onClick={() => setView((current) => zoomAt(current, 0.8, 0.5))}
              >
                <ZoomOut size={16} aria-hidden="true" />
              </button>
              <button
                type="button"
                className="text-button"
                aria-label="Zoom in"
                onClick={() => setView((current) => zoomAt(current, 1.25, 0.5))}
              >
                <ZoomIn size={16} aria-hidden="true" />
              </button>
            </div>
          )}
          {!isIdentity(view) && zoomable && (
            <button
              type="button"
              className="text-button"
              title="Show the whole plot again"
              onClick={() => setView({ ...IDENTITY })}
            >
              <RotateCcw size={14} aria-hidden="true" /> Reset zoom
            </button>
          )}
          <span className="chart-tools-end">
            {/* What downloads is the chart as drawn — this page's rows, this
                window — so the buttons say "download", not "export the data". */}
            <button
              type="button"
              className="text-button"
              title="Save this chart as an SVG image, as drawn on this page"
              onClick={downloadSvg}
            >
              <ArrowDownToLine size={14} aria-hidden="true" /> SVG
            </button>
            <button
              type="button"
              className="text-button"
              title="Save this chart as a PNG image, as drawn on this page"
              onClick={downloadPng}
            >
              <ArrowDownToLine size={14} aria-hidden="true" /> PNG
            </button>
          </span>
        </div>
      </div>
      {exportError && (
        <p className="alert warning" role="status">
          {exportError}
        </p>
      )}
      <div className="chart-plot" ref={plotRef}>
        <p className="chart-readout" role="status">
          {readout ?? (
            <span className="muted">
              Select a mark to see its rows. Use arrow keys to move between
              marks.
              {zoomable ? " Drag to pan after zooming." : ""}
            </span>
          )}
        </p>
        {hovered && pointer && (
          <div
            className="chart-tooltip"
            role="tooltip"
            style={{ left: tooltipLeft, top: pointer.y + 18 }}
          >
            {describe(hovered, layout)}
          </div>
        )}
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="group"
          aria-label={title}
          onMouseLeave={() => {
            setHovered(null);
            setPointer(null);
          }}
          onMouseMove={(event) => {
            const box = plotRef.current?.getBoundingClientRect();
            if (box)
              setPointer({
                x: event.clientX - box.left,
                y: event.clientY - box.top,
              });
          }}
          onPointerDown={(event) => {
            if (!zoomable || event.button !== 0) return;
            drag.current = {
              pointerId: event.pointerId,
              startClientX: event.clientX,
              startPan: view.pan,
              moved: false,
            };
            // Capturing here redirects a normal mark click to the SVG root.
            // Wait until this gesture actually becomes a drag.
            suppressClick.current = false;
          }}
          onPointerMove={(event) => {
            const active = drag.current;
            if (!active || active.pointerId !== event.pointerId) return;
            const delta =
              (clientToViewX(event.clientX) -
                clientToViewX(active.startClientX)) /
              PLOT.width;
            if (!active.moved && Math.abs(delta) > 0.005) {
              active.moved = true;
              event.currentTarget.setPointerCapture(event.pointerId);
              setPanning(true);
            }
            if (active.moved) {
              setView((current) =>
                panBy({ scale: current.scale, pan: active.startPan }, delta),
              );
            }
          }}
          onPointerUp={(event) => {
            const active = drag.current;
            if (!active || active.pointerId !== event.pointerId) return;
            suppressClick.current = active.moved;
            // The click that follows release must see the flag, then the flag
            // must not outlive the gesture.
            setTimeout(() => (suppressClick.current = false), 0);
            drag.current = null;
            setPanning(false);
          }}
          onPointerCancel={() => {
            drag.current = null;
            setPanning(false);
          }}
          onPointerLeave={() => {
            if (drag.current && !drag.current.moved) drag.current = null;
          }}
          onLostPointerCapture={() => {
            drag.current = null;
            setPanning(false);
          }}
          onDoubleClick={() => setView({ ...IDENTITY })}
          style={{
            width: "100%",
            height: "auto",
            cursor: zoomable ? (panning ? "grabbing" : "grab") : "default",
            touchAction: "pan-y",
          }}
        >
          <title>{title}</title>
          <defs>
            <clipPath id={clipId}>
              <rect
                x={
                  layout.shape === "points" && isIdentity(view)
                    ? PAD.left - 26
                    : PAD.left
                }
                y={0}
                width={
                  PLOT.width +
                  (layout.shape === "points" && isIdentity(view) ? 50 : 0)
                }
                height={
                  PAD.top + PLOT.height + (layout.shape === "points" ? 26 : 0)
                }
              />
            </clipPath>
          </defs>

          {/* A horizontal ranking puts its category names where the eye starts,
            one per row, rather than rotating them under the axis. */}
          {cartesian &&
            horizontalBars &&
            layout.x.categories.map((name, index) => (
              <text
                key={`row-${name}`}
                x={leftPad - 10}
                y={PAD.top + (index + 0.5) * rowHeight + 4}
                textAnchor="end"
                fontSize="11"
                fill={TEXT}
              >
                {name.length > 30 ? `${name.slice(0, 29)}…` : name}
              </text>
            ))}

          {/* Horizontal gridlines and their values, or vertical ones when the
            bars run left-to-right. */}
          {cartesian &&
            horizontalBars &&
            layout.y.ticks.map((tick) => (
              <g key={`h-${tick.at}`}>
                <line
                  x1={hx(tick.at)}
                  x2={hx(tick.at)}
                  y1={PAD.top}
                  y2={PAD.top + PLOT.height}
                  stroke={GRID}
                />
                <text
                  x={hx(tick.at)}
                  y={PAD.top + PLOT.height + 17}
                  fontSize="11"
                  textAnchor="middle"
                  fill={TEXT}
                >
                  {tick.label}
                </text>
              </g>
            ))}
          {cartesian &&
            !horizontalBars &&
            layout.y.kind === "number" &&
            layout.y.min < 0 &&
            layout.y.max > 0 && (
              <line
                x1={PAD.left}
                x2={WIDTH - PAD.right}
                y1={yAt(0)}
                y2={yAt(0)}
                stroke={AXIS}
                strokeWidth={1.2}
              />
            )}
          {cartesian &&
            horizontalBars &&
            layout.y.min < 0 &&
            layout.y.max > 0 && (
              <line
                x1={hx(0)}
                x2={hx(0)}
                y1={PAD.top}
                y2={PAD.top + PLOT.height}
                stroke={AXIS}
                strokeWidth={1.2}
              />
            )}
          {cartesian &&
            !horizontalBars &&
            layout.y.kind === "number" &&
            layout.y.ticks.map((tick) => (
              <g key={tick.at}>
                <line
                  x1={PAD.left}
                  x2={WIDTH - PAD.right}
                  y1={yAt(tick.at)}
                  y2={yAt(tick.at)}
                  stroke={GRID}
                />
                <text
                  x={PAD.left - 8}
                  y={yAt(tick.at) + 4}
                  fontSize="11"
                  textAnchor="end"
                  fill={TEXT}
                >
                  {tick.label}
                </text>
              </g>
            ))}
          {cartesian &&
            layout.y.kind === "category" &&
            layout.y.categories.map((name, index) => (
              <text
                key={name}
                x={PAD.left - 8}
                y={
                  PAD.top +
                  ((index + 0.5) / layout.y.categories.length) * PLOT.height +
                  4
                }
                fontSize="11"
                textAnchor="end"
                fill={TEXT}
              >
                {name.length > 14 ? `${name.slice(0, 13)}…` : name}
              </text>
            ))}

          {/* The clip stays untransformed: it is exactly the viewport, and the
            marks inside it carry the view. */}
          <g clipPath={`url(#${clipId})`}>
            <g transform={plotShift}>
              {layout.shape === "bars" &&
                layout.series.map((series, seriesIndex) =>
                  series.marks.map((mark) => {
                    if (horizontalBars) {
                      const inset = rowHeight * 0.09;
                      const slot = (rowHeight - 2 * inset) / groupCount;
                      const index = Math.max(0, Math.round(mark.x - 0.5));
                      const top =
                        PAD.top +
                        (index + 0.5) * rowHeight -
                        rowHeight / 2 +
                        inset +
                        seriesIndex * slot;
                      const left = Math.min(hx(0), hx(mark.y));
                      const width = Math.max(1, Math.abs(hx(mark.y) - hx(0)));
                      return (
                        <g
                          key={mark.key}
                          opacity={dim(mark.label, mark.group, mark.key)}
                          onMouseEnter={() => setHovered(mark)}
                          style={{ cursor: "pointer" }}
                        >
                          <rect
                            x={left}
                            y={top + 1}
                            width={width}
                            height={Math.max(1, slot - 2)}
                            rx={Math.min(4, slot / 4)}
                            fill={series.color}
                            fillOpacity={isHovered(mark) ? 1 : 0.88}
                            stroke={isHovered(mark) ? "#fff" : "none"}
                            strokeWidth={1.5}
                            {...markKeys(mark, () => pick(mark))}
                          />
                          {showBarValues &&
                            (() => {
                              const outside =
                                mark.y >= 0 ? left + width + 6 : left - 6;
                              const fits =
                                mark.y >= 0
                                  ? outside < WIDTH - PAD.right - 18
                                  : outside > leftPad + 18;
                              return (
                                <text
                                  x={
                                    fits
                                      ? outside
                                      : mark.y >= 0
                                        ? left + width - 6
                                        : left + 6
                                  }
                                  y={top + slot / 2 + 4}
                                  textAnchor={
                                    fits
                                      ? mark.y >= 0
                                        ? "start"
                                        : "end"
                                      : mark.y >= 0
                                        ? "end"
                                        : "start"
                                  }
                                  fontSize="10"
                                  fontWeight="600"
                                  fill={fits ? TEXT : "#fff"}
                                  pointerEvents="none"
                                >
                                  {formatNumber(mark.y)}
                                </text>
                              );
                            })()}
                        </g>
                      );
                    }
                    const slot = (bandWidth - 2 * bandInset) / groupCount;
                    const left =
                      xAt(mark.x) -
                      bandWidth / 2 +
                      bandInset +
                      seriesIndex * slot;
                    const top = Math.min(yAt(mark.y), yAt(0));
                    const height = Math.abs(yAt(mark.y) - yAt(0));
                    const mid = left + slot / 2;
                    return (
                      <g
                        key={mark.key}
                        opacity={dim(mark.label, mark.group, mark.key)}
                        onMouseEnter={() => setHovered(mark)}
                        style={{ cursor: "pointer" }}
                      >
                        <rect
                          x={left + 1}
                          y={top}
                          width={Math.max(1, slot - 2)}
                          height={Math.max(1, height)}
                          rx={Math.min(4, slot / 4)}
                          fill={series.color}
                          fillOpacity={isHovered(mark) ? 1 : 0.88}
                          stroke={isHovered(mark) ? "#fff" : "none"}
                          strokeWidth={1.5}
                          {...markKeys(mark, () => pick(mark))}
                        />
                        {showBarValues &&
                          height > 18 &&
                          (mark.y >= 0 ||
                            top + height < PAD.top + PLOT.height - 16) && (
                            <text
                              x={mid}
                              y={mark.y >= 0 ? top - 5 : top + height + 12}
                              textAnchor="middle"
                              fontSize="10"
                              fontWeight="600"
                              fill={TEXT}
                              pointerEvents="none"
                            >
                              {formatNumber(mark.y)}
                            </text>
                          )}
                      </g>
                    );
                  }),
                )}

              {layout.shape === "line" &&
                layout.series.map((series) => {
                  const points = series.marks
                    .map((mark) => `${xAt(mark.x)},${yAt(mark.y)}`)
                    .join(" ");
                  const baseline = yAt(0);
                  const stepPath = layout.step
                    ? `M ${xAt(layout.x.min)} ${baseline} ${series.marks.map((mark) => `H ${xAt(mark.x)} V ${yAt(mark.y)}`).join(" ")} H ${xAt(layout.x.max)}`
                    : null;
                  return (
                    <g key={series.group}>
                      {!layout.step &&
                        layout.series.length === 1 &&
                        series.marks.length > 1 && (
                          <polygon
                            points={`${points} ${xAt(series.marks[series.marks.length - 1].x)},${baseline} ${xAt(series.marks[0].x)},${baseline}`}
                            fill={series.color}
                            fillOpacity={0.1}
                            stroke="none"
                          />
                        )}
                      {stepPath ? (
                        <path
                          d={stepPath}
                          fill="none"
                          stroke={series.color}
                          strokeWidth="2.4"
                          opacity={selected ? 0.35 : 1}
                        />
                      ) : (
                        <polyline
                          points={points}
                          fill="none"
                          stroke={series.color}
                          strokeWidth="2.4"
                          strokeLinejoin="round"
                          strokeLinecap="round"
                          opacity={selected ? 0.35 : 1}
                        />
                      )}
                      {series.marks.map((mark) => (
                        <circle
                          key={mark.key}
                          cx={xAt(mark.x)}
                          cy={yAt(mark.y)}
                          r={isHovered(mark) ? 5 : 3.4}
                          fill={series.color}
                          stroke="#fff"
                          strokeWidth={1.4}
                          opacity={dim(mark.label, mark.group, mark.key)}
                          onMouseEnter={() => setHovered(mark)}
                          style={{ cursor: "pointer" }}
                          {...markKeys(mark, () => pick(mark))}
                        />
                      ))}
                    </g>
                  );
                })}

              {layout.shape === "points" &&
                layout.series.map((series) =>
                  series.marks.map((mark) => (
                    <circle
                      key={mark.key}
                      cx={xAt(mark.x)}
                      cy={yAt(mark.y)}
                      r={
                        mark.size === undefined
                          ? 4
                          : 3 + Math.sqrt(mark.size) * BUBBLE_MAX
                      }
                      fill={series.color}
                      fillOpacity={mark.size === undefined ? 0.8 : 0.42}
                      stroke={isHovered(mark) ? "#fff" : series.color}
                      strokeWidth={isHovered(mark) ? 2 : 1}
                      opacity={dim(mark.label, mark.group, mark.key)}
                      onMouseEnter={() => setHovered(mark)}
                      style={{ cursor: "pointer" }}
                      {...markKeys(mark, () => pick(mark))}
                    />
                  )),
                )}

              {layout.shape === "box" &&
                layout.boxes.map((box) => {
                  const centre =
                    PAD.left + (layout.boxes.indexOf(box) + 0.5) * bandWidth;
                  const width = Math.max(6, bandWidth * 0.5);
                  return (
                    <g
                      key={box.key}
                      opacity={dim(box.label, box.group, box.key)}
                      onMouseEnter={() => setHovered(box)}
                      style={{ cursor: "pointer" }}
                      {...markKeys(box, () => pick(box))}
                    >
                      <line
                        x1={centre}
                        x2={centre}
                        y1={yAt(box.max)}
                        y2={yAt(box.min)}
                        stroke={box.color}
                      />
                      <rect
                        x={centre - width / 2}
                        y={yAt(box.q3)}
                        width={width}
                        height={Math.max(1, yAt(box.q1) - yAt(box.q3))}
                        rx={2}
                        fill={box.color}
                        fillOpacity={isHovered(box) ? 0.52 : 0.32}
                        stroke={box.color}
                        strokeWidth={isHovered(box) ? 2 : 1}
                      />
                      <line
                        x1={centre - width / 2}
                        x2={centre + width / 2}
                        y1={yAt(box.median)}
                        y2={yAt(box.median)}
                        stroke={box.color}
                        strokeWidth="2"
                      />
                      {box.points
                        .filter(
                          (_, index) =>
                            index %
                              Math.max(1, Math.ceil(box.points.length / 60)) ===
                            0,
                        )
                        .map((value, index) => (
                          <circle
                            key={`${box.key}-${index}`}
                            cx={centre + (frac(index * 0.618) - 0.5) * width}
                            cy={yAt(value)}
                            r={1.7}
                            fill={box.color}
                            fillOpacity={0.55}
                            pointerEvents="none"
                          />
                        ))}
                    </g>
                  );
                })}

              {layout.shape === "heatmap" &&
                (() => {
                  const cellWidth =
                    PLOT.width / Math.max(1, layout.columns.length);
                  const cellHeight =
                    PLOT.height / Math.max(1, layout.rows.length);
                  return layout.cells.map((cell) => {
                    const column = layout.columns.indexOf(cell.column);
                    const row = layout.rows.indexOf(cell.row);
                    const color = cellColor(cell.value, colorMin, colorMax);
                    return (
                      <g
                        key={cell.key}
                        opacity={dim(cell.column, cell.row)}
                        onMouseEnter={() => setHovered(cell)}
                        style={{ cursor: "pointer" }}
                      >
                        <rect
                          x={PAD.left + column * cellWidth + 1}
                          y={PAD.top + row * cellHeight + 1}
                          width={Math.max(1, cellWidth - 2)}
                          height={Math.max(1, cellHeight - 2)}
                          rx={2}
                          fill={color.fill}
                          stroke={isHovered(cell) ? TEXT : "#d7e1e9"}
                          strokeWidth={isHovered(cell) ? 1.5 : 0.6}
                          {...markKeys(cell, () =>
                            pick({ label: cell.column, group: cell.row }),
                          )}
                        />
                        {cellWidth > 34 && cellHeight > 18 && (
                          <text
                            x={PAD.left + (column + 0.5) * cellWidth}
                            y={PAD.top + (row + 0.5) * cellHeight + 4}
                            textAnchor="middle"
                            fontSize="10"
                            fontWeight="600"
                            fill={color.ink}
                            pointerEvents="none"
                          >
                            {formatNumber(cell.value)}
                          </text>
                        )}
                      </g>
                    );
                  });
                })()}

              {layout.shape === "pie" &&
                layout.slices &&
                (() => {
                  const cx = PAD.left + PLOT.width / 2;
                  const cy = PAD.top + PLOT.height / 2;
                  const radius = Math.min(PLOT.width, PLOT.height) * 0.46;
                  return (
                    <g>
                      {layout.slices.map((slice) => {
                        const mid = (slice.startAngle + slice.endAngle) / 2;
                        const push = isHovered(slice) ? 5 : 0;
                        const label = percent(slice.share);
                        // A label on a sliver is noise: below roughly a
                        // twentieth of the circle, the legend and the hover
                        // carry the category instead.
                        const wide = slice.endAngle - slice.startAngle > 0.3;
                        return (
                          <g
                            key={slice.key}
                            opacity={dim(slice.label, slice.group, slice.key)}
                            onMouseEnter={() => setHovered(slice)}
                            style={{ cursor: "pointer" }}
                            {...markKeys(slice, () => pick(slice))}
                          >
                            <path
                              d={wedgePath(
                                cx,
                                cy,
                                radius,
                                slice.startAngle,
                                slice.endAngle,
                              )}
                              transform={`translate(${Math.cos(mid) * push} ${Math.sin(mid) * push})`}
                              fill={slice.color}
                              stroke="#fff"
                              strokeWidth={2}
                            />
                            {label && wide && (
                              <text
                                x={cx + Math.cos(mid) * radius * 0.66}
                                y={cy + Math.sin(mid) * radius * 0.66}
                                textAnchor="middle"
                                dominantBaseline="middle"
                                fontSize="11"
                                fontWeight="650"
                                fill="#fff"
                                pointerEvents="none"
                              >
                                {label}
                              </text>
                            )}
                          </g>
                        );
                      })}
                    </g>
                  );
                })()}

              {layout.shape === "sunburst" &&
                layout.arcs &&
                (() => {
                  const cx = PAD.left + PLOT.width / 2;
                  const cy = PAD.top + PLOT.height / 2;
                  const radius = Math.min(PLOT.width, PLOT.height) * 0.5;
                  return (
                    <g>
                      {layout.arcs.map((arc) => {
                        const mid = (arc.startAngle + arc.endAngle) / 2;
                        const span = arc.endAngle - arc.startAngle;
                        const labelAt = (arc.inner + arc.outer) / 2;
                        return (
                          <g
                            key={arc.key}
                            opacity={dim(arc.label, arc.group, arc.key)}
                            onMouseEnter={() => setHovered(arc)}
                            style={{ cursor: "pointer" }}
                            {...markKeys(arc, () => pick(arc))}
                          >
                            <path
                              d={ringPath(
                                cx,
                                cy,
                                arc.inner * radius,
                                arc.outer * radius,
                                arc.startAngle,
                                arc.endAngle,
                              )}
                              fill={arc.color}
                              stroke="#fff"
                              strokeWidth={1.5}
                            />
                            {span > 0.22 &&
                              (arc.outer - arc.inner) * radius > 24 && (
                                <text
                                  x={cx + Math.cos(mid) * labelAt * radius}
                                  y={cy + Math.sin(mid) * labelAt * radius}
                                  textAnchor="middle"
                                  dominantBaseline="middle"
                                  fontSize="9"
                                  fill="#fff"
                                  paintOrder="stroke"
                                  stroke="rgba(0,0,0,.28)"
                                  strokeWidth="2"
                                  pointerEvents="none"
                                >
                                  {arc.label.split(" · ").pop()}
                                </text>
                              )}
                          </g>
                        );
                      })}
                    </g>
                  );
                })()}

              {layout.shape === "treemap" &&
                layout.tiles &&
                (() => (
                  <g transform={`translate(${PAD.left} ${PAD.top})`}>
                    {layout.tiles.map((tile) => {
                      const width = tile.width * PLOT.width;
                      const height = tile.height * PLOT.height;
                      return (
                        <g
                          key={tile.key}
                          opacity={dim(tile.label, tile.group, tile.key)}
                          onMouseEnter={() => setHovered(tile)}
                          style={{ cursor: "pointer" }}
                          {...markKeys(tile, () => pick(tile))}
                        >
                          <rect
                            x={tile.x * PLOT.width + 1}
                            y={tile.y * PLOT.height + 1}
                            width={Math.max(1, width - 2)}
                            height={Math.max(1, height - 2)}
                            rx={2}
                            fill={tile.color}
                            stroke={isHovered(tile) ? "#fff" : "none"}
                            strokeWidth={2}
                          />
                          {width > 78 && height > 30 && (
                            <text
                              x={tile.x * PLOT.width + 7}
                              y={tile.y * PLOT.height + 17}
                              fontSize="10"
                              fontWeight="600"
                              fill="#fff"
                              paintOrder="stroke"
                              stroke="rgba(0,0,0,.32)"
                              strokeWidth="2"
                              pointerEvents="none"
                            >
                              {tile.label.split(" · ").pop()?.slice(0, 18)}
                            </text>
                          )}
                          {width > 78 && height > 46 && (
                            <text
                              x={tile.x * PLOT.width + 7}
                              y={tile.y * PLOT.height + 31}
                              fontSize="10"
                              fill="rgba(255,255,255,.9)"
                              paintOrder="stroke"
                              stroke="rgba(0,0,0,.32)"
                              strokeWidth="2"
                              pointerEvents="none"
                            >
                              {formatNumber(tile.value)}
                            </text>
                          )}
                        </g>
                      );
                    })}
                  </g>
                ))()}

              {layout.shape === "violin" &&
                layout.violins &&
                (() => {
                  const band =
                    PLOT.width / Math.max(1, layout.x.categories.length);
                  return layout.violins.map((violin) => {
                    const width = (band / Math.max(1, violin.slots)) * 0.44;
                    const centre =
                      PAD.left +
                      (violin.xIndex + (violin.slot + 0.5) / violin.slots) *
                        band;
                    const path = violin.density.length
                      ? [
                          ...violin.density.map(
                            (point) =>
                              `${centre + point.y * width},${yAt(point.x)}`,
                          ),
                          ...violin.density
                            .slice()
                            .reverse()
                            .map(
                              (point) =>
                                `${centre - point.y * width},${yAt(point.x)}`,
                            ),
                        ].join(" ")
                      : "";
                    const step = Math.max(
                      1,
                      Math.ceil(violin.values.length / 70),
                    );
                    return (
                      <g
                        key={violin.key}
                        opacity={dim(violin.label, violin.group, violin.key)}
                        onMouseEnter={() => setHovered(violin)}
                        style={{ cursor: "pointer" }}
                        {...markKeys(violin, () => pick(violin))}
                      >
                        {path && (
                          <polygon
                            points={path}
                            fill={violin.color}
                            fillOpacity={isHovered(violin) ? 0.4 : 0.24}
                            stroke={violin.color}
                            strokeWidth={1.5}
                          />
                        )}
                        <line
                          x1={centre}
                          x2={centre}
                          y1={yAt(violin.stats.max)}
                          y2={yAt(violin.stats.min)}
                          stroke={violin.color}
                          strokeWidth={1.2}
                        />
                        <rect
                          x={centre - width * 0.18}
                          y={yAt(violin.stats.q3)}
                          width={width * 0.36}
                          height={Math.max(
                            1,
                            yAt(violin.stats.q1) - yAt(violin.stats.q3),
                          )}
                          fill="#fff"
                          fillOpacity={0.72}
                          stroke={violin.color}
                        />
                        <line
                          x1={centre - width * 0.24}
                          x2={centre + width * 0.24}
                          y1={yAt(violin.stats.median)}
                          y2={yAt(violin.stats.median)}
                          stroke={violin.color}
                          strokeWidth={2.2}
                        />
                        {violin.values
                          .filter((_, index) => index % step === 0)
                          .map((value, index) => (
                            <circle
                              key={`${violin.key}-${index}`}
                              cx={centre + (frac(index * 0.618) - 0.5) * width}
                              cy={yAt(value)}
                              r={1.7}
                              fill={violin.color}
                              fillOpacity={0.55}
                              pointerEvents="none"
                            />
                          ))}
                      </g>
                    );
                  });
                })()}

              {layout.shape === "radar" &&
                layout.radar &&
                (() => {
                  const cx = PAD.left + PLOT.width / 2;
                  const cy = PAD.top + PLOT.height / 2;
                  const radius = Math.min(PLOT.width, PLOT.height) * 0.34;
                  const spokes = layout.radar.spokes;
                  const angleAt = (index: number) =>
                    TWELVE + (index * TAU) / Math.max(3, spokes.length);
                  const at = (index: number, value: number) => {
                    const span = layout.radar!.max - layout.radar!.min || 1;
                    const share = (value - layout.radar!.min) / span;
                    return polar(
                      cx,
                      cy,
                      radius * Math.max(0.04, share),
                      angleAt(index),
                    );
                  };
                  const ring = (share: number) =>
                    spokes
                      .map((_, index) => {
                        const point = polar(
                          cx,
                          cy,
                          radius * share,
                          angleAt(index),
                        );
                        return `${point.x},${point.y}`;
                      })
                      .join(" ");
                  return (
                    <g>
                      {[0.25, 0.5, 0.75, 1].map((share) => (
                        <polygon
                          key={share}
                          points={ring(share)}
                          fill="none"
                          stroke={GRID}
                          strokeWidth={1}
                        />
                      ))}
                      {spokes.map((spoke, index) => {
                        const end = polar(cx, cy, radius, angleAt(index));
                        const label = polar(
                          cx,
                          cy,
                          radius + 16,
                          angleAt(index),
                        );
                        return (
                          <g key={spoke}>
                            <line
                              x1={cx}
                              y1={cy}
                              x2={end.x}
                              y2={end.y}
                              stroke={GRID}
                            />
                            <text
                              x={label.x}
                              y={label.y}
                              textAnchor={
                                Math.abs(label.x - cx) < 8
                                  ? "middle"
                                  : label.x > cx
                                    ? "start"
                                    : "end"
                              }
                              dominantBaseline="middle"
                              fontSize="10"
                              fill={TEXT}
                            >
                              {spoke.length > 16
                                ? `${spoke.slice(0, 15)}…`
                                : spoke}
                            </text>
                          </g>
                        );
                      })}
                      {layout.radar.series.map((series) => {
                        const points = series.values.map((value, index) =>
                          at(index, value),
                        );
                        return (
                          <g key={series.group}>
                            <polygon
                              points={points
                                .map((point) => `${point.x},${point.y}`)
                                .join(" ")}
                              fill={series.color}
                              fillOpacity={0.12}
                              stroke={series.color}
                              strokeWidth={2}
                            />
                            {points.map((point, index) => {
                              const mark = layout.series.find(
                                (entry) => entry.group === series.group,
                              )?.marks[index];
                              if (!mark) return null;
                              return (
                                <circle
                                  key={mark.key}
                                  cx={point.x}
                                  cy={point.y}
                                  r={isHovered(mark) ? 5 : 3.2}
                                  fill={series.color}
                                  stroke="#fff"
                                  strokeWidth={1.5}
                                  opacity={dim(
                                    mark.label,
                                    mark.group,
                                    mark.key,
                                  )}
                                  onMouseEnter={() => setHovered(mark)}
                                  style={{ cursor: "pointer" }}
                                  {...markKeys(mark, () => pick(mark))}
                                />
                              );
                            })}
                          </g>
                        );
                      })}
                    </g>
                  );
                })()}

              {layout.shape === "waffle" &&
                layout.waffle &&
                (() => {
                  const size = Math.min(PLOT.width, PLOT.height) * 0.94;
                  const cell = size / 10;
                  const left = PAD.left + (PLOT.width - size) / 2;
                  const top = PAD.top + (PLOT.height - size) / 2;
                  return (
                    <g>
                      {layout.waffle!.squares.map((square) => (
                        <rect
                          key={square.key}
                          x={left + square.column * cell + 1}
                          y={top + square.row * cell + 1}
                          width={Math.max(1, cell - 2)}
                          height={Math.max(1, cell - 2)}
                          rx={2}
                          fill={square.color}
                          opacity={dim(square.label, square.group, square.key)}
                          stroke={isHovered(square) ? "#fff" : "none"}
                          strokeWidth={2}
                          onMouseEnter={() => setHovered(square)}
                          style={{ cursor: "pointer" }}
                          {...markKeys(square, () => pick(square))}
                        />
                      ))}
                    </g>
                  );
                })()}

              {layout.calendar &&
                (() => {
                  const { months, cells } = layout.calendar;
                  const labelWidth = 64;
                  const header = 24;
                  const cellWidth = (PLOT.width - labelWidth) / 31;
                  const cellHeight = (PLOT.height - header) / months.length;
                  const dayX = (day: number) =>
                    PAD.left + labelWidth + (day - 1) * cellWidth;
                  const monthY = (month: number) =>
                    PAD.top + header + month * cellHeight;
                  const labelEvery = Math.max(1, Math.ceil(15 / cellHeight));
                  return (
                    <g>
                      {[1, 5, 10, 15, 20, 25, 31].map((day) => (
                        <text
                          key={day}
                          x={dayX(day) + cellWidth / 2}
                          y={PAD.top + 13}
                          textAnchor="middle"
                          fontSize="12"
                          fill={TEXT}
                        >
                          {day}
                        </text>
                      ))}
                      {months
                        .filter((month) => month.index % labelEvery === 0)
                        .map((month) => (
                          <text
                            key={month.key}
                            x={PAD.left}
                            y={monthY(month.index) + cellHeight / 2 + 4}
                            fontSize="12"
                            fill={TEXT}
                          >
                            {month.label}
                          </text>
                        ))}
                      {cells.map((cell) => (
                        <rect
                          key={cell.key}
                          opacity={dim(cell.label, cell.group, cell.key)}
                          onMouseEnter={() => setHovered(cell)}
                          style={{ cursor: "pointer" }}
                          {...markKeys(cell, () => pick(cell))}
                          x={dayX(cell.day) + 1}
                          y={monthY(cell.month) + Math.min(1, cellHeight / 4)}
                          width={cellWidth - 2}
                          height={cellHeight - Math.min(2, cellHeight / 2)}
                          rx={Math.min(2, cellHeight / 4)}
                          fill={cellColor(cell.value, colorMin, colorMax).fill}
                          stroke={isHovered(cell) ? TEXT : "#d7e1e9"}
                          strokeWidth={isHovered(cell) ? 1.5 : 0.6}
                        />
                      ))}
                    </g>
                  );
                })()}

              {layout.shape === "sankey" &&
                layout.flows &&
                (() => {
                  const left = PAD.left + 78;
                  const right = WIDTH - PAD.right - 78;
                  const top = PAD.top + 28;
                  const height = PLOT.height - 36;
                  const nodeWidth = 11;
                  const y = (value: number) => top + value * height;
                  return (
                    <g>
                      <text
                        x={left}
                        y={PAD.top + 10}
                        fontSize="10"
                        fill={TEXT}
                        textAnchor="middle"
                      >
                        {layout.x.label}
                      </text>
                      <text
                        x={right}
                        y={PAD.top + 10}
                        fontSize="10"
                        fill={TEXT}
                        textAnchor="middle"
                      >
                        {layout.y.label}
                      </text>
                      {layout.flows!.links.map((link) => {
                        const sy0 = y(link.sy0);
                        const sy1 = y(link.sy1);
                        const ty0 = y(link.ty0);
                        const ty1 = y(link.ty1);
                        const mid = (left + nodeWidth + right) / 2;
                        const path = `M ${left + nodeWidth} ${sy0} C ${mid} ${sy0} ${mid} ${ty0} ${right} ${ty0} L ${right} ${ty1} C ${mid} ${ty1} ${mid} ${sy1} ${left + nodeWidth} ${sy1} Z`;
                        return (
                          <path
                            key={link.key}
                            d={path}
                            fill={link.color}
                            fillOpacity={isHovered(link) ? 0.5 : 0.24}
                            stroke="none"
                            opacity={dim(link.label, link.group, link.key)}
                            onMouseEnter={() => setHovered(link)}
                            style={{ cursor: "pointer" }}
                            {...markKeys(link, () => pick(link))}
                          />
                        );
                      })}
                      {layout.flows!.nodes.map((node) => {
                        const isSource = node.side === "source";
                        return (
                          <g key={node.key}>
                            <rect
                              x={isSource ? left : right}
                              y={y(node.y0)}
                              width={nodeWidth}
                              height={Math.max(1, y(node.y1) - y(node.y0))}
                              rx={2}
                              fill={node.color}
                            />
                            <text
                              x={isSource ? left - 7 : right + nodeWidth + 7}
                              y={(y(node.y0) + y(node.y1)) / 2 + 3}
                              textAnchor={isSource ? "end" : "start"}
                              fontSize="10"
                              fill={TEXT}
                            >
                              {node.label.length > 22
                                ? `${node.label.slice(0, 21)}…`
                                : node.label}
                            </text>
                          </g>
                        );
                      })}
                    </g>
                  );
                })()}
            </g>
          </g>

          {/* Axes last, so marks never paint over them. */}
          {cartesian && (
            <line
              x1={leftPad}
              x2={WIDTH - PAD.right}
              y1={PAD.top + PLOT.height}
              y2={PAD.top + PLOT.height}
              stroke={AXIS}
            />
          )}
          {layout.heatmapMeasure && (
            <ColorKey
              x={WIDTH - PAD.right - 112}
              y={HEIGHT - 27}
              label={layout.heatmapMeasure}
              min={colorMin}
              max={colorMax}
            />
          )}
          {layout.calendar && (
            <ColorKey
              x={254}
              y={HEIGHT - 35}
              label={layout.y.label}
              min={colorMin}
              max={colorMax}
            />
          )}
          {cartesian && !horizontalBars && (
            <line
              x1={PAD.left}
              x2={PAD.left}
              y1={PAD.top}
              y2={PAD.top + PLOT.height}
              stroke={AXIS}
            />
          )}
          {cartesian && (
            <g>
              {!horizontalBars &&
                (() => {
                  const ticks =
                    layout.shape === "heatmap"
                      ? layout.columns.map((label, index) => ({
                          label,
                          at: index + 0.5,
                        }))
                      : ticksToShow(layout);
                  const visible = ticks.flatMap((tick) => {
                    const position =
                      PAD.left + panPx + (xAt(tick.at) - PAD.left) * view.scale;
                    return position >= PAD.left - 0.01 &&
                      position <= WIDTH - PAD.right + 0.01
                      ? [{ ...tick, position }]
                      : [];
                  });
                  const slot = PLOT.width / Math.max(1, visible.length);
                  const rotate =
                    layout.x.kind === "category" &&
                    visible.some(
                      (tick) => textWidth(tick.label, 12) + 12 > slot,
                    );
                  const labels =
                    layout.shape === "heatmap" && visible.length > 24
                      ? visible.filter(
                          (_, index) =>
                            index % Math.ceil(visible.length / 24) === 0,
                        )
                      : visible;
                  return labels.map(({ at, label, position }) => {
                    const y = PAD.top + PLOT.height + 18;
                    const shown = fitText(
                      label,
                      rotate ? 86 : Math.min(170, slot - 8),
                      12,
                    );
                    return (
                      <text
                        key={at}
                        x={position}
                        y={y}
                        fontSize="12"
                        textAnchor={rotate ? "end" : "middle"}
                        fill={TEXT}
                        transform={
                          rotate ? `rotate(-32 ${position} ${y})` : undefined
                        }
                      >
                        {shown !== label && <title>{label}</title>}
                        {shown}
                      </text>
                    );
                  });
                })()}
            </g>
          )}
          {cartesian && (
            <text
              x={WIDTH / 2}
              y={HEIGHT - 6}
              fontSize="11"
              textAnchor="middle"
              fill={TEXT}
            >
              {horizontalBars ? layout.y.label : layout.x.label}
            </text>
          )}
          {cartesian && !horizontalBars && (
            <text
              x={14}
              y={PAD.top + PLOT.height / 2}
              fontSize="11"
              textAnchor="middle"
              fill={TEXT}
              transform={`rotate(-90 14 ${PAD.top + PLOT.height / 2})`}
            >
              {layout.y.label}
            </text>
          )}
        </svg>
      </div>
      {layout.series.length > 1 && (
        <ul className="chart-legend">
          {legendItems.map((series) => (
            <li key={series.group}>
              <span style={{ background: series.color }} aria-hidden="true" />
              {series.group}
              {layout.shape === "pie" && series.marks[0] && (
                <small> {formatNumber(series.marks[0].y)}</small>
              )}
            </li>
          ))}
          {legendRest > 0 && (
            <li className="chart-legend-note">
              and {legendRest} more — hover a mark to read it
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

/** What the pointer (or the keyboard's focus) is over, in words. */
function describe(item: HoverItem | null, layout: Layout): string | null {
  if (!item) return null;
  if (
    layout.step &&
    "x" in item &&
    "y" in item &&
    "group" in item &&
    "n" in item
  ) {
    const group = item.group === "(all)" ? "" : `${item.group} · `;
    return `${group}${formatNumber(item.y)}% at or below ${formatNumber(item.x)} (${item.n} ${item.n === 1 ? "row" : "rows"})`;
  }
  if ("stats" in item) {
    return `${item.label}${item.group && item.group !== item.label ? ` · ${item.group}` : ""} — median ${formatNumber(
      item.stats.median,
    )}, middle half ${formatNumber(item.stats.q1)} to ${formatNumber(
      item.stats.q3,
    )}, ${item.n} observation${item.n === 1 ? "" : "s"}`;
  }
  if ("median" in item) {
    return `${item.label} — median ${formatNumber(item.median)}, middle half ${formatNumber(
      item.q1,
    )} to ${formatNumber(item.q3)}, ${item.n} row${item.n === 1 ? "" : "s"}`;
  }
  if ("source" in item && "target" in item) {
    return `${item.label} → ${item.group} — ${formatNumber(item.value)}`;
  }
  if ("share" in item) {
    const share = `${(item.share * 100).toFixed(item.share >= 0.1 ? 0 : 1)}%`;
    if ("value" in item)
      return `${item.label} — ${formatNumber(item.value)} (${share})`;
    return `${item.label} — ${share} of the whole`;
  }
  if ("day" in item) {
    return `${item.label} — ${formatNumber(item.value)}${item.n > 1 ? ` (${item.n} rows)` : ""}`;
  }
  if ("column" in item) {
    return `${item.row} × ${item.column} — ${formatNumber(item.value)} (${item.n} row${item.n === 1 ? "" : "s"})`;
  }
  const group = item.group && item.group !== "(all)" ? ` · ${item.group}` : "";
  const rows = item.n > 1 ? ` (${item.n} rows)` : "";
  return `${item.label || "(blank)"}${group} — ${layout.y.label}: ${formatNumber(item.y)}${rows}`;
}
