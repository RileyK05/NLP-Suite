import { useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import { ArrowDownToLine, RotateCcw } from "lucide-react";
import type { Box, Cell, Layout, Mark } from "./chartLayout";
import { formatNumber } from "./chartLayout";
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
const HEIGHT = 380;
const PAD = { top: 18, right: 18, bottom: 64, left: 72 };
const PLOT = {
  width: WIDTH - PAD.left - PAD.right,
  height: HEIGHT - PAD.top - PAD.bottom,
};
const AXIS = "#b3bdb2";
const GRID = "#e8ece5";
const TEXT = "#5d6a5c";
const BUBBLE_MAX = 22;

export type Selection = { label: string; group: string } | null;

function ticksToShow(layout: Layout): { at: number; label: string }[] {
  const ticks = layout.x.ticks;
  if (layout.x.kind !== "category") return ticks;
  // Category labels overlap once there are more than a couple of dozen; thin
  // them evenly rather than letting them collide into an unreadable band.
  const every = Math.ceil(ticks.length / 24);
  return every <= 1 ? ticks : ticks.filter((_, index) => index % every === 0);
}

export function ChartCanvas({
  layout,
  selected,
  onSelect,
  title,
  name,
}: {
  layout: Layout;
  selected: Selection;
  onSelect: (selection: Selection) => void;
  title: string;
  /** Filename base for downloads; defaults to the title. */
  name?: string;
}) {
  const [hovered, setHovered] = useState<Mark | Box | Cell | null>(null);
  const [view, setView] = useState<ViewTransform>(IDENTITY);
  const [panning, setPanning] = useState(false);
  const [focusIndex, setFocusIndex] = useState(0);
  const [exportError, setExportError] = useState("");
  const svgRef = useRef<SVGSVGElement>(null);
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
  }, []);

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

  const isSelected = (label: string, group: string) =>
    !selected || (selected.label === label && selected.group === group);
  const dim = (label: string, group: string) =>
    isSelected(label, group) ? 1 : 0.25;

  const bandWidth =
    layout.x.kind === "category"
      ? PLOT.width / Math.max(1, layout.x.categories.length)
      : 0;
  const groupCount = Math.max(1, layout.series.length);

  const pick = (mark: { label: string; group: string }) => {
    if (suppressClick.current) return;
    onSelect(
      selected && selected.label === mark.label && selected.group === mark.group
        ? null
        : mark,
    );
  };

  // Marks in draw order, counted before rendering so the roving tab stop can
  // survive the mark list shrinking (a re-layout with fewer marks must not
  // leave the tab index pointing past the end of the list).
  const markTotal =
    layout.shape === "box"
      ? layout.boxes.length
      : layout.shape === "heatmap"
        ? layout.cells.length
        : layout.series.reduce(
            (total, series) => total + series.marks.length,
            0,
          );
  const activeMark = Math.min(focusIndex, Math.max(0, markTotal - 1));

  /**
   * The keyboard half of a mark: roving tab stop, arrows to move, Enter to
   * pick — the same selection a click produces, so pointing and tabbing never
   * disagree about what a mark means.
   */
  let markOrdinal = 0;
  const markKeys = (item: Mark | Box | Cell, activate: () => void) => {
    const index = markOrdinal++;
    return {
      tabIndex: index === activeMark ? 0 : -1,
      role: "button" as const,
      "aria-label": describe(item, layout) ?? "",
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

  const svgMarkup = () => {
    const svg = svgRef.current;
    if (!svg) throw new Error("The chart is not on the page yet.");
    // Standalone file: pin the size and namespace the inline SVG inherits
    // from the document but would not carry into a downloaded file.
    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width", String(WIDTH));
    clone.setAttribute("height", String(HEIGHT));
    return new XMLSerializer().serializeToString(clone);
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
        new Blob([svgMarkup()], { type: "image/svg+xml" }),
        `${fileBase}.svg`,
      );
    } catch (error) {
      setExportError(error instanceof Error ? error.message : String(error));
    }
  };

  const downloadPng = () => {
    try {
      setExportError("");
      const url = URL.createObjectURL(
        new Blob([svgMarkup()], { type: "image/svg+xml" }),
      );
      const image = new Image();
      image.onload = () => {
        // 2x the drawing size so the PNG stays crisp on high-density screens.
        const canvas = document.createElement("canvas");
        canvas.width = WIDTH * 2;
        canvas.height = HEIGHT * 2;
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

  return (
    <div className="chart-canvas">
      <div className="chart-tools">
        {!isIdentity(view) && (
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
      {exportError && (
        <p className="alert warning" role="status">
          {exportError}
        </p>
      )}
      <p className="chart-readout" role="status">
        {readout ?? (
          <span className="muted">
            Point to read a value, click to filter the table, wheel to zoom,
            drag to pan.
          </span>
        )}
      </p>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={title}
        onMouseLeave={() => setHovered(null)}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
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
          cursor: panning ? "grabbing" : "grab",
          touchAction: "pan-y",
        }}
      >
        <title>{title}</title>
        <defs>
          <clipPath id={clipId}>
            <rect
              x={PAD.left}
              y={PAD.top}
              width={PLOT.width}
              height={PLOT.height}
            />
          </clipPath>
        </defs>

        {/* Horizontal gridlines and their values. */}
        {layout.y.kind === "number" &&
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
        {layout.y.kind === "category" &&
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
                  const slot = bandWidth / groupCount;
                  const left = xAt(mark.x) - bandWidth / 2 + seriesIndex * slot;
                  const top = Math.min(yAt(mark.y), yAt(0));
                  const height = Math.abs(yAt(mark.y) - yAt(0));
                  return (
                    <rect
                      key={mark.key}
                      x={left + 1}
                      y={top}
                      width={Math.max(1, slot - 2)}
                      height={Math.max(1, height)}
                      fill={series.color}
                      opacity={dim(mark.label, mark.group)}
                      onMouseEnter={() => setHovered(mark)}
                      style={{ cursor: "pointer" }}
                      {...markKeys(mark, () => pick(mark))}
                    />
                  );
                }),
              )}

            {layout.shape === "line" &&
              layout.series.map((series) => (
                <g key={series.group}>
                  <polyline
                    points={series.marks
                      .map((mark) => `${xAt(mark.x)},${yAt(mark.y)}`)
                      .join(" ")}
                    fill="none"
                    stroke={series.color}
                    strokeWidth="2"
                    opacity={selected ? 0.35 : 1}
                  />
                  {series.marks.map((mark) => (
                    <circle
                      key={mark.key}
                      cx={xAt(mark.x)}
                      cy={yAt(mark.y)}
                      r={4}
                      fill={series.color}
                      opacity={dim(mark.label, mark.group)}
                      onMouseEnter={() => setHovered(mark)}
                      style={{ cursor: "pointer" }}
                      {...markKeys(mark, () => pick(mark))}
                    />
                  ))}
                </g>
              ))}

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
                    fillOpacity={mark.size === undefined ? 0.75 : 0.5}
                    stroke={series.color}
                    opacity={dim(mark.label, mark.group)}
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
                    opacity={dim(box.label, box.group)}
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
                      fill={box.color}
                      fillOpacity={0.35}
                      stroke={box.color}
                    />
                    <line
                      x1={centre - width / 2}
                      x2={centre + width / 2}
                      y1={yAt(box.median)}
                      y2={yAt(box.median)}
                      stroke={box.color}
                      strokeWidth="2"
                    />
                  </g>
                );
              })}

            {layout.shape === "heatmap" &&
              (() => {
                const high = Math.max(
                  ...layout.cells.map((cell) => Math.abs(cell.value)),
                  1,
                );
                const cellWidth =
                  PLOT.width / Math.max(1, layout.columns.length);
                const cellHeight =
                  PLOT.height / Math.max(1, layout.rows.length);
                return layout.cells.map((cell) => {
                  const column = layout.columns.indexOf(cell.column);
                  const row = layout.rows.indexOf(cell.row);
                  return (
                    <rect
                      key={cell.key}
                      x={PAD.left + column * cellWidth}
                      y={PAD.top + row * cellHeight}
                      width={Math.max(1, cellWidth - 1)}
                      height={Math.max(1, cellHeight - 1)}
                      fill={OKABE_BLUE}
                      fillOpacity={0.12 + 0.88 * (Math.abs(cell.value) / high)}
                      opacity={dim(cell.column, cell.row)}
                      onMouseEnter={() => setHovered(cell)}
                      style={{ cursor: "pointer" }}
                      {...markKeys(cell, () =>
                        pick({ label: cell.column, group: cell.row }),
                      )}
                    />
                  );
                });
              })()}
          </g>
        </g>

        {/* Axes last, so marks never paint over them. */}
        <line
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={PAD.top + PLOT.height}
          y2={PAD.top + PLOT.height}
          stroke={AXIS}
        />
        <line
          x1={PAD.left}
          x2={PAD.left}
          y1={PAD.top}
          y2={PAD.top + PLOT.height}
          stroke={AXIS}
        />
        <g transform={plotShift}>
          {layout.shape !== "heatmap" &&
            ticksToShow(layout).map((tick, index) => (
              <text
                key={`${tick.label}-${index}`}
                x={
                  layout.x.kind === "category"
                    ? PAD.left + tick.at * bandWidth
                    : xAt(tick.at)
                }
                y={PAD.top + PLOT.height + 16}
                fontSize="11"
                textAnchor="end"
                fill={TEXT}
                transform={`rotate(-35 ${
                  layout.x.kind === "category"
                    ? PAD.left + tick.at * bandWidth
                    : xAt(tick.at)
                } ${PAD.top + PLOT.height + 16})`}
              >
                {tick.label.length > 18
                  ? `${tick.label.slice(0, 17)}…`
                  : tick.label}
              </text>
            ))}
          {layout.shape === "heatmap" &&
            layout.columns.map((name, index) => {
              const at =
                PAD.left + ((index + 0.5) / layout.columns.length) * PLOT.width;
              return (
                <text
                  key={name}
                  x={at}
                  y={PAD.top + PLOT.height + 16}
                  fontSize="11"
                  textAnchor="end"
                  fill={TEXT}
                  transform={`rotate(-35 ${at} ${PAD.top + PLOT.height + 16})`}
                >
                  {name.length > 18 ? `${name.slice(0, 17)}…` : name}
                </text>
              );
            })}
        </g>
        <text
          x={WIDTH / 2}
          y={HEIGHT - 6}
          fontSize="11"
          textAnchor="middle"
          fill={TEXT}
        >
          {layout.x.label}
        </text>
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
      </svg>
      {layout.series.length > 1 && (
        <ul className="chart-legend">
          {layout.series.map((series) => (
            <li key={series.group}>
              <span style={{ background: series.color }} aria-hidden="true" />
              {series.group}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const OKABE_BLUE = "#0072B2";

/** What the pointer (or the keyboard's focus) is over, in words. */
function describe(
  item: Mark | Box | Cell | null,
  layout: Layout,
): string | null {
  if (!item) return null;
  if ("median" in item) {
    return `${item.label} — median ${formatNumber(item.median)}, middle half ${formatNumber(
      item.q1,
    )} to ${formatNumber(item.q3)}, ${item.n} row${item.n === 1 ? "" : "s"}`;
  }
  if ("column" in item) {
    return `${item.row} × ${item.column} — ${formatNumber(item.value)} (${item.n} row${item.n === 1 ? "" : "s"})`;
  }
  const group = item.group && item.group !== "(all)" ? ` · ${item.group}` : "";
  const rows = item.n > 1 ? ` (${item.n} rows)` : "";
  return `${item.label || "(blank)"}${group} — ${layout.y.label}: ${formatNumber(item.y)}${rows}`;
}
