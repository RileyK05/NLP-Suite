/**
 * Axis ticks, label widths and label placement for the panel figures.
 *
 * Pure functions, no DOM, so every rule a reader sees on an axis is pinned by
 * `panelTicks.test.ts`. Each one exists because a real figure broke without
 * it (docs/FIGURE_QUALITY_PLAN.md, classes C1, C3 and C4):
 *
 * * **No "-0" tick.** `Math.round(-0.4) * step` is negative zero, and
 *   `(-0).toLocaleString()` prints "-0"; floating-point stepping also leaves
 *   residue like 1.3e-17 at zero. Both read as a negative result.
 * * **One axis, one number of decimals.** Formatting each tick alone printed
 *   "1.50" next to "2" on the co-occurrence axis.
 * * **Label room from the labels.** A fixed 84px margin and a 15-character
 *   cut turned every entity name and n-gram into "…".
 * * **Labels that do not overlap.** Centring every label above its point
 *   printed "income · private · sector" through "high · priority".
 */

/** Ticks at round numbers between `min` and `max`, zero exactly zero. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [clean(min, Math.abs(min) || 1)];
  const step = niceStep(min, max, count);
  const ticks: number[] = [];
  for (
    let at = Math.ceil(min / step - 1e-9) * step;
    at <= max + step / 1000;
    at += step
  ) {
    ticks.push(clean(Math.round(at / step) * step, step));
  }
  return ticks;
}

function niceStep(min: number, max: number, count: number): number {
  const rough = (max - min) / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  return (
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough) ??
    magnitude * 10
  );
}

/** A value within a billionth of a step of zero is zero, and positive. */
function clean(value: number, step: number): number {
  if (Math.abs(value) < Math.abs(step) * 1e-9) return 0;
  // Trim binary residue (0.30000000000000004) to the step's precision.
  const digits = Math.min(12, Math.max(0, decimalsOf(step) + 2));
  const trimmed = Number(value.toFixed(digits));
  return trimmed === 0 ? 0 : trimmed;
}

/** The fewest decimals that write `value` exactly (to 1e-6 of a unit). */
function decimalsOf(value: number): number {
  for (let digits = 0; digits <= 8; digits++) {
    const scaled = Math.abs(value) * 10 ** digits;
    if (Math.abs(scaled - Math.round(scaled)) < 1e-6 * Math.max(1, scaled))
      return digits;
  }
  return 8;
}

/**
 * Labels for one axis's ticks, all at the same precision.
 *
 * `plain` keeps whole numbers ungrouped (a year axis reads 1990, never
 * "1,990"). Large values share one suffix: an axis reaching a million reads
 * 0, 0.5M, 1M rather than mixing "500k" and "1.0M".
 */
export function tickLabels(
  ticks: number[],
  options: { plain?: boolean } = {},
): string[] {
  if (!ticks.length) return [];
  const largest = Math.max(...ticks.map((tick) => Math.abs(tick)));
  const step =
    ticks.length > 1
      ? Math.min(
          ...ticks.slice(1).map((tick, index) => Math.abs(tick - ticks[index])),
        )
      : Math.abs(ticks[0]) || 1;
  if (options.plain) {
    const digits = decimalsOf(step);
    return ticks.map((tick) => positiveZero(tick).toFixed(digits));
  }
  const [divisor, suffix] =
    largest >= 1e9
      ? [1e9, "B"]
      : largest >= 1e6
        ? [1e6, "M"]
        : largest >= 1e4
          ? [1e3, "k"]
          : [1, ""];
  const digits = Math.min(6, decimalsOf(step / divisor));
  return ticks.map((tick) => {
    const value = positiveZero(tick / divisor);
    if (value === 0) return "0";
    const text =
      divisor === 1 && digits === 0
        ? value.toLocaleString("en-US", { maximumFractionDigits: 0 })
        : value.toLocaleString("en-US", {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
          });
    return `${text}${suffix}`;
  });
}

function positiveZero(value: number): number {
  return Object.is(value, -0) || Math.abs(value) < 1e-12 ? 0 : value;
}

/** `[min, max]` widened to the nearest round ticks outside it, so a bar
 *  axis ends on a labelled value and the longest bar is read against one. */
export function niceDomain(
  min: number,
  max: number,
  count = 5,
): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max)
    return [min, max];
  const step = niceStep(min, max, count);
  return [
    clean(Math.floor(min / step + 1e-9) * step, step),
    clean(Math.ceil(max / step - 1e-9) * step, step),
  ];
}

/**
 * Ticks for an axis that holds log10 of a quantity: round quantities
 * (1, 2, 5 × 10^k; only powers of ten past two decades) placed at their log,
 * labelled with the quantity. `core/viz/static/shapes.log_tick_values` is
 * the same rule.
 */
export function logTicks(
  low: number,
  high: number,
): { value: number; label: string }[] {
  const values: number[] = [];
  const steps = high - low > 2 ? [1] : [1, 2, 5];
  for (let power = Math.floor(low) - 1; power <= Math.ceil(high); power++) {
    for (const step of steps) {
      const quantity = step * 10 ** power;
      const at = Math.log10(quantity);
      if (at >= low - 1e-9 && at <= high + 1e-9) values.push(quantity);
    }
  }
  const labels = tickLabelsForValues(values);
  return values.map((quantity, index) => ({
    value: Math.log10(quantity),
    label: labels[index],
  }));
}

/** Labels for irregular values (1, 2, 5, 10 ...): each written plainly,
 *  thousands grouped, one suffix for the axis. */
function tickLabelsForValues(values: number[]): string[] {
  const largest = Math.max(0, ...values.map((value) => Math.abs(value)));
  const [divisor, suffix] =
    largest >= 1e6 ? [1e6, "M"] : largest >= 1e4 ? [1e3, "k"] : [1, ""];
  return values.map((value) => {
    const scaled = value / divisor;
    return `${scaled.toLocaleString("en-US", { maximumFractionDigits: 2 })}${suffix}`;
  });
}

/** Ticks and their labels together, the way every figure wants them. */
export function axisTicks(
  min: number,
  max: number,
  count = 5,
  options: { plain?: boolean } = {},
): { value: number; label: string }[] {
  const ticks = niceTicks(min, max, count);
  const labels = tickLabels(ticks, options);
  return ticks.map((value, index) => ({ value, label: labels[index] }));
}

// ------------------------------------------------------------ label widths --

/** Approximate advance widths, in em, of the app's sans-serif at small
 *  sizes. An estimate: good to a few percent over ordinary words, which is
 *  what sizing a margin needs. */
function charEm(char: string): number {
  if (" ,.;:'|!ijlIft()[]".includes(char)) return 0.3;
  if ("mwMW@%".includes(char)) return 0.86;
  if (char >= "A" && char <= "Z") return 0.66;
  if (char >= "0" && char <= "9") return 0.56;
  if (char === "…" || char === "—") return 0.9;
  return 0.54;
}

/** The estimated width of `text` in px at `fontPx`. */
export function textWidth(text: string, fontPx = 10): number {
  let em = 0;
  for (const char of text) em += charEm(char);
  return em * fontPx;
}

/** `text` cut with "…" to fit `maxPx`, or whole when it fits. */
export function fitText(text: string, maxPx: number, fontPx = 10): string {
  if (textWidth(text, fontPx) <= maxPx) return text;
  const room = maxPx - textWidth("…", fontPx);
  let width = 0;
  let cut = 0;
  for (const char of text) {
    const next = width + charEm(char) * fontPx;
    if (next > room) break;
    width = next;
    cut += char.length;
  }
  return `${text.slice(0, cut).trimEnd()}…`;
}

/**
 * The left margin a set of row labels needs: the widest label plus a gap,
 * no less than `floor`, no more than `share` of the canvas width. Labels
 * wider than that are cut to fit by `fitText`, with the full text kept in
 * the mark's evidence.
 */
export function labelMargin(
  labels: string[],
  width: number,
  { floor = 84, share = 0.38, fontPx = 10, gap = 14 } = {},
): number {
  const widest = Math.max(
    0,
    ...labels.map((label) => textWidth(label, fontPx)),
  );
  return Math.round(Math.min(width * share, Math.max(floor, widest + gap)));
}

// --------------------------------------------------------- label placement --

export type LabelBox = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

export type LabelRequest = {
  /** The point's position in px. */
  x: number;
  y: number;
  /** The point's drawn radius, so a label clears its own mark. */
  radius: number;
  text: string;
};

export type PlacedLabel = {
  index: number;
  text: string;
  /** The text's anchor point and alignment, as SVG draws it. */
  x: number;
  y: number;
  anchor: "start" | "middle" | "end";
  box: LabelBox;
};

function intersects(a: LabelBox, b: LabelBox): boolean {
  return (
    a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom
  );
}

/**
 * Collision-free labels, placed greedily in the order given.
 *
 * The caller orders requests by importance (largest, most significant
 * first). Each label tries eight positions around its point — above, below,
 * right, left, then the diagonals — and takes the first that stays inside
 * `bounds` and meets no label already placed; among those that meet no
 * label, one that also covers no other point is preferred. A label with no
 * free position is dropped: the reader can still point at the mark, and an
 * overprinted label is not readable anyway. Deterministic: same input, same
 * picture.
 */
export function placeLabels(
  requests: LabelRequest[],
  bounds: LabelBox,
  fontPx = 10,
): PlacedLabel[] {
  const placed: PlacedLabel[] = [];
  const height = fontPx * 1.15;
  const points: LabelBox[] = requests.map((request) => ({
    left: request.x - request.radius,
    right: request.x + request.radius,
    top: request.y - request.radius,
    bottom: request.y + request.radius,
  }));
  requests.forEach((request, index) => {
    const width = textWidth(request.text, fontPx);
    const gap = request.radius + 3;
    const candidates: {
      x: number;
      y: number;
      anchor: PlacedLabel["anchor"];
      box: LabelBox;
    }[] = [];
    const add = (
      x: number,
      baseline: number,
      anchor: PlacedLabel["anchor"],
    ) => {
      const left =
        anchor === "start" ? x : anchor === "end" ? x - width : x - width / 2;
      candidates.push({
        x,
        y: baseline,
        anchor,
        box: {
          left,
          right: left + width,
          top: baseline - fontPx * 0.85,
          bottom: baseline - fontPx * 0.85 + height,
        },
      });
    };
    const { x, y } = request;
    add(x, y - gap, "middle"); // above
    add(x, y + gap + fontPx * 0.8, "middle"); // below
    add(x + gap, y + fontPx * 0.35, "start"); // right
    add(x - gap, y + fontPx * 0.35, "end"); // left
    add(x + gap * 0.75, y - gap * 0.75, "start"); // upper right
    add(x - gap * 0.75, y - gap * 0.75, "end"); // upper left
    add(x + gap * 0.75, y + gap * 0.75 + fontPx * 0.7, "start"); // lower right
    add(x - gap * 0.75, y + gap * 0.75 + fontPx * 0.7, "end"); // lower left
    const free = candidates.filter(
      (candidate) =>
        candidate.box.left >= bounds.left &&
        candidate.box.right <= bounds.right &&
        candidate.box.top >= bounds.top &&
        candidate.box.bottom <= bounds.bottom &&
        !placed.some((label) => intersects(label.box, candidate.box)),
    );
    if (!free.length) return;
    const clear = free.find(
      (candidate) =>
        !points.some(
          (point, other) => other !== index && intersects(point, candidate.box),
        ),
    );
    const chosen = clear ?? free[0];
    placed.push({ index, text: request.text, ...chosen });
  });
  return placed;
}
