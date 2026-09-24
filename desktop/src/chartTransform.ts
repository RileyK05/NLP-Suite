/**
 * Zooming and panning a chart, as pure arithmetic.
 *
 * A view maps a data position (a fraction of the plot's width, 0–1) to a
 * screen position: `screen = pan + data * scale`. `scale: 1, pan: 0` is the
 * whole plot; zooming in raises `scale`, panning shifts `pan` so a window of
 * width `1/scale` is visible.
 *
 * Everything here is a pure function so the geometry of the view — clamping,
 * zoom-about-a-point, round-trips — is testable without a browser. The canvas
 * translates pointer events into these calls and applies the result as an SVG
 * transform; it holds no view logic of its own.
 */

export type ViewTransform = {
  /** How wide the plot is drawn relative to its viewport; 1 = fit, >1 = zoomed. */
  scale: number;
  /** Where the left edge of the plot sits, as a fraction of the viewport. */
  pan: number;
};

export const IDENTITY: ViewTransform = { scale: 1, pan: 0 };

/** Past this, one viewport pixel covers less than a data point's width anyway. */
export const MAX_SCALE = 40;

/** True when the view shows exactly the fitted plot — nothing to reset. */
export function isIdentity(view: ViewTransform): boolean {
  return view.scale === 1 && view.pan === 0;
}

/**
 * A view with impossible values brought back into range.
 *
 * The pan is limited so the plot always fills the viewport: zoomed out or
 * exactly fitted, panning is meaningless and collapses to zero; zoomed in, the
 * pan may range only over `[1 - scale, 0]`, the two positions where one plot
 * edge aligns with one viewport edge. Without that, a reader could drag the
 * data entirely off screen and be left with an empty chart that looks broken.
 */
export function clampView(view: ViewTransform): ViewTransform {
  const scale = Math.min(MAX_SCALE, Math.max(1, view.scale));
  if (scale <= 1) return { ...IDENTITY };
  const pan = Math.min(0, Math.max(1 - scale, view.pan));
  return { scale, pan };
}

/**
 * Zoom by `factor` while keeping the data under `anchor` stationary on screen.
 *
 * `anchor` is the pointer's position as a fraction of the *viewport* width
 * (0 = left edge, 1 = right edge). The reader expects the point under the
 * cursor to stay under the cursor, the way maps and image editors behave;
 * zooming about the plot's own origin instead makes the data slide away from
 * the pointer.
 *
 * The formula is exact in both directions: zooming by `f` and then by `1/f`
 * about the same anchor returns pan and scale to their previous values, which
 * the tests pin down.
 */
export function zoomAt(
  view: ViewTransform,
  factor: number,
  anchor: number,
): ViewTransform {
  const scale = Math.min(MAX_SCALE, Math.max(1, view.scale * factor));
  const applied = scale / view.scale;
  const pan = view.pan * applied + anchor * (1 - applied);
  return clampView({ scale, pan });
}

/** Shift the view by `delta` viewport-width fractions (positive moves data right). */
export function panBy(view: ViewTransform, delta: number): ViewTransform {
  return clampView({ scale: view.scale, pan: view.pan + delta });
}
