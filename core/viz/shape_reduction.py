"""Story-shape reduction charts — hand-rolled SVG, no plotting backend.

Both charts are drawn as plain SVG rather than through Plotly on purpose:
the dendrogram and the component profiles are exactly the artifacts a class
needs to read, and a chart that needs a plotting install to render at all is
a chart that disappears from a lab machine. The coordinates come straight
from :mod:`core.analysis.shape_reduction`'s result objects, so the picture
and the CSV tables can never disagree.
"""

from __future__ import annotations

import html as html_lib

import pandas as pd

from core.analysis.shape_reduction import HcResult, ReductionResult
from core.result import Diagnostic, Result

__all__ = ["components_html", "dendrogram_html"]

_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#4b5563")
_PANEL_GAP = 34
_LABEL_LIMIT = 18


def _short(label: str) -> str:
    return label if len(label) <= _LABEL_LIMIT else f"{label[: _LABEL_LIMIT - 1]}…"


def dendrogram_html(hc: HcResult, *, title: str = "Hierarchical clustering") -> Result[str]:
    """The merge tree as SVG: leaves along the bottom, merge heights up.

    The U-shapes are the standard dendrogram drawing — a horizontal bar at the
    merge height spanning the two children, with verticals dropping to each
    child's own height — so the picture is literally the merge table beside
    it. Leaf labels carry the cluster each document was cut into.
    """
    merges = hc.merges
    n_docs = len(hc.leaf_order) or 1
    x_of: dict[int, float] = {leaf: float(pos) for pos, leaf in enumerate(hc.leaf_order)}
    y_of: dict[int, float] = {leaf: 0.0 for leaf in hc.leaf_order}
    segments: list[tuple[float, float, float, float]] = []
    max_h = max(hc.heights, default=1.0) or 1.0
    for index, row in enumerate(merges.to_dict("records")):
        left, right = int(row["Left id"]), int(row["Right id"])
        height = float(row["Height"])
        left_x, right_x = x_of.get(left, 0.0), x_of.get(right, 0.0)
        if left_x > right_x:
            left_x, right_x = right_x, left_x
        segments.append((left_x, y_of.get(left, 0.0), left_x, height))
        segments.append((right_x, y_of.get(right, 0.0), right_x, height))
        segments.append((left_x, height, right_x, height))
        node = n_docs + index
        x_of[node] = (left_x + right_x) / 2.0
        y_of[node] = height

    step = 46.0
    pad_left, pad_right, pad_top, pad_bottom = 30.0, 30.0, 28.0, 92.0
    width = pad_left + pad_right + step * max(n_docs - 1, 1)
    plot_h = 180.0
    height = pad_top + plot_h + pad_bottom
    if not segments:
        return Result.success(f"<h3>{html_lib.escape(title)}</h3><p>No merges to draw (fewer than two documents).</p>")

    def sx(x: float) -> float:
        return pad_left + x * step

    def sy(y: float) -> float:
        return pad_top + plot_h - (y / max_h) * plot_h

    lines = "\n".join(
        f'<line x1="{sx(x1):.1f}" y1="{sy(y1):.1f}" x2="{sx(x2):.1f}" y2="{sy(y2):.1f}" '
        'stroke="#334155" stroke-width="1.5" />'
        for x1, y1, x2, y2 in segments
    )
    cut = max(hc.heights[: max(len(hc.heights) - hc.n_clusters + 1, 0)], default=None)
    guide = (
        f'<line x1="{pad_left}" y1="{sy(cut):.1f}" x2="{width - pad_right}" y2="{sy(cut):.1f}" '
        'stroke="#dc2626" stroke-width="1" stroke-dasharray="4 3" />'
        f'<text x="{width - pad_right}" y="{sy(cut) - 4:.1f}" text-anchor="end" font-size="10" fill="#dc2626">'
        f"cut at {cut:.3g} → {hc.n_clusters} clusters</text>"
        if cut is not None
        else ""
    )
    clusters = dict(zip(hc.assignments["Document"], hc.assignments["Cluster"], strict=True))
    labels = "\n".join(
        f'<text x="{sx(x_of[leaf]):.1f}" y="{pad_top + plot_h + 14:.1f}" text-anchor="end" '
        f'font-size="10" fill="#0f172a" transform="rotate(-38 {sx(x_of[leaf]):.1f} {pad_top + plot_h + 14:.1f})">'
        f"{html_lib.escape(_short(label))} [{clusters.get(label, '?')}]</text>"
        for leaf, label in zip(hc.leaf_order, hc.assignments["Document"].tolist(), strict=True)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" '
        f'aria-label="{html_lib.escape(title)}">'
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="white" />'
        f'<text x="{pad_left}" y="16" font-size="13" font-weight="600" fill="#0f172a">'
        f"{html_lib.escape(title)} ({html_lib.escape(hc.method)} linkage)</text>"
        f"{guide}{lines}{labels}</svg>"
    )
    note = (
        f"<p>Height is the distance at which two groups merge; the dashed line is the cut that "
        f"leaves {hc.n_clusters} clusters"
        + (f" (cophenetic correlation {hc.cophenetic:.3f})" if hc.cophenetic is not None else "")
        + ".</p>"
    )
    return Result.success(f"{svg}\n{note}")


def components_html(result: ReductionResult, *, title: str = "") -> Result[str]:
    """One panel per story metric: how each component rides the narrative x-axis.

    The x axis inside a panel is the resampled story position (opening →
    ending), so a line is a SHAPE: up early and down late is one story arc,
    the mirror image is another. The share of variance (SVD) or mass (NMF)
    rides along as the legend suffix, because that is what tells you how much
    of the corpus the component is responsible for.
    """
    loadings = result.loadings
    if loadings.empty:
        return Result.success("<p>No components to draw.</p>")
    heading = title or ("SVD shape components" if result.method == "svd" else "NMF shape parts")
    panels: dict[str, list[tuple[int, pd.Series]]] = {}
    for _, row in loadings.iterrows():
        feature = str(row["Feature"])
        metric, _, raw_index = feature.rpartition(" ")
        try:
            index = int(raw_index)
        except ValueError:
            index = len(panels.get(metric, ())) + 1
        panels.setdefault(metric, []).append((index, row))
    shares = result.explained.to_dict("records")
    share_name = "Explained variance" if result.method == "svd" else "Share of mass"
    legend = " · ".join(
        f'<span style="color:{_COLORS[i % len(_COLORS)]}">Component {int(entry["Component"])} '
        f"({entry[share_name]:.0%})</span>"
        for i, entry in enumerate(shares)
    )
    chunks: list[str] = [
        f"<h3>{html_lib.escape(heading)}</h3>",
        f'<p style="font-size:12px">{legend}</p>',
    ]
    panel_w, panel_h = 520.0, 120.0
    width = 560.0
    height = 40.0 + (panel_h + _PANEL_GAP) * max(len(panels), 1)
    parts_svg = []
    y_top = 30.0
    for metric, raw_entries in panels.items():
        entries = sorted(raw_entries, key=lambda pair: pair[0])
        xs = [float(i) for i, _ in entries]
        lo = min(
            (float(row[column]) for _, row in entries for column in loadings.columns if column.startswith("Component")),
            default=0.0,
        )
        hi = max(
            (float(row[column]) for _, row in entries for column in loadings.columns if column.startswith("Component")),
            default=1.0,
        )
        if hi - lo < 1e-9:
            hi = lo + 1.0
        x_pad, y_pad = 36.0, 12.0
        scale_x = (panel_w - 2 * x_pad) / max(max(xs) - min(xs), 1.0)
        scale_y = (panel_h - 2 * y_pad) / (hi - lo)
        x0 = min(xs)

        def px(x: float, *, _x0: float = x0, _pad: float = x_pad, _scale: float = scale_x) -> float:
            return _pad + (x - _x0) * _scale

        def py(v: float, *, _hi: float = hi, _pad: float = y_pad, _scale: float = scale_y) -> float:
            return _pad + (_hi - v) * _scale

        axis = (
            f'<rect x="20" y="{y_top:.0f}" width="{panel_w:.0f}" height="{panel_h:.0f}" fill="#f8fafc" stroke="#cbd5e1" />'
            f'<text x="26" y="{y_top + 14:.0f}" font-size="11" font-weight="600" fill="#0f172a">'
            f"{html_lib.escape(metric)}</text>"
        )
        curves = []
        for slot, column in enumerate(c for c in loadings.columns if c.startswith("Component")):
            points = " ".join(
                f"{px(x):.1f},{(y_top + py(float(row[column]))):.1f}" for (_, row), x in zip(entries, xs, strict=True)
            )
            curves.append(
                f'<polyline points="{points}" fill="none" stroke="{_COLORS[slot % len(_COLORS)]}" stroke-width="1.8" />'
            )
        parts_svg.append(axis + "".join(curves))
        y_top += panel_h + _PANEL_GAP
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" aria-label="{html_lib.escape(heading)}">'
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="white" />'
        f"{''.join(parts_svg)}</svg>"
    )
    note = (
        "<p>Each panel is one story metric along the narrative (opening to ending); "
        "a line is a shape component's profile over that path.</p>"
    )
    if result.method == "nmf" and result.shift:
        note += (
            f"<p>NMF saw values shifted up by {result.shift:.4f} (non-negative input), "
            "and the shift is not undone — the parts are non-negative contributions.</p>"
        )
    _ = Diagnostic  # kept import: chart copy mirrors the engine's vocabulary
    return Result.success(f"{''.join(chunks)}\n{svg}\n{note}")
