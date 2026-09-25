"""Publication-only figures: views of a whole table that no panel draws.

A panel is one question about a run, drawn the same way in the app and in
print. Some questions are only worth asking of the whole table at once, and
only answerable on paper: how every measure a tool computes moves with every
other; whether a measure is just document length in disguise, with both
distributions at the margins; which topic each decade leaned on; where each
speech sits among all the others. These are the "bundles" of
``docs/FIGURE_QUALITY_PLAN.md`` (4.3).

Each :class:`Bundle` names the columns it reads, like a panel, so the same
``best_table`` rule picks its table; it draws a matplotlib figure and the
sentences its caption owes the reader. :func:`render_bundle` adds the title,
subtitle and provenance line exactly as a panel's publication figure has
them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import io
from typing import Any

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panelspec import Provenance

__all__ = ["BUNDLES", "Bundle", "bundles_for", "render_bundle"]

#: The Greek letter for Spearman's correlation, spelled out for the linter.
RHO = chr(0x3C1)
#: The en dash of a range ("1930s-1940s" with the right dash), likewise.
DASH = chr(0x2013)

#: Drawn figure, its subtitle, and the caption lines it owes the reader.
Drawing = tuple[Any, str, list[str]]


@dataclass(frozen=True, slots=True)
class Bundle:
    """One publication-only figure over one table of a tool's run."""

    name: str
    tool: str
    title: str
    question: str
    requires: tuple[str, ...]
    draw: Callable[[pd.DataFrame], Drawing] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "tool": self.tool, "title": self.title, "question": self.question}


# -------------------------------------------------------------- measures --


def _measure_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce") for c in columns if c in frame.columns})
    return numeric.loc[:, [c for c in numeric.columns if numeric[c].nunique(dropna=True) >= 3]]


Prepare = Callable[[pd.DataFrame], Result[pd.DataFrame]]
#: Lengths spanning more than this ratio are drawn on a log axis.
_LOG_SPREAD = 10


def _prepared(frame: pd.DataFrame, prepare: Prepare) -> pd.DataFrame:
    """The tool's own canonical table (derived measures such as % Passive)."""
    ready = prepare(frame)
    if ready.value is None:
        raise ValueError("; ".join(d.message for d in ready.diagnostics) or "the table could not be read")
    return ready.unwrap()


def _correlations(columns: Sequence[str], prepare: Prepare) -> Callable[[pd.DataFrame], Drawing]:
    def draw(raw: pd.DataFrame) -> Drawing:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        import seaborn as sns

        from core.viz.static.labels import freeze_layout
        from core.viz.static.shapes import _colormap_named, _figure

        values = _measure_frame(_prepared(raw, prepare), columns)
        if values.shape[1] < 3:
            raise ValueError("fewer than three measures vary across these documents")
        rho = values.corr(method="spearman")
        # Measures that rank documents alike sit together.
        distance = np.clip(1 - rho.abs().to_numpy(), 0, None)
        np.fill_diagonal(distance, 0)
        order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
        rho = rho.iloc[order, order]
        size = max(4.8, 1.6 + 0.62 * len(rho))
        fig = _figure(size + 1.4, size)
        ax = fig.add_subplot()
        mask = np.triu(np.ones_like(rho, dtype=bool), k=1)
        sns.heatmap(
            rho,
            mask=mask,
            ax=ax,
            cmap=_colormap_named("diverging"),
            vmin=-1,
            vmax=1,
            center=0,
            # Written out, not formatted by seaborn: "-0.00" for a tiny
            # negative reads as a result (the tick rule, applied to cells).
            annot=rho.map(lambda v: f"{0.0 if abs(v) < 0.005 else v:.2f}").to_numpy(),
            fmt="",
            annot_kws={"fontsize": 8},
            linewidths=0.6,
            linecolor="white",
            square=True,
            cbar_kws={"label": "Spearman correlation", "shrink": 0.7},
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        ax.tick_params(length=0)
        freeze_layout(fig)
        strongest = rho.where(~np.tril(np.ones_like(rho, dtype=bool))).abs().stack().sort_values(ascending=False)
        lines = [
            "Spearman rank correlation between every pair of measures across the documents; measures are ordered "
            "so those that rank documents alike sit together.",
            "A correlation near ±1 means two measures rank the documents almost the same way: one of them adds "
            "little the other does not already say.",
        ]
        if len(strongest):
            a, b = strongest.index[0]
            lines.append(f"Most alike: {a} and {b} ({RHO} = {rho.loc[a, b]:+.2f}).")
        return fig, f"{len(values)} documents · {values.shape[1]} measures", lines

    return draw


def _against_length(measure: str, length: str, label: str, prepare: Prepare) -> Callable[[pd.DataFrame], Drawing]:
    def draw(raw: pd.DataFrame) -> Drawing:
        from scipy.stats import spearmanr

        from core.viz.panel_helpers import document_labels
        from core.viz.static.labels import LabelRequest, freeze_layout, place_labels
        from core.viz.static.shapes import _figure
        from core.viz.static.style import MUTED, OKABE_ITO
        from core.viz.static.text import AxisFormatter

        frame = _prepared(raw, prepare)
        working = frame.assign(
            _x=pd.to_numeric(frame[length], errors="coerce"), _y=pd.to_numeric(frame[measure], errors="coerce")
        ).dropna(subset=["_x", "_y"])
        if len(working) < 5:
            raise ValueError("fewer than five documents have both values")
        fig = _figure(8.5, 6.6)
        grid = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5), wspace=0.03, hspace=0.03)
        ax = fig.add_subplot(grid[1, 0])
        top = fig.add_subplot(grid[0, 0], sharex=ax)
        side = fig.add_subplot(grid[1, 1], sharey=ax)
        ax.scatter(working["_x"], working["_y"], s=24, color=OKABE_ITO[0], alpha=0.75, linewidths=0)
        # The running median of the measure along length: a trend a reader can
        # see without trusting a straight line.
        ordered = working.sort_values("_x")
        window = max(5, len(ordered) // 5) | 1
        # A few very long documents (written messages) squeeze the rest into
        # a corner; past a tenfold spread, length is read on a log scale.
        logged = float(working["_x"].min()) > 0 and float(working["_x"].max() / working["_x"].min()) > _LOG_SPREAD
        smooth = ordered["_y"].rolling(window, center=True, min_periods=window // 2 + 1).median()
        ax.plot(ordered["_x"], smooth, color="#2b3329", linewidth=1.8)
        bins: int | list[float] = (
            [float(v) for v in np.geomspace(working["_x"].min(), working["_x"].max(), 21)] if logged else 20
        )
        top.hist(working["_x"], bins=bins, color=MUTED, alpha=0.5)
        if logged:
            ax.set_xscale("log")
        side.hist(working["_y"], bins=20, color=MUTED, alpha=0.5, orientation="horizontal")
        for axis in (top, side):
            axis.grid(False)
            axis.tick_params(labelbottom=False, labelleft=False, length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
        ax.set_xlabel(label)
        ax.set_ylabel(measure)
        if not logged:
            ax.xaxis.set_major_formatter(AxisFormatter())
        ax.yaxis.set_major_formatter(AxisFormatter())
        rho, p = spearmanr(working["_x"], working["_y"])
        verdict = (
            "tracks length closely: read it as partly a measure of how long the document is"
            if abs(rho) >= 0.6
            else "moves with length somewhat"
            if abs(rho) >= 0.3
            else "is largely independent of length"
        )
        badge = ax.text(
            0.99, 0.98, f"Spearman {RHO} = {rho:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=9
        )
        renderer = freeze_layout(fig)
        names = document_labels(working["Document"].astype(str)) if "Document" in working.columns else {}
        extremes = pd.concat([working.nlargest(2, "_y"), working.nsmallest(2, "_y"), working.nlargest(1, "_x")])
        requests = [
            LabelRequest(float(row["_x"]), float(row["_y"]), names.get(str(row.get("Document", "")), ""), 3.0)
            for _, row in extremes.drop_duplicates().iterrows()
            if names.get(str(row.get("Document", "")))
        ]
        place_labels(ax, requests, renderer, fontsize=7, obstacles=[badge.get_window_extent(renderer)])
        lines = [
            f"{measure} {verdict} (Spearman {RHO} = {rho:+.2f}, p = {p:.2g}, n = {len(working)}).",
            f"The dark line is a running median over {window} documents; the bars at the edges are each axis's "
            f"distribution.{' Length is on a log scale.' if logged else ''}",
        ]
        return fig, f"{len(working)} documents · is {measure} just a measure of length?", lines

    return draw


def _decades(frame: pd.DataFrame) -> tuple[pd.Series, dict[str, str]]:
    """Each row's decade ("1930s") from its document name, and the short names."""
    from core.viz.panel_helpers import document_labels
    from core.viz.static.stats import decade_of

    if "Document" not in frame.columns:
        raise ValueError("the table has no Document column to date")
    names = document_labels(frame["Document"].astype(str))
    decades = frame["Document"].astype(str).map(lambda d: decade_of(names.get(d) or d) or "")
    return decades, names


#: A ridge needs this many documents per group to be a shape and not a guess.
_RIDGE_MIN = 4


def _eras(decades: pd.Series) -> pd.Series:
    """Decades, merged into longer eras until a typical group has a few documents.

    Twelve speeches over nine decades is one or two a decade: nothing to draw
    a distribution from. The span grows (10, 20, 30, 50 years) until the
    median group holds ``_RIDGE_MIN`` documents or more.
    """
    years = decades.str.slice(0, 4).astype(int)
    for span in (10, 20, 30, 50):
        start = years // span * span
        if start.value_counts().median() >= _RIDGE_MIN or span == 50:
            if span == 10:
                return decades
            # Named by the decades actually present ("1930s to 1940s", with an en dash), never a
            # span that runs past the last document.
            present = decades.groupby(start).agg(lambda d: (min(d), max(d)))
            names = {key: first if first == last else f"{first}{DASH}{last}" for key, (first, last) in present.items()}
            return start.map(names)
    return decades


def _ridgeline(measure: str, prepare: Prepare) -> Callable[[pd.DataFrame], Drawing]:
    """One distribution of *measure* per decade, stacked like a mountain range."""

    def draw(raw: pd.DataFrame) -> Drawing:
        import matplotlib as mpl
        from scipy.stats import gaussian_kde

        from core.viz.static.labels import freeze_layout
        from core.viz.static.shapes import _figure
        from core.viz.static.text import AxisFormatter

        frame = _prepared(raw, prepare)
        decades, _ = _decades(frame)
        working = frame.assign(_v=pd.to_numeric(frame[measure], errors="coerce"), _d=decades)
        working = working[(working["_d"] != "") & working["_v"].notna()]
        working = working.assign(_d=_eras(working["_d"]))
        groups = sorted(working["_d"].unique())
        if len(groups) < 2:
            raise ValueError("the documents span fewer than two decades")
        low, high = float(working["_v"].min()), float(working["_v"].max())
        pad = (high - low) * 0.08 or 1.0
        grid = np.linspace(low - pad, high + pad, 300)
        ramp = mpl.colormaps["viridis"]
        fig = _figure(8.5, max(3.8, 1.2 + 0.52 * len(groups)))
        ax = fig.add_subplot()
        step = 1.0
        overall = float(working["_v"].median())
        ticks: list[str] = []
        for row, decade in enumerate(reversed(groups)):
            values = working.loc[working["_d"] == decade, "_v"].to_numpy(dtype=float)
            base = row * step
            colour = ramp(1 - row / max(1, len(groups) - 1))
            if len(values) >= 3 and np.ptp(values) > 0:
                # Each ridge stays inside its own row: overlapping ridges hide
                # the one behind, which with few periods is most of the figure.
                density = gaussian_kde(values)(grid)
                height = density / density.max() * step * 0.85
                ax.fill_between(grid, base, base + height, color=colour, alpha=0.7, linewidth=0)
                ax.plot(grid, base + height, color=colour, linewidth=1.0)
            # Every document as a tick: a decade of three is three speeches.
            ax.scatter(values, np.full(len(values), base), marker="|", s=40, color="#2b3329", zorder=4)
            median = float(np.median(values))
            ax.plot([median, median], [base, base + step * 0.85], color="#2b3329", linewidth=1.3, zorder=5)
            ticks.append(f"{decade}\n(n={len(values)})")
        ax.axvline(overall, color="#9aa39a", linewidth=0.9, linestyle="--", zorder=0)
        ax.set_yticks([row * step for row in range(len(groups))], ticks)
        ax.set_ylim(-0.25, (len(groups) - 1) * step + 1.05)
        ax.set_xlabel(measure)
        ax.xaxis.set_major_formatter(AxisFormatter())
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        freeze_layout(fig)
        by = working.groupby("_d")["_v"].median()
        lines = [
            f"Each ridge is the distribution of {measure} over one period's documents (a smoothed density), with every "
            "document as a tick beneath it and its median as the dark stroke. The dashed line is the median of all "
            "documents.",
            f"Highest median: {by.idxmax()} ({by.max():.3g}); lowest: {by.idxmin()} ({by.min():.3g}). A period with few "
            "documents has a ridge drawn from few points -- read its ticks, not its shape. Decades are merged into "
            "longer periods when most would hold fewer than four documents.",
        ]
        span = "periods" if any(DASH in g for g in groups) else "decades"
        return fig, f"{len(working)} documents across {len(groups)} {span}", lines

    return draw


def _pair_grid(columns: Sequence[str], prepare: Prepare) -> Callable[[pd.DataFrame], Drawing]:
    """Every measure against every other, documents coloured by decade."""

    def draw(raw: pd.DataFrame) -> Drawing:
        import matplotlib as mpl

        from core.viz.static.labels import freeze_layout
        from core.viz.static.shapes import _figure
        from core.viz.static.text import AxisFormatter

        frame = _prepared(raw, prepare)
        values = _measure_frame(frame, columns)
        if values.shape[1] < 3:
            raise ValueError("fewer than three measures vary across these documents")
        # Five at most: past that the cells are too small to read a pattern.
        chosen = list(values.columns[:5])
        values = values[chosen].dropna()
        decades, _ = _decades(frame.loc[values.index])
        known = sorted({d for d in decades if d})
        ramp = mpl.colormaps["viridis"]
        tone = {d: ramp(i / max(1, len(known) - 1)) for i, d in enumerate(known)}
        colours = [tone.get(d, "#9aa39a") for d in decades]
        size = len(chosen)
        fig = _figure(1.9 * size + 1.6, 1.9 * size)
        axes = [[fig.add_subplot(size, size, r * size + c + 1) for c in range(size)] for r in range(size)]
        for r, row_name in enumerate(chosen):
            for c, col_name in enumerate(chosen):
                ax = axes[r][c]
                if r == c:
                    # The histogram's counts go on a hidden twin, so this cell's
                    # own y axis keeps the row's scale like every other cell.
                    ax.set_xlim(values[col_name].min(), values[col_name].max())
                    ax.set_ylim(values[row_name].min(), values[row_name].max())
                    twin = ax.twinx()
                    twin.hist(values[col_name], bins=15, color="#9aa39a", alpha=0.8)
                    twin.set_yticks([])
                    twin.grid(False)
                    for spine in twin.spines.values():
                        spine.set_visible(False)
                else:
                    ax.scatter(values[col_name], values[row_name], s=9, c=colours, linewidths=0, alpha=0.85)
                ax.tick_params(labelsize=6.5, length=2)
                ax.xaxis.set_major_formatter(AxisFormatter())
                ax.yaxis.set_major_formatter(AxisFormatter())
                ax.locator_params(nbins=3)
                if r < size - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(col_name, fontsize=7.5)
                if c > 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(row_name, fontsize=7.5)
        if known:
            handles = [
                mpl.lines.Line2D([], [], marker="o", linestyle="", color=tone[d], markersize=5, label=d) for d in known
            ]
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Decade", fontsize=7)
        freeze_layout(fig)
        rho = values.corr(method="spearman")
        pairs = rho.where(~np.tril(np.ones_like(rho, dtype=bool))).abs().stack().sort_values()
        lines = [
            "Every measure against every other, one point per document, coloured by decade; the diagonal is each "
            "measure's own distribution.",
            f"Least alike: {pairs.index[0][0]} and {pairs.index[0][1]} ({RHO} = {rho.loc[pairs.index[0]]:+.2f}); most "
            f"alike: {pairs.index[-1][0]} and {pairs.index[-1][1]} ({RHO} = {rho.loc[pairs.index[-1]]:+.2f}). A "
            "diagonal band means two measures rank the documents the same way.",
        ]
        if values.shape[1] < len(_measure_frame(frame, columns).columns):
            lines.append("The first five measures are shown; the correlation figure has them all.")
        return fig, f"{len(values)} documents · {size} measures", lines

    return draw


def _meaning_map(frame: pd.DataFrame) -> Drawing:
    """The doc_embeddings map, clusters outlined, every point named where it fits."""
    from scipy.spatial import ConvexHull

    from core.viz.panel_helpers import document_labels
    from core.viz.static.labels import LabelRequest, freeze_layout, place_labels
    from core.viz.static.shapes import _figure
    from core.viz.static.style import OKABE_ITO

    working = frame.assign(_x=pd.to_numeric(frame["X"], errors="coerce"), _y=pd.to_numeric(frame["Y"], errors="coerce"))
    working = working.dropna(subset=["_x", "_y"])
    if len(working) < 4:
        raise ValueError("fewer than four mapped points")
    clusters = sorted(working["Cluster"].astype(str).unique(), key=lambda c: (len(c), c))
    names = document_labels(working["Document"].astype(str))
    fig = _figure(9.5, 7.2)
    ax = fig.add_subplot()
    for index, cluster in enumerate(clusters):
        members = working[working["Cluster"].astype(str) == cluster]
        colour = OKABE_ITO[index % len(OKABE_ITO)]
        points = members[["_x", "_y"]].to_numpy(dtype=float)
        if len(points) >= 3 and np.linalg.matrix_rank(points - points.mean(axis=0)) == 2:
            hull = ConvexHull(points)
            ring = points[hull.vertices]
            ax.fill(ring[:, 0], ring[:, 1], color=colour, alpha=0.12, linewidth=0, zorder=1)
            ax.plot(*np.vstack([ring, ring[:1]]).T, color=colour, alpha=0.5, linewidth=0.8, zorder=2)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=36,
            color=colour,
            label=f"{cluster} ({len(points)})",
            zorder=3,
            linewidths=0.5,
            edgecolors="white",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7.5)
    renderer = freeze_layout(fig)
    place_labels(
        ax,
        [
            LabelRequest(float(row["_x"]), float(row["_y"]), names.get(str(row["Document"]), str(row["Document"])), 3.0)
            for _, row in working.iterrows()
        ],
        renderer,
        fontsize=6.5,
    )
    lines = [
        "Each point is a document placed by t-SNE from its sentence-model vector (the first two principal components "
        "when there are fewer than eight). Near points mean alike; distances across the map carry little meaning.",
        "Colour and outline are k-means clusters found in the full vector space, with k chosen by silhouette; a point "
        "far from its cluster's others sits between neighbourhoods.",
    ]
    return fig, f"{len(working)} points in {len(clusters)} clusters", lines


# ---------------------------------------------------------------- topics --


def _topics_by_decade(frame: pd.DataFrame) -> Drawing:
    import seaborn as sns

    from core.viz.panel_helpers import decimal_year
    from core.viz.static.labels import freeze_layout
    from core.viz.static.shapes import _colormap_named, _figure

    when = frame["Date"] if "Date" in frame.columns else frame.get("Year")
    years = pd.Series([decimal_year(v) for v in when], index=frame.index) if when is not None else None
    if years is None or years.isna().all():
        raise ValueError("the documents carry no dates")
    working = frame.assign(_decade=(years // 10 * 10), _topic=pd.to_numeric(frame["Dominant topic"], errors="coerce"))
    working = working.dropna(subset=["_decade", "_topic"])
    words = (
        working.groupby("_topic")["Topic keywords"].first().map(lambda k: ", ".join(str(k).split(",")[:3]).strip())
        if "Topic keywords" in working.columns
        else pd.Series(dtype=str)
    )
    share = pd.crosstab(working["_topic"], working["_decade"], normalize="columns")
    share.index = [f"Topic {int(t)}: {words.get(t, '')}".rstrip(": ") for t in share.index]
    share.columns = [f"{int(d)}s" for d in share.columns]
    counts = working.groupby("_decade").size()
    fig = _figure(max(6.5, 2.8 + 0.7 * share.shape[1]), max(3.4, 1.4 + 0.45 * share.shape[0]))
    ax = fig.add_subplot()
    sns.heatmap(
        share * 100,
        ax=ax,
        cmap=_colormap_named("sequential"),
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 8},
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "% of the decade's documents"},
    )
    ax.set_xticklabels([f"{label.get_text()}\n(n={counts.iloc[i]})" for i, label in enumerate(ax.get_xticklabels())])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    freeze_layout(fig)
    lines = [
        "Each column is one decade and sums to 100%: the share of its documents whose dominant topic is each row. "
        "n under each decade is how many documents it has -- a decade of four is four speeches, not a trend."
    ]
    return fig, f"{len(working)} documents across {share.shape[1]} decades", lines


# ------------------------------------------------------------- similarity --


def _similarity_map(frame: pd.DataFrame) -> Drawing:
    from sklearn.manifold import MDS

    from core.viz.panel_helpers import document_labels
    from core.viz.static.labels import LabelRequest, freeze_layout, place_labels
    from core.viz.static.shapes import _figure
    from core.viz.static.stats import decade_of

    names = sorted({*frame["Document A"].astype(str), *frame["Document B"].astype(str)})
    index = {name: i for i, name in enumerate(names)}
    values = pd.to_numeric(frame["Similarity"], errors="coerce")
    top = float(values.max())
    distance = np.full((len(names), len(names)), top - float(values.min()))
    np.fill_diagonal(distance, 0.0)
    for a, b, value in zip(frame["Document A"].astype(str), frame["Document B"].astype(str), values, strict=True):
        if np.isfinite(value):
            distance[index[a], index[b]] = distance[index[b], index[a]] = top - value
    if len(names) < 4:
        raise ValueError("fewer than four documents")
    import warnings

    with warnings.catch_warnings():
        # sklearn 1.8 is renaming MDS's arguments; the call below means the
        # same under both names, and the notice is not the reader's concern.
        warnings.simplefilter("ignore", FutureWarning)
        coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=4, init="random")
        points = coords.fit_transform(distance)
    labels = document_labels(names)
    decades = [decade_of(labels[name]) for name in names]
    known = sorted({d for d in decades if d})
    import matplotlib as mpl

    ramp = mpl.colormaps["viridis"]
    tone = {d: ramp(i / max(1, len(known) - 1)) for i, d in enumerate(known)}
    fig = _figure(9.5, 7.2)
    ax = fig.add_subplot()
    for decade in known:
        chosen = [i for i, d in enumerate(decades) if d == decade]
        ax.scatter(
            points[chosen, 0],
            points[chosen, 1],
            s=42,
            color=tone[decade],
            label=decade,
            linewidths=0.5,
            edgecolors="white",
            zorder=3,
        )
    rest = [i for i, d in enumerate(decades) if not d]
    if rest:
        ax.scatter(points[rest, 0], points[rest, 1], s=42, color="#9aa39a", label="undated", zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="Decade", fontsize=7.5, title_fontsize=8)
    renderer = freeze_layout(fig)
    place_labels(
        ax,
        [LabelRequest(float(points[i, 0]), float(points[i, 1]), labels[name], 4.0) for i, name in enumerate(names)],
        renderer,
        fontsize=6.5,
    )
    lines = [
        "Each point is a document, placed by multidimensional scaling so that documents that read alike sit close "
        "together; the axes have no units, only distances matter. Colour is decade: an era that clusters is a "
        "period with its own vocabulary.",
        f"Kruskal stress {_stress1(distance, points):.2f} (0 is a perfect map of every pairwise distance; below "
        "0.1 is good, above 0.2 means the flat map distorts much of the matrix).",
    ]
    return fig, f"{len(names)} documents", lines


def _stress1(distance: np.ndarray, points: np.ndarray) -> float:
    """Kruskal's stress-1: how far the map's distances depart from the matrix's."""
    drawn = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    upper = np.triu_indices(len(distance), k=1)
    target, got = distance[upper], drawn[upper]
    scale = float((target * got).sum() / (got * got).sum()) if (got * got).sum() else 1.0
    return float(np.sqrt(((target - scale * got) ** 2).sum() / (target**2).sum()))


# -------------------------------------------------------------- registry --


def _measure_bundles() -> list[Bundle]:
    from core.viz.panels import PANELS
    from core.viz.panels_document_measures import MEASURE_TOOLS

    out: list[Bundle] = []
    for name, tools in MEASURE_TOOLS.items():
        tool = max(tools, key=lambda t: len(t.measures))
        columns = tuple(dict.fromkeys([*tool.measures, *tool.length_measures]))
        if len(tool.measures) >= 3:
            out.append(
                Bundle(
                    name=f"{name}_measure_correlations",
                    tool=name,
                    title=f"How {name.replace('_', ' ')} measures move together",
                    question="Which of these measures say the same thing, and which add something?",
                    requires=tool.requires,
                    draw=_correlations(columns, tool.prepare),
                )
            )
        if len(tool.measures) >= 3:
            out.append(
                Bundle(
                    name=f"{name}_pair_grid",
                    tool=name,
                    title=f"Every {name.replace('_', ' ')} measure against every other",
                    question="What does each pair of measures look like document by document, and does an era stand out?",
                    requires=tool.requires,
                    draw=_pair_grid(columns, tool.prepare),
                )
            )
        if tool.default_measure:
            out.append(
                Bundle(
                    name=f"{name}_by_decade",
                    tool=name,
                    title=f"{tool.default_measure} over the decades",
                    question=f"How does the spread of {tool.default_measure} change from decade to decade?",
                    requires=tool.requires,
                    draw=_ridgeline(tool.default_measure, tool.prepare),
                )
            )
        if tool.length_column:
            out.append(
                Bundle(
                    name=f"{name}_against_length",
                    tool=name,
                    title=f"{tool.default_measure} against document length",
                    question=f"Is {tool.default_measure} just a proxy for how long a document is?",
                    requires=tool.requires,
                    draw=_against_length(
                        tool.default_measure, tool.length_column, tool.length_label or tool.length_column, tool.prepare
                    ),
                )
            )
    return out


def _all_bundles() -> tuple[Bundle, ...]:
    return (
        *_measure_bundles(),
        Bundle(
            name="lda_topics_by_decade",
            tool="lda_gensim",
            title="Which topics each decade leaned on",
            question="Which topics dominate the documents of each decade?",
            requires=("Document", "Dominant topic", "Topic keywords"),
            draw=_topics_by_decade,
        ),
        Bundle(
            name="doc_similarity_map",
            tool="doc_similarity",
            title="A map of the documents by similarity",
            question="Which documents read alike, and do eras form neighbourhoods?",
            requires=("Document A", "Document B", "Similarity"),
            draw=_similarity_map,
        ),
        Bundle(
            name="doc_embeddings_decades_map",
            tool="doc_embeddings",
            title="A map of the documents by meaning, by decade",
            question="Do eras form neighbourhoods of meaning?",
            requires=("Document A", "Document B", "Similarity"),
            draw=_similarity_map,
        ),
        Bundle(
            name="doc_embeddings_clusters",
            tool="doc_embeddings",
            title="Neighbourhoods of meaning",
            question="Which documents form clusters by meaning, and which sit between them?",
            requires=("Document", "X", "Y", "Cluster"),
            draw=_meaning_map,
        ),
    )


BUNDLES: tuple[Bundle, ...] = _all_bundles()


def bundles_for(tool: str) -> list[Bundle]:
    return [bundle for bundle in BUNDLES if bundle.tool == tool]


def get_bundle(name: str) -> Bundle | None:
    return next((bundle for bundle in BUNDLES if bundle.name == name), None)


def render_bundle(  # noqa: PLR0913 - provenance travels as keywords
    bundle: Bundle,
    frame: pd.DataFrame,
    fmt: str = "png",
    *,
    source: str = "",
    settings: Mapping[str, Any] | None = None,
    sha256: str = "",
    dpi: int = 200,
) -> Result[bytes]:
    """A bundle's figure as PNG/SVG/PDF bytes, with title and provenance."""
    from core.viz.static import LIBRARY, STATIC_FORMATS, add_chrome
    from core.viz.static.lint import lint_figure
    from core.viz.static.style import house_style

    if fmt not in STATIC_FORMATS:
        return Result.failure(Diagnostic.error("STATIC_BAD_FORMAT", f"format must be one of {STATIC_FORMATS}"))
    missing = [c for c in bundle.requires if c not in frame.columns]
    if missing:
        return Result.failure(Diagnostic.error("BUNDLE_MISSING_COLUMN", f"{bundle.name} needs {missing}"))
    provenance = Provenance(
        tool=bundle.tool,
        panel=bundle.name,
        source=source,
        source_sha256=sha256 or hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        settings=dict(settings or {}),
        library=LIBRARY,
    )
    try:
        with house_style():
            fig, subtitle, lines = bundle.draw(frame)
            fig.set_dpi(dpi)
            fig.canvas.draw()
            add_chrome(fig, bundle.title, subtitle, lines, provenance.caption())
            problems = lint_figure(fig)
            buffer = io.BytesIO()
            metadata = {"svg": {"Date": None}, "pdf": {"CreationDate": None, "ModDate": None}}.get(fmt, {})
            fig.savefig(buffer, format=fmt, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.3, metadata=metadata)
    except (ValueError, KeyError) as exc:
        return Result.failure(Diagnostic.info("BUNDLE_NOT_DRAWN", f"{bundle.title}: {exc}", panel=bundle.name))
    except Exception as exc:  # a drawing bug must not take the caller down (R4)
        return Result.failure(
            Diagnostic.error("STATIC_RENDER_FAILED", f"{bundle.name}: {type(exc).__name__}: {exc}", panel=bundle.name)
        )
    return Result.success(
        buffer.getvalue(),
        *(Diagnostic.warning(f"STATIC_{p.code}", f"{bundle.name}: {p.message}", panel=bundle.name) for p in problems),
    )
