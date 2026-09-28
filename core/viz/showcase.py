"""Showcase figures: one multi-panel publication figure per tool, built with the kit.

``docs/internal/SHOWCASE_FIGURES_PLAN.md`` section 2: a showcase is a figure that
shows three zoom levels at once -- the whole corpus, a few documents close up,
and what the method itself learned -- under a title that is the finding or the
question, with the world on the axis and model internals shown rather than
hidden. The reference figure (``mallet_what_it_does.py``) is the model.

Each :class:`Showcase` is written *with the public kit* (:mod:`core.viz.kit`,
exposed to scripts as ``nlp.viz``), so the same figure is a worked example and
a Scripts template: there is no private drawing code. A run that can support a
showcase publishes it beside its panels; :func:`render_showcase` saves it with
the same provenance footer and overlap check every other figure has.

A tool with no showcase is not a gap: it is left to its panels (the plan's
section 5, "probably not worth a showcase"). The registry below is the
explicit list of tools that can support one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import io
import re
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.kit import Canvas, canvas
from core.viz.panelspec import Source

__all__ = ["SHOWCASES", "Showcase", "render_showcase", "showcase_for", "showcases_for"]

#: The provenance line a showcase's footers carry.
LIBRARY = "NLP Suite figure kit (matplotlib)"

#: A showcase's builder: the frames it asked for, and the canvas it drew.
#: A showcase builder: the run's frames and the project's dated events (decimal
#: year and label), which it may draw as stoplines. Returns the unfinished canvas.
Build = Callable[[Mapping[str, pd.DataFrame], Sequence[tuple[float, str]]], Canvas]


@dataclass(frozen=True, slots=True)
class Showcase:
    """One tool's multi-panel publication figure."""

    name: str
    tool: str
    title: str
    question: str
    #: Table file names the builder reads; the first present, with its columns,
    #: is the run's source. Missing tables make the figure refuse, not crash.
    tables: tuple[str, ...]
    build: Build = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "tool": self.tool, "title": self.title, "question": self.question}


# --------------------------------------------------------------- shared ---

#: The x axis every dated showcase puts time on, and the fallback when the
#: corpus is undated (document order, from the corpus axis).
_X_CANDIDATES = ("Year", "Order", "Position")


def _time_axis_column(frame: pd.DataFrame) -> tuple[str, str]:
    """(column, axis label) for the time axis: Year when dated, else order.

    The plan's section 6: an undated corpus has no stream over years, so the
    figure falls back to document order rather than refusing outright.
    """
    for column, label in (("Year", "Year"), ("Order", "Document order"), ("Position", "Position in the axis")):
        if column in frame.columns and frame[column].notna().any():
            return column, label
    raise ValueError("the run has no dated or ordered column to place documents on")


def _topic_columns(frame: pd.DataFrame) -> list[str]:
    """The ``Topic <i>`` share columns of a document x topic matrix, in order."""
    found = [column for column in frame.columns if column.startswith("Topic ") and column[6:].isdigit()]
    return sorted(found, key=lambda column: int(column[6:]))


def _short(text: str, limit: int = 26) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _document_label(row: pd.Series, *, date_column: str | None) -> str:
    """A short label for a document in the close-up panel: year + a name stub."""
    name = str(row.get("Document", row.get("Document ID", "?")))
    year = row.get(date_column) if date_column else None
    if year is not None and not pd.isna(year):
        return f"{int(year)}"
    return _short(name, 14)


def _yearly_shares(matrix: pd.DataFrame, topics: Sequence[str], x: str) -> pd.DataFrame:
    """Topic shares averaged within each x value (a year has several speeches)."""
    grouped = matrix.groupby(x, sort=True)[list(topics)].mean()
    return grouped


def _topic_labels(topics: Sequence[str], keywords: Mapping[str, str]) -> dict[str, str]:
    """A direct label per topic column: its first distinctive words."""
    labels: dict[str, str] = {}
    for topic in topics:
        words = keywords.get(topic, "")
        labels[topic] = words if words else topic
    return labels


def _keywords_by_topic(topics_frame: pd.DataFrame | None, limit: int = 2) -> dict[str, str]:
    """Topic column -> its top words, from ``topics.csv`` (Topic, Word, Weight)."""
    if topics_frame is None or topics_frame.empty:
        return {}
    found: dict[str, str] = {}
    for topic_id, group in topics_frame.groupby("Topic", sort=True):
        words = [str(word) for word in group["Word"].tolist()[:limit]]
        found[f"Topic {int(topic_id)}"] = ", ".join(words)
    return found


def _pick_closeups(matrix: pd.DataFrame, x: str, count: int = 6) -> pd.DataFrame:
    """Up to *count* documents spread across the x range, evenly spaced."""
    ordered = matrix.sort_values(x).reset_index(drop=True)
    if len(ordered) <= count:
        return ordered
    positions = sorted({round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)})
    return ordered.iloc[positions].reset_index(drop=True)


# ------------------------------------------------------------ lda_gensim ---

_GENERIC = frozenset(
    {
        "year",
        "years",
        "people",
        "american",
        "americans",
        "america",
        "nation",
        "nations",
        "world",
        "government",
        "congress",
        "great",
        "make",
        "time",
        "states",
        "united",
        "national",
        "new",
        "president",
        "country",
        "today",
        "must",
        "shall",
        "may",
        "will",
        "upon",
        "every",
    }
)


def _lda_gensim_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """What the topic model sees: the corpus over time, six documents, the terms."""
    from core.viz import kit

    matrix = _require(frames, "doc_topics.csv")
    topics_frame = frames.get("topics.csv")
    relevance = frames.get("terms_by_relevance.csv")
    x, x_label = _time_axis_column(matrix)
    topics = _topic_columns(matrix)
    if len(topics) < 2:
        raise ValueError("a topic stream needs at least two topics")
    keywords = _keywords_by_topic(topics_frame)
    labels = _topic_labels(topics, keywords)
    # The most prevalent topic is drawn grey, as the reference greys the
    # background topic, so the eye goes to the topics that rise and fall.
    means = matrix[list(topics)].mean()
    background = str(means.idxmax())
    colors = kit.palette(topics, colors={})

    event_positions = tuple(position for position, _ in events)
    stream_shares = _yearly_shares(matrix, topics, x)
    # Order topics by the x position of their peak, so the stack reads in time.
    order = sorted(topics, key=lambda topic: _peak(stream_shares[topic], stream_shares.index))
    order = [topic for topic in order if topic != background] + [background]

    fig = canvas(
        "What the topic model sees",
        "Every document is a mixture of all topics; the shares rise and fall over the corpus",
        size=(15.0, 13.0),
        library=LIBRARY,
    )
    fig.rows([1.05, 1.0], hspace=0.34)
    row_a = fig.row(0)
    row_a.cols([1.0])
    ax_a = row_a.axis(0)
    kit.stream(
        ax_a,
        stream_shares,
        labels=labels,
        colors=colors,
        background=background,
        order=order,
        event_at=event_positions,
    )
    ax_a.set(xlabel=x_label, ylabel="Average topic share")
    kit.time_axis(ax_a, matrix, x=x)
    if events and x == "Year":
        kit.events(ax_a, list(events))
    ax_a.set_title(
        "A. The corpus over time: each document split among every topic"
        + ("" if x == "Year" else " (this corpus is undated; documents are in order)"),
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=24,
    )

    row_b = fig.row(1)
    row_b.cols([1.05, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    closeups = _pick_closeups(matrix, x, 6)
    date_column = x if x in closeups.columns else None
    shares = closeups.set_index(closeups.index)[list(topics)]
    shares.index = [_document_label(row, date_column=date_column) for _, row in closeups.iterrows()]
    kit.mixture_bars(ax_b, shares, colors=colors, background=background, order=order)
    ax_b.set_title(
        "B. Six documents, as the model reads them",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax_b.set_xlabel("Share of the document")

    _draw_terms(ax_c, topics, keywords, relevance, colors)
    ax_c.set_title(
        "C. What the model learned: each topic's top words", loc="left", fontsize=11, fontweight="bold", pad=10
    )
    return fig


def _peak(series: pd.Series, index: Any) -> int:
    smooth = pd.Series(series.to_numpy()).rolling(5, center=True, min_periods=1).mean()
    return int(smooth.to_numpy().argmax())


def _require(frames: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    frame = frames.get(name)
    if frame is None or frame.empty:
        raise ValueError(f"this run has no {name}")
    return frame


def labels_for(topic: str, keywords: Mapping[str, str]) -> str:
    """A topic's short direct label for panel A: its first distinctive words.

    ``keywords`` are the model's top words (comma-joined). Two words are kept
    so a band is identifiable in the stream without a legend; the full list is
    in panel C.
    """
    words = [word.strip() for word in keywords.get(topic, "").split(",") if word.strip()]
    return ", ".join(words[:2]) if words else topic


def _draw_terms(
    ax: Any,
    topics: Sequence[str],
    keywords: Mapping[str, str],
    relevance: pd.DataFrame | None,
    colors: Mapping[str, str],
) -> None:
    """The method's own output: each topic's top words, shaded by corpus frequency.

    A compact grid: one row per topic (in the shared colour), the topic's top
    words as cells, each cell shaded by how often the word occurs corpus-wide.
    A word dark in every row is common everywhere; a word dark in one row is
    that topic's own. One axis, so it scales to any topic count.
    """
    from matplotlib.patches import Rectangle

    from core.viz.static.style import INK, SEQUENTIAL, house_style

    ax.set_axis_off()
    rows: list[tuple[str, list[tuple[str, float]]]] = []
    for topic in topics:
        words = _topic_words(topic, keywords, relevance)
        if words:
            rows.append((topic, words))
    if not rows:
        return
    width = max(len(words) for _, words in rows)
    all_counts = [count for _, words in rows for _, count in words]
    low, high = (min(all_counts), max(all_counts)) if all_counts else (0.0, 1.0)
    span = (high - low) or 1.0
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("kit_topic_words", list(SEQUENTIAL))
    with house_style():
        for row, (topic, words) in enumerate(rows):
            y = len(rows) - row
            ax.add_patch(Rectangle((0.0, y - 0.92), 0.14, 0.84, facecolor=colors.get(topic, "#0072B2"), linewidth=0))
            ax.text(
                0.02,
                y - 0.5,
                labels_for(topic, keywords),
                ha="left",
                va="center",
                fontsize=7.0,
                color="white",
                clip_on=True,
            )
            for col, (word, count) in enumerate(words):
                shade = 0.18 + 0.7 * (count - low) / span
                ax.add_patch(
                    Rectangle(
                        (0.16 + col * 0.168, y - 0.92),
                        0.16,
                        0.84,
                        facecolor=cmap(shade),
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
                ax.text(0.24 + col * 0.168, y - 0.5, word, ha="center", va="center", fontsize=7.2, color=INK)
    ax.set_xlim(0, 0.16 + width * 0.168)
    ax.set_ylim(0, len(rows) + 0.1)


def _topic_words(
    topic: str, keywords: Mapping[str, str], relevance: pd.DataFrame | None, limit: int = 5
) -> list[tuple[str, float]]:
    """The top ``limit`` words for one topic, as (word, corpus frequency)."""
    if relevance is not None and not relevance.empty:
        rows = relevance.loc[pd.to_numeric(relevance["Topic"], errors="coerce") == int(topic[6:])]
        if not rows.empty:
            rows = rows.sort_values("Relevance", ascending=False).head(limit)
            counts = pd.to_numeric(rows["Corpus frequency"], errors="coerce").fillna(0).tolist()
            return [(str(word), float(count)) for word, count in zip(rows["Word"], counts, strict=True)]
    words = [word.strip() for word in keywords.get(topic, "").split(",") if word.strip()][:limit]
    return [(word, float(len(words) - index)) for index, word in enumerate(words)]


def _draw_terms_from_csv(
    ax: Any, topics: Sequence[str], keywords: Mapping[str, str], colors: Mapping[str, str]
) -> None:
    _draw_terms(ax, topics, keywords, None, colors)


# --------------------------------------------------------------- registry --


def _lda_mallet_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """What MALLET sees: the corpus over time, six documents, and its internals.

    MALLET's own readout is thinner than Gensim's (topic keys, and a
    dominant-topic table from which the shares are recovered), so panel C
    shows the topic weights it does publish rather than inventing per-word
    probabilities.
    """
    from core.viz import kit

    dominant = _require(frames, "topics_dominant.csv")
    topics_frame = frames.get("topics.csv")
    keywords = _keyword_columns(topics_frame)
    matrix = _mallet_matrix(dominant, keywords)
    if matrix is None or len(_topic_columns(matrix)) < 2:
        raise ValueError("MALLET dominant-topic output carries no mixture to draw")
    x, x_label = _time_axis_column(dominant)
    topics = _topic_columns(matrix)
    labels = _topic_labels(topics, keywords)
    means = matrix[list(topics)].mean()
    background = str(means.idxmax())
    colors = kit.palette(topics, colors={})
    stream_shares = _yearly_shares(matrix, topics, x)
    order = sorted(topics, key=lambda t: _peak(stream_shares[t], stream_shares.index))
    order = [t for t in order if t != background] + [background]

    fig = canvas(
        "What MALLET sees in the corpus",
        "Every document is a mixture of all topics; the shares rise and fall over the corpus",
        size=(15.0, 13.0),
        library=LIBRARY,
    )
    fig.rows([1.05, 1.0], hspace=0.34)
    row_a = fig.row(0)
    row_a.cols([1.0])
    ax_a = row_a.axis(0)
    kit.stream(ax_a, stream_shares, labels=labels, colors=colors, background=background, order=order)
    ax_a.set(xlabel=x_label, ylabel="Average topic share")
    kit.time_axis(ax_a, matrix, x=x)
    if events and x == "Year":
        kit.events(ax_a, list(events))
    ax_a.set_title(
        "A. The corpus over time: each document split among every topic",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=24,
    )

    row_b = fig.row(1)
    row_b.cols([1.05, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    closeups = _pick_closeups(matrix, x, 6)
    shares = closeups[list(topics)]
    shares.index = [_document_label(row, date_column=x) for _, row in closeups.iterrows()]
    kit.mixture_bars(ax_b, shares, colors=colors, background=background, order=order)
    ax_b.set_title("B. Six documents, as MALLET reads them", loc="left", fontsize=11, fontweight="bold", pad=10)
    ax_b.set_xlabel("Share of the document")
    _draw_mallet_weights(ax_c, topics_frame, keywords, colors)
    ax_c.set_title(
        "C. MALLET's own readout: topic weight and words", loc="left", fontsize=11, fontweight="bold", pad=10
    )
    return fig


def _trend_panel(  # noqa: PLR0913 - keyword-only drawing options, like a kit piece
    ax: Any,
    frame: pd.DataFrame,
    *,
    value: str,
    x: str,
    x_label: str,
    title: str,
    events: Sequence[tuple[float, str]] = (),
) -> None:
    """A per-document measure over time: points, a rolling median, a band.

    The plan's panel A for a one-measure tool: the whole corpus at a glance,
    with the median line so a single document is not read as a trend.
    """
    from core.viz import kit

    values = pd.to_numeric(frame[value], errors="coerce")
    usable = frame.loc[values.notna()].sort_values(x)
    if usable.empty:
        raise ValueError(f"no numeric {value} to draw")
    xs = usable[x].to_numpy(dtype=float)
    ys = pd.to_numeric(usable[value], errors="coerce").to_numpy(dtype=float)
    window = max(3, min(9, len(xs) // 4 * 2 + 1))
    ax.plot(xs, ys, "o", ms=4.0, alpha=0.5, color="#0072B2", label="One document")
    median = pd.Series(ys).rolling(window, center=True, min_periods=1).median().to_numpy()
    ax.plot(xs, median, "-", lw=1.8, color="#D55E00", label=f"{window}-document median")
    if events:
        kit.events(ax, list(events))
    ax.set(xlabel=x_label, ylabel=value)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=24 if events else 8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def _joint_panel(  # noqa: PLR0913 - keyword-only drawing options, like a kit piece
    ax: Any, frame: pd.DataFrame, *, x: str, y: str, x_label: str, y_label: str
) -> None:
    """Two measures against each other, with the Spearman rho named."""
    from scipy.stats import spearmanr

    from core.viz.static.style import MUTED

    pairs = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pairs) < 3:
        raise ValueError(f"too few documents to compare {x} and {y}")
    ax.scatter(pairs[x], pairs[y], s=18, color="#0072B2", alpha=0.6, linewidths=0)
    rho = float(spearmanr(pairs[x], pairs[y]).statistic)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.text(0.02, 0.98, f"Spearman rho = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=8.4, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)


def _group_box(ax: Any, frame: pd.DataFrame, *, value: str, group: str | None, year_column: str | None) -> None:
    """One box per group (a decade, or a detail) for a measure."""
    values = pd.to_numeric(frame[value], errors="coerce")
    if group is not None and group in frame.columns:
        labels = frame[group].astype(str)
    elif year_column is not None and year_column in frame.columns:
        labels = pd.to_numeric(frame[year_column], errors="coerce").map(
            lambda year: f"{int(year) // 10 * 10}s" if pd.notna(year) else ""
        )
    else:
        labels = pd.Series(["(all)"] * len(frame), index=frame.index)
    usable = frame.assign(_group=labels, _value=values).loc[lambda d: d["_group"].ne("") & d["_value"].notna()]
    groups = sorted(usable["_group"].unique())
    if len(groups) < 2:
        raise ValueError("too few groups to compare")
    data = [usable.loc[usable["_group"] == name, "_value"].to_numpy() for name in groups]
    ax.boxplot(
        data,
        tick_labels=groups,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#c6dbef", "edgecolor": "#2171b5"},
        medianprops={"color": "#D55E00"},
    )
    ax.set_ylabel(value)
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)


def _extreme_sentences(ax: Any, sentences: pd.DataFrame | None) -> None:
    """The most positive and most negative sentences, as text (the plan's panel B).

    Drawn as six labelled lines on one axis: the score in colour, the sentence
    beside it. No nested axes, so the panel never collapses and the text is
    measured by the figure's own linter like any other.
    """
    ax.set_axis_off()
    if sentences is None or sentences.empty or "Compound" not in sentences.columns:
        ax.text(0.0, 0.5, "sentence-level scores are not in this run", fontsize=9, color="#5d6a5c")
        return
    scores = pd.to_numeric(sentences["Compound"], errors="coerce")
    usable = sentences.loc[scores.notna()].assign(_score=scores.dropna())
    if usable.empty:
        return
    picks = pd.concat([usable.nlargest(3, "_score"), usable.nsmallest(3, "_score")]).reset_index(drop=True)
    rows = len(picks)
    top, bottom = 0.96, 0.04
    step = (top - bottom) / max(1, rows)
    for index, row in picks.iterrows():
        y = top - step * (index + 0.5)
        score = float(row["_score"])
        color = "#009E73" if score >= 0 else "#D55E00"
        ax.text(0.0, y, f"{score:+.2f}", fontsize=9.0, fontweight="bold", color=color, ha="left", va="center")
        ax.text(0.09, y, _short(str(row.get("Sentence", "")), 88), fontsize=8.0, ha="left", va="center")


def _share_by_document(frame: pd.DataFrame, x: str, columns: Sequence[str]) -> pd.DataFrame:
    """Each column's share of the row total, indexed by x, averaged within x."""
    present = [column for column in columns if column in frame.columns]
    values = frame[present].apply(pd.to_numeric, errors="coerce").fillna(0)
    totals = values.sum(axis=1).replace(0, pd.NA)
    shares = values.div(totals, axis=0).fillna(0)
    shares = shares.assign(**{x: frame[x].to_numpy()}).dropna(subset=[x])
    return shares.groupby(x, sort=True)[present].mean()


def _readability_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """Did the speeches get simpler: the trend, a length check, decade spread."""
    frame = _require(frames, "readability.csv")
    x, x_label = _time_axis_column(frame)
    fig = canvas(
        "Did the speeches get simpler?",
        "Flesch Reading Ease rises as prose gets easier; length is checked before trusting any of it",
        size=(15.0, 13.0),
        library=LIBRARY,
    )
    fig.rows([1.05, 1.0], hspace=0.34)
    ax_a = fig.row(0).cols([1.0]).axis(0)
    _trend_panel(
        ax_a,
        frame,
        value="Flesch Reading Ease",
        x=x,
        x_label=x_label,
        title="A. The whole corpus: reading ease by document",
        events=events if x == "Year" else (),
    )
    row_b = fig.row(1)
    row_b.cols([1.0, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    _joint_panel(ax_b, frame, x="Words", y="Flesch Reading Ease", x_label="Words", y_label="Flesch Reading Ease")
    ax_b.set_title(
        "B. Length check: is reading ease just document length?", loc="left", fontsize=11, fontweight="bold", pad=8
    )
    _group_box(ax_c, frame, value="Flesch Reading Ease", group=None, year_column="Year")
    ax_c.set_title("C. Reading ease by decade", loc="left", fontsize=11, fontweight="bold", pad=8)
    return fig


def _sentiment_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """How the tone moved: the trend, the extremes in text, the spread by group."""
    from core.viz import kit

    frame = _require(frames, "vader.csv")
    x, x_label = _time_axis_column(frame)
    fig = canvas(
        "How did the tone move?",
        "VADER compound sentiment per document, with the most positive and negative sentences read back",
        size=(15.0, 13.5),
        library=LIBRARY,
    )
    fig.rows([1.0, 1.0], hspace=0.36)
    ax_a = fig.row(0).cols([1.0]).axis(0)
    _trend_panel(
        ax_a,
        frame,
        value="Compound",
        x=x,
        x_label=x_label,
        title="A. The whole corpus: VADER compound per document",
        events=events if x == "Year" else (),
    )
    ax_a.set_ylim(-1, 1)
    row_b = fig.row(1)
    row_b.cols([1.0, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    _extreme_sentences(ax_b, frames.get("vader_sentences.csv"))
    ax_b.set_title("B. The extremes, as text", loc="left", fontsize=11, fontweight="bold", pad=8)
    detail = next((column for column in ("Speaker", "Party", "Kind") if column in frame.columns), None)
    if detail is not None:
        _group_box(ax_c, frame, value="Compound", group=detail, year_column=None)
        ax_c.set_title(f"C. Compound by {detail}", loc="left", fontsize=11, fontweight="bold", pad=8)
    else:
        _group_box(ax_c, frame, value="Compound", group=None, year_column="Year")
        ax_c.set_title("C. Compound by decade", loc="left", fontsize=11, fontweight="bold", pad=8)
    return fig


def _verb_analysis_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """How the grammar changed: tense mix over time, the passive's share, modality."""
    from core.viz import kit

    frame = _require(frames, "verb_summary.csv")
    x, x_label = _time_axis_column(frame)
    tense = ["Past", "Present", "Future", "Gerund"]
    fig = canvas(
        "How did the grammar change?",
        "The mix of verb tense and voice across the corpus, from the verbs the parser tagged",
        size=(15.0, 13.0),
        library=LIBRARY,
    )
    fig.rows([1.05, 1.0], hspace=0.34)
    ax_a = fig.row(0).cols([1.0]).axis(0)
    kit.stream(ax_a, _share_by_document(frame, x, tense), labels={name: name for name in tense}, order=tense)
    ax_a.set(xlabel=x_label, ylabel="Share of verbs")
    kit.time_axis(ax_a, frame, x=x)
    if events and x == "Year":
        kit.events(ax_a, list(events))
    ax_a.set_title("A. The whole corpus: verb-tense mix over time", loc="left", fontsize=11, fontweight="bold", pad=24)
    row_b = fig.row(1)
    row_b.cols([1.05, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    kit.stream(
        ax_b,
        _share_by_document(frame, x, ["Passive", "Active"]),
        labels={"Passive": "Passive", "Active": "Active"},
        order=["Active", "Passive"],
    )
    ax_b.set(xlabel=x_label, ylabel="Share of voice")
    ax_b.set_title("B. The passive's share of the voice", loc="left", fontsize=11, fontweight="bold", pad=8)
    modality = [name for name in ("Ability", "Possibility", "Obligation", "No modality") if name in frame.columns]
    totals = [float(pd.to_numeric(frame[name], errors="coerce").sum()) for name in modality]
    kit.bars(ax_c, totals, labels=modality)
    ax_c.set_title("C. Modality across the corpus", loc="left", fontsize=11, fontweight="bold", pad=8)
    return fig


def _ner_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """Who and where the corpus talks about: type mix, top entities per era."""
    from core.viz import kit

    timeline = _require(frames, "entity_timeline.csv")
    x, x_label = _time_axis_column(timeline)
    # The narrative types a reader watches (the reference figure's focus);
    # DATE/CARDINAL/ORDINAL are noise for a "who and where" title.
    kinds = ["PERSON", "ORG", "GPE", "NORP", "MONEY", "LAW"]
    present = [kind for kind in kinds if (timeline["NER Tag"] == kind).any()]
    if len(present) < 2:
        present = list(timeline["NER Tag"].value_counts().head(6).index)
    per_doc = _entity_type_counts(timeline, x, present)
    kit.palette(present, colors={})
    fig = canvas(
        "Who and where the corpus talks about",
        "The mix of named-entity types over the corpus, and the entities each era names most",
        size=(15.0, 13.0),
        library=LIBRARY,
    )
    fig.rows([1.05, 1.0], hspace=0.34)
    ax_a = fig.row(0).cols([1.0]).axis(0)
    shares = _shares_of_total(per_doc, present)
    kit.stream(ax_a, shares, labels={kind: kind for kind in present}, order=present)
    ax_a.set(xlabel=x_label, ylabel="Share of entity mentions")
    kit.time_axis(ax_a, timeline, x=x)
    if events and x == "Year":
        kit.events(ax_a, list(events))
    ax_a.set_title("A. The whole corpus: entity-type mix over time", loc="left", fontsize=11, fontweight="bold", pad=24)
    row_b = fig.row(1)
    row_b.cols([1.0, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    counts = timeline.groupby("Entity")["Count"].sum().sort_values(ascending=False).head(12)
    kit.bars(ax_b, counts.to_numpy(), labels=[str(name) for name in counts.index])
    ax_b.set_title("B. The entities named most often", loc="left", fontsize=11, fontweight="bold", pad=8)
    _era_entities(ax_c, timeline, x)
    ax_c.set_title("C. The top entity in each era", loc="left", fontsize=11, fontweight="bold", pad=8)
    return fig


def _entity_type_counts(timeline: pd.DataFrame, x: str, kinds: Sequence[str]) -> pd.DataFrame:
    counts = timeline.assign(_kind=timeline["NER Tag"].astype(str))
    table = counts.pivot_table(index=x, columns="_kind", values="Count", aggfunc="sum", fill_value=0)
    return table.reindex(columns=list(kinds), fill_value=0).sort_index()


def _shares_of_total(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = table[list(columns)].apply(pd.to_numeric, errors="coerce").astype(float)
    totals = numeric.sum(axis=1)
    # A zero total is no share, not a division by zero; the rows are dropped
    # by fillna(0) on a float frame, so no object downcast warning.
    safe = totals.where(totals.ne(0))
    return numeric.div(safe, axis=0).fillna(0.0)


def _era_entities(ax: Any, timeline: pd.DataFrame, x: str) -> None:
    """For each era (a decade of x), the entity its documents name most."""
    if x == "Year":
        era = (pd.to_numeric(timeline[x], errors="coerce") // 10 * 10).astype("Int64").astype(str) + "s"
    else:
        era = timeline[x].astype(str)
    working = timeline.assign(_era=era, _count=pd.to_numeric(timeline["Count"], errors="coerce").fillna(0))
    tops = (
        working.groupby(["_era", "Entity"], as_index=False)["_count"]
        .sum()
        .sort_values(["_era", "_count"], ascending=[True, False])
        .groupby("_era")
        .head(1)
    )
    from core.viz.static.style import INK

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(tops) + 0.1)
    for row, (_, record) in enumerate(tops.iterrows()):
        y = len(tops) - row
        ax.text(0.0, y - 0.5, f"{record['_era']}", fontsize=8.4, color="#6b7268", va="center")
        ax.text(0.18, y - 0.5, str(record["Entity"]), fontsize=9.2, fontweight="bold", color=INK, va="center")
        ax.text(
            1.0, y - 0.5, f"{int(record['_count'])} mentions", fontsize=7.6, color="#6b7268", va="center", ha="right"
        )


def _doc_similarity_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """Which documents read alike: the pair distribution and the strongest links."""
    pairs = _require(frames, "doc_pairs.csv")
    fig = canvas(
        "Which documents read alike?",
        "Similarity between documents, and how far back each document's nearest neighbour reaches",
        size=(15.0, 12.0),
        library=LIBRARY,
    )
    fig.rows([1.05, 1.0], hspace=0.34)
    ax_a = fig.row(0).cols([1.0]).axis(0)
    _similarity_distribution(ax_a, pairs)
    ax_a.set_title("A. The shape of document similarity", loc="left", fontsize=11, fontweight="bold", pad=8)
    row_b = fig.row(1)
    row_b.cols([1.0, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    _top_pairs(ax_b, pairs, 10)
    ax_b.set_title("B. The most similar document pairs", loc="left", fontsize=11, fontweight="bold", pad=8)
    _reach(ax_c, pairs)
    ax_c.set_title(
        "C. How far back each document's nearest match reaches", loc="left", fontsize=11, fontweight="bold", pad=8
    )
    return fig


def _pair_x(pairs: pd.DataFrame) -> str:
    return "Year" if "Year" in pairs.columns else "Document ID A"


def _similarity_distribution(ax: Any, pairs: pd.DataFrame) -> None:
    values = pd.to_numeric(pairs["Similarity"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("no similarity values to draw")
    ax.hist(values, bins=min(20, max(4, len(values) // 3)), color="#c6dbef", edgecolor="#2171b5")
    ax.set(xlabel="Cosine similarity (%)", ylabel="Document pairs")
    ax.spines[["top", "right"]].set_visible(False)


def _top_pairs(ax: Any, pairs: pd.DataFrame, top_n: int) -> None:
    working = pairs.assign(_sim=pd.to_numeric(pairs["Similarity"], errors="coerce")).dropna(subset=["_sim"])
    working = working.nlargest(top_n, "_sim")
    labels = [
        f"{_short_pair(a)} ↔ {_short_pair(b)}"
        for a, b in zip(working["Document A"], working["Document B"], strict=False)
    ]
    from core.viz.static.style import house_style

    ys = list(range(len(working)))
    with house_style():
        ax.barh(ys, working["_sim"].to_numpy(), color="#2171b5", height=0.66)
    for y, value in zip(ys, working["_sim"], strict=True):
        ax.text(float(value) + 1.0, y, f"{float(value):.0f}%", va="center", fontsize=8.0, color="#5d6a5c")
    ax.set_yticks(ys, labels, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0, float(working["_sim"].max()) * 1.2 if len(working) else 1)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)


def _short_pair(name: str) -> str:
    """A document name shortened to its year (the part that identifies it in time)."""
    text = str(name)
    return text[:4] if text[:4].isdigit() else _short(text, 16)


def _reach(ax: Any, pairs: pd.DataFrame) -> None:
    """The strongest neighbour of each document, plotted at its own position."""
    if "Document ID A" not in pairs.columns or "Document ID B" not in pairs.columns:
        raise ValueError("needs document ids on both sides of each pair")
    working = pairs.assign(_sim=pd.to_numeric(pairs["Similarity"], errors="coerce")).dropna(subset=["_sim"])
    best = working.sort_values("_sim", ascending=False).groupby("Document ID A").head(1).sort_values("Document ID A")
    if best.empty:
        raise ValueError("no pairs to plot")
    ax.scatter(best["Document ID A"], best["_sim"], s=26, color="#D55E00", linewidths=0)
    for _, row in best.iterrows():
        ax.annotate(
            str(int(row["Document ID B"])),
            (row["Document ID A"], row["_sim"]),
            fontsize=6.6,
            color="#5d6a5c",
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set(xlabel="Document ID (position in the corpus)", ylabel="Best similarity (%)")
    ax.spines[["top", "right"]].set_visible(False)


def _keyness_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """What sets the groups apart: effect against evidence, and the top words each way."""
    from core.viz import kit
    from core.viz.static.style import INK

    frame = _require(frames, "keyness.csv")
    working = frame.assign(
        _g2=pd.to_numeric(frame["G2 (log-likelihood)"], errors="coerce"),
        _lr=pd.to_numeric(frame["Log Ratio"], errors="coerce"),
        _p=pd.to_numeric(frame["p-value"], errors="coerce"),
    ).dropna(subset=["_g2", "_lr"])
    if working.empty:
        raise ValueError("no keyness rows with G2 and Log Ratio")
    fig = canvas(
        "What sets the groups apart?",
        "Effect (log ratio) against evidence (G2); the words far right and high moved most between the groups",
        size=(15.0, 12.0),
        library=LIBRARY,
    )
    fig.rows([1.0, 1.0], hspace=0.34)
    ax_a = fig.row(0).cols([1.0]).axis(0)
    colors = {"Group A": "#0072B2", "Group B": "#D55E00"}
    for side in ("Group A", "Group B"):
        part = working.loc[working["Overrepresented in"].astype(str) == side]
        ax_a.scatter(part["_g2"], part["_lr"], s=22, alpha=0.6, color=colors[side], linewidths=0, label=side)
    ax_a.axhline(0, color="#c9cfc6", linewidth=0.8)
    top = working.nlargest(8, "_g2")
    for _, row in top.iterrows():
        ax_a.annotate(
            str(row["Word"]),
            (row["_g2"], row["_lr"]),
            fontsize=6.8,
            color=INK,
            xytext=(3, 2),
            textcoords="offset points",
        )
    ax_a.set(xlabel="Evidence: G2 (log-likelihood)", ylabel="Effect: log ratio")
    ax_a.legend(frameon=False, fontsize=8)
    ax_a.set_title("A. Effect against evidence", loc="left", fontsize=11, fontweight="bold", pad=8)
    row_b = fig.row(1)
    row_b.cols([1.0, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    _keyness_side(ax_b, working, "Group A", colors["Group A"])
    ax_b.set_title("B. Words that set group A apart", loc="left", fontsize=11, fontweight="bold", pad=8)
    _keyness_side(ax_c, working, "Group B", colors["Group B"])
    ax_c.set_title("C. Words that set group B apart", loc="left", fontsize=11, fontweight="bold", pad=8)
    return fig


def _keyness_side(ax: Any, working: pd.DataFrame, side: str, color: str) -> None:
    from core.viz import kit

    part = working.loc[working["Overrepresented in"].astype(str) == side].nlargest(10, "_g2")
    if part.empty:
        raise ValueError(f"no words overrepresented in {side}")
    kit.bars(ax, part["_lr"].to_numpy(), labels=[str(word) for word in part["Word"]], colors=color)


def _lexicon_series_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """When the corpus started saying X: a category's rate over time, by facet."""
    from core.viz import kit

    frame = _require(frames, "lexicon_series.csv")
    facet = "Facet"
    categories = sorted(frame["Category"].astype(str).unique())
    table = frame.pivot_table(index=facet, columns="Category", values="Per 1000", aggfunc="mean").sort_index()
    table.index = pd.to_numeric(table.index, errors="coerce")
    table = table.loc[table.index.notna()].sort_index()
    colors = kit.palette(categories, colors={})
    fig = canvas(
        "When did the corpus start saying this?",
        "Each category's rate per 1,000 words over the corpus, with the categories as an overlay",
        size=(15.0, 11.0),
        library=LIBRARY,
    )
    fig.rows([1.0, 1.0], hspace=0.34)
    ax_a = fig.row(0).cols([1.0]).axis(0)
    for category in categories:
        series = pd.to_numeric(table[category], errors="coerce")
        ax_a.plot(table.index, series, "-o", ms=3.0, lw=1.6, color=colors[category], label=str(category))
    if events:
        kit.events(ax_a, list(events))
    ax_a.set(xlabel=table.index.name or "Facet", ylabel="Occurrences per 1,000 words")
    ax_a.legend(frameon=False, fontsize=7.6, ncol=min(4, len(categories)))
    ax_a.set_title(
        "A. The whole corpus: rate per 1,000 words", loc="left", fontsize=11, fontweight="bold", pad=24 if events else 8
    )
    row_b = fig.row(1)
    row_b.cols([1.0, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    coverage = frame.pivot_table(
        index=facet, columns="Category", values="Sentence Percent", aggfunc="mean"
    ).sort_index()
    coverage.index = pd.to_numeric(coverage.index, errors="coerce")
    coverage = coverage.loc[coverage.index.notna()].sort_index()
    for category in categories:
        ax_b.plot(
            coverage.index, pd.to_numeric(coverage[category], errors="coerce"), "-", lw=1.6, color=colors[category]
        )
    ax_b.set(xlabel=coverage.index.name or "Facet", ylabel="% of sentences with a match")
    ax_b.set_title(
        "B. Coverage: how many sentences mention the category", loc="left", fontsize=11, fontweight="bold", pad=8
    )
    totals = frame.groupby("Category")["Occurrences"].sum().reindex(categories).fillna(0)
    kit.bars(ax_c, totals.to_numpy(), labels=[str(c) for c in totals.index], colors=[colors[c] for c in totals.index])
    ax_c.set_title("C. Occurrences across the corpus", loc="left", fontsize=11, fontweight="bold", pad=8)
    return fig


def _bert_topics_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """Which topics each era talks about: topic shares over time and the top words."""
    from core.viz import kit

    docs = _require(frames, "bert_topic_docs.csv")
    topics_frame = frames.get("bert_topics.csv")
    x, x_label = _time_axis_column(docs)
    topics_frame = topics_frame if topics_frame is not None else pd.DataFrame()
    topic_ids = sorted(pd.to_numeric(docs["Topic"], errors="coerce").dropna().unique())
    if len(topic_ids) < 2:
        raise ValueError("BERT topics needs at least two topics to compare")
    working = docs.assign(_topic=pd.to_numeric(docs["Topic"], errors="coerce"))
    counts = working.pivot_table(index=x, columns="_topic", values="Document ID", aggfunc="count", fill_value=0)
    counts.columns = [f"Topic {int(t)}" for t in counts.columns]
    shares = _shares_of_total(counts, list(counts.columns))
    keywords = _keywords_by_topic(topics_frame)
    labels = {column: (keywords.get(column) or column) for column in shares.columns}
    background = str(shares.mean().idxmax())
    colors = kit.palette(list(shares.columns), colors={})
    order = sorted(shares.columns, key=lambda c: _peak(shares[c], shares.index))
    order = [c for c in order if c != background] + [background]
    fig = canvas(
        "Which topics does each era talk about?",
        "The share of documents each BERT topic claims over the corpus, and the words the topic learned",
        size=(15.0, 13.0),
        library=LIBRARY,
    )
    fig.rows([1.05, 1.0], hspace=0.34)
    ax_a = fig.row(0).cols([1.0]).axis(0)
    kit.stream(
        ax_a,
        shares,
        labels=labels,
        colors=colors,
        background=background,
        order=order,
        event_at=tuple(p for p, _ in events),
    )
    ax_a.set(xlabel=x_label, ylabel="Share of documents")
    kit.time_axis(ax_a, docs, x=x)
    if events and x == "Year":
        kit.events(ax_a, list(events))
    ax_a.set_title("A. Topic shares over time", loc="left", fontsize=11, fontweight="bold", pad=24)
    row_b = fig.row(1)
    row_b.cols([1.05, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    dominant = docs.sort_values("Distance").groupby(x).head(1).sort_values(x)
    ax_b.scatter(
        pd.to_numeric(dominant[x], errors="coerce"),
        pd.to_numeric(dominant["Topic"], errors="coerce"),
        c=[colors.get(f"Topic {int(t)}", "#0072B2") for t in pd.to_numeric(dominant["Topic"], errors="coerce")],
        s=30,
    )
    ax_b.set(xlabel=x_label, ylabel="Nearest BERT topic per document")
    ax_b.set_title("B. Each document's nearest topic", loc="left", fontsize=11, fontweight="bold", pad=8)
    _draw_terms(ax_c, list(shares.columns), keywords, None, colors)
    ax_c.set_title("C. What each topic learned", loc="left", fontsize=11, fontweight="bold", pad=8)
    return fig


def _clause_svo_showcase(frames: Mapping[str, pd.DataFrame], events: Sequence[tuple[float, str]] = ()) -> Canvas:
    """Who acts, and on whom: the most common subjects and verbs across the corpus."""
    from core.viz import kit

    frame = _require(frames, "svo.csv")
    subjects = frame["Subject"].astype(str)
    # The passive placeholder is a parser artefact, not an actor; count it separately.
    usable = frame.loc[~subjects.str.startswith("Inferred_Subject")]
    fig = canvas(
        "Who acts, and on whom?",
        "The subjects and verbs that recur in the corpus's subject-verb-object triples",
        size=(15.0, 12.0),
        library=LIBRARY,
    )
    fig.rows([1.0, 1.0], hspace=0.34)
    ax_a = fig.row(0).cols([1.0]).axis(0)
    top_subjects = usable["Subject"].astype(str).str.lower().value_counts().head(15)
    kit.bars(ax_a, top_subjects.to_numpy(), labels=[str(s) for s in top_subjects.index])
    ax_a.set_title("A. The subjects that recur most", loc="left", fontsize=11, fontweight="bold", pad=8)
    row_b = fig.row(1)
    row_b.cols([1.0, 1.0])
    ax_b = row_b.axis(0)
    ax_c = row_b.axis(1)
    top_verbs = usable["Verb"].astype(str).str.lower().value_counts().head(15)
    kit.bars(ax_b, top_verbs.to_numpy(), labels=[str(v) for v in top_verbs.index], colors="#009E73")
    ax_b.set_title("B. The verbs that recur most", loc="left", fontsize=11, fontweight="bold", pad=8)
    passive = int(subjects.str.startswith("Inferred_Subject").sum())
    _svo_shape(ax_c, frame, usable, passive)
    ax_c.set_title("C. The shape of the triples", loc="left", fontsize=11, fontweight="bold", pad=8)
    return fig


def _svo_shape(ax: Any, frame: pd.DataFrame, usable: pd.DataFrame, passive: int) -> None:
    from core.viz.static.style import INK, MUTED

    ax.set_axis_off()
    total = len(frame)
    lines = [
        f"{total:,} triples",
        f"{len(usable):,} active ({len(usable) / total:.0%})",
        f"{passive:,} passive (an inferred actor)",
        f"{usable['Subject'].nunique():,} distinct subjects",
        f"{usable['Verb'].nunique():,} distinct verbs",
        f"{usable['Object'].nunique():,} distinct objects",
    ]
    for index, line in enumerate(lines):
        ax.text(
            0.0,
            0.9 - index * 0.14,
            line,
            fontsize=10.5 if index == 0 else 9.2,
            fontweight="bold" if index == 0 else "normal",
            color=INK if index == 0 else MUTED,
        )


def _keyword_columns(topics_frame: pd.DataFrame | None) -> dict[str, str]:
    """Topic column -> its words, from MALLET's ``topics.csv`` (Topic, Words)."""
    if topics_frame is None or topics_frame.empty or "Words" not in topics_frame.columns:
        return {}
    found: dict[str, str] = {}
    for _, row in topics_frame.iterrows():
        words = str(row["Words"]).split()[:3]
        found[f"Topic {int(row['Topic'])}"] = ", ".join(words)
    return found


def _mallet_matrix(dominant: pd.DataFrame, keywords: Mapping[str, str]) -> pd.DataFrame | None:
    """Recover a document x topic share matrix from MALLET's proportion string."""
    from core.viz.panels_mallet import _parts

    if "Topic proportions" not in dominant.columns:
        return None
    rows: list[dict[str, Any]] = []
    columns: set[int] = set()
    for _, row in dominant.iterrows():
        shares = _parts(row["Topic proportions"])
        if shares is None:
            continue
        entry: dict[str, Any] = {"Document": row.get("Document", row.get("Document ID"))}
        if "Document ID" in dominant.columns:
            entry["Document ID"] = row["Document ID"]
        if "Year" in dominant.columns:
            entry["Year"] = row["Year"]
        if "Order" in dominant.columns:
            entry["Order"] = row["Order"]
        for topic, share in shares:
            entry[f"Topic {topic}"] = share
            columns.add(topic)
        rows.append(entry)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    return frame.sort_values("Document", ignore_index=True)


def _draw_mallet_weights(
    ax: Any, topics_frame: pd.DataFrame | None, keywords: Mapping[str, str], colors: Mapping[str, str]
) -> None:
    from core.viz import kit

    ax.axis("off")
    if topics_frame is None or topics_frame.empty or "Weight" not in topics_frame.columns:
        return
    ordered = topics_frame.sort_values("Topic")
    values = pd.to_numeric(ordered["Weight"], errors="coerce").fillna(0).tolist()
    names = [f"Topic {int(topic)}: {keywords.get(f'Topic {int(topic)}', '')}" for topic in ordered["Topic"]]
    bar_colors = [colors.get(f"Topic {int(topic)}", "#0072B2") for topic in ordered["Topic"]]
    kit.bars(ax, values, labels=names, colors=bar_colors)


def _registry() -> tuple[Showcase, ...]:
    return (
        Showcase(
            name="lda_gensim_showcase",
            tool="lda_gensim",
            title="What the topic model sees",
            question="Which topics does the corpus use, how do they move, and what did the model learn?",
            tables=("doc_topics.csv", "topics.csv", "terms_by_relevance.csv"),
            build=_lda_gensim_showcase,
        ),
        Showcase(
            name="lda_mallet_showcase",
            tool="lda_mallet",
            title="What MALLET sees in the corpus",
            question="Which topics does MALLET find, how do they move, and how does it weight them?",
            tables=("topics_dominant.csv", "topics.csv"),
            build=_lda_mallet_showcase,
        ),
        Showcase(
            name="readability_showcase",
            tool="readability",
            title="Did the speeches get simpler?",
            question="Does reading ease move over the corpus, or is it just document length?",
            tables=("readability.csv",),
            build=_readability_showcase,
        ),
        Showcase(
            name="sentiment_vader_anew_showcase",
            tool="sentiment_vader_anew",
            title="How did the tone move?",
            question="How does per-document sentiment move, and what are the most positive and negative sentences?",
            tables=("vader.csv", "vader_sentences.csv"),
            build=_sentiment_showcase,
        ),
        Showcase(
            name="verb_analysis_showcase",
            tool="verb_analysis",
            title="How did the grammar change?",
            question="How do verb tense, voice and modality shift across the corpus?",
            tables=("verb_summary.csv",),
            build=_verb_analysis_showcase,
        ),
        Showcase(
            name="ner_showcase",
            tool="ner",
            title="Who and where the corpus talks about",
            question="How does the mix of named-entity types move, and which entities does each era name most?",
            tables=("entity_timeline.csv",),
            build=_ner_showcase,
        ),
        Showcase(
            name="doc_similarity_showcase",
            tool="doc_similarity",
            title="Which documents read alike?",
            question="How similar are the documents, and how far back does each one's nearest match reach?",
            tables=("doc_pairs.csv",),
            build=_doc_similarity_showcase,
        ),
        Showcase(
            name="keyness_showcase",
            tool="keyness",
            title="What sets the groups apart?",
            question="Which words distinguish the groups, by effect size and by statistical evidence?",
            tables=("keyness.csv",),
            build=_keyness_showcase,
        ),
        Showcase(
            name="lexicon_series_showcase",
            tool="lexicon_series",
            title="When did the corpus start saying this?",
            question="How does each lexicon category's rate move across the corpus?",
            tables=("lexicon_series.csv",),
            build=_lexicon_series_showcase,
        ),
        Showcase(
            name="bert_topics_showcase",
            tool="bert_topics",
            title="Which topics does each era talk about?",
            question="How do the BERT topic shares move over the corpus, and what did each topic learn?",
            tables=("bert_topic_docs.csv", "bert_topics.csv"),
            build=_bert_topics_showcase,
        ),
        Showcase(
            name="clause_svo_showcase",
            tool="clause_svo",
            title="Who acts, and on whom?",
            question="Which subjects and verbs recur in the corpus's subject-verb-object triples?",
            tables=("svo.csv",),
            build=_clause_svo_showcase,
        ),
    )


SHOWCASES: tuple[Showcase, ...] = _registry()


def showcases_for(tool: str) -> list[Showcase]:
    return [showcase for showcase in SHOWCASES if showcase.tool == tool]


def showcase_for(name: str) -> Showcase | None:
    return next((showcase for showcase in SHOWCASES if showcase.name == name), None)


def render_showcase(  # noqa: PLR0913 - provenance travels as keywords, like render_bundle
    showcase: Showcase,
    frames: Mapping[str, pd.DataFrame],
    fmt: str = "png",
    *,
    source: str = "",
    settings: Mapping[str, Any] | None = None,
    sha256: str = "",
    dpi: int = 200,
    events: Sequence[tuple[float, str]] = (),
) -> Result[bytes]:
    """A showcase figure as PNG/SVG/PDF bytes, with provenance and the lint check.

    ``events`` are the project's dated events (decimal year and label), drawn
    as stoplines on every time axis. A showcase that cannot be drawn (a missing
    table, an undated corpus with no order) is a :class:`Diagnostic`, never an
    exception: the run's tables are the result and they stand without their
    picture (R4).
    """
    from core.viz.static import STATIC_FORMATS
    from core.viz.static.style import house_style

    if fmt not in STATIC_FORMATS:
        return Result.failure(Diagnostic.error("STATIC_BAD_FORMAT", f"format must be one of {STATIC_FORMATS}"))
    try:
        finished = showcase.build(frames, events)
    except (ValueError, KeyError) as exc:
        return Result.failure(Diagnostic.info("SHOWCASE_NOT_DRAWN", f"{showcase.title}: {exc}", showcase=showcase.name))
    except Exception as exc:  # a drawing bug must not take the caller down (R4)
        return Result.failure(
            Diagnostic.error("SHOWCASE_FAILED", f"{showcase.name}: {type(exc).__name__}: {exc}", showcase=showcase.name)
        )
    source_table = next((name for name in showcase.tables if name in frames), "")
    digest = sha256 or (
        hashlib.sha256(frames[source_table].to_csv(index=False).encode()).hexdigest() if source_table else ""
    )
    finished.source = Source(path=source or source_table, sha256=digest, settings=dict(settings or {}))
    try:
        fig = finished.done()
        from core.viz.static.lint import lint_figure

        problems = lint_figure(fig)
        buffer = io.BytesIO()
        metadata = {"svg": {"Date": None}, "pdf": {"CreationDate": None, "ModDate": None}}.get(fmt, {})
        with house_style():
            fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.3, metadata=metadata)
    except Exception as exc:
        return Result.failure(
            Diagnostic.error("SHOWCASE_FAILED", f"{showcase.name}: save failed: {type(exc).__name__}: {exc}")
        )
    saved = buffer.getvalue()
    return Result.success(
        _stable_svg_ids(saved) if fmt == "svg" else saved,
        *(
            Diagnostic.warning(f"STATIC_{problem.code}", f"{showcase.name}: {problem.message}", showcase=showcase.name)
            for problem in problems
        ),
    )


#: matplotlib's SVG clip-path ids: "p" and ten hex digits of a hash of the clip
#: rectangle's exact floats, in the definition and in each reference to it.
_SVG_CLIP_ID = re.compile(rb'(id="|url\(#)(p[0-9a-f]{10})')


def _stable_svg_ids(svg: bytes) -> bytes:
    """Number the clip-path ids in order of appearance (clip1, clip2, ...).

    The hash covers the rectangle's full-precision floats, and on macOS those
    differ in the last bits between two renders of the same figure, so the
    same drawing got different ids and a different file. Numbering them keeps
    every reference pointing at its definition and makes the bytes repeat.
    """
    names: dict[bytes, bytes] = {}

    def rename(match: re.Match[bytes]) -> bytes:
        name = names.setdefault(match.group(2), b"clip%d" % (len(names) + 1))
        return match.group(1) + name

    return _SVG_CLIP_ID.sub(rename, svg)
