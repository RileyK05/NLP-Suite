"""Intertopic Distance Map — pyLDAvis's left-hand chart, natively.

``core.analysis.lda.intertopic_map`` places each fitted topic by
multidimensional scaling (MDS) of the Jensen-Shannon distance between the
topics' word distributions, and scales each topic's marker by its
prevalence. That is exactly the geometry pyLDAvis has drawn since it
popularised topic-model diagnostics, computed here without the bundled JS
dependency: the numbers are the graded thing, and this panel is one Plotly
call away from them.

The figure answers a question no bar chart of top words can: *which topics
overlap in vocabulary, and which stand alone?* Two topics close together use
similar words even if a human reader would never guess it from either
topic's top-10 list; two topics far apart share almost no vocabulary. That
is a property of the whole fit, not of any single topic, which is why it
needs its own panel rather than living beside the per-topic word tables.

**The single most misread thing about this chart**: the X/Y axes are MDS
coordinates with no inherent unit and no individual meaning. MDS only
promises to preserve *relative* distances between points as faithfully as a
2-D layout of an inherently higher-dimensional dissimilarity matrix allows;
it promises nothing about what "up" or "right" mean, and a rotated or
mirrored layout is exactly as correct as the original. Reading "topic 3 has
a high X value" as a finding is reading noise. Reading "topic 3 sits far
from every other topic" is reading the map correctly. This panel says so in
an :class:`~core.viz.panelspec.Annotation`, in ``notes``, and in the axis
labels themselves, because it is easy to build a chart that is technically
correct and confidently misleading.

The second thing worth stating plainly: ``Prevalence`` here is *not* a
literal share of corpus tokens that sums to 1 across topics. Reading
``intertopic_map``, the engine rescales each topic's mass by the single
largest topic's mass (``scale = masses.max()``), so the biggest topic is
always exactly 1.0 and every other topic is reported relative to it. Bubble
area follows that rescaled number, matching pyLDAvis's convention (circle
area is prevalence) without pyLDAvis's separate percentage-of-tokens axis,
which this panel does not attempt to reconstruct.

Like ``panels_keyness.keyness_volcano``, this is a pure function of (frame,
params): no plotly import, no filesystem, no corpus access. It is the
second panel written against the contract in ``core/viz/panelspec.py`` and
copies ``panels_keyness.py``'s structure deliberately.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panelspec import (
    Annotation,
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = ["LDA_INTERTOPIC", "lda_intertopic"]

TOPIC = "Topic"
X = "X"
Y = "Y"
PREVALENCE = "Prevalence"


def lda_intertopic(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Prepare the map: one bubble per topic, positioned by MDS, sized by prevalence."""
    label_topics = str(params["label-topics"])
    label_top = int(params["label-top"])

    diagnostics: list[Diagnostic] = []

    # `intertopic_map` itself returns an empty table for k < 2 -- there is no
    # MDS layout for fewer than two points, since MDS exists to place points
    # *relative to one another*. Checked here too, defensively, because a
    # builder is a pure function and must not assume its caller already
    # enforced that (this module's own tests call it directly).
    if len(frame) < 2:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                "fewer than 2 topics to plot. A distance map exists to show how topics sit relative to one "
                "another; with 0 or 1 topics there is nothing to be relative to. Re-fit LDA with n_topics >= 2.",
            )
        )

    working = frame.copy()
    # Numbers first, same discipline as the keyness volcano: a topic whose X,
    # Y or Prevalence is not a finite number cannot be placed on the page,
    # and placing it at the origin or at zero size would invent a finding.
    for column in (X, Y, PREVALENCE):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    placeable = working[X].apply(_is_finite) & working[Y].apply(_is_finite) & working[PREVALENCE].apply(_is_finite)
    unplaceable = int((~placeable).sum())
    if unplaceable:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_BAD_NUMERIC",
                f"{unplaceable} topic(s) had a non-numeric or infinite X, Y or Prevalence and were left out of "
                "the map; the table beside it still shows them.",
                dropped=unplaceable,
            )
        )
    working = working[placeable]

    if len(working) < 2:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                "fewer than 2 topics left to plot once unplaceable rows were dropped. A distance map of one "
                "topic is not a thing.",
                remaining=len(working),
            ),
            *diagnostics,
        )

    # A real branch in `intertopic_map`: when every pairwise Jensen-Shannon
    # distance is exactly zero, the engine skips MDS's random init entirely
    # and returns every topic at (0, 0) rather than an arbitrary scatter from
    # a zero-distance matrix. That is a fact about the fit worth surfacing,
    # not a rendering accident -- checked generally (same X and Y across all
    # rows), not by assuming the engine's particular (0, 0) convention.
    distinct_positions = working[[X, Y]].drop_duplicates()
    if len(distinct_positions) == 1:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_TOPICS_IDENTICAL",
                f"all {len(working)} topics landed at the same point. Their word distributions are effectively "
                "identical, which usually means more topics were requested than the corpus actually supports -- "
                "try refitting with a smaller n_topics.",
                topics=len(working),
            )
        )

    labelled_topics = _labelled_topics(working, label_topics, label_top)

    marks: list[PanelMark] = []
    for _, row in working.iterrows():
        topic = int(row[TOPIC])
        x_value, y_value, prevalence = float(row[X]), float(row[Y]), float(row[PREVALENCE])
        marks.append(
            PanelMark(
                key=str(topic),
                label=f"Topic {topic}",
                x=x_value,
                y=y_value,
                group="",
                size=prevalence,
                labelled=topic in labelled_topics,
                evidence=Evidence(
                    scope="rows",
                    filters=((TOPIC, str(topic)),),
                    count=1,
                    describe=(
                        f"Topic {topic}: relative prevalence {prevalence:.3f} "
                        "(1.0 = the most prevalent topic in this fit)"
                    ),
                ),
            )
        )

    prepared = PreparedPanel(
        panel=LDA_INTERTOPIC.name,
        shape="scatter_labelled",
        title="Intertopic Distance Map",
        subtitle=_subtitle(len(marks)),
        marks=tuple(marks),
        # Named as MDS coordinates, not as any interpretable dimension --
        # inventing an axis name here ("formality", "sentiment", whatever)
        # would be the exact misreading this panel exists to head off.
        x_label="MDS dimension 1 (no meaning alone -- only distance between points does)",
        y_label="MDS dimension 2 (no meaning alone -- only distance between points does)",
        provenance=provenance,
        data=working,
        annotations=(
            Annotation(
                kind="note",
                value=0.0,
                label="axes have no independent meaning",
                note=(
                    "These are MDS coordinates, not measurements. MDS only preserves the relative distance "
                    "between topics as faithfully as a 2-D layout of a higher-dimensional distance can; it "
                    "promises nothing about what either axis individually means, and a mirrored or rotated "
                    "layout is exactly as correct as this one. Only how far apart two bubbles are is meaningful."
                ),
            ),
        ),
        groups=(),
        notes=_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def _labelled_topics(frame: pd.DataFrame, label_topics: str, label_top: int) -> set[int]:
    """Which topic numbers get their label written, per ``label-topics``.

    Labelling every bubble in a fifty-topic model produces an unreadable
    smear over the densest part of the map, so ``"largest"`` exists to let a
    caller ask for only the topics that matter most by prevalence. Ties
    break on topic number so a redraw labels the same bubbles.
    """
    if label_topics == "none":
        return set()
    if label_topics == "largest":
        ranked = frame.sort_values([PREVALENCE, TOPIC], ascending=[False, True], kind="stable")
        return set(ranked.head(label_top)[TOPIC].astype(int))
    return set(frame[TOPIC].astype(int))


def _subtitle(topic_count: int) -> str:
    """One line of orientation: how many topics, and what bubble size means."""
    return f"{topic_count} topics · bubble area is prevalence relative to the largest topic"


def _is_finite(value: Any) -> bool:
    """True for a real, finite number. NaN and +/-inf cannot be placed."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


_NOTES: tuple[str, ...] = (
    "The X and Y axes are MDS coordinates with no meaning on their own. Only the distance between two "
    "bubbles is meaningful: close together means similar word distributions, far apart means dissimilar. "
    "Do not read either axis as a dimension with a direction or a name.",
    "The layout is a 2-D projection of a distance that is computed in a much higher-dimensional space (one "
    "dimension per vocabulary word). Multidimensional scaling preserves relative distance as well as it can "
    "in two dimensions, but some distortion from the true distances is unavoidable.",
    "Bubble area is prevalence, relative to the single most prevalent topic in this fit (the largest bubble "
    "is always 1.0). A large bubble sitting far from the others is a big topic that is genuinely unlike the "
    "rest of the model, not an outlier to discard.",
    "Topic numbers are arbitrary labels assigned during this fit's word-sorting, not stable identities. "
    "Refitting the same corpus with a different seed or a different n_topics can and usually will renumber, "
    "reposition, or split and merge topics; do not track a topic across runs by its number alone.",
)


LDA_INTERTOPIC = PanelDefinition(
    name="lda_intertopic",
    title="Intertopic Distance Map",
    question="Which topics overlap, and which are the model's biggest?",
    tool="lda_gensim",
    shape="scatter_labelled",
    summary="Topics placed by MDS distance between their word distributions, one bubble per topic sized by prevalence.",
    requires=(TOPIC, X, Y, PREVALENCE),
    params=(
        PanelParam(
            name="label-topics",
            type="choice",
            default="all",
            choices=("all", "none", "largest"),
            label="Label topics",
            help=(
                "Which bubbles get their topic number written on the map. 'all' labels every topic, 'none' "
                "labels none, and 'largest' labels only the most prevalent few -- useful for models with many "
                "topics, where labelling every bubble produces an unreadable smear."
            ),
        ),
        PanelParam(
            name="label-top",
            type="int",
            default=10,
            minimum=1,
            maximum=100,
            label="Topics to label",
            help=(
                "How many bubbles get a label when label-topics is 'largest', ranked by prevalence. "
                "Ignored when label-topics is 'all' or 'none'."
            ),
        ),
    ),
    build=lda_intertopic,
    notes=_NOTES,
)
