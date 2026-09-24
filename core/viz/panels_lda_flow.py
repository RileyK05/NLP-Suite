"""Topic flow — each speech as a ribbon of its paragraphs' topics.

The topic model scores a speech as a mixture and the dominant-topic table
reduces that to one topic per speech. Neither can show structure: whether a
speech opens on the economy, turns to war and closes on a conclusion, or stays
on one subject for pages. This panel draws each speech as a horizontal band
cut into its paragraphs, each coloured by the topic that dominates it, one band
per speech. Read across a band, it is that speech's outline; read down the
bands, it is how the shape of the address changed.

The table comes from ``lda_gensim``'s ``topic_flow.csv``: the fitted model
scores every paragraph while it is still in hand (``fit_lda(..., segments=)``),
with paragraphs cut by ``core/analysis/topic_flow.py``.

Four decisions worth stating:

* **The segmentation rule is reported, every time** (``PANEL_SEGMENTATION``).
  Blank-line paragraphs, one paragraph per line, or fixed sentence windows --
  the State of the Union files have no blank lines at all, so the rule changes
  the picture, and a figure that did not say which one drew it would be asking
  for trust in an unstated decision. A speech whose tokens could not be placed
  in its text is named: it fell back to sentence windows rather than being cut
  by guesswork.
* **A paragraph too short to score is a gap, not a guess** (``min-tokens``).
  A one-line salutation leaves two or three words after stopwords, and its
  "topic" is noise. It is left blank on the ribbon, counted in a notice, and
  never recoloured to match its neighbours.
* **Colours match the prevalence panel.** Bands are grouped ``Topic n`` in
  topic order, exactly as ``lda_prevalence`` groups them, so topic 3 is the
  same colour in both figures and a reader can move between them.
* **Switching is counted, and not judged.** Each speech's describe line says
  how many times its dominant topic changes between paragraphs and how long
  its longest unbroken run is. Frequent switching is not incoherence -- a
  State of the Union is a list of subjects by design -- and the notes say so,
  because a count invites exactly that reading.

Ribbon geometry, as ``panel_plotters`` and ``panelLayout.ts`` draw it: a
mark's ``x`` is where its segment starts along the band, ``size`` is how wide
it is (same units), and ``y`` is the band's row, 0 at the top.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
import math
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = ["LDA_FLOW", "lda_flow"]

DOCUMENT_ID = "Document ID"
DOCUMENT = "Document"
SEGMENT = "Segment"
SEGMENTS = "Segments"
RULE = "Rule"
ALIGNED = "Aligned"
TOKENS = "Tokens"
DOMINANT = "Dominant topic"
CONTRIBUTION = "Contribution"
KEYWORDS = "Topic keywords"

_KEYWORDS_SHOWN = 3


def lda_flow(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Prepare the ribbons: one band per speech, one mark per scored paragraph."""
    limit = int(params["documents"])
    min_tokens = int(params["min-tokens"])
    align = str(params["align"])
    order = str(params["order"])

    diagnostics: list[Diagnostic] = []
    working = frame.copy()
    for column in (SEGMENT, SEGMENTS, TOKENS):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working[DOMINANT] = pd.to_numeric(working[DOMINANT], errors="coerce")
    working[CONTRIBUTION] = pd.to_numeric(working[CONTRIBUTION], errors="coerce")
    working = working[working[SEGMENT].notna() & working[SEGMENTS].notna()]
    if working.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error("PANEL_NO_DATA", "the topic flow table has no segments to draw")
        )
    working[DOCUMENT] = working[DOCUMENT].astype(str)

    diagnostics.append(_segmentation_notice(working))

    scored = working[DOMINANT].notna() & (working[TOKENS] >= min_tokens)
    short = int((~scored).sum())
    if short:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_SHORT_SEGMENTS",
                f"{short} of {len(working)} paragraphs had fewer than {min_tokens} words left after stopwords, "
                "and are shown as gaps rather than coloured by a guess. Lower min-tokens to show them.",
                segments=short,
                minimum=min_tokens,
            )
        )

    speeches = _speeches(working, scored)
    if not speeches:
        return Result[PreparedPanel].failure(
            *diagnostics,
            Diagnostic.error(
                "PANEL_NO_DATA",
                f"no paragraph has {min_tokens} or more words to score; lower min-tokens",
            ),
        )
    if order == "switches":
        # Most switching first: the speeches that move between subjects most
        # often. Name breaks ties so a redraw orders them the same way.
        speeches.sort(key=lambda s: (-s["switches"], s["name"]))
    else:
        # Names in this corpus lead with an ISO date, so name order is
        # chronological; for any other naming it is at least stable.
        speeches.sort(key=lambda s: s["name"])
    if len(speeches) > limit:
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_DOCUMENTS_CAPPED",
                f"{len(speeches)} speeches is more than a readable set of ribbons; drew the first {limit} in "
                f"{order} order. Raise documents to see more.",
                drawn=limit,
                available=len(speeches),
            )
        )
        speeches = speeches[:limit]

    topics = sorted({int(t) for s in speeches for t in s["rows"][DOMINANT].dropna()})
    keywords = _keywords(working)
    # A topic is named by its words ("Topic 2: world, peace, nations"): a
    # legend of "Topic 0 ... Topic 7" says nothing a reader can use.
    names = {t: f"Topic {t}: {keywords[t]}" if keywords.get(t) else f"Topic {t}" for t in topics}
    # Speeches as a reader names them ("1934 Roosevelt"), not by file.
    speech_label = document_labels(s["name"] for s in speeches)
    marks: list[PanelMark] = []
    kept_rows: list[pd.DataFrame] = []
    for row_number, speech in enumerate(speeches):
        rows = speech["rows"]
        total = int(rows[SEGMENTS].iloc[0])
        kept_rows.append(rows)
        for _, segment in rows[rows["_scored"]].iterrows():
            number = int(segment[SEGMENT])
            topic = int(segment[DOMINANT])
            start, width = ((number - 1) / total, 1 / total) if align == "relative" else (float(number - 1), 1.0)
            share = float(segment[CONTRIBUTION]) if _finite(segment[CONTRIBUTION]) else float("nan")
            words = keywords.get(topic, "")
            marks.append(
                PanelMark(
                    key=f"{speech['name']}:{number}",
                    label=speech_label[speech["name"]],
                    x=start,
                    y=float(row_number),
                    size=width,
                    group=names[topic],
                    evidence=Evidence(
                        scope="documents",
                        filters=((DOCUMENT, speech["name"]), (SEGMENT, str(number))),
                        count=1,
                        describe=(
                            f"{speech['name']}, paragraph {number} of {total}: topic {topic}"
                            f"{f' ({words})' if words else ''}"
                            f"{f' at {share:.0%}' if math.isfinite(share) else ''}, "
                            f"{int(segment[TOKENS])} words · this speech changes topic {speech['switches']} "
                            f"time(s); its longest run is {speech['longest']} paragraph(s)"
                        ),
                    ),
                )
            )

    data = pd.concat(kept_rows).drop(columns=["_scored"]).reset_index(drop=True)
    data[SEGMENT] = data[SEGMENT].astype(int)
    prepared = PreparedPanel(
        panel=LDA_FLOW.name,
        shape="ribbon",
        title="Topic flow: each speech, paragraph by paragraph",
        subtitle=(
            f"{len(speeches)} speeches · {order} order · "
            f"{'stretched to one length' if align == 'relative' else 'true length, one unit per paragraph'}"
        ),
        marks=tuple(marks),
        x_label="Position in the speech (start to end)" if align == "relative" else "Paragraph",
        y_label="Speeches",
        provenance=provenance,
        data=data,
        groups=tuple(names[t] for t in topics),
        notes=_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _speeches(working: pd.DataFrame, scored: pd.Series) -> list[dict[str, Any]]:
    """Each speech's rows in paragraph order, with its switch count and longest run."""
    marked = working.assign(_scored=scored)
    speeches: list[dict[str, Any]] = []
    for name, group in marked.groupby(DOCUMENT, sort=False):
        rows = group.sort_values(SEGMENT, kind="stable")
        sequence = [int(t) for t in rows.loc[rows["_scored"], DOMINANT]]
        if not sequence:
            continue
        # Counted over scored paragraphs only: a gap is "too short to say",
        # and treating it as a change of topic would count a salutation line
        # as two switches.
        switches = sum(1 for a, b in pairwise(sequence) if a != b)
        longest = run = 1
        for a, b in pairwise(sequence):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        speeches.append({"name": str(name), "rows": rows, "switches": switches, "longest": longest})
    return speeches


def _keywords(working: pd.DataFrame) -> dict[int, str]:
    """The first few keywords the model gave each topic, for readable labels."""
    found: dict[int, str] = {}
    if KEYWORDS not in working.columns:
        return found
    for topic, words in zip(working[DOMINANT], working[KEYWORDS], strict=True):
        if not _finite(topic) or int(topic) in found or not isinstance(words, str):
            continue
        found[int(topic)] = ", ".join(word.strip() for word in words.split(",")[:_KEYWORDS_SHOWN])
    return found


def _segmentation_notice(working: pd.DataFrame) -> Diagnostic:
    """Which rule cut the speeches -- always stated, because it shapes the figure."""
    per_speech = working.groupby(DOCUMENT, sort=False)[RULE].first()
    counts = per_speech.value_counts()
    described = {
        "blank-line": "by blank lines",
        "line": "one paragraph per line",
        "sentences": "into fixed sentence windows",
    }
    parts = [f"{int(n)} {described.get(str(rule), str(rule))}" for rule, n in counts.items()]
    unaligned: list[str] = []
    if ALIGNED in working.columns:
        aligned = working.groupby(DOCUMENT, sort=False)[ALIGNED].first()
        unaligned = [str(name) for name, ok in aligned.items() if str(ok).lower() in ("false", "0")]
    tail = ""
    if unaligned:
        shown = ", ".join(unaligned[:3])
        tail = (
            f" {len(unaligned)} speech(es) could not be matched to their own text and were cut into sentence "
            f"windows instead, e.g. {shown}."
        )
    return Diagnostic.info(
        "PANEL_SEGMENTATION",
        f"Speeches were cut into paragraphs as follows: {'; '.join(parts)}.{tail}",
        rules={str(rule): int(n) for rule, n in counts.items()},
        unaligned=unaligned,
    )


_NOTES: tuple[str, ...] = (
    "Each paragraph is coloured by the one topic with the largest share of it. A paragraph mixing two "
    "subjects is shown as whichever leads, so a ribbon is an outline, not a full account.",
    "The topics were learned from whole speeches and then scored on paragraphs. A paragraph is short, so "
    "its topic is a noisier estimate than a speech's; gaps mark paragraphs too short to score at all.",
    "Changing topic often is not incoherence. An annual address is a list of subjects by design; the switch "
    "count describes structure, not quality, and says nothing about how well the speech reads.",
    "Stretched to one length, a long speech and a short one start and end together, which suits comparing "
    "their shape and hides how much longer one is. Switch align to true length to see that.",
    "How speeches were cut into paragraphs changes the picture; the segmentation notice says which rule "
    "applied to which speeches.",
)


LDA_FLOW = PanelDefinition(
    name="lda_flow",
    title="Topic flow",
    question="How does each speech move from subject to subject, paragraph by paragraph?",
    tool="lda_gensim",
    shape="ribbon",
    summary="Each speech as a band of its paragraphs, each coloured by the topic that dominates it.",
    requires=(DOCUMENT, SEGMENT, SEGMENTS, RULE, TOKENS, DOMINANT, CONTRIBUTION),
    params=(
        PanelParam(
            name="documents",
            type="int",
            default=30,
            minimum=1,
            maximum=200,
            label="Speeches to show",
            help="How many speeches to draw as ribbons, taken from the start of the chosen order.",
        ),
        PanelParam(
            name="order",
            type="choice",
            default="chronological",
            choices=("chronological", "switches"),
            label="Order speeches by",
            help=(
                "'chronological' by name (these names lead with a date); 'switches' puts the speeches "
                "that change topic most often first."
            ),
        ),
        PanelParam(
            name="align",
            type="choice",
            default="relative",
            choices=("relative", "absolute"),
            label="Length",
            help=(
                "'relative' stretches every speech to one length so their shapes line up; 'absolute' gives "
                "each paragraph one unit so a longer speech is a longer band."
            ),
        ),
        PanelParam(
            name="min-tokens",
            type="int",
            default=5,
            minimum=1,
            maximum=500,
            label="Words needed to score a paragraph",
            help=(
                "Paragraphs with fewer words left after stopwords are shown as gaps: a topic read from two "
                "or three words is a guess."
            ),
        ),
    ),
    build=lda_flow,
    notes=_NOTES,
)
