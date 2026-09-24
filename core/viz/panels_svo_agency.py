"""SVO agency -- who acts, and who is acted upon.

``core/analysis/clause_svo.py::extract_svo`` walks a parsed corpus and emits
one row per (Subject, Verb, Object) triple it can find via dependency
structure (``ClauseSvoResult.svo_frame``, written out as ``svo.csv``). A word
frequency count over that table answers "how often does 'government'
appear?" It cannot answer the question this panel exists for: does
"government" *act* -- is it the one doing things -- or is it *acted upon*?
"Citizens" and "the bill" invite the same question, and the answer is not in
how often the word occurs, but in which grammatical slot it occupies when it
does.

For each entity (a word that shows up as a Subject or an Object somewhere in
the table), this panel counts the two slots separately and draws them as two
bars sharing one row -- the same geometry ``panels_lda_relevance.py`` uses
for Relevance against Saliency, for the same reason: the two numbers are on
one scale (both are counts of triples) so overlaying them is honest, and
their *disagreement* -- all-subject "we announce", all-object "the bill was
passed" -- is the finding a single ranked list of word frequencies would
hide.

Entity identity is normalised (stripped, casefolded) so "Government" and
"government" count as one entity rather than two accidental near-duplicates,
but the panel still needs something to put on the page -- so it keeps the
most frequent original spelling as the display form. Because a single
normalised key can stand for several original spellings, a (column, value)
filter on the raw ``Subject``/``Object`` column cannot express "any of
these"; the fix, following ``panels_lda_prevalence.py``'s ``Period`` column,
is to publish the normalisation itself as a real column
(``Subject (normalised)``, ``Object (normalised)``) in ``PreparedPanel.data``
and filter on that.

Like ``panels_keyness.py``, this is a pure function of (frame, params,
provenance): no plotly import, no filesystem, no corpus access.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
)

__all__ = ["SVO_AGENCY", "svo_agency"]

SUBJECT = "Subject"
OBJECT = "Object"
#: The derived columns this builder publishes alongside the source ones.
#: Not something ``clause_svo.py`` produces -- the normalisation is this
#: panel's own decision, and it is added to ``PreparedPanel.data`` (rather
#: than kept as a private lookup only the builder understands) so a mark's
#: evidence can filter on a real column instead of a computation a reader
#: cannot see. Mirrors the ``Period`` column in ``panels_lda_prevalence.py``.
SUBJECT_NORM = "Subject (normalised)"
OBJECT_NORM = "Object (normalised)"

_SUBJECT_GROUP = "As subject"
_OBJECT_GROUP = "As object"

#: Closed-class pronouns. Small and explicit on purpose: a heuristic that
#: tried to guess "is this a pronoun?" from shape (short, lowercase, common)
#: would misfire on plenty of real nouns ("it" the word is unambiguous;
#: "well" or "so" are not). These are the forms that, in a subject or object
#: slot, name no one -- coreference resolution is what would turn "we" back
#: into the entity it refers to.
_PRONOUNS = frozenset(
    {
        "i",
        "we",
        "you",
        "he",
        "she",
        "it",
        "they",
        "me",
        "us",
        "him",
        "her",
        "them",
        "this",
        "that",
        "who",
        "which",
    }
)


def svo_agency(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
) -> Result[PreparedPanel]:
    """Prepare ranked bars: each entity's count as Subject against as Object.

    ``top-n`` entities are drawn, ordered by ``order-by``. Each entity
    produces two marks sharing one row (one y position) so its two counts sit
    side by side rather than one hiding the other -- the same shape
    ``panels_lda_relevance.py`` uses for Relevance against Saliency.
    """
    top_n = int(params["top-n"])
    order_by = str(params["order-by"])
    min_count = int(params["min-count"])

    diagnostics: list[Diagnostic] = []
    working = frame.copy()

    subject_raw = _stripped(working[SUBJECT])
    object_raw = _stripped(working[OBJECT])
    working[SUBJECT_NORM] = subject_raw.str.casefold()
    working[OBJECT_NORM] = object_raw.str.casefold()

    # A row with neither a Subject nor an Object says nothing about who acts
    # on whom -- it is left in the published table (so a reader can still see
    # it happened) but it is worth naming, not silently invisible.
    both_blank = (working[SUBJECT_NORM] == "") & (working[OBJECT_NORM] == "")
    empty_triples = int(both_blank.sum())
    if empty_triples:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_EMPTY_TRIPLES",
                f"{empty_triples} triple(s) have neither a Subject nor an Object and contribute nothing to "
                "who-acts-on-whom; they stay in the published table but nothing is drawn from them.",
                count=empty_triples,
            )
        )
    working = working.loc[~both_blank].copy()

    if working.empty:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                "no triple in this table has a Subject or an Object to plot",
            ),
            *diagnostics,
        )

    subject_norm = working[SUBJECT_NORM]
    object_norm = working[OBJECT_NORM]
    subject_counts = subject_norm[subject_norm != ""].value_counts()
    object_counts = object_norm[object_norm != ""].value_counts()
    entities = sorted(set(subject_counts.index) | set(object_counts.index))

    display_by_entity = _display_forms(working)
    total_by_entity = {
        entity: int(subject_counts.get(entity, 0)) + int(object_counts.get(entity, 0)) for entity in entities
    }

    kept = [entity for entity in entities if total_by_entity[entity] >= min_count]
    removed = len(entities) - len(kept)
    if not kept:
        return Result[PreparedPanel].failure(
            Diagnostic.error(
                "PANEL_NO_DATA",
                f"no entity has at least {min_count} total appearance(s) as Subject or Object; lower "
                "min-count to see more entities.",
                minimum=min_count,
            ),
            *diagnostics,
        )
    if removed:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_MIN_COUNT_FILTERED",
                f"{removed} entit{'y' if removed == 1 else 'ies'} appearing fewer than {min_count} time(s) "
                "total (as Subject plus as Object) were left out; a subject/object ratio built on one or two "
                "appearances is noise, not a pattern.",
                removed=removed,
                minimum=min_count,
            )
        )

    ranked = sorted(kept, key=lambda entity: (_rank_key(order_by, subject_counts, object_counts, entity), entity))
    drawn = ranked[:top_n]

    pronoun_count = sum(1 for entity in drawn if entity in _PRONOUNS)
    if drawn and pronoun_count * 2 > len(drawn):
        diagnostics.append(
            Diagnostic.warning(
                "PANEL_PRONOUN_HEAVY",
                f"{pronoun_count} of the {len(drawn)} entities shown are pronouns ('we', 'it', 'they'...), "
                "which hide who they actually refer to -- a pronoun's role is real but its identity is not. "
                "Running coreference resolution (tool: coreference) first would resolve pronouns back to the "
                "entities they stand for before this panel counts them.",
                pronouns=pronoun_count,
                entities=len(drawn),
            )
        )

    marks: list[PanelMark] = []
    for position, entity in enumerate(drawn):
        subject_count = int(subject_counts.get(entity, 0))
        object_count = int(object_counts.get(entity, 0))
        total = total_by_entity[entity]
        display = display_by_entity[entity]
        marks.append(
            PanelMark(
                key=f"{entity}:subject",
                label=display,
                x=float(subject_count),
                y=float(position),
                group=_SUBJECT_GROUP,
                evidence=Evidence(
                    scope="sentences",
                    filters=((SUBJECT_NORM, entity),),
                    count=subject_count,
                    describe=f"{display!r} as subject: {subject_count} of {total} total appearance(s)",
                ),
            )
        )
        marks.append(
            PanelMark(
                key=f"{entity}:object",
                label=display,
                x=float(object_count),
                y=float(position),
                group=_OBJECT_GROUP,
                evidence=Evidence(
                    scope="sentences",
                    filters=((OBJECT_NORM, entity),),
                    count=object_count,
                    describe=f"{display!r} as object: {object_count} of {total} total appearance(s)",
                ),
            )
        )

    prepared = PreparedPanel(
        panel=SVO_AGENCY.name,
        shape="ranked_bars",
        title="Who acts, who is acted upon",
        subtitle=f"{len(drawn)} of {len(kept)} entities (min-count={min_count}), ranked by {order_by}",
        marks=tuple(marks),
        x_label="Triples (count)",
        y_label=f"Entities, ranked by {order_by}",
        provenance=provenance,
        data=working,
        groups=(_SUBJECT_GROUP, _OBJECT_GROUP),
        notes=_NOTES,
    )
    return Result.success(prepared, *diagnostics)


def _stripped(series: pd.Series) -> pd.Series:
    """Original text with NaN/None treated as empty and whitespace trimmed.

    Blank detection has to happen before normalisation touches the text, so
    this is the one place both the display form and the casefolded key are
    built from -- a row where Subject is genuinely ``NaN`` (a clause with no
    object, or vice versa) must read as "nothing here", not as the four
    characters "n a n".
    """
    text = series.where(series.notna(), "")
    return text.astype(str).str.strip()


def _display_forms(working: pd.DataFrame) -> dict[str, str]:
    """The most frequent original spelling for each normalised entity.

    Casefolding merges "Government" and "government" so they are counted as
    one entity, but the figure still has to print something. Ties (equally
    frequent spellings) break alphabetically, so a redraw over the same data
    always picks the same display form.
    """
    subject_pairs = pd.DataFrame({"norm": working[SUBJECT_NORM], "raw": _stripped(working[SUBJECT])})
    object_pairs = pd.DataFrame({"norm": working[OBJECT_NORM], "raw": _stripped(working[OBJECT])})
    pairs = pd.concat([subject_pairs, object_pairs], ignore_index=True)
    pairs = pairs.loc[pairs["norm"] != ""]

    display: dict[str, str] = {}
    for norm, group in pairs.groupby("norm"):
        counts = group["raw"].value_counts()
        best = counts.max()
        display[str(norm)] = min(counts[counts == best].index)
    return display


def _rank_key(
    order_by: str,
    subject_counts: pd.Series,
    object_counts: pd.Series,
    entity: str,
) -> float:
    """The sort key that puts the requested ordering first (ascending sort).

    Negated so ``sorted(..., key=...)`` -- ascending by construction --
    produces a descending ranking without a separate ``reverse=True``, which
    would also have reversed the alphabetical tie-break the caller appends.
    "balance" is (subject - object) / total: +1 means the entity is never
    acted upon, -1 means it never acts, 0 means the two slots are even. There
    is no blended default across these four; each answers a different
    question, the same refusal ``panels_keyness.py``'s ``label-by`` and
    ``panels_lda_relevance.py``'s ``order-by`` make.
    """
    subject = int(subject_counts.get(entity, 0))
    obj = int(object_counts.get(entity, 0))
    total = subject + obj
    if order_by == "subject":
        return -float(subject)
    if order_by == "object":
        return -float(obj)
    if order_by == "balance":
        return -((subject - obj) / total) if total else 0.0
    return -float(total)


_NOTES: tuple[str, ...] = (
    "Subject and Object are the output of a dependency parser and a heuristic role-remapper "
    "(core/analysis/clause_svo.py::extract_svo), not hand-checked ground truth. Passive voice is corrected "
    "only when the parser tags it unambiguously: a clear nsubj:pass/nsubjpass patient, together with an "
    "obl:agent/agent phrase when one is present, is swapped so the true agent lands in Subject and the "
    "patient in Object. A passive clause the parser only partly tags -- the auxiliary marked aux:pass but the "
    "patient left as a plain nsubj -- is not corrected, and the patient sits in the Subject column looking "
    "like an agent it is not.",
    "An agentless passive ('the bill was passed.', no 'by X') has no surface agent to put in Subject at all; "
    "extract_svo fills that slot with the literal placeholder 'Inferred_Subject_Passive' rather than a word "
    "from the text. It appears here as an entity like any other -- a high subject count for it usually means "
    "a lot of agentless passives, not a person or institution acting.",
    "A pronoun-heavy corpus hides its real entities behind 'we', 'it' and 'they' unless coreference "
    "resolution has been run first: a pronoun's grammatical role is counted correctly, but its identity is "
    "not resolved to the noun it stands for. PANEL_PRONOUN_HEAVY names when this is happening to more than "
    "half of what is drawn.",
    "Counts are over extracted triples, not over mentions of the word anywhere in the corpus: an entity "
    "discussed constantly but rarely placed in a subject or object slot -- inside a prepositional phrase, a "
    "modifier, an appositive -- will look minor here even though it is central to the text.",
    "'balance' -- (subject count - object count) / total -- is a ratio, and a ratio built on a handful of "
    "appearances swings wildly with one more or fewer triple. That instability is exactly what min-count "
    "exists to guard against; raising it trades coverage for a ranking worth trusting.",
)


SVO_AGENCY = PanelDefinition(
    name="svo_agency",
    title="SVO agency",
    question="Who acts, and who is acted upon?",
    tool="clause_svo",
    shape="ranked_bars",
    summary="Per entity, how often it is the grammatical Subject versus the Object of an extracted clause.",
    requires=(SUBJECT, OBJECT),
    params=(
        PanelParam(
            name="top-n",
            type="int",
            default=20,
            minimum=1,
            maximum=100,
            label="Entities to show",
            help="How many entities to draw, taken from the top of the chosen ranking.",
        ),
        PanelParam(
            name="order-by",
            type="choice",
            default="total",
            choices=("total", "subject", "object", "balance"),
            label="Rank by",
            help=(
                "Which quantity orders the entities: 'total' (subject count + object count, the most "
                "frequently-placed entities overall), 'subject' (most often the one acting), 'object' (most "
                "often the one acted upon), or 'balance' ((subject - object) / total, most agentive first, "
                "most acted-upon last). There is no blended default."
            ),
        ),
        PanelParam(
            name="min-count",
            type="int",
            default=3,
            minimum=1,
            maximum=10_000,
            label="Minimum total appearances",
            help=(
                "Leave out entities appearing fewer than this many times total (as Subject plus as Object). "
                "A subject/object ratio built on one or two appearances is noise, not a pattern."
            ),
        ),
    ),
    build=svo_agency,
    notes=_NOTES,
)
