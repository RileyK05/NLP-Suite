"""Two tools whose tables support one more honest figure each.

* ``clause_svo`` -- the most common subject-verb-object triples, with how
  many speeches each comes from (the agency view already answers "who acts";
  this answers "what is said to be done, and how widely").
* ``quote_annotator`` -- who is quoted most, and how many quotes each
  speech carries and how many name their speaker, through the shared
  per-document factory.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import is_function_word
from core.viz.panels_document_measures import MeasureTool, measure_panels
from core.viz.panelspec import Evidence, PanelDefinition, PanelMark, PanelParam, PreparedPanel, Provenance

__all__ = ["EXTRAS_PANELS"]

DOC_ID = "Document ID"
DOC = "Document"
DATE = "Date"
TRIPLE = "Triple"
#: The SVO extractor's placeholder for a subject it inferred (a passive with
#: no agent), not a word anyone said.
_INFERRED = "Inferred_"


def _ranked(  # noqa: PLR0913 - keyword-only; shared by the two rankings
    counts: pd.DataFrame, *, key: str, definition: PanelDefinition, provenance: Provenance, unit: str, documents: int
) -> PreparedPanel:
    marks = tuple(
        PanelMark(
            key=str(row[key]),
            label=str(row[key]),
            x=float(row["Count"]),
            y=float(rank),
            evidence=Evidence(
                scope="rows",
                filters=((key, str(row[key])),),
                count=int(row["Count"]),
                describe=f"{row[key]}: {int(row['Count'])} {unit}, in {int(row['Documents'])} of {documents} speeches",
            ),
        )
        for rank, (_, row) in enumerate(counts.iterrows())
    )
    return PreparedPanel(
        panel=definition.name,
        shape="ranked_bars",
        title=definition.title,
        subtitle=f"top {len(marks)}, with the number of speeches each comes from",
        marks=marks,
        x_label=unit.capitalize(),
        y_label="",
        provenance=provenance,
        data=counts.reset_index(drop=True),
        notes=definition.notes,
    )


def _triples(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    working = frame.copy()
    for column in ("Subject", "Verb", "Object"):
        working[column] = working[column].astype(str).str.strip().str.lower()
    removed: list[str] = []
    if bool(params.get("hide-inferred", True)):
        inferred = working["Subject"].str.startswith(_INFERRED.lower())
        if inferred.any():
            removed.append(f"{int(inferred.sum())} with an inferred subject")
        working = working[~inferred]
    if bool(params.get("hide-pronoun-only", False)):
        pronouns = working["Subject"].map(is_function_word) & working["Object"].map(is_function_word)
        removed.append(f"{int(pronouns.sum())} made only of pronouns")
        working = working[~pronouns]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no triples left to rank; turn a filter off"))
    working[TRIPLE] = working["Subject"] + " / " + working["Verb"] + " / " + working["Object"]
    counts = (
        working.groupby(TRIPLE, sort=False)
        .agg(Count=(TRIPLE, "size"), Documents=(DOC_ID, "nunique"))
        .reset_index()
        .sort_values(["Count", TRIPLE], ascending=[False, True], kind="stable")
        .head(int(params.get("top-n", 25)))
    )
    diagnostics = (
        [Diagnostic.info("PANEL_TRIPLES_FILTERED", "Left out: " + "; ".join(removed) + ".")] if removed else []
    )
    panel = _ranked(
        counts,
        key=TRIPLE,
        definition=SVO_TOP_TRIPLES,
        provenance=provenance,
        unit="occurrences",
        documents=int(frame[DOC_ID].nunique()),
    )
    return Result.success(panel, *diagnostics)


SVO_TOP_TRIPLES = PanelDefinition(
    name="clause_svo_top_triples",
    title="Most common subject / verb / object",
    question="What is most often said to be done, by whom, and to what?",
    tool="clause_svo",
    shape="ranked_bars",
    summary="Subject-verb-object triples ranked by how often they occur, with speech coverage.",
    requires=("Subject", "Verb", "Object", DOC_ID),
    params=(
        PanelParam(
            name="top-n", type="int", default=25, minimum=1, maximum=200, label="Triples", help="How many to rank."
        ),
        PanelParam(
            name="hide-inferred",
            type="bool",
            default=True,
            label="Hide inferred subjects",
            help="Leave out triples whose subject the extractor inferred for a passive with no agent.",
        ),
        PanelParam(
            name="hide-pronoun-only",
            type="bool",
            default=False,
            label="Hide pronoun-only triples",
            help="Leave out triples whose subject and object are both pronouns ('we / do / it').",
        ),
    ),
    build=_triples,
    notes=(
        "A triple is counted each time the extractor finds it; 'in N speeches' says whether it is a formula of "
        "the genre or one speech's refrain.",
        "Pronouns are kept by default because 'we' is the genre's main actor; hide them to see what is done to "
        "named things.",
    ),
)


def _speakers(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    working = frame.copy()
    working["Speaker"] = working["Speaker"].astype(str).str.strip()
    attributed = working[~working["Speaker"].str.lower().isin(("", "none", "nan"))].copy()
    # "he" and "He" are one speaker; shown in the form the table uses most.
    attributed["_key"] = attributed["Speaker"].str.casefold()
    display = attributed.groupby("_key")["Speaker"].agg(lambda names: names.value_counts().idxmax())
    attributed["Speaker"] = attributed["_key"].map(display)
    hidden = 0
    if bool(params.get("hide-function-words", True)):
        # On the real corpus the top "speakers" were he, I, Shall, have: a
        # pronoun or a verb picked up as the subject of a speech verb.
        function = attributed["_key"].map(is_function_word)
        hidden = int(function.sum())
        attributed = attributed[~function]
    if attributed.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no quote in this table is attributed to a named speaker")
        )
    counts = (
        attributed.groupby("Speaker", sort=False)
        .agg(Count=("Speaker", "size"), Documents=(DOC_ID, "nunique"))
        .reset_index()
        .sort_values(["Count", "Speaker"], ascending=[False, True], kind="stable")
        .head(int(params.get("top-n", 25)))
    )
    unattributed = len(working) - len(attributed) - hidden
    panel = _ranked(
        counts,
        key="Speaker",
        definition=QUOTE_SPEAKERS,
        provenance=provenance,
        unit="quotes",
        documents=int(frame[DOC_ID].nunique()),
    )
    note = Diagnostic.info(
        "PANEL_UNATTRIBUTED",
        f"{unattributed} of {len(working)} quotes have no speaker, and {hidden} are attributed to a pronoun or "
        "function word; neither is ranked.",
        unattributed=unattributed,
        hidden=hidden,
    )
    return Result.success(panel, note)


QUOTE_SPEAKERS = PanelDefinition(
    name="quote_annotator_speakers",
    title="Who is quoted",
    question="Whose words does the corpus quote most, and across how many speeches?",
    tool="quote_annotator",
    shape="ranked_bars",
    summary="Quote speakers ranked by how many quotes are attributed to them.",
    requires=("Quote", "Speaker", "Cue", DOC_ID),
    params=(
        PanelParam(
            name="top-n", type="int", default=25, minimum=1, maximum=200, label="Speakers", help="How many to rank."
        ),
        PanelParam(
            name="hide-function-words",
            type="bool",
            default=True,
            label="Hide pronoun speakers",
            help="Leave out quotes attributed to a pronoun or function word ('he', 'I', 'have').",
        ),
    ),
    build=_speakers,
    notes=(
        "Attribution takes the subject of the speech verb, so a pronoun ('he said') or a mis-parsed word can "
        "come out as the speaker; those are hidden by default, and the notice says how many.",
        "Unattributed quotes are counted in a notice, not drawn.",
    ),
)


def _quote_shares(frame: pd.DataFrame) -> Result[pd.DataFrame]:
    """The share of a speech's quotes that name their speaker.

    ``Words`` in this summary counts the *quoted* words, not the speech's,
    so no per-length rate can be made from it: a "quotes per 1,000 words"
    here would be quotes per thousand quoted words, which measures nothing.
    """
    working = frame.copy()
    quotes = pd.to_numeric(working["Quotes"], errors="coerce")
    working["% attributed"] = np.where(quotes > 0, 100.0 * pd.to_numeric(working["Attributed"]) / quotes, np.nan)
    return Result.success(working)


_QUOTES = MeasureTool(
    tool="quote_annotator",
    requires=(DOC_ID, DOC, DATE, "Quotes", "Attributed", "Unattributed", "Words"),
    prepare=_quote_shares,
    measures=("Quotes", "% attributed", "Words"),
    default_measure="Quotes",
    size_column="Words",
    size_unit="quoted words",
    general_notes=(
        "Counts are not divided by speech length: the summary carries the quoted words, not the speech's, so a "
        "long speech can show more quotes for that reason alone. '% attributed' is the length-free measure.",
        "A speech with no quotes has no row, so it is absent from these figures rather than drawn at zero.",
    ),
)

EXTRAS_PANELS: tuple[PanelDefinition, ...] = (
    SVO_TOP_TRIPLES,
    QUOTE_SPEAKERS,
    *measure_panels(_QUOTES, title_word="Quotation", prefix="quote_annotator_rate"),
)
