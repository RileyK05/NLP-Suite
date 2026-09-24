"""Ranked-term panels for the "terms" tool family.

Every raw-frequency ranking on this corpus tops out with "of", "the", "be" --
technically correct and informative about nothing. This module is one shared
factory (:func:`ranked_terms`) applied to every tool whose output is, at
bottom, a table of terms with a count: n-grams, the CoNLL wordlist,
nominalizations, and the resource-lookup tools that map a word to a category
(NRC via its own heatmap, VerbNet, FrameNet, the symbolic typologies,
WordNet). Two house rules make the difference between "correct" and
"informative" here:

* **Document coverage beside frequency.** A word that is frequent because
  one speech repeats it forty times is not a corpus theme; the hover says
  "in N of M documents" wherever the source table carries a document id, so
  a reader can tell the two apart without a second figure.
* **A function-word toggle**, on by default for word rankings. A term is
  hidden when it begins or ends with a function word, or is punctuation:
  "the world" and "of congress" are fragments, "united states" and "state
  of the union" are phrases. (Hiding only all-function-word n-grams left the
  real top bigrams as "the world, the congress, the united".)

``core/analysis/verbnet.py``, ``framenet.py``, ``symbolic.py`` and
``wordnet.py`` (the ``up`` command) all take a flat word-list CSV with no
``Document`` column at all -- they are literally "classify these words",
run outside the per-document corpus pipeline. Their panels here inherit that
limit honestly: no document coverage is offered, and the notes say why. None
of the four are in the real-data audit pickle (the machine that produced it
has no WordNet/VerbNet/FrameNet corpora installed), so their real-data
checks are unverified and this file says so wherever they matter -- the
schemas are read from ``core/analysis/*.py`` and ``tools/*.py`` directly, not
guessed.

``tfidf`` and ``nrc`` get their own designs (a per-document top-terms
ranking plus a document x term heatmap; a document x emotion heatmap) because
their real tables do not reduce to one ranked list: tf-idf is inherently
per-document, and NRC already publishes one row per document with eight
normalised emotion shares as columns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import re
from typing import Any

import pandas as pd

from core.result import Diagnostic, Result
from core.viz.panel_helpers import document_labels, is_function_word
from core.viz.panelspec import (
    Evidence,
    PanelDefinition,
    PanelMark,
    PanelParam,
    PreparedPanel,
    Provenance,
    counted_on_lemmas,
)

__all__ = [
    "FRAMENET_RANKED_FRAMES",
    "NGRAMS_RANKED_TERMS",
    "NOMINALIZATION_RANKED_TERMS",
    "NRC_EMOTION_HEATMAP",
    "SEMANTIC_RANKED_LEMMAS",
    "SYMBOLIC_RANKED_ACTORS",
    "SYMBOLIC_RANKED_SPACES",
    "TERMS_PANELS",
    "TFIDF_HEATMAP",
    "TFIDF_TOP_TERMS",
    "VERBNET_RANKED_CLASSES",
    "WORDLIST_RANKED_TERMS",
    "WORDNET_RANKED_CATEGORIES",
    "ranked_terms",
]

_TOP_N_DEFAULT = 25
_TOP_N_MAX = 300


def _modes(frame: pd.DataFrame, key: str, column: str) -> pd.Series:
    """Each *key*'s most frequent non-blank *column* value (ties: alphabetical).

    Vectorised: a per-group Python mode over the semantic tool's 11,018
    lemmas took seven seconds on the real corpus.
    """
    values = frame[[key, column]].astype(str)
    values = values[values[column].str.strip() != ""]
    counts = values.groupby([key, column], sort=False).size().reset_index(name="n")
    counts = counts.sort_values(["n", column], ascending=[False, True], kind="stable").drop_duplicates(key)
    return counts.set_index(key)[column].reindex(frame[key].unique()).fillna("")


def _is_function_term(term: str) -> bool:
    """True for a term that begins or ends with a function word, or has no letters.

    The phrase-boundary rule. Hiding only terms made *entirely* of function
    words was tried first and failed on the real corpus: the top bigrams
    were "the world", "the congress", "the united" -- a content word glued to
    an article, which says nothing a unigram list does not. A real phrase
    starts and ends on content ("united states", "state of the union"),
    so this keeps those and drops the fragments. Punctuation and numbers are
    not terms at all: the word-list tool counts "." and "," as tokens.
    """
    words = str(term).split()
    if not words or not any(character.isalpha() for character in str(term)):
        return True
    return is_function_word(words[0]) or is_function_word(words[-1])


def ranked_terms(  # noqa: PLR0913 - one factory covers every ranked-term table; each keyword names one real choice
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    provenance: Provenance,
    *,
    definition: PanelDefinition,
    term_column: str,
    count_column: str | None,
    doc_id_column: str | None,
    unit_name: str,
    x_label: str,
    exclude_values: tuple[str, ...] = (),
    hide_function_words_supported: bool = False,
    ngram_n_supported: bool = False,
    phrase_of: Callable[[str], str] | None = None,
    lemma: bool = False,
    extra_columns: tuple[str, ...] = (),
) -> Result[PreparedPanel]:
    """Rank terms by corpus frequency, with document coverage in the hover.

    *count_column* is ``None`` for a table with one row per occurrence
    (frequency = rows per term) and a column name for a table that already
    carries the term's total (ngrams' ``Frequency in Corpus`` is repeated on
    every occurrence row, so it is taken with ``max`` per term, not summed --
    summing would multiply it by the number of documents the term appears
    in). *doc_id_column* is ``None`` for a table with no document
    association at all (the resource-lookup tools), which turns off the
    coverage column and the coverage/frequency-per-document sort orders
    rather than faking a number.
    """
    diagnostics: list[Diagnostic] = []
    working = frame.copy()
    working[term_column] = working[term_column].astype(str).str.strip()
    working = working[working[term_column] != ""]
    if exclude_values:
        working = working[~working[term_column].isin(exclude_values)]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"no {term_column} values to rank after cleaning"))

    grouped = working.groupby(term_column, sort=False)
    if count_column is not None:
        working[count_column] = pd.to_numeric(working[count_column], errors="coerce").fillna(0.0)
        frequency = working.groupby(term_column, sort=False)[count_column].max()
    else:
        frequency = grouped.size().astype(float)

    agg = pd.DataFrame({"Frequency": frequency})

    total_docs = 0
    if doc_id_column is not None and doc_id_column in working.columns:
        total_docs = int(working[doc_id_column].nunique())
        coverage = grouped[doc_id_column].nunique()
        agg["Documents"] = coverage
        agg["Frequency per document"] = agg["Frequency"] / coverage.replace(0, pd.NA)

    for column in extra_columns:
        if column in working.columns:
            agg[column] = _modes(working, term_column, column)

    agg = agg.reset_index()

    if ngram_n_supported:
        n_param = int(params.get("n", 0) or 0)
        if n_param > 0:
            before = len(agg)
            agg = agg[agg[term_column].str.split().str.len() == n_param]
            removed = before - len(agg)
            if removed:
                diagnostics.append(
                    Diagnostic.info(
                        "PANEL_NGRAM_LENGTH_FILTERED",
                        f"{removed} term(s) were not {n_param}-word n-grams and were left out.",
                        removed=removed,
                        n=n_param,
                    )
                )

    if hide_function_words_supported and bool(params.get("hide-function-words", True)):
        before = len(agg)
        agg = agg[~agg[term_column].map(_is_function_term)]
        removed = before - len(agg)
        if removed:
            diagnostics.append(
                Diagnostic.info(
                    "PANEL_FUNCTION_WORDS_HIDDEN",
                    f"{removed} term(s) that begin or end with a function word, or are punctuation, were hidden. "
                    "Turn off hide-function-words to see them.",
                    removed=removed,
                )
            )

    if agg.empty:
        return Result.failure(
            Diagnostic.error("PANEL_NO_DATA", "no terms left to rank after filtering; loosen a filter"),
            *diagnostics,
        )

    order = str(params.get("order", "frequency"))
    sort_column = {
        "frequency": "Frequency",
        "coverage": "Documents",
        "frequency-per-document": "Frequency per document",
    }.get(order, "Frequency")
    if sort_column not in agg.columns:
        sort_column = "Frequency"
    agg = agg.sort_values([sort_column, term_column], ascending=[False, True], kind="stable")

    top_n = int(params.get("top-n", _TOP_N_DEFAULT))
    drawn = agg.head(top_n).reset_index(drop=True)

    marks: list[PanelMark] = []
    for rank, row in drawn.iterrows():
        term = str(row[term_column])
        freq = float(row["Frequency"])
        parts = [f"{unit_name} {freq:g}"]
        if "Documents" in drawn.columns:
            docs = int(row["Documents"])
            per_doc = float(row["Frequency per document"]) if pd.notna(row["Frequency per document"]) else 0.0
            parts.append(f"in {docs} of {total_docs} documents ({per_doc:.1f} per covered document)")
        for column in extra_columns:
            if column in drawn.columns:
                value = str(row[column]).strip()
                if value:
                    parts.append(f"{column} {value}")
        describe = f"{term}: " + "; ".join(parts)
        phrase = phrase_of(term) if phrase_of is not None else ""
        marks.append(
            PanelMark(
                key=term,
                label=term,
                x=freq,
                y=float(rank),
                evidence=Evidence(
                    scope="terms" if phrase else "rows",
                    filters=((term_column, term),),
                    count=max(round(freq), 1),
                    describe=describe,
                    phrase=phrase,
                    lemma=bool(lemma and phrase),
                ),
            )
        )

    coverage_note = (
        "Coverage is 'in N of M documents': M is the number of documents present in this table, and a term "
        "heavy in one document but absent from the rest cannot pass for a corpus theme."
        if "Documents" in drawn.columns
        else "This table carries no document association, so no document-coverage figure is offered here -- "
        "every bar is a corpus-wide total with no way to say how many documents it came from."
    )
    notes: tuple[str, ...] = (
        f"Bar length is {unit_name}, always -- the 'order' parameter changes which terms are shown, not what "
        "the bar encodes.",
        coverage_note,
    )
    if hide_function_words_supported:
        notes = (
            *notes,
            "Function words are hidden by default: a term that begins or ends with one ('the world', "
            "'of congress') is a fragment, not a phrase, and is left out; 'united states' and 'state of the "
            "union' are kept. Punctuation counted as a token is hidden too.",
        )

    prepared = PreparedPanel(
        panel=definition.name,
        shape="ranked_bars",
        title=definition.title,
        subtitle=f"{len(drawn)} of {len(agg)} term(s) shown, ordered by {order}",
        marks=tuple(marks),
        x_label=x_label,
        y_label=f"Term, ranked by {order}",
        provenance=provenance,
        data=drawn,
        notes=notes,
    )
    return Result.success(prepared, *diagnostics)


# ------------------------------------------------------------- ngrams --

_NGRAM = "N-gram"
_FREQ_DOC = "Frequency in Document"
_FREQ_CORPUS = "Frequency in Corpus"
_DOC_ID = "Document ID"

_HIDE_FUNCTION_WORDS = PanelParam(
    name="hide-function-words",
    type="bool",
    default=True,
    label="Hide function-word terms",
    help=(
        "Hide terms that begin or end with a function word ('the world', 'of congress') and punctuation. "
        "Phrases that start and end on content ('united states', 'state of the union') are kept."
    ),
)
_TOP_N = PanelParam(
    name="top-n",
    type="int",
    default=_TOP_N_DEFAULT,
    minimum=1,
    maximum=_TOP_N_MAX,
    label="Terms to show",
    help="How many ranked terms to draw.",
)


def _order_param(coverage_supported: bool) -> PanelParam:
    choices = ("frequency", "coverage", "frequency-per-document") if coverage_supported else ("frequency",)
    return PanelParam(
        name="order",
        type="choice",
        default="frequency",
        choices=choices,
        label="Rank by",
        help=(
            "'frequency' ranks by raw corpus count (bar length always shows this). 'coverage' ranks by how "
            "many documents contain the term. 'frequency-per-document' ranks by average occurrences per "
            "document it appears in -- a term used sparingly across many documents ranks differently from one "
            "used heavily in a few."
            if coverage_supported
            else "This table has no document association, so 'frequency' is the only rank available."
        ),
    )


def _build_ngrams(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    on_lemmas = counted_on_lemmas(provenance.settings)
    return ranked_terms(
        frame,
        params,
        provenance,
        definition=NGRAMS_RANKED_TERMS,
        term_column=_NGRAM,
        count_column=_FREQ_CORPUS,
        doc_id_column=_DOC_ID,
        unit_name="occurrences",
        x_label="Occurrences across the corpus",
        hide_function_words_supported=True,
        ngram_n_supported=True,
        phrase_of=lambda term: term,
        lemma=on_lemmas,
    )


NGRAMS_RANKED_TERMS = PanelDefinition(
    name="ngrams_ranked_terms",
    title="Ranked n-grams",
    question="Which word sequences does the corpus actually use most, once function words are set aside?",
    tool="ngrams",
    shape="ranked_bars",
    summary="N-grams ranked by corpus frequency, with document coverage and a function-word toggle.",
    requires=(_NGRAM, _FREQ_CORPUS, _DOC_ID),
    params=(
        _order_param(coverage_supported=True),
        _TOP_N,
        _HIDE_FUNCTION_WORDS,
        PanelParam(
            name="n",
            type="int",
            default=0,
            minimum=0,
            maximum=10,
            label="N-gram length",
            help=(
                "Keep only n-grams with exactly this many words (counted by spaces). 0 keeps every length "
                "present -- most ngrams runs produce a single fixed n, so this usually has no effect unless "
                "several runs were concatenated."
            ),
        ),
    ),
    build=_build_ngrams,
    notes=(
        "Frequency in Corpus is repeated on every document row for a given n-gram; this panel takes it once "
        "per n-gram (not summed across the rows) so the bar is the true corpus total, not a multiple of it.",
        "On the real 87-speech State of the Union corpus, raw frequency without the function-word toggle is "
        "topped by 'of the', 'in the', 'to the'; with the default toggle on, content bigrams such as "
        "'united states', 'the congress' and 'the world' surface instead.",
    ),
)


# ------------------------------------------------------- conll_wordlist --

_WORD = "Word"
_COUNT = "Count"
#: Fewer content words than this after hiding, and the run's own top-n is the limit.
_FEW_WORDS = 5


def _build_wordlist(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    on_lemmas = counted_on_lemmas(provenance.settings)

    def rank(chosen: Mapping[str, Any]) -> Result[PreparedPanel]:
        return ranked_terms(
            frame,
            chosen,
            provenance,
            definition=WORDLIST_RANKED_TERMS,
            term_column=_WORD,
            count_column=_COUNT,
            doc_id_column=None,
            unit_name="occurrences",
            x_label="Occurrences across the corpus",
            hide_function_words_supported=True,
            phrase_of=lambda term: term,
            lemma=on_lemmas,
        )

    result = rank(params)
    if result.value is None and bool(params.get("hide-function-words", True)):
        # Every word in the run's own top-n was a function word (the real
        # corpus's top 20): a refusal draws nothing, while the unfiltered
        # list at least shows what the run kept, and the warning below says
        # how to get content words.
        unfiltered = rank({**params, "hide-function-words": False})
        if unfiltered.value is not None:
            result = Result.success(
                unfiltered.value,
                *unfiltered.diagnostics,
                Diagnostic.warning(
                    "PANEL_FUNCTION_WORDS_SHOWN",
                    "Every word in this list is a function word, so hiding them would leave nothing; they are shown.",
                ),
            )
    # The tool keeps only its own top N, and on real prose that is almost all
    # function words and punctuation: hiding them leaves a handful of bars or
    # none. The remedy is in the run, so say which setting to change.
    content = [mark for mark in result.value.marks if not _is_function_term(mark.label)] if result.value else []
    kept = len(content)
    if kept < _FEW_WORDS and len(frame) > kept:
        advice = Diagnostic.warning(
            "PANEL_WORDLIST_TOO_SHORT",
            f"Only {kept} of the run's {len(frame)} words are content words. The word list keeps just its own "
            "top-n, which on real prose is mostly function words; re-run conll_wordlist with a larger top-n or "
            "a part-of-speech category (for example nouns) to rank content words.",
            kept=kept,
        )
        if result.value is None:
            return Result.failure(*result.diagnostics, advice)
        return Result.success(result.value, *result.diagnostics, advice)
    return result


WORDLIST_RANKED_TERMS = PanelDefinition(
    name="conll_wordlist_ranked_terms",
    title="Ranked word list",
    question="Which words dominate this corpus, once function words are set aside?",
    tool="conll_wordlist",
    shape="ranked_bars",
    summary="The tool's own top-N word list, ranked by count, with a function-word toggle.",
    requires=(_WORD, _COUNT),
    params=(_order_param(coverage_supported=False), _TOP_N, _HIDE_FUNCTION_WORDS),
    build=_build_wordlist,
    notes=(
        "conll_wordlist already returns a top-N table (the tool's own --top-n), not the whole vocabulary, and "
        "it carries no per-document breakdown -- there is no honest document-coverage figure to add here.",
        "On the real 87-speech State of the Union corpus, the tool's own top 20 is almost entirely function "
        "words ('the', 'of', 'and', 'to'); hiding them is the only way this panel says anything about content.",
    ),
)


# ------------------------------------------------------- nominalization --

_BASE_VERB = "Base Verb"


def _build_nominalization(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    on_lemmas = counted_on_lemmas(provenance.settings)
    return ranked_terms(
        frame,
        params,
        provenance,
        definition=NOMINALIZATION_RANKED_TERMS,
        term_column=_WORD,
        count_column=None,
        doc_id_column=_DOC_ID,
        unit_name="occurrences",
        x_label="Occurrences across the corpus",
        hide_function_words_supported=False,
        phrase_of=lambda term: term,
        lemma=on_lemmas,
        extra_columns=(_BASE_VERB,),
    )


NOMINALIZATION_RANKED_TERMS = PanelDefinition(
    name="nominalization_ranked_terms",
    title="Ranked nominalizations",
    question="Which nominalizations (verb-derived nouns like 'decision' from 'decide') does the corpus use most?",
    tool="nominalization",
    shape="ranked_bars",
    summary="Nominalizations ranked by occurrence count, with document coverage and the base verb in the hover.",
    requires=(_WORD, _BASE_VERB, _DOC_ID),
    params=(_order_param(coverage_supported=True), _TOP_N),
    build=_build_nominalization,
    notes=(
        "Every row of the source table is one occurrence of one nominalization in one sentence; frequency here "
        "is the row count per word, not a rate, so a longer document contributes more rows on its own.",
        "The hover names the word's most common Base Verb; a small share of nominalizations can match more "
        "than one base verb across the corpus, and only the most frequent one is shown.",
        "A function-word toggle is not offered: nominalizations are, by the tool's own detection, content "
        "words (verb-derived nouns), so the raw-frequency trap this toggle guards against does not arise here.",
    ),
)


# -------------------------------------------------------------- semantic --

_LEMMA = "Lemma"
_SEM_COUNT = "Count"
_SEM_EXTRA = ("WordNet", "VerbNet", "FrameNet")


def _build_semantic(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    on_lemmas = counted_on_lemmas(provenance.settings)
    return ranked_terms(
        frame,
        params,
        provenance,
        definition=SEMANTIC_RANKED_LEMMAS,
        term_column=_LEMMA,
        count_column=_SEM_COUNT,
        doc_id_column=None,
        unit_name="occurrences",
        x_label="Occurrences across the corpus",
        hide_function_words_supported=True,
        phrase_of=lambda term: term,
        lemma=on_lemmas,
        extra_columns=_SEM_EXTRA,
    )


SEMANTIC_RANKED_LEMMAS = PanelDefinition(
    name="semantic_ranked_lemmas",
    title="Ranked semantic-tag lemmas",
    question="Which lemmas does the corpus use most among the ones semantic tagging could resolve?",
    tool="semantic",
    shape="ranked_bars",
    summary="Lemmas ranked by count, with their WordNet/VerbNet/FrameNet tags in the hover.",
    requires=(_LEMMA, _SEM_COUNT, "WordNet", "VerbNet", "FrameNet"),
    params=(_order_param(coverage_supported=False), _TOP_N, _HIDE_FUNCTION_WORDS),
    build=_build_semantic,
    notes=(
        "This table is a corpus-wide lemma count with no per-document breakdown, so no document-coverage "
        "figure is offered.",
        "A lemma with an empty WordNet/VerbNet/FrameNet cell means the resource had nothing for that lemma "
        "(VerbNet only ever tags verbs), not that the panel failed to look it up.",
        "Unverified on the real 87-speech corpus: the machine that produced the real-data audit has no "
        "WordNet/VerbNet/FrameNet corpora installed, so 'semantic' fell back to its own labelled demonstration "
        "vocabulary there rather than a real run. This panel is built and tested against the tool's real "
        "output schema (core/analysis/semantic.py), not against a real run.",
    ),
)


# ---------------------------------------------- category-lookup factory --


def _category_builder(  # noqa: PLR0913 - declaration maps each real word->category tool to a panel
    *,
    name: str,
    tool: str,
    title: str,
    question: str,
    term_column: str,
    category_column: str,
    exclude_value: str,
    summary: str,
    notes: tuple[str, ...],
) -> PanelDefinition:
    """One ranked-bars panel for a word -> category lookup table with no document axis.

    Shared by VerbNet, FrameNet, the two symbolic typologies and WordNet UP:
    every one of these tools reads a flat word-list CSV (``tools/<tool>.py``
    -- no ``Document`` column exists to join against) and assigns each word
    one category, with unresolved words recorded as ``exclude_value`` rather
    than dropped. The panel ranks the *categories* by how many words landed
    in each -- a corpus-level answer to "what does this word list mostly
    consist of?" -- with the unresolved bucket excluded, matching the tools'
    own ``category_counts`` helpers.
    """
    holder: dict[str, PanelDefinition] = {}

    def build(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
        return ranked_terms(
            frame,
            params,
            provenance,
            definition=holder["definition"],
            term_column=category_column,
            count_column=None,
            doc_id_column=None,
            unit_name="words classified",
            x_label="Words classified in this category",
            exclude_values=(exclude_value,),
        )

    definition = PanelDefinition(
        name=name,
        title=title,
        question=question,
        tool=tool,
        shape="ranked_bars",
        summary=summary,
        requires=(term_column, category_column),
        params=(
            PanelParam(
                name="top-n", type="int", default=_TOP_N_DEFAULT, minimum=1, maximum=_TOP_N_MAX, help=_TOP_N.help
            ),
        ),
        build=build,
        notes=notes,
    )
    holder["definition"] = definition
    return definition


_LOOKUP_NOTES = (
    "This tool classifies a flat word list, not the per-document corpus table: its output has no Document "
    "column at all, so no document-coverage figure is offered and 'in N of M documents' cannot be shown.",
    "No function-word toggle: the terms ranked here are category names (classes/frames/types), not the "
    "underlying words, so the raw-frequency trap the toggle guards against does not apply.",
    "Words the resource could not classify are excluded from this ranking, matching the tool's own frequency "
    "table -- an empty resource (no WordNet/VerbNet/FrameNet corpus installed) would rank nothing at all "
    "rather than silently ranking 'not found' as a category.",
    "Unverified on the real 87-speech corpus: the machine that produced the real-data audit has no "
    "WordNet/VerbNet/FrameNet corpora installed, so this tool never ran there. This panel is built and tested "
    "against the tool's real output schema (core/analysis/*.py, tools/*.py), not against a real run.",
)

VERBNET_RANKED_CLASSES = _category_builder(
    name="verbnet_ranked_classes",
    tool="verbnet",
    title="Ranked VerbNet classes",
    question="Which VerbNet verb classes does this word list mostly consist of?",
    term_column=_WORD,
    category_column="VerbNet Class",
    exclude_value="Not found",
    summary="VerbNet classes ranked by how many verbs in the list resolved to each.",
    notes=_LOOKUP_NOTES,
)

FRAMENET_RANKED_FRAMES = _category_builder(
    name="framenet_ranked_frames",
    tool="framenet",
    title="Ranked FrameNet frames",
    question="Which FrameNet frames does this word list mostly evoke?",
    term_column=_WORD,
    category_column="FrameNet Frame",
    exclude_value="Not found",
    summary="FrameNet frames ranked by how many words in the list resolved to each.",
    notes=_LOOKUP_NOTES,
)

SYMBOLIC_RANKED_SPACES = _category_builder(
    name="symbolic_ranked_spaces",
    tool="symbolic",
    title="Ranked symbolic space types",
    question="Which symbolic space types does this word list mostly consist of?",
    term_column=_WORD,
    category_column="Space Type",
    exclude_value="unclassified",
    summary="Symbolic space types ranked by how many location words resolved to each.",
    notes=_LOOKUP_NOTES,
)

SYMBOLIC_RANKED_ACTORS = _category_builder(
    name="symbolic_ranked_actors",
    tool="symbolic",
    title="Ranked social actor types",
    question="Which social actor types does this word list mostly consist of?",
    term_column="Actor",
    category_column="Actor Type",
    exclude_value="unclassified",
    summary="Social actor types ranked by how many person words resolved to each.",
    notes=_LOOKUP_NOTES,
)

WORDNET_RANKED_CATEGORIES = _category_builder(
    name="wordnet_ranked_categories",
    tool="wordnet",
    title="Ranked WordNet supersenses",
    question="Which WordNet top-level categories (or chosen anchors) does this word list mostly consist of?",
    term_column=_WORD,
    category_column="WordNet Category",
    exclude_value="Not found",
    summary="WordNet supersenses (or anchor categories) ranked by how many words resolved to each.",
    notes=_LOOKUP_NOTES,
)


# ------------------------------------------------------------------- nrc --

_DOC = "Document"
_NRC_EIGHT: tuple[str, ...] = ("fear", "anger", "anticipation", "trust", "surprise", "sadness", "disgust", "joy")
_LEADING_DATE = re.compile(r"^(\d{4})(?:-(\d{2})-(\d{2}))?")


def _document_position(name: str) -> float:
    """A sortable position for a document row: the year (plus day-of-year fraction) if the filename is dated.

    A filename that does not start ``YYYY`` or ``YYYY-MM-DD`` sorts after
    every dated one (``float('inf')``), so undated rows fall to the bottom of
    the axis instead of scattering through it.
    """
    match = _LEADING_DATE.match(str(name).strip())
    if not match:
        return float("inf")
    year = int(match.group(1))
    if match.group(2) and match.group(3):
        try:
            import datetime as _dt

            day = _dt.date(year, int(match.group(2)), int(match.group(3)))
            start = _dt.date(year, 1, 1)
            span = (_dt.date(year + 1, 1, 1) - start).days
            return year + (day - start).days / span
        except ValueError:
            return float(year)
    return float(year)


def _nrc_emotion_heatmap(
    frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance
) -> Result[PreparedPanel]:
    """Document x emotion heatmap: NRC already publishes one row per document."""
    _ = params
    working = frame.copy()
    working[_DOC] = working[_DOC].astype(str)
    missing = [emotion for emotion in _NRC_EIGHT if emotion not in working.columns]
    if missing:
        return Result.failure(
            Diagnostic.error("PANEL_MISSING_COLUMN", f"nrc table is missing emotion column(s): {missing}")
        )
    for emotion in _NRC_EIGHT:
        working[emotion] = pd.to_numeric(working[emotion], errors="coerce").fillna(0.0)

    working = working.sort_values(_DOC, key=lambda col: col.map(_document_position), kind="stable").reset_index(
        drop=True
    )
    labels = document_labels(working[_DOC].tolist())
    y_categories = tuple(labels[name] for name in working[_DOC])

    marks: list[PanelMark] = []
    for row_index, row in working.iterrows():
        doc = str(row[_DOC])
        doc_id = str(row.get(_DOC_ID, doc))
        tokens = int(row.get("Tokens", 0))
        for col_index, emotion in enumerate(_NRC_EIGHT):
            share = float(row[emotion])
            marks.append(
                PanelMark(
                    key=f"{doc_id}:{emotion}",
                    label=f"{y_categories[row_index]} / {emotion}",
                    x=float(col_index),
                    y=float(row_index),
                    value=share,
                    evidence=Evidence(
                        scope="rows",
                        filters=((_DOC_ID, doc_id),),
                        count=tokens,
                        describe=(
                            f"{y_categories[row_index]}: {emotion} is {share:.0%} of this document's "
                            f"eight-emotion hits ({tokens} tokens scored)"
                        ),
                    ),
                )
            )

    prepared = PreparedPanel(
        panel=NRC_EMOTION_HEATMAP.name,
        shape="heatmap",
        title="NRC emotion mix by document",
        subtitle=f"{len(working)} document(s) x {len(_NRC_EIGHT)} emotions -- value = share of each document's hits",
        marks=tuple(marks),
        x_label="Emotion",
        y_label="Document (dated order where the file name carries a date)",
        provenance=provenance,
        data=working,
        x_categories=_NRC_EIGHT,
        y_categories=y_categories,
        color_scale="sequential",
        value_label="Share of the document's emotion hits",
        height=max(500, 24 * len(working) + 120),
        notes=(
            "Value is each emotion's share of the document's own eight-emotion hits (NRC's own normalisation, "
            "already summing to 1 per document across fear/anger/anticipation/trust/surprise/sadness/disgust/joy). "
            "It is not a per-1,000-token rate: a document with very few lexicon hits can still show a large "
            "share for one emotion, so read a lone bright cell in a low-Tokens row cautiously.",
            "Positive/negative valence is scored separately by NRC and is not one of the eight columns here.",
            "Rows are ordered by the date parsed from the file name (YYYY or YYYY-MM-DD); a document whose "
            "name carries no such date sorts to the bottom, not into the timeline.",
        ),
    )
    return Result.success(prepared)


NRC_EMOTION_HEATMAP = PanelDefinition(
    name="nrc_emotion_heatmap",
    title="NRC emotion mix by document",
    question="Which emotions does the NRC lexicon find each document leaning on, and how does that change across documents?",
    tool="nrc",
    shape="heatmap",
    summary="Document x emotion heatmap; value is each emotion's share of that document's lexicon hits.",
    requires=(_DOC_ID, _DOC, "Tokens", "Hits", *_NRC_EIGHT),
    params=(),
    build=_nrc_emotion_heatmap,
    notes=(
        "Unverified on the real 87-speech corpus: the machine that produced the real-data audit has no NRC "
        "lexicon installed (nrclex is absent and no nrc-lexicon asset is configured), so 'nrc' never ran "
        "there. This panel is built and tested against core/analysis/nrc.py's real, documented output schema.",
    ),
)


# ------------------------------------------------------------------ tfidf --

_TERM = "Term"
_TFIDF = "TF-IDF"
_RANK = "Rank"
_DATE = "Date"
_YEAR = "Year"


def _corpus_wide_terms(frame: pd.DataFrame) -> list[Diagnostic]:
    """A warning when most of the table's top terms occur in every document.

    Smoothed IDF gives a word found everywhere an IDF of 1, not 0, so with
    raw term frequency "the" outranks every distinctive word: on the real
    87-speech corpus every speech's top 20 was "the, of, and, to, ...". The
    fix is in the run, not the figure -- the table holds only each
    document's top N, so the distinctive terms were never kept. With
    max-df-ratio 0.5 the same corpus gives 1934 "industrial, restoration,
    recovery" and 2024 "gaza, roe, predecessor".
    """
    if "Document Frequency" not in frame.columns or _DOC not in frame.columns:
        return []
    documents = frame[_DOC].nunique()
    df = pd.to_numeric(frame["Document Frequency"], errors="coerce")
    share = float((df >= documents).mean()) if documents > 1 and len(df) else 0.0
    if share <= 0.5:
        return []
    return [
        Diagnostic.warning(
            "PANEL_CORPUS_WIDE_TERMS",
            f"{share:.0%} of this table's top terms appear in every document, so TF-IDF here is ranking raw "
            "frequency ('the', 'of'). Re-run tfidf with max-df-ratio below 1 (0.5 drops words found in more "
            "than half the documents) to see what distinguishes each document.",
            share=round(share, 3),
        )
    ]


def _tfidf_top_terms(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    working = frame.copy()
    working[_TFIDF] = pd.to_numeric(working[_TFIDF], errors="coerce")
    working[_COUNT] = pd.to_numeric(working.get(_COUNT, 0), errors="coerce").fillna(0)
    working["IDF"] = pd.to_numeric(working.get("IDF", 0), errors="coerce").fillna(0)
    diagnostics: list[Diagnostic] = _corpus_wide_terms(working)

    requested = str(params.get("document", "") or "").strip()
    by_doc = working[[_DOC, _DATE]].drop_duplicates(subset=[_DOC]) if _DATE in working.columns else None
    if requested:
        matches = working[_DOC].astype(str) == requested
        if not matches.any():
            return Result.failure(
                Diagnostic.error(
                    "PANEL_DOCUMENT_NOT_FOUND",
                    f"no document named {requested!r} in this table's Document column",
                )
            )
        document = requested
    elif by_doc is not None and not by_doc.empty:
        document = str(by_doc.sort_values(_DATE, kind="stable").iloc[0][_DOC])
    else:
        document = str(working[_DOC].iloc[0])

    subset = working[working[_DOC].astype(str) == document].copy()
    subset = subset[subset[_TFIDF].notna()]
    if subset.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", f"no usable TF-IDF rows for document {document!r}"))

    if _RANK in subset.columns:
        subset["_rank"] = pd.to_numeric(subset[_RANK], errors="coerce")
        subset = subset.sort_values(["_rank", _TERM], ascending=[True, True], kind="stable")
    else:
        subset = subset.sort_values([_TFIDF, _TERM], ascending=[False, True], kind="stable")

    top_n = int(params.get("top-n", 20))
    drawn = subset.head(top_n).reset_index(drop=True)

    marks: list[PanelMark] = []
    for rank, row in drawn.iterrows():
        term = str(row[_TERM])
        score = float(row[_TFIDF])
        count = int(row[_COUNT])
        idf = float(row["IDF"])
        marks.append(
            PanelMark(
                key=term,
                label=term,
                x=score,
                # ranked_bars reads y as the rank, 0 at the top (panelspec).
                y=float(rank),
                evidence=Evidence(
                    scope="terms",
                    filters=((_DOC, document), (_TERM, term)),
                    count=count,
                    describe=f"{term} in {document}: TF-IDF {score:.4f}, appears {count}x here (IDF {idf:.2f})",
                    phrase=term,
                    lemma=counted_on_lemmas(provenance.settings),
                ),
            )
        )

    prepared = PreparedPanel(
        panel=TFIDF_TOP_TERMS.name,
        shape="ranked_bars",
        title="Top TF-IDF terms for one document",
        subtitle=f"{document} -- {len(drawn)} term(s), ranked by TF-IDF",
        marks=tuple(marks),
        x_label="TF-IDF",
        y_label="Term",
        provenance=provenance,
        data=drawn,
        notes=(
            "TF-IDF is high for a term that is frequent in this document but rare across the corpus -- it "
            "surfaces what makes this document distinctive, not what it talks about most (a word used in "
            "every document scores low here even if it is common in this one).",
            "This table is already the tool's own top-N per document (its --top-n), so a high rank near the "
            "bottom of this list may not be the document's true Nth-highest TF-IDF term if the run capped "
            "lower than what is shown here.",
        ),
    )
    return Result.success(prepared, *diagnostics)


TFIDF_TOP_TERMS = PanelDefinition(
    name="tfidf_top_terms",
    title="Top TF-IDF terms for one document",
    question="What makes this document distinctive against the rest of the corpus, by TF-IDF?",
    tool="tfidf",
    shape="ranked_bars",
    summary="One document's terms ranked by TF-IDF, with raw count and IDF in the hover.",
    requires=(_DOC_ID, _DOC, _TERM, _COUNT, "IDF", _TFIDF),
    params=(
        PanelParam(
            name="document",
            type="str",
            default="",
            label="Document",
            help="Which document to show, matched against the Document column. Empty picks the earliest by date.",
        ),
        PanelParam(
            name="top-n",
            type="int",
            default=20,
            minimum=1,
            maximum=100,
            label="Terms to show",
            help="How many of the document's top TF-IDF terms to draw.",
        ),
    ),
    build=_tfidf_top_terms,
    notes=(
        "TF-IDF rewards a term that is common here and rare elsewhere; a genuinely important but corpus-wide "
        "word (e.g. 'america' in every State of the Union address) can score low.",
    ),
)


def _tfidf_heatmap(frame: pd.DataFrame, params: Mapping[str, Any], provenance: Provenance) -> Result[PreparedPanel]:
    working = frame.copy()
    working[_TFIDF] = pd.to_numeric(working[_TFIDF], errors="coerce")
    working = working[working[_TFIDF].notna()]
    if working.empty:
        return Result.failure(Diagnostic.error("PANEL_NO_DATA", "no usable TF-IDF rows"))

    top_k = int(params.get("top-k", 25))
    totals = working.groupby(_TERM, sort=False)[_TFIDF].sum().sort_values(ascending=False)
    terms = tuple(str(term) for term in totals.head(top_k).index)
    diagnostics: list[Diagnostic] = _corpus_wide_terms(working)
    if len(totals) > top_k:
        diagnostics.append(
            Diagnostic.info(
                "PANEL_TERMS_CAPPED",
                f"{len(totals)} distinct terms appear across the corpus's top-TF-IDF lists; showing the {top_k} "
                "with the highest summed TF-IDF. Raise top-k to see more.",
                available=len(totals),
                shown=top_k,
            )
        )

    by_doc = working[[_DOC, _DATE]].drop_duplicates(subset=[_DOC]) if _DATE in working.columns else None
    if by_doc is not None and not by_doc.empty:
        ordered_docs = by_doc.sort_values(_DATE, kind="stable")[_DOC].astype(str).tolist()
    else:
        ordered_docs = sorted(working[_DOC].astype(str).unique())
    labels = document_labels(ordered_docs)
    y_categories = tuple(labels[name] for name in ordered_docs)
    doc_index = {name: index for index, name in enumerate(ordered_docs)}
    term_index = {term: index for index, term in enumerate(terms)}
    # Key and filter a cell by the stable Document ID where the table carries
    # one, matching the other heatmaps; the display label keeps the file name.
    has_doc_id = _DOC_ID in working.columns
    doc_ids = (
        working[[_DOC, _DOC_ID]].drop_duplicates(subset=[_DOC]).set_index(_DOC)[_DOC_ID].astype(str).to_dict()
        if has_doc_id
        else {}
    )
    filter_column = _DOC_ID if has_doc_id else _DOC

    cell = working[working[_TERM].astype(str).isin(terms)].copy()
    cell = cell[cell[_DOC].astype(str).isin(doc_index)]

    marks: list[PanelMark] = []
    seen: set[tuple[int, int]] = set()
    for _, row in cell.iterrows():
        term = str(row[_TERM])
        doc = str(row[_DOC])
        doc_key = doc_ids.get(doc, doc)
        key = (doc_index[doc], term_index[term])
        if key in seen:
            continue
        seen.add(key)
        score = float(row[_TFIDF])
        marks.append(
            PanelMark(
                key=f"{doc_key}:{term}",
                label=f"{labels[doc]} / {term}",
                x=float(key[1]),
                y=float(key[0]),
                value=score,
                evidence=Evidence(
                    scope="terms",
                    filters=((filter_column, doc_key), (_TERM, term)),
                    count=int(pd.to_numeric(row.get(_COUNT, 1), errors="coerce") or 1),
                    describe=f"{labels[doc]} / {term}: TF-IDF {score:.4f}",
                    phrase=term,
                    lemma=counted_on_lemmas(provenance.settings),
                ),
            )
        )

    prepared = PreparedPanel(
        panel=TFIDF_HEATMAP.name,
        shape="heatmap",
        title="TF-IDF across documents and top terms",
        subtitle=f"{len(ordered_docs)} document(s) x {len(terms)} term(s), by summed TF-IDF",
        marks=tuple(marks),
        x_label="Term",
        y_label="Document (dated order)",
        provenance=provenance,
        data=cell.reset_index(drop=True),
        x_categories=terms,
        y_categories=y_categories,
        color_scale="sequential",
        value_label="TF-IDF",
        height=max(500, 24 * len(ordered_docs) + 120),
        width=max(900, 40 * len(terms) + 200),
        notes=(
            "This table already holds only each document's own top-N TF-IDF terms (the tfidf run's --top-n). "
            "A blank cell means the term was not in that document's top list, not that its TF-IDF is zero -- "
            "the term may still occur there, just outside the ranked terms this table records.",
            "Columns are the terms with the highest TF-IDF summed across every document's top list, which "
            "favours a term that is distinctive in several documents over one that is extremely distinctive "
            "in only one.",
        ),
    )
    return Result.success(prepared, *diagnostics)


TFIDF_HEATMAP = PanelDefinition(
    name="tfidf_heatmap",
    title="TF-IDF across documents and top terms",
    question="Which distinctive terms recur across documents, and which are unique to one?",
    tool="tfidf",
    shape="heatmap",
    summary="Documents (dated) x the corpus's overall top TF-IDF terms; value is TF-IDF.",
    requires=(_DOC_ID, _DOC, _TERM, _TFIDF),
    params=(
        PanelParam(
            name="top-k",
            type="int",
            default=25,
            minimum=2,
            maximum=100,
            label="Terms to show",
            help="How many of the highest summed-TF-IDF terms become the heatmap's columns.",
        ),
    ),
    build=_tfidf_heatmap,
    notes=(
        "Only terms already present in some document's own top-N TF-IDF list can appear at all -- this is a "
        "map of the top of the distribution, not the full term-by-document matrix.",
    ),
)


TERMS_PANELS: tuple[PanelDefinition, ...] = (
    NGRAMS_RANKED_TERMS,
    WORDLIST_RANKED_TERMS,
    NOMINALIZATION_RANKED_TERMS,
    SEMANTIC_RANKED_LEMMAS,
    VERBNET_RANKED_CLASSES,
    FRAMENET_RANKED_FRAMES,
    SYMBOLIC_RANKED_SPACES,
    SYMBOLIC_RANKED_ACTORS,
    WORDNET_RANKED_CATEGORIES,
    NRC_EMOTION_HEATMAP,
    TFIDF_TOP_TERMS,
    TFIDF_HEATMAP,
)
