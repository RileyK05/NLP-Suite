"""Profiler executor (FR-7.3) — runs a validated Plan over shared inputs.

One shared corpus and at most one shared parse feed every tool; each tool
runs through a thin adapter that calls the existing ``core.analysis``
API (no analysis logic lives here). A failing tool becomes a failed
outcome — the batch always continues, mirroring the legacy ``run_profile``
rule. Timings are recorded per tool for the report child.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
import json
from os import PathLike
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from core.analysis import word_meaning
from core.analysis.bert_extract import summarize as summarize_bert
from core.analysis.bert_topics import bert_topics
from core.analysis.clause_svo import clause_frequencies, extract_svo
from core.analysis.collocations import collocations as collocation_measures
from core.analysis.conll_wordlist import run as wordlist
from core.analysis.contextual import contextual_vectors, sense_uses, wsi_senses
from core.analysis.coreference import run as run_coreference
from core.analysis.corpus_statistics import run as corpus_statistics
from core.analysis.dispersion import dispersion as dispersion_measures
from core.analysis.doc_duplicates import exact_groups, fuzzy_pairs, normalized_groups
from core.analysis.doc_similarity import find_duplicates, pairwise_similarity
from core.analysis.k_sentences import run as k_sentences
from core.analysis.keyness import keyness as keyness_scores
from core.analysis.knowledge_graph import build as kg_build
from core.analysis.kwic import concordance as kwic_concordance
from core.analysis.lda import fit_lda, segment_tokens_for, tokens_from_frame
from core.analysis.lexical_diversity import run as run_lexdiv
from core.analysis.lexicon_series import facet_labels, lexicon_series, parse_lexicon
from core.analysis.movement import movement_summary, movement_tracks
from core.analysis.ner import entity_timeline, location_tracking
from core.analysis.ngram_cooccurrence import run as cooccurrence
from core.analysis.ngrams import collocations, ngrams
from core.analysis.nominalization import (
    detect_nominalizations,
    load_curated_words,
    sentence_frequency,
)
from core.analysis.nrc import emotions
from core.analysis.readability import run as run_readability
from core.analysis.semantic import aggregate as semantic_tags
from core.analysis.sentence_complexity import run as run_complexity
from core.analysis.sentiment_swn_hedono import hedonometer, sentiwordnet
from core.analysis.sentiment_vader_anew import anew, load_vader_analyzer, vader, vader_sentences
from core.analysis.spellcheck import load_wordlist, run as run_spellcheck
from core.analysis.style import concreteness, iconic_words, iconicity
from core.analysis.svo_compare import compare as svo_compare
from core.analysis.table_search import SearchFilter, search as table_search
from core.analysis.text_statistics import run as text_statistics
from core.analysis.tfidf import tfidf as tfidf_terms
from core.analysis.topic_stability import topic_stability
from core.analysis.word_embeddings import project_tsne, train as train_w2v
from core.conll.schema import Col
from core.file_ops.search import search_in_text
from core.io.reader import Corpus, display_names
from core.narrative.arcs import emotion_arc, length_arc
from core.narrative.characters import character_arcs, character_mentions
from core.profiler.plan import Plan
from core.research.phrase import EvidenceRequest, PhraseQuery, Tokenization, track_phrase
from core.result import Diagnostic, Result
from core.viz.shapes import story_shape

__all__ = [
    "ADAPTERS",
    "ADAPTER_NEEDS",
    "Adapter",
    "BatchContext",
    "BatchResult",
    "ToolOutcome",
    "execute",
]

Needs = frozenset[str]  # subset of {"corpus", "table"}


@dataclass(frozen=True, slots=True)
class BatchContext:
    """Shared inputs for one batch run (parse happens once, outside)."""

    corpus: Corpus | None = None
    table: pd.DataFrame | None = None
    #: How the parser that produced *table* splits text into tokens, for the
    #: tools that have to split a question the same way the corpus was split.
    #: Absent, a phrase question answers by a guess at some parser's rules and
    #: reports a confident zero where the live workspace reports occurrences --
    #: the same defect in the same corpus, differing only by which door the
    #: question came through.
    tokenizer: Tokenization | None = None


#: The column every per-document result carries, and the key the corpus is
#: joined on. Always ``str(doc.doc_id)``, assigned at intake and never parsed
#: out of a filename, so the join cannot drift with a renamed file.
DOCUMENT_ID = "Document ID"
DOCUMENT = "Document"
#: What a dated corpus contributes to a per-document result. ``Date`` is the
#: ISO day, ``Year`` the integer, because the two answer different questions:
#: a timeline needs the day's order, a cohort comparison needs the year as a
#: grouping label.
DATE = "Date"
YEAR = "Year"


def _document_dates(corpus: Corpus | None) -> dict[str, date]:
    """When each document is from, by the id every result reports it under.

    Empty when nothing is dated, which is the common case for a directory of
    files named by subject rather than by day.
    """
    if corpus is None:
        return {}
    return {str(doc.doc_id): doc.date for doc in corpus.docs if doc.date is not None}


def _dated(frame: pd.DataFrame, dates: dict[str, date]) -> pd.DataFrame:
    """One per-document result with the corpus's dates joined onto it.

    The corpus has known each document's date since intake and no result has
    ever carried it, so "has this changed over ninety years" -- the question a
    corpus spanning ninety years exists to answer -- could not be asked of any
    table this suite produced. Every chart was a ranking of documents against
    each other, which is why a run over 87 State of the Union addresses drew
    87 bars in no order anyone chose.

    Added here rather than in each tool so that live and published results
    stay the same rows: the preview is the thing itself, and a chart drawn
    over time in the workbench is one that can be published.

    Deliberately narrow. A frame with no ``Document ID`` is not per-document;
    a frame already carrying either column has its own, better answer; and a
    frame whose ids match nothing dated gets no columns at all rather than two
    empty ones, which would be an axis offering nothing but blanks.
    """
    if DOCUMENT_ID not in frame.columns or DATE in frame.columns or YEAR in frame.columns:
        return frame
    keys = frame[DOCUMENT_ID].astype(str)
    resolved = keys.map(dates)
    if resolved.isna().all():
        return frame
    out = frame.copy()
    # Beside the document's own name, where a reader looks for what a row is
    # about -- not appended after the measures, where the axis pickers bury it.
    at = out.columns.get_loc(DOCUMENT) + 1 if DOCUMENT in out.columns else out.columns.get_loc(DOCUMENT_ID) + 1
    out.insert(int(at), DATE, [value.isoformat() if isinstance(value, date) else None for value in resolved])
    out.insert(int(at) + 1, YEAR, [value.year if isinstance(value, date) else None for value in resolved])
    return out


def _with_dates(frames: dict[str, pd.DataFrame], corpus: Corpus | None) -> dict[str, pd.DataFrame]:
    """Every per-document frame a tool produced, dated where the corpus can."""
    dates = _document_dates(corpus)
    if not dates:
        return frames
    return {name: _dated(frame, dates) for name, frame in frames.items()}


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What one tool produced (or why it did not)."""

    name: str
    ok: bool
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Ordered per-tool outcomes for one Plan."""

    outcomes: tuple[ToolOutcome, ...] = ()


Adapter = Callable[[BatchContext, dict[str, object]], Result[dict[str, pd.DataFrame]]]

if TYPE_CHECKING:  # pragma: no cover -- typing only; the modules load lazily
    from core.analysis.shape_reduction import ReductionResult, ShapeMatrix


def _int(params: dict[str, object], key: str, default: int) -> int:
    """Validated int param (plan guarantees the type; direct-Plan misuse fails fast)."""
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"param {key!r} must be int, got {value!r}")
    return value


def _tokens(params: dict[str, object], key: str) -> list[str] | None:
    """A JSON list of already-resolved tokens, or None if the caller has none.

    Plans carry scalars, so a resolved subject travels as JSON text. Unreadable
    text is None rather than an error: the question can still be answered by
    tokenizing it again, and the answer says that is what happened.
    """
    raw = params.get(key, "")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(value, list):
        return None
    tokens = [str(item) for item in value if str(item).strip()]
    return tokens or None


def _float(params: dict[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"param {key!r} must be a number, got {value!r}")
    return float(value)


def _str(params: dict[str, object], key: str, default: str) -> str:
    value = params.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"param {key!r} must be str, got {value!r}")
    return value


def _bool(params: dict[str, object], key: str, default: bool) -> bool:
    value = params.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"param {key!r} must be bool, got {value!r}")
    return value


def _missing_input(tool: str, what: str) -> Result[dict[str, pd.DataFrame]]:
    code = "PROFILER_NO_TABLE" if what == "table" else "PROFILER_NO_CORPUS"
    return Result.failure(Diagnostic.error(code, f"{tool} needs {what}, which the batch did not provide", tool=tool))


def _lexicon_param(params: dict[str, object], key: str) -> str | None:
    """Validated optional lexicon-path param (plan guarantees str-or-absent)."""
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, (str, PathLike)):
        raise TypeError(f"param {key!r} must be a path, got {value!r}")
    return str(value)


def _adapt_readability(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    _ = params
    if ctx.corpus is None:
        return _missing_input("readability", "corpus")
    result = run_readability(ctx.corpus)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"readability.csv": result.unwrap().to_frame()}, *result.diagnostics)


def _adapt_lexical_diversity(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.corpus is None:
        return _missing_input("lexical_diversity", "corpus")
    result = run_lexdiv(ctx.corpus, seed=_int(params, "seed", 42))
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"lexical_diversity.csv": result.unwrap().to_frame()}, *result.diagnostics)


def _adapt_doc_similarity(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.corpus is None:
        return _missing_input("doc_similarity", "corpus")
    stopwords: list[str] | None = None
    stopwords_path = params.get("stopwords")
    if stopwords_path is not None:
        try:
            stopwords = [
                line.strip()
                for line in Path(str(stopwords_path)).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as exc:
            return Result.failure(Diagnostic.error("PROFILER_BAD_PARAM", f"cannot read stopwords: {exc}"))
    pairs = pairwise_similarity(ctx.corpus, stopwords)
    if pairs.value is None:
        return Result.failure(*pairs.diagnostics)
    dupes = find_duplicates(ctx.corpus, _float(params, "threshold", 80.0), stopwords)
    if dupes.value is None:
        return Result.failure(*dupes.diagnostics)
    return Result.success(
        {"doc_pairs.csv": pairs.unwrap().to_frame(), "duplicates.csv": dupes.unwrap().to_frame()},
        *pairs.diagnostics,
        *dupes.diagnostics,
    )


def _adapt_doc_duplicates(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.corpus is None:
        return _missing_input("doc_duplicates", "corpus")
    exact = exact_groups(ctx.corpus)
    if exact.value is None:
        return Result.failure(*exact.diagnostics)
    normalized = normalized_groups(ctx.corpus)
    if normalized.value is None:
        return Result.failure(*normalized.diagnostics)
    fuzzy = fuzzy_pairs(ctx.corpus, _float(params, "threshold", 80.0))
    if fuzzy.value is None:
        return Result.failure(*fuzzy.diagnostics)
    return Result.success(
        {
            "exact_groups.csv": exact.unwrap().to_frame(),
            "normalized_groups.csv": normalized.unwrap().to_frame(),
            "fuzzy_pairs.csv": fuzzy.unwrap().to_frame(),
        },
        *exact.diagnostics,
        *normalized.diagnostics,
        *fuzzy.diagnostics,
    )


def _adapt_spellcheck(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.corpus is None:
        return _missing_input("spellcheck", "corpus")
    if params.get("correct"):
        # Corrected copies are files, not batch frames — CLI-only surface.
        return Result.failure(
            Diagnostic.error(
                "PROFILER_UNSUPPORTED_OPTION",
                "spellcheck 'correct' writes file copies and cannot run in a batch; use the CLI",
                tool="spellcheck",
            )
        )
    vocab_result = load_wordlist(Path(str(params["wordlist"])))
    if vocab_result.value is None:
        return Result.failure(*vocab_result.diagnostics)
    result = run_spellcheck(ctx.corpus, vocab_result.unwrap(), _float(params, "threshold", 80.0))
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"spellcheck.csv": result.unwrap().to_frame()}, *result.diagnostics)


def _search_conll(
    ctx: BatchContext, query: str, case_sensitive: bool, use_regex: bool
) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("search", "table")
    filt = SearchFilter(
        field="Lemma", op="regex" if use_regex else "contains", value=query, case_sensitive=case_sensitive
    )
    result = table_search(ctx.table, [filt])
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"conll_hits.csv": result.unwrap().to_frame()}, *result.diagnostics)


def _search_text(
    ctx: BatchContext, query: str, case_sensitive: bool, use_regex: bool
) -> Result[dict[str, pd.DataFrame]]:
    if ctx.corpus is None:
        return _missing_input("search", "corpus")
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    names = display_names(ctx.corpus.docs)
    for doc in ctx.corpus.docs:
        hits = search_in_text(doc.text, query, case_sensitive=case_sensitive, use_regex=use_regex)
        if hits.value is None:
            return Result.failure(*hits.diagnostics)
        for line_no, line in hits.unwrap():
            rows.append({"Document": names[doc.doc_id], "Line": line_no, "Match": line})
        diags.extend(hits.diagnostics)
    return Result.success({"text_hits.csv": pd.DataFrame(rows, columns=["Document", "Line", "Match"])}, *diags)


def _adapt_search(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    mode = _str(params, "mode", "text")
    query = _str(params, "query", "")
    case_sensitive = _bool(params, "case-sensitive", False)
    use_regex = _bool(params, "regex", False)
    if mode == "csv":
        return Result.failure(
            Diagnostic.error(
                "PROFILER_MODE_UNAVAILABLE",
                "search csv mode needs an analyst-chosen CSV file, unavailable in a batch",
                tool="search",
            )
        )
    if mode == "conll":
        return _search_conll(ctx, query, case_sensitive, use_regex)
    return _search_text(ctx, query, case_sensitive, use_regex)


def _adapt_sentence_complexity(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    _ = params
    if ctx.table is None:
        return _missing_input("sentence_complexity", "table")
    result = run_complexity(ctx.table)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"sentence_complexity.csv": result.unwrap().to_frame()}, *result.diagnostics)


def _adapt_lda_gensim(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    """Gensim LDA plus the two views the Gensim GUI is graded on (HW2).

    Also cuts the corpus into paragraphs (``core.analysis.topic_flow``) and
    scores each one with the fitted model, as ``topic_flow.csv`` — the
    paragraph-level table the topic-flow ribbon draws. Segmentation needs the
    corpus; it is optional because a CSV-sourced run has no text to cut.
    """
    if ctx.table is None:
        return _missing_input("lda_gensim", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    nouns_only = _bool(params, "nouns-only", False)
    tokens = tokens_from_frame(ctx.table, field=field, nouns_only=nouns_only)
    if not tokens:
        return Result.failure(Diagnostic.error("PROFILER_NO_INPUT", "lda_gensim found no usable tokens"))
    lam = params.get("lambda", 0.6)
    if not isinstance(lam, (int, float)) or isinstance(lam, bool):
        raise TypeError(f"param 'lambda' must be a number, got {lam!r}")
    remove_stopwords = not _bool(params, "keep-stopwords", False)

    # Paragraph segmentation, before the model exists, so the same token
    # filter feeds both the fit and the segments. Needs the corpus text; a
    # live preview without one simply skips the flow table.
    plan = None
    segment_bags: dict[str, pd.Series] = {}
    diagnostics_flow: list[Diagnostic] = []
    if ctx.corpus is not None:
        try:
            from core.analysis.topic_flow import plan_segments

            plan = plan_segments(ctx.table, ctx.corpus)
        except Exception as exc:  # the flow table is a convenience; the fit still counts
            plan = None
            diagnostics_flow.append(Diagnostic.warning("TOPIC_FLOW_FAILED", f"paragraph segmentation failed: {exc}"))
        if plan is not None:
            # The fit's own filter, nouns-only included: a flow scored on
            # tokens the model never trained on describes another model.
            segment_bags = segment_tokens_for(
                ctx.table,
                plan.labels,
                field=field,
                nouns_only=nouns_only,
                remove_stopwords=remove_stopwords,
            )

    fitted = fit_lda(
        tokens,
        n_topics=_int(params, "topics", 3),
        top_n=_int(params, "top-n", 5),
        seed=_int(params, "seed", 100),
        remove_stopwords=remove_stopwords,
        # c_v coherence re-scans the whole corpus and cost over six minutes on
        # the 87-speech corpus at 20 topics -- forty times the model fit. A
        # live preview is a look, not a provenance-bearing run, so it skips
        # that scan; publishing still computes and records it, where the wait
        # is expected and the number is persisted with the artifact.
        coherence=False,
        relevance_lambda=float(lam),
        segments=plan,
        segment_tokens=segment_bags or None,
    )
    if fitted.value is None:
        return Result.failure(*fitted.diagnostics)
    model = fitted.unwrap()
    frames: dict[str, pd.DataFrame] = {
        "topics.csv": model.topics,
        "topics_dominant.csv": model.dominant,
    }
    if model.relevance is not None:
        frames["terms_by_relevance.csv"] = model.relevance
    if model.intertopic is not None:
        frames["intertopic_distances.csv"] = model.intertopic
        from core.viz.embeddings import tsne_html

        # The Intertopic Distance Map, drawn from the same placement data.
        # Knowable degradation: without plotly the coordinates table stands.
        renamed = model.intertopic.rename(columns={"Topic": "Word", "X": "X", "Y": "Y"})
        plot = tsne_html(renamed, title="Intertopic Distance Map")
        if plot.value is not None:
            frames["intertopic_map.html"] = pd.DataFrame({"html": [plot.unwrap()]})
    if model.flow is not None and not model.flow.empty:
        frames["topic_flow.csv"] = model.flow
    return Result.success(frames, *fitted.diagnostics, *diagnostics_flow)


def _adapt_lda_stability(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    """Topic stability: refit at several seeds, match topics back to the first.

    ``seeds`` is a CSV list (default "100,101,102"); the first entry is the
    reference seed the others are matched against. Writes the summary the
    stability panel draws plus the raw per-seed matches.
    """
    if ctx.table is None:
        return _missing_input("lda_stability", "table")
    raw = params.get("seeds", "100,101,102")
    seeds: list[int] = []
    for part in str(raw).split(","):
        try:
            seeds.append(int(part.strip()))
        except ValueError as exc:
            raise TypeError(f"param 'seeds' must be comma-separated whole numbers, got {raw!r}") from exc
    # Comparing seeds is the point: one refit has nothing to compare against.
    _MINIMUM_SEEDS = 2
    if len(seeds) < _MINIMUM_SEEDS:
        return Result.failure(
            Diagnostic.error(
                "STABILITY_ONE_SEED",
                "topic stability compares refits: give at least two seeds, e.g. seeds=100,101,102 "
                "(the first is the reference).",
            )
        )
    result = topic_stability(
        tokens_from_frame(ctx.table, nouns_only=_bool(params, "nouns-only", False)),
        n_topics=_int(params, "topics", 3),
        seeds=seeds,
        top_n=_int(params, "top-n", 10),
        passes=_int(params, "passes", 10),
        stable_at=_float(params, "stable-at", 0.5),
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    stability = result.unwrap()
    return Result.success(
        {"summary.csv": stability.summary, "matches.csv": stability.matches},
        *result.diagnostics,
    )


def _adapt_word_embeddings(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("word2vec_gensim", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    trained = train_w2v(
        ctx.table,
        field=field,
        min_count=_int(params, "min-count", 2),
        vector_size=_int(params, "vector-size", 100),
        window=_int(params, "window", 5),
        seed=_int(params, "seed", 42),
        epochs=_int(params, "epochs", 20),
        sg=_int(params, "sg", 1),
        remove_stopwords=_bool(params, "remove-stopwords", False),
    )
    if trained.value is None:
        return Result.failure(*trained.diagnostics)
    model = trained.unwrap()
    frames = {
        "vectors.csv": pd.DataFrame(
            [
                {"Word": word, "Count": count, "Vector": ",".join(f"{v:.4f}" for v in vec)}
                for word, count, vec in zip(model.words, model.counts, model.vectors, strict=True)
            ],
            columns=["Word", "Count", "Vector"],
        )
    }
    diags = list(trained.diagnostics)
    projected = project_tsne(model, seed=model.seed)
    diags.extend(projected.diagnostics)
    if projected.value is not None and not projected.unwrap().empty:
        frames["tsne.csv"] = _tsne_with_counts(projected.unwrap(), frames["vectors.csv"])
        # Legacy parity: the interactive scatter alongside the coordinates
        # (word2vec_tsne_plot_util.py). Knowable degradation without plotly.
        # The batch writer routes *.html names to write_html (see batch.py).
        from core.viz.embeddings import tsne_html

        plot = tsne_html(projected.unwrap(), title="t-SNE word vectors")
        diags.extend(plot.diagnostics)
        if plot.value is not None:
            frames["tsne.html"] = pd.DataFrame({"html": [plot.unwrap()]})
    if model.words:
        space = word_meaning.space_of(model.words, model.counts, np.asarray(model.vectors, dtype=float))
        _meaning_columns(frames, ctx.table, "form" if field == Col.FORM else "lemma", space)
    query = params.get("query")
    if query is not None:
        if not isinstance(query, str):
            raise TypeError(f"param 'query' must be str, got {query!r}")
        key = query.strip().lower()
        if key not in model.words:
            return Result.failure(Diagnostic.error("W2V_UNKNOWN_WORD", f"{query!r} not in vocabulary"))
        scored = sorted(
            ((word, model.similarity(key, word)) for word in model.words if word != key),
            key=lambda pair: pair[1],
            reverse=True,
        )
        top_n = _int(params, "top-n", 5)
        frames["neighbours.csv"] = pd.DataFrame(
            [{"Word": key, "Neighbor": word, "Cosine": round(score, 4)} for word, score in scored[:top_n]],
            columns=["Word", "Neighbor", "Cosine"],
        )
    return Result.success(frames, *diags)


def _adapt_sentiment_vader_anew(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("sentiment_vader_anew", "table")
    field = Col.FORM if params.get("field", "form") == "form" else Col.LEMMA
    anew_field = Col.FORM if params.get("anew-field", "lemma") == "form" else Col.LEMMA
    loaded = load_vader_analyzer(_lexicon_param(params, "vader-lexicon"))
    if not loaded.ok:
        return Result.failure(*loaded.diagnostics)
    docs = vader(ctx.table, field=field, analyzer=loaded.unwrap())
    if docs.value is None:
        return Result.failure(*docs.diagnostics)
    sents = vader_sentences(ctx.table, field=field, analyzer=loaded.unwrap())
    if sents.value is None:
        return Result.failure(*sents.diagnostics)
    frames = {"vader.csv": docs.unwrap(), "vader_sentences.csv": sents.unwrap()}
    diagnostics = [*docs.diagnostics, *sents.diagnostics, *loaded.diagnostics]
    if params.get("analysis") != "vader":
        anew_result = anew(ctx.table, field=anew_field, lexicon=_lexicon_param(params, "anew-lexicon"))
        if anew_result.value is None:
            return Result.failure(*anew_result.diagnostics)
        frames["anew.csv"] = anew_result.unwrap()
        diagnostics.extend(anew_result.diagnostics)
    return Result.success(frames, *diagnostics)


def _adapt_sentiment_swn_hedono(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("sentiment_swn_hedono", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    swn = sentiwordnet(ctx.table, field=field)
    if swn.value is None:
        return Result.failure(*swn.diagnostics)
    hedo = hedonometer(ctx.table, field=field, lexicon=_lexicon_param(params, "hedonometer-lexicon"))
    if hedo.value is None:
        return Result.failure(*hedo.diagnostics)
    return Result.success(
        {"swn.csv": swn.unwrap(), "hedonometer.csv": hedo.unwrap()}, *swn.diagnostics, *hedo.diagnostics
    )


def _adapt_nrc(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("nrc", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    result = emotions(ctx.table, field=field, lexicon=_lexicon_param(params, "lexicon"))
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"nrc.csv": result.unwrap()}, *result.diagnostics)


def _adapt_nominalization(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("nominalization", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    check_ending = params.get("check-ending", True)
    if not isinstance(check_ending, bool):
        raise TypeError(f"param 'check-ending' must be bool, got {check_ending!r}")
    curated: tuple[str, ...] = ()
    curated_list = _lexicon_param(params, "curated-list")
    if curated_list is not None:
        loaded = load_curated_words(curated_list)
        if loaded.value is None:
            return Result.failure(*loaded.diagnostics)
        curated = tuple(sorted(loaded.unwrap()))
    noms = detect_nominalizations(ctx.table, field=field, check_ending=check_ending, curated_words=curated)
    if noms.value is None:
        return Result.failure(*noms.diagnostics)
    sents = sentence_frequency(ctx.table, field=field, check_ending=check_ending, curated_words=curated)
    if sents.value is None:
        return Result.failure(*sents.diagnostics)
    return Result.success(
        {"nominalization.csv": noms.unwrap(), "nominalization_by_sentence.csv": sents.unwrap()},
        *noms.diagnostics,
        *sents.diagnostics,
    )


def _adapt_style(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("style", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    analysis = params.get("analysis", "concreteness")
    if analysis not in ("concreteness", "iconicity"):
        raise TypeError(f"param 'analysis' must be concreteness|iconicity, got {analysis!r}")
    if analysis == "concreteness":
        scored = concreteness(ctx.table, field=field, lexicon=_lexicon_param(params, "concreteness-lexicon"))
        if scored.value is None:
            return Result.failure(*scored.diagnostics)
        return Result.success({"style_concreteness.csv": scored.unwrap()}, *scored.diagnostics)
    icon_lex = _lexicon_param(params, "iconicity-lexicon")
    scored = iconicity(ctx.table, field=field, lexicon=icon_lex)
    if scored.value is None:
        return Result.failure(*scored.diagnostics)
    min_rating = _float(params, "min-rating", 5.0)
    max_sd = _float(params, "max-rating-sd", 2.0)
    words = iconic_words(ctx.table, field=field, lexicon=icon_lex, min_rating=min_rating, max_rating_sd=max_sd)
    if words.value is None:
        return Result.failure(*words.diagnostics)
    return Result.success(
        {"style_iconicity.csv": scored.unwrap(), "style_iconic_words.csv": words.unwrap()},
        *scored.diagnostics,
        *words.diagnostics,
    )


def _adapt_coreference(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("coreference", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    result = run_coreference(ctx.table, field=field)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"coreference.csv": result.unwrap()}, *result.diagnostics)


def _adapt_narrative(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("narrative", "table")
    field = Col.FORM if params.get("field", "form") == "form" else Col.LEMMA
    scored = emotion_arc(ctx.table, field=field)
    if scored.value is None:
        return Result.failure(*scored.diagnostics)
    lengths = length_arc(ctx.table)
    if lengths.value is None:
        return Result.failure(*lengths.diagnostics)
    mentions = character_mentions(ctx.table)
    if mentions.value is None:
        return Result.failure(*mentions.diagnostics)
    arcs = character_arcs(ctx.table, field=field)
    if arcs.value is None:
        return Result.failure(*arcs.diagnostics)
    return Result.success(
        {
            "emotion_arc.csv": scored.unwrap(),
            "length_arc.csv": lengths.unwrap(),
            "character_mentions.csv": mentions.unwrap(),
            "character_arcs.csv": arcs.unwrap(),
        },
        *scored.diagnostics,
        *lengths.diagnostics,
        *mentions.diagnostics,
        *arcs.diagnostics,
    )


def _adapt_contextual(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("word_sense_induction", "table")
    field = Col.FORM if params.get("field", "form") == "form" else Col.LEMMA
    model = params.get("model", "bert-base-uncased")
    if not isinstance(model, str):
        raise TypeError(f"param 'model' must be str, got {model!r}")
    vectors = contextual_vectors(ctx.table, field=field, model=model)
    if vectors.value is None:
        return Result.failure(*vectors.diagnostics)
    # Senses group by lemma over the same token vectors; reuse them rather
    # than embedding every token of the corpus a second time.
    senses = wsi_senses(
        ctx.table, field=Col.LEMMA if field == Col.FORM else Col.FORM, model=model, vectors=vectors.unwrap()
    )
    if senses.value is None:
        return Result.failure(*senses.diagnostics)
    frames = {"contextual_vectors.csv": vectors.unwrap(), "wsi.csv": senses.unwrap()}
    # The sentences behind each sense, for the figures that let a reader judge the split.
    shown = display_names(ctx.corpus.docs) if ctx.corpus is not None else {}
    uses = sense_uses(ctx.table, senses.unwrap(), names={str(doc): name for doc, name in shown.items()} or None)
    if not uses.empty:
        classes = word_meaning.word_classes(ctx.table, Col.LEMMA.value)
        frames["senses.csv"] = uses.assign(
            **{word_meaning.WORD_CLASS: uses["Lemma"].astype(str).str.lower().map(classes).fillna("")}
        )
    return Result.success(frames, *vectors.diagnostics, *senses.diagnostics)


def _adapt_clause_svo(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    _ = params
    if ctx.table is None:
        return _missing_input("clause_svo", "table")
    clauses = clause_frequencies(ctx.table)
    if clauses.value is None:
        return Result.failure(*clauses.diagnostics)
    svos = extract_svo(ctx.table)
    if svos.value is None:
        return Result.failure(*svos.diagnostics)
    clause_frame = pd.DataFrame(
        [{"Clause Tag": row.tag, "Count": row.count} for row in clauses.unwrap()],
        columns=["Clause Tag", "Count"],
    )
    svo_frame = pd.DataFrame(
        [
            {
                "Subject": row.subject,
                "Verb": row.verb,
                "Object": row.obj,
                "Sentence ID": row.sentence_id,
                "Document ID": row.document_id,
                "Document": row.document,
            }
            for row in svos.unwrap()
        ],
        columns=["Subject", "Verb", "Object", "Sentence ID", "Document ID", "Document"],
    )
    return Result.success({"clauses.csv": clause_frame, "svo.csv": svo_frame}, *clauses.diagnostics, *svos.diagnostics)


def _adapt_ner(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    _ = params
    if ctx.table is None:
        return _missing_input("ner", "table")
    timeline = entity_timeline(ctx.table)
    if timeline.value is None:
        return Result.failure(*timeline.diagnostics)
    locations = location_tracking(ctx.table)
    if locations.value is None:
        return Result.failure(*locations.diagnostics)
    frames = {"entity_timeline.csv": timeline.unwrap(), "locations.csv": locations.unwrap()}
    diags = [*timeline.diagnostics, *locations.diagnostics]
    # CAP-NER-03: person-location co-occurrence pairs (no geocode in batch;
    # the offline KB would silently bound the place list — CLI --geocode for it).
    tracks = movement_tracks(ctx.table, geocode=False)
    diags.extend(tracks.diagnostics)
    if tracks.value is not None:
        pairs = tracks.unwrap()
        frames["movement_tracks.csv"] = pairs
        frames["movement_summary.csv"] = movement_summary(pairs)
    return Result.success(frames, *diags)


def _adapt_bert_extract(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("bert_extract", "table")
    model = params.get("model", "bert-base-uncased")
    if not isinstance(model, str):
        raise TypeError(f"param 'model' must be str, got {model!r}")
    result = summarize_bert(ctx.table, sentences=_int(params, "sentences", 3), model=model)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"bert_extract.csv": result.unwrap()}, *result.diagnostics)


def _adapt_ngrams(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("ngrams", "table")
    field = Col.FORM if params.get("field", "form") == "form" else Col.LEMMA
    grams = ngrams(ctx.table, n=_int(params, "n", 2), field=field)
    if grams.value is None:
        return Result.failure(*grams.diagnostics)
    colloc = collocations(ctx.table, field=field, min_count=_int(params, "min-count", 2))
    if colloc.value is None:
        return Result.failure(*colloc.diagnostics)
    return Result.success({"ngrams.csv": grams.unwrap(), "collocations.csv": colloc.unwrap()}, *grams.diagnostics)


def _adapt_phrase_distribution(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    """Publish the same question contract used by the interactive workspace."""
    if ctx.table is None:
        return _missing_input("phrase_distribution", "table")
    if ctx.corpus is None:
        return _missing_input("phrase_distribution", "corpus")

    phrase = _str(params, "phrase", "").strip()
    comparison = _str(params, "comparison", "").strip()
    case_sensitive = _bool(params, "case-sensitive", False)
    normalize = _bool(params, "normalize", False)
    match_lemma = _bool(params, "match-lemma", False)
    match_nominalization = _bool(params, "match-nominalization", False)
    bins = _int(params, "position-bins", 10)
    snapshot_id = _str(params, "snapshot-id", "")
    evidence_year = _int(params, "evidence-year", 0)
    evidence_document = _str(params, "evidence-document", "")
    subjects = [phrase, *([comparison] if comparison else [])]
    # The tokens each subject was resolved into when the question was saved.
    # Publishing asks the saved question, not today's reading of its text.
    saved_tokens = [
        _tokens(params, "resolved-tokens"),
        *([_tokens(params, "comparison-tokens")] if comparison else []),
    ]

    answers = [
        track_phrase(
            ctx.table,
            ctx.corpus,
            PhraseQuery(
                text=subject,
                case_sensitive=case_sensitive,
                position_bins=bins,
                normalize=normalize,
                match_lemma=match_lemma,
                match_nominalization=match_nominalization,
            ),
            snapshot_id=snapshot_id,
            evidence=EvidenceRequest(all_rows=True),
            tokenize=ctx.tokenizer.split if ctx.tokenizer else None,
            tokenizer_name=ctx.tokenizer.name if ctx.tokenizer else "",
            resolved_tokens=tokens,
        )
        for subject, tokens in zip(subjects, saved_tokens, strict=True)
    ]

    def with_subject(rows: list[dict[str, object]], subject: str) -> list[dict[str, object]]:
        return [{"subject": subject, **row} for row in rows]

    summary_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    document_rows: list[dict[str, object]] = []
    occurrence_rows: list[dict[str, object]] = []
    for subject, answer in zip(subjects, answers, strict=True):
        summary_rows.extend(with_subject([answer["summary"]], subject))
        time_rows.extend(with_subject(answer["time"], subject))
        position_rows.extend(with_subject(answer["position"], subject))
        document_rows.extend(with_subject(answer["documents"], subject))
        for row in with_subject(answer["evidence"]["rows"], subject):
            row["source_token_ids"] = json.dumps(row.get("source_token_ids", []), ensure_ascii=False)
            row["selected_evidence"] = (not evidence_year or row.get("year") == evidence_year) and (
                not evidence_document or row.get("document_id") == evidence_document
            )
            occurrence_rows.append(row)

    question_row = {
        "question_id": _str(params, "question-id", ""),
        "question_name": _str(params, "question-name", ""),
        "question_revision": _int(params, "question-revision", 1),
        "snapshot_id": snapshot_id,
        "phrase": phrase,
        "comparison": comparison or None,
        "case_sensitive": case_sensitive,
        "normalize": normalize,
        "match_lemma": match_lemma,
        "match_nominalization": match_nominalization,
        "position_bins": bins,
        "evidence_year": evidence_year or None,
        "evidence_document_id": evidence_document or None,
        "token_field": "surface",
        "boundary": "sentence",
        "punctuation": "preserve",
        "overlapping_matches": True,
        # What "a match" meant here, so a reader of the published tables can
        # tell an answer produced under the saved rules from one produced
        # under whatever rules this machine happens to have.
        "matching_version": answers[0]["question"]["matching_profile"]["version"],
        "matching_source": answers[0]["question"]["matching_profile"]["source"],
        "matching_tokenizer": answers[0]["question"]["matching_profile"]["tokenizer"],
        "resolved_tokens": json.dumps(answers[0]["question"]["subject"]["tokens"], ensure_ascii=False),
        # Every form each token was taken to mean in this corpus. Without it a
        # published count of 174 "promise" occurrences cannot be checked
        # against the documents, because the documents do not contain 174 of
        # the word that was typed.
        "counted_forms": json.dumps(answers[0]["question"]["subject"]["counted_forms"], ensure_ascii=False),
        # An option that was asked for and could not be applied. Empty is the
        # claim that every option asked for was used.
        "matching_unavailable": json.dumps(
            answers[0]["question"]["matching_profile"]["unavailable"], ensure_ascii=False
        ),
        "comparison_tokens": json.dumps(
            answers[1]["question"]["subject"]["tokens"] if comparison else [], ensure_ascii=False
        ),
    }
    time_columns = (
        "subject",
        "year",
        "observed",
        "occurrences",
        "word_tokens",
        "eligible_documents",
        "matching_documents",
        "occurrences_per_10000",
        "document_prevalence_percent",
    )
    occurrence_columns = (
        "subject",
        "id",
        "document_id",
        "document",
        "content_sha256",
        "date",
        "year",
        "sentence_id",
        "token_start",
        "token_end",
        "source_token_ids",
        "word_start",
        "relative_position",
        "position_bin",
        "left",
        "match",
        "right",
        "sentence",
        "character_start",
        "character_end",
        "browser_character_start",
        "browser_character_end",
        "character_offset_unit",
        "left_source",
        "match_source",
        "right_source",
        "exact_source_highlight",
        "selected_evidence",
    )
    return Result.success(
        {
            "phrase_question.csv": pd.DataFrame([question_row]),
            "phrase_summary.csv": pd.DataFrame(summary_rows),
            "phrase_time.csv": pd.DataFrame(time_rows, columns=time_columns),
            "phrase_positions.csv": pd.DataFrame(position_rows),
            "phrase_documents.csv": pd.DataFrame(document_rows),
            "phrase_occurrences.csv": pd.DataFrame(occurrence_rows, columns=occurrence_columns),
        }
    )


def _adapt_ngram_cooccurrence(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("ngram_cooccurrence", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    result = cooccurrence(
        ctx.table, window=_int(params, "window", 5), field=field, min_count=_int(params, "min-count", 1)
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"cooccurrences.csv": result.unwrap()}, *result.diagnostics)


def _adapt_conll_wordlist(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("conll_wordlist", "table")
    field = Col.FORM if params.get("field", "form") == "form" else Col.LEMMA
    category = str(params.get("category", "all"))
    top_n = _int(params, "top-n", 20)
    result = wordlist(
        ctx.table,
        field=field,
        category=category,  # type: ignore[arg-type]  # registry choices pin the literal
        top_n=None if top_n == 0 else top_n,
        case_sensitive=bool(params.get("case-sensitive", False)),
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"wordlist.csv": result.unwrap().to_frame()}, *result.diagnostics)


def _adapt_corpus_statistics(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("corpus_statistics", "table")
    field = Col.FORM if params.get("field", "form") == "form" else Col.LEMMA
    result = corpus_statistics(ctx.table, field=field)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"corpus_statistics.csv": result.unwrap().to_frame()}, *result.diagnostics)


def _adapt_k_sentences(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("k_sentences", "table")
    result = k_sentences(ctx.table, k_first=_int(params, "k-first", 2), k_last=_int(params, "k-last", 2))
    if result.value is None:
        return Result.failure(*result.diagnostics)
    unwrapped = result.unwrap()
    return Result.success(
        {"k_sentences.csv": unwrapped.counts_frame(), "k_repetitions.csv": unwrapped.repetitions_frame()},
        *result.diagnostics,
    )


def _adapt_svo_compare(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    _ = params
    if ctx.table is None:
        return _missing_input("svo_compare", "table")
    result = svo_compare(ctx.table)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    pairs = result.unwrap()
    # The comparison names documents by id alone, so every figure of it was
    # labelled "Doc 1 / Doc 2". The names are appended, never substituted:
    # the id columns stay the key the rows are joined on.
    if Col.DOCUMENT.value in ctx.table.columns and not pairs.empty:
        names = ctx.table.groupby(Col.DOCUMENT_ID.value)[Col.DOCUMENT.value].first()
        names.index = names.index.astype(str)
        pairs = pairs.assign(
            **{
                "Document A": pairs["Doc A"].astype(str).map(names),
                "Document B": pairs["Doc B"].astype(str).map(names),
            }
        )
    return Result.success({"svo_compare.csv": pairs}, *result.diagnostics)


def _adapt_text_statistics(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    _ = params
    if ctx.table is None:
        return _missing_input("text_statistics", "table")
    result = text_statistics(ctx.table)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"text_statistics.csv": result.unwrap().to_frame()}, *result.diagnostics)


def _adapt_table_search(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("table_search", "table")
    field = str(params.get("field") or "")
    if not field:
        return Result.failure(Diagnostic.error("TS_BAD_FIELD", "table_search needs a CoNLL column to search"))
    filt = SearchFilter(
        field=field,
        op=str(params.get("op", "eq")),  # type: ignore[arg-type]
        value=str(params.get("value", "")),
        case_sensitive=bool(params.get("case-sensitive", False)),
        negate=bool(params.get("negate", False)),
    )
    result = table_search(ctx.table, [filt], logic="AND" if params.get("logic", "AND") == "AND" else "OR")
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"search_results.csv": result.unwrap().frame}, *result.diagnostics)


def _adapt_semantic(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("semantic", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    result = semantic_tags(ctx.table, field=field)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"semantic_tags.csv": result.unwrap()}, *result.diagnostics)


def _adapt_knowledge_graph(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("knowledge_graph", "table")
    field = Col.FORM if params.get("field", "form") == "form" else Col.LEMMA
    source = str(params.get("source", "stub"))
    if source == "dbpedia":
        return Result.failure(
            Diagnostic.error(
                "KG_LIVE_UNAVAILABLE_IN_BATCH",
                "the live DBpedia source needs network access; "
                "run the knowledge_graph CLI with --source dbpedia instead (batch runs stay offline)",
            )
        )
    result = kg_build(ctx.table, field=field, source="stub")
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"knowledge_graph.csv": result.unwrap()}, *result.diagnostics)


def _adapt_shapes(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("shapes", "table")
    result = story_shape(ctx.table)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    frames: dict[str, pd.DataFrame] = {"story_shape.csv": result.unwrap()}
    # CAP-VIZ-08: KMeans over shape trajectories when the corpus has enough
    # documents; the diagnostic travels either way.
    from core.viz.shape_clusters import cluster_shapes

    clustered = cluster_shapes(result.unwrap())
    if clustered.value is not None:
        assignments, _k = clustered.unwrap()
        frames["shape_clusters.csv"] = assignments
    return Result.success(frames, *result.diagnostics, *clustered.diagnostics)


def _adapt_kwic(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("kwic", "table")
    field = Col.FORM if params.get("field", "form") == "form" else Col.LEMMA
    query = params.get("query")
    if not isinstance(query, str):
        raise TypeError(f"param 'query' must be str, got {query!r}")
    result = kwic_concordance(
        ctx.table,
        query,
        field=field,
        window=_int(params, "window", 5),
        max_hits=_int(params, "max-hits", 1000),
        case_sensitive=_bool(params, "case-sensitive", False),
        regex=_bool(params, "regex", False),
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"kwic.csv": result.unwrap()}, *result.diagnostics)


def _adapt_dispersion(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("dispersion", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    top_n = _int(params, "top-n", 500)
    result = dispersion_measures(
        ctx.table,
        field=field,
        parts=_str(params, "parts", "document"),
        chunks=_int(params, "chunks", 10),
        min_count=_int(params, "min-count", 5),
        min_length=_int(params, "min-length", 1),
        top_n=top_n or None,
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    frames = {"dispersion.csv": result.unwrap()}
    if _bool(params, "plot", False) and not frames["dispersion.csv"].empty:
        # The registry promises dispersion_plot.html when --plot is set (the
        # CLI writes it); the desktop adapter must keep that promise. Plotting
        # the most frequent surviving terms, as the CLI default does. A failed
        # plot must not discard the table that already succeeded.
        from core.viz.dispersion_plot import MAX_TERMS, dispersion_plot

        terms = frames["dispersion.csv"]["Term"].head(MAX_TERMS).tolist()
        plotted = dispersion_plot(ctx.table, terms, field=field, title="Lexical dispersion")
        if plotted.value is not None:
            # The batch writer routes *.html names to write_html (see batch.py).
            frames["dispersion_plot.html"] = pd.DataFrame({"html": [plotted.unwrap()]})
        diagnostics = (*result.diagnostics, *plotted.diagnostics)
        return Result.success(frames, *diagnostics)
    return Result.success(frames, *result.diagnostics)


def _adapt_tfidf(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("tfidf", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    result = tfidf_terms(
        ctx.table,
        field=field,
        top_n=_int(params, "top-n", 20),
        min_df=_int(params, "min-df", 1),
        max_df_ratio=_float(params, "max-df-ratio", 1.0),
        min_length=_int(params, "min-length", 1),
        sublinear_tf=_bool(params, "sublinear-tf", False),
        normalize=not _bool(params, "no-normalize", False),
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"tfidf.csv": result.unwrap()}, *result.diagnostics)


def _adapt_collocations(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("collocations", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    top_n = _int(params, "top-n", 200)
    result = collocation_measures(
        ctx.table,
        field=field,
        span=_str(params, "span", "adjacent"),
        window=_int(params, "window", 5),
        min_count=_int(params, "min-count", 3),
        min_length=_int(params, "min-length", 1),
        top_n=top_n or None,
        case_sensitive=_bool(params, "case-sensitive", False),
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"collocations.csv": result.unwrap()}, *result.diagnostics)


def _adapt_keyness(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("keyness", "table")
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    pattern = params.get("group-pattern")
    if not isinstance(pattern, str):
        raise TypeError(f"param 'group-pattern' must be str, got {pattern!r}")
    result = keyness_scores(
        ctx.table,
        pattern,
        field=field,
        smoothing=_float(params, "smoothing", 0.5),
        top_n=_int(params, "top-n", 200),
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    return Result.success({"keyness.csv": result.unwrap()}, *result.diagnostics)


def _adapt_lexicon_series(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    """Word groups across an axis. Needs the corpus as well as the parse.

    The parse has the tokens; only the corpus knows when each document is from,
    and a count over time with no time is not the question anyone asked.
    """
    if ctx.table is None or ctx.corpus is None:
        return _missing_input("lexicon_series", "table" if ctx.table is None else "corpus")
    groups = parse_lexicon(_str(params, "terms", ""))
    if groups.value is None:
        return Result.failure(*groups.diagnostics)
    within = None
    written = _str(params, "within", "").strip()
    if written:
        restricted = parse_lexicon(written)
        if restricted.value is None:
            return Result.failure(*restricted.diagnostics)
        within = restricted.unwrap()
    labels = facet_labels(
        [(doc.name, doc.date) for doc in ctx.corpus.docs],
        _str(params, "by", "year"),
        group_pattern=_str(params, "group-pattern", ""),
    )
    if labels.value is None:
        return Result.failure(*labels.diagnostics)
    field = Col.FORM if params.get("field", "lemma") == "form" else Col.LEMMA
    result = lexicon_series(ctx.table, groups.unwrap(), labels.unwrap(), field=field, within=within)
    if result.value is None:
        return Result.failure(*labels.diagnostics, *result.diagnostics)
    return Result.success({"lexicon_series.csv": result.unwrap()}, *labels.diagnostics, *result.diagnostics)


def _adapt_bert_topics(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    if ctx.table is None:
        return _missing_input("bert_topics", "table")
    model = params.get("model", "bert-base-uncased")
    if not isinstance(model, str):
        raise TypeError(f"param 'model' must be str, got {model!r}")
    result = bert_topics(
        ctx.table,
        n_topics=_int(params, "topics", 3),
        top_n=_int(params, "top-n", 5),
        seed=_int(params, "seed", 42),
        model=model,
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    topics, documents = result.unwrap()
    return Result.success(
        {"bert_topics.csv": topics, "bert_topic_docs.csv": documents},
        *result.diagnostics,
    )


def _geocode_places(provider: str, places: list[str]) -> Result[pd.DataFrame]:
    """Provider dispatch to the shared (Place, Lat, Lon, Status) contract.

    Mirrors ``tools.geocode.geocode_places`` — the CLI keeps its own copy so
    ``core/`` never imports the tools layer. An empty place list is an empty
    table, not a failure.
    """
    if not places:
        return Result.success(
            pd.DataFrame(columns=["Place", "Lat", "Lon", "Status"]),
            Diagnostic.info("GEOCODE_EMPTY_LIST", "no places to geocode"),
        )
    if provider == "offline":
        from core.gis.geocode import geocode_many

        return geocode_many(places)
    if provider == "nominatim":
        from core.gis.nominatim import NominatimGeocoder

        return NominatimGeocoder().geocode_many(places)
    from core.gis.online import GoogleGeocoder, api_key_from_env

    key = api_key_from_env()
    if key is None:
        return Result.failure(
            Diagnostic.error(
                "GEOCODE_NO_KEY",
                "GOOGLE_MAPS_KEY is not set",
                fix="set the GOOGLE_MAPS_KEY environment variable or use --provider nominatim",
            )
        )
    client = GoogleGeocoder(key)
    rows: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    for place in places:
        resolved = client.geocode(place)
        if resolved.ok:
            lat, lon = resolved.unwrap()
            rows.append({"Place": place, "Lat": lat, "Lon": lon, "Status": "OK"})
        else:
            rows.append({"Place": place, "Lat": 0.0, "Lon": 0.0, "Status": "UNKNOWN"})
            diags.extend(resolved.diagnostics)
    return Result.success(pd.DataFrame(rows, columns=["Place", "Lat", "Lon", "Status"]), *diags)


def _adapt_gender_annotator(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.gender_annotator import annotate_names, summarize_names

    if ctx.table is None:
        return _missing_input("gender_annotator", "table")
    annotated = annotate_names(
        ctx.table,
        dictionary=_str(params, "dictionary", "nltk"),
        names_dir=_lexicon_param(params, "names-dir"),
        source=_str(params, "source", "ner"),
    )
    if annotated.value is None:
        return Result.failure(*annotated.diagnostics)
    names_frame = annotated.unwrap()
    summarized = summarize_names(names_frame)
    if summarized.value is None:
        return Result.failure(*summarized.diagnostics)
    return Result.success(
        {"gender_names.csv": names_frame, "gender_summary.csv": summarized.unwrap()},
        *annotated.diagnostics,
        *summarized.diagnostics,
    )


def _adapt_date_annotator(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.date_annotator import annotate_corpus, summarize_dates

    if ctx.corpus is None:
        return _missing_input("date_annotator", "corpus")
    names = display_names(ctx.corpus.docs)
    docs = [(names[doc.doc_id], str(doc.doc_id), doc.text) for doc in ctx.corpus.docs]
    annotated = annotate_corpus(
        docs,
        min_year=_int(params, "min-year", 1000),
        max_year=_int(params, "max-year", 2100),
    )
    if annotated.value is None:
        return Result.failure(*annotated.diagnostics)
    dates_frame = annotated.unwrap()
    summarized = summarize_dates(dates_frame)
    if summarized.value is None:
        return Result.failure(*summarized.diagnostics)
    return Result.success(
        {"dates.csv": dates_frame, "dates_summary.csv": summarized.unwrap()},
        *annotated.diagnostics,
        *summarized.diagnostics,
    )


def _adapt_quote_annotator(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.quote_annotator import annotate_quotes, summarize_quotes

    if ctx.table is None:
        return _missing_input("quote_annotator", "table")
    annotated = annotate_quotes(ctx.table, min_length=_int(params, "min-length", 1))
    if annotated.value is None:
        return Result.failure(*annotated.diagnostics)
    quotes_frame = annotated.unwrap()
    summarized = summarize_quotes(quotes_frame)
    if summarized.value is None:
        return Result.failure(*summarized.diagnostics)
    return Result.success(
        {"quotes.csv": quotes_frame, "quotes_summary.csv": summarized.unwrap()},
        *annotated.diagnostics,
        *summarized.diagnostics,
    )


def _adapt_gender_guess(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.gender_guess import guess_gender, summarize_labels

    if ctx.table is None:
        return _missing_input("gender_guess", "table")
    result = guess_gender(ctx.table, mode=_str(params, "mode", "both"))
    if result.value is None:
        return Result.failure(*result.diagnostics)
    annotated = result.unwrap()
    summary = summarize_labels(annotated)
    if summary.value is None:
        return Result.failure(*result.diagnostics, *summary.diagnostics)
    return Result.success(
        {"gender_guess.csv": annotated, "gender_guess_summary.csv": summary.unwrap()},
        *result.diagnostics,
        *summary.diagnostics,
    )


def _adapt_verb_analysis(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.verb_analysis import analyze_verbs, summarize_verbs

    if ctx.table is None:
        return _missing_input("verb_analysis", "table")
    analysis = _str(params, "analysis", "all")
    result = analyze_verbs(ctx.table, analysis=analysis)
    if result.value is None:
        return Result.failure(*result.diagnostics)
    annotated = result.unwrap()
    summary = summarize_verbs(annotated, analysis=analysis)
    if summary.value is None:
        return Result.failure(*result.diagnostics, *summary.diagnostics)
    return Result.success(
        {"verbs.csv": annotated, "verb_summary.csv": summary.unwrap()},
        *result.diagnostics,
        *summary.diagnostics,
    )


def _adapt_ngram_viewer(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.ngram_viewer import frame_tokens, ngram_series

    if ctx.table is None or ctx.corpus is None:
        return _missing_input("ngram_viewer", "table" if ctx.table is None else "corpus")
    queries = [part.strip() for part in _str(params, "queries", "").split(",") if part.strip()]
    doc_tokens = frame_tokens(ctx.table)
    # Same key as _document_dates: str(doc.doc_id), the id every result
    # reports a document under. frame_tokens keys match it.
    dates = {str(doc.doc_id): (doc.date.year if doc.date is not None else None) for doc in ctx.corpus.docs}
    result = ngram_series(
        doc_tokens,
        dates,
        queries,
        case_sensitive=_bool(params, "case-sensitive", False),
        smooth=_int(params, "smooth", 1),
    )
    if result.value is None:
        return Result.failure(*result.diagnostics)
    series = result.unwrap()
    from core.viz.ngram_viewer import ngram_viewer_html

    chart = ngram_viewer_html(series)
    if chart.value is None:
        return Result.failure(*result.diagnostics, *chart.diagnostics)
    return Result.success(
        {"ngram_series.csv": series, "ngram_viewer.html": pd.DataFrame({"html": [chart.unwrap()]})},
        *result.diagnostics,
        *chart.diagnostics,
    )


def _adapt_sentiment_neural_bert(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.sentiment_neural import bert_sentences

    if ctx.table is None:
        return _missing_input("sentiment_neural_bert", "table")
    result = bert_sentences(ctx.table, model=_str(params, "model", "distilbert-sst2"))
    return _neural_sentiment_frames(result)


def _adapt_sentiment_neural_spacy(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.sentiment_neural import spacy_sentences

    if ctx.table is None:
        return _missing_input("sentiment_neural_spacy", "table")
    result = spacy_sentences(ctx.table, model=_str(params, "model", "en_core_web_sm"))
    return _neural_sentiment_frames(result)


def _adapt_sentiment_neural_stanza(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.sentiment_neural import stanza_sentences

    if ctx.table is None:
        return _missing_input("sentiment_neural_stanza", "table")
    result = stanza_sentences(ctx.table, language=_str(params, "language", "en"))
    return _neural_sentiment_frames(result)


def _neural_sentiment_frames(result: Result[pd.DataFrame]) -> Result[dict[str, pd.DataFrame]]:
    """The four neural backends share one table contract; fold it to frames."""
    from core.analysis.sentiment_neural import summarize_neural

    if result.value is None:
        return Result.failure(*result.diagnostics)
    sentences = result.unwrap()
    summary = summarize_neural(sentences)
    if summary.value is None:
        return Result.failure(*result.diagnostics, *summary.diagnostics)
    return Result.success(
        {"sentiment_sentences.csv": sentences, "sentiment_documents.csv": summary.unwrap()},
        *result.diagnostics,
        *summary.diagnostics,
    )


def _adapt_sentiment_neural_corenlp(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.sentiment_neural import corenlp_sentences

    if ctx.corpus is None:
        return _missing_input("sentiment_neural_corenlp", "corpus")
    names = display_names(ctx.corpus.docs)
    docs = [(str(doc.doc_id), names[doc.doc_id], doc.text) for doc in ctx.corpus.docs]
    result = corenlp_sentences(docs, server_url=_str(params, "server", "http://localhost:9000"))
    return _neural_sentiment_frames(result)


def _shape_frames(
    ctx: BatchContext,
    params: dict[str, object],
    tool: str,
) -> Result[tuple[ShapeMatrix, pd.DataFrame]]:
    """story_shape -> matrix, shared by the three data-reduction adapters."""
    from core.analysis.shape_reduction import build_shape_matrix
    from core.viz.shapes import story_shape

    if ctx.table is None:
        return Result.failure(Diagnostic.error("PROFILER_NO_TABLE", f"{tool} needs the parsed table"))
    shaped = story_shape(ctx.table)
    if shaped.value is None:
        return Result.failure(*shaped.diagnostics)
    matrix = build_shape_matrix(shaped.unwrap(), resample=_int(params, "resample", 32))
    if matrix.value is None:
        return Result.failure(*shaped.diagnostics, *matrix.diagnostics)
    from core.analysis.shape_reduction import matrix_frame

    return Result.success((matrix.unwrap(), matrix_frame(matrix.unwrap())), *shaped.diagnostics, *matrix.diagnostics)


def _named_shapes(
    result: Result[dict[str, pd.DataFrame]], table: pd.DataFrame | None
) -> Result[dict[str, pd.DataFrame]]:
    """Put the documents' own names back into the story-shape tables.

    The shape step keys documents as ``doc-<id>``, so every figure of these
    tables -- a dendrogram above all -- was labelled "doc-14 + doc-15" over a
    corpus of dated, named speeches. Only exact ``doc-<id>`` values in the
    name columns are replaced; ids and every number are untouched.
    """
    if result.value is None or table is None or Col.DOCUMENT.value not in table.columns:
        return result
    names = table.groupby(Col.DOCUMENT_ID.value)[Col.DOCUMENT.value].first()
    lookup = {f"doc-{doc_id}": str(name) for doc_id, name in names.items()}
    frames = {
        name: frame.assign(
            **{
                column: frame[column].map(lambda value: lookup.get(str(value), value))
                for column in ("Document", "Left", "Right")
                if column in frame.columns
            }
        )
        for name, frame in result.unwrap().items()
    }
    return Result.success(frames, *result.diagnostics)


def _adapt_shape_hc(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.shape_reduction import hierarchical_cluster
    from core.viz.shape_reduction import dendrogram_html

    prepared = _shape_frames(ctx, params, "shape_hc")
    if prepared.value is None:
        return Result.failure(*prepared.diagnostics)
    shape, matrix_table = prepared.unwrap()
    clustered = hierarchical_cluster(
        shape, method=_str(params, "method", "ward"), n_clusters=_int(params, "n-clusters", 2)
    )
    if clustered.value is None:
        return Result.failure(*prepared.diagnostics, *clustered.diagnostics)
    result = clustered.unwrap()
    frames: dict[str, pd.DataFrame] = {
        "shape_matrix.csv": matrix_table,
        "shape_hc_assignments.csv": result.assignments,
        "shape_hc_merges.csv": result.merges,
    }
    chart = dendrogram_html(result, title="Story shapes: hierarchical clustering")
    if chart.value is not None:
        frames["shape_hc_dendrogram.html"] = pd.DataFrame({"html": [chart.unwrap()]})
    return _named_shapes(
        Result.success(frames, *prepared.diagnostics, *clustered.diagnostics, *chart.diagnostics), ctx.table
    )


def _adapt_shape_svd(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.shape_reduction import svd_reduce
    from core.viz.shape_reduction import components_html

    prepared = _shape_frames(ctx, params, "shape_svd")
    if prepared.value is None:
        return Result.failure(*prepared.diagnostics)
    shape, matrix_table = prepared.unwrap()
    reduced = svd_reduce(shape, n_components=_int(params, "n-components", 2), seed=_int(params, "seed", 42))
    return _named_shapes(
        _reduction_frames(prepared.diagnostics, matrix_table, reduced, "svd", components_html), ctx.table
    )


def _adapt_shape_nmf(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.shape_reduction import nmf_reduce
    from core.viz.shape_reduction import components_html

    prepared = _shape_frames(ctx, params, "shape_nmf")
    if prepared.value is None:
        return Result.failure(*prepared.diagnostics)
    shape, matrix_table = prepared.unwrap()
    reduced = nmf_reduce(
        shape,
        n_components=_int(params, "n-components", 2),
        seed=_int(params, "seed", 42),
        max_iter=_int(params, "max-iter", 200),
    )
    return _named_shapes(
        _reduction_frames(prepared.diagnostics, matrix_table, reduced, "nmf", components_html), ctx.table
    )


def _reduction_frames(
    earlier: Sequence[Diagnostic],
    matrix_table: pd.DataFrame,
    reduced: Result[ReductionResult],
    method: str,
    components_html: Callable[..., Result[str]],
) -> Result[dict[str, pd.DataFrame]]:
    """Fold one reduction result into its five artifacts."""
    prior = tuple(earlier)
    if reduced.value is None:
        return Result.failure(*prior, *reduced.diagnostics)
    result = reduced.unwrap()
    frames: dict[str, pd.DataFrame] = {
        "shape_matrix.csv": matrix_table,
        f"shape_{method}_scores.csv": result.scores,
        f"shape_{method}_loadings.csv": result.loadings,
        f"shape_{method}_explained.csv": result.explained,
    }
    heading = "Story shapes: SVD components" if method == "svd" else "Story shapes: NMF parts"
    chart = components_html(result, title=heading)
    if chart.value is not None:
        frames[f"shape_{method}_components.html"] = pd.DataFrame({"html": [chart.unwrap()]})
    return Result.success(frames, *prior, *reduced.diagnostics, *chart.diagnostics)


def _adapt_geocode(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.location_extract import extract_locations

    if ctx.table is None:
        return _missing_input("geocode", "table")
    found = extract_locations(ctx.table, min_count=_int(params, "min-count", 1))
    if found.value is None:
        return Result.failure(*found.diagnostics)
    locations = found.unwrap()
    ranked = locations.groupby("Place", sort=False)["Mentions"].sum().sort_values(ascending=False)
    places = [str(place) for place in ranked.head(_int(params, "limit", 200)).index.tolist()]
    geocoded = _geocode_places(_str(params, "provider", "offline"), places)
    if geocoded.value is None:
        return Result.failure(*found.diagnostics, *geocoded.diagnostics)
    return Result.success(
        {"locations.csv": locations, "geocoded.csv": geocoded.unwrap()},
        *found.diagnostics,
        *geocoded.diagnostics,
    )


def _adapt_svo_map(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:  # noqa: PLR0911 -- one named failure per pipeline stage
    from core.analysis.clause_svo import extract_svo
    from core.analysis.location_extract import locations_by_document
    from core.analysis.svo_map import map_svo, svo_geo_summary

    if ctx.table is None:
        return _missing_input("svo_map", "table")
    svos = extract_svo(ctx.table)
    if svos.value is None:
        return Result.failure(*svos.diagnostics)
    svo_rows = pd.DataFrame(
        [
            {
                "Subject": row.subject,
                "Verb": row.verb,
                "Object": row.obj,
                "Sentence ID": row.sentence_id,
                "Document ID": str(row.document_id),
                "Document": row.document,
            }
            for row in svos.unwrap()
        ],
        columns=["Subject", "Verb", "Object", "Sentence ID", "Document ID", "Document"],
    )
    locs = locations_by_document(ctx.table)
    if locs.value is None:
        return Result.failure(*svos.diagnostics, *locs.diagnostics)
    locations = locs.unwrap()
    ranked = locations.groupby("Place", sort=False)["Mentions"].sum().sort_values(ascending=False)
    places = [str(place) for place in ranked.head(_int(params, "limit", 200)).index.tolist()]
    geocoded = _geocode_places(_str(params, "provider", "offline"), places)
    if geocoded.value is None:
        return Result.failure(*svos.diagnostics, *locs.diagnostics, *geocoded.diagnostics)
    mapped = map_svo(svo_rows, locations, geocoded=geocoded.unwrap())
    if mapped.value is None:
        return Result.failure(*svos.diagnostics, *locs.diagnostics, *geocoded.diagnostics, *mapped.diagnostics)
    table = mapped.unwrap()
    summary = svo_geo_summary(table)
    if summary.value is None:
        return Result.failure(*mapped.diagnostics, *summary.diagnostics)
    geo = summary.unwrap()
    from core.gis.mapping import kml
    from core.gis.pins import pinmap_html

    frames: dict[str, pd.DataFrame] = {
        "svo_locations.csv": locations,
        "svo_geocoded.csv": geo,
    }
    diags: list[Diagnostic] = [
        *svos.diagnostics,
        *locs.diagnostics,
        *geocoded.diagnostics,
        *mapped.diagnostics,
        *summary.diagnostics,
    ]
    plot = pinmap_html(geo, weight_col="SVO rows", title="Where narrative events happen")
    diags.extend(plot.diagnostics)
    if plot.value is not None:
        frames["svo_map.html"] = pd.DataFrame({"html": [plot.unwrap()]})
    document = kml(geo, name_col="Place")
    diags.extend(document.diagnostics)
    if document.value is not None:
        frames["svo_map.kml"] = pd.DataFrame({"html": [document.unwrap()]})
    return Result.success(frames, *diags)


def _tsne_with_counts(tsne: pd.DataFrame, vectors: pd.DataFrame) -> pd.DataFrame:
    """tsne.csv plus each word's corpus Count, from vectors.csv (additive).

    The coordinates come out in alphabetical order with no frequency, so a
    map of 7,478 words could only label its outliers -- usually rare words.
    With Count, the figure can show the words the corpus actually uses.
    """
    if "Count" in tsne.columns or not {"Word", "Count"} <= set(vectors.columns):
        return tsne
    counts = dict(zip(vectors["Word"].astype(str), vectors["Count"], strict=True))
    return tsne.assign(Count=tsne["Word"].astype(str).map(counts))


#: Groups a word map is coloured by, and how many of the most frequent
#: content words the groups are found among.
_MAP_GROUPS = 8
_MAP_GROUP_WORDS = 2000
#: A before and an after.
_PERIODS_NEEDED = 2


def _meaning_columns(
    frames: dict[str, pd.DataFrame], table: pd.DataFrame, field: str, space: word_meaning.Space
) -> None:
    """Word class on vectors.csv and a meaning group on tsne.csv, in place.

    The word class lets a figure keep to nouns (themes) or adjectives
    (qualities); the group turns the t-SNE cloud into named regions. Both are
    additive columns: every earlier reader of these tables still works.
    """
    from core.analysis.lda import STOPWORDS

    column = Col.FORM.value if field == "form" else Col.LEMMA.value
    classes = word_meaning.word_classes(table, column)
    vectors = frames["vectors.csv"]
    frames["vectors.csv"] = vectors.assign(
        **{word_meaning.WORD_CLASS: vectors["Word"].astype(str).str.lower().map(classes).fillna("")}
    )
    if "tsne.csv" not in frames:
        return
    among = word_meaning.candidates(space, exclude=STOPWORDS)[:_MAP_GROUP_WORDS]
    if len(among) < 2 * _MAP_GROUPS:
        return
    group_of: dict[str, str] = {}
    for group in word_meaning.meaning_groups(space, among, groups=_MAP_GROUPS):
        for member in group.members:
            group_of[space.words[member]] = group.name
    tsne = frames["tsne.csv"]
    frames["tsne.csv"] = tsne.assign(Group=tsne["Word"].astype(str).str.lower().map(group_of).fillna(""))


def _periods(corpus: Corpus | None) -> dict[str, str]:
    """Each dated document's period: its decade, or its year when all share one decade.

    Empty when fewer than two periods would result: change needs a before
    and an after.
    """
    dates = _document_dates(corpus)
    decades = {doc: f"{when.year // 10 * 10}s" for doc, when in dates.items()}
    if len(set(decades.values())) >= _PERIODS_NEEDED:
        return decades
    years = {doc: str(when.year) for doc, when in dates.items()}
    return years if len(set(years.values())) >= _PERIODS_NEEDED else {}


def _adapt_word2vec_bert(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.word2vec_bert import (
        neighbours as bert_neighbours,
        project_tsne as bert_tsne,
        train_bert,
        vectors as bert_vectors,
    )

    if ctx.table is None:
        return _missing_input("word2vec_bert", "table")
    field = _str(params, "field", "lemma")
    trained = train_bert(
        ctx.table,
        field=field,
        model=_str(params, "model", "bert-base-uncased"),
        min_count=_int(params, "min-count", 2),
        periods=_periods(ctx.corpus),
    )
    if trained.value is None:
        return Result.failure(*trained.diagnostics)
    model = trained.unwrap()
    vecs = bert_vectors(model)
    if vecs.value is None:
        return Result.failure(*trained.diagnostics, *vecs.diagnostics)
    frames: dict[str, pd.DataFrame] = {"vectors.csv": vecs.unwrap()}
    diags = [*trained.diagnostics, *vecs.diagnostics]
    projected = bert_tsne(model, seed=_int(params, "seed", 42))
    diags.extend(projected.diagnostics)
    if projected.value is not None and not projected.unwrap().empty:
        frames["tsne.csv"] = _tsne_with_counts(projected.unwrap(), frames["vectors.csv"])
        # Same scatter contract as word2vec_gensim so HW2 compares like with like.
        from core.viz.embeddings import tsne_html

        plot = tsne_html(projected.unwrap(), title="t-SNE BERT type vectors")
        diags.extend(plot.diagnostics)
        if plot.value is not None:
            frames["tsne.html"] = pd.DataFrame({"html": [plot.unwrap()]})
    if model.words:
        from core.analysis.lda import STOPWORDS

        space = word_meaning.space_of(model.words, model.counts, np.asarray(model.vectors, dtype=float))
        _meaning_columns(frames, ctx.table, field, space)
        over_time, change = word_meaning.meaning_over_time(space, model.by_period, exclude=STOPWORDS)
        if not change.empty:
            classes = dict(
                zip(frames["vectors.csv"]["Word"], frames["vectors.csv"][word_meaning.WORD_CLASS], strict=True)
            )
            frames["meaning_over_time.csv"] = over_time
            frames["meaning_change.csv"] = change.assign(
                **{word_meaning.WORD_CLASS: change["Word"].map(classes).fillna("")}
            )
        elif model.by_period:
            diags.append(
                Diagnostic.info(
                    "W2V_BERT_NO_CHANGE",
                    "No word was used often enough in two periods to compare its meaning over time; "
                    "a larger or longer corpus gives the change figures something to show.",
                )
            )
    query = params.get("query")
    if isinstance(query, str) and query.strip():
        # The neighbours of the space just built: embedding the corpus again
        # for them doubled the cost of every run with a query.
        found = bert_neighbours(model, query, top_n=_int(params, "top-n", 5))
        diags.extend(found.diagnostics)
        if found.value is None:
            return Result.failure(*diags)
        frames["neighbours.csv"] = found.unwrap()
    return Result.success(frames, *diags)


def _adapt_doc_embeddings(ctx: BatchContext, params: dict[str, object]) -> Result[dict[str, pd.DataFrame]]:
    from core.analysis.doc_embeddings import DEFAULT_MODEL, embed_corpus, semantic_search

    if ctx.table is None:
        return _missing_input("doc_embeddings", "table")
    model = _str(params, "model", DEFAULT_MODEL)
    embedded = embed_corpus(
        ctx.table,
        model=model,
        unit=_str(params, "unit", "document"),
        top_n=_int(params, "top-n", 5),
        seed=_int(params, "seed", 42),
    )
    if embedded.value is None:
        return Result.failure(*embedded.diagnostics)
    tables = embedded.unwrap()
    frames: dict[str, pd.DataFrame] = {"doc_vectors.csv": tables.vectors, "doc_map.csv": tables.map}
    if not tables.pairs.empty:
        frames["doc_pairs.csv"] = tables.pairs
        frames["doc_neighbours.csv"] = tables.neighbours
    diags = list(embedded.diagnostics)
    query = params.get("query")
    if isinstance(query, str) and query.strip():
        found = semantic_search(ctx.table, query, model=model)
        diags.extend(found.diagnostics)
        if found.value is None:
            return Result.failure(*diags)
        frames["search_results.csv"] = found.unwrap()
    return Result.success(frames, *diags)


ADAPTERS: dict[str, Adapter] = {
    "readability": _adapt_readability,
    "lexical_diversity": _adapt_lexical_diversity,
    "doc_similarity": _adapt_doc_similarity,
    "doc_duplicates": _adapt_doc_duplicates,
    "spellcheck": _adapt_spellcheck,
    "search": _adapt_search,
    "sentence_complexity": _adapt_sentence_complexity,
    "lda_gensim": _adapt_lda_gensim,
    "lda_stability": _adapt_lda_stability,
    "word2vec_gensim": _adapt_word_embeddings,
    "sentiment_vader_anew": _adapt_sentiment_vader_anew,
    "sentiment_swn_hedono": _adapt_sentiment_swn_hedono,
    "nrc": _adapt_nrc,
    "nominalization": _adapt_nominalization,
    "style": _adapt_style,
    "coreference": _adapt_coreference,
    "narrative": _adapt_narrative,
    "word_sense_induction": _adapt_contextual,
    "clause_svo": _adapt_clause_svo,
    "ner": _adapt_ner,
    "bert_extract": _adapt_bert_extract,
    "ngrams": _adapt_ngrams,
    "phrase_distribution": _adapt_phrase_distribution,
    "ngram_cooccurrence": _adapt_ngram_cooccurrence,
    "conll_wordlist": _adapt_conll_wordlist,
    "corpus_statistics": _adapt_corpus_statistics,
    "k_sentences": _adapt_k_sentences,
    "svo_compare": _adapt_svo_compare,
    "text_statistics": _adapt_text_statistics,
    "table_search": _adapt_table_search,
    "semantic": _adapt_semantic,
    "knowledge_graph": _adapt_knowledge_graph,
    "shapes": _adapt_shapes,
    "kwic": _adapt_kwic,
    "dispersion": _adapt_dispersion,
    "tfidf": _adapt_tfidf,
    "collocations": _adapt_collocations,
    "keyness": _adapt_keyness,
    "lexicon_series": _adapt_lexicon_series,
    "bert_topics": _adapt_bert_topics,
    "gender_annotator": _adapt_gender_annotator,
    "date_annotator": _adapt_date_annotator,
    "quote_annotator": _adapt_quote_annotator,
    "gender_guess": _adapt_gender_guess,
    "verb_analysis": _adapt_verb_analysis,
    "ngram_viewer": _adapt_ngram_viewer,
    "sentiment_neural_bert": _adapt_sentiment_neural_bert,
    "sentiment_neural_spacy": _adapt_sentiment_neural_spacy,
    "sentiment_neural_stanza": _adapt_sentiment_neural_stanza,
    "sentiment_neural_corenlp": _adapt_sentiment_neural_corenlp,
    "shape_hc": _adapt_shape_hc,
    "shape_svd": _adapt_shape_svd,
    "shape_nmf": _adapt_shape_nmf,
    "geocode": _adapt_geocode,
    "svo_map": _adapt_svo_map,
    "word2vec_bert": _adapt_word2vec_bert,
    "doc_embeddings": _adapt_doc_embeddings,
}

ADAPTER_NEEDS: dict[str, Needs] = {
    "readability": frozenset({"corpus"}),
    "lexical_diversity": frozenset({"corpus"}),
    "doc_similarity": frozenset({"corpus"}),
    "doc_duplicates": frozenset({"corpus"}),
    "spellcheck": frozenset({"corpus"}),
    "search": frozenset(),
    "sentence_complexity": frozenset({"table"}),
    "lda_gensim": frozenset({"table"}),
    "lda_stability": frozenset({"table"}),
    "word2vec_gensim": frozenset({"table"}),
    "sentiment_vader_anew": frozenset({"table"}),
    "sentiment_swn_hedono": frozenset({"table"}),
    "nrc": frozenset({"table"}),
    "nominalization": frozenset({"table"}),
    "style": frozenset({"table"}),
    "coreference": frozenset({"table"}),
    "narrative": frozenset({"table"}),
    "word_sense_induction": frozenset({"table"}),
    "clause_svo": frozenset({"table"}),
    "ner": frozenset({"table"}),
    "bert_extract": frozenset({"table"}),
    "ngrams": frozenset({"table"}),
    "phrase_distribution": frozenset({"table", "corpus"}),
    "ngram_cooccurrence": frozenset({"table"}),
    "conll_wordlist": frozenset({"table"}),
    "corpus_statistics": frozenset({"table"}),
    "k_sentences": frozenset({"table"}),
    "svo_compare": frozenset({"table"}),
    "text_statistics": frozenset({"table"}),
    "table_search": frozenset({"table"}),
    "semantic": frozenset({"table"}),
    "knowledge_graph": frozenset({"table"}),
    "shapes": frozenset({"table"}),
    "kwic": frozenset({"table"}),
    "dispersion": frozenset({"table"}),
    "tfidf": frozenset({"table"}),
    "collocations": frozenset({"table"}),
    "keyness": frozenset({"table"}),
    "lexicon_series": frozenset({"table", "corpus"}),
    "bert_topics": frozenset({"table"}),
    "gender_annotator": frozenset({"table"}),
    "date_annotator": frozenset({"corpus"}),
    "quote_annotator": frozenset({"table"}),
    "gender_guess": frozenset({"table"}),
    "verb_analysis": frozenset({"table"}),
    "ngram_viewer": frozenset({"table", "corpus"}),
    "sentiment_neural_bert": frozenset({"table"}),
    "sentiment_neural_spacy": frozenset({"table"}),
    "sentiment_neural_stanza": frozenset({"table"}),
    "sentiment_neural_corenlp": frozenset({"corpus"}),
    "shape_hc": frozenset({"table"}),
    "shape_svd": frozenset({"table"}),
    "shape_nmf": frozenset({"table"}),
    "geocode": frozenset({"table"}),
    "svo_map": frozenset({"table"}),
    "word2vec_bert": frozenset({"table"}),
}


def execute(
    plan: Plan,
    *,
    corpus: Corpus | None = None,
    table: pd.DataFrame | None = None,
    parse_diagnostics: tuple[Diagnostic, ...] = (),
    tokenizer: Tokenization | None = None,
) -> BatchResult:
    """Run every planned tool; a failure isolates to its own outcome.

    *tokenizer* belongs to whatever produced *table*, and callers that parsed
    the corpus themselves should pass it: see :class:`BatchContext`.
    """
    ctx = BatchContext(corpus=corpus, table=table, tokenizer=tokenizer)
    outcomes: list[ToolOutcome] = []
    for tool in plan.tools:
        adapter = ADAPTERS.get(tool.name)
        if adapter is None:
            outcomes.append(
                ToolOutcome(
                    name=tool.name,
                    ok=False,
                    diagnostics=(
                        Diagnostic.error("PROFILER_NO_ADAPTER", f"no batch adapter for {tool.name!r}", tool=tool.name),
                    ),
                )
            )
            continue
        missing = [need for need in sorted(ADAPTER_NEEDS.get(tool.name, frozenset())) if getattr(ctx, need) is None]
        if missing:
            need = missing[0]
            outcomes.append(
                ToolOutcome(
                    name=tool.name,
                    ok=False,
                    diagnostics=(
                        Diagnostic.error(
                            "PROFILER_NO_TABLE" if need == "table" else "PROFILER_NO_CORPUS",
                            f"{tool.name} needs {need}, which the batch did not provide",
                            tool=tool.name,
                        ),
                    ),
                )
            )
            continue
        start = time.perf_counter()
        try:
            result = adapter(ctx, tool.params)
        except Exception as exc:
            outcomes.append(
                ToolOutcome(
                    name=tool.name,
                    ok=False,
                    seconds=round(time.perf_counter() - start, 3),
                    diagnostics=(
                        Diagnostic.error(
                            "PROFILER_TOOL_CRASH", f"{tool.name} raised {type(exc).__name__}: {exc}", tool=tool.name
                        ),
                    ),
                )
            )
            continue
        seconds = round(time.perf_counter() - start, 3)
        if result.value is None:
            outcomes.append(
                ToolOutcome(name=tool.name, ok=False, seconds=seconds, diagnostics=tuple(result.diagnostics))
            )
        else:
            outcomes.append(
                ToolOutcome(
                    name=tool.name,
                    ok=result.ok,
                    frames=_with_dates(dict(result.value), corpus),
                    seconds=seconds,
                    diagnostics=tuple(result.diagnostics),
                )
            )
    parsed_names = {tool.name for tool in plan.tools if tool.requires_parse}
    parse_failed = any(diag.severity.value == "ERROR" for diag in parse_diagnostics)
    return BatchResult(
        outcomes=tuple(
            replace(outcome, ok=outcome.ok and not parse_failed, diagnostics=(*outcome.diagnostics, *parse_diagnostics))
            if outcome.name in parsed_names
            else outcome
            for outcome in outcomes
        )
    )
