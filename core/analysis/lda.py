"""Gensim LDA topic modeling (FR-5.7) — real backend, seeded, reproducible.

Ports the modeling core of legacy ``topic_modeling_gensim_util.py``: seeded
``LdaModel`` over per-document lemma lists, per-topic top words, and a
per-document dominant-topic table. The TF-IDF round-robin stand-in that used
to live in ``topic_model.py`` is retired by this module (C4 rewires it).

Randomized algorithm, so verification is structural + recorded-reference
(see ``tests/fixtures/topics/ORACLE_NOTES.md``), not byte-pinned weights:
same seed twice is byte-identical in this environment, themes partition,
document distributions sum to 1.

Intentional differences from the legacy (defect-grade, recorded):

- ``stop_words.append(['from', 'subject', 're', 'edu', 'use'])`` appended one
  LIST element, so those words were never filtered — the port unions the
  five words as a real set with sklearn's English stopwords;
- topic ids are 0-based Gensim-native everywhere (the legacy used +1 in the
  keywords CSV but 0-based in the dominant CSV);
- no phrase detection in v1 (legacy ``Phrases`` min_count=5/threshold=100
  almost never fires on class-size corpora) — follow-up, not silent;
- no pyLDAvis HTML or browser launch; the viewer renders the CSV artifacts;
- MALLET stays FR-5.8; ``nouns_only`` keeps universal NOUN and PROPN (the
  legacy's spacy ``['NOUN']`` silently dropped proper nouns — people and
  places users model topics to find).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from core.analysis.postags import NOUN_TAGS
from core.conll.schema import Col
from core.result import Diagnostic, Result

__all__ = [
    "EXTRA_STOPWORDS",
    "FLOW_COLUMNS",
    "LdaResult",
    "STOPWORDS",
    "corpus_size_advice",
    "fit_lda",
    "intertopic_map",
    "kept_tokens",
    "relevance_terms",
    "segment_tokens_for",
    "tokens_from_frame",
]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gensim.models.ldamodel import LdaModel

    from core.analysis.topic_flow import SegmentPlan

# sklearn's 318 English stopwords plus the five words the legacy *meant* to
# add (it appended them as one list element, filtering nothing).
EXTRA_STOPWORDS: frozenset[str] = frozenset({"from", "subject", "re", "edu", "use"})
STOPWORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS

_FEW_DOCUMENTS = 50
_DOMINANT_KEYWORDS = 10


@dataclass(frozen=True, slots=True)
class LdaResult:
    """One fitted LDA model, as data (the Gensim object itself is not kept)."""

    topics: pd.DataFrame  # Topic (0-based), Word, Weight
    dominant: pd.DataFrame  # Document ID, Document, Dominant topic, Contribution, Topic keywords
    coherence: float | None  # c_v; None when it could not be computed
    perplexity: float  # log-perplexity (negative)
    seed: int
    n_topics: int
    #: Terms per topic at the requested lambda (HW2's "salient or relevant
    #: terms"): Topic, Word, Relevance, Saliency.
    relevance: pd.DataFrame | None = None
    #: Topic-topic distances laid out in 2-D (the Intertopic Distance Map's
    #: data): Topic, X, Y, Prevalence.
    intertopic: pd.DataFrame | None = None
    #: Paragraph-level topic flow (``fit_lda(..., segments=...)``): one row
    #: per (document, segment) -- Document ID, Document, Segment, Segments,
    #: Rule, Aligned, Tokens, Dominant topic, Contribution, Topic keywords,
    #: Start, End. ``None`` when no segments were given.
    flow: pd.DataFrame | None = None


def corpus_size_advice(n_docs: int) -> tuple[str, str]:
    """('error'|'advice'|'', message) for a corpus of *n_docs* documents.

    Port of the legacy pure helper: LDA compares word co-occurrence ACROSS
    documents, so 0/1 documents are errors; few documents are runnable but
    unstable (non-blocking advice, after the legacy TIPS wording).
    """
    if n_docs == 0:
        return (
            "error",
            "No documents to model.\n\nTopic modeling compares how words co-occur ACROSS documents.",
        )
    if n_docs == 1:
        return (
            "error",
            "A single document cannot be topic-modeled.\n\nThe comparison needs more than one document: "
            "with a single file there is nothing to compare. Split a long document into sections "
            "(chapters, articles, speeches) and use those as documents.",
        )
    if n_docs < _FEW_DOCUMENTS:
        return (
            "advice",
            f"The corpus has {n_docs} documents; this will run.\n\nWhat to watch for: topics are estimated "
            "per document as well as overall, so with few documents they can shift from one run to the next. "
            "Run it twice and check the same topics come back. Splitting long documents into sections "
            "usually steadies the topics.",
        )
    return ("", "")


def kept_tokens(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    nouns_only: bool = False,
    remove_stopwords: bool = True,
) -> pd.Series:
    """The frame's kept tokens (lower-cased, alphabetic, filtered), indexed
    like the frame.

    The single filter every consumer shares: :func:`tokens_from_frame` reads
    it per document, and the segment scorer (``fit_lda(segments=...)``) reads
    it per paragraph, so a segment's bag of words is built from *exactly* the
    tokens the model was trained on -- never a filter that drifted from the
    one that defined the vocabulary.

    The returned Series is in frame order, holding only the kept rows, with
    the original frame index -- so ``frame.loc[kept.index]`` aligns a kept
    token with its row's other columns (document, sentence, position).
    """
    empty = pd.Series(dtype=object, name=field.value)
    if frame.empty or field.value not in frame.columns:
        return empty
    values = frame[field.value]
    # notna() covers None and NaN alike, which the row loop checked separately.
    text = values[values.notna()].astype(str).str.strip().str.lower()
    # str.isalpha is False for "", so this also drops blank cells.
    keep = text.str.isalpha()
    if nouns_only and Col.POS.value in frame.columns:
        pos = frame.loc[text.index, Col.POS.value].astype(str).str.strip().str.upper()
        # The one noun rule (core/analysis/postags.py): Penn NN* or
        # Universal NOUN/PROPN, vectorised here for the same answer.
        keep &= pos.str.startswith("NN") | pos.isin(NOUN_TAGS)
    if remove_stopwords:
        keep &= ~text.isin(STOPWORDS)
    return text[keep]


def segment_tokens_for(
    frame: pd.DataFrame,
    documents: Iterable[str],
    *,
    field: Col = Col.LEMMA,
    nouns_only: bool = False,
    remove_stopwords: bool = True,
) -> dict[str, pd.Series]:
    """Each named document's kept tokens, indexed like *frame*.

    The segment scorer joins these to a segment plan's labels by row index.
    Taking the same filter arguments as the fit is the point: a flow scored
    on tokens the model was not trained on (nouns-only in the fit, every
    word in the segments) describes a different model.
    """
    kept = kept_tokens(frame, field=field, nouns_only=nouns_only, remove_stopwords=remove_stopwords)
    owner = frame.loc[kept.index, Col.DOCUMENT.value].astype(str)
    return {name: kept[owner == name] for name in documents}


def tokens_from_frame(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    nouns_only: bool = False,
    remove_stopwords: bool = True,
) -> dict[str, list[str]]:
    """Per-document token lists from a canonical CoNLL frame.

    Lower-cased alpha-only tokens in row order, keyed by Document name.
    ``nouns_only`` keeps nouns under either tagset -- Penn ``NN*`` or
    Universal ``NOUN``/``PROPN`` (the shared rule in
    ``core/analysis/postags.py``). Unknown/missing POS never drops a token
    when nouns_only is off.

    Two fixes, both proved against the previous row-by-row version on the
    87-address corpus (identical keys, order and tokens in every setting
    except nouns_only):

    * The noun filter used to accept Universal tags only. spaCy's parse here
      carries Penn tags (``NN``, ``NNS``), so ``nouns_only`` silently returned
      no tokens at all, and the topic model then failed with "no documents to
      model" -- pointing the reader at their corpus, not at the tagset. It now
      keeps 136,515 noun tokens across all 87 documents.
    * It iterated ``frame.iterrows()``, 10-16 s on the 600k-row corpus and
      paid on every topic-model run. Vectorised, it takes about 0.25 s.
    """
    tokens: dict[str, list[str]] = {}
    if frame.empty or field.value not in frame.columns or Col.DOCUMENT.value not in frame.columns:
        return tokens
    kept = kept_tokens(frame, field=field, nouns_only=nouns_only, remove_stopwords=remove_stopwords)
    documents = frame.loc[kept.index, Col.DOCUMENT.value].astype(str)
    # Insertion order is first appearance, exactly as the row loop built it.
    for document, word in zip(documents.tolist(), kept.tolist(), strict=True):
        tokens.setdefault(document, []).append(word)
    return tokens


def relevance_terms(
    topic_word: object,
    vocabulary: Sequence[str],
    doc_tokens: Mapping[str, Sequence[str]],
    *,
    lam: float = 0.6,
    top_n: int = 10,
    topic_tokens: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Terms per topic at relevance lambda, after Chuang et al./pyLDAvis.

    ``relevance = lam * log p(word|topic) + (1 - lam) * log p(word)``. At
    lam=1 the ranking is the topic's own probability (the "salient" reading);
    lowering lam lifts terms distinctive of the topic even if rare corpus-wide.
    HW2 asks what varying lam *does*, so lam is a parameter and the table
    always carries both readings (Relevance and Saliency).

    Pure data in, data out: the two probabilities come from the fitted model
    and the corpus the caller already holds.

    Two counts ride along for drawing, because Relevance and Saliency are
    log-scale scores that cannot be bar lengths: relevance is a sum of logs of
    probabilities, so it is always negative and the best-ranked term has the
    *smallest* magnitude. pyLDAvis draws counts instead and uses relevance only
    to order the rows, and so does the suite:

    * ``Corpus frequency`` -- how many times the word occurs in the corpus.
    * ``Topic frequency`` -- the model's *estimate* of how many of those
      occurrences belong to this topic: ``p(word|topic) * N_t``, where
      ``topic_tokens[t]`` is ``N_t``, the token-weighted mass of the topic
      (sum over documents of topic share times document length). It is an
      expectation under the model, not an observed count. Without
      ``topic_tokens`` it cannot be estimated and is left missing (NaN)
      rather than guessed.

    The gap between the two is the finding: a word whose corpus count dwarfs
    its topic count is common everywhere, not a feature of this topic.
    """
    columns = ["Topic", "Word", "Relevance", "Saliency", "Corpus frequency", "Topic frequency"]
    if not 0.0 <= lam <= 1.0:
        return pd.DataFrame(columns=columns)
    counts: dict[str, int] = {}
    total = 0
    for tokens in doc_tokens.values():
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
            total += 1
    total = total or 1
    rows: list[dict[str, object]] = []
    phi = np.asarray(topic_word, dtype=float)
    masses = None if topic_tokens is None else np.asarray(topic_tokens, dtype=float)
    # Chuang et al. 2010's term salience (pyLDAvis's "saliency" per term):
    # s(w) = sum_t p(t|w) * log(p(t|w) / p(t)) -- a KL divergence of the
    # word's topic distribution from the topic marginal, so it is never
    # negative and grows as the word's topic preferences diverge from how
    # common each topic is. pyLDAvis scales the plot by p(w) on top; that is
    # a display weighting, so it is NOT folded in here -- the column stays
    # the pure divergence, in bits, comparable across words. The previous
    # engine's p(w|t)*(log p(w|t) - E log p(w|t)) was inverted and negative
    # on real data; the panel used to order by magnitude to survive both.
    # See TestRelevanceLambda for the definitional checks.
    prevalence = np.clip(phi.sum(axis=1), 1e-12, None)  # p(t) over the topic margin
    prevalence = prevalence / prevalence.sum()
    # p(t|w) = phi[t, w] / p(w); 0 where the word never occurs.
    ptw = np.divide(
        phi, np.clip(phi.sum(axis=0), 1e-12, None)[None, :], out=np.zeros_like(phi), where=phi.sum(axis=0) > 0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(np.clip(ptw, 1e-12, None) / prevalence[:, None])
    saliency = np.sum(ptw * log_ratio, axis=0)
    for topic_id in range(phi.shape[0]):
        log_phi = np.log(np.clip(phi[topic_id], 1e-12, None))
        freq = np.array([counts.get(str(word), 0) for word in vocabulary], dtype=float)
        log_pw = np.log(np.clip(freq / total, 1e-12, None))
        relevance = lam * log_phi + (1.0 - lam) * log_pw
        order = np.argsort(-relevance)[:top_n]
        for index in order:
            rows.append(
                {
                    "Topic": topic_id,
                    "Word": str(vocabulary[index]),
                    "Relevance": round(float(relevance[index]), 4),
                    "Saliency": round(float(saliency[index]), 4),
                    "Corpus frequency": int(freq[index]),
                    "Topic frequency": (
                        round(float(phi[topic_id, index]) * float(masses[topic_id]), 2)
                        if masses is not None
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def intertopic_map(
    topic_word: object,
    prevalence: Sequence[float],
    *,
    seed: int = 100,
) -> pd.DataFrame:
    """The Intertopic Distance Map's data: topics laid out by their distance.

    pyLDAvis places topics by multidimensional scaling of the Jensen-Shannon
    distance between their word distributions (circle area is prevalence).
    That placement is computed here natively -- the numbers are the graded
    thing, and a chart of them is one Plotly call away -- so the suite grows
    no dependency on pyLDAvis's bundled JS. Same formula; recorded in
    tests/fixtures/topics/ORACLE_NOTES.md.
    """
    import warnings

    from scipy.spatial.distance import jensenshannon
    from sklearn.manifold import MDS

    phi = np.asarray(topic_word, dtype=float)
    k = phi.shape[0]
    if k < 2:
        return pd.DataFrame(columns=["Topic", "X", "Y", "Prevalence"])
    distances = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            distances[i, j] = distances[j, i] = float(jensenshannon(phi[i], phi[j]))
    masses = np.asarray(prevalence, dtype=float)
    scale = float(masses.max()) or 1.0
    if not distances.max():
        # Every topic identical: one point, not a coin toss from MDS's
        # random init on a zero matrix. "Same words, same place."
        return pd.DataFrame(
            {
                "Topic": list(range(k)),
                "X": [0.0] * k,
                "Y": [0.0] * k,
                "Prevalence": [round(float(m / scale), 4) for m in masses],
            }
        )
    # sklearn is renaming MDS's arguments across releases and the placement is
    # what matters, not which keyword carried it. Pinned down by test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        layout = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=seed,
            n_init=4,
            normalized_stress="auto",
        ).fit_transform(distances)
    return pd.DataFrame(
        {
            "Topic": list(range(k)),
            "X": [round(float(x), 4) for x in layout[:, 0]],
            "Y": [round(float(y), 4) for y in layout[:, 1]],
            "Prevalence": [round(float(m / scale), 4) for m in masses],
        }
    )


#: The columns of ``LdaResult.flow``, exactly as ``panels_lda_flow`` reads
#: them (its ``requires`` plus the span columns a reader needs to open the
#: paragraph as a passage). One tuple, so a second copy cannot drift.
FLOW_COLUMNS: tuple[str, ...] = (
    "Document ID",
    "Document",
    "Segment",
    "Segments",
    "Rule",
    "Aligned",
    "Tokens",
    "Dominant topic",
    "Contribution",
    "Topic keywords",
    "Start",
    "End",
)

_DOCUMENT_ID = "Document ID"
_DOCUMENT = "Document"
_SEGMENT = "Segment"
_SEGMENTS = "Segments"
_RULE = "Rule"
_ALIGNED = "Aligned"
_TOKENS = "Tokens"
_DOMINANT = "Dominant topic"
_CONTRIBUTION = "Contribution"
_TOPIC_KEYWORDS = "Topic keywords"
_START = "Start"
_END = "End"


def _score_segments(
    model: LdaModel,
    plan: SegmentPlan,
    segment_tokens: Mapping[str, pd.Series],
    *,
    names: Sequence[str],
    keywords_by_topic: Mapping[int, str],
) -> pd.DataFrame:
    """Score each segment's bag of words with the fitted model, as rows.

    A segment with no kept tokens has no basis for a topic: its
    ``Dominant topic`` is left blank rather than filled with the prior,
    which would colour a salutation by a subject the model never saw. The
    vocabulary lookup is the model's own dictionary -- out-of-vocabulary
    words simply do not reach the bow, which is what "the model cannot read
    this word" means, not an error.
    """
    doc_ids = {name: number for number, name in enumerate(names, start=1)}
    dictionary = model.id2word
    rows: list[dict[str, object]] = []
    for document, labels in plan.labels.items():
        if document not in doc_ids:
            continue
        spans = plan.spans.get(document, [])
        rules = plan.rules.get(document, "")
        counts: dict[int, int] = {}
        bags: dict[int, list[str]] = {}
        # Paired by row index, never by position: the kept tokens are a
        # filtered subset of the document's rows, and ``labels`` covers every
        # row, so zipping the two lists put each word in the paragraph of
        # whichever row happened to share its ordinal.
        tokens = segment_tokens.get(document)
        if tokens is None:
            tokens = pd.Series(dtype=object)
        for token, segment in zip(tokens.tolist(), labels.reindex(tokens.index).tolist(), strict=True):
            if pd.isna(segment):
                continue
            counts[int(segment)] = counts.get(int(segment), 0) + 1
            bags.setdefault(int(segment), []).append(token)
        total = max(labels.tolist(), default=0)
        for number in range(1, max(total, 1) + 1):
            tokens_here = bags.get(number, ())
            if tokens_here:
                dist = sorted(
                    model.get_document_topics(dictionary.doc2bow(list(tokens_here)), minimum_probability=0.0),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                best, share = int(dist[0][0]), round(float(dist[0][1]), 4)
                keywords = keywords_by_topic.get(best, "")
            else:
                # A blank dominant topic, not the prior: nothing was left to score.
                best, share, keywords = -1, float("nan"), ""
            # plan.spans is a per-segment list indexed from 0 (segment 1 = 0).
            span = spans[number - 1] if 0 < number <= len(spans) else None
            start, end = span if span is not None else (None, None)
            rows.append(
                {
                    _DOCUMENT_ID: doc_ids[document],
                    _DOCUMENT: document,
                    _SEGMENT: number,
                    _SEGMENTS: total,
                    _RULE: rules,
                    _ALIGNED: document not in plan.unaligned,
                    _TOKENS: counts.get(number, 0),
                    _DOMINANT: best if best >= 0 else None,
                    _CONTRIBUTION: share if best >= 0 else None,
                    _TOPIC_KEYWORDS: keywords,
                    _START: start,
                    _END: end,
                }
            )
    return pd.DataFrame(rows, columns=FLOW_COLUMNS)


def fit_lda(
    doc_tokens: Mapping[str, Sequence[str]],
    *,
    n_topics: int = 3,
    top_n: int = 5,
    seed: int = 100,
    passes: int = 10,
    remove_stopwords: bool = True,
    coherence: bool = True,
    relevance_lambda: float = 0.6,
    segments: SegmentPlan | None = None,
    segment_tokens: Mapping[str, pd.Series] | None = None,
) -> Result[LdaResult]:
    """Fit seeded Gensim LDA over per-document token lists.

    Guards fire before any gensim import, so they hold on machines without
    the optional extra. ``seed`` is recorded in the result and the envelope.

    ``segments`` (from ``core.analysis.topic_flow.plan_segments``) asks for
    paragraph-level scoring while the model is still in hand: each segment's
    bag of words is scored with ``get_document_topics`` and written to
    ``LdaResult.flow``. ``segment_tokens`` supplies each document's kept
    tokens as a Series indexed like the parsed frame (build it with
    :func:`segment_tokens_for`), so each token meets its own row's
    segment label; this is because the
    training filter, not a caller's re-derivation, must define what a
    segment contains. Giving ``segments`` without ``segment_tokens`` is a
    caller bug and fails with ``TOPIC_SEGMENTS_INCOMPLETE`` rather than
    returning a flow of empty rows.
    """
    if n_topics < 1 or top_n < 1:
        return Result.failure(
            Diagnostic.error("TOPIC_BAD_K", f"n_topics and top_n must be >=1, got {n_topics}, {top_n}")
        )
    level, message = corpus_size_advice(len(doc_tokens))
    if level == "error":
        code = "TOPIC_NO_DOCUMENTS" if len(doc_tokens) == 0 else "TOPIC_SINGLE_DOCUMENT"
        return Result.failure(Diagnostic.error(code, message))
    diags: list[Diagnostic] = []
    if level == "advice":
        diags.append(Diagnostic.warning("TOPIC_FEW_DOCUMENTS", message))

    names = list(doc_tokens.keys())
    texts: list[list[str]] = []
    for name in names:
        kept = [tok for tok in doc_tokens[name] if not (remove_stopwords and tok in STOPWORDS)]
        texts.append(kept)
    if not any(texts):
        return Result.failure(
            Diagnostic.error(
                "TOPIC_EMPTY_VOCABULARY",
                "nothing left to model after stopword filtering; all documents are empty or stopwords-only",
            )
        )

    try:
        from gensim.corpora import Dictionary
        from gensim.models.ldamodel import LdaModel
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "TOPIC_BACKEND_MISSING",
                "Gensim is not installed; topic modeling needs the optional 'topics' extra",
                fix="python -m pip install nlp-suite-ng[topics]",
            )
        )

    dictionary = Dictionary(texts)
    if len(dictionary) == 0:
        return Result.failure(
            Diagnostic.error("TOPIC_EMPTY_VOCABULARY", "the Gensim dictionary is empty; nothing to model")
        )
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=seed,
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha="auto",
        per_word_topics=True,
    )

    topic_rows: list[dict[str, object]] = []
    keywords_by_topic: dict[int, str] = {}
    for topic_id in range(n_topics):
        shown = model.show_topic(topic_id, topn=max(top_n, _DOMINANT_KEYWORDS))
        keywords_by_topic[topic_id] = ", ".join(word for word, _ in shown[:_DOMINANT_KEYWORDS])
        for word, weight in shown[:top_n]:
            topic_rows.append({"Topic": topic_id, "Word": str(word), "Weight": round(float(weight), 4)})
    topics = pd.DataFrame(topic_rows, columns=["Topic", "Word", "Weight"])

    dominant_rows: list[dict[str, object]] = []
    for doc_id, (name, bow) in enumerate(zip(names, corpus, strict=True), start=1):
        dist = sorted(model.get_document_topics(bow, minimum_probability=0.0), key=lambda pair: pair[1], reverse=True)
        best_id, best_share = int(dist[0][0]), round(float(dist[0][1]), 4)
        dominant_rows.append(
            {
                "Document ID": doc_id,
                "Document": name,
                "Dominant topic": best_id,
                "Contribution": best_share,
                "Topic keywords": keywords_by_topic[best_id],
            }
        )
    dominant = pd.DataFrame(
        dominant_rows, columns=["Document ID", "Document", "Dominant topic", "Contribution", "Topic keywords"]
    )

    flow: pd.DataFrame | None = None
    if segments is not None:
        if segment_tokens is None:
            return Result.failure(
                Diagnostic.error(
                    "TOPIC_SEGMENTS_INCOMPLETE",
                    "segments were given without segment_tokens; the segment bags must come from the same "
                    "token filter the model was trained on, not from a caller's re-derivation.",
                )
            )
        flow = _score_segments(model, segments, segment_tokens, names=names, keywords_by_topic=keywords_by_topic)

    coherence_value: float | None = None
    if coherence:
        try:
            from gensim.models import CoherenceModel

            coherence_value = float(
                CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v").get_coherence()
            )
        except Exception as exc:
            diags.append(Diagnostic.warning("TOPIC_COHERENCE_FAILED", f"c_v coherence could not be computed: {exc}"))

    # The two Gensim-GUI views HW2 grades, computed from the fitted model
    # while it is still in hand: terms at the requested lambda, and the
    # Intertopic Distance Map's placement. Both are data; the drawing is
    # somebody else's job.
    vocabulary = [dictionary[i] for i in range(len(dictionary))]
    phi = model.get_topics()
    doc_topic_mass = np.zeros(n_topics, dtype=float)
    # Token-weighted too: a topic's expected token count, the N_t pyLDAvis
    # scales p(word|topic) by. A long address contributes more tokens than a
    # short one, and the share-weighted mass above does not know that.
    topic_token_mass = np.zeros(n_topics, dtype=float)
    for bow in corpus:
        length = float(sum(count for _, count in bow))
        for topic_id, share in model.get_document_topics(bow, minimum_probability=0.0):
            doc_topic_mass[int(topic_id)] += float(share)
            topic_token_mass[int(topic_id)] += float(share) * length
    views: tuple[Diagnostic, ...] = tuple(diags)
    try:
        by_name = dict(zip(names, texts, strict=True))
        relevance = relevance_terms(
            phi, vocabulary, by_name, lam=relevance_lambda, top_n=max(top_n, 10), topic_tokens=topic_token_mass.tolist()
        )
        intertopic = intertopic_map(phi, doc_topic_mass.tolist(), seed=seed)
    except Exception as exc:  # views are a convenience; the fit still counts
        views = (*tuple(diags), Diagnostic.warning("TOPIC_VIEWS_FAILED", f"topic views could not be computed: {exc}"))
        relevance = None
        intertopic = None

    return Result.success(
        LdaResult(
            topics=topics,
            dominant=dominant,
            coherence=coherence_value,
            perplexity=float(model.log_perplexity(corpus)),
            seed=seed,
            n_topics=n_topics,
            relevance=relevance,
            intertopic=intertopic,
            flow=flow,
        ),
        *views,
    )
