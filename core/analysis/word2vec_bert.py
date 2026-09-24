"""Word2Vec via BERT — mean-pooled contextual type vectors (HW2).

The HW2 comparison asks which of the two Word2Vec approaches reads a corpus
better, so this module publishes the SAME table contracts as
:mod:`core.analysis.word_embeddings` (``Word/Count/Vector``,
``Word/Neighbor/Cosine``, ``Word/X/Y``) and the same trained-model surface.
The algorithms are deliberately different, and the difference is the finding:

* ``word2vec_gensim`` TRAINS on your corpus; a word has a vector only if the
  corpus teaches the model one.
* this module READS your corpus with a pretrained transformer. BERT splits
  words into subword pieces, each occurrence contributes the mean of its
  pieces' contextual embeddings in that sentence, and the word-type vector is
  the mean of its occurrence vectors. Nothing is trained on your corpus; the
  model downloads on first use.

Model loading and per-occurrence embedding are reused from
:mod:`core.analysis.contextual` (its ``TransformerBackend``); the type-level
aggregation lives here. ``transformers``/``torch`` are lazy imports; a missing
install fails as ``W2V_BERT_UNAVAILABLE`` with the fix command instead of
returning fake vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
import math

import numpy as np
import pandas as pd

from core.analysis.contextual import EmbeddingBackend, TransformerBackend, default_backend
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["TrainedBert", "distances", "project_tsne", "train_bert", "vectors"]

_TSNE_MIN_WORDS = 5
_FIX = "pip install transformers torch (the 'embeddings' extra); the model downloads on first use"
_VECTOR_COLUMNS = ["Word", "Count", "Vector"]
_NEIGHBOR_COLUMNS = ["Word", "Neighbor", "Cosine"]
_TSNE_COLUMNS = ["Word", "X", "Y"]


@dataclass(frozen=True, slots=True)
class TrainedBert:
    """One BERT type-vector space, as data (no model object retained).

    Mirrors ``core.analysis.word_embeddings.TrainedW2V``'s public surface so
    the two Word2Vec outputs stay interchangeable.
    """

    words: tuple[str, ...]
    counts: tuple[int, ...]
    vectors: tuple[tuple[float, ...], ...]
    vector_size: int
    model: str
    field: str
    min_count: int
    layers: int

    def _index(self, word: str) -> int:
        try:
            return self.words.index(word)
        except ValueError:
            raise KeyError(f"{word!r} not in vocabulary") from None

    def similarity(self, word_a: str, word_b: str) -> float:
        """Cosine similarity between two in-vocabulary words."""
        vec_a = np.array(self.vectors[self._index(word_a)], dtype=float)
        vec_b = np.array(self.vectors[self._index(word_b)], dtype=float)
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    def neighbors(self, query: str, top_n: int = 5) -> list[tuple[str, float]]:
        """``(word, cosine)`` pairs nearest to *query*, best first."""
        key = query.strip().lower()
        scored = sorted(
            ((word, self.similarity(key, word)) for word in self.words if word != key),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return scored[:top_n]


def _occurrences(frame: pd.DataFrame, column: str) -> list[tuple[str, str]]:
    """(type, sentence context) per usable token, in row order.

    Types are lowercased and alphabetic-only, matching
    ``word_embeddings._sentences_from_frame``, so the two Word2Vec
    vocabularies are comparable in the HW2 tables.
    """
    sent_text: dict[tuple[object, object], str] = {}
    for (doc_id, sent_id), group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        sent_text[(doc_id, sent_id)] = " ".join(str(form) for form in group[Col.FORM.value].tolist())
    pairs: list[tuple[str, str]] = []
    for _, row in frame.iterrows():
        raw = row[column]
        if raw is None:
            continue
        try:
            if bool(pd.isna(raw)):
                continue
        except (TypeError, ValueError):
            pass
        text = str(raw).strip().lower()
        if not text or not text.isalpha():
            continue
        key = (row[Col.DOCUMENT_ID.value], row[Col.SENTENCE_ID.value])
        pairs.append((text, sent_text.get(key, "")))
    return pairs


def _resolve_backend(backend: EmbeddingBackend | None, model: str) -> tuple[EmbeddingBackend | None, list[Diagnostic]]:
    if backend is not None:
        return backend, []
    resolved = default_backend(model)
    if resolved.value is None:
        return None, [
            Diagnostic.error(
                "W2V_BERT_UNAVAILABLE",
                "transformers is not installed; Word2Vec via BERT needs the optional 'embeddings' extra",
                fix=_FIX,
            )
        ]
    return resolved.unwrap(), list(resolved.diagnostics)


def _embed_last_hidden(backend: EmbeddingBackend, pairs: list[tuple[str, str]]) -> list[list[float]]:
    """Per-occurrence vectors: mean of the word's subword pieces in context."""
    return backend.embed([word for word, _ in pairs], [context for _, context in pairs])


def _embed_layer(backend: TransformerBackend, pairs: list[tuple[str, str]], layers: int) -> list[list[float]]:
    """Per-occurrence vectors from hidden state ``layers`` (same piece rule).

    Layer selection is not on the shared ``EmbeddingBackend`` protocol (it
    answers one question: embed these words in these contexts), so it reaches
    the transformer loader :mod:`core.analysis.contextual` already provides
    rather than re-implementing model plumbing here.
    """
    from core.analysis.contextual import _piece_text

    pipe = backend._load()
    tokenizer = pipe.tokenizer
    vectors: list[list[float]] = []
    for word, context in pairs:
        encoding = tokenizer(context, return_tensors="pt", truncation=True)
        word_ids = encoding.word_ids()
        pieces = [
            i
            for i, wid in enumerate(word_ids)
            if wid is not None and _piece_text(tokenizer, encoding, i).lower().lstrip("#") in word
        ]
        if not pieces:
            pieces = [i for i, wid in enumerate(word_ids) if wid is not None]
        import torch

        with torch.no_grad():
            outputs = pipe.model(
                **{k: v for k, v in encoding.items() if k != "overflow_to_sample_mapping"},
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[layers]
        selected = hidden[0][pieces].mean(dim=0).tolist()
        norm = math.sqrt(sum(value * value for value in selected)) or 1.0
        vectors.append([value / norm for value in selected])
    return vectors


def train_bert(
    frame: pd.DataFrame,
    *,
    field: str = "lemma",
    model: str = "bert-base-uncased",
    min_count: int = 2,
    layers: int = -1,
    backend: EmbeddingBackend | None = None,
) -> Result[TrainedBert]:
    """Mean-pooled BERT type vectors over the corpus's token occurrences.

    Each occurrence of a word type is embedded in its sentence context (the
    mean of its subword pieces), and the type vector is the (renormalized)
    mean of those occurrence vectors. ``layers=-1`` is the last hidden state;
    other indices read that hidden state through the transformer backend.
    """
    if field not in ("form", "lemma"):
        return Result.failure(Diagnostic.error("W2V_BERT_BAD_FIELD", f"field must be 'form' or 'lemma', got {field!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[TrainedBert](None, checked.diagnostics)
    column = Col.FORM.value if field == "form" else Col.LEMMA.value
    if column not in frame.columns:
        return Result.failure(Diagnostic.error("W2V_BERT_MISSING_COLUMN", f"missing {column!r}"))
    if min_count < 1:
        return Result.failure(Diagnostic.error("W2V_BERT_BAD_PARAM", f"min_count must be >=1, got {min_count}"))
    if frame.empty:
        # Nothing to embed is an empty space, not a broken run: the HW2 tables
        # then arrive empty with their columns, matching distances/vectors.
        return Result.success(
            TrainedBert(
                words=(),
                counts=(),
                vectors=(),
                vector_size=0,
                model=model,
                field=field,
                min_count=min_count,
                layers=layers,
            )
        )

    occurrences = _occurrences(frame, column)
    if not occurrences:
        return Result.failure(Diagnostic.error("W2V_BERT_NO_TOKENS", "no usable tokens; nothing to embed"))
    counts: dict[str, int] = {}
    for word, _ in occurrences:
        counts[word] = counts.get(word, 0) + 1
    kept = {word for word, count in counts.items() if count >= min_count}
    if not kept:
        return Result.failure(
            Diagnostic.error(
                "W2V_BERT_EMPTY_VOCAB", f"no vocabulary left at min_count={min_count}", min_count=min_count
            )
        )

    resolved, diags = _resolve_backend(backend, model)
    if resolved is None:
        return Result[TrainedBert](None, tuple(diags))
    if layers != -1 and not isinstance(resolved, TransformerBackend):
        return Result.failure(
            Diagnostic.error(
                "W2V_BERT_BAD_PARAM",
                f"layers={layers} needs the transformer backend; use layers=-1 with an injected backend",
                layers=layers,
            )
        )
    targets = [pair for pair in occurrences if pair[0] in kept]
    try:
        if layers == -1:
            occurrence_vectors = _embed_last_hidden(resolved, targets)
        else:
            occurrence_vectors = _embed_layer(cast(TransformerBackend, resolved), targets, layers)
    except RuntimeError as exc:
        return Result.failure(Diagnostic.error("W2V_BERT_UNAVAILABLE", str(exc), fix=_FIX))

    width = len(occurrence_vectors[0]) if occurrence_vectors else 0
    sums: dict[str, list[float]] = {}
    for (word, _), vec in zip(targets, occurrence_vectors, strict=True):
        accumulator = sums.setdefault(word, [0.0] * len(vec))
        for index, value in enumerate(vec):
            accumulator[index] += value

    words = sorted(kept)
    vectors: list[tuple[float, ...]] = []
    for word in words:
        mean = [total / counts[word] for total in sums[word]]
        norm = math.sqrt(sum(value * value for value in mean)) or 1.0
        vectors.append(tuple(value / norm for value in mean))
    return Result.success(
        TrainedBert(
            words=tuple(words),
            counts=tuple(counts[word] for word in words),
            vectors=tuple(vectors),
            vector_size=width,
            model=model,
            field=field,
            min_count=min_count,
            layers=layers,
        ),
        *diags,
    )


def vectors(trained: TrainedBert) -> Result[pd.DataFrame]:
    """(Word, Count, Vector) frame; Vector is comma-joined 4dp floats."""
    rows = [
        {"Word": word, "Count": count, "Vector": ",".join(f"{value:.4f}" for value in vec)}
        for word, count, vec in zip(trained.words, trained.counts, trained.vectors, strict=True)
    ]
    return Result.success(pd.DataFrame(rows, columns=_VECTOR_COLUMNS))


def _near_miss_fix(words: tuple[str, ...], query: str, field: str) -> str | None:
    """Name a trivial morphological near-miss, when one exists.

    Called only when the lowercased query is already out of vocabulary (a
    capitalisation miss never gets here: lookup is case-insensitive, matching
    ``word_embeddings.distances``). The check is deliberately naive — a
    trailing -s — because its job is to catch the form/lemma gap ("cats"
    against a lemma vocabulary), which is the miss HW2 readers actually make,
    not to do morphological analysis.
    """
    lowered = query.strip().lower()
    if not lowered:
        return None
    for candidate in (lowered[:-1], lowered + "s"):
        if candidate and candidate in words:
            return f"the vocabulary holds {field}s; try {candidate!r}"
    return None


def distances(
    frame: pd.DataFrame,
    query: str,
    *,
    field: str = "lemma",
    model: str = "bert-base-uncased",
    top_n: int = 5,
    min_count: int = 2,
    layers: int = -1,
    backend: EmbeddingBackend | None = None,
) -> Result[pd.DataFrame]:
    """Cosine nearest neighbours for *query* (Word, Neighbor, Cosine)."""
    if not query or not query.strip():
        return Result.failure(Diagnostic.error("W2V_BERT_BAD_QUERY", "query must be non-empty"))
    if top_n < 1:
        return Result.failure(Diagnostic.error("W2V_BERT_BAD_K", f"top_n must be >=1, got {top_n}"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_NEIGHBOR_COLUMNS))
    trained = train_bert(frame, field=field, model=model, min_count=min_count, layers=layers, backend=backend)
    if trained.value is None:
        return Result.failure(*trained.diagnostics)
    model_view = trained.unwrap()
    key = query.strip().lower()
    if key not in model_view.words:
        context: dict[str, object] = {"query": query}
        near = _near_miss_fix(model_view.words, query, field)
        if near:
            context["fix"] = near
        return Result.failure(Diagnostic.error("W2V_BERT_UNKNOWN_WORD", f"{query!r} not in vocabulary", **context))
    rows = [
        {"Word": key, "Neighbor": word, "Cosine": round(score, 4)} for word, score in model_view.neighbors(key, top_n)
    ]
    return Result.success(pd.DataFrame(rows, columns=_NEIGHBOR_COLUMNS))


def project_tsne(trained: TrainedBert, *, seed: int = 42) -> Result[pd.DataFrame]:
    """Seeded 2-D t-SNE coordinates (Word, X, Y) for viewer scatter plots."""
    if len(trained.words) < _TSNE_MIN_WORDS:
        return Result.success(
            pd.DataFrame(columns=_TSNE_COLUMNS),
            Diagnostic.warning(
                "W2V_TSNE_SKIPPED",
                f"t-SNE needs at least {_TSNE_MIN_WORDS} words, got {len(trained.words)}; empty result",
            ),
        )
    from sklearn.manifold import TSNE

    matrix = np.array([list(vec) for vec in trained.vectors], dtype=np.float64)
    perplexity = min(30.0, float(len(trained.words) - 1))
    coords = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="random").fit_transform(matrix)
    rows = [
        {"Word": word, "X": round(float(x), 4), "Y": round(float(y), 4)}
        for word, (x, y) in zip(trained.words, coords.tolist(), strict=True)
    ]
    return Result.success(pd.DataFrame(rows, columns=_TSNE_COLUMNS))
