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
:mod:`core.analysis.contextual` (``default_backend``: the installed ONNX
model, else ``transformers`` in a source checkout); the type-level
aggregation lives here. A missing model fails as ``W2V_BERT_UNAVAILABLE``
telling the reader to add it from the Models page, instead of returning
fake vectors.

Each occurrence is located by its surface form even when the vocabulary is
lemmas: "was" is found in its sentence and counted towards "be".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from core.analysis.contextual import EmbeddingBackend, default_backend, truncation_note
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["TrainedBert", "distances", "neighbours", "project_tsne", "train_bert", "vectors"]

_TSNE_MIN_WORDS = 5
_FIX = "Open Models in the sidebar and add BERT base."
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
    #: ``(word, period, uses, vector)``: the word's mean occurrence vector
    #: within one period, for words used at least twice there. Empty unless
    #: ``train_bert`` was given the documents' periods.
    by_period: tuple[tuple[str, str, int, tuple[float, ...]], ...] = ()

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
        index = self._index(key)
        matrix = np.array(self.vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        cosines = (matrix @ matrix[index]) / (norms * norms[index])
        # Stable sort on -cosine keeps the alphabetical order of ties, as before.
        order = [i for i in np.argsort(-cosines, kind="stable").tolist() if i != index]
        return [(self.words[i], float(cosines[i])) for i in order[:top_n]]


def _occurrences(frame: pd.DataFrame, column: str) -> list[tuple[str, str, str, str]]:
    """(type, surface form, sentence context, document id) per usable token, in row order.

    Types are lowercased and alphabetic-only, matching
    ``word_embeddings._sentences_from_frame``, so the two Word2Vec
    vocabularies are comparable in the HW2 tables.
    """
    sent_text: dict[tuple[object, object], str] = {}
    for (doc_id, sent_id), group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        sent_text[(doc_id, sent_id)] = " ".join(str(form) for form in group[Col.FORM.value].tolist())
    pairs: list[tuple[str, str, str, str]] = []
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
        pairs.append((text, str(row[Col.FORM.value]).lower(), sent_text.get(key, ""), str(key[0])))
    return pairs


def _resolve_backend(backend: EmbeddingBackend | None, model: str) -> tuple[EmbeddingBackend | None, list[Diagnostic]]:
    if backend is not None:
        return backend, []
    resolved = default_backend(model)
    if resolved.value is None:
        reason = resolved.diagnostics[0].message if resolved.diagnostics else f"{model} is not installed"
        return None, [Diagnostic.error("W2V_BERT_UNAVAILABLE", reason, model=model, fix=_FIX)]
    return resolved.unwrap(), list(resolved.diagnostics)


def _embed(backend: EmbeddingBackend, pairs: list[tuple[str, str, str, str]], layers: int) -> list[list[float]]:
    """Per-occurrence vectors: the mean of the form's pieces in hidden state *layers*.

    Layer selection is not on the shared ``EmbeddingBackend`` protocol (it
    answers one question: embed these words in these contexts); the real
    backends (ONNX and transformers) both offer ``embed_layer``.
    """
    forms = [pair[1] for pair in pairs]
    contexts = [pair[2] for pair in pairs]
    if layers == -1:
        return backend.embed(forms, contexts)
    layered = backend.embed_layer  # type: ignore[attr-defined]  # checked by the caller
    return list(layered(forms, contexts, layers))


def train_bert(
    frame: pd.DataFrame,
    *,
    field: str = "lemma",
    model: str = "bert-base-uncased",
    min_count: int = 2,
    layers: int = -1,
    backend: EmbeddingBackend | None = None,
    periods: Mapping[str, str] | None = None,
) -> Result[TrainedBert]:
    """Mean-pooled BERT type vectors over the corpus's token occurrences.

    Each occurrence of a word type is embedded in its sentence context (the
    mean of its subword pieces), and the type vector is the (renormalized)
    mean of those occurrence vectors. ``layers=-1`` is the last hidden state;
    other indices read that hidden state through the transformer backend.

    *periods* maps document ids to a period label ("1940s"). Given, the same
    occurrence vectors are also averaged per word within each period
    (``by_period``), which costs no further embedding: a contextual model
    reads every period into one space, so a word's vector in the 1940s and
    in the 2000s can be compared directly, with no alignment step.
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
    for occurrence in occurrences:
        counts[occurrence[0]] = counts.get(occurrence[0], 0) + 1
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
    if layers != -1 and not callable(getattr(resolved, "embed_layer", None)):
        return Result.failure(
            Diagnostic.error(
                "W2V_BERT_BAD_PARAM",
                f"layers={layers} needs a backend that reads hidden layers; use layers=-1 with an injected backend",
                layers=layers,
            )
        )
    targets = [pair for pair in occurrences if pair[0] in kept]
    try:
        occurrence_vectors = _embed(resolved, targets, layers)
    except (RuntimeError, ValueError) as exc:
        return Result.failure(Diagnostic.error("W2V_BERT_UNAVAILABLE", str(exc), fix=_FIX))
    diags.extend(truncation_note(resolved))

    width = len(occurrence_vectors[0]) if occurrence_vectors else 0
    matrix = np.asarray(occurrence_vectors, dtype=float).reshape(len(targets), width)
    order = [pair[0] for pair in targets]
    sums: dict[str, list[float]] = {}
    for word, total in _sums(order, matrix).items():
        sums[word] = total.tolist()
    by_period = _period_means(targets, matrix, periods) if periods else ()

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
            by_period=by_period,
        ),
        *diags,
    )


def _sums(keys: list[str], matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Row sums of *matrix* per key, keys in first-seen order."""
    codes, uniques = pd.factorize(pd.Series(keys, dtype=object), sort=False)
    totals = np.zeros((len(uniques), matrix.shape[1]))
    np.add.at(totals, codes, matrix)
    return {str(key): totals[i] for i, key in enumerate(uniques)}


def _period_means(
    targets: list[tuple[str, str, str, str]], matrix: np.ndarray, periods: Mapping[str, str]
) -> tuple[tuple[str, str, int, tuple[float, ...]], ...]:
    """Each word's mean occurrence vector per period, where it was used twice or more.

    Occurrences in undated documents belong to no period and are left out.
    """
    rows = [i for i, pair in enumerate(targets) if pair[3] in periods]
    if not rows:
        return ()
    keys = [(targets[i][0], periods[targets[i][3]]) for i in rows]
    codes, uniques = pd.factorize(pd.Series(keys, dtype=object), sort=False)
    totals = np.zeros((len(uniques), matrix.shape[1]))
    np.add.at(totals, codes, matrix[rows])
    uses = np.bincount(codes, minlength=len(uniques))
    out: list[tuple[str, str, int, tuple[float, ...]]] = []
    for position, (word, period) in enumerate(uniques):
        n = int(uses[position])
        if n < 2:
            continue
        mean = totals[position] / n
        norm = float(np.linalg.norm(mean)) or 1.0
        out.append((str(word), str(period), n, tuple(float(v) for v in mean / norm)))
    out.sort(key=lambda row: (row[0], row[1]))
    return tuple(out)


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
    return neighbours(trained.unwrap(), query, top_n=top_n)


def neighbours(model_view: TrainedBert, query: str, *, top_n: int = 5) -> Result[pd.DataFrame]:
    """Cosine nearest neighbours for *query* in an already-built vector space."""
    if not query or not query.strip():
        return Result.failure(Diagnostic.error("W2V_BERT_BAD_QUERY", "query must be non-empty"))
    if top_n < 1:
        return Result.failure(Diagnostic.error("W2V_BERT_BAD_K", f"top_n must be >=1, got {top_n}"))
    field = model_view.field
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
