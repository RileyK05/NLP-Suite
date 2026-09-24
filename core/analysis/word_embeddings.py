"""Word embeddings — seeded Gensim Word2Vec (FR-5.5).

Trains real skip-gram/CBOW vectors over CoNLL sentences, with cosine
neighbours, seeded sklearn t-SNE coordinates, and gensim-free save/load
(npz + json sidecar: a saved model serves distances without the optional
extra installed). The old MD5 hash vectors are gone.

Intentional differences from the legacy (defect-grade, recorded):

- the legacy passed NO seed (a fresh model every run) — ``seed`` is
  required plumbing here (default 42, ``workers=1`` fixed) and recorded in
  every envelope; same seed twice is byte-identical;
- training sentences come from the CoNLL table ((Document ID, Sentence ID)
  groups) instead of a second stanza run over the raw text;
- stopword filtering defaults OFF (function words carry distributional
  signal; removal is a toggle);
- skip-gram default (``sg=1``): better rare-word vectors on class-size
  corpora;
- t-SNE is seeded sklearn output as data, not an interactive plot call.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "TrainedW2V",
    "distances",
    "load_model",
    "project_tsne",
    "save_model",
    "train",
    "vectors",
]

_TSNE_MIN_WORDS = 5


@dataclass(frozen=True, slots=True)
class TrainedW2V:
    """One trained embedding model, as data (no gensim object retained)."""

    words: tuple[str, ...]
    counts: tuple[int, ...]
    vectors: tuple[tuple[float, ...], ...]
    vector_size: int
    window: int
    min_count: int
    sg: int
    seed: int
    epochs: int

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


def _sentences_from_frame(frame: pd.DataFrame, field: Col, remove_stopwords: bool) -> list[list[str]]:
    """CoNLL rows -> per-sentence token lists in row order."""
    from core.analysis.lda import STOPWORDS

    sentences: dict[tuple[str, str], list[str]] = {}
    for _, row in frame.iterrows():
        raw = row[field.value]
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
        if remove_stopwords and text in STOPWORDS:
            continue
        key = (str(row[Col.DOCUMENT_ID.value]), str(row["Sentence ID"]))
        sentences.setdefault(key, []).append(text)
    return [toks for toks in sentences.values() if toks]


def train(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    min_count: int = 2,
    vector_size: int = 100,
    window: int = 5,
    seed: int = 42,
    epochs: int = 20,
    sg: int = 1,
    remove_stopwords: bool = False,
) -> Result[TrainedW2V]:
    """Train seeded Word2Vec over the frame's sentences."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("W2V_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[TrainedW2V](None, checked.diagnostics)
    if field.value not in frame.columns:
        return Result.failure(Diagnostic.error("W2V_MISSING_COLUMN", f"missing {field.value!r}"))
    for name, value in (("min_count", min_count), ("vector_size", vector_size), ("window", window), ("epochs", epochs)):
        if value < 1:
            return Result.failure(Diagnostic.error("W2V_BAD_PARAM", f"{name} must be >=1, got {value}"))
    if sg not in (0, 1):
        return Result.failure(Diagnostic.error("W2V_BAD_PARAM", f"sg must be 0 (CBOW) or 1 (skip-gram), got {sg}"))
    if frame.empty:
        return Result.failure(Diagnostic.error("W2V_NO_SENTENCES", "empty input frame; nothing to train on"))
    sentences = _sentences_from_frame(frame, field, remove_stopwords)
    if not sentences:
        return Result.failure(Diagnostic.error("W2V_NO_SENTENCES", "no usable sentences; nothing to train on"))

    try:
        from gensim.models import Word2Vec
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "W2V_BACKEND_MISSING",
                "Gensim is not installed; embeddings need the optional 'topics' extra",
                fix="python -m pip install nlp-suite-ng[topics]",
            )
        )
    try:
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            seed=seed,
            workers=1,
            epochs=epochs,
        )
    except RuntimeError as exc:
        # Gensim raises when min_count filters every word ("must first build
        # vocabulary"): that is the empty-vocabulary contract outcome.
        return Result.failure(
            Diagnostic.error(
                "W2V_EMPTY_VOCAB",
                f"no vocabulary left to train on (min_count={min_count}): {exc}",
                min_count=min_count,
            )
        )
    from collections import Counter

    counts = Counter(tok for sent in sentences for tok in sent)
    words = sorted(model.wv.key_to_index)
    return Result.success(
        TrainedW2V(
            words=tuple(words),
            counts=tuple(int(counts[w]) for w in words),
            vectors=tuple(tuple(float(x) for x in model.wv[w]) for w in words),
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            seed=seed,
            epochs=epochs,
        )
    )


def vectors(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    min_count: int = 2,
    vector_size: int = 100,
    window: int = 5,
    seed: int = 42,
    epochs: int = 20,
    sg: int = 1,
    remove_stopwords: bool = False,
) -> Result[pd.DataFrame]:
    """(Word, Count, Vector) frame; empty input yields an empty frame."""
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Word", "Count", "Vector"]))
    trained = train(
        frame,
        field=field,
        min_count=min_count,
        vector_size=vector_size,
        window=window,
        seed=seed,
        epochs=epochs,
        sg=sg,
        remove_stopwords=remove_stopwords,
    )
    if trained.value is None:
        if any(d.code == "W2V_EMPTY_VOCAB" for d in trained.diagnostics):
            return Result.success(
                pd.DataFrame(columns=["Word", "Count", "Vector"]),
                Diagnostic.warning("W2V_EMPTY_VOCAB", "min_count filtered the whole vocabulary; empty result"),
            )
        return Result.failure(*trained.diagnostics)
    model = trained.unwrap()
    rows = [
        {"Word": word, "Count": count, "Vector": ",".join(f"{v:.4f}" for v in vec)}
        for word, count, vec in zip(model.words, model.counts, model.vectors, strict=True)
    ]
    return Result.success(pd.DataFrame(rows, columns=["Word", "Count", "Vector"]))


def distances(
    frame: pd.DataFrame,
    query: str,
    *,
    field: Col = Col.LEMMA,
    top_n: int = 5,
    min_count: int = 2,
    vector_size: int = 100,
    window: int = 5,
    seed: int = 42,
    epochs: int = 20,
    sg: int = 1,
    remove_stopwords: bool = False,
) -> Result[pd.DataFrame]:
    """Cosine nearest neighbours for *query* (Word, Neighbor, Cosine)."""
    if not query or not query.strip():
        return Result.failure(Diagnostic.error("W2V_BAD_QUERY", "query must be non-empty"))
    if top_n < 1:
        return Result.failure(Diagnostic.error("W2V_BAD_K", f"top_n must be >=1, got {top_n}"))
    trained = train(
        frame,
        field=field,
        min_count=min_count,
        vector_size=vector_size,
        window=window,
        seed=seed,
        epochs=epochs,
        sg=sg,
        remove_stopwords=remove_stopwords,
    )
    if trained.value is None:
        return Result.failure(*trained.diagnostics)
    model = trained.unwrap()
    key = query.strip().lower()
    if key not in model.words:
        return Result.failure(Diagnostic.error("W2V_UNKNOWN_WORD", f"{query!r} not in vocabulary", query=query))
    scored = sorted(
        ((word, model.similarity(key, word)) for word in model.words if word != key),
        key=lambda pair: pair[1],
        reverse=True,
    )
    out = pd.DataFrame(
        [{"Word": key, "Neighbor": word, "Cosine": round(score, 4)} for word, score in scored[:top_n]],
        columns=["Word", "Neighbor", "Cosine"],
    )
    return Result.success(out)


def save_model(trained: TrainedW2V, path: str | Path) -> Result[Path]:
    """Persist a trained model (npz + json sidecar values, no gensim)."""
    target = Path(path)
    try:
        np.savez(
            target,
            words=np.array(trained.words),
            counts=np.array(trained.counts, dtype=np.int64),
            matrix=np.array([list(vec) for vec in trained.vectors], dtype=np.float64),
            meta=np.array(
                [
                    json.dumps(
                        {
                            "vector_size": trained.vector_size,
                            "window": trained.window,
                            "min_count": trained.min_count,
                            "sg": trained.sg,
                            "seed": trained.seed,
                            "epochs": trained.epochs,
                        }
                    )
                ]
            ),
        )
    except OSError as exc:
        return Result.failure(Diagnostic.error("W2V_MODEL_UNWRITABLE", f"cannot write model to {target}: {exc}"))
    return Result.success(target)


def load_model(path: str | Path) -> Result[TrainedW2V]:
    """Reload a saved model; works without gensim installed."""
    target = Path(path)
    try:
        with np.load(target, allow_pickle=False) as data:
            words = tuple(str(w) for w in data["words"].tolist())
            counts = tuple(int(c) for c in data["counts"].tolist())
            matrix = tuple(tuple(float(x) for x in row) for row in data["matrix"].tolist())
            meta = json.loads(str(data["meta"].tolist()[0]))
    except (OSError, ValueError, KeyError) as exc:
        return Result.failure(Diagnostic.error("W2V_MODEL_UNREADABLE", f"cannot read model from {target}: {exc}"))
    try:
        params = {key: int(meta[key]) for key in ("vector_size", "window", "min_count", "sg", "seed", "epochs")}
    except (KeyError, TypeError, ValueError) as exc:
        return Result.failure(Diagnostic.error("W2V_MODEL_UNREADABLE", f"model metadata is corrupt: {exc}"))
    if len(words) != len(counts) or len(words) != len(matrix):
        return Result.failure(Diagnostic.error("W2V_MODEL_UNREADABLE", "model arrays have mismatched lengths"))
    return Result.success(TrainedW2V(words=words, counts=counts, vectors=matrix, **params))


def project_tsne(trained: TrainedW2V, *, seed: int = 42) -> Result[pd.DataFrame]:
    """Seeded 2-D t-SNE coordinates (Word, X, Y) for viewer scatter plots."""
    if len(trained.words) < _TSNE_MIN_WORDS:
        return Result.success(
            pd.DataFrame(columns=["Word", "X", "Y"]),
            Diagnostic.warning(
                "W2V_TSNE_SKIPPED",
                f"t-SNE needs at least {_TSNE_MIN_WORDS} words, got {len(trained.words)}; empty result",
            ),
        )
    matrix = np.array([list(vec) for vec in trained.vectors], dtype=np.float64)
    perplexity = min(30.0, float(len(trained.words) - 1))
    coords = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="random").fit_transform(matrix)
    rows = [
        {"Word": word, "X": round(float(x), 4), "Y": round(float(y), 4)}
        for word, (x, y) in zip(trained.words, coords.tolist(), strict=True)
    ]
    return Result.success(pd.DataFrame(rows, columns=["Word", "X", "Y"]))
