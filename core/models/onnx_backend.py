"""Run a registered model through ONNX Runtime, with no torch or transformers.

Three backends, one per :data:`~core.models.registry.ModelKind`:

* :class:`OnnxTokenBackend` — a vector for a word in its sentence (BERT).
  It implements :class:`core.analysis.contextual.EmbeddingBackend`, plus
  ``embed_layer`` (any hidden state, for Word2Vec via BERT) and
  ``embed_texts`` (a sentence as the mean of its word pieces).
* :class:`OnnxSentenceBackend` — one vector per text, pooled the way the
  model was trained to be read (CLS for Granite, last token for Qwen).
* :class:`OnnxClassifier` — a label and its probability per text.

Every vector is L2-normalised, so a dot product is a cosine. Inputs are
batched by length (so a batch pads little) under a token budget, and
anything longer than the model's window is truncated and counted in
``truncated`` for the caller to report. Sessions are cached per model
folder: loading one costs seconds, running one costs milliseconds.

Word alignment: contexts are the parse's forms joined by spaces, so each
context is split back into those words and tokenized pre-split. The
tokenizer then says which word every piece came from, and a word's vector
is the mean of its own pieces — not of every piece that happens to be a
substring of it. The n-th request for the same word in the same sentence
reads the n-th occurrence.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import threading
from typing import Any

import numpy as np

from core.models.align import plan_requests
from core.models.locate import model_dir, status
from core.models.registry import ModelSpec, get_model
from core.result import Diagnostic, Result

__all__ = [
    "OnnxClassifier",
    "OnnxSentenceBackend",
    "OnnxTokenBackend",
    "forget",
    "open_model",
    "runtime_available",
    "runtime_missing",
]

#: Padded tokens per batch. Token models return every hidden state, so a
#: batch costs hidden_states x budget x dims floats (13 x 4096 x 768 = 160 MB).
_TOKEN_BUDGET = 4096
_MAX_BATCH = 64

_sessions: dict[str, Any] = {}
_tokenizers: dict[str, Any] = {}
_lock = threading.Lock()


def runtime_missing() -> list[str]:
    """Import names of the runtime pieces this installation lacks."""
    import importlib.util

    return [name for name in ("onnxruntime", "tokenizers") if importlib.util.find_spec(name) is None]


def runtime_available() -> bool:
    return not runtime_missing()


def _session(path: Path) -> Any:
    key = str(path)
    with _lock:
        cached = _sessions.get(key)
        if cached is not None:
            return cached
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        # ONNX Runtime's default intra-op pool is one thread per physical core.
        session = ort.InferenceSession(key, sess_options=options, providers=["CPUExecutionProvider"])
        _sessions[key] = session
        return session


def _tokenizer(path: Path, max_tokens: int) -> Any:
    key = f"{path}|{max_tokens}"
    with _lock:
        cached = _tokenizers.get(key)
        if cached is not None:
            return cached
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=max_tokens)
        _tokenizers[key] = tokenizer
        return tokenizer


def forget(folder: Path) -> None:
    """Drop cached sessions and tokenizers of one model folder (before deleting it)."""
    prefix = str(folder)
    with _lock:
        for cache in (_sessions, _tokenizers):
            for key in [key for key in cache if key.startswith(prefix)]:
                del cache[key]


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised: np.ndarray = matrix / norms
    return normalised


class _Runner:
    """Tokenize, batch by length, pad, run: what every backend shares."""

    def __init__(self, spec: ModelSpec, folder: Path) -> None:
        self.spec = spec
        self.folder = folder
        self._tokenizer = _tokenizer(folder / "tokenizer.json", spec.max_tokens)
        self._session = _session(folder / "model.onnx")
        self._inputs = {item.name for item in self._session.get_inputs()}
        #: Inputs cut at the model's window since this backend was made.
        self.truncated = 0

    @property
    def model_name(self) -> str:
        return self.spec.id

    def dimension(self) -> int:
        return self.spec.dims

    def encode(self, texts: Sequence[str] | Sequence[list[str]], *, pretokenized: bool = False) -> list[Any]:
        if not texts:
            return []
        encodings = self._tokenizer.encode_batch(list(texts), is_pretokenized=pretokenized)
        self.truncated += sum(1 for encoding in encodings if encoding.overflowing)
        return list(encodings)

    def batches(self, encodings: list[Any]) -> list[list[int]]:
        """Indices grouped so each batch pads to a similar length."""
        order = sorted(range(len(encodings)), key=lambda index: len(encodings[index].ids))
        groups: list[list[int]] = []
        current: list[int] = []
        for index in order:
            length = max(1, len(encodings[index].ids))
            if current and (length * (len(current) + 1) > _TOKEN_BUDGET or len(current) >= _MAX_BATCH):
                groups.append(current)
                current = []
            current.append(index)
        if current:
            groups.append(current)
        return groups

    def run(self, encodings: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        """The model's first output for a batch, and its attention mask."""
        width = max(1, *(len(encoding.ids) for encoding in encodings))
        ids = np.zeros((len(encodings), width), dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            length = len(encoding.ids)
            ids[row, :length] = encoding.ids
            mask[row, :length] = 1
        feeds: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        if "position_ids" in self._inputs:
            feeds["position_ids"] = np.broadcast_to(np.arange(width, dtype=np.int64), ids.shape).copy()
        output = self._session.run(None, {name: value for name, value in feeds.items() if name in self._inputs})[0]
        return np.asarray(output, dtype=np.float32), mask


class OnnxTokenBackend(_Runner):
    """Word-in-context vectors from a token model's hidden states."""

    def embed(self, words: Sequence[str], contexts: Sequence[str]) -> list[list[float]]:
        return self.embed_layer(words, contexts, -1)

    def embed_layer(self, words: Sequence[str], contexts: Sequence[str], layer: int) -> list[list[float]]:
        """One vector per (word, context) from hidden state *layer* (-1 = last)."""
        plan = plan_requests(words, contexts)
        if not words:
            return []
        if not -self.spec.hidden_states <= layer < self.spec.hidden_states:
            raise ValueError(f"layer {layer} is outside this model's {self.spec.hidden_states} hidden states")
        encodings = self.encode(plan.contexts, pretokenized=True)
        vectors = np.zeros((len(words), self.spec.dims), dtype=np.float32)
        for batch in self.batches(encodings):
            hidden, mask = self.run([encodings[slot] for slot in batch])
            states = hidden[layer]
            for row, slot in enumerate(batch):
                word_ids = encodings[slot].word_ids
                real = [index for index, word_id in enumerate(word_ids) if word_id is not None and mask[row, index]]
                for request in plan.by_slot.get(slot, []):
                    target = plan.requests[request][1]
                    pieces = [index for index in real if word_ids[index] == target] if target is not None else []
                    # A word the tokenizer cut off (or a lemma that never
                    # surfaces) reads the whole sentence, as before.
                    chosen = pieces or real or [0]
                    vectors[request] = states[row, chosen].mean(axis=0)
        rows: list[list[float]] = _normalise(vectors).tolist()
        return rows

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """A text as the mean of its word pieces in the last hidden state."""
        if not texts:
            return np.zeros((0, self.spec.dims), dtype=np.float32)
        encodings = self.encode(list(texts))
        out = np.zeros((len(texts), self.spec.dims), dtype=np.float32)
        for batch in self.batches(encodings):
            hidden, mask = self.run([encodings[index] for index in batch])
            states = hidden[-1]
            for row, index in enumerate(batch):
                real = [
                    position
                    for position, word_id in enumerate(encodings[index].word_ids)
                    if word_id is not None and mask[row, position]
                ]
                out[index] = states[row, real or [0]].mean(axis=0)
        return _normalise(out)


class OnnxSentenceBackend(_Runner):
    """One vector per text, pooled as the model was trained to be read."""

    def embed_texts(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.spec.dims), dtype=np.float32)
        prompt = self.spec.query_prompt if query else ""
        encodings = self.encode([prompt + text for text in texts])
        out = np.zeros((len(texts), self.spec.dims), dtype=np.float32)
        for batch in self.batches(encodings):
            hidden, mask = self.run([encodings[index] for index in batch])
            lengths = mask.sum(axis=1)
            for row, index in enumerate(batch):
                if self.spec.pooling == "cls":
                    out[index] = hidden[row, 0]
                elif self.spec.pooling == "last_token":
                    out[index] = hidden[row, max(0, int(lengths[row]) - 1)]
                else:
                    out[index] = hidden[row, : int(lengths[row])].mean(axis=0)
        return _normalise(out)


class OnnxClassifier(_Runner):
    """A label and its probability per text."""

    def classify(self, texts: Sequence[str]) -> list[tuple[str, float]]:
        if not texts:
            return []
        encodings = self.encode(list(texts))
        verdicts: list[tuple[str, float]] = [("", 0.0)] * len(texts)
        for batch in self.batches(encodings):
            logits, _ = self.run([encodings[index] for index in batch])
            shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = shifted / shifted.sum(axis=1, keepdims=True)
            for row, index in enumerate(batch):
                best = int(probabilities[row].argmax())
                verdicts[index] = (self.spec.labels[best], float(probabilities[row, best]))
        return verdicts

    def __call__(self, text: str) -> list[dict[str, Any]]:
        """The ``transformers`` sentiment pipeline's answer shape, for one text."""
        label, score = self.classify([text])[0]
        return [{"label": label, "score": score}]


_KIND_CLASS: dict[str, type[_Runner]] = {
    "token_embeddings": OnnxTokenBackend,
    "sentence_embeddings": OnnxSentenceBackend,
    "classifier": OnnxClassifier,
}


def not_installed(spec: ModelSpec) -> Diagnostic:
    """The one sentence a reader needs when a model is missing."""
    size = f" ({spec.size_mb:,} MB)" if spec.published else ""
    return Diagnostic.error(
        "MODEL_NOT_INSTALLED",
        f"{spec.display_name} isn't installed. Open Models to add it.",
        model=spec.id,
        fix=f"Open Models in the sidebar and download {spec.display_name}{size}.",
    )


def open_model(name: str, *kinds: str) -> Result[Any]:  # noqa: PLR0911 -- one return per reason a model cannot open
    """A ready backend for *name*, or the diagnostic that says why not.

    *kinds* limits which kinds of model are acceptable here (a sentiment
    tool cannot read an embedding model).
    """
    spec = get_model(name)
    if spec is None:
        return Result.failure(Diagnostic.error("MODEL_UNKNOWN", f"no model called {name!r}", model=name))
    if kinds and spec.kind not in kinds:
        return Result.failure(
            Diagnostic.error(
                "MODEL_WRONG_KIND",
                f"{spec.display_name} is a {spec.kind.replace('_', ' ')} model; this analysis needs "
                + " or ".join(kind.replace("_", " ") for kind in kinds),
                model=spec.id,
            )
        )
    missing = runtime_missing()
    if missing:
        return Result.failure(
            Diagnostic.error(
                "MODEL_RUNTIME_MISSING",
                "the model runtime is not installed (" + ", ".join(missing) + ")",
                fix='python -m pip install "nlp-suite-ng[models]"',
            )
        )
    folder = model_dir(spec)
    if folder is None:
        state = status(spec)
        if state == "corrupt":
            return Result.failure(
                Diagnostic.error(
                    "MODEL_CORRUPT",
                    f"{spec.display_name} is incomplete or damaged. Open Models to download it again.",
                    model=spec.id,
                )
            )
        return Result.failure(not_installed(spec))
    try:
        return Result.success(_KIND_CLASS[spec.kind](spec, folder))
    except Exception as exc:  # a broken file must be a diagnostic, not a traceback
        return Result.failure(
            Diagnostic.error("MODEL_LOAD_FAILED", f"could not load {spec.display_name}: {exc}", model=spec.id)
        )
