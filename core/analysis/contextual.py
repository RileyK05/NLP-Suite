"""Contextual embeddings + word-sense induction (FR-5.6).

``contextual_vectors`` embeds each token in its sentence through an
injectable backend; ``wsi_senses`` clusters each lemma's uses in two and
keeps the split only when the two groups separate clearly (so a word used
one way stays one sense). It replaced a median split on the vectors' first
component, which split every word in half whatever its uses.

The production backend is the registered model run through ONNX Runtime
(:mod:`core.models.onnx_backend`; BERT base ships with the app). A source
checkout without the model falls back to ``transformers``
(``TransformerBackend``); without either, the run fails with the one
thing to do — add the model from the Models page — instead of returning
fake vectors. The test suite never downloads a model.

A word is always located by its surface form in the sentence, whichever
field names it: a lemma ("be") seldom occurs verbatim in its sentence
("was"), and reading the whole sentence instead would give every such
lemma the same vector.

The old MD5 hash vectors were production fraud (deterministic,
untrained, and labeled "contextual"); they now live only in
``tests/test_contextual.py`` as the offline fake behind the same
protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import sys
from typing import Any, Protocol

import numpy as np
import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "EmbeddingBackend",
    "TransformerBackend",
    "contextual_vectors",
    "default_backend",
    "embed_sentences",
    "sense_uses",
    "text_backend",
    "truncation_note",
    "wsi_senses",
]

_DEFAULT_MODEL = "bert-base-uncased"

_VECTOR_COLUMNS = ["Word", "Lemma", "Sentence ID", "Document ID", "Vector"]
_WSI_COLUMNS = ["Lemma", "Sense", "Separation", "Sentence ID", "Document ID"]


class EmbeddingBackend(Protocol):
    """One question the analyses need: embed these words in these sentences."""

    def dimension(self) -> int:
        """Embedding width."""
        ...

    def embed(self, words: Sequence[str], contexts: Sequence[str]) -> list[list[float]]:
        """One vector per (word, context) pair, in order."""
        ...


class TransformerBackend:
    """Development fallback: the same model through PyTorch ``transformers``.

    The installed app runs models through ONNX Runtime
    (:mod:`core.models.onnx_backend`); this class is what a source checkout
    with ``transformers`` uses when a model has not been installed, and what
    ``scripts/export_models.py`` checks the ONNX files against. It reads
    exactly the same pieces (:mod:`core.models.align`), so the two agree.

    The model loads on the first ``embed`` (CPU, no gradients); the
    package import is lazy so ``import core.analysis.contextual`` never
    requires ``transformers`` or ``torch``.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, revision: str | None = None, source: str = "") -> None:
        self._model_name = model_name
        self._source = source or model_name
        self._revision = revision
        self._loaded: tuple[Any, Any] | None = None
        #: Inputs cut at the model's window since this backend was made.
        self.truncated = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> tuple[Any, Any]:
        if self._loaded is not None:
            return self._loaded
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed; contextual embeddings need the optional 'embeddings' extra"
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._source, revision=self._revision)
            # Inference mode (no dropout), spelled train(False) so the layering
            # scan for dynamic-evaluation calls has nothing to flag.
            model = AutoModel.from_pretrained(self._source, revision=self._revision, dtype="float32")
            model.train(False)
        except Exception as exc:
            raise RuntimeError(
                f"cannot load embedding model {self._model_name!r} "
                "(no cached copy and no network, or a broken download)"
            ) from exc
        self._loaded = (tokenizer, model)
        return self._loaded

    def dimension(self) -> int:
        return int(self._load()[1].config.hidden_size)

    def _hidden(self, encoding: Any) -> Any:
        import torch

        _, model = self._load()
        with torch.no_grad():
            return model(**encoding, output_hidden_states=True).hidden_states

    def embed(self, words: Sequence[str], contexts: Sequence[str]) -> list[list[float]]:
        return self.embed_layer(words, contexts, -1)

    def embed_layer(self, words: Sequence[str], contexts: Sequence[str], layer: int) -> list[list[float]]:
        """One vector per (word, context) from hidden state *layer* (-1 = last)."""
        from core.models.align import plan_requests

        plan = plan_requests(words, contexts)
        tokenizer, _ = self._load()
        vectors: list[list[float]] = [[] for _ in words]
        for slot, split in enumerate(plan.contexts):
            encoding = tokenizer(split, is_split_into_words=True, truncation=True, return_tensors="pt")
            if encoding["input_ids"].shape[1] >= tokenizer.model_max_length:
                self.truncated += 1
            states = self._hidden(encoding)[layer][0]
            word_ids = encoding.word_ids()
            real = [index for index, word_id in enumerate(word_ids) if word_id is not None]
            for request in plan.by_slot.get(slot, []):
                target = plan.requests[request][1]
                pieces = [index for index in real if word_ids[index] == target] if target is not None else []
                vectors[request] = _unit(states[pieces or real or [0]].mean(dim=0).tolist())
        return vectors

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """A text as the mean of its word pieces in the last hidden state."""
        tokenizer, _ = self._load()
        out: list[list[float]] = []
        for text in texts:
            encoding = tokenizer(text, truncation=True, return_tensors="pt")
            states = self._hidden(encoding)[-1][0]
            real = [index for index, word_id in enumerate(encoding.word_ids()) if word_id is not None]
            out.append(_unit(states[real or [0]].mean(dim=0).tolist()))
        return out


def _unit(values: list[float]) -> list[float]:
    import math

    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _transformers_present() -> bool:
    """Is the development fallback importable? (A test may block it in sys.modules.)"""
    import importlib.util

    if "transformers" in sys.modules:
        return sys.modules["transformers"] is not None
    try:
        return importlib.util.find_spec("transformers") is not None
    except (ImportError, ValueError):
        return False


def default_backend(model_name: str = _DEFAULT_MODEL) -> Result[EmbeddingBackend]:
    """The installed ONNX model, else (in a source checkout) ``transformers``.

    A model the registry knows and has installed runs through ONNX Runtime.
    Otherwise a development install with ``transformers`` reads the same
    model from the Hugging Face cache; without either, the reader is told
    the one thing to do: add the model from the Models page.
    """
    from core.models.onnx_backend import open_model
    from core.models.registry import get_model

    opened = open_model(model_name, "token_embeddings")
    if opened.value is not None:
        return Result.success(opened.value)
    spec = get_model(model_name)
    wrong_kind = any(diag.code == "MODEL_WRONG_KIND" for diag in opened.diagnostics)
    if not wrong_kind and _transformers_present():
        if spec is not None:
            return Result.success(TransformerBackend(spec.id, spec.revision, spec.source))
        return Result.success(TransformerBackend(model_name))
    return Result.failure(*opened.diagnostics)


def text_backend(model_name: str = _DEFAULT_MODEL) -> Result[Any]:
    """A backend that embeds whole sentences: a sentence model or a token model.

    Summaries and BERT topics read one vector per sentence. A sentence
    model (Granite, Qwen) is trained for exactly that; a token model
    (BERT) answers with the mean of the sentence's word pieces.
    """
    from core.models.onnx_backend import open_model
    from core.models.registry import get_model

    opened = open_model(model_name, "token_embeddings", "sentence_embeddings")
    if opened.value is not None:
        return Result.success(opened.value)
    spec = get_model(model_name)
    wrong_kind = any(diag.code == "MODEL_WRONG_KIND" for diag in opened.diagnostics)
    token_model = spec is None or spec.kind == "token_embeddings"
    if not wrong_kind and token_model and _transformers_present():
        if spec is not None:
            return Result.success(TransformerBackend(spec.id, spec.revision, spec.source))
        return Result.success(TransformerBackend(model_name))
    return Result.failure(*opened.diagnostics)


def embed_sentences(backend: Any, texts: Sequence[str]) -> list[list[float]]:
    """One unit vector per text, from whichever sentence route the backend has.

    A backend without ``embed_texts`` (a test double) is asked for each
    text as a word in itself, which is what the older protocol meant.
    """
    if not texts:
        return []
    embed_texts = getattr(backend, "embed_texts", None)
    if callable(embed_texts):
        rows = embed_texts(list(texts))
        return [list(map(float, row)) for row in rows]
    return list(backend.embed(list(texts), list(texts)))


def truncation_note(backend: object) -> list[Diagnostic]:
    """A warning when the backend cut sentences at the model's window."""
    count = int(getattr(backend, "truncated", 0) or 0)
    if not count:
        return []
    return [
        Diagnostic.warning(
            "MODEL_INPUT_TRUNCATED",
            f"{count} sentence(s) were longer than the model reads at once and were cut at its window",
            count=count,
        )
    ]


def _resolve(backend: EmbeddingBackend | None, model: str | None) -> tuple[EmbeddingBackend | None, list[Diagnostic]]:
    if backend is not None:
        return backend, []
    resolved = default_backend(model or _DEFAULT_MODEL)
    if resolved.value is None:
        return None, list(resolved.diagnostics)
    return resolved.unwrap(), list(resolved.diagnostics)


def contextual_vectors(
    frame: pd.DataFrame,
    *,
    field: Col = Col.FORM,
    backend: EmbeddingBackend | None = None,
    model: str | None = None,
) -> Result[pd.DataFrame]:
    """One embedding per token: word vector in its sentence context."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("CTX_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_VECTOR_COLUMNS))

    resolved, diags = _resolve(backend, model)
    if resolved is None:
        return Result.failure(*diags)

    sent_text: dict[tuple[object, object], str] = {}
    for (did, sid), group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        sent_text[(did, sid)] = " ".join(str(x) for x in group[Col.FORM.value].tolist())

    forms: list[str] = []
    metas: list[tuple[str, str, object, object]] = []  # word, lemma, sid, did
    for _, row in frame.iterrows():
        word = str(row[field.value])
        if not word.strip() or word.lower() in ("nan", "none"):
            continue
        did = row[Col.DOCUMENT_ID.value]
        sid = row[Col.SENTENCE_ID.value]
        forms.append(str(row[Col.FORM.value]).lower())
        metas.append((word, str(row[Col.LEMMA.value]) if Col.LEMMA.value in row else "", sid, did))
    if not forms:
        return Result.success(pd.DataFrame(columns=_VECTOR_COLUMNS), *diags)
    try:
        vectors = resolved.embed(forms, [sent_text.get((did, sid), "") for _, _, sid, did in metas])
    except RuntimeError as exc:
        return Result.failure(Diagnostic.error("CTX_MODEL_MISSING", str(exc)))
    diags.extend(truncation_note(resolved))
    rows = [
        {
            "Word": word,
            "Lemma": lemma,
            "Sentence ID": sid,
            "Document ID": str(did),
            "Vector": ",".join(f"{v:.4f}" for v in vec),
        }
        for (word, lemma, sid, did), vec in zip(metas, vectors, strict=True)
    ]
    return Result.success(pd.DataFrame(rows, columns=_VECTOR_COLUMNS), *diags)


#: A word needs this many uses before two senses are worth testing for.
_WSI_MIN_USES = 4
#: Two clusters count as two senses only when they separate at least this well
#: (mean silhouette, cosine) ...
_WSI_SEPARATION = 0.2
#: ... and the smaller one holds at least this share of the uses (and two uses):
#: with few uses, one odd sentence separates well on its own.
_WSI_MIN_SHARE = 0.15


def _two_senses(vectors: list[list[float]]) -> list[int]:
    """1 or 2 per use: two senses only when the uses really fall into two groups."""
    return _split(vectors)[0]


def _split(vectors: list[list[float]]) -> tuple[list[int], float]:
    """Sense per use, and how clearly the two senses separate (0.0 for one sense).

    Seeded k-means with k=2 over the uses' vectors, kept when the clusters
    separate (silhouette >= ``_WSI_SEPARATION``) and neither is a stray use
    or two; otherwise every use is sense 1. Sense 1 is the more frequent
    reading. Calibrated on BERT base over hand-built uses: "bank" (money and
    river, 0.47), "play" (0.35) and "run" (0.22) split; "the" (0.17) and
    one-sense nouns whose best split is 3 against 1 do not.
    """
    if len(vectors) < _WSI_MIN_USES:
        return [1] * len(vectors), 0.0
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    matrix = np.asarray(vectors, dtype=float)
    if len(np.unique(matrix.round(6), axis=0)) < _WSI_MIN_USES:
        # Mostly the same sentence repeated: nothing to split.
        return [1] * len(vectors), 0.0
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(matrix)
    smaller = int(np.bincount(labels, minlength=2).min())
    if smaller < max(2, _WSI_MIN_SHARE * len(vectors)):
        return [1] * len(vectors), 0.0
    separation = float(silhouette_score(matrix, labels, metric="cosine"))
    if separation < _WSI_SEPARATION:
        return [1] * len(vectors), 0.0
    larger = int(np.bincount(labels).argmax())
    return [1 if int(label) == larger else 2 for label in labels], round(separation, 4)


def wsi_senses(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    min_count: int = 2,
    backend: EmbeddingBackend | None = None,
    model: str | None = None,
    vectors: pd.DataFrame | None = None,
) -> Result[pd.DataFrame]:
    """Word-sense induction: does a word's use split into two readings?

    The uses of each lemma seen in at least two sentences are clustered in
    two (seeded k-means over their contextual vectors); the split is kept
    only when the two groups separate clearly, and otherwise every use is
    sense 1. Sense 1 is the more frequent reading. Two senses at most: this
    asks "one meaning or two?", which is what a reader can check by reading
    the sentences, not how many a dictionary lists. Function words (the
    topic model's stopword list) and non-words are left out.

    *vectors* is a :func:`contextual_vectors` table already made for this
    frame; senses group by its Lemma column whichever field named the
    words, so passing it skips embedding every token a second time.
    """
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("WSI_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_WSI_COLUMNS))

    if vectors is not None:
        cdf = vectors
    else:
        ctx_res = contextual_vectors(frame, field=field, backend=backend, model=model)
        if not ctx_res.ok:
            return Result[pd.DataFrame](None, ctx_res.diagnostics)
        cdf = ctx_res.unwrap()
    if cdf.empty:
        return Result.success(pd.DataFrame(columns=_WSI_COLUMNS))

    rows: list[dict[str, object]] = []
    from core.analysis.lda import STOPWORDS

    for lemma, group in cdf.groupby("Lemma", sort=False):
        if len(group) < min_count:
            continue
        # Function words have no senses to find: "the" splits by where it
        # sits in the sentence, which reads as two senses and is none.
        if str(lemma).lower() in STOPWORDS or not str(lemma).isalpha():
            continue
        if group["Sentence ID"].nunique() < 2:
            for _, row in group.iterrows():
                rows.append(
                    {
                        "Lemma": lemma,
                        "Sense": 1,
                        "Separation": 0.0,
                        "Sentence ID": row["Sentence ID"],
                        "Document ID": row["Document ID"],
                    }
                )
            continue
        senses, separation = _split([[float(v) for v in str(text).split(",")] for text in group["Vector"]])
        for (_, row), sense in zip(group.iterrows(), senses, strict=True):
            rows.append(
                {
                    "Lemma": lemma,
                    "Sense": sense,
                    "Separation": separation,
                    "Sentence ID": row["Sentence ID"],
                    "Document ID": row["Document ID"],
                }
            )

    if not rows:
        return Result.success(pd.DataFrame(columns=_WSI_COLUMNS))
    out = pd.DataFrame(rows, columns=_WSI_COLUMNS)
    out = out.sort_values(["Lemma", "Sense", "Document ID"]).reset_index(drop=True)
    return Result.success(out)


#: The uses of every word that splits into two senses, with their sentences.
SENSE_USE_COLUMNS = ["Lemma", "Sense", "Separation", "Share", "Document", "Sentence ID", "Document ID", "Sentence"]


def sense_uses(frame: pd.DataFrame, senses: pd.DataFrame, *, names: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Each use of each two-sense word, with the sentence it is used in.

    ``wsi.csv`` says which sentence holds which sense by id; a reader needs
    the sentence itself to judge whether the split is a real difference of
    meaning, so this joins the text back on. ``Share`` is the sense's share
    of the word's uses. Words with one sense are left out: there is nothing
    to compare. *names* gives each document id the name a reader knows it
    by; without it the table's own Document column is used.
    """
    if senses.empty or "Separation" not in senses.columns:
        return pd.DataFrame(columns=SENSE_USE_COLUMNS)
    split = senses[pd.to_numeric(senses["Separation"], errors="coerce").fillna(0) > 0].copy()
    if split.empty:
        return pd.DataFrame(columns=SENSE_USE_COLUMNS)
    doc, sent, form = Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value, Col.FORM.value
    keys = frame[[doc, sent]].astype(str)
    text = frame[form].astype(str).groupby([keys[doc], keys[sent]], sort=False).agg(" ".join)
    if names is None and Col.DOCUMENT.value in frame.columns:
        firsts = frame.drop_duplicates(doc)
        names = {str(k): _document_name(str(v)) for k, v in zip(firsts[doc], firsts[Col.DOCUMENT.value], strict=True)}
    names = dict(names or {})
    split["Document ID"] = split["Document ID"].astype(str)
    split["Sentence ID"] = split["Sentence ID"].astype(str)
    # One row per sentence and sense: a word used twice in one sentence is one passage to read.
    split = split.drop_duplicates(["Lemma", "Sense", "Document ID", "Sentence ID"])
    totals = split.groupby("Lemma")["Sense"].transform("size")
    split["Share"] = (split.groupby(["Lemma", "Sense"])["Sense"].transform("size") / totals).round(4)
    split["Document"] = split["Document ID"].map(lambda value: names.get(value, value))
    split["Sentence"] = [
        text.get((d, s_), "") for d, s_ in zip(split["Document ID"], split["Sentence ID"], strict=True)
    ]
    return split[SENSE_USE_COLUMNS].reset_index(drop=True)


def _document_name(value: str) -> str:
    """A document's file name without its folder or extension."""
    stem = value.replace("\\", "/").rsplit("/", 1)[-1]
    return stem.rsplit(".", 1)[0] if "." in stem else stem
