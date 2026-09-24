"""Contextual embeddings + word-sense induction (FR-5.6).

``contextual_vectors`` embeds each token in its sentence through an
injectable backend; ``wsi_senses`` splits each lemma's occurrences into
two senses on the median of the vectors' first component (a documented
baseline, not a clustering claim).

The production backend is a lazily loaded Hugging Face transformer
(``TransformerBackend``): no model ships with the suite and none is
downloaded by the test suite — the first real embed pulls the named
model through the ``transformers`` cache, and a missing package or
model fails loudly instead of returning fake vectors.

The old MD5 hash vectors were production fraud (deterministic,
untrained, and labeled "contextual"); they now live only in
``tests/test_contextual.py`` as the offline fake behind the same
protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "EmbeddingBackend",
    "TransformerBackend",
    "contextual_vectors",
    "default_backend",
    "wsi_senses",
]

_DEFAULT_MODEL = "bert-base-uncased"

_VECTOR_COLUMNS = ["Word", "Lemma", "Sentence ID", "Document ID", "Vector"]
_WSI_COLUMNS = ["Lemma", "Sense", "Sentence ID", "Document ID"]


class EmbeddingBackend(Protocol):
    """One question the analyses need: embed these words in these sentences."""

    def dimension(self) -> int:
        """Embedding width."""
        ...

    def embed(self, words: Sequence[str], contexts: Sequence[str]) -> list[list[float]]:
        """One vector per (word, context) pair, in order."""
        ...


class TransformerBackend:
    """Production backend: last-hidden-state means over a word's pieces.

    The model loads on the first ``embed`` (CPU, no gradients); the
    package import is lazy so ``import core.analysis.contextual`` never
    requires ``transformers`` or ``torch``.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._pipe: Any = None
        self._dim: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed; contextual embeddings need the optional 'embeddings' extra"
            ) from exc
        try:
            self._pipe = pipeline("feature-extraction", model=self._model_name, device="cpu", truncation=True)
        except Exception as exc:
            raise RuntimeError(
                f"cannot load embedding model {self._model_name!r} "
                "(no cached copy and no network, or a broken download — "
                'pre-fetch with: python -c "from transformers import pipeline; '
                f"pipeline('feature-extraction', model={self._model_name!r})\")"
            ) from exc
        return self._pipe

    def dimension(self) -> int:
        if self._dim is None:
            pipe = self._load()
            self._dim = int(pipe.model.config.hidden_size)
        return self._dim

    def embed(self, words: Sequence[str], contexts: Sequence[str]) -> list[list[float]]:
        pipe = self._load()
        tokenizer = pipe.tokenizer
        vectors: list[list[float]] = []
        for word, context in zip(words, contexts, strict=True):
            encoding = tokenizer(context, return_tensors="pt", truncation=True)
            word_ids = encoding.word_ids()
            target = str(word).lower()
            # Subword pieces of the target word (first occurrence wins by
            # construction: word_ids order follows the context). Falls back
            # to the whole sentence when the word is not found (e.g. a
            # lemma that never surfaces verbatim).
            pieces = [
                i
                for i, wid in enumerate(word_ids)
                if wid is not None and _piece_text(tokenizer, encoding, i).lower().lstrip("#") in target
            ]
            if not pieces:
                pieces = [i for i, wid in enumerate(word_ids) if wid is not None]
            import torch

            with torch.no_grad():
                hidden = pipe.model(**{k: v for k, v in encoding.items() if k != "overflow_to_sample_mapping"})[0][0]
            import math

            selected = hidden[pieces].mean(dim=0).tolist()
            norm = math.sqrt(sum(v * v for v in selected)) or 1.0
            vectors.append([v / norm for v in selected])
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors


def _piece_text(tokenizer: Any, encoding: Any, index: int) -> str:
    try:
        return str(tokenizer.convert_ids_to_tokens(int(encoding["input_ids"][0][index])))
    except Exception:
        return ""


def default_backend(model_name: str = _DEFAULT_MODEL) -> Result[TransformerBackend]:
    """Resolve the transformer backend (package present, model load deferred)."""
    try:
        import transformers
    except ImportError:
        return Result.failure(
            Diagnostic.error(
                "CTX_BACKEND_MISSING",
                "transformers is not installed; contextual embeddings need the optional 'embeddings' extra",
                fix='python -m pip install "nlp-suite-ng[embeddings]"',
            )
        )
    return Result.success(TransformerBackend(model_name))


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

    words: list[str] = []
    metas: list[tuple[str, str, object, object]] = []  # word, lemma, sid, did
    for _, row in frame.iterrows():
        word = str(row[field.value])
        if not word.strip() or word.lower() in ("nan", "none"):
            continue
        did = row[Col.DOCUMENT_ID.value]
        sid = row[Col.SENTENCE_ID.value]
        words.append(word)
        metas.append((word, str(row[Col.LEMMA.value]) if Col.LEMMA.value in row else "", sid, did))
    if not words:
        return Result.success(pd.DataFrame(columns=_VECTOR_COLUMNS), *diags)
    try:
        vectors = resolved.embed(
            [w.lower() for w in words], [sent_text.get((did, sid), "") for _, _, sid, did in metas]
        )
    except RuntimeError as exc:
        message = str(exc)
        code = "CTX_BACKEND_MISSING" if "not installed" in message else "CTX_MODEL_MISSING"
        return Result.failure(Diagnostic.error(code, message, fix='python -m pip install "nlp-suite-ng[embeddings]"'))
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


def wsi_senses(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    min_count: int = 2,
    backend: EmbeddingBackend | None = None,
    model: str | None = None,
) -> Result[pd.DataFrame]:
    """Baseline word-sense induction: median split (k=2) over backend vectors.

    Occurrences of a lemma seen in at least two sentences split on the
    median of the vectors' first component; single-sentence lemmas stay
    sense 1. A baseline for plumbing, not a clustering result.
    """
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("WSI_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if frame.empty:
        return Result.success(pd.DataFrame(columns=_WSI_COLUMNS))

    ctx_res = contextual_vectors(frame, field=field, backend=backend, model=model)
    if not ctx_res.ok:
        return Result[pd.DataFrame](None, ctx_res.diagnostics)
    cdf = ctx_res.unwrap()
    if cdf.empty:
        return Result.success(pd.DataFrame(columns=_WSI_COLUMNS))

    rows: list[dict[str, object]] = []
    for lemma, group in cdf.groupby("Lemma", sort=False):
        if len(group) < min_count:
            continue
        if group["Sentence ID"].nunique() < 2:
            for _, row in group.iterrows():
                rows.append(
                    {
                        "Lemma": lemma,
                        "Sense": 1,
                        "Sentence ID": row["Sentence ID"],
                        "Document ID": row["Document ID"],
                    }
                )
            continue
        group_copy = group.copy()
        group_copy["v0"] = group_copy["Vector"].apply(lambda s: float(str(s).split(",")[0]))
        median = float(group_copy["v0"].median())
        for _, row in group_copy.iterrows():
            sense = 1 if float(row["v0"]) <= median else 2
            rows.append(
                {
                    "Lemma": lemma,
                    "Sense": sense,
                    "Sentence ID": row["Sentence ID"],
                    "Document ID": row["Document ID"],
                }
            )

    if not rows:
        return Result.success(pd.DataFrame(columns=_WSI_COLUMNS))
    out = pd.DataFrame(rows, columns=_WSI_COLUMNS)
    out = out.sort_values(["Lemma", "Sense", "Document ID"]).reset_index(drop=True)
    return Result.success(out)
