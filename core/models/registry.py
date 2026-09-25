"""The pretrained models the suite can run, as data.

One :class:`ModelSpec` per model. A spec says what the model is for
(``kind``), how its output becomes a vector or a label (``pooling``,
``labels``), where it came from (``source`` at a pinned ``revision``) and
whether it ships inside the installer (``bundled``) or is downloaded from
the Models page. The files each model is made of, with their sizes and
SHA-256, are written by ``scripts/export_models.py`` into
:mod:`core.models._files`; a model with no files there has not been
published yet.

Model ids are the names readers already know from the syllabus and from
Hugging Face ("bert-base-uncased"), and every older spelling a saved run
may carry ("google-bert/bert-base-uncased") resolves through ``aliases``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.models._files import FILES, PRECISION, RELEASE

__all__ = [
    "MODELS",
    "ModelFile",
    "ModelKind",
    "ModelSpec",
    "get_model",
    "models_of_kind",
    "release_url",
]

ModelKind = Literal["token_embeddings", "sentence_embeddings", "classifier"]
Pooling = Literal["cls", "mean", "last_token"]

#: Where published model files live: one GitHub release per model set,
#: versioned apart from the app so a new app never re-downloads a model.
_RELEASES = "https://github.com/RileyK05/NLP-Suite/releases/download"


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One file of a published model: its name, size in bytes and SHA-256."""

    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What one model is, where it came from, and how to read its output."""

    id: str
    display_name: str
    kind: ModelKind
    #: Hugging Face repository and the commit it was exported from.
    source: str
    revision: str
    dims: int
    pooling: Pooling
    #: Tokens per input the model reads; longer inputs are truncated (and said so).
    max_tokens: int
    #: Plain-language purpose, shown on the Models page.
    description: str
    license: str = "Apache-2.0"
    bundled: bool = False
    #: Hidden states a token model exposes (embeddings + one per layer).
    hidden_states: int = 0
    #: Prefix for search queries (Qwen reads queries and documents differently).
    query_prompt: str = ""
    #: Output classes, in logit order (classifiers only).
    labels: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    @property
    def files(self) -> tuple[ModelFile, ...]:
        """The published files, or ``()`` before the model is exported."""
        return tuple(ModelFile(name, size, sha) for name, (size, sha) in sorted(FILES.get(self.id, {}).items()))

    @property
    def published(self) -> bool:
        return bool(FILES.get(self.id))

    @property
    def size_bytes(self) -> int:
        return sum(item.size for item in self.files)

    @property
    def size_mb(self) -> int:
        return round(self.size_bytes / 1_000_000)

    @property
    def precision(self) -> str:
        """How the weights are stored (``fp16`` or ``int8``); ``""`` if unpublished."""
        return PRECISION.get(self.id, "")


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="bert-base-uncased",
        display_name="BERT base (uncased)",
        kind="token_embeddings",
        source="google-bert/bert-base-uncased",
        revision="86b5e0934494bd15c9632b12f734a8a67f723594",
        dims=768,
        pooling="mean",
        max_tokens=512,
        hidden_states=13,
        description="The original BERT: a vector for every word in its sentence. "
        "Used by word senses, Word2Vec via BERT, BERT topics and summaries.",
        bundled=True,
        aliases=("google-bert/bert-base-uncased", "bert"),
    ),
    ModelSpec(
        id="distilbert-sst2",
        display_name="DistilBERT sentiment (SST-2)",
        kind="classifier",
        source="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0fa89d2f80546fda750413ed43d93601a13",
        dims=768,
        pooling="cls",
        max_tokens=512,
        labels=("NEGATIVE", "POSITIVE"),
        description="Sentence sentiment, positive or negative. It has no neutral class: "
        "a sentence it is unsure about scores near zero.",
        bundled=True,
        aliases=(
            "distilbert-base-uncased-finetuned-sst-2-english",
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        ),
    ),
    ModelSpec(
        id="granite-embedding-english-r2",
        display_name="Granite Embedding English R2",
        kind="sentence_embeddings",
        source="ibm-granite/granite-embedding-english-r2",
        revision="47ea694b257b703fee9253d75c2b1f2985180498",
        dims=768,
        pooling="cls",
        max_tokens=512,
        description="One vector per sentence or document, trained for similarity and search. "
        "Used by document embeddings and semantic search.",
        bundled=True,
        aliases=("ibm-granite/granite-embedding-english-r2", "granite"),
    ),
    ModelSpec(
        id="qwen3-embedding-0.6b",
        display_name="Qwen3 Embedding 0.6B",
        kind="sentence_embeddings",
        source="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        dims=1024,
        pooling="last_token",
        max_tokens=512,
        query_prompt="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        description="A larger, stronger sentence and document embedding model. "
        "Slower than Granite; worth it for semantic search and fine distinctions.",
        aliases=("Qwen/Qwen3-Embedding-0.6B", "qwen", "qwen3-embedding"),
    ),
)

_BY_NAME: dict[str, ModelSpec] = {}
for _spec in MODELS:
    for _name in (_spec.id, *_spec.aliases):
        _BY_NAME[_name.lower()] = _spec


def get_model(name: str) -> ModelSpec | None:
    """The spec for an id or any alias, case-insensitively."""
    return _BY_NAME.get(str(name).strip().lower())


def models_of_kind(*kinds: ModelKind) -> tuple[ModelSpec, ...]:
    """Specs of the given kinds (every spec when none is named), in registry order."""
    return tuple(spec for spec in MODELS if not kinds or spec.kind in kinds)


def release_url(spec: ModelSpec, file_name: str) -> str:
    """Download URL of one published file (assets are flat: ``<id>--<file>``)."""
    return f"{_RELEASES}/{RELEASE}/{spec.id}--{file_name}"
