"""Pretrained models: the registry, where their files live, and how they run.

The analyses never import ``torch`` or ``transformers`` to use a model: a
registered model is an ONNX graph plus a ``tokenizer.json``, run by ONNX
Runtime (:mod:`core.models.onnx_backend`). Small models ship inside the
installer; larger ones are downloaded once from the Models page
(:mod:`core.models.download`).
"""

from __future__ import annotations

from core.models.locate import ModelStatus, model_dir, status, user_models_dir
from core.models.registry import MODELS, ModelKind, ModelSpec, get_model, models_of_kind

__all__ = [
    "MODELS",
    "ModelKind",
    "ModelSpec",
    "ModelStatus",
    "get_model",
    "model_dir",
    "models_of_kind",
    "status",
    "user_models_dir",
]
