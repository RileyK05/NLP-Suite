"""Where a model's files live, and whether they are usable.

Three places are searched, in order:

1. ``NLP_SUITE_MODELS`` — a directory of ``<model id>/`` folders. For
   development and tests only; never mentioned to users. When set, it is
   the only place searched.
2. Bundled — ``models/<id>/`` inside the frozen engine (``sys._MEIPASS``)
   or at the root of a source checkout.
3. Downloaded — the user's models directory (:func:`user_models_dir`),
   which survives app updates because the updater replaces the app, not
   the user's data.

A folder counts only when every published file is present at its
published size. Checking sizes costs a ``stat`` per file, so status is
cheap enough for every tool listing; the SHA-256 is checked when a file is
downloaded (:mod:`core.models.download`), not on each look.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Literal

from core.models.registry import ModelSpec

__all__ = [
    "ModelStatus",
    "bundled_models_dir",
    "model_dir",
    "search_roots",
    "status",
    "user_models_dir",
]

ModelStatus = Literal["ready", "not_downloaded", "corrupt", "unpublished"]

#: Set by the desktop engine at start-up (``<app data>/models``) so its worker
#: processes find the same downloads.
USER_DIR_VARIABLE = "NLP_SUITE_MODELS_DIR"
OVERRIDE_VARIABLE = "NLP_SUITE_MODELS"


def user_models_dir() -> Path:
    """Where downloaded models go: the engine's choice, else the platform's."""
    chosen = os.environ.get(USER_DIR_VARIABLE, "").strip()
    if chosen:
        return Path(chosen)
    system: str = sys.platform  # a plain str, so each branch type-checks on every OS
    if system == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "NLP Suite" / "models"
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / "NLP Suite" / "models"
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "nlp-suite" / "models"


def bundled_models_dir() -> Path:
    """``models/`` inside the frozen engine, or at the source checkout root."""
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return Path(frozen) / "models"
    return Path(__file__).resolve().parents[2] / "models"


def search_roots() -> list[Path]:
    """Where to look, in order. The override, when set, is the only place:
    the test suite points it at its tiny fixture models so real ones in a
    developer's checkout never change a result."""
    override = os.environ.get(OVERRIDE_VARIABLE, "").strip()
    if override:
        return [Path(override)]
    return [bundled_models_dir(), user_models_dir()]


def _state(spec: ModelSpec, folder: Path) -> ModelStatus:
    files = spec.files
    present = [item for item in files if (folder / item.name).is_file()]
    if not present:
        return "not_downloaded"
    if len(present) < len(files):
        return "corrupt"
    for item in files:
        if (folder / item.name).stat().st_size != item.size:
            return "corrupt"
    return "ready"


def model_dir(spec: ModelSpec) -> Path | None:
    """The first folder holding a complete copy of *spec*, if any."""
    if not spec.published:
        return None
    for root in search_roots():
        folder = root / spec.id
        if folder.is_dir() and _state(spec, folder) == "ready":
            return folder
    return None


def status(spec: ModelSpec) -> ModelStatus:
    """``ready`` anywhere wins; a half-written download reads ``corrupt``."""
    if not spec.published:
        return "unpublished"
    seen: ModelStatus = "not_downloaded"
    for root in search_roots():
        folder = root / spec.id
        if not folder.is_dir():
            continue
        state = _state(spec, folder)
        if state == "ready":
            return "ready"
        if state == "corrupt":
            seen = "corrupt"
    return seen
