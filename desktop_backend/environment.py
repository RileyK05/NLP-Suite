"""Cheap, offline environment inventory. Presence is not scientific validation."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

from core.profiler.registry import ToolSpec

# Scanning packages (nltk import, model find_spec) costs seconds on a cold
# disk under real-time antivirus scanning. The result is process-stable, so
# serve it from a cache; Setup-driven installs re-check after this TTL.
_INVENTORY_TTL_SECONDS = 60.0
_inventory_lock = threading.Lock()
_inventory_cache: dict[str, tuple[float, dict[str, bool]]] = {}


def inventory() -> dict[str, bool]:
    with _inventory_lock:
        cached = _inventory_cache.get("scan")
        if cached is not None and time.monotonic() - cached[0] < _INVENTORY_TTL_SECONDS:
            return dict(cached[1])
        found = _scan()
        _inventory_cache["scan"] = (time.monotonic(), found)
        return dict(found)


def _nltk_corpora_paths() -> list[str]:
    """Mirror nltk.data.path without importing nltk (its __init__ pulls scipy)."""
    paths: list[str] = []
    for variable in ("NLTK_DATA", "NLP_SUITE_NLTK_DATA"):
        value = os.environ.get(variable)
        if value:
            paths.extend(part for part in value.split(os.pathsep) if part)
    # The app's vendored .nltk_data lives next to the package, wherever the
    # process was started from.
    paths.append(str(_repo_root() / ".nltk_data"))
    paths.extend(
        [
            os.path.join(sys.prefix, "nltk_data"),
            os.path.join(sys.prefix, "share", "nltk_data"),
            os.path.join(sys.prefix, "lib", "nltk_data"),
            os.path.expanduser(os.path.join("~", "nltk_data")),
            os.path.expanduser(os.path.join("~", ".nltk_data")),
        ]
    )
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scan() -> dict[str, bool]:
    names = (
        "spacy",
        "en_core_web_sm",
        "stanza",
        "gensim",
        "nltk",
        "vaderSentiment",
        "torch",
        "transformers",
        "pdfminer",
        "docx",
        "striprtf",
        "plotly",
        # Static chart export (PNG/SVG/PDF) goes through kaleido. Without it
        # the dialog was still offering those formats and the job failed after
        # running; scanning for it lets the interface say so beforehand.
        "kaleido",
        "wordcloud",
        "openpyxl",
        # Publication figures (core/viz/static) and a run's figures/.
        "matplotlib",
        "seaborn",
    )
    found = {name: importlib.util.find_spec(name) is not None for name in names}
    # NRC ships its lexicon inside the nrclex wheel; no separate asset needed.
    found["nrclex"] = importlib.util.find_spec("nrclex") is not None
    # A parse needs *a* backend with a model, not one particular backend (see
    # core.pipelines.resolve). Stanza keeps its models in a resources directory
    # rather than a package, so presence is a filesystem question -- loading one
    # to be certain takes seconds and belongs in `nlp-doctor`, not in a scan
    # that /api/tools waits on. Execution validates it either way.
    found["stanza_model_en"] = found["stanza"] and any(
        os.path.isdir(os.path.join(root, "en"))
        for root in (
            os.environ.get("STANZA_RESOURCES_DIR", ""),
            os.path.expanduser(os.path.join("~", "stanza_resources")),
            os.path.expanduser(os.path.join("~", "AppData", "Local", "StanfordNLP", "stanza")),
        )
        if root
    )
    if found["nltk"]:
        # Corpora presence is a filesystem question; importing nltk here would
        # drag scipy/BLAS into the API server and stall /api/tools for minutes
        # on a cold disk.
        roots = _nltk_corpora_paths()
        for name in ("wordnet", "sentiwordnet"):
            found[name] = any(
                os.path.exists(os.path.join(root, "corpora", candidate))
                for root in roots
                for candidate in (name, name + ".zip")
            )
    return found


# What each scanned name is, in words, and what stops working without it.
# The scan keys are import names and corpus directory names; showing them to a
# reader as "en_core_web_sm" or "vaderSentiment" asks them to know the packaging
# to understand their own installation. ``tests/test_labels.py`` requires an
# entry for every name ``_scan`` reports, so the two cannot drift.
COMPONENT_LABELS: dict[str, tuple[str, str]] = {
    "spacy": ("spaCy", "The faster of the two English parsers."),
    "en_core_web_sm": ("spaCy English model", "The language model spaCy parses with."),
    "stanza": ("Stanza", "The more detailed English parser."),
    "stanza_model_en": ("Stanza English model", "The language model Stanza parses with."),
    "gensim": ("Gensim", "Topic discovery and word relationships."),
    "nltk": ("NLTK", "Word senses, lemma lookup and sentiment lexicons."),
    "wordnet": ("WordNet", "The dictionary behind nominalization and word senses."),
    "sentiwordnet": ("SentiWordNet", "Sentiment scores attached to word senses."),
    "vaderSentiment": ("VADER", "Sentence-level sentiment scoring."),
    "nrclex": ("NRC emotion lexicon", "The ten-emotion vocabulary."),
    "torch": ("PyTorch", "Runs the transformer models."),
    "transformers": ("Transformers", "Contextual embeddings and transformer topics."),
    "pdfminer": ("PDF reader", "Imports PDF documents as text."),
    "docx": ("Word reader", "Imports .docx documents as text."),
    "striprtf": ("RTF reader", "Imports .rtf documents as text."),
    "plotly": ("Plotly", "Draws the interactive charts."),
    "kaleido": ("Kaleido", "Saves charts as PNG, SVG or PDF."),
    "wordcloud": ("Wordcloud", "Renders wordclouds as pictures."),
    "openpyxl": ("Excel writer", "Writes native .xlsx workbooks and their charts."),
    "matplotlib": ("Matplotlib", "Draws publication figures and a finished run's figures folder."),
    "seaborn": ("Seaborn", "The statistical layers of publication figures: violins, clustered heatmaps."),
}


def components(found: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    """The inventory as something a reader can act on.

    ``inventory`` answers in import names. This answers in what the person
    would lose, which is the question someone opens Settings to ask.
    """
    scanned = inventory() if found is None else found
    described = []
    for name, present in sorted(scanned.items()):
        label, purpose = COMPONENT_LABELS.get(name, (name, ""))
        described.append({"name": name, "label": label, "purpose": purpose, "present": present})
    return described


def _parser_missing(found: dict[str, bool]) -> list[str]:
    """What is missing for a parse, or nothing if any backend can do it.

    Either backend carries a run, so a machine with a complete Stanza install
    and no spaCy must not be told that 24 of its analyses are unavailable. When
    neither is ready the message names the spaCy route, because it is one
    command and a far smaller download.
    """
    if found.get("spacy", False) and found.get("en_core_web_sm", False):
        return []
    if found.get("stanza_model_en", False):
        return []
    return ["spacy", "en_core_web_sm"]


def availability(spec: ToolSpec, found: dict[str, bool]) -> dict[str, Any]:
    required = []
    if spec.requires_parse:
        required.extend(_parser_missing(found))
    optional = {
        "topics": ["gensim"],
        "sentiment": ["vaderSentiment"],
        "wordnet": ["nltk", "wordnet"],
        "embeddings": ["torch", "transformers"],
    }
    required.extend(optional.get(spec.optional_package, []))
    if spec.name == "sentiment_swn_hedono":
        required.append("sentiwordnet")
    if spec.name == "nrc":
        required.append("nrclex")
    if spec.name == "table_charts":
        required.extend(["plotly", "openpyxl"])
    if spec.name == "table_wordcloud_gephi":
        required.append("wordcloud")
    missing = sorted({name for name in required if not found.get(name, False)})
    if missing:
        return {
            "state": "needs_setup",
            "message": "This analysis needs a component that is unavailable in this installation. Check Settings & backups.",
            "missing": missing,
        }
    if spec.assets or any(param.required and param.type == "path" for param in spec.params):
        return {"state": "needs_input", "message": "Choose the required resource/table file in settings", "missing": []}
    return {"state": "available", "message": "Local prerequisites found; execution validates them", "missing": []}
