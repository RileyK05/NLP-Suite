"""Install check — verify optional extras and environment."""

from __future__ import annotations

import importlib.util
import sys

import pandas as pd

from core.result import Result

__all__ = ["check_environment"]

_EXTRAS: dict[str, list[str]] = {
    "stanza": ["stanza"],
    "spacy": ["spacy"],
    "app": ["streamlit", "plotly"],
    "plotly": ["plotly"],
    "dev": ["pytest", "ruff", "mypy"],
}


def check_environment() -> Result[pd.DataFrame]:
    """Return a DataFrame of extra -> installed?"""
    rows: list[dict[str, object]] = []
    for extra, mods in _EXTRAS.items():
        for mod in mods:
            found = importlib.util.find_spec(mod) is not None
            rows.append(
                {
                    "Extra": extra,
                    "Module": mod,
                    "Installed": found,
                    "Python": sys.version.split()[0],
                }
            )
    df = pd.DataFrame(rows, columns=["Extra", "Module", "Installed", "Python"])
    return Result.success(df)
