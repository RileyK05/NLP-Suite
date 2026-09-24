"""The statistics a publication figure adds to what the panel prepared.

Kept small and dependency-free (numpy only): the frozen engine carries no
statsmodels, and each quantity here is simple enough to state in a caption.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np

__all__ = ["decade_of", "rolling_band"]


def rolling_band(
    xs: Sequence[float], ys: Sequence[float], window: int, low: float = 0.25, high: float = 0.75
) -> list[tuple[float, float, float]]:
    """The interquartile range over the same windows as the rolling median.

    ``panel_helpers.rolling_median``'s windows exactly -- centred where they
    fit, the first or last ``window`` points near an end -- so the band and
    the line describe the same documents: the line is their middle, the band
    the middle half of them. "No averages without spread", drawn.
    """
    pairs = sorted((float(x), float(y)) for x, y in zip(xs, ys, strict=True) if np.isfinite(x) and np.isfinite(y))
    count = len(pairs)
    if window < 1 or count < window:
        return []
    half = window // 2
    out = []
    for index in range(count):
        start = min(max(0, index - half), count - window)
        values = np.array([y for _, y in pairs[start : start + window]])
        out.append((pairs[index][0], float(np.quantile(values, low)), float(np.quantile(values, high))))
    return out


_LEADING_YEAR = re.compile(r"^\s*(\d{4})\b")


def decade_of(label: str) -> str:
    """ "1934 Roosevelt" -> "1930s"; "" when the label carries no year."""
    match = _LEADING_YEAR.match(label)
    if not match:
        return ""
    year = int(match.group(1))
    if not 1000 <= year <= 2500:
        return ""
    return f"{year // 10 * 10}s"
