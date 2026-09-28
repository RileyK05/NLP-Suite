"""Comparing document sets: sides, measures, words, focus, topics and tone (docs/internal/PLAN_0.5.0.md 3.4).

Pure analysis over one parse of the joined corpus. Every document carries its
side in its details (``Side``), set by whoever built the corpus -- the runner
for a comparison job, a script for ``nlp.compare``. :func:`core.contrast.run.contrast`
is the one entry point.
"""

from core.contrast.run import CONTRAST_METHODS, ContrastSpec, contrast

__all__ = ["CONTRAST_METHODS", "ContrastSpec", "contrast"]
