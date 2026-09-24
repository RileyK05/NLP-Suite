"""Document matching and duplicates (FR-3.3) — exact, normalized, fuzzy.

Three reproducible tiers: byte-identical SHA groups, whitespace/case-folded
SHA groups, and TF-IDF cosine pairs at an explicit 0-100 threshold (via
``doc_similarity``). The legacy NER actor-overlap intruder workflow needs
actor lists and stays follow-up work — these tiers cover the packet's
exact/normalized/fuzzy contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import pandas as pd

from core.analysis.doc_similarity import DuplicatesResult, find_duplicates
from core.io.reader import Corpus, display_names
from core.result import Diagnostic, Result

__all__ = ["GroupsResult", "exact_groups", "fuzzy_pairs", "normalize_text", "normalized_groups"]

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class GroupsResult:
    frame: pd.DataFrame  # Group, Size, Members

    def to_frame(self) -> pd.DataFrame:
        return self.frame.copy()


def normalize_text(text: str) -> str:
    """Casefolded, whitespace-collapsed text for normalized matching."""
    return _WS_RE.sub(" ", text.casefold()).strip()


def _groups(corpus: Corpus, key: str) -> GroupsResult:
    buckets: dict[str, list[tuple[int, str]]] = {}
    names = display_names(corpus.docs)
    for doc in corpus.docs:
        digest = hashlib.sha256((doc.text if key == "exact" else normalize_text(doc.text)).encode("utf-8")).hexdigest()
        buckets.setdefault(digest, []).append((doc.doc_id, names[doc.doc_id]))
    rows = [
        {"Group": idx, "Size": len(members), "Members": ", ".join(sorted(name for _doc_id, name in members))}
        for idx, members in enumerate((members for members in buckets.values() if len(members) > 1), start=1)
    ]
    return GroupsResult(frame=pd.DataFrame(rows, columns=["Group", "Size", "Members"]))


def exact_groups(corpus: Corpus) -> Result[GroupsResult]:
    """Groups of byte-identical documents."""
    return Result.success(_groups(corpus, "exact"))


def normalized_groups(corpus: Corpus) -> Result[GroupsResult]:
    """Groups identical after casefold + whitespace collapse."""
    return Result.success(_groups(corpus, "normalized"))


def fuzzy_pairs(corpus: Corpus, threshold: float = 80.0) -> Result[DuplicatesResult]:
    """TF-IDF cosine pairs at or above *threshold* (0-100, validated)."""
    from core.analysis.doc_similarity import _validate_threshold

    problem = _validate_threshold(threshold)
    if problem is not None:
        return Result.failure(problem)
    return find_duplicates(corpus, threshold)
