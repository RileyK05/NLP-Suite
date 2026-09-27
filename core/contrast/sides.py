"""Which side each document is on, and which group it is compared within.

A comparison of every State of the Union with every inaugural mostly shows
that addresses to Congress differ from addresses to the nation -- true, and
nothing anyone needed a tool for. Lining the sides up ("alignment") compares
like with like: the same president in both settings, or the same decade.
Groups that only one side has are left out, and said to be.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.corpus_axis import axis_of
from core.io.document_fields import field_value
from core.io.reader import Corpus, Document

__all__ = ["ALL", "SIDE", "Alignment", "align", "group_label", "period_groups", "side_of"]

#: The detail every compared document carries.
SIDE = "Side"
#: The one group of an unaligned comparison.
ALL = "All documents"


def side_of(doc: Document) -> str:
    return field_value(doc.details, SIDE)


def period_groups(corpus: Corpus) -> dict[int, str] | None:
    """The order axis's period labels per document, or None for a dated corpus.

    A corpus with chapters and no dates compares within blocks of chapters
    ("Chapters 1-15") -- the shared-period question asked of a book. A dated
    corpus keeps the decade rule of :func:`group_label`, unchanged; a corpus
    with neither places nothing and the comparison says so.
    """
    axis = axis_of(corpus)
    if axis.kind != "order":
        return None
    return {int(doc_id): label for doc_id, label in axis.periods().items()}


def group_label(doc: Document, alignment: str, periods: Mapping[int, str] | None = None) -> str | None:
    """The group *doc* is compared within, or None when it has no value for it.

    ``periods`` is the order axis's period labels (from
    :func:`period_groups`), needed because a block of chapters is a cut of
    the whole corpus, not of one document. Without it -- a dated corpus -- a
    period is the decade, as it always was.
    """
    if alignment == "none":
        return ALL
    if alignment == "period":
        if periods is not None:
            return periods.get(doc.doc_id)
        return f"{doc.date.year // 10 * 10}s" if doc.date is not None else None
    name = alignment.removeprefix("field:")
    return field_value(doc.details, name) or None


@dataclass(frozen=True)
class Alignment:
    """The groups every side has, and what was left out."""

    #: doc_id -> group, for documents in a group every side has.
    groups: dict[int, str]
    #: Groups every side has, in order of first appearance.
    shared: tuple[str, ...]
    #: side -> groups only that side has (left out).
    only: dict[str, tuple[str, ...]]
    #: Documents with no value for the alignment (left out).
    unplaced: tuple[str, ...]
    #: What one period is called in the coverage sentence ("decade", "period").
    period_noun: str = "decade"

    def coverage(self, alignment: str) -> str:
        """The plain sentence plan 3.6 rule 4 asks for."""
        if alignment == "none":
            return ""
        noun = self.period_noun if alignment == "period" else alignment.removeprefix("field:")
        parts = [f"{len(self.shared)} {noun} value{'s are' if len(self.shared) != 1 else ' is'} on every side"]
        for side, groups in self.only.items():
            if groups:
                shown = ", ".join(groups[:6]) + (" ..." if len(groups) > 6 else "")  # noqa: PLR2004 - a readable list
                parts.append(f"{len(groups)} only on {side} and were left out ({shown})")
        if self.unplaced:
            parts.append(f"{len(self.unplaced)} document(s) have no {noun} and were left out")
        return "; ".join(parts) + "."


def align(corpus: Corpus, sides: Sequence[str], alignment: str) -> Alignment:
    """Group every document for *alignment*, keeping only groups on every side."""
    by_side: dict[str, list[str]] = {side: [] for side in sides}
    labels: dict[int, str] = {}
    unplaced: list[str] = []
    periods = period_groups(corpus) if alignment == "period" else None
    for doc in corpus.docs:
        side = side_of(doc)
        if side not in by_side:
            continue
        label = group_label(doc, alignment, periods)
        if label is None:
            unplaced.append(doc.name)
            continue
        labels[doc.doc_id] = label
        if label not in by_side[side]:
            by_side[side].append(label)
    first = by_side[sides[0]] if sides else []
    shared = tuple(label for label in first if all(label in by_side[side] for side in sides))
    only = {side: tuple(label for label in labels_ if label not in shared) for side, labels_ in by_side.items()}
    kept = {doc_id: label for doc_id, label in labels.items() if label in shared}
    return Alignment(kept, shared, only, tuple(unplaced), period_noun="period" if periods is not None else "decade")
