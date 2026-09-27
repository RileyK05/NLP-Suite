"""Find the closest sentence-level passages shared across two sides."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from core.analysis.doc_embeddings import sentence_vectors
from core.analysis.lexicon_series import term_hits
from core.conll.schema import Col
from core.contrast.sides import Alignment, side_of
from core.io.reader import Corpus
from core.models.vector_cache import VectorCache
from core.result import Diagnostic, Result

__all__ = ["CARRYOVER_COLUMNS", "MIN_SIMILARITY", "carried_over"]

MIN_SIMILARITY = 0.78
_MAX_ROWS = 500
_BLOCK_ROWS = 128
_MAX_GROUP_SENTENCES_PER_SIDE = 2_000
CARRYOVER_COLUMNS = [
    "Group",
    "Similarity",
    "Project ID A",
    "Document ID A",
    "Document A",
    "Sentence ID A",
    "Passage A",
    "Project ID B",
    "Document ID B",
    "Document B",
    "Sentence ID B",
    "Passage B",
]


@dataclass(frozen=True)
class CarryoverSpec:
    """Which two sides and which focused passages should be matched."""

    sides: tuple[str, str]
    focus: Mapping[str, Sequence[str]]
    match: str = "lemma"


def _nearest_pairs(left: np.ndarray, right: np.ndarray) -> list[tuple[int, int, float]]:
    """Exact mutual nearest neighbors, with bounded temporary memory."""
    best_left = np.full(len(left), -np.inf)
    match_right = np.full(len(left), -1, dtype=int)
    best_right = np.full(len(right), -np.inf)
    match_left = np.full(len(right), -1, dtype=int)
    for start in range(0, len(left), _BLOCK_ROWS):
        similarities = left[start : start + _BLOCK_ROWS] @ right.T
        right_slots = similarities.argmax(axis=1)
        scores = similarities[np.arange(len(similarities)), right_slots]
        best_left[start : start + len(scores)] = scores
        match_right[start : start + len(scores)] = right_slots
        left_slots = similarities.argmax(axis=0)
        right_scores = similarities[left_slots, np.arange(len(left_slots))]
        improved = right_scores > best_right
        best_right[improved] = right_scores[improved]
        match_left[improved] = start + left_slots[improved]
    return [
        (i, int(j), float(score))
        for i, (j, score) in enumerate(zip(match_right, best_left, strict=True))
        if j >= 0 and match_left[j] == i
    ]


def carried_over(  # noqa: PLR0912 -- validate, filter and publish the complete optional method
    table: pd.DataFrame,
    corpus: Corpus,
    alignment: Alignment,
    spec: CarryoverSpec,
    *,
    cache: VectorCache | None = None,
) -> Result[pd.DataFrame]:
    """Return reciprocal nearest sentence pairs within shared alignment groups.

    Each row keeps source project, document, and sentence identifiers next to
    the original sentence text, so readers can follow both passages back to
    their corpus records. Only sentences containing one of the selected focus
    terms are embedded. The threshold suppresses weak nearest matches.
    """
    if not spec.focus:
        return Result.success(
            pd.DataFrame(columns=CARRYOVER_COLUMNS),
            Diagnostic.warning("CONTRAST_CARRYOVER_NEEDS_FOCUS", "Carried-over passages need focus words."),
        )
    column = Col.LEMMA.value if spec.match == "lemma" else Col.FORM.value
    terms = {name: tuple(term.lower() for term in words) for name, words in spec.focus.items()}
    focused: set[tuple[str, str]] = set()
    for (document, sentence), part in table.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        words = part[column].fillna("").astype(str).str.lower().tolist()
        if term_hits(words, terms):
            focused.add((str(document), str(sentence)))
    if not focused:
        return Result.success(
            pd.DataFrame(columns=CARRYOVER_COLUMNS),
            Diagnostic.info("CONTRAST_CARRYOVER_NO_FOCUS", "No parsed sentences contained the selected focus words."),
        )
    documents = {str(doc.doc_id): doc for doc in corpus.docs}
    candidate_counts: dict[tuple[str, str], int] = {}
    for document, _sentence in focused:
        doc = documents.get(document)
        if doc is None or doc.doc_id not in alignment.groups:
            continue
        key = (alignment.groups[doc.doc_id], side_of(doc))
        candidate_counts[key] = candidate_counts.get(key, 0) + 1
    oversized = [
        (group, side, count)
        for (group, side), count in candidate_counts.items()
        if count > _MAX_GROUP_SENTENCES_PER_SIDE
    ]
    if oversized:
        group, side, count = max(oversized, key=lambda item: item[2])
        return Result.success(
            pd.DataFrame(columns=CARRYOVER_COLUMNS),
            Diagnostic.warning(
                "CONTRAST_CARRYOVER_TOO_MANY_CANDIDATES",
                f"{count:,} focus sentences in {group} on {side} exceed the {_MAX_GROUP_SENTENCES_PER_SIDE:,} limit. "
                "Narrow the document selection or use more specific focus words.",
                group=group,
                side=side,
                candidates=count,
                limit=_MAX_GROUP_SENTENCES_PER_SIDE,
            ),
        )
    selected = table.loc[
        [
            (str(document), str(sentence)) in focused
            for document, sentence in zip(table[Col.DOCUMENT_ID.value], table[Col.SENTENCE_ID.value], strict=True)
        ]
    ]
    shas = {str(doc.doc_id): doc.sha256 for doc in corpus.docs}
    embedded = sentence_vectors(selected, shas=shas, cache=cache)
    if embedded.value is None:
        message = "; ".join(item.message for item in embedded.diagnostics)
        return Result.success(
            pd.DataFrame(columns=CARRYOVER_COLUMNS),
            Diagnostic.warning(
                "CONTRAST_CARRYOVER_UNAVAILABLE",
                f"Carried-over passages could not be computed: {message}",
            ),
        )
    vectors = embedded.unwrap()
    groups: dict[str, dict[str, list[int]]] = {}
    for index, sentence in enumerate(vectors.sentences):
        doc = documents.get(sentence.doc_id)
        if doc is None or doc.doc_id not in alignment.groups:
            continue
        side = side_of(doc)
        if side not in spec.sides:
            continue
        label = alignment.groups[doc.doc_id]
        groups.setdefault(label, {spec.sides[0]: [], spec.sides[1]: []})[side].append(index)

    rows: list[dict[str, object]] = []
    for label, by_side in groups.items():
        left_indices, right_indices = by_side[spec.sides[0]], by_side[spec.sides[1]]
        if not left_indices or not right_indices:
            continue
        left_vectors = vectors.vectors[left_indices]
        right_vectors = vectors.vectors[right_indices]
        for left_slot, right_slot, cosine in _nearest_pairs(left_vectors, right_vectors):
            if cosine < MIN_SIMILARITY:
                continue
            left = vectors.sentences[left_indices[left_slot]]
            right = vectors.sentences[right_indices[right_slot]]
            left_doc, right_doc = documents[left.doc_id], documents[right.doc_id]
            rows.append(
                {
                    "Group": label,
                    "Similarity": round(cosine, 4),
                    "Project ID A": left_doc.source_project_id,
                    "Document ID A": left_doc.source_id,
                    "Document A": left_doc.name,
                    "Sentence ID A": left.sent_id,
                    "Passage A": left.text,
                    "Project ID B": right_doc.source_project_id,
                    "Document ID B": right_doc.source_id,
                    "Document B": right_doc.name,
                    "Sentence ID B": right.sent_id,
                    "Passage B": right.text,
                }
            )
    rows.sort(key=lambda row: (-cast(float, row["Similarity"]), str(row["Document A"]), str(row["Sentence ID A"])))
    frame = pd.DataFrame(rows[:_MAX_ROWS], columns=CARRYOVER_COLUMNS)
    diagnostics = list(vectors.diagnostics)
    diagnostics.append(
        Diagnostic.info(
            "CONTRAST_CARRYOVER",
            f"Found {len(frame)} reciprocal sentence match(es) at cosine similarity {MIN_SIMILARITY:.2f} or higher.",
            matches=len(frame),
            minimum_similarity=MIN_SIMILARITY,
            truncated=len(rows) > _MAX_ROWS,
        )
    )
    return Result.success(frame, *diagnostics)
