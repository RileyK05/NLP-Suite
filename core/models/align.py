"""Which word of which sentence each embedding request reads.

Shared by the ONNX and the PyTorch token backends, so the two read the
same pieces and a parity check between them compares models, not
alignment rules.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["Plan", "plan_requests"]


class Plan:
    """Unique contexts (split into words) and, per request, its context and word.

    ``requests[i] = (slot, position)``: request *i* reads word ``position``
    of ``contexts[slot]``, or the whole sentence when ``position`` is None
    (the word does not occur verbatim, e.g. a lemma like "be" in "was").
    """

    __slots__ = ("by_slot", "contexts", "requests")

    def __init__(self, contexts: list[list[str]], requests: list[tuple[int, int | None]]) -> None:
        self.contexts = contexts
        self.requests = requests
        self.by_slot: dict[int, list[int]] = {}
        for index, (slot, _) in enumerate(requests):
            self.by_slot.setdefault(slot, []).append(index)


def plan_requests(words: Sequence[str], contexts: Sequence[str]) -> Plan:
    """Group requests by sentence; the n-th ask for a word reads its n-th occurrence."""
    if len(words) != len(contexts):
        raise ValueError("words and contexts must pair up")
    unique: dict[str, int] = {}
    for context in contexts:
        unique.setdefault(context, len(unique))
    split = [context.split() or [""] for context in unique]
    seen: dict[tuple[int, str], int] = {}
    requests: list[tuple[int, int | None]] = []
    for word, context in zip(words, contexts, strict=True):
        slot = unique[context]
        target = str(word).lower()
        positions = [index for index, token in enumerate(split[slot]) if token.lower() == target]
        nth = seen.get((slot, target), 0)
        seen[(slot, target)] = nth + 1
        requests.append((slot, positions[nth % len(positions)] if positions else None))
    return Plan(split, requests)
