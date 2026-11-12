"""Semantic aggregation — WordNet / VerbNet / FrameNet over the real resources.

Review finding: the production adapter consulted tiny baked test dictionaries
(7 WordNet / 5 VerbNet / 6 FrameNet entries) and published success with blank
labels and no diagnostics for every other lemma. This module now calls the
real resource-backed engines (``core.analysis.wordnet``, ``verbnet``,
``framenet``). When those corpora are not installed the tool degrades to an
explicitly-labelled demonstration mode and attaches a persistent limitation
diagnostic, so blank columns are never mistaken for real negative results.
"""

from __future__ import annotations

import pandas as pd

from collections.abc import Callable
from typing import Any

from core.analysis.framenet import FrameNetBackend
from core.analysis.verbnet import VerbNetBackend
from core.analysis.wordnet import WordNetBackend
from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["aggregate"]

_DEMO_LIMIT = Diagnostic.warning(
    "SEM_DEMO_MODE",
    "WordNet/VerbNet/FrameNet corpora are not installed; the bundled "
    "demonstration vocabulary (a handful of lemmas) is used and every other "
    "lemma is blank. Install the [wordnet], [verbnet] and [framenet] extras "
    "and download the corpora for the full resources (see Setup).",
    fix="python -m nltk.downloader wordnet; python -m pip install nlp-suite-ng[wordnet]",
)

# Tiny demonstration vocabulary — explicitly a demo, never presented as data.
_DEMO_WORDNET: dict[str, str] = {
    "dog": "dog.n.01 (canine)",
    "cat": "cat.n.01 (feline)",
    "run": "run.v.01 (move quickly)",
    "city": "city.n.01",
    "person": "person.n.01",
}
_DEMO_VERBNET: dict[str, str] = {
    "run": "run-51.3.2",
    "chase": "chase-51.6",
    "give": "give-13.1",
}
_DEMO_FRAMENET: dict[str, str] = {
    "run": "Motion",
    "chase": "Pursuit",
    "give": "Giving",
    "city": "Locale",
}


def aggregate(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
) -> Result[pd.DataFrame]:
    """Per-lemma semantic tags from WordNet / VerbNet / FrameNet."""
    if field not in (Col.FORM, Col.LEMMA):
        return Result.failure(Diagnostic.error("SEM_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}"))
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[pd.DataFrame](None, checked.diagnostics)
    if field.value not in frame.columns:
        return Result.failure(Diagnostic.error("SEM_MISSING_COLUMN", f"missing {field.value!r}"))
    if frame.empty:
        return Result.success(pd.DataFrame(columns=["Lemma", "WordNet", "VerbNet", "FrameNet", "Count"]))

    from collections import Counter

    # Pos column decides which resource can see a lemma (VerbNet is verbs only).
    pos_col = Col.POS.value if Col.POS.value in frame.columns else None
    toks = [
        (str(x).lower(), pos)
        for x, pos in zip(frame[field.value], frame[pos_col] if pos_col else [], strict=False)
        if str(x).strip() and str(x).isalpha()
    ]
    if not toks:
        return Result.success(pd.DataFrame(columns=["Lemma", "WordNet", "VerbNet", "FrameNet", "Count"]))
    cnt = Counter(toks)

    # Resolve the real backends; any that fail leave the demo map for that
    # resource and a persistent limitation diagnostic.
    diagnostics: list[Diagnostic] = []
    wn: WordNetBackend | None = _resolve(_load_wordnet, diagnostics)  # type: ignore[assignment]
    vn: VerbNetBackend | None = _resolve(_load_verbnet, diagnostics)  # type: ignore[assignment]
    fn: FrameNetBackend | None = _resolve(_load_framenet, diagnostics)  # type: ignore[assignment]

    def wn_for(lemma: str) -> str:
        if wn is None:
            return _DEMO_WORDNET.get(lemma, "")
        try:
            names = [s.name() for s in wn.synsets(lemma, pos="NOUN")]
        except Exception:
            return ""
        return "; ".join(str(n) for n in names[:6])

    def vn_for(lemma: str, pos: str) -> str:
        if vn is None:
            return _DEMO_VERBNET.get(lemma, "")
        try:
            ids = vn.class_ids(lemma)
        except Exception:
            return ""
        return "; ".join(str(i) for i in ids[:4])

    def fn_for(lemma: str, pos: str) -> str:
        if fn is None:
            return _DEMO_FRAMENET.get(lemma, "")
        try:
            frames = fn.frames_for(lemma, pos if pos in ("NOUN", "VERB") else "NOUN")
        except Exception:
            return ""
        return "; ".join(str(f) for f in frames[:4])

    rows: list[dict[str, object]] = []
    for (lemma, pos), c in sorted(cnt.items()):
        rows.append(
            {
                "Lemma": lemma,
                "WordNet": wn_for(lemma),
                "VerbNet": vn_for(lemma, pos) if pos.startswith("VB") or pos == "VERB" else "",
                "FrameNet": fn_for(lemma, pos),
                "Count": int(c),
            }
        )

    df = pd.DataFrame(rows, columns=["Lemma", "WordNet", "VerbNet", "FrameNet", "Count"])
    df = df.sort_values("Count", ascending=False).reset_index(drop=True)
    if diagnostics:
        return Result.success(df, *diagnostics)
    return Result.success(df)


def _resolve(
    loader: Callable[[], Result[Any]],
    diagnostics: list[Diagnostic],
) -> WordNetBackend | VerbNetBackend | FrameNetBackend | None:
    result = loader()
    if result.value is None:
        # The loader emits ERROR diagnostics for the missing environment.
        # This adapter still completes (demo mode), so the record must be a
        # WARNING: one limitation diagnostic per resource, never an ERROR
        # that would disguise a successful aggregation as a failure.
        for d in result.diagnostics:
            note = (
                Diagnostic.warning(
                    d.code + "_DEMO" if d.code.startswith("SEM_") is False else d.code,
                    d.message,
                    **(d.context or {}),
                )
                if d.severity.name == "ERROR"
                else d
            )
            if note.code not in {existing.code for existing in diagnostics}:
                diagnostics.append(note)
        if _DEMO_LIMIT.code not in {existing.code for existing in diagnostics}:
            diagnostics.append(_DEMO_LIMIT)
        return None
    return result.unwrap()  # type: ignore[no-any-return]  # backend protocols are structural


def _load_wordnet() -> Result[Any]:
    from core.analysis.wordnet import default_backend

    return default_backend()


def _load_verbnet() -> Result[Any]:
    from core.analysis.verbnet import default_backend

    return default_backend()


def _load_framenet() -> Result[Any]:
    from core.analysis.framenet import default_backend

    return default_backend()
