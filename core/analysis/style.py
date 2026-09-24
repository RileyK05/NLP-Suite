"""Abstract/concrete + iconicity style analysis (FR-2.8).

Ports the two legacy ``style_analysis_*`` routines onto the shared parse
table:

- concreteness: Brysbaert et al. 2014 norms (``Word`` -> ``Conc.M``),
  alpha-only tokens minus English stopwords, lemma matching;
- iconicity: Winter et al. 2024 norms (``word`` -> ``rating`` /
  ``rating_sd``), alpha-only tokens, lemma matching, plus the
  most-iconic-words list (defaults ``min_rating=5.0``,
  ``max_rating_sd=2.0``).

Both lexicons are user-supplied registry assets (no bytes vendored);
tests inject tiny hand-built mappings. Per-sentence rows mirror the
legacy columns (mean/median/stdev, found/all words, sentence, document)
and sentences with no scored words are skipped, as in the legacy.

Intentional differences from the legacy (all defect-grade, none silent):

- one shared parse instead of a Stanza re-parse per sentence and per word;
- stopwords come from scikit-learn's ``ENGLISH_STOP_WORDS`` (already a
  core dependency) instead of the legacy ``lib/wordLists/stopwords.txt``;
- empty input files fail loudly (``STYLE_EMPTY_DOCUMENT``) instead of a
  Tkinter error popup with no output row.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import statistics

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "concreteness",
    "iconic_words",
    "iconicity",
    "load_concreteness_lexicon",
    "load_iconicity_lexicon",
]

_CONC_COLUMNS: tuple[str, ...] = (
    "Concreteness Mean",
    "Concreteness Median",
    "Concreteness SD",
    "Words Found",
    "Coverage %",
    "Found Words",
    "All Words",
    "Sentence ID",
    "Sentence",
    "Document ID",
    "Document",
)
_ICON_COLUMNS: tuple[str, ...] = (
    "Iconicity Mean",
    "Iconicity Median",
    "Iconicity SD",
    "Words Found",
    "Coverage %",
    "Found Words",
    "All Words",
    "Sentence ID",
    "Sentence",
    "Document ID",
    "Document",
)
_ICONIC_COLUMNS: tuple[str, ...] = ("Word", "Rating", "Document ID", "Document")


def _stopwords() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(ENGLISH_STOP_WORDS)


def _resolve_asset_path(name: str, override: object, legacy_lib: str) -> tuple[Path | None, list[Diagnostic]]:
    if override is not None:
        if not isinstance(override, (str, Path)):
            return None, [
                Diagnostic.error(
                    "STYLE_LEXICON_NOT_A_PATH", f"lexicon override must be a path, got {type(override).__name__}"
                )
            ]
        target = Path(override)
        if not target.is_file():
            return None, [
                Diagnostic.error("STYLE_LEXICON_NOT_FOUND", f"lexicon file not found at {target}", path=str(target))
            ]
        return target, []
    from core.assets.registry import default_registry

    resolved = default_registry().path(name)
    if resolved.value is None:
        hint = next((d for d in resolved.diagnostics if d.code == "ASSET_NOT_USABLE"), None)
        status = hint.context.get("status", "missing") if hint is not None else "missing"
        return None, [
            Diagnostic.error(
                f"STYLE_{name.upper().replace('-', '_')}_{status}",
                f"{name} asset is {status}; copy it from the legacy {legacy_lib} "
                "or install the documented source (see the asset registry)",
                asset=name,
            )
        ]
    return resolved.unwrap(), list(resolved.diagnostics)


def load_concreteness_lexicon(path: str | Path) -> Result[dict[str, float]]:
    """Load a Brysbaert-format CSV (``Word``, ``Conc.M``) to {word: mean}; bad rows skipped."""
    import csv

    try:
        handle = open(path, encoding="utf-8-sig", newline="")  # noqa: SIM115
    except OSError as exc:
        return Result.failure(
            Diagnostic.error("STYLE_CONCRETENESS_UNREADABLE", f"cannot read concreteness CSV {path}: {exc}")
        )
    table: dict[str, float] = {}
    skipped = 0
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Word" not in reader.fieldnames or "Conc.M" not in reader.fieldnames:
            return Result.failure(
                Diagnostic.error(
                    "STYLE_CONCRETENESS_BAD_COLUMNS",
                    f"concreteness CSV needs 'Word' and 'Conc.M', got {reader.fieldnames}",
                )
            )
        for row in reader:
            try:
                word = str(row["Word"]).strip().lower()
                score = round(float(str(row["Conc.M"])), 2)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if word:
                table[word] = score
    diags = (
        [Diagnostic.warning("STYLE_CONCRETENESS_SKIPPED_ROWS", f"skipped {skipped} malformed row(s)")]
        if skipped
        else []
    )
    return Result.success(table, *diags)


def load_iconicity_lexicon(path: str | Path) -> Result[dict[str, tuple[float, float]]]:
    """Load a Winter-format CSV (``word``, ``rating``, ``rating_sd``); bad rows skipped."""
    import csv

    try:
        handle = open(path, encoding="utf-8-sig", newline="")  # noqa: SIM115
    except OSError as exc:
        return Result.failure(
            Diagnostic.error("STYLE_ICONICITY_UNREADABLE", f"cannot read iconicity CSV {path}: {exc}")
        )
    table: dict[str, tuple[float, float]] = {}
    skipped = 0
    with handle:
        reader = csv.DictReader(handle)
        if (
            reader.fieldnames is None
            or "word" not in reader.fieldnames
            or "rating" not in reader.fieldnames
            or "rating_sd" not in reader.fieldnames
        ):
            return Result.failure(
                Diagnostic.error(
                    "STYLE_ICONICITY_BAD_COLUMNS",
                    f"iconicity CSV needs 'word', 'rating' and 'rating_sd', got {reader.fieldnames}",
                )
            )
        for row in reader:
            try:
                word = str(row["word"]).strip().lower()
                rating = round(float(str(row["rating"])), 2)
                sd_raw = str(row["rating_sd"]).strip()
                sd = round(float(sd_raw), 3) if sd_raw and sd_raw.lower() != "nan" else 0.0
            except (TypeError, ValueError):
                skipped += 1
                continue
            if word:
                table[word] = (rating, sd)
    diags = (
        [Diagnostic.warning("STYLE_ICONICITY_SKIPPED_ROWS", f"skipped {skipped} malformed row(s)")] if skipped else []
    )
    return Result.success(table, *diags)


def _check(frame: pd.DataFrame, field: Col) -> list[Diagnostic]:
    if field not in (Col.FORM, Col.LEMMA):
        return [Diagnostic.error("STYLE_BAD_FIELD", f"field must be Form or Lemma, got {field.value!r}")]
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return list(checked.diagnostics)
    for column in (field.value, Col.SENTENCE_ID.value, Col.DOCUMENT_ID.value):
        if column not in frame.columns:
            return [Diagnostic.error("STYLE_MISSING_COLUMN", f"parse table needs column {column!r}")]
    return []


def _summarize(scores: list[float]) -> tuple[float, float, float]:
    mean = round(float(statistics.mean(scores)), 2)
    median = round(float(statistics.median(scores)), 2)
    sd = round(float(statistics.stdev(scores)), 2) if len(scores) > 1 else 0.0
    return mean, median, sd


def concreteness(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    lexicon: Mapping[str, float] | str | Path | None = None,
) -> Result[pd.DataFrame]:
    """Per-sentence Brysbaert concreteness over alpha tokens minus stopwords."""
    diags = _check(frame, field)
    if diags:
        return Result.failure(*diags)
    table: Mapping[str, float] | None = lexicon if isinstance(lexicon, Mapping) else None
    if table is None:
        target, asset_diags = _resolve_asset_path("brysbaert-concreteness", lexicon, "lib/concretenessLib")
        diags.extend(asset_diags)
        if target is None:
            return Result.failure(*diags)
        loaded = load_concreteness_lexicon(target)
        if loaded.value is None:
            return Result.failure(*diags, *loaded.diagnostics)
        diags.extend(loaded.diagnostics)
        table = loaded.unwrap()
    if frame.empty:
        return Result.success(pd.DataFrame(columns=list(_CONC_COLUMNS)), *diags)
    stops = _stopwords()
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for (doc_id, sent_id), group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        forms = [str(v) for v in group[Col.FORM.value].tolist()]
        all_words: list[str] = []
        found: list[str] = []
        scores: list[float] = []
        for raw in group[field.value].tolist():
            word = str(raw).lower()
            if not word.isalpha() or word in stops:
                continue
            all_words.append(word)
            score = table.get(word)
            if score is not None:
                found.append(f"({word}, {score})")
                scores.append(score)
        if not scores or not all_words:
            continue
        mean, median, sd = _summarize(scores)
        if mean == 0 or median == 0:
            continue
        rows.append(
            {
                "Concreteness Mean": mean,
                "Concreteness Median": median,
                "Concreteness SD": sd,
                "Words Found": f"{len(found)} out of {len(all_words)}",
                "Coverage %": f"{100 * round(len(found) / len(all_words), 2)}%",
                "Found Words": ", ".join(found),
                "All Words": ", ".join(all_words),
                "Sentence ID": sent_id,
                "Sentence": " ".join(forms),
                "Document ID": doc_id,
                "Document": str(group[doc_col].iloc[0]) if doc_col is not None else "",
            }
        )
    return Result.success(pd.DataFrame(rows, columns=list(_CONC_COLUMNS)), *diags)


def iconicity(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    lexicon: Mapping[str, tuple[float, float]] | str | Path | None = None,
) -> Result[pd.DataFrame]:
    """Per-sentence Winter iconicity over alpha tokens (stopwords kept, as in the legacy)."""
    diags = _check(frame, field)
    if diags:
        return Result.failure(*diags)
    table: Mapping[str, tuple[float, float]] | None = lexicon if isinstance(lexicon, Mapping) else None
    if table is None:
        target, asset_diags = _resolve_asset_path("iconicity-ratings", lexicon, "lib/iconicityLib")
        diags.extend(asset_diags)
        if target is None:
            return Result.failure(*diags)
        loaded = load_iconicity_lexicon(target)
        if loaded.value is None:
            return Result.failure(*diags, *loaded.diagnostics)
        diags.extend(loaded.diagnostics)
        table = loaded.unwrap()
    if frame.empty:
        return Result.success(pd.DataFrame(columns=list(_ICON_COLUMNS)), *diags)
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for (doc_id, sent_id), group in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        forms = [str(v) for v in group[Col.FORM.value].tolist()]
        all_words: list[str] = []
        found: list[str] = []
        scores: list[float] = []
        for raw in group[field.value].tolist():
            word = str(raw).lower()
            if not word.isalpha():
                continue
            all_words.append(word)
            hit = table.get(word)
            if hit is None:
                continue
            found.append(f"({word}, {hit[0]})")
            scores.append(hit[0])
        if not scores or not all_words:
            continue
        mean, median, sd = _summarize(scores)
        if mean == 0 or median == 0:
            continue
        rows.append(
            {
                "Iconicity Mean": mean,
                "Iconicity Median": median,
                "Iconicity SD": sd,
                "Words Found": f"{len(found)} out of {len(all_words)}",
                "Coverage %": f"{100 * round(len(found) / len(all_words), 2)}%",
                "Found Words": ", ".join(found),
                "All Words": ", ".join(all_words),
                "Sentence ID": sent_id,
                "Sentence": " ".join(forms),
                "Document ID": doc_id,
                "Document": str(group[doc_col].iloc[0]) if doc_col is not None else "",
            }
        )
    return Result.success(pd.DataFrame(rows, columns=list(_ICON_COLUMNS)), *diags)


def iconic_words(
    frame: pd.DataFrame,
    *,
    field: Col = Col.LEMMA,
    lexicon: Mapping[str, tuple[float, float]] | str | Path | None = None,
    min_rating: float = 5.0,
    max_rating_sd: float = 2.0,
) -> Result[pd.DataFrame]:
    """Most-iconic words per document (rating > min_rating, sd < max_rating_sd)."""
    diags = _check(frame, field)
    if diags:
        return Result.failure(*diags)
    table: Mapping[str, tuple[float, float]] | None = lexicon if isinstance(lexicon, Mapping) else None
    if table is None:
        target, asset_diags = _resolve_asset_path("iconicity-ratings", lexicon, "lib/iconicityLib")
        diags.extend(asset_diags)
        if target is None:
            return Result.failure(*diags)
        loaded = load_iconicity_lexicon(target)
        if loaded.value is None:
            return Result.failure(*diags, *loaded.diagnostics)
        diags.extend(loaded.diagnostics)
        table = loaded.unwrap()
    if frame.empty:
        return Result.success(pd.DataFrame(columns=list(_ICONIC_COLUMNS)), *diags)
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby(Col.DOCUMENT_ID.value, sort=False):
        doc = str(group[doc_col].iloc[0]) if doc_col is not None else ""
        for raw in group[field.value].tolist():
            word = str(raw).lower()
            if not word.isalpha():
                continue
            hit = table.get(word)
            if hit is None:
                continue
            rating, sd = hit
            if rating > min_rating and sd < max_rating_sd:
                rows.append({"Word": word, "Rating": rating, "Document ID": doc_id, "Document": doc})
    return Result.success(pd.DataFrame(rows, columns=list(_ICONIC_COLUMNS)), *diags)
