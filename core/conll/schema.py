"""The CoNLL table contract.

ARCHITECTURE.md invariant **R9**: the CoNLL table is versioned and
name-addressed. Columns are reached through :class:`Col` by *name*; positional
indexing is banned. Every table ships a sidecar recording the schema version
and, critically, the **POS tagset**.

Why the tagset stamp exists: the legacy analyzer normalized Universal tags to
Penn in memory, then re-read the raw file and filtered Universal tags with
Penn-shaped patterns — producing silently empty outputs. Stamping the tagset
on the table makes "which tags are these?" answerable at load time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
from typing import Final, Literal

from core.result import Diagnostic, Result

__all__ = [
    "SCHEMA_VERSION",
    "CoNLLSchema",
    "Col",
    "PosTagset",
    "auto_generated_columns",
    "canonical_columns",
    "read_sidecar",
    "required_columns",
    "sidecar_path",
    "validate_columns",
    "write_sidecar",
]

SCHEMA_VERSION: Final[int] = 1

PosTagset = Literal["penn", "universal"]


class Col(Enum):
    """Every CoNLL column, by name. Never index a table by position."""

    ID = "ID"
    FORM = "Form"
    LEMMA = "Lemma"
    POS = "POS"
    NER = "NER"
    HEAD = "Head"
    DEPREL = "DepRel"
    DEPS = "Deps"
    CLAUSE_TAG = "Clause Tag"
    RECORD_ID = "Record ID"
    SENTENCE_ID = "Sentence ID"
    DOCUMENT_ID = "Document ID"
    DOCUMENT = "Document"

    def __str__(self) -> str:
        return self.value


# Canonical order. Every table is normalized to this layout regardless of which
# parser produced it, so a downstream reader never has to guess.
_CANONICAL: Final[tuple[Col, ...]] = (
    Col.ID,
    Col.FORM,
    Col.LEMMA,
    Col.POS,
    Col.NER,
    Col.HEAD,
    Col.DEPREL,
    Col.DEPS,
    Col.CLAUSE_TAG,
    Col.RECORD_ID,
    Col.SENTENCE_ID,
    Col.DOCUMENT_ID,
    Col.DOCUMENT,
)

# Present in every valid table.
_REQUIRED: Final[frozenset[Col]] = frozenset(
    {
        Col.ID,
        Col.FORM,
        Col.LEMMA,
        Col.POS,
        Col.NER,
        Col.HEAD,
        Col.DEPREL,
        Col.SENTENCE_ID,
        Col.DOCUMENT_ID,
        Col.DOCUMENT,
    }
)

# Filled in when the parser did not supply them.
_AUTO_GENERATED: Final[frozenset[Col]] = frozenset({Col.RECORD_ID, Col.DEPS, Col.CLAUSE_TAG})


def canonical_columns() -> tuple[str, ...]:
    """The canonical column names, in canonical order."""
    return tuple(c.value for c in _CANONICAL)


def required_columns() -> frozenset[str]:
    """Columns a table must have to be analyzable."""
    return frozenset(c.value for c in _REQUIRED)


def auto_generated_columns() -> frozenset[str]:
    """Columns we synthesize when the parser omitted them."""
    return frozenset(c.value for c in _AUTO_GENERATED)


@dataclass(frozen=True, slots=True)
class CoNLLSchema:
    """What a CoNLL table means: its version, its tagset, and who produced it."""

    parser: str = ""
    parser_version: str = ""
    pos_tagset: PosTagset = "penn"
    schema_version: int = SCHEMA_VERSION
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    extra_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported CoNLL schema version {self.schema_version}; this build reads {SCHEMA_VERSION}"
            )
        if self.pos_tagset not in ("penn", "universal"):
            raise ValueError(f"unknown pos_tagset {self.pos_tagset!r}; expected 'penn' or 'universal'")

    @property
    def columns(self) -> tuple[str, ...]:
        """The canonical columns plus any extras this table carries."""
        return canonical_columns() + tuple(self.extra_columns)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pos_tagset": self.pos_tagset,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "created": self.created,
            "columns": list(self.columns),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CoNLLSchema:
        extras = payload.get("columns")
        canonical = set(canonical_columns())
        extra_columns = tuple(str(c) for c in extras if str(c) not in canonical) if isinstance(extras, list) else ()
        return cls(
            parser=str(payload.get("parser", "")),
            parser_version=str(payload.get("parser_version", "")),
            pos_tagset=str(payload.get("pos_tagset", "penn")),  # type: ignore[arg-type]
            schema_version=_coerce_int(payload.get("schema_version", SCHEMA_VERSION), "schema_version"),
            created=str(payload.get("created", "")),
            extra_columns=extra_columns,
        )


def _coerce_int(value: object, field_name: str) -> int:
    """Read an integer out of an untrusted JSON payload, or fail loudly."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}")
    return int(value)


def validate_columns(columns: list[str]) -> Result[tuple[str, ...]]:
    """Check a table's header row, reporting every problem rather than the first.

    Missing auto-generated columns are fine (we synthesize them); missing
    required columns are not.
    """
    seen = [str(c) for c in columns]
    missing = sorted(required_columns() - set(seen))
    diagnostics: list[Diagnostic] = []

    if missing:
        diagnostics.append(
            Diagnostic.error(
                "CONLL_MISSING_COLUMN",
                f"table is missing required CoNLL column(s): {missing}",
                missing=missing,
            )
        )
    if len(seen) != len(set(seen)):
        duplicates = sorted({c for c in seen if seen.count(c) > 1})
        diagnostics.append(
            Diagnostic.error(
                "CONLL_DUPLICATE_COLUMN",
                f"table has duplicate column name(s): {duplicates}",
                duplicates=duplicates,
            )
        )

    unknown = [c for c in seen if c not in set(canonical_columns())]
    if unknown:
        diagnostics.append(
            Diagnostic.info(
                "CONLL_EXTRA_COLUMN",
                f"table carries non-canonical column(s): {sorted(unknown)}",
                extra=sorted(unknown),
            )
        )

    if any(d.severity.value == "ERROR" for d in diagnostics):
        return Result.failure(*diagnostics)
    return Result.success(tuple(seen), *diagnostics)


def sidecar_path(table_path: Path) -> Path:
    """``foo.csv`` -> ``foo.csv.schema.json``."""
    table_path = Path(table_path)
    return table_path.with_name(table_path.name + ".schema.json")


def write_sidecar(table_path: Path, schema: CoNLLSchema) -> Result[Path]:
    """Write the sidecar next to its table. The only place a sidecar is written."""
    target = sidecar_path(table_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(schema.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        return Result.failure(
            Diagnostic.error("SIDECAR_WRITE_FAILED", f"could not write {target}: {exc}", path=str(target))
        )
    return Result.success(target)


def read_sidecar(table_path: Path) -> Result[CoNLLSchema]:
    """Read a table's sidecar. A missing sidecar is an error, not a default.

    The legacy suite inferred structure from filenames and position; a table
    without a schema has no defined meaning, so we refuse to guess.
    """
    source = sidecar_path(table_path)
    if not source.is_file():
        return Result.failure(
            Diagnostic.error(
                "SIDECAR_MISSING",
                f"no schema sidecar for {Path(table_path).name}; expected {source.name}",
                path=str(source),
            )
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Result.failure(
            Diagnostic.error("SIDECAR_UNREADABLE", f"could not read {source}: {exc}", path=str(source))
        )
    if not isinstance(payload, dict):
        return Result.failure(
            Diagnostic.error("SIDECAR_MALFORMED", f"sidecar {source} is not a JSON object", path=str(source))
        )
    try:
        return Result.success(CoNLLSchema.from_dict(payload))
    except (ValueError, TypeError) as exc:
        return Result.failure(
            Diagnostic.error("SIDECAR_INVALID", f"sidecar {source} is not a valid schema: {exc}", path=str(source))
        )
