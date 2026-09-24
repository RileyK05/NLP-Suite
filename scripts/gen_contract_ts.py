"""Render desktop_backend/schemas.py into desktop/src/contract.ts (R-C6).

Run after changing any contract model:

    python scripts/gen_contract_ts.py

The output is checked in and covered by tests/test_contract_parity.py, which
regenerates and diffs. The desktop must never hand-write these shapes: the
engine and the interface drifted silently for years over tool names, parameter
labels and chart kinds, and drift must fail a test rather than a user.

The mapping is deliberately dumb — field types only — because a clever mapper
is a second thing to keep in step. Pydantic remains the definition.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types
import typing

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import BaseModel  # noqa: E402

from desktop_backend.schemas import CONTRACT_MODELS  # noqa: E402

TARGET = ROOT / "desktop" / "src" / "contract.ts"

_PRIMITIVES = {
    "str": "string",
    "int": "number",
    "float": "number",
    "bool": "boolean",
}


def _field_ts(annotation: object, name: str) -> str:
    """One TS type for one Python annotation. Fails loudly on the unknown."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        return " | ".join(repr(value) if isinstance(value, str) else str(value) for value in args)
    if origin in (types.UnionType, typing.Union):
        parts = [arg for arg in args if arg is not type(None)]
        rendered = " | ".join(_field_ts(part, name) for part in parts)
        return f"{rendered} | null" if len(parts) < len(args) else rendered
    if origin in (dict, dict):
        value = args[1] if args else "object"
        mapped = _field_ts(value, name) if args else "unknown"
        return f"Record<string, {mapped}>"
    if origin in (list, list):
        return f"{_field_ts(args[0], name)}[]"
    if annotation in (str, int, float, bool):
        return _PRIMITIVES[annotation.__name__]
    if annotation is object:
        return "unknown"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__
    raise ValueError(f"contract.ts generator: unhandled type {annotation!r} on {name}")


def render() -> str:
    """The full contract.ts body from CONTRACT_MODELS, in declaration order."""
    chunks: list[str] = [
        "/** GENERATED FILE — do not edit.",
        " *",
        " * Rendered from desktop_backend/schemas.py by scripts/gen_contract_ts.py",
        " * (communication contract R-C6). Change the models, re-run the script,",
        " * and let tests/test_contract_parity.py confirm the two agree.",
        " */",
        "",
    ]
    for model in CONTRACT_MODELS:
        chunks.append(f"export interface {model.__name__} {{")
        for field_name, field in model.model_fields.items():
            annotation = field.annotation
            # A defaulted field is always in model_dump(); only a None default
            # also brings null (and may be omitted by request senders), so it
            # is optional+nullable and everything else is simply required.
            ts = _field_ts(annotation or "str", field_name)
            if field.default is None:
                if not ts.endswith("| null"):
                    ts = f"{ts} | null"
                key = f"{field_name}?"
            else:
                key = field_name
            comment = field.description or ""
            if comment:
                chunks.append(f"  /** {comment} */")
            chunks.append(f"  {key}: {ts};")
        chunks.append("}")
        chunks.append("")
    return "\n".join(chunks)


def main() -> int:
    TARGET.write_text(render(), encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
