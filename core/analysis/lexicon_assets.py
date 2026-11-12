"""Shared lexicon-asset resolution for the lexicon-based sentiment tools.

VADER/ANEW, SentiWordNet/Hedonometer and the NRC emotion lexicon each need a
registry asset (a full lexicon file) or an explicit override. The resolution
rules -- a path override wins, otherwise the registry asset, with a diagnostic
that names the missing asset and how to get it -- live here once.
"""

from __future__ import annotations

from pathlib import Path

from core.result import Diagnostic

__all__ = ["resolve_asset_path"]


def resolve_asset_path(name: str, override: object) -> tuple[Path | None, list[Diagnostic]]:
    """Resolve a lexicon asset, honouring an explicit path override first."""
    if override is not None:
        if not isinstance(override, (str, Path)):
            return None, [
                Diagnostic.error(
                    "LEXICON_NOT_A_PATH", f"lexicon override must be a path, got {type(override).__name__}"
                )
            ]
        target = Path(override)
        if not target.is_file():
            return None, [
                Diagnostic.error("LEXICON_NOT_FOUND", f"lexicon file not found at {target}", path=str(target))
            ]
        return target, []
    from core.assets.registry import default_registry

    resolved = default_registry().path(name)
    if resolved.value is None:
        hint = next((d for d in resolved.diagnostics if d.code == "ASSET_NOT_USABLE"), None)
        status = hint.context.get("status", "missing") if hint is not None else "missing"
        return None, [
            Diagnostic.error(
                f"{name.upper().replace('-', '_')}_ASSET_{status}",
                f"{name} asset is {status}; copy it from the legacy lib/sentimentLib "
                "or install the documented source (see the asset registry)",
                asset=name,
            )
        ]
    return resolved.unwrap(), list(resolved.diagnostics)
