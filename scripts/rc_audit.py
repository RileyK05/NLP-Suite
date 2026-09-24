"""rc_audit - release-candidate consistency audit (FR-9.8, Gates A-E support).

Fast static checks (no pytest, no models): ledger review rows name files
that exist, the registry/migration/pyproject triple agrees, and the
required docs are present. Prints failures; exit 0 means consistent.
The full quality gate (pytest, ruff, mypy) still runs separately — this
audit catches drift between documents and code, which the gate cannot.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import tomllib
from contextlib import suppress
from pathlib import Path

__all__ = ["audit_capabilities", "audit_docs", "audit_ledger_files", "audit_triple", "main"]

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DOCS = (
    "docs/MIGRATION.md",
    "docs/PERFORMANCE.md",
    "docs/SECURITY.md",
    "docs/LICENSE_REVIEW.md",
    "docs/USABILITY_TEST.md",
    "docs/PCACE_DECOMPOSITION.md",
    "docs/INSTALL.md",
)


def audit_docs(root: Path = ROOT) -> list[str]:
    """Required release docs present."""
    return [f"missing doc: {name}" for name in REQUIRED_DOCS if not (root / name).is_file()]


def audit_ledger_files(root: Path = ROOT) -> list[str]:
    """Every `code` path named in a `review` ledger row exists on disk."""
    ledger = root / "docs" / "REPLACEMENT_LEDGER.md"
    try:
        text = ledger.read_text(encoding="utf-8")
    except OSError:
        return ["missing docs/REPLACEMENT_LEDGER.md"]
    failures: list[str] = []
    for line in text.splitlines():
        if "| review |" not in line:
            continue
        for candidate in re.findall(r"`((?:core|tools|app|scripts|tests)/[^`]+?)`", line):
            striped = candidate.split("::")[0].split("#")[0].rstrip("/")
            if striped.endswith((".csv", ".md", ".json", ".txt")):
                continue  # data/asset references, not code paths
            if not (root / striped).exists():
                failures.append(f"ledger names missing path: {candidate}")
    return failures


def audit_capabilities(root: Path = ROOT) -> list[str]:
    """Every capability a registry spec claims has a row in the ledger.

    The capability ID is the only join between a tool and its replacement
    record. A spec claiming an ID the ledger does not define means the tool is
    reporting coverage that nothing tracks, so it is a drift failure here.
    """
    ledger = root / "docs" / "REPLACEMENT_LEDGER.md"
    try:
        text = ledger.read_text(encoding="utf-8")
    except OSError:
        return ["missing docs/REPLACEMENT_LEDGER.md"]
    defined = set(re.findall(r"\|\s*(CAP-[A-Z]+-\d+)\s*\|", text))
    sys.path.insert(0, str(root))
    try:
        from core.profiler.registry import TOOL_REGISTRY
    except ImportError as exc:
        return [f"cannot import registry: {exc}"]
    finally:
        with suppress(ValueError):
            sys.path.remove(str(root))
    failures: list[str] = []
    for spec in TOOL_REGISTRY:
        for capability in spec.capability_ids:
            if capability not in defined:
                failures.append(f"{spec.name} claims {capability}, which has no ledger row")
    return failures


def audit_triple(root: Path = ROOT) -> list[str]:
    """Registry, migration guide, and pyproject entry points agree."""
    failures: list[str] = []
    sys.path.insert(0, str(root))
    try:
        from core.profiler.registry import TOOL_REGISTRY_EXCLUSIONS, tool_names
    except ImportError as exc:
        return [f"cannot import registry: {exc}"]
    finally:
        with suppress(ValueError):
            sys.path.remove(str(root))
    try:
        guide = (root / "docs" / "MIGRATION.md").read_text(encoding="utf-8")
    except OSError:
        return ["missing docs/MIGRATION.md"]
    try:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"unreadable pyproject.toml: {exc}"]
    scripts = pyproject["project"]["scripts"]
    for name in tool_names():
        if name not in guide:
            failures.append(f"registry tool missing from migration guide: {name}")
    cli_names = {path.stem for path in (root / "tools").glob("*.py") if path.stem not in {"_cli", "__init__"}}
    covered = set(tool_names()) | {name for name, _reason in TOOL_REGISTRY_EXCLUSIONS}
    for name in sorted(cli_names - covered):
        failures.append(f"CLI neither registered nor excluded: {name}")
    for script, target in scripts.items():
        module_name, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            failures.append(f"script {script!r} points at unimportable {module_name}")
            continue
        if not callable(getattr(module, attr, None)):
            failures.append(f"script {script!r} points at non-callable {target}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Release-candidate consistency audit")
    parser.parse_args(argv)
    failures = audit_docs() + audit_ledger_files() + audit_capabilities() + audit_triple()
    for failure in failures:
        print(f"audit: {failure}")
    print(f"audit: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
