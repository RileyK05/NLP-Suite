"""R-C6: the desktop's contract types are generated, and the generator is current.

The engine and the interface drifted silently over tool names, parameter
labels and chart kinds — each side looked fine alone. The contract now has one
definition (``desktop_backend/schemas.py``) and one generated mirror
(``desktop/src/contract.ts``), and this file is the ratchet: change a model
without re-running ``scripts/gen_contract_ts.py`` and the suite fails with the
command to run.

A second, independent check reads both files as data and compares field sets,
so a generator bug that emits *something* still cannot emit the wrong shape
without being caught.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

from desktop_backend.schemas import CONTRACT_MODELS

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "desktop" / "src" / "contract.ts"


def test_generated_contract_is_committed_and_current() -> None:
    """Regenerating must be a no-op. A stale contract.ts is drift in progress."""
    assert GENERATED.is_file(), "desktop/src/contract.ts is missing; run python scripts/gen_contract_ts.py"
    before = GENERATED.read_text(encoding="utf-8")
    subprocess.run([sys.executable, "scripts/gen_contract_ts.py"], cwd=ROOT, check=True, capture_output=True)
    after = GENERATED.read_text(encoding="utf-8")
    assert before == after, (
        "desktop/src/contract.ts is stale against desktop_backend/schemas.py. Run: python scripts/gen_contract_ts.py"
    )


def _ts_interfaces(text: str) -> dict[str, set[str]]:
    """Interface name -> field names, parsed out of the generated TypeScript."""
    found: dict[str, set[str]] = {}
    for match in re.finditer(r"export interface (\w+) \{(.*?)\n\}", text, re.S):
        name, body = match.group(1), match.group(2)
        fields = set(re.findall(r"^\s{2}(\w+)\??:", body, re.M))
        found[name] = fields
    return found


def test_every_contract_model_reaches_the_desktop() -> None:
    """Same fields on both sides, or a caller reads a name the engine never sends."""
    ts = _ts_interfaces(GENERATED.read_text(encoding="utf-8"))
    for model in CONTRACT_MODELS:
        assert model.__name__ in ts, f"{model.__name__} never reaches desktop/src/contract.ts"
        python_fields = set(model.model_fields)
        assert ts[model.__name__] == python_fields, (
            f"{model.__name__} disagrees across the boundary: "
            f"python={sorted(python_fields)} ts={sorted(ts[model.__name__])}"
        )


def test_the_racy_envelope_carries_request_identity() -> None:
    """R-C2 lives or dies on these two fields being in both envelopes.

    Without them a superseded answer cannot be recognised, which is the whole
    of the "setting changed while running" bug class.
    """
    request = CONTRACT_MODELS[[m.__name__ for m in CONTRACT_MODELS].index("AnalyseRequest")]
    response = CONTRACT_MODELS[[m.__name__ for m in CONTRACT_MODELS].index("AnalyseResponse")]
    assert "request_id" in request.model_fields
    assert "request_id" in response.model_fields
    assert "snapshot_id" in request.model_fields, "R-C4: analyse must name the corpus it asks about"
