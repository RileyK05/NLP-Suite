"""SPEC_ONLY registry (FR-1.6) — Gate B evidence where no golden can exist.

Some legacy paths are unreachable here (a Java server, an isolated Python
3.8 env, a model download) or broken upstream. Capturing a "golden" from
such a path would record our own failure diagnostic, not legacy behavior —
so instead each such capability gets a SPEC_ONLY entry: why no valid golden
can be produced, which documentation and domain sources define the behavior,
the contract the real backend must satisfy, and the substitute evidence.

Rules:

* An entry never blesses a stub as complete. The stub stays a stub (its
  parse fails loudly); the entry only records what "done" means later.
* Adding a capability here requires flipping its ledger Oracle column to
  ``S`` and extending the scope test in ``tests/test_spec_only.py`` in the
  same change — registry, ledger, and test move together or not at all.
* The packet review that accepts an entry is its approval. There is no
  per-entry status field to drift out of sync with the ledger.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from core.result import Diagnostic

__all__ = [
    "SPEC_ONLY_REGISTRY",
    "SpecCategory",
    "SpecOnlyEntry",
    "check_consistency",
    "get_spec",
    "spec_capability_ids",
]


class SpecCategory(Enum):
    """Why no valid golden can be produced."""

    UNREACHABLE = "UNREACHABLE"  # needs an env absent from test and CI
    CAPTURE_PENDING = "CAPTURE_PENDING"  # a bounded owner-approved capture is possible
    BROKEN = "BROKEN"  # the legacy path itself never worked as a suite tool


@dataclass(frozen=True, slots=True)
class SpecOnlyEntry:
    """The golden-substitute for one capability."""

    capability_id: str
    category: SpecCategory
    reason: str  # why a valid golden cannot be produced
    legacy_refs: tuple[str, ...]  # exact legacy files, repo-relative
    basis: tuple[str, ...]  # docs/formulas the behavior is defined from
    contract: str  # what the real backend must do instead
    evidence: tuple[str, ...]  # substitute evidence (tests, property checks)


SPEC_ONLY_REGISTRY: tuple[SpecOnlyEntry, ...] = (
    SpecOnlyEntry(
        capability_id="CAP-PARSE-03",
        # C6-16: the reviewer was right — a bounded probe showed Java 21.0.9
        # + CoreNLP 4.5.8 (installed at C:/Users/moomi/NLP_Software/) serves
        # requests on localhost (HTTP 200, JSON with parse + dependencies
        # for "The cat sat."). UNREACHABLE was wrong; the blocker is only
        # that CI/offline environments lack the server. A REAL golden can be
        # captured against this installation with owner authorization, so
        # this entry is downgraded to a capture-plan placeholder until
        # FR-1.4 runs with the recorded environment.
        category=SpecCategory.CAPTURE_PENDING,
        reason=(
            "Golden capture REQUIRES the local CoreNLP 4.5.8 install (present "
            "on the dev machine: Java 21.0.9 + stanford-corenlp-4.5.8 with "
            "models jar; probe returned HTTP 200 with a constituency parse "
            "for 'The cat sat.' on 2026-09-03). Ordinary test/CI environments "
            "do not have the server, so automated tests still cannot depend "
            "on it — but the correct next step is a bounded, read-only "
            "capture under FR-1.4 with the owner's go-ahead, NOT a claim "
            "that no golden can exist."
        ),
        legacy_refs=(
            "src/Stanford_CoreNLP_util.py",
            "src/Stanford_CoreNLP_clause_util.py",
            "src/Stanford_CoreNLP_SVO_enhanced_dependencies_util.py",
            "src/Stanford_CoreNLP_tags_util.py",
            "src/Stanford_CoreNLP_port_util.py",
            "lib/CoreNLP_enhanced_dependencies",
        ),
        basis=(
            "Stanford CoreNLP server API: annotators tokenize/ssplit/pos/lemma/parse/depparse over HTTP",
            "legacy TIPS_NLP_Stanford CoreNLP CoNLL table.pdf (column contract for the CoNLL output)",
            "legacy src/Stanford_CoreNLP_util.py output columns (canonical names the conversion must hit)",
        ),
        contract=(
            "The FR-5.2 backend must be a lazy optional extra speaking to a CoreNLP "
            "server (default http://localhost:9000) with a startup health check, "
            "bounded request timeouts, and conversion into the canonical name-addressed "
            "CoNLL table plus schema sidecar. Until FR-5.2 lands, parse() fails with "
            "PIPELINE_NOT_CONFIGURED naming the server URL — never a blank table."
        ),
        evidence=(
            "tests/test_backends.py::TestCoreNLP (fail-big: build succeeds, parse requires the server)",
            "FR-5.2 server-backed model integration tests, marked to skip offline",
        ),
    ),
    SpecOnlyEntry(
        capability_id="CAP-PARSE-04",
        category=SpecCategory.UNREACHABLE,
        reason=(
            "Legacy SRL runs inside an isolated conda env (nlp_srl, Python 3.8, torch 1.7, "
            "allennlp 1.2) with transformer_srl and a model tarball fetched from Dropbox — "
            "per the src/SRL_worker.py header it executes as a subprocess there, not in the "
            "suite env. None of that exists offline, so no golden can be captured."
        ),
        legacy_refs=(
            "src/SRL_worker.py",
            "src/SRL_main.py",
            "src/SRL_util.py",
            "setup_SRL.py",
            "lib/SRL",
        ),
        basis=(
            "src/SRL_worker.py header and HEADERS: exact CSV schema (Document, Date, "
            "Sentence ID, Sentence, Predicate, Frame, VerbNet class, FrameNet frame, "
            "PropBank ARG columns, Refined roles, Description)",
            "PropBank ARG0/ARG1/ARGM-LOC/TMP/MNR/CAU role semantics via the worker ROLE_COLUMNS map",
            "Palmer et al. SemLink pb-vn2 / vn-fn2 linking (fetched by setup_SRL.py) for VerbNet/FrameNet columns",
        ),
        contract=(
            "The FR-5.3 backend must invoke the isolated Python 3.8 worker through the one "
            "subprocess wrapper with timeouts and safe argv, and emit the worker HEADERS "
            "schema including Frame, VerbNet class, and FrameNet frame per predicate. Until "
            "FR-5.3 lands, parse() fails with PIPELINE_NOT_CONFIGURED naming the isolated "
            "env — never an empty table."
        ),
        evidence=(
            "tests/test_backends.py::TestSRL (fail-big: English-only build, parse requires the isolated env)",
            "FR-5.3 worker-backed integration tests with a recorded worker response, marked to skip offline",
        ),
    ),
)


def get_spec(capability_id: str) -> SpecOnlyEntry | None:
    """Return the SPEC_ONLY entry for *capability_id*, or None."""
    for entry in SPEC_ONLY_REGISTRY:
        if entry.capability_id == capability_id:
            return entry
    return None


def spec_capability_ids() -> tuple[str, ...]:
    """Capability IDs covered by the registry, in registry order."""
    return tuple(entry.capability_id for entry in SPEC_ONLY_REGISTRY)


def check_consistency(
    entries: Sequence[SpecOnlyEntry],
    s_row_ids: set[str] | frozenset[str],
) -> tuple[Diagnostic, ...]:
    """Check registry entries against the ledger's ``S``-marked capability IDs.

    Unknown references, missing entries, and duplicates are ERROR diagnostics;
    ledger rows without entries but also without an ``S`` mark are fine — they
    are golden-tracked, which is none of this registry's business.
    """
    diags: list[Diagnostic] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in entries:
        if entry.capability_id in seen:
            duplicates.add(entry.capability_id)
        seen.add(entry.capability_id)
    if duplicates:
        diags.append(
            Diagnostic.error(
                "SPEC_ONLY_DUPLICATE",
                f"duplicate SPEC_ONLY entries for {sorted(duplicates)}",
                capability_ids=sorted(duplicates),
            )
        )
    unknown = sorted(seen - set(s_row_ids))
    if unknown:
        diags.append(
            Diagnostic.error(
                "SPEC_ONLY_UNKNOWN_CAPABILITY",
                f"SPEC_ONLY entries reference capabilities with no ledger S mark: {unknown}",
                capability_ids=unknown,
            )
        )
    missing = sorted(set(s_row_ids) - seen)
    if missing:
        diags.append(
            Diagnostic.error(
                "SPEC_ONLY_MISSING_ENTRY",
                f"ledger S-marked capabilities have no SPEC_ONLY entry: {missing}",
                capability_ids=missing,
            )
        )
    return tuple(diags)
