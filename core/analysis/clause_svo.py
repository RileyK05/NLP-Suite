"""Clause and SVO extraction from a canonical CoNLL frame."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = ["ClauseRow", "ClauseSvoResult", "SvoRow", "clause_frequencies", "extract_svo"]

_LEGACY_CLAUSE_TAGS = frozenset(["S", "SBAR", "SBARQ", "SQ", "SINV", "NP", "VP", "ADJP", "ADVP", "PP"])


@dataclass(frozen=True, slots=True)
class ClauseRow:
    tag: str
    count: int


@dataclass(frozen=True, slots=True)
class SvoRow:
    subject: str
    verb: str
    obj: str
    sentence_id: object
    document_id: object
    document: str


@dataclass(frozen=True, slots=True)
class ClauseSvoResult:
    clauses: tuple[ClauseRow, ...]
    svos: tuple[SvoRow, ...]

    def clauses_frame(self) -> pd.DataFrame:
        if not self.clauses:
            return pd.DataFrame(columns=["Clause Tag", "Count"])
        return pd.DataFrame([{"Clause Tag": r.tag, "Count": r.count} for r in self.clauses])

    def svo_frame(self) -> pd.DataFrame:
        if not self.svos:
            return pd.DataFrame(columns=["Subject", "Verb", "Object", "Sentence ID", "Document ID", "Document"])
        return pd.DataFrame(
            [
                {
                    "Subject": r.subject,
                    "Verb": r.verb,
                    "Object": r.obj,
                    "Sentence ID": r.sentence_id,
                    "Document ID": r.document_id,
                    "Document": r.document,
                }
                for r in self.svos
            ]
        )


def clause_frequencies(frame: pd.DataFrame) -> Result[tuple[ClauseRow, ...]]:
    """Count occurrences of each clause tag in the Clause Tag column."""
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[tuple[ClauseRow, ...]](None, checked.diagnostics)

    if Col.CLAUSE_TAG.value not in frame.columns:
        return Result.failure(
            Diagnostic.error(
                "CLAUSE_MISSING_COLUMN", f"missing column {Col.CLAUSE_TAG.value!r}", column=Col.CLAUSE_TAG.value
            )
        )

    if frame.empty:
        return Result.success(())

    series = frame[Col.CLAUSE_TAG.value].astype(str).str.strip()
    # Empty string, "_" or "nan" means "no clause annotation" — not a tag.
    series = series[~series.isin(["", "_", "nan", "None"])]
    counts = series.value_counts().sort_index()
    rows = tuple(ClauseRow(tag=str(tag), count=int(c)) for tag, c in counts.items())
    return Result.success(rows)


_SUBJ_DEPRELS = frozenset(["nsubj", "nsubj:pass", "csubj", "csubj:pass"])
_OBJ_DEPRELS = frozenset(["obj", "dobj", "iobj", "obl", "xcomp", "ccomp"])
# Passive markings across conventions:
# * UD (Stanford/Stanza): aux:pass on the verb + nsubj:pass patient + obl:agent agent
# * spaCy coarse deps: nsubjpass patient + auxpass auxiliary + agent case head of pobj
_PASSIVE_PATIENT_DEPRELS = frozenset(["nsubj:pass", "nsubjpass"])
_PASSIVE_AGENT_CASES = frozenset(["agent", "obl:agent"])
_AUX_PASS_DEPRELS = frozenset(["aux:pass", "auxpass"])


def _agent_name(dep: pd.Series, sent: pd.DataFrame, full_rel: str) -> str:
    """The surface form of the passive agent introduced by this dependent.

    * UD ``obl:agent``: the dependent itself carries the agent noun, with a
      ``case`` child "by".
    * spaCy ``agent``: the dependent is the preposition "by"; the agent is
      its ``pobj`` child.
    """
    if full_rel == "agent":
        for _, child in sent.iterrows():
            try:
                if int(child[Col.HEAD.value]) == int(dep[Col.ID.value]) and str(child[Col.DEPREL.value]).lower() in (
                    "pobj",
                    "obl",
                    "obj",
                ):
                    return str(child[Col.FORM.value])
            except (ValueError, TypeError):
                continue
        return ""
    return str(dep[Col.FORM.value])


def _is_agent_case(dep: pd.Series, sent: pd.DataFrame) -> bool:
    """Is this dependent a genuine passive-agent marking?

    UD ``obl:agent`` always is. A bare ``case``/``agent`` dependent counts
    only in the spaCy shape, where the preposition "by" heads a ``pobj``
    child that itself hangs off the passive verb — i.e. the preposition's
    head IS the verb being examined (checked by the caller) and its child
    is the agent noun.
    """
    full_rel = str(dep[Col.DEPREL.value]).lower()
    return full_rel in ("agent", "obl:agent")


def extract_svo(frame: pd.DataFrame) -> Result[tuple[SvoRow, ...]]:
    """Extract (subject, verb, object) triples via head-dependency structure."""
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[tuple[SvoRow, ...]](None, checked.diagnostics)

    required = [
        Col.ID.value,
        Col.FORM.value,
        Col.HEAD.value,
        Col.DEPREL.value,
        Col.SENTENCE_ID.value,
        Col.DOCUMENT_ID.value,
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return Result.failure(
            Diagnostic.error("SVO_MISSING_COLUMN", f"missing column(s): {missing}", missing=missing),
        )

    if frame.empty:
        return Result.success(())

    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    pos_col = Col.POS.value if Col.POS.value in frame.columns else None

    svos: list[SvoRow] = []
    clipped: list[Diagnostic] = []

    # Treat each (Document ID, Sentence ID) as an independent sentence partition.
    for (_doc_id, _sent_id), sent in frame.groupby([Col.DOCUMENT_ID.value, Col.SENTENCE_ID.value], sort=False):
        # Index tokens by ID (CoNLL IDs are 1-based per sentence).
        # HEAD of 0 means root — no governing token.
        by_id: dict[int, pd.Series] = {}
        for _, row in sent.iterrows():
            try:
                tid = int(row[Col.ID.value])
            except (ValueError, TypeError):
                continue
            by_id[tid] = row

        # A token is a plausible predicate if it is a verb (when POS is available) or
        # if it is the head of at least one nsubj/object dependent (when POS is missing).
        for _, tok in sent.iterrows():
            is_verb = True
            if pos_col is not None:
                pos = str(tok[pos_col])
                is_verb = pos.startswith("VB") or pos in ("VERB", "AUX")

            try:
                tid2 = int(tok[Col.ID.value])
            except (ValueError, TypeError):
                continue

            # Collect dependents of this token.
            subjects: list[str] = []
            objects: list[str] = []
            passive_patients: list[str] = []
            passive_agents: list[str] = []
            passive_aux_seen = False
            for _, dep in sent.iterrows():
                try:
                    head = int(dep[Col.HEAD.value])
                except (ValueError, TypeError):
                    continue
                if head != tid2:
                    continue
                deprel = str(dep[Col.DEPREL.value]).split(":")[0].lower()
                full_rel = str(dep[Col.DEPREL.value]).lower()
                if full_rel in _PASSIVE_PATIENT_DEPRELS:
                    # Passive patient: semantic OBJECT (legacy maps nsubj:pass to the
                    # object slot — "Mary was hired by John" is (John, hired, Mary)).
                    passive_patients.append(str(dep[Col.Form.value] if hasattr(Col, "Form") else dep[Col.FORM.value]))
                elif full_rel in _PASSIVE_AGENT_CASES and _is_agent_case(dep, sent):
                    # obl:agent (UD) or the agent preposition's pobj child (spaCy).
                    name = _agent_name(dep, sent, full_rel)
                    if name:
                        passive_agents.append(name)
                elif full_rel in _AUX_PASS_DEPRELS:
                    passive_aux_seen = True
                if full_rel in _SUBJ_DEPRELS or deprel == "nsubj" or full_rel == "nsubj" or deprel == "csubj":
                    subjects.append(str(dep[Col.FORM.value]))
                if full_rel in _OBJ_DEPRELS or deprel in ("obj", "dobj", "iobj", "obl", "xcomp"):
                    objects.append(str(dep[Col.FORM.value]))

            passive = bool(passive_patients or passive_agents or passive_aux_seen)
            if passive:
                # Semantic roles, regardless of notation: the obl:agent (or
                # spaCy pobj-of-agent) is the SUBJECT; the nsubj:pass /
                # nsubjpass patient is the OBJECT. Legacy reference:
                # src/Stanford_CoreNLP_SVO_enhanced_dependencies_util.py:245-258.
                if passive_agents and passive_patients:
                    subjects = passive_agents
                    objects = passive_patients
                elif passive_agents:
                    # Agent present, patient not annotated: agent acts on the
                    # clause; emit agent as subject with an empty object? The
                    # legacy engine still emits (agent, verb, ...) triples only
                    # when an object exists, so skip to avoid role reversal.
                    continue
                elif passive_patients:
                    # Agentless passive: the patient is acted upon — emit it as
                    # the object with the legacy inferred-subject placeholder.
                    subjects = ["Inferred_Subject_Passive"]
                    objects = passive_patients
            if subjects and objects:
                # Emit the full cross-product; when it would explode, record a
                # clipping diagnostic instead of silently dropping rows
                # (research output must not silently omit relationships).
                if len(subjects) > 3 or len(objects) > 3:
                    clipped.append(
                        Diagnostic.warning(
                            "SVO_MANY_ROLES",
                            f"verb {str(tok[Col.FORM.value])!r} has {len(subjects)} subject(s) and "
                            f"{len(objects)} object(s); clipping the cross-product to 3x3 "
                            "(split the sentence or filter the parse for the full set)",
                            sentence_id=_sent_id,
                            document_id=_doc_id,
                            subjects=len(subjects),
                            objects=len(objects),
                        )
                    )
                    subjects = subjects[:3]
                    objects = objects[:3]
                verb_form = str(tok[Col.FORM.value])
                if not is_verb and pos_col is not None:
                    # If POS says this is not a verb, skip — it is likely a parsing artefact.
                    continue
                for subj in subjects:
                    for obj in objects:
                        svos.append(
                            SvoRow(
                                subject=subj,
                                verb=verb_form,
                                obj=obj,
                                sentence_id=_sent_id,
                                document_id=_doc_id,
                                document=str(tok[doc_col]) if doc_col is not None else "",
                            )
                        )

    return Result.success(tuple(svos), *clipped)
