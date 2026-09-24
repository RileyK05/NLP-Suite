"""Stanza backend — lazy, cached, validated."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.conll.schema import Col, canonical_columns
from core.io.reader import Corpus, display_names
from core.pipelines.quiet import CapturedLogs, captured_logs
from core.result import Diagnostic, Result

__all__ = ["StanzaPipeline", "build_stanza_pipeline"]

_CANONICAL = canonical_columns()


def _word_id(token: object) -> int | None:
    """Integer word id, or None for multi-word-token tuple ids."""
    raw = getattr(token, "id", None)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    return None


def _canonical_ner(tag: str) -> str:
    """BIOES (Stanza) -> bare entity type (spaCy convention).

    C6-13: one canonical NER representation across backends. ``B-PERSON``,
    ``E-PERSON``, ``S-PERSON`` all become ``PERSON``; ``I-*`` keeps the
    type of its chunk. ``O`` stays ``O``.
    """
    if not tag or tag == "O":
        return "O"
    if "-" in tag:
        prefix, rest = tag.split("-", 1)
        if prefix.upper() in ("B", "I", "E", "S") and rest:
            return rest
    return tag


def _word_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return 0


@dataclass(frozen=True, slots=True)
class StanzaPipeline:
    _nlp: object
    language: str
    tasks: frozenset[str]
    model_name: str

    @property
    def backend(self) -> str:
        return "stanza"

    def supports(self, language: str, tasks: object = ()) -> bool:
        from core.config import supports_language

        return supports_language("stanza", language)

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Stanza's own segmentation, through the loaded pipeline.

        Stanza has no separable tokenizer object, so this runs the pipeline
        over a phrase-length string. Words rather than tokens, to match what
        :meth:`parse` writes into the Form column for multi-word tokens.
        """
        document = self._nlp(text)  # type: ignore[operator]
        return tuple(word.text for sentence in document.sentences for word in sentence.words)

    def parse(self, corpus: Corpus) -> Result[pd.DataFrame]:
        nlp = self._nlp
        rows: list[dict[str, object]] = []
        diagnostics: list[Diagnostic] = []
        record_id = 1
        names = display_names(corpus.docs)
        for doc in corpus.docs:
            if not doc.text.strip():
                diagnostics.append(
                    Diagnostic.warning("EMPTY_DOC", f"{names[doc.doc_id]} has no content", doc_id=doc.doc_id)
                )
                continue
            try:
                stanza_doc = nlp(doc.text)  # type: ignore[operator]
            except Exception as exc:
                diagnostics.append(
                    Diagnostic.error("PARSE_FAILED", f"Stanza failed on {names[doc.doc_id]}: {exc}", doc_id=doc.doc_id)
                )
                continue
            for sent_idx, sent in enumerate(stanza_doc.sentences, start=1):
                # NER lives on tokens (BIOES spans), not words: map word id -> tag.
                ner_by_word: dict[int, str] = {}
                for parent in sent.tokens:
                    tag = _canonical_ner(str(getattr(parent, "ner", "O") or "O"))
                    for sub in parent.words:
                        wid = _word_id(sub)
                        if wid is not None:
                            ner_by_word[wid] = tag
                for token in sent.words:
                    wid = _word_id(token)
                    word_text = str(getattr(token, "text", ""))
                    lemma = str(getattr(token, "lemma", "") or word_text)
                    pos = str(getattr(token, "xpos", "") or getattr(token, "upos", "") or "")
                    head = _word_int(getattr(token, "head", 0))
                    deprel = str(getattr(token, "deprel", "") or "")
                    rows.append(
                        {
                            Col.ID.value: wid if wid is not None else record_id,
                            Col.FORM.value: word_text,
                            Col.LEMMA.value: lemma,
                            Col.POS.value: pos,
                            Col.NER.value: ner_by_word.get(wid, "O") if wid is not None else "O",
                            Col.HEAD.value: head,
                            Col.DEPREL.value: deprel,
                            Col.DEPS.value: "_",
                            Col.CLAUSE_TAG.value: "",
                            Col.RECORD_ID.value: record_id,
                            Col.SENTENCE_ID.value: sent_idx,
                            Col.DOCUMENT_ID.value: str(doc.doc_id),
                            Col.DOCUMENT.value: names[doc.doc_id],
                        }
                    )
                    record_id += 1
        if not rows:
            frame = pd.DataFrame(columns=list(_CANONICAL))
            if diagnostics:
                return Result[pd.DataFrame](frame, tuple(diagnostics))
            return Result.success(frame, *diagnostics)
        frame = pd.DataFrame(rows, columns=list(_CANONICAL))
        return Result.success(frame, *diagnostics)


#: A token this long is a path or a model name, not a common word. If one
#: appears in both the log line and the exception, the two are describing the
#: same artifact and only one of them belongs in the diagnostic.
_SHARED_TOKEN_LENGTH = 12


def _is_redundant(message: str, exception_text: str) -> bool:
    """Whether a captured log line adds nothing to what the exception says."""
    if message in exception_text:
        return True
    return any(
        len(token) >= _SHARED_TOKEN_LENGTH and token in exception_text for token in message.replace(",", " ").split()
    )


def _held_back(logged: CapturedLogs, exc: BaseException) -> str:
    """Detail Stanza logged that its exception did not already say.

    Dropping the library's logging outright would lose information whenever the
    raise is terser than the log line. Keeping all of it produces a diagnostic
    that says the same thing twice: Stanza logs "Cannot load model from <path>"
    and then raises "Could not find model file <path>". So a line is kept only
    when it names something the exception does not, which puts the whole story
    in the one place the suite reports failures and puts none of it there
    twice.
    """
    text = str(exc)
    extra = [message for message in logged.messages() if not _is_redundant(message, text)]
    return f" ({'; '.join(extra)})" if extra else ""


def build_stanza_pipeline(language: str, tasks: frozenset[str]) -> Result[StanzaPipeline]:
    # C6-13: language gate BEFORE any stanza import so an unsupported
    # language is a pure config answer, independent of the environment.
    from core.config import supports_language

    if not supports_language("stanza", language):
        return Result.failure(
            Diagnostic.error(
                "PIPELINE_UNSUPPORTED",
                f"Stanza has no confirmed model for language {language!r}",
                backend="stanza",
                language=language,
            )
        )
    # Task gate: like the language gate, a pure config answer before any
    # import. Build only the processors requested by the caller. An empty task
    # set is the full canonical parse; task-specific callers no longer pay for
    # NER/dependency models they cannot consume.
    processor_for_task = {
        "tokenize": "tokenize",
        "sentences": "tokenize",
        "pos": "pos",
        "lemma": "lemma",
        "depparse": "depparse",
        "dependency": "depparse",
        "ner": "ner",
    }
    normalized_tasks = {str(task).strip().lower() for task in tasks}
    requested = {processor_for_task.get(task, task) for task in normalized_tasks}
    unknown = sorted(p for p in requested if p not in {"tokenize", "pos", "lemma", "depparse", "ner"})
    if unknown:
        return Result.failure(
            Diagnostic.error("PIPELINE_BAD_TASK", f"unsupported Stanza task(s): {unknown}", tasks=unknown)
        )
    try:
        import stanza
    except ImportError as exc:
        return Result.failure(
            Diagnostic.error("PIPELINE_MISSING_DEP", f"stanza is not installed: {exc}", backend="stanza")
        )
    # Bound by the ``with`` below before Stanza runs, so the failure branches
    # can read whatever it logged on its way to raising.
    logged = CapturedLogs()
    try:
        # Stanza enforces processor prerequisites: depparse needs pos+lemma,
        # lemma needs pos, and every processor needs tokenize. Expand the
        # request so a task-specific build cannot fail with a
        # pipeline-requirements error (misdiagnosed as a corrupt model when
        # it is an incomplete processor list).
        if requested:
            requested |= {"tokenize"}
        if "depparse" in requested:
            requested |= {"pos", "lemma"}
        if "lemma" in requested:
            requested |= {"pos"}
        processors = (
            ",".join(p for p in ("tokenize", "pos", "lemma", "depparse", "ner") if p in requested)
            if requested
            else "tokenize,pos,lemma,depparse,ner"
        )
        # Stanza logs its own load failures to stderr and *then* raises. The
        # raise is what this function reports on, so the log line is a
        # duplicate that arrives ahead of whatever the tool was printing --
        # `nlp-doctor` opened with a raw ERROR above its own report. Held
        # back here and folded into the diagnostic below instead.
        with captured_logs("stanza") as logged:
            nlp = stanza.Pipeline(lang=language, processors=processors, verbose=False, download_method=None)
        return Result.success(StanzaPipeline(_nlp=nlp, language=language, tasks=tasks, model_name=f"stanza:{language}"))
    except PermissionError as exc:
        return Result.failure(
            Diagnostic.error(
                "PIPELINE_MODEL_UNREADABLE",
                f"Stanza model resources for {language!r} are not readable: {exc}; "
                f"repair with stanza.download('{language}')" + _held_back(logged, exc),
                backend="stanza",
                language=language,
                fix=f"python -c \"import stanza; stanza.download('{language}')\"",
            )
        )
    except FileNotFoundError as exc:
        return Result.failure(
            Diagnostic.error(
                "PIPELINE_MODEL_MISSING",
                f"Stanza model resources for {language!r} are missing: {exc}" + _held_back(logged, exc),
                backend="stanza",
                language=language,
                fix=f"python -c \"import stanza; stanza.download('{language}')\"",
            )
        )
    except Exception as exc:
        code = "PIPELINE_MODEL_CORRUPT"
        return Result.failure(
            Diagnostic.error(
                code,
                f"could not build Stanza pipeline for {language!r}: {exc}{_held_back(logged, exc)}. "
                f'Stanza models are downloaded once and cached: run `python -c "import stanza; '
                f"stanza.download('{language}')\"` from the suite environment, then retry.",
                backend="stanza",
                language=language,
                fix=f"python -c \"import stanza; stanza.download('{language}')\"",
            )
        )
