"""spaCy backend — lazy, cached, validated.

The legacy suite built a spaCy pipeline per sentence and never checked whether
the requested language was actually available. Here a pipeline is built once,
behind :class:`PipelineCache`, and the language gate in :mod:`core.config`
fails loudly before a model is touched.

**No silent degradation.** If no trained model is installed for the language
the build FAILS with ``PIPELINE_MODEL_MISSING`` and the exact fix command in
the message. A blank pipeline has no tagger/lemmatizer/parser, so falling
back to one would make every downstream analysis quietly return hollow
numbers under a green exit code — the worst class of failure this suite
bans. Use ``tools/doctor.py`` to check what is installed before a run.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.conll.schema import Col, canonical_columns
from core.io.reader import Corpus, display_names
from core.result import Diagnostic, Result

__all__ = ["SpacyPipeline", "build_spacy_pipeline", "spacy_model_name"]

_CANONICAL = canonical_columns()

# spaCy's published models use two naming families: *_core_web_* (en, zh,
# xx, uk) and *_core_news_* (de, fr, es, it, pt, nl, ru, pl, ro, hr, da,
# el, ja). Both are tried, small -> large, so an installed de_core_news_sm
# is found just like en_core_web_sm.
_MODEL_FAMILIES = ("web", "news")
_MODEL_SIZES = ("sm", "md", "lg")


def spacy_model_name(language: str) -> str:
    """The canonical model name this backend looks for first.

    Used by the doctor and by failure messages, so one place defines the
    name a user must install.
    """
    return f"{language}_core_web_sm"


@dataclass(frozen=True, slots=True)
class SpacyPipeline:
    """Thin wrapper around a loaded spaCy ``Language`` object."""

    _nlp: object
    language: str
    tasks: frozenset[str]
    model_name: str

    @property
    def backend(self) -> str:
        return "spacy"

    def supports(self, language: str, tasks: object = ()) -> bool:
        from core.config import supports_language

        return supports_language("spacy", language)

    def tokenize(self, text: str) -> tuple[str, ...]:
        """The tokenizer alone -- spaCy's first step, and the one that fixes
        the Form column. Running the whole pipeline would give the same tokens
        for many times the cost."""
        return tuple(token.text for token in self._nlp.tokenizer(text))  # type: ignore[attr-defined]

    def parse(self, corpus: Corpus) -> Result[pd.DataFrame]:
        """Parse *corpus* into a canonical CoNLL table."""
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
                spacy_doc = nlp(doc.text)  # type: ignore[operator]
            except Exception as exc:
                diagnostics.append(
                    Diagnostic.error(
                        "PARSE_FAILED",
                        f"spaCy failed on {names[doc.doc_id]}: {exc}",
                        doc_id=doc.doc_id,
                        path=str(doc.path),
                    )
                )
                continue
            for sent_idx, sent in enumerate(spacy_doc.sents, start=1):
                for token in sent:
                    head_id = 0
                    try:
                        head = token.head
                        # spaCy may expose distinct Python wrapper objects for
                        # the same token; identity comparison mislabels roots
                        # as self-headed. Compare stable token indices.
                        if int(head.i) != int(token.i):
                            head_id = int(head.i - sent.start + 1)
                    except Exception:
                        head_id = 0
                    pos = ""
                    try:
                        pos = str(getattr(token, "tag_", "") or getattr(token, "pos_", "") or "")
                    except Exception:
                        pos = ""
                    lemma = ""
                    try:
                        lemma = str(getattr(token, "lemma_", "") or token.text)
                    except Exception:
                        lemma = str(token.text)
                    ner = "O"
                    try:
                        ent = str(getattr(token, "ent_type_", "") or "")
                        if ent:
                            ner = ent
                    except Exception:
                        ner = "O"
                    dep = ""
                    try:
                        dep = str(getattr(token, "dep_", "") or "")
                    except Exception:
                        dep = ""
                    rows.append(
                        {
                            Col.ID.value: int(token.i - sent.start + 1),
                            Col.FORM.value: str(token.text),
                            Col.LEMMA.value: lemma,
                            Col.POS.value: pos,
                            Col.NER.value: ner,
                            Col.HEAD.value: head_id,
                            Col.DEPREL.value: dep,
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


def build_spacy_pipeline(language: str, tasks: frozenset[str]) -> Result[SpacyPipeline]:
    """Build a spaCy pipeline for *language*, or fail with the fix command.

    Tries ``{language}_core_web_{sm,md,lg}`` then ``{language}_core_news_{sm,md,lg}``.
    Missing models are a hard failure (``PIPELINE_MODEL_MISSING``), never a
    blank-pipeline fallback. No model is ever downloaded at import time.
    """
    try:
        import spacy
    except ImportError as exc:
        return Result.failure(
            Diagnostic.error("PIPELINE_MISSING_DEP", f"spaCy is not installed: {exc}", backend="spacy")
        )

    from core.config import supports_language

    if not supports_language("spacy", language):
        return Result.failure(
            Diagnostic.error(
                "PIPELINE_UNSUPPORTED",
                f"spaCy has no confirmed model for language {language!r}",
                backend="spacy",
                language=language,
            )
        )

    # spaCy's published models use two naming families: *_core_web_* (en, zh,
    # xx, uk) and *_core_news_* (de, fr, es, it, pt, nl, ru, pl, ro, hr, da,
    # el, ja). Trying only the web family meant e.g. an installed
    # de_core_news_sm was never found and the pipeline silently degraded to
    # a blank tokenizer. Try both families, small -> large.
    model_candidates = [f"{language}_core_{family}_{size}" for family in _MODEL_FAMILIES for size in _MODEL_SIZES]
    for model_name in model_candidates:
        try:
            nlp = spacy.load(model_name)
            return Result.success(SpacyPipeline(_nlp=nlp, language=language, tasks=tasks, model_name=model_name))
        except OSError:
            continue  # not installed under this name; try the next candidate
        except Exception as exc:  # a model that exists but cannot load is a real failure
            return Result.failure(
                Diagnostic.error(
                    "PIPELINE_BUILD_FAILED",
                    f"spaCy model {model_name} is installed but failed to load: {exc}",
                    backend="spacy",
                    language=language,
                    model=model_name,
                )
            )

    # No silent degradation: without a trained model there is no POS, no
    # lemma, no dependency parse, and every downstream number would be
    # hollow while the run still exits green. Fail with the exact fix.
    return Result.failure(
        Diagnostic.error(
            "PIPELINE_MODEL_MISSING",
            f"no spaCy model for {language!r} is installed; a blank pipeline would silently "
            "strip POS/lemma/dependency information from every analysis.",
            backend="spacy",
            language=language,
            tried=model_candidates,
            fix=f"python -m spacy download {spacy_model_name(language)}",
        )
    )
