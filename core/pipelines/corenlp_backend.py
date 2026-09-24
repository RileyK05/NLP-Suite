"""CoreNLP backend — parses through a running Stanford CoreNLP Java server.

The server itself is outside this suite (FR-5.2): start it with
``java -mx4g -cp "stanford-corenlp-*-models.jar" edu.stanford.nlp.pipeline.StanfordCoreNLPServer``.
Every entry point probes the server first over plain ``urllib`` (no new
dependency); a dead server fails with ``CORENLP_UNAVAILABLE`` naming the
exact start command instead of a connection traceback.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from core.conll.schema import Col
from core.io.reader import Corpus, display_names
from core.result import Diagnostic, Result

__all__ = ["CoreNLPPipeline", "build_corenlp_pipeline", "probe_server"]

_START_FIX = (
    'start the server: java -mx4g -cp "stanford-corenlp-*-models.jar" '
    "edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000"
)

_TIMEOUT = 30.0


def _checked_url(server_url: str) -> str | None:
    """Reject non-HTTP(S) server URLs before any request is built."""
    scheme = urllib.parse.urlsplit(server_url).scheme.lower()
    return None if scheme in ("http", "https") else scheme or "(missing)"


class CoreNLPPipeline:
    backend = "corenlp"

    def __init__(self, language: str, tasks: frozenset[str], server_url: str = "http://localhost:9000") -> None:
        self.language = language
        self.tasks = tasks
        self.server_url = server_url

    def supports(self, language: str, tasks: object = ()) -> bool:
        from core.config import supports_language

        return supports_language("corenlp", language)

    def tokenize(self, text: str) -> tuple[str, ...]:
        """CoreNLP tokenizes on the server, so this is a round trip.

        Raises when the server cannot be reached rather than guessing. A
        caller that cannot get this backend's opinion about token boundaries
        has to say so; quietly substituting a different opinion is how a
        phrase that is in the text comes back as absent.
        """
        annotated = self._annotate(text)
        if annotated.value is None:
            raise ValueError("; ".join(d.message for d in annotated.diagnostics))
        payload = annotated.unwrap()
        sentences = payload.get("sentences", [])
        if not isinstance(sentences, list):
            raise ValueError("CoreNLP returned no sentences for that phrase.")
        words: list[str] = []
        for sentence in sentences:
            for token in sentence.get("tokens", []) if isinstance(sentence, dict) else []:
                words.append(str(token.get("originalText") or token.get("word", "")))
        return tuple(words)

    def parse(self, corpus: Corpus) -> Result[pd.DataFrame]:
        probed = probe_server(self.server_url)
        if probed.value is None:
            return Result.failure(*probed.diagnostics)
        rows: list[dict[str, object]] = []
        record_id = 1
        diags: list[Diagnostic] = list(probed.diagnostics)
        names = display_names(corpus.docs)
        for doc in corpus:
            annotated = self._annotate(doc.text)
            if annotated.value is None:
                return Result.failure(*diags, *annotated.diagnostics)
            sentences = annotated.unwrap().get("sentences", [])
            if not isinstance(sentences, list):
                return Result.failure(
                    *diags,
                    Diagnostic.error(
                        "CORENLP_BAD_RESPONSE",
                        f"CoreNLP server at {self.server_url} returned no sentences list",
                        backend="corenlp",
                    ),
                )
            for sent_idx, sentence in enumerate(sentences, start=1):
                if not isinstance(sentence, dict):
                    continue
                raw_deps = sentence.get("basicDependencies", [])
                deps = (
                    {d["dependent"]: d for d in raw_deps if isinstance(d, dict) and "dependent" in d}
                    if isinstance(raw_deps, list)
                    else {}
                )
                raw_tokens = sentence.get("tokens", [])
                for token in raw_tokens if isinstance(raw_tokens, list) else []:
                    if not isinstance(token, dict):
                        continue
                    index = int(token.get("index", 0))
                    dep = deps.get(index, {})
                    rows.append(
                        {
                            Col.ID.value: index,
                            Col.FORM.value: str(token.get("word", "")),
                            Col.LEMMA.value: str(token.get("lemma", "") or token.get("word", "")),
                            Col.POS.value: str(token.get("pos", "")),
                            Col.NER.value: str(token.get("ner", "O") or "O"),
                            Col.HEAD.value: int(dep.get("governor", 0)),
                            Col.DEPREL.value: str(dep.get("dep", "root" if index == 1 else "")),
                            Col.DEPS.value: "_",
                            Col.CLAUSE_TAG.value: "",
                            Col.RECORD_ID.value: record_id,
                            Col.SENTENCE_ID.value: sent_idx,
                            Col.DOCUMENT_ID.value: str(doc.doc_id),
                            Col.DOCUMENT.value: names[doc.doc_id],
                        }
                    )
                    record_id += 1
        return Result.success(pd.DataFrame(rows), *diags)

    def _annotate(self, text: str) -> Result[dict[str, object]]:
        """POST one document to the server; network failures stay loud."""
        bad = _checked_url(self.server_url)
        if bad is not None:
            return Result.failure(
                Diagnostic.error(
                    "CORENLP_BAD_URL",
                    f"CoreNLP server URL must be http(s), got scheme {bad!r}",
                    backend="corenlp",
                )
            )
        properties = json.dumps({"annotators": "tokenize,ssplit,pos,lemma,ner,depparse", "outputFormat": "json"})
        url = self.server_url.rstrip("/") + "/?" + urllib.parse.urlencode({"properties": properties})
        request = urllib.request.Request(url, data=text.encode("utf-8"), method="POST")  # noqa: S310
        try:
            # Scheme pre-validated http(s) by _checked_url above.
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return Result.failure(
                Diagnostic.error(
                    "CORENLP_REQUEST_FAILED",
                    f"CoreNLP server at {self.server_url} failed mid-parse: {exc}",
                    backend="corenlp",
                    fix=_START_FIX,
                )
            )
        if not isinstance(payload, dict):
            return Result.failure(
                Diagnostic.error(
                    "CORENLP_BAD_RESPONSE", f"CoreNLP server at {self.server_url} returned non-JSON", backend="corenlp"
                )
            )
        return Result.success(payload)


def probe_server(server_url: str = "http://localhost:9000", timeout: float = 3.0) -> Result[dict[str, object]]:
    """POST a one-word tokenize request; proves the server is alive."""
    bad = _checked_url(server_url)
    if bad is not None:
        return Result.failure(
            Diagnostic.error(
                "CORENLP_BAD_URL", f"CoreNLP server URL must be http(s), got scheme {bad!r}", backend="corenlp"
            )
        )
    properties = json.dumps({"annotators": "tokenize", "outputFormat": "json"})
    url = server_url.rstrip("/") + "/?" + urllib.parse.urlencode({"properties": properties})
    request = urllib.request.Request(url, data=b"ping", method="POST")  # noqa: S310
    try:
        # Scheme pre-validated http(s) by _checked_url above.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            json.loads(response.read().decode("utf-8"))
    except http.client.BadStatusLine as exc:
        # A listener that answers with non-HTTP bytes (e.g. a Jupyter ZMQ
        # kernel squatting the port) — classify as unavailable, not a crash.
        return Result.failure(
            Diagnostic.error(
                "CORENLP_UNAVAILABLE",
                f"no CoreNLP server at {server_url} (port answers, but not HTTP — is another "
                f"program listening there? [{exc}]); use --parser spacy or stanza for the default path",
                backend="corenlp",
                server_url=server_url,
                fix=_START_FIX,
            )
        )
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Result.failure(
            Diagnostic.error(
                "CORENLP_UNAVAILABLE",
                f"no CoreNLP server at {server_url} ({exc}); use --parser spacy or stanza for the default path",
                backend="corenlp",
                server_url=server_url,
                fix=_START_FIX,
            )
        )
    return Result.success({"server_url": server_url})


def build_corenlp_pipeline(language: str, tasks: frozenset[str]) -> Result[CoreNLPPipeline]:
    from core.config import supports_language

    if not supports_language("corenlp", language):
        return Result.failure(
            Diagnostic.error(
                "PIPELINE_UNSUPPORTED",
                f"CoreNLP has no confirmed model for language {language!r}",
                backend="corenlp",
                language=language,
            )
        )
    return Result.success(CoreNLPPipeline(language=language, tasks=tasks))
