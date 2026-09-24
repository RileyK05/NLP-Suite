"""Neural sentiment — four backends, one table contract (HW5).

The syllabus grades the COMPARISON: "Neural network approaches to sentiment
analysis: BERT, Stanford CoreNLP, spaCy, Stanza" and HW5 asks which annotator
produces better results, and which produces better shape-of-stories results.
So these are four separate tools over one shared table contract — the
``Compound`` column is the signed sentiment the story-shape tools consume.

What each backend actually is, and the suite never papers over the gaps:

* ``bert``      — a Hugging Face sequence classifier over each sentence
  (``transformers``/``torch``, the optional ``embeddings`` extra). Long
  sentences are truncated at 256 words with a warning, not silently cut.
* ``spacy``     — a spaCy pipeline's ``textcat`` head over each sentence read
  as a one-sentence document. Vanilla ``en_core_web_sm`` has NO sentiment
  head; the tool says so and stops rather than inventing a score.
* ``stanza``    — Stanza's own ``sentiment`` processor, a three-way hard class
  (0/1/2) shipped for some languages only.
* ``corenlp``   — Stanford CoreNLP's ``sentiment`` annotator (the RNTN) via a
  running Java server; the graded special annotator of HW3/HW5.

Missing backends fail as diagnostics with an install/start fix (the MALLET
rule: never substitute another backend, because "compare four annotators" must
not become "compare one annotator with itself"). Third-party imports are lazy;
tests inject fakes through the same seams production uses.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from core.conll.schema import Col, validate_columns
from core.result import Diagnostic, Result

__all__ = [
    "bert_sentences",
    "corenlp_sentences",
    "sentence_rows",
    "spacy_sentences",
    "stanza_sentences",
    "summarize_neural",
]

_SENTENCE_COLUMNS = ["Document ID", "Document", "Sentence ID", "Sentence", "Label", "Score", "Compound"]
_DOCUMENT_COLUMNS = ["Document ID", "Document", "Sentences", "Mean Compound", "Positive", "Negative", "Neutral"]

#: BERT-style classifiers have a context window; 256 whitespace words is a
#: conservative cut that keeps almost every real sentence inside the model.
_TRUNCATE_WORDS = 256
_FIX_TRANSFORMERS = "pip install transformers torch (the 'embeddings' extra); the model downloads on first use"
_FIX_SPACY = (
    "install a spaCy pipeline that carries a sentiment textcat head (a trained "
    "textcat project or a model that ships one); en_core_web_sm has none"
)
_FIX_STANZA = "pip install stanza, then stanza.download('<language>') — the sentiment processor ships with some language packages only"
_FIX_SERVER = (
    'start the server: java -mx4g -cp "stanford-corenlp-*-models.jar" '
    "edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000"
)

_STANZA_CLASS_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
_CORENLP_VALUE = {
    "very negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very positive": 4,
}


def _bucket(label: str) -> str:
    """Map any backend's label words onto positive / negative / neutral."""
    low = str(label).strip().lower()
    if "neg" in low:
        return "negative"
    if "pos" in low:
        return "positive"
    return "neutral"


def _compound(label: str, score: float) -> float:
    """Signed sentiment in [-1, +1]; +1 is confidently positive.

    A hard-class backend passes score=1.0 and so lands on -1 / 0 / +1.
    """
    bucket = _bucket(label)
    if bucket == "positive":
        return round(abs(score), 4)
    if bucket == "negative":
        return round(-abs(score), 4)
    return 0.0


def _truncated(sentence: str) -> tuple[str, bool]:
    words = sentence.split()
    if len(words) <= _TRUNCATE_WORDS:
        return sentence, False
    return " ".join(words[:_TRUNCATE_WORDS]), True


def sentence_rows(frame: pd.DataFrame) -> Result[list[tuple[str, str, object, str]]]:
    """``(doc_id, doc_name, sent_id, sentence text)`` in document order.

    Shared by the three parse-based backends so their tables are row-comparable;
    the sentence is its Form tokens joined, because a sentiment model must see
    the words as written (not lemmas).
    """
    checked = validate_columns([str(c) for c in frame.columns])
    if not checked.ok:
        return Result[list[tuple[str, str, object, str]]](None, checked.diagnostics)
    missing = [c for c in (Col.FORM.value, Col.DOCUMENT_ID.value, "Sentence ID") if c not in frame.columns]
    if missing:
        return Result.failure(Diagnostic.error("SENTIMENT_NN_MISSING_COLUMN", f"missing {missing[0]!r}"))
    doc_col = Col.DOCUMENT.value if Col.DOCUMENT.value in frame.columns else None
    rows: list[tuple[str, str, object, str]] = []
    for (doc_id, sent_id), group in frame.groupby([Col.DOCUMENT_ID.value, "Sentence ID"], sort=False):
        name = str(group[doc_col].iloc[0]) if doc_col is not None else f"doc-{doc_id}"
        text = " ".join(str(form) for form in group[Col.FORM.value].tolist())
        rows.append((str(doc_id), name, sent_id, text))
    return Result.success(rows)


def _frame_from_scores(scored: list[dict[str, object]], extra: Sequence[Diagnostic] = ()) -> Result[pd.DataFrame]:
    if not scored:
        return Result.success(pd.DataFrame(columns=_SENTENCE_COLUMNS), *extra)
    return Result.success(pd.DataFrame(scored, columns=_SENTENCE_COLUMNS), *extra)


def _score_with(
    rows: list[tuple[str, str, object, str]],
    scorer: Callable[[str], tuple[str, float, float]],
) -> tuple[list[dict[str, object]], list[Diagnostic]]:
    """Run *scorer* per sentence; it answers (label, score, compound)."""
    out: list[dict[str, object]] = []
    diags: list[Diagnostic] = []
    cut = 0
    for doc_id, name, sent_id, text in rows:
        shortened, was_cut = _truncated(text)
        cut += 1 if was_cut else 0
        label, score, compound = scorer(shortened)
        out.append(
            {
                "Document ID": doc_id,
                "Document": name,
                "Sentence ID": sent_id,
                "Sentence": text,
                "Label": label,
                "Score": round(abs(score), 4),
                "Compound": compound,
            }
        )
    if cut:
        diags.append(
            Diagnostic.warning(
                "SENTIMENT_NN_TRUNCATED",
                f"{cut} sentence(s) longer than {_TRUNCATE_WORDS} words were truncated for the model window",
                count=cut,
                limit=_TRUNCATE_WORDS,
            )
        )
    return out, diags


def summarize_neural(annotated: pd.DataFrame) -> Result[pd.DataFrame]:
    """Per-document roll-up of the sentence table (the story-shape feed)."""
    if annotated.empty:
        return Result.success(pd.DataFrame(columns=_DOCUMENT_COLUMNS))
    for column in _SENTENCE_COLUMNS:
        if column not in annotated.columns:
            return Result.failure(Diagnostic.error("SENTIMENT_NN_MISSING_COLUMN", f"missing {column!r}"))
    rows: list[dict[str, object]] = []
    for (doc_id, name), group in annotated.groupby(["Document ID", "Document"], sort=False):
        buckets = group["Label"].map(_bucket)
        rows.append(
            {
                "Document ID": doc_id,
                "Document": name,
                "Sentences": len(group),
                "Mean Compound": round(float(group["Compound"].mean()), 4),
                "Positive": int((buckets == "positive").sum()),
                "Negative": int((buckets == "negative").sum()),
                "Neutral": int((buckets == "neutral").sum()),
            }
        )
    return Result.success(pd.DataFrame(rows, columns=_DOCUMENT_COLUMNS))


# ---------------------------------------------------------------- BERT


def bert_sentences(
    frame: pd.DataFrame,
    *,
    model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    pipeline: Callable[[str], list[dict[str, Any]]] | None = None,
) -> Result[pd.DataFrame]:
    """Per-sentence sentiment from a Hugging Face sequence classifier.

    ``pipeline`` is the seam: the real ``transformers.pipeline(...)``
    ``sentiment-analysis`` callable, or a fake in tests. A two-class head
    (POSITIVE/NEGATIVE) is signed by its winning label; any head with a
    neutral-ish label collapses to 0.
    """
    rows = sentence_rows(frame)
    if rows.value is None:
        return Result.failure(*rows.diagnostics)
    scorer = pipeline
    if scorer is None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            return Result.failure(
                Diagnostic.error(
                    "SENTIMENT_NN_UNAVAILABLE",
                    f"transformers is not installed; neural BERT sentiment needs it ({exc})",
                    fix=_FIX_TRANSFORMERS,
                )
            )
        try:
            scorer = hf_pipeline("sentiment-analysis", model=model)  # type: ignore[call-overload,unused-ignore]
        except Exception as exc:
            return Result.failure(
                Diagnostic.error(
                    "SENTIMENT_NN_UNAVAILABLE", f"could not load model {model!r}: {exc}", fix=_FIX_TRANSFORMERS
                )
            )

    def score(text: str) -> tuple[str, float, float]:
        verdict = scorer(text)
        best = verdict[0] if isinstance(verdict, list) else verdict
        label = str(best.get("label", "neutral"))
        value = float(best.get("score", 0.0))
        return label, value, _compound(label, value)

    scored, extra = _score_with(rows.unwrap(), score)
    return _frame_from_scores(scored, extra)


# ---------------------------------------------------------------- spaCy


def _spacy_head(nlp: Any) -> bool:
    """Does this pipeline carry a text categorizer? Unknown shape counts as yes."""
    has = getattr(nlp, "has", None)
    if callable(has):
        return bool(has("textcat") or has("textcat_multilabel") or has("textcat_single_label"))
    return True


def spacy_sentences(
    frame: pd.DataFrame,
    *,
    model: str = "en_core_web_sm",
    nlp: Any = None,
) -> Result[pd.DataFrame]:
    """Per-sentence sentiment from a spaCy pipeline's textcat head.

    Each sentence is scored as a one-sentence document and its ``cats`` read,
    because spaCy's textcat attaches to the Doc. A pipeline without any text
    categorizer fails loudly — vanilla ``en_core_web_sm`` is exactly that case,
    and HW5's comparison would be dishonest if this tool quietly fell back to
    a lexicon.
    """
    rows = sentence_rows(frame)
    if rows.value is None:
        return Result.failure(*rows.diagnostics)
    pipe = nlp
    if pipe is None:
        try:
            import spacy
        except ImportError as exc:
            return Result.failure(
                Diagnostic.error("SENTIMENT_NN_UNAVAILABLE", f"spacy is not installed ({exc})", fix="pip install spacy")
            )
        try:
            pipe = spacy.load(model)
        except Exception as exc:
            return Result.failure(
                Diagnostic.error(
                    "SENTIMENT_NN_UNAVAILABLE", f"could not load spaCy model {model!r}: {exc}", fix=_FIX_SPACY
                )
            )
    if not _spacy_head(pipe):
        return Result.failure(
            Diagnostic.error(
                "SENTIMENT_NN_NO_HEAD",
                f"the spaCy pipeline {model!r} carries no sentiment textcat head",
                fix=_FIX_SPACY,
            )
        )

    missing_head = False

    def score(text: str) -> tuple[str, float, float]:
        nonlocal missing_head
        doc = pipe(text)
        cats = dict(getattr(doc, "cats", {}) or {})
        if not cats:
            missing_head = True
            return "neutral", 0.0, 0.0
        label = max(cats, key=lambda key: float(cats[key]))
        value = float(cats[label])
        return label, value, _compound(label, value)

    scored, extra = _score_with(rows.unwrap(), score)
    if missing_head and not scored:
        return Result.failure(
            Diagnostic.error("SENTIMENT_NN_NO_HEAD", "the pipeline scored no sentence", fix=_FIX_SPACY)
        )
    if missing_head:
        extra.append(Diagnostic.warning("SENTIMENT_NN_NO_HEAD", "some sentences carried no cats and scored neutral"))
    return _frame_from_scores(scored, extra)


# ---------------------------------------------------------------- Stanza


def stanza_sentences(
    frame: pd.DataFrame,
    *,
    language: str = "en",
    pipeline: Any = None,
) -> Result[pd.DataFrame]:
    """Per-sentence sentiment from Stanza's own ``sentiment`` processor.

    Stanza answers a hard three-way class per sentence (0 negative, 1 neutral,
    2 positive), so Score is 1.0 and Compound lands on -1 / 0 / +1 — the
    coarseness is the backend's, and it is why the shape tools must treat
    these scores as a different instrument from BERT's confidences.
    """
    rows = sentence_rows(frame)
    if rows.value is None:
        return Result.failure(*rows.diagnostics)
    pipe = pipeline
    if pipe is None:
        try:
            import stanza
        except ImportError as exc:
            return Result.failure(
                Diagnostic.error(
                    "SENTIMENT_NN_UNAVAILABLE", f"stanza is not installed ({exc})", fix="pip install stanza"
                )
            )
        try:
            pipe = stanza.Pipeline(language, processors="tokenize,pos,sentiment")
        except Exception as exc:
            text = str(exc)
            fix = _FIX_STANZA.replace("<language>", language)
            if "sentiment" in text.lower():
                return Result.failure(Diagnostic.error("SENTIMENT_NN_NO_HEAD", text, fix=fix))
            return Result.failure(Diagnostic.error("SENTIMENT_NN_UNAVAILABLE", text, fix=fix))

    def score(text: str) -> tuple[str, float, float]:
        doc = pipe(text)
        sentences = list(getattr(doc, "sentences", []) or [])
        if not sentences:
            return "neutral", 1.0, 0.0
        votes = [int(getattr(sent, "sentiment", 1)) for sent in sentences]
        value = max(set(votes), key=votes.count)
        label = _STANZA_CLASS_LABELS.get(value, "neutral")
        return label, 1.0, _compound(label, 1.0)

    scored, extra = _score_with(rows.unwrap(), score)
    return _frame_from_scores(scored, extra)


# ---------------------------------------------------------------- CoreNLP


def _corenlp_scores(
    server_url: str,
    properties: str,
    text: str,
    request_json: Callable[..., object] | None,
) -> list[tuple[str, str, float]]:
    """One CoreNLP round trip -> ``[(label, sentence text, compound)]``.

    Raises ``ConnectionError`` (no server) or ``ValueError`` (unusable answer)
    — both map to diagnostics at the call site, never to a traceback.
    """
    import http.client
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    url = f"{server_url.rstrip('/')}/?properties=" + urllib.parse.quote(properties)
    if request_json is not None:
        payload = request_json(url, properties, text.encode("utf-8"), 30.0)
    else:
        request = urllib.request.Request(url, data=text.encode("utf-8"), method="POST")  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, http.client.BadStatusLine) as exc:
            raise ConnectionError(str(exc)) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("sentences"), list):
        raise ValueError("CoreNLP returned no sentences list")
    scores: list[tuple[str, str, float]] = []
    for sentence in payload["sentences"]:
        if not isinstance(sentence, dict):
            continue
        label = str(sentence.get("sentiment", "")).strip() or "Neutral"
        raw = sentence.get("sentimentValue")
        value = int(raw) if raw is not None else _CORENLP_VALUE.get(label.lower(), 2)
        tokens = sentence.get("tokens")
        if isinstance(tokens, list):
            sent_text = " ".join(str(t.get("word", "")) for t in tokens if isinstance(t, dict))
        else:
            sent_text = str(sentence.get("text", ""))
        # The RNTN's 0..4 scale is already signed around 2: Very negative .. Very positive.
        scores.append((label, sent_text, round((value - 2) / 2, 4)))
    if not scores:
        raise ValueError("CoreNLP returned no scored sentence")
    return scores


def corenlp_sentences(
    docs: Sequence[tuple[str, str, str]],
    *,
    server_url: str = "http://localhost:9000",
    request_json: Callable[..., object] | None = None,
) -> Result[pd.DataFrame]:
    """Per-sentence sentiment from Stanford CoreNLP's ``sentiment`` annotator.

    ``docs`` is ``(doc_id, doc_name, raw text)`` — CoreNLP tokenizes and splits
    server-side, so this backend needs no local parse. One row per SERVER
    sentence, which is also why the four backends' tables are parallel in
    columns and not in row counts: the sentence splitter is part of what HW5
    asks you to compare.
    """
    from core.pipelines.corenlp_backend import _START_FIX, _checked_url

    bad = _checked_url(server_url)
    if bad is not None:
        return Result.failure(
            Diagnostic.error(
                "CORENLP_BAD_URL",
                f"CoreNLP server URL must be http(s), got scheme {bad!r}",
                backend="corenlp",
            )
        )
    properties = '{"annotators":"tokenize,ssplit,parse,sentiment","outputFormat":"json"}'
    out: list[dict[str, object]] = []
    for doc_id, name, text in docs:
        try:
            scored = _corenlp_scores(server_url, properties, text, request_json)
        except (ConnectionError, ValueError) as exc:
            return Result.failure(
                Diagnostic.error(
                    "SENTIMENT_NN_UNAVAILABLE",
                    f"no CoreNLP sentiment from {server_url} ({exc})",
                    fix=_START_FIX,
                )
            )
        for index, (label, sent_text, compound) in enumerate(scored, start=1):
            out.append(
                {
                    "Document ID": doc_id,
                    "Document": name,
                    "Sentence ID": index,
                    "Sentence": sent_text,
                    "Label": label,
                    "Score": 1.0,
                    "Compound": compound,
                }
            )
    return _frame_from_scores(out)
