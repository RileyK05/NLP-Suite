"""Reviewed desktop workflows. Research registration never publishes a tool."""

from dataclasses import replace
import sys

from core.profiler.labels import tool_description
from core.profiler.registry import ToolSpec

# Keep experimental, demonstration, network and separately installed model
# workflows in the developer interfaces until they have a desktop contract.
CORPUS_TOOLS = (
    "readability",
    "lexical_diversity",
    "doc_similarity",
    "doc_duplicates",
    "spellcheck",
    "search",
    "sentence_complexity",
    "lda_gensim",
    "lda_stability",
    "lda_mallet",
    "word2vec_gensim",
    "sentiment_vader_anew",
    "nominalization",
    "style",
    "narrative",
    "clause_svo",
    "ner",
    "ngrams",
    "ngram_cooccurrence",
    "conll_wordlist",
    "corpus_statistics",
    "k_sentences",
    "svo_compare",
    "text_statistics",
    "table_search",
    "shapes",
    "kwic",
    "keyness",
    "collocations",
    "tfidf",
    "dispersion",
    "lexicon_series",
    "gender_annotator",
    "date_annotator",
    "quote_annotator",
    "gender_guess",
    "verb_analysis",
    "ngram_viewer",
    "sentiment_neural_bert",
    "sentiment_neural_spacy",
    "sentiment_neural_stanza",
    "sentiment_neural_corenlp",
    "shape_hc",
    "shape_svd",
    "shape_nmf",
    "geocode",
    "svo_map",
    "word2vec_bert",
    # BERT-family tools: the models ship with the app (core.models) and run
    # through ONNX Runtime, so they have a desktop contract now.
    "word_sense_induction",
    "bert_extract",
    "bert_topics",
    "doc_embeddings",
)

INTERNAL_TOOLS = frozenset(
    {
        "coreference",
        "semantic",
        "knowledge_graph",
        "sentiment_swn_hedono",
        "nrc",
    }
)


def desktop_spec(spec: ToolSpec) -> ToolSpec:
    """Packaged exports must work without a separately installed browser.

    Also the point where a registry description — written for whoever wires the
    tool up — is replaced by one written for whoever is choosing it. The three
    specs below say more than a description, so they keep their own wording.
    """
    if spec.name not in {"sentiment_vader_anew", "search", "spellcheck"}:
        spec = replace(spec, description=tool_description(spec.name, spec.description))
    if spec.name == "sentiment_vader_anew":
        return replace(
            spec,
            assets=(),
            description="Measure positive and negative tone in sentences and documents. VADER is included; ANEW needs your lexicon file.",
            params=tuple(
                replace(param, default="vader")
                if param.name == "analysis"
                else replace(param, help="Choose your ANEW ratings CSV when using both analyses.")
                for param in spec.params
                if param.name in {"analysis", "anew-lexicon"}
            ),
        )
    if spec.name == "search":
        return replace(
            spec,
            description="Find words or phrases in your documents, with optional case-sensitive or pattern matching.",
        )
    if spec.name == "spellcheck":
        return replace(
            spec,
            description="Find unfamiliar words and spelling suggestions using a wordlist you choose. Your documents stay unchanged.",
        )
    if spec.name == "table_charts" and getattr(sys, "frozen", False):
        return replace(
            spec,
            params=tuple(
                replace(param, choices=("html", "xlsx")) if param.name == "format" else param for param in spec.params
            ),
        )
    return spec
