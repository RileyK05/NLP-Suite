"""FR-7 child 1 (C2) — registry completion contract tests.

The six FR-4/FR-5 surfaces must be spec'd, and every spec parameter must
mirror a real CLI flag (drift between forms and flags is exactly what this
guards). Fails until the specs land (C3).
"""

from __future__ import annotations

from core.profiler.registry import TOOL_REGISTRY, get_tool, validate_specs

# spec name -> tools.<module> exposing _parse_args(argv) -> Namespace
NEW_TOOLS = {
    "wordnet": "tools.wordnet",
    "lda_gensim": "tools.lda_gensim",
    "lda_mallet": "tools.lda_mallet",
    "word2vec_gensim": "tools.word2vec_gensim",
    "sentiment_vader_anew": "tools.sentiment_vader_anew",
    "sentiment_swn_hedono": "tools.sentiment_swn_hedono",
    "nrc": "tools.nrc",
    "nominalization": "tools.nominalization",
    "style": "tools.style",
    "verbnet": "tools.verbnet",
    "framenet": "tools.framenet",
    "symbolic": "tools.symbolic",
    "coreference": "tools.coreference",
    "narrative": "tools.narrative",
    "charts": "tools.charts",
    "wordcloud_gephi": "tools.wordcloud_gephi",
    "word_sense_induction": "tools.word_sense_induction",
    "clause_svo": "tools.clause_svo",
    "ner": "tools.ner",
    "bert_extract": "tools.bert_extract",
    "kwic": "tools.kwic",
    "keyness": "tools.keyness",
    "bert_topics": "tools.bert_topics",
    "gender_annotator": "tools.gender_annotator",
    "date_annotator": "tools.date_annotator",
    "quote_annotator": "tools.quote_annotator",
    "gender_guess": "tools.gender_guess",
    "verb_analysis": "tools.verb_analysis",
    "ngram_viewer": "tools.ngram_viewer",
    "sentiment_neural_bert": "tools.sentiment_neural_bert",
    "sentiment_neural_spacy": "tools.sentiment_neural_spacy",
    "sentiment_neural_stanza": "tools.sentiment_neural_stanza",
    "sentiment_neural_corenlp": "tools.sentiment_neural_corenlp",
    "shape_hc": "tools.shape_hc",
    "shape_svd": "tools.shape_svd",
    "shape_nmf": "tools.shape_nmf",
    "geocode": "tools.geocode",
    "gis_map": "tools.gis_map",
    "svo_map": "tools.svo_map",
    "word2vec_bert": "tools.word2vec_bert",
}

# Spec params with no single CLI flag (each documented; the mirror test
# enforces this stays the complete list).
PARAM_EXCEPTIONS = {
    # wordnet selects up/down via subcommands, not a --mode flag
    ("wordnet", "mode"),
}


def _cli_flags(module_name: str) -> set[str]:  # noqa: PLR0912 -- one branch per CLI with non-standard argv
    import importlib

    module = importlib.import_module(module_name)
    if module_name == "tools.wordnet":
        argsets = [["up", "w", "o"], ["down", "o", "--keyword", "k"]]
    elif module_name == "tools.charts":
        argsets = [["in.csv", "o", "--x", "X", "--y", "Y"]]
    elif module_name == "tools.wordcloud_gephi":
        argsets = [["in.csv", "o", "--word-col", "W", "--weight-col", "C"]]
    elif module_name == "tools.symbolic":
        argsets = [["w", "o"], ["w", "o", "--analysis", "actor"]]
    elif module_name == "tools.style":
        argsets = [["corpus", "out"], ["corpus", "out", "--analysis", "iconicity"]]
    elif module_name == "tools.kwic":
        argsets = [["corpus", "out", "--query", "peace"]]
    elif module_name == "tools.keyness":
        argsets = [["corpus", "out", "--group-pattern", "g"]]
    elif module_name == "tools.bert_topics":
        argsets = [["corpus", "out"]]
    elif module_name == "tools.ngram_viewer":
        argsets = [["corpus", "out", "--queries", "war, peace"]]
    elif module_name == "tools.gis_map":
        argsets = [["in.csv", "o"]]
    elif module_name == "tools.framenet":
        argsets = [["w", "o"]]
    else:
        argsets = [["corpus", "out"]]
    seen: set[str] = set()
    for argv in argsets:
        namespace = module._parse_args(argv)
        seen |= {f"--{key.replace('_', '-')}" for key in vars(namespace)}
    return seen - {"--corpus", "--output", "--words", "--command", "--input"}


class TestNewSpecs:
    def test_specs_exist(self) -> None:
        for name in NEW_TOOLS:
            assert get_tool(name) is not None, f"missing spec: {name}"

    def test_registry_validates_clean(self) -> None:
        assert validate_specs(TOOL_REGISTRY) == ()

    def test_params_mirror_cli_flags(self) -> None:
        for name, module in NEW_TOOLS.items():
            spec = get_tool(name)
            assert spec is not None
            flags = _cli_flags(module)
            for param in spec.params:
                if (name, param.name) in PARAM_EXCEPTIONS:
                    continue
                assert f"--{param.name}" in flags, f"{name}.{param.name} has no CLI flag"

    def test_parse_tools_name_config_backend(self) -> None:
        for name in (
            "lda_gensim",
            "word2vec_gensim",
            "sentiment_vader_anew",
            "sentiment_swn_hedono",
            "nrc",
            "nominalization",
            "style",
            "coreference",
            "narrative",
            "word_sense_induction",
            "clause_svo",
            "ner",
            "bert_extract",
            "kwic",
            "keyness",
            "bert_topics",
            "gender_annotator",
            "quote_annotator",
            "gender_guess",
            "verb_analysis",
            "ngram_viewer",
            "sentiment_neural_bert",
            "sentiment_neural_spacy",
            "sentiment_neural_stanza",
            "shape_hc",
            "shape_svd",
            "shape_nmf",
            "geocode",
            "svo_map",
            "word2vec_bert",
        ):
            spec = get_tool(name)
            assert spec is not None
            assert spec.requires_parse and spec.parser_backend == "config"

    def test_wordnet_is_not_profiler_eligible(self) -> None:
        spec = get_tool("wordnet")
        assert spec is not None
        # word-list input (words CSV / keyword), not a corpus batch tool
        assert spec.profiler_eligible is False
