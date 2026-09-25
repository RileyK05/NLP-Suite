"""Declarative tool registry (FR-7.1) — prerequisite for UI forms and profiler.

Each tool declares its packet, capability refs (empty until the FR-2.x/3.x
rows are mapped to CAP IDs), human description, whether it needs a parse,
its parameters, and the artifacts it emits. Downstream (FR-7.2+) consumes
these declarations instead of re-discovering flags per tool.

Rules: the registry is data, not dispatch — it never imports tool modules
(R1: no import-time model work through the back door). Adding a tool updates
``TOOL_REGISTRY`` and the scope test together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.models.registry import ModelKind, models_of_kind
from core.result import Diagnostic
from core.viz.chartspec import CHART_KINDS
from core.viz.panels import panel_names

__all__ = [
    "AGG_FUNCS",
    "FAMILY_LABELS",
    "SHARED_CAPABILITIES",
    "TOOL_REGISTRY",
    "TOOL_REGISTRY_EXCLUSIONS",
    "ParamSpec",
    "ToolSpec",
    "get_tool",
    "tool_names",
    "validate_specs",
]

AGG_FUNCS: tuple[str, ...] = ("sum", "mean", "median", "count")

#: Syllabus-aligned families, in the order the course meets them. The
#: desktop gallery groups by these; a tool without one is a validation
#: error (see validate_specs), because ungrouped tools are exactly the
#: "weird chunking" this taxonomy exists to end.
FAMILY_LABELS: dict[str, str] = {
    "file_intake": "Files & intake",
    "corpus_statistics": "Corpus & table statistics",
    "words": "Words: n-grams, search & vocabulary",
    "style": "Style & readability",
    "topics": "Topic modeling",
    "embeddings": "Word embeddings & senses",
    "parsers_conll": "Parsers & the CoNLL table",
    "annotators": "Annotators & semantic lexicons",
    "gis": "Maps & geography",
    "narrative_svo": "Narrative & SVO",
    "sentiment": "Sentiment & emotion",
    "story_shape": "Shape of stories",
    "visualization": "Visualization",
    "utilities": "Utilities",
}

# The registry is intentionally the contract for profiler/UI-selectable tools,
# not an import-time inventory of every thin CLI. Every CLI that is not yet
# safe to select declaratively is named here with an explicit reason; silently
# omitting it would make coverage appear larger than it is.
TOOL_REGISTRY_EXCLUSIONS: tuple[tuple[str, str], ...] = (
    ("assets", "environment/reporting command; no analysis selection"),
    ("compare", "harness command; consumes run directories"),
    ("explain", "reads a finished result and recommends charts; operates on output, not a corpus"),
    ("visualizations", "catalog listing; describes what can be drawn rather than drawing it"),
    ("coreference", "optional/model-dependent backend"),
    ("corpus_validation", "intake validation command"),
    ("data_manipulation", "table-transform command with conditional schemas"),
    ("doctor", "environment/reporting command"),
    ("earth", "GIS artifact workflow; consumes a geocoded table"),
    ("freeze_legacy", "maintenance command"),
    ("geocode", "external-service workflow"),
    ("html_annotator", "standalone HTML artifact workflow"),
    ("install", "environment/reporting command"),
    ("jobs", "background runner over tool CLIs; not an analysis selection"),
    ("mapping", "GIS artifact workflow"),
    ("models", "environment/reporting command; read-only inventory"),
    ("pcace", "database workflow"),
    ("pcace_analysis", "database workflow"),
    ("sql", "database workflow"),
    ("unified", "meta-dispatcher over all CLIs; not an analysis selection"),
)

# A capability ID is the join key to docs/REPLACEMENT_LEDGER.md, so two specs
# claiming one ID normally means a copy-paste that quietly reassigns a ledger
# row to unrelated code. Where a capability genuinely has two entry points it
# is named here with its reason, exactly as exclusions are.
SHARED_CAPABILITIES: dict[str, str] = {
    "CAP-STATS-06": "correlations are surfaced by both csv_stats and stats_trends over the same engine",
    "CAP-SEM-01": "WordNet aggregation has a words-CSV entry point (wordnet) and a per-lemma one (semantic)",
}


@dataclass(frozen=True, slots=True)
class ParamSpec:
    name: str
    type: str  # str | int | float | bool | path
    default: object
    required: bool
    help: str
    choices: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    capability_ids: tuple[str, ...]
    packet: str  # implementing FR-x.y
    description: str
    requires_parse: bool
    params: tuple[ParamSpec, ...]
    outputs: tuple[str, ...]
    version: str
    input_kind: str = "corpus"  # corpus | csv | conll | database | asset | document | directory
    family: str = ""  # one of FAMILY_LABELS; how the UI groups the tool
    parser_backend: str = ""  # "" = none needed; else stanza|spacy|corenlp
    optional_package: str = ""  # extra that must be installed, if any
    assets: tuple[str, ...] = ()  # asset names from the asset registry
    profiler_eligible: bool = True  # can the batch profiler select it?
    execution_phase: int = 2  # 0=env 1=intake 2=analysis 3=post/aggregate
    parse_when: str = ""  # conditional parse requirement, e.g. mode == 'conll'


def _model_param(default: str, *kinds: ModelKind) -> ParamSpec:
    """The ``model`` choice: every registered model of the kinds this tool reads.

    A model added to :mod:`core.models.registry` appears here on its own; one
    that is not installed yet fails its run with "Open Models to add it".
    """
    return _flag(
        "model",
        "str",
        default,
        "model (add more on the Models page)",
        choices=tuple(spec.id for spec in models_of_kind(*kinds)),
    )


def _flag(  # noqa: PLR0913
    name: str,
    type: str,
    default: object,
    help: str,
    *,
    choices: tuple[object, ...] = (),
    minimum: float | None = None,
    maximum: float | None = None,
) -> ParamSpec:
    return ParamSpec(
        name=name,
        type=type,
        default=default,
        required=False,
        help=help,
        choices=choices,
        minimum=minimum,
        maximum=maximum,
    )


TOOL_REGISTRY: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="readability",
        family="style",
        capability_ids=("CAP-STATS-04",),
        packet="FR-2.1",
        description="Flesch/Fog/SMOG readability scores per document (raw text)",
        requires_parse=False,
        input_kind="corpus",
        params=(),
        outputs=("readability.csv",),
        version="1",
    ),
    ToolSpec(
        name="lexical_diversity",
        family="words",
        capability_ids=("CAP-STATS-03",),
        packet="FR-2.2",
        description="TTR, Guiraud, MTLD, vocd-D per document (raw text, seeded)",
        requires_parse=False,
        input_kind="corpus",
        params=(_flag("seed", "int", 42, "vocd sampling seed"),),
        outputs=("lexical_diversity.csv",),
        version="1",
    ),
    ToolSpec(
        name="doc_similarity",
        family="words",
        capability_ids=("CAP-SEM-04",),
        packet="FR-2.9",
        description="Pairwise TF-IDF document similarity + duplicate pairs",
        requires_parse=False,
        input_kind="corpus",
        optional_package="",
        params=(
            _flag("threshold", "float", 80.0, "duplicate threshold 0-100"),
            _flag("stopwords", "path", None, "optional stopwords file"),
        ),
        outputs=("doc_pairs.csv", "duplicates.csv"),
        version="1",
    ),
    ToolSpec(
        name="doc_duplicates",
        family="file_intake",
        capability_ids=("CAP-INTAKE-14",),
        packet="FR-3.3",
        description="Exact, normalized, and fuzzy duplicate matching",
        requires_parse=False,
        input_kind="corpus",
        optional_package="",
        params=(_flag("threshold", "float", 80.0, "fuzzy threshold 0-100"),),
        outputs=("exact_groups.csv", "normalized_groups.csv", "fuzzy_pairs.csv"),
        version="1",
    ),
    ToolSpec(
        name="spellcheck",
        family="words",
        capability_ids=("CAP-INTAKE-13",),
        packet="FR-3.2",
        description="Unknown-word findings against an explicit wordlist (+ optional corrected copies)",
        requires_parse=False,
        input_kind="corpus",
        assets=(),
        params=(
            ParamSpec(name="wordlist", type="path", default=None, required=True, help="wordlist file"),
            _flag("threshold", "float", 80.0, "suggestion threshold 0-100"),
            _flag("correct", "bool", False, "write corrected copies"),
        ),
        outputs=("spellcheck.csv",),
        version="1",
    ),
    ToolSpec(
        name="search",
        family="words",
        capability_ids=("CAP-INTAKE-11", "CAP-INTAKE-12"),
        packet="FR-3.4",
        description="Text, CSV-row, or CoNLL-row search with provenance (modes: text/csv/conll)",
        requires_parse=False,  # only conll mode parses; see parse_when
        parse_when="mode == 'conll'",
        params=(
            ParamSpec(
                name="mode",
                type="str",
                default=None,
                required=True,
                help="text | csv | conll",
                choices=("text", "csv", "conll"),
            ),
            ParamSpec(name="query", type="str", default=None, required=True, help="search pattern"),
            _flag("case-sensitive", "bool", False, "case-sensitive match"),
            _flag("regex", "bool", False, "treat query as regex"),
        ),
        outputs=("text_hits.csv", "csv_hits.csv", "conll_hits.csv"),
        version="1",
    ),
    ToolSpec(
        name="stats_categorical",
        family="corpus_statistics",
        capability_ids=("CAP-STATS-07",),
        packet="FR-2.3",
        description="Crosstabs, chi-square, keyness over a CSV file (tests: chi2/crosstab/keyness)",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # needs an analyst-chosen CSV file, not a corpus batch input
        params=(
            ParamSpec(
                name="test",
                type="str",
                default=None,
                required=True,
                help="chi2 | crosstab | keyness",
                choices=("chi2", "crosstab", "keyness"),
            ),
            _flag("alpha", "float", 0.05, "significance level in (0,1)", minimum=0.0, maximum=1.0),
        ),
        outputs=("chi2_summary.csv", "crosstab_counts.csv", "keyness.csv"),
        version="1",
    ),
    ToolSpec(
        name="stats_groups",
        family="corpus_statistics",
        capability_ids=("CAP-STATS-08",),
        packet="FR-2.4",
        description="Mann-Whitney and Kruskal-Wallis (+Dunn) over a CSV file (tests: mw/kw)",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # needs an analyst-chosen CSV file, not a corpus batch input
        params=(
            ParamSpec(name="test", type="str", default=None, required=True, help="mw | kw", choices=("mw", "kw")),
            _flag("alpha", "float", 0.05, "significance level in (0,1)", minimum=0.0, maximum=1.0),
            _flag(
                "posthoc-method", "str", "bonferroni", "Bonferroni or Holm correction", choices=("bonferroni", "holm")
            ),
        ),
        outputs=("mann_whitney_summary.csv", "kruskal_wallis_summary.csv"),
        version="1",
    ),
    ToolSpec(
        name="stats_trends",
        family="corpus_statistics",
        capability_ids=("CAP-STATS-09", "CAP-STATS-06"),
        packet="FR-2.5",
        description="Mann-Kendall trends and rank correlations over a CSV file (tests: trend/rankcorr)",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # needs an analyst-chosen CSV file, not a corpus batch input
        params=(ParamSpec(name="test", type="str", default=None, required=True, help="trend | rankcorr"),),
        outputs=("mann_kendall_summary.csv", "rank_correlation.csv"),
        version="1",
    ),
    ToolSpec(
        name="sentence_complexity",
        family="style",
        capability_ids=("CAP-SEM-06",),
        packet="FR-2.6",
        description="Per-sentence dependency complexity (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        params=(),
        outputs=("sentence_complexity.csv",),
        version="1",
    ),
    ToolSpec(
        name="convert",
        family="file_intake",
        capability_ids=("CAP-INTAKE-03", "CAP-INTAKE-04"),
        packet="FR-3.1",
        description="Document-to-text intake conversion with per-file report",
        requires_parse=False,
        input_kind="directory",
        optional_package="converters",
        profiler_eligible=False,
        execution_phase=1,
        params=(),
        outputs=("conversion_report.csv",),
        version="1",
    ),
    ToolSpec(
        name="filenames",
        family="file_intake",
        capability_ids=("CAP-INTAKE-10",),
        packet="FR-3.5",
        description="Preview-first filename standardization (mutates inputs only via --apply)",
        requires_parse=False,
        input_kind="directory",
        profiler_eligible=False,
        execution_phase=1,
        params=(_flag("apply", "bool", False, "actually rename (default previews)"),),
        outputs=(),
        version="1",
    ),
    ToolSpec(
        name="csv_stats",
        family="corpus_statistics",
        capability_ids=("CAP-STATS-06", "CAP-STATS-10"),
        packet="scaffold",
        description="CSV describe + Pearson/Spearman/Kendall correlations",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # needs an analyst-chosen CSV file, not a corpus batch input
        params=(),
        outputs=("csv_stats.csv", "correlation.csv", "correlation_matrix.csv"),
        version="1",
    ),
    ToolSpec(
        name="profiler",
        family="corpus_statistics",
        capability_ids=("CAP-PROF-01",),
        packet="scaffold",
        description="Batch profiler over the shared parse (plan + linked envelopes)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        profiler_eligible=False,  # the batch itself, not a batch member (no recursive profiles)
        params=(
            _flag("analyses", "str", None, "comma-separated tool names (default: all eligible)"),
            _flag("resume-from", "path", None, "prior run dir for fingerprint-gated reuse"),
        ),
        outputs=("batch.json", "report.md"),
        version="1",
        execution_phase=3,
    ),
    ToolSpec(
        name="wordnet",
        family="annotators",
        capability_ids=("CAP-SEM-01",),
        packet="FR-4.4",
        description="WordNet aggregation UP (words CSV) and hyponym expansion DOWN (modes: up/down)",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # word-list input, not a corpus batch tool
        params=(
            ParamSpec(
                name="mode",
                type="str",
                default=None,
                required=True,
                help="up (words CSV) | down (--keyword)",
                choices=("up", "down"),
            ),
            ParamSpec(
                name="pos", type="str", default="NOUN", required=False, help="NOUN | VERB", choices=("NOUN", "VERB")
            ),
            _flag("anchors", "str", "", "comma-separated anchor terms or synset names (up mode)"),
            _flag("keyword", "str", None, "anchor keyword (down mode)"),
        ),
        outputs=("wordnet_up.csv", "wordnet_up_frequency.csv", "wordnet_down.csv"),
        version="1",
    ),
    ToolSpec(
        name="lda_gensim",
        family="topics",
        capability_ids=("CAP-TOPIC-02",),
        packet="FR-5.7",
        description="Gensim LDA: topics, dominant topic per document, Intertopic Distance Map and lambda-ranked terms (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="topics",
        params=(
            _flag("field", "str", "lemma", "token column to model", choices=("form", "lemma")),
            _flag("topics", "int", 3, "number of topics", minimum=1),
            _flag("top-n", "int", 5, "top words per topic", minimum=1),
            _flag("seed", "int", 100, "LDA random seed"),
            _flag(
                "nouns-only",
                "bool",
                False,
                "model noun lemmas only (the shared rule: Penn NN* or Universal NOUN/PROPN)",
            ),
            _flag("keep-stopwords", "bool", False, "do not filter English stopwords"),
            _flag(
                "lambda",
                "float",
                0.6,
                "term relevance lambda (1 = topic probability, 0 = distinctiveness)",
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        outputs=(
            "topics.csv",
            "topics_dominant.csv",
            "terms_by_relevance.csv",
            "intertopic_distances.csv",
            "intertopic_map.html",
            "topic_flow.csv",
        ),
        version="2",
    ),
    ToolSpec(
        name="lda_stability",
        family="topics",
        capability_ids=("CAP-TOPIC-05",),
        packet="FR-5.7",
        description=(
            "Topic stability: refit the model at several random seeds, match topics back to the reference "
            "by top-word overlap (Jaccard), and report which topics come back — a topic only worth naming "
            "if it is reproducible (needs a parse)"
        ),
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="topics",
        params=(
            _flag("topics", "int", 3, "number of topics", minimum=1),
            _flag(
                "seeds",
                "str",
                "100,101,102",
                "random seeds to refit at (comma-separated; the first is the reference)",
            ),
            _flag("top-n", "int", 10, "top words used to match topics", minimum=1),
            _flag("passes", "int", 10, "training passes per refit", minimum=1),
            _flag(
                "stable-at",
                "float",
                0.5,
                "worst-seed overlap a topic needs to count as Stable",
                minimum=0.0,
                maximum=1.0,
            ),
            _flag(
                "nouns-only",
                "bool",
                False,
                "model noun lemmas only (the shared rule: Penn NN* or Universal NOUN/PROPN)",
            ),
        ),
        outputs=("summary.csv", "matches.csv"),
        version="1",
    ),
    ToolSpec(
        name="lda_mallet",
        family="topics",
        capability_ids=("CAP-TOPIC-03",),
        packet="FR-5.8",
        description="MALLET LDA: topic keys and document-topic composition (needs the MALLET binary)",
        requires_parse=False,
        input_kind="corpus",
        params=(
            _flag("topics", "int", 10, "number of topics", minimum=2),
            _flag("seed", "int", 42, "MALLET random seed"),
        ),
        outputs=("topics.csv", "topics_dominant.csv"),
        version="1",
    ),
    ToolSpec(
        name="word2vec_gensim",
        family="embeddings",
        capability_ids=("CAP-EMBED-01",),
        packet="FR-5.5",
        description="Seeded Gensim Word2Vec vectors, neighbours, t-SNE (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="topics",
        params=(
            _flag("field", "str", "lemma", "token column to model", choices=("form", "lemma")),
            _flag("min-count", "int", 2, "minimum word count", minimum=1),
            _flag("vector-size", "int", 100, "embedding dimensions", minimum=1),
            _flag("window", "int", 5, "context window", minimum=1),
            _flag("epochs", "int", 20, "training epochs", minimum=1),
            ParamSpec(name="sg", type="int", default=1, required=False, help="0 = CBOW, 1 = skip-gram", choices=(0, 1)),
            _flag("seed", "int", 42, "training seed"),
            _flag("remove-stopwords", "bool", False, "exclude English stopwords before training"),
            _flag("query", "str", None, "word to find neighbours for"),
            _flag("top-n", "int", 5, "neighbours returned", minimum=1),
        ),
        outputs=("vectors.csv", "neighbours.csv", "tsne.csv", "tsne.html"),
        # 2: vectors.csv names each word's class and tsne.csv its meaning
        # group, for the meaning figures (axes, groups, networks).
        version="2",
    ),
    ToolSpec(
        name="sentiment_vader_anew",
        family="sentiment",
        capability_ids=("CAP-SENT-01", "CAP-SENT-02"),
        packet="FR-4.2",
        description="VADER per-sentence + per-document (package scorer) and ANEW means (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="sentiment",
        assets=("anew-lexicon",),
        params=(
            _flag("analysis", "str", "both", "VADER alone or VADER with an ANEW lexicon", choices=("vader", "both")),
            ParamSpec(
                name="field",
                type="str",
                default="form",
                required=False,
                help="column for VADER",
                choices=("form", "lemma"),
            ),
            ParamSpec(
                name="anew-field",
                type="str",
                default="lemma",
                required=False,
                help="column for ANEW",
                choices=("form", "lemma"),
            ),
            _flag("vader-lexicon", "path", None, "legacy-exact VADER lexicon file"),
            _flag("anew-lexicon", "path", None, "ANEW CSV (defaults to the registry asset)"),
        ),
        outputs=("vader.csv", "vader_sentences.csv", "anew.csv"),
        version="1",
    ),
    ToolSpec(
        name="sentiment_swn_hedono",
        family="sentiment",
        capability_ids=("CAP-SENT-03", "CAP-SENT-04"),
        packet="FR-4.2",
        description="SentiWordNet means (NLTK) and hedonometer happiness (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="wordnet",
        assets=("hedonometer",),
        params=(
            ParamSpec(
                name="field",
                type="str",
                default="lemma",
                required=False,
                help="column for both analyses",
                choices=("form", "lemma"),
            ),
            _flag("hedonometer-lexicon", "path", None, "labMT JSON (defaults to the registry asset)"),
        ),
        outputs=("swn.csv", "hedonometer.csv"),
        version="1",
    ),
    ToolSpec(
        name="nrc",
        family="sentiment",
        capability_ids=("CAP-SENT-05",),
        packet="FR-4.3",
        description="NRC emotion proportions over ten fixed categories (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            ParamSpec(
                name="field",
                type="str",
                default="lemma",
                required=False,
                help="column for matching",
                choices=("form", "lemma"),
            ),
            _flag("lexicon", "path", None, "NRC JSON (defaults to asset, then nrclex bundled)"),
        ),
        outputs=("nrc.csv",),
        version="1",
    ),
    ToolSpec(
        name="nominalization",
        family="style",
        capability_ids=("CAP-SEM-05",),
        packet="FR-2.7",
        description="Deverbal nominalization detection via WordNet derivational morphology (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="wordnet",
        params=(
            ParamSpec(
                name="field",
                type="str",
                default="lemma",
                required=False,
                help="column for base-verb lookup",
                choices=("form", "lemma"),
            ),
            _flag("check-ending", "bool", True, "apply the nominal-suffix prefilter"),
            _flag("curated-list", "path", None, "one-word-per-row CSV bypassing the suffix filter"),
        ),
        outputs=("nominalization.csv", "nominalization_by_sentence.csv"),
        version="1",
    ),
    ToolSpec(
        name="style",
        family="style",
        capability_ids=("CAP-SEM-07",),
        packet="FR-2.8",
        description="Per-sentence concreteness + iconicity over user-supplied norm assets (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        assets=("brysbaert-concreteness", "iconicity-ratings"),
        params=(
            ParamSpec(
                name="analysis",
                type="str",
                default="concreteness",
                required=False,
                help="which norms to score",
                choices=("concreteness", "iconicity"),
            ),
            ParamSpec(
                name="field",
                type="str",
                default="lemma",
                required=False,
                help="column for norm lookup",
                choices=("form", "lemma"),
            ),
            _flag("concreteness-lexicon", "path", None, "Brysbaert CSV (defaults to the registry asset)"),
            _flag("iconicity-lexicon", "path", None, "Winter CSV (defaults to the registry asset)"),
            _flag("min-rating", "float", 5.0, "most-iconic word threshold"),
            _flag("max-rating-sd", "float", 2.0, "most-iconic word SD ceiling"),
        ),
        outputs=("style_concreteness.csv", "style_iconicity.csv", "style_iconic_words.csv"),
        version="1",
    ),
    ToolSpec(
        name="verbnet",
        family="annotators",
        capability_ids=("CAP-SEM-02",),
        packet="FR-4.5",
        description="VerbNet class aggregation over a verb-list CSV (first class wins)",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # word-list input, not a corpus batch tool
        optional_package="wordnet",
        params=(),
        outputs=("verbnet.csv", "verbnet_frequency.csv"),
        version="1",
    ),
    ToolSpec(
        name="framenet",
        family="annotators",
        capability_ids=("CAP-SEM-03",),
        packet="FR-4.5",
        description="FrameNet frame aggregation over a word-list CSV (first frame wins)",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # word-list input, not a corpus batch tool
        optional_package="wordnet",
        params=(
            ParamSpec(
                name="pos", type="str", default="NOUN", required=False, help="NOUN | VERB", choices=("NOUN", "VERB")
            ),
        ),
        outputs=("framenet.csv", "framenet_frequency.csv"),
        version="1",
    ),
    ToolSpec(
        name="symbolic",
        family="annotators",
        capability_ids=("CAP-GIS-06",),
        packet="FR-4.6",
        description="Symbolic space + social actor typologies over a word-list CSV",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # word-list input, not a corpus batch tool
        optional_package="wordnet",
        assets=("space-typology", "actor-typology"),
        params=(
            ParamSpec(
                name="analysis",
                type="str",
                default="space",
                required=False,
                help="which typology to apply",
                choices=("space", "actor"),
            ),
            _flag("space-lexicon", "path", None, "space typology CSV (defaults to the registry asset)"),
            _flag("actor-lexicon", "path", None, "actor typology CSV (defaults to the registry asset)"),
        ),
        outputs=(
            "symbolic_space.csv",
            "symbolic_space_frequency.csv",
            "symbolic_actor.csv",
            "symbolic_actor_frequency.csv",
        ),
        version="1",
    ),
    ToolSpec(
        name="coreference",
        family="annotators",
        capability_ids=("CAP-COREF-01",),
        packet="FR-5.4",
        description="Deterministic lemma coreference baseline over noun mentions (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            ParamSpec(
                name="field",
                type="str",
                default="lemma",
                required=False,
                help="column for mention grouping",
                choices=("form", "lemma"),
            ),
        ),
        outputs=("coreference.csv",),
        version="1",
    ),
    ToolSpec(
        name="narrative",
        family="narrative_svo",
        capability_ids=("CAP-NARR-01", "CAP-NARR-02", "CAP-NARR-03"),
        packet="FR-6.3",
        description="Per-sentence emotion (VADER) + length arcs (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="sentiment",
        params=(
            ParamSpec(
                name="field",
                type="str",
                default="form",
                required=False,
                help="column for VADER scoring",
                choices=("form", "lemma"),
            ),
        ),
        outputs=("emotion_arc.csv", "length_arc.csv", "character_mentions.csv", "character_arcs.csv"),
        version="1",
    ),
    ToolSpec(
        name="charts",
        family="visualization",
        capability_ids=("CAP-VIZ-01", "CAP-VIZ-02"),
        packet="FR-6.5",
        description=(
            "Publication-ready charts (bar/line/scatter/histogram/box/heatmap plus pie/sunburst/"
            "treemap/violin/radar/waffle/calendar) over an analyst CSV "
            "with explicit aggregation/normalization/rate semantics; export as HTML, PNG/SVG/PDF "
            "(kaleido) or a native Excel workbook (openpyxl, legacy-parity)"
        ),
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # needs an analyst-chosen CSV file, not a corpus batch input
        optional_package="plotly",
        params=(
            ParamSpec(
                name="kind",
                type="str",
                default="bar",
                required=False,
                help="chart kind",
                # Derived from the engine rather than restated, so a new kind
                # reaches the desktop and the profiler without a second edit.
                choices=CHART_KINDS,
            ),
            ParamSpec(name="x", type="str", default=None, required=True, help="x column"),
            ParamSpec(name="y", type="str", default=None, required=True, help="y column"),
            _flag(
                "group",
                "str",
                None,
                "group column (color/series; heatmap row axis; sunburst/treemap inner ring)",
            ),
            _flag("agg", "str", None, "aggregation for duplicates: sum|mean|median|count", choices=AGG_FUNCS),
            _flag(
                "top-n",
                "int",
                None,
                "keep n categories by total (bar/line/pie/sunburst/treemap/waffle)",
                minimum=1,
            ),
            _flag(
                "normalize", "str", "none", "percent|share (bar/line/histogram)", choices=("none", "percent", "share")
            ),
            _flag(
                "scale-by",
                "str",
                "total",
                "normalization level: total|group|category (bar/line)",
                choices=("none", "total", "group", "category"),
            ),
            _flag(
                "rate-per", "float", None, "rate multiplier (with --denominator-column, --agg sum)", minimum=0.000001
            ),
            _flag("denominator-column", "str", None, "numeric denominator column for rates"),
            _flag("horizontal", "bool", False, "horizontal bars (bar only)"),
            _flag(
                "bar-mode",
                "str",
                "group",
                "grouped bar layout: group (side-by-side, default) | stack | relative",
                choices=("group", "stack", "relative"),
            ),
            _flag("bins", "int", None, "histogram bin count", minimum=1),
            _flag("title", "str", "", "chart title"),
            _flag("subtitle", "str", "", "chart subtitle"),
            _flag("x-label", "str", "", "x axis label override"),
            _flag("y-label", "str", "", "y axis label override"),
            _flag("width", "int", 900, "figure width in px", minimum=200),
            _flag("height", "int", 500, "figure height in px", minimum=200),
            _flag("wrap-labels", "int", 24, "wrap x tick labels at this width", minimum=1),
            _flag("x-tick-angle", "int", None, "explicit tick label angle in degrees (default per-orientation)"),
            _flag("delimiter", "str", ",", "input delimiter: , ; tab | space"),
            _flag("encoding", "str", "utf-8-sig", "input text encoding"),
            _flag(
                "format",
                "str",
                "html",
                "output format: html|png|svg|pdf (png/svg/pdf need [plotly-image]) | xlsx native Excel workbook ([excel])",
                choices=("html", "png", "svg", "pdf", "xlsx"),
            ),
            _flag(
                "browser-path",
                "str",
                None,
                "per-call kaleido browser override (invalid path fails); default BROWSER_PATH env or auto-detect",
            ),
            _flag("cdn", "bool", False, "reference plotly.js from CDN instead of inlining"),
            _flag("json", "bool", False, "print one JSON result object on stdout"),
        ),
        outputs=("chart.html", "chart_data.csv", "chart.png", "chart.xlsx"),
        version="2",
    ),
    ToolSpec(
        name="panels",
        family="visualization",
        capability_ids=("CAP-VIZ-07", "CAP-VIZ-08"),
        packet="FR-6.5",
        description=(
            "Tool-specific figures over an analysis CSV: a panel that knows one analysis's result shape, "
            "for results a generic x/y/group chart cannot express. Every mark carries the evidence it "
            "stands for and every figure carries its provenance caption inside the image "
            "(HTML, PNG/SVG/PDF via kaleido)"
        ),
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # needs an analyst-chosen CSV file, not a corpus batch input
        optional_package="plotly",
        params=(
            ParamSpec(
                name="panel",
                type="str",
                default=None,
                required=True,
                help="which panel to draw",
                # Derived from the panel registry rather than restated, so a
                # new panel reaches the desktop and the CLI without a second
                # edit -- the same rule as CHART_KINDS above.
                choices=panel_names(),
            ),
            _flag(
                "set",
                "str",
                None,
                "a panel parameter as NAME=VALUE, repeatable; run --list to see what each panel takes",
            ),
            _flag(
                "format",
                "str",
                "none",
                "also export a static image: png|svg|pdf (needs [plotly-image])",
                choices=("none", "png", "svg", "pdf"),
            ),
            _flag(
                "browser-path",
                "str",
                None,
                "per-call kaleido browser override (invalid path fails); default BROWSER_PATH env or auto-detect",
            ),
            _flag("cdn", "bool", False, "reference plotly.js from CDN instead of inlining"),
            _flag("delimiter", "str", ",", "input delimiter: , ; tab |"),
            _flag("encoding", "str", "utf-8-sig", "input text encoding"),
            _flag("list", "bool", False, "list the registered panels and exit"),
            _flag("json", "bool", False, "print one JSON result object on stdout"),
        ),
        outputs=("panel.html", "panel_data.csv", "panel.png"),
        version="1",
    ),
    ToolSpec(
        name="wordcloud_gephi",
        family="visualization",
        capability_ids=("CAP-VIZ-05", "CAP-VIZ-06"),
        packet="FR-6.6",
        description="Wordcloud HTML + raster PNG + validated GEXF network over an analyst-chosen CSV",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # needs an analyst-chosen CSV file, not a corpus batch input
        optional_package="wordcloud",
        params=(
            ParamSpec(name="word-col", type="str", default=None, required=True, help="word column"),
            ParamSpec(name="weight-col", type="str", default=None, required=True, help="weight column"),
            _flag("source-col", "str", None, "GEXF source column (enables graph output)"),
            _flag("target-col", "str", None, "GEXF target column (enables graph output)"),
            _flag("title", "str", "", "wordcloud title"),
            _flag("image", "bool", False, "also render a raster PNG wordcloud (needs [wordcloud] extra)"),
            _flag("mask", "path", None, "image mask: words render in non-white regions"),
            _flag("shape", "str", None, "procedural mask shape instead of --mask (available: butterfly)"),
            _flag("group-col", "str", None, "color words by this column (with --colors)"),
            _flag("colors", "str", "", "group=hexcolor pairs, e.g. 'Subject=#ff0000,Verb=#0000ff'"),
            _flag("max-words", "int", 200, "maximum words in the cloud (image mode)", minimum=1),
            _flag("width", "int", 800, "image width in px", minimum=64),
            _flag("height", "int", 800, "image height in px", minimum=64),
            _flag("colormap", "str", "viridis", "matplotlib colormap (image mode, no --group-col)"),
            _flag("background", "str", "white", "background color (image mode)"),
            _flag("seed", "int", 42, "layout RNG seed (image mode)"),
        ),
        outputs=("wordcloud.html", "wordcloud.png", "graph.gexf"),
        version="2",
    ),
    ToolSpec(
        name="word_sense_induction",
        family="embeddings",
        capability_ids=("CAP-EMBED-02", "CAP-EMBED-03"),
        packet="FR-5.6",
        description="BERT token embeddings and word senses: which words split into two uses, with the sentences "
        "behind each (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="embeddings",
        params=(
            ParamSpec(
                name="field",
                type="str",
                default="form",
                required=False,
                help="column for vectors",
                choices=("form", "lemma"),
            ),
            _model_param("bert-base-uncased", "token_embeddings"),
        ),
        outputs=("contextual_vectors.csv", "wsi.csv", "senses.csv"),
        # 2: each word is read at its own surface form in the sentence (a
        # lemma was read as the whole sentence), through ONNX Runtime.
        # 3: wsi.csv says how clearly each split separates, and senses.csv
        # carries the sentences behind each sense.
        version="3",
    ),
    ToolSpec(
        name="clause_svo",
        family="parsers_conll",
        capability_ids=("CAP-CONLL-02", "CAP-CONLL-04"),
        packet="FR-6.1",
        description="Clause-tag frequencies + SVO triples over the shared parse (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(),
        outputs=("clauses.csv", "svo.csv"),
        version="1",
    ),
    ToolSpec(
        name="ner",
        family="annotators",
        capability_ids=("CAP-NER-01", "CAP-NER-02", "CAP-NER-03"),
        packet="FR-6.4",
        description="NER entity timeline + location tracking (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(),
        outputs=("entity_timeline.csv", "locations.csv", "movement_tracks.csv", "movement_summary.csv"),
        version="1",
    ),
    ToolSpec(
        name="bert_extract",
        family="narrative_svo",
        capability_ids=("CAP-SEM-08",),
        packet="FR-5.9",
        description="Centroid extractive summarization over transformer sentence embeddings (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="embeddings",
        params=(
            _flag("sentences", "int", 3, "summary sentences per document", minimum=1),
            _model_param("bert-base-uncased", "token_embeddings", "sentence_embeddings"),
        ),
        outputs=("bert_extract.csv",),
        # 2: sentence vectors through ONNX Runtime; sentence models accepted.
        version="2",
    ),
    ToolSpec(
        name="ngrams",
        family="words",
        capability_ids=("CAP-NGRAM-01",),
        packet="FR-2.5",
        description="N-gram frequency tables over form or lemma (1..5-grams, needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("n", "int", 2, "n-gram size (1..5)", minimum=1),
            _flag("field", "str", "form", "token field", choices=("form", "lemma")),
            _flag("min-count", "int", 2, "minimum count for collocations", minimum=1),
        ),
        outputs=("ngrams.csv", "collocations.csv"),
        version="1",
    ),
    ToolSpec(
        name="ngram_cooccurrence",
        family="words",
        capability_ids=("CAP-NGRAM-02",),
        packet="FR-2.5",
        description="Word-pair co-occurrence counts within a sentence window (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("window", "int", 5, "sentence window size", minimum=1),
            _flag("field", "str", "lemma", "token field", choices=("form", "lemma")),
            _flag("min-count", "int", 1, "minimum pair count", minimum=1),
        ),
        outputs=("cooccurrences.csv",),
        version="1",
    ),
    ToolSpec(
        name="conll_wordlist",
        family="parsers_conll",
        capability_ids=("CAP-CONLL-01",),
        packet="FR-2.5",
        description="Word frequency list with POS-category filter over the parse (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("field", "str", "form", "token field", choices=("form", "lemma")),
            _flag(
                "category",
                "str",
                "all",
                "POS filter",
                choices=("all", "noun", "verb", "adjective", "adverb", "function", "noun-verb"),
            ),
            _flag("top-n", "int", 20, "keep the top n words (0 for all)", minimum=0),
            _flag("case-sensitive", "bool", False, "do not lower-case words"),
        ),
        outputs=("wordlist.csv",),
        version="1",
    ),
    ToolSpec(
        name="corpus_statistics",
        family="corpus_statistics",
        capability_ids=("CAP-STATS-02",),
        packet="FR-2.6",
        description="Per-document lexical diversity (TTR, RootTTR, Herdan, Yule K) over the parse (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(_flag("field", "str", "form", "token field", choices=("form", "lemma")),),
        outputs=("corpus_statistics.csv",),
        version="1",
    ),
    ToolSpec(
        name="k_sentences",
        family="words",
        capability_ids=("CAP-CONLL-07",),
        packet="FR-2.5",
        description="First/last K sentence windows per document (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("k-first", "int", 2, "K for first sentences", minimum=1),
            _flag("k-last", "int", 2, "K for last sentences", minimum=1),
        ),
        outputs=("k_sentences.csv",),
        version="1",
    ),
    ToolSpec(
        name="svo_compare",
        family="narrative_svo",
        capability_ids=("CAP-CONLL-05",),
        packet="FR-2.5",
        description="Pairwise SVO triple similarity (Jaccard) across documents (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(),
        outputs=("svo_compare.csv",),
        version="1",
    ),
    ToolSpec(
        name="text_statistics",
        family="corpus_statistics",
        capability_ids=("CAP-VIZ-03",),
        packet="FR-2.5",
        description="Per-document sentence/word/syllable summary over the parse (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(),
        outputs=("text_statistics.csv",),
        version="1",
    ),
    ToolSpec(
        name="table_search",
        family="parsers_conll",
        capability_ids=("CAP-CONLL-06",),
        packet="FR-2.5",
        description="Search the CoNLL table with a predicate (eq/contains/starts_with/ends_with/regex)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("field", "str", None, "CoNLL column to search (e.g. Form, Lemma, POS)"),
            _flag("value", "str", None, "value to match"),
            _flag(
                "op", "str", "eq", "match operation", choices=("eq", "contains", "starts_with", "ends_with", "regex")
            ),
            _flag("logic", "str", "AND", "AND | OR (single predicate)", choices=("AND", "OR")),
            _flag("case-sensitive", "bool", False, "case-sensitive match"),
            _flag("negate", "bool", False, "negate the predicate"),
        ),
        outputs=("search_results.csv",),
        version="1",
    ),
    ToolSpec(
        name="semantic",
        family="annotators",
        capability_ids=("CAP-SEM-01",),
        packet="FR-4.6",
        description="Per-lemma semantic tags from the baked semantic maps (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(_flag("field", "str", "lemma", "token field", choices=("form", "lemma")),),
        outputs=("semantic_tags.csv",),
        version="1",
    ),
    ToolSpec(
        name="knowledge_graph",
        family="annotators",
        capability_ids=("CAP-KG-01",),
        packet="FR-6.8",
        description="Knowledge-graph triples over known entities (offline stub; DBpedia needs --source dbpedia)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("field", "str", "form", "token field", choices=("form", "lemma")),
            _flag("source", "str", "stub", "offline stub or live DBpedia Spotlight", choices=("stub", "dbpedia")),
            _flag("confidence", "float", 0.5, "Spotlight confidence bound (dbpedia source)", minimum=0, maximum=1),
        ),
        outputs=("knowledge_graph.csv",),
        version="1",
    ),
    ToolSpec(
        name="shapes",
        family="story_shape",
        capability_ids=("CAP-VIZ-04",),
        packet="FR-6.2",
        description="Story-shape sentence metrics (tokens, noun/verb ratios) + optional Sankey HTML (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("sankey-source", "str", None, "source column for the Sankey diagram (optional)"),
            _flag("sankey-target", "str", None, "target column for the Sankey diagram (optional)"),
            _flag("sankey-value", "str", "", "value column for the Sankey diagram (optional)"),
        ),
        outputs=("story_shape.csv", "sankey.html", "shape_clusters.csv"),
        version="1",
    ),
    ToolSpec(
        name="kwic",
        family="words",
        capability_ids=("CAP-INTAKE-19",),
        packet="FR-3.4",
        description="KWIC concordance: every match of a query with left/right context (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            ParamSpec(name="query", type="str", default=None, required=True, help="word, fragment, or regex"),
            _flag("field", "str", "form", "token field", choices=("form", "lemma")),
            _flag("window", "int", 5, "context tokens per side", minimum=1, maximum=50),
            _flag("max-hits", "int", 1000, "stop after this many matches", minimum=1),
            _flag("case-sensitive", "bool", False, "match case exactly"),
            _flag("regex", "bool", False, "treat the query as a regular expression"),
        ),
        outputs=("kwic.csv",),
        version="1",
    ),
    ToolSpec(
        name="dispersion",
        family="words",
        capability_ids=("CAP-STATS-13",),
        packet="FR-2.9",
        description="Lexical dispersion: how evenly each word spreads across the corpus (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("field", "str", "lemma", "token field", choices=("form", "lemma")),
            _flag("parts", "str", "document", "split by document or into equal chunks", choices=("document", "chunk")),
            _flag("chunks", "int", 10, "number of chunks when parts=chunk", minimum=2),
            _flag("min-count", "int", 5, "drop terms rarer than this", minimum=1),
            _flag("min-length", "int", 1, "drop tokens shorter than this", minimum=1),
            _flag("top-n", "int", 500, "rows kept after sorting by frequency (0 = all)", minimum=0),
            _flag("plot", "bool", False, "also write a dispersion plot showing where each word falls"),
        ),
        outputs=("dispersion.csv", "dispersion_plot.html"),
        version="1",
    ),
    ToolSpec(
        name="lexicon_series",
        family="words",
        capability_ids=("CAP-STATS-14",),
        packet="FR-2.9",
        description="Named word groups counted across a corpus axis, as rates per 1,000 tokens (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            ParamSpec(
                name="terms",
                type="str",
                default=None,
                required=True,
                help="word groups: 'Iraq: iraq, iraqi, saddam; Vietnam: vietnam, hanoi' (phrases allowed)",
            ),
            # Spelled out rather than imported: the registry is data and does
            # not import tool modules. A test pins these to lexicon_series.BY_CHOICES.
            _flag("by", "str", "year", "the axis to count along", choices=("year", "decade", "document", "pattern")),
            _flag(
                "group-pattern",
                "str",
                "",
                "regex over document names when by=pattern; its first group names the axis value",
            ),
            _flag("field", "str", "lemma", "token field", choices=("form", "lemma")),
            _flag(
                "within",
                "str",
                "",
                "only count inside sentences matching these word groups, e.g. 'Iraq: iraq, saddam'",
            ),
        ),
        outputs=("lexicon_series.csv",),
        version="1",
    ),
    ToolSpec(
        name="tfidf",
        family="words",
        capability_ids=("CAP-STATS-12",),
        packet="FR-2.9",
        description="TF-IDF distinctive terms per document (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("field", "str", "lemma", "token field", choices=("form", "lemma")),
            _flag("top-n", "int", 20, "terms kept per document", minimum=1),
            _flag("min-df", "int", 1, "drop terms in fewer documents than this", minimum=1),
            _flag(
                "max-df-ratio",
                "float",
                1.0,
                "drop terms in more than this fraction of documents",
                minimum=0.0,
                maximum=1.0,
            ),
            _flag("min-length", "int", 1, "drop tokens shorter than this", minimum=1),
            _flag("sublinear-tf", "bool", False, "use 1 + log(count) instead of the raw count"),
            _flag("no-normalize", "bool", False, "skip L2 normalization per document"),
        ),
        outputs=("tfidf.csv",),
        version="1",
    ),
    ToolSpec(
        name="collocations",
        family="words",
        capability_ids=("CAP-NGRAM-04",),
        packet="FR-2.3",
        description="Collocation association measures: PMI, t-score, G2, Dice over word pairs (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("field", "str", "lemma", "token field", choices=("form", "lemma")),
            _flag("span", "str", "adjacent", "adjacent bigrams or a token window", choices=("adjacent", "window")),
            _flag("window", "int", 5, "window size when span=window", minimum=1, maximum=25),
            _flag("min-count", "int", 3, "drop pairs seen fewer times than this", minimum=1),
            _flag("min-length", "int", 1, "drop tokens shorter than this", minimum=1),
            _flag("top-n", "int", 200, "rows kept after sorting by G2 (0 = all)", minimum=0),
            _flag("case-sensitive", "bool", False, "do not casefold tokens"),
        ),
        outputs=("collocations.csv",),
        version="1",
    ),
    ToolSpec(
        name="keyness",
        family="words",
        capability_ids=("CAP-STATS-11",),
        packet="FR-2.3",
        description="G2 log-likelihood keyness between two text groups (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            ParamSpec(
                name="group-pattern",
                type="str",
                default=None,
                required=True,
                help="regex over document names; group A matches, group B is the rest",
            ),
            _flag("field", "str", "lemma", "token field", choices=("form", "lemma")),
            _flag("smoothing", "float", 0.5, "Log Ratio smoothing (Hardie 0.5)", minimum=0.000001),
            _flag("top-n", "int", 200, "keyness rows kept (0 = all)", minimum=0),
        ),
        outputs=("keyness.csv",),
        version="1",
    ),
    ToolSpec(
        name="bert_topics",
        family="topics",
        capability_ids=("CAP-TOPIC-04",),
        packet="FR-5.9",
        description="KMeans topics over BERT document embeddings (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="embeddings",
        params=(
            _flag("topics", "int", 3, "number of topics", minimum=2),
            _flag("top-n", "int", 5, "top terms per topic", minimum=1),
            _flag("seed", "int", 42, "KMeans seed"),
            _model_param("bert-base-uncased", "token_embeddings", "sentence_embeddings"),
        ),
        outputs=("bert_topics.csv", "bert_topic_docs.csv"),
        # 2: sentence vectors through ONNX Runtime; sentence models accepted.
        version="2",
    ),
    ToolSpec(
        name="gender_annotator",
        family="annotators",
        capability_ids=("CAP-ANNO-10",),
        packet="FR-4.8",
        description="name-gender annotation from a choice of four dictionaries (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag(
                "dictionary",
                "str",
                "nltk",
                "name dictionary: census, carnegie_mellon, nltk, social_security",
                choices=("census", "carnegie_mellon", "nltk", "social_security"),
            ),
            _flag("names-dir", "path", None, "folder holding <dictionary>/male.txt and female.txt"),
            _flag(
                "source", "str", "ner", "where names come from: NER PERSON or PROPN tokens", choices=("ner", "propn")
            ),
        ),
        outputs=("gender_names.csv", "gender_summary.csv"),
        version="1",
    ),
    ToolSpec(
        name="date_annotator",
        family="annotators",
        capability_ids=("CAP-ANNO-11",),
        packet="FR-4.8",
        description="normalized date extraction and ISO date/times (raw text, no parse)",
        requires_parse=False,
        input_kind="corpus",
        params=(
            _flag("min-year", "int", 1000, "earliest plausible year", minimum=0, maximum=3000),
            _flag("max-year", "int", 2100, "latest plausible year", minimum=0, maximum=3000),
        ),
        outputs=("dates.csv", "dates_summary.csv"),
        version="1",
    ),
    ToolSpec(
        name="quote_annotator",
        family="annotators",
        capability_ids=("CAP-ANNO-12",),
        packet="FR-4.8",
        description="quote/dialogue extraction with a two-stage speaker sieve (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(_flag("min-length", "int", 1, "fewest words for a span to count as a quote", minimum=1),),
        outputs=("quotes.csv", "quotes_summary.csv"),
        version="1",
    ),
    ToolSpec(
        name="gender_guess",
        family="style",
        capability_ids=("CAP-STYLE-10",),
        packet="FR-2.10",
        description="writing-style gender guesses from weak-word rates, own transparent lexicon (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag(
                "mode",
                "str",
                "both",
                "which scoring axis: formal, informal, or both",
                choices=("formal", "informal", "both"),
            ),
        ),
        outputs=("gender_guess.csv", "gender_guess_summary.csv"),
        version="1",
    ),
    ToolSpec(
        name="verb_analysis",
        family="parsers_conll",
        capability_ids=("CAP-CONLL-10",),
        packet="FR-5.10",
        description="verb modality, tense and voice per verb token from the shared parse (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag(
                "analysis",
                "str",
                "all",
                "facet to summarize: all, modality, tense, or voice",
                choices=("all", "modality", "tense", "voice"),
            ),
        ),
        outputs=("verbs.csv", "verb_summary.csv"),
        version="1",
    ),
    ToolSpec(
        name="ngram_viewer",
        family="words",
        capability_ids=("CAP-NGRAM-10",),
        packet="FR-1.6",
        description="culturomics n-gram frequency per year from dated filenames; table + line chart (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            ParamSpec(
                name="queries",
                type="str",
                default=None,
                required=True,
                help="comma-separated words or phrases to track",
            ),
            _flag("case-sensitive", "bool", False, "distinguish capitalized forms"),
            _flag("smooth", "int", 1, "moving-average window in years", minimum=1),
        ),
        outputs=("ngram_series.csv", "ngram_viewer.html"),
        version="2",
    ),
    ToolSpec(
        name="sentiment_neural_bert",
        family="sentiment",
        capability_ids=("CAP-SENT-10",),
        packet="FR-4.9",
        description="neural sentiment per sentence via a BERT classifier: positive or negative, no neutral (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="embeddings",
        params=(_model_param("distilbert-sst2", "classifier"),),
        outputs=("sentiment_sentences.csv", "sentiment_documents.csv"),
        # 2: the model runs through ONNX Runtime, batched.
        version="2",
    ),
    ToolSpec(
        name="sentiment_neural_spacy",
        family="sentiment",
        capability_ids=("CAP-SENT-11",),
        packet="FR-4.9",
        description="neural sentiment per sentence via a spaCy textcat head (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="spacy",
        params=(_flag("model", "str", "en_core_web_sm", "spaCy pipeline with a sentiment textcat head"),),
        outputs=("sentiment_sentences.csv", "sentiment_documents.csv"),
        version="1",
    ),
    ToolSpec(
        name="sentiment_neural_stanza",
        family="sentiment",
        capability_ids=("CAP-SENT-12",),
        packet="FR-4.9",
        description="neural sentiment per sentence via Stanza's sentiment processor (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="stanza",
        params=(_flag("language", "str", "en", "Stanza language package carrying the sentiment processor"),),
        outputs=("sentiment_sentences.csv", "sentiment_documents.csv"),
        version="1",
    ),
    ToolSpec(
        name="sentiment_neural_corenlp",
        family="sentiment",
        capability_ids=("CAP-SENT-13",),
        packet="FR-4.9",
        description="neural sentiment via the CoreNLP sentiment annotator (needs the Java server)",
        requires_parse=False,
        input_kind="corpus",
        params=(_flag("server", "str", "http://localhost:9000", "Stanford CoreNLP server URL"),),
        outputs=("sentiment_sentences.csv", "sentiment_documents.csv"),
        version="1",
    ),
    ToolSpec(
        name="shape_hc",
        family="story_shape",
        capability_ids=("CAP-SHAPE-10",),
        packet="FR-6.8",
        description="hierarchical clustering of story-shape trajectories with a dendrogram (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag(
                "method",
                "str",
                "ward",
                "linkage method: ward, average, complete, single",
                choices=("ward", "average", "complete", "single"),
            ),
            _flag("n-clusters", "int", 2, "clusters to cut the tree into", minimum=2),
            _flag("resample", "int", 32, "points per document trajectory", minimum=4),
        ),
        outputs=(
            "shape_matrix.csv",
            "shape_hc_assignments.csv",
            "shape_hc_merges.csv",
            "shape_hc_dendrogram.html",
        ),
        version="1",
    ),
    ToolSpec(
        name="shape_svd",
        family="story_shape",
        capability_ids=("CAP-SHAPE-11",),
        packet="FR-6.8",
        description="SVD shape components over story-shape trajectories (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("n-components", "int", 2, "components to keep", minimum=1),
            _flag("seed", "int", 42, "randomized SVD seed"),
            _flag("resample", "int", 32, "points per document trajectory", minimum=4),
        ),
        outputs=(
            "shape_matrix.csv",
            "shape_svd_scores.csv",
            "shape_svd_loadings.csv",
            "shape_svd_explained.csv",
            "shape_svd_components.html",
        ),
        version="1",
    ),
    ToolSpec(
        name="shape_nmf",
        family="story_shape",
        capability_ids=("CAP-SHAPE-12",),
        packet="FR-6.8",
        description="non-negative shape parts over story-shape trajectories (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag("n-components", "int", 2, "non-negative components to extract", minimum=1),
            _flag("seed", "int", 42, "NMF init seed"),
            _flag("resample", "int", 32, "points per document trajectory", minimum=4),
            _flag("max-iter", "int", 200, "optimizer iterations", minimum=10),
        ),
        outputs=(
            "shape_matrix.csv",
            "shape_nmf_scores.csv",
            "shape_nmf_loadings.csv",
            "shape_nmf_explained.csv",
            "shape_nmf_components.html",
        ),
        version="1",
    ),
    ToolSpec(
        name="geocode",
        family="gis",
        capability_ids=("CAP-GIS-10",),
        packet="FR-6.9",
        description="geocode the places named in a corpus to lat/lon (offline, Google, or Nominatim)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag(
                "provider",
                "str",
                "offline",
                "geocoder: offline baked KB, google, or nominatim",
                choices=("offline", "google", "nominatim"),
            ),
            _flag("limit", "int", 200, "most frequent places to geocode", minimum=1),
            _flag("min-count", "int", 1, "fewest mentions for a place to count", minimum=1),
        ),
        outputs=("locations.csv", "geocoded.csv"),
        version="1",
    ),
    ToolSpec(
        name="gis_map",
        family="gis",
        capability_ids=("CAP-GIS-11",),
        packet="FR-6.9",
        description="pin map, heatmap and KML from a geocoded CSV table",
        requires_parse=False,
        input_kind="csv",
        profiler_eligible=False,  # needs an analyst-chosen CSV file, not a corpus batch input
        params=(
            ParamSpec(name="lat-col", type="str", default="Lat", required=False, help="latitude column"),
            ParamSpec(name="lon-col", type="str", default="Lon", required=False, help="longitude column"),
            ParamSpec(name="name-col", type="str", default="Place", required=False, help="place-name column"),
            _flag("weight-col", "str", "", "column driving pin radius / heat intensity"),
            _flag("group-col", "str", "", "column grouping placemarks into a Google Earth tour"),
            _flag("title", "str", "", "map title"),
        ),
        outputs=("pin_map.html", "heatmap.html", "pins.kml", "pins_tour.kml"),
        version="1",
    ),
    ToolSpec(
        name="svo_map",
        family="gis",
        capability_ids=("CAP-GIS-12",),
        packet="FR-6.9",
        description="map where narrative events happen: geocoded SVO triples as pins, heat and KML (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        params=(
            _flag(
                "provider",
                "str",
                "offline",
                "geocoder: offline baked KB, google, or nominatim",
                choices=("offline", "google", "nominatim"),
            ),
            _flag("limit", "int", 200, "most frequent places to geocode", minimum=1),
        ),
        outputs=("svo_locations.csv", "svo_geocoded.csv", "svo_map.html", "svo_map.kml"),
        version="1",
    ),
    ToolSpec(
        name="word2vec_bert",
        family="embeddings",
        capability_ids=("CAP-EMBED-10",),
        packet="FR-5.11",
        description="Word2Vec via BERT: mean-pooled contextual type vectors, neighbours, t-SNE, and each word's "
        "company per decade in a dated corpus (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="embeddings",
        params=(
            ParamSpec(
                name="field",
                type="str",
                default="lemma",
                required=False,
                help="column for vectors",
                choices=("form", "lemma"),
            ),
            _model_param("bert-base-uncased", "token_embeddings"),
            _flag("min-count", "int", 2, "minimum word count", minimum=1),
            _flag("query", "str", None, "word to find neighbours for"),
            _flag("top-n", "int", 5, "neighbours returned", minimum=1),
            _flag("seed", "int", 42, "t-SNE seed"),
        ),
        outputs=(
            "vectors.csv",
            "neighbours.csv",
            "tsne.csv",
            "tsne.html",
            "meaning_over_time.csv",
            "meaning_change.csv",
        ),
        # 2: lemmas are read at their surface forms; neighbours reuse the
        # vectors instead of embedding the corpus a second time.
        # 3: word classes, meaning groups, and each word's company per
        # decade (meaning_over_time.csv, meaning_change.csv) when dated.
        version="3",
    ),
    ToolSpec(
        name="doc_embeddings",
        family="embeddings",
        capability_ids=("CAP-EMBED-11",),
        packet="FR-5.12",
        description="Sentence-model document or sentence embeddings: pairwise similarity by meaning, "
        "neighbours, a clustered map and semantic search (needs a parse)",
        requires_parse=True,
        parser_backend="config",
        input_kind="corpus",
        optional_package="embeddings",
        params=(
            _model_param("granite-embedding-english-r2", "sentence_embeddings", "token_embeddings"),
            _flag(
                "unit", "str", "document", "embed whole documents or single sentences", choices=("document", "sentence")
            ),
            _flag("top-n", "int", 5, "neighbours per document", minimum=1),
            _flag("query", "str", None, "semantic search: find the sentences closest in meaning"),
            _flag("seed", "int", 42, "map and clustering seed"),
        ),
        outputs=("doc_vectors.csv", "doc_pairs.csv", "doc_neighbours.csv", "doc_map.csv", "search_results.csv"),
        version="1",
    ),
)


def get_tool(name: str) -> ToolSpec | None:
    """Return the spec for *name*, or None."""
    for spec in TOOL_REGISTRY:
        if spec.name == name:
            return spec
    return None


def tool_names() -> tuple[str, ...]:
    """Registered tool names, in registry order."""
    return tuple(spec.name for spec in TOOL_REGISTRY)


_VALID_INPUT_KINDS = frozenset({"corpus", "csv", "conll", "database", "asset", "document", "directory"})


def _capability_conflicts(specs: Sequence[ToolSpec]) -> list[Diagnostic]:
    """Two specs claiming one capability ID, minus the documented shares.

    A capability ID is the join key to docs/REPLACEMENT_LEDGER.md, so an
    accidental second claimant silently reassigns a ledger row to unrelated
    code. Genuine two-entry-point capabilities live in SHARED_CAPABILITIES.
    """
    diags: list[Diagnostic] = []
    owner_of: dict[str, str] = {}
    for spec in specs:
        for capability in spec.capability_ids:
            owner = owner_of.setdefault(capability, spec.name)
            if owner != spec.name and capability not in SHARED_CAPABILITIES:
                diags.append(
                    Diagnostic.error(
                        "REGISTRY_DUPLICATE_CAPABILITY",
                        f"capability {capability} is claimed by both {owner!r} and {spec.name!r}",
                        tool=spec.name,
                        capability=capability,
                        other=owner,
                    )
                )
    return diags


def _family_conflicts(specs: Sequence[ToolSpec]) -> list[Diagnostic]:
    """A tool with no shelf (or a typo'd one) is ungroupable in every UI."""
    return [
        Diagnostic.error(
            "REGISTRY_BAD_FAMILY",
            f"{spec.name} family {spec.family!r} is not one of FAMILY_LABELS",
            tool=spec.name,
            family=spec.family,
        )
        for spec in specs
        if spec.family not in FAMILY_LABELS
    ]


def validate_specs(specs: Sequence[ToolSpec]) -> tuple[Diagnostic, ...]:
    """Duplicates, bad param types, output-less specs, empty capability
    lists, unknown input kinds, and parse claims without a backend are errors."""
    diags: list[Diagnostic] = [*_capability_conflicts(specs), *_family_conflicts(specs)]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            duplicates.add(spec.name)
        seen.add(spec.name)
        if not spec.capability_ids:
            diags.append(
                Diagnostic.error("REGISTRY_NO_CAPABILITY", f"{spec.name} declares no capability IDs", tool=spec.name)
            )
        if spec.input_kind not in _VALID_INPUT_KINDS:
            diags.append(
                Diagnostic.error(
                    "REGISTRY_BAD_INPUT_KIND",
                    f"{spec.name} input_kind {spec.input_kind!r} unknown",
                    tool=spec.name,
                    input_kind=spec.input_kind,
                )
            )
        if spec.requires_parse and not spec.parser_backend:
            diags.append(
                Diagnostic.error(
                    "REGISTRY_PARSE_NO_BACKEND",
                    f"{spec.name} requires a parse but names no backend",
                    tool=spec.name,
                )
            )
        param_names: set[str] = set()
        for param in spec.params:
            if param.name in param_names:
                diags.append(
                    Diagnostic.error(
                        "REGISTRY_DUPLICATE_PARAM", f"{spec.name}.{param.name} is repeated", tool=spec.name
                    )
                )
            param_names.add(param.name)
            if param.type not in ("str", "int", "float", "bool", "path"):
                diags.append(
                    Diagnostic.error(
                        "REGISTRY_BAD_PARAM",
                        f"{spec.name}.{param.name} has unknown type {param.type!r}",
                        tool=spec.name,
                        param=param.name,
                    )
                )
            if param.choices and param.default is not None and param.default not in param.choices:
                diags.append(
                    Diagnostic.error(
                        "REGISTRY_BAD_DEFAULT",
                        f"{spec.name}.{param.name} default is not one of its choices",
                        tool=spec.name,
                        param=param.name,
                    )
                )
            if param.minimum is not None and param.maximum is not None and param.minimum >= param.maximum:
                diags.append(
                    Diagnostic.error(
                        "REGISTRY_BAD_BOUNDS",
                        f"{spec.name}.{param.name} minimum must be below maximum",
                        tool=spec.name,
                        param=param.name,
                    )
                )
        # filenames is the one tool whose product is stdout + in-place renames,
        # not run-dir artifacts — its empty outputs are correct by design.
        if not spec.outputs and spec.name != "filenames":
            diags.append(Diagnostic.error("REGISTRY_NO_OUTPUTS", f"{spec.name} declares no outputs", tool=spec.name))
    if duplicates:
        diags.append(
            Diagnostic.error(
                "REGISTRY_DUPLICATE", f"duplicate tool specs: {sorted(duplicates)}", tools=sorted(duplicates)
            )
        )
    return tuple(diags)
