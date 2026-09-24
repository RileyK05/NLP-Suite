"""Human-readable names for tools and their parameters.

A tool's registry ``name`` is an identifier (``ngram_cooccurrence``), and a
parameter's ``name`` is a command-line flag (``--max-df-ratio``). Neither is
something to put in front of a reader. Until this module existed the desktop
kept its own hand-written table of tool names and derived parameter names by
replacing underscores with spaces, which produced labels like "sg", "op",
"col x" and "no normalize" -- and silently fell back to the raw identifier for
any tool added after the table was written, so ``collocations``, ``tfidf`` and
``dispersion`` appeared in lower case beside "Readability".

Labels live here, once, on the engine side:

* the desktop gets them in the ``/api/tools`` payload and never invents one,
* ``tools/explain`` and other reporting surfaces can use the same words,
* ``tests/test_labels.py`` fails if a tool or parameter has no label, or if a
  label names something that no longer exists.

They are presentation, not contract, which is why they are not fields on
``ToolSpec``: adding a tool must not require touching the registry twice, and
a missing label is caught by a test rather than by a reader noticing that the
interface has started speaking in flag names.

``help`` still carries the sentence that explains a parameter. A label is the
two or three words that go above the input.
"""

from __future__ import annotations

__all__ = [
    "PARAM_LABELS",
    "PARAM_LABEL_OVERRIDES",
    "TOOL_DESCRIPTIONS",
    "TOOL_LABELS",
    "humanize",
    "param_label",
    "tool_description",
    "tool_label",
]

# Every tool in ``TOOL_REGISTRY`` plus the desktop's table workflows. Tools the
# desktop does not publish are labelled too: the CLI, the profiler and the
# visualization catalog all name them.
TOOL_LABELS: dict[str, str] = {
    # --- corpus analyses -------------------------------------------------
    "readability": "Readability scores",
    "lexical_diversity": "Lexical diversity (TTR, MTLD)",
    "doc_similarity": "Document similarity",
    "doc_duplicates": "Duplicate detection",
    "spellcheck": "Spelling suggestions",
    "search": "Text search",
    "sentence_complexity": "Sentence complexity",
    "lda_gensim": "Topic discovery (Gensim LDA)",
    "lda_stability": "Topic stability across seeds",
    "lda_mallet": "Topic discovery (MALLET LDA)",
    "gender_annotator": "Gender of names mentioned",
    "date_annotator": "Dates mentioned in the text",
    "quote_annotator": "Quotes and their speakers",
    "gender_guess": "Author gender from writing style",
    "verb_analysis": "Verb tense, modality and voice",
    "ngram_viewer": "N-grams over time (culturomics)",
    "sentiment_neural_bert": "Neural sentiment (BERT)",
    "sentiment_neural_spacy": "Neural sentiment (spaCy)",
    "sentiment_neural_stanza": "Neural sentiment (Stanza)",
    "sentiment_neural_corenlp": "Neural sentiment (CoreNLP)",
    "shape_hc": "Story shapes: clustering tree",
    "shape_svd": "Story shapes: SVD components",
    "shape_nmf": "Story shapes: NMF parts",
    "geocode": "Geocode places to coordinates",
    "gis_map": "Pin maps and heatmaps",
    "svo_map": "Map of who does what where",
    "word2vec_bert": "Word relationships (BERT)",
    "word2vec_gensim": "Word relationships (Gensim Word2Vec)",
    "sentiment_vader_anew": "Sentiment (VADER / ANEW)",
    "sentiment_swn_hedono": "Sentiment (SentiWordNet / happiness)",
    "nrc": "Emotion vocabulary (NRC)",
    "nominalization": "Nouns derived from verbs",
    "style": "Concreteness and iconicity",
    "narrative": "Character sentiment over time",
    "clause_svo": "Subjects, verbs and objects",
    "ner": "Named entities",
    "ngrams": "Repeated word sequences",
    "ngram_cooccurrence": "Words occurring together",
    "collocations": "Collocation strength (PMI, t-score, G²)",
    "conll_wordlist": "Word frequencies by part of speech",
    "corpus_statistics": "Vocabulary richness per document",
    "k_sentences": "Opening and closing sentences",
    "svo_compare": "Compare subject–verb–object patterns",
    "text_statistics": "Document statistics",
    "table_search": "Search annotated words",
    "kwic": "Keyword in context",
    "keyness": "Distinctive words between groups (keyness)",
    "tfidf": "Distinctive terms (TF–IDF)",
    "dispersion": "Where words fall across the corpus",
    "lexicon_series": "Track your own word groups over time",
    "coreference": "Mention grouping (lemma baseline)",
    "semantic": "Semantic tags",
    "wordnet": "WordNet senses and hierarchy",
    "verbnet": "VerbNet verb classes",
    "framenet": "FrameNet frames",
    "symbolic": "Symbolic space and social actors",
    "knowledge_graph": "Knowledge graph",
    "word_sense_induction": "Word senses from context (BERT)",
    "bert_extract": "Extractive summary (transformer)",
    "bert_topics": "Topic clusters (transformer)",
    # --- visualization ---------------------------------------------------
    "shapes": "Sentence structure over time",
    "charts": "Charts from a table",
    "panels": "Purpose-built figures for one analysis",
    "wordcloud_gephi": "Wordclouds and networks",
    # --- statistics over an analyst's CSV --------------------------------
    "stats_categorical": "Categorical association (chi-square)",
    "stats_groups": "Group comparisons (Mann-Whitney, Kruskal-Wallis)",
    "stats_trends": "Trends over time (Mann-Kendall, rank correlation)",
    "csv_stats": "Table summary and correlations",
    # --- intake and batch ------------------------------------------------
    "convert": "Convert documents to text",
    "filenames": "Standardize filenames",
    "profiler": "Batch profiler",
}

# The desktop's CSV workflows are separate specs (desktop_backend/tables.py).
TOOL_LABELS.update(
    {
        "table_chi2": "Column association (chi-square)",
        "table_crosstab": "Cross-tabulation with percentages",
        "table_keyness": "Distinctive words between columns (keyness)",
        "table_mw": "Two-group comparison (Mann–Whitney U)",
        "table_kw": "Three-plus-group comparison (Kruskal–Wallis)",
        "table_trend": "Trend over time (Mann–Kendall)",
        "table_rankcorr": "Rank correlation (Spearman, Kendall)",
        "table_charts": "Charts from a table",
        "table_wordcloud_gephi": "Wordclouds and networks",
    }
)

# Parameter labels are keyed by flag name. A flag means the same thing
# wherever it appears -- ``--field`` is always the token column, ``--top-n`` is
# always how many rows survive -- so one table keeps the interface consistent
# instead of letting each form invent its own wording.
PARAM_LABELS: dict[str, str] = {
    "agg": "Combine repeated rows by",
    "alpha": "Significance level (α)",
    "analysis": "Analysis",
    "anew-field": "ANEW rating",
    "anew-lexicon": "ANEW ratings file",
    "bins": "Number of bins",
    "case-sensitive": "Match capitalisation exactly",
    "category": "Part of speech",
    "check-ending": "Require a nominal ending",
    "chunks": "Number of chunks",
    "lambda": "Relevance lambda",
    "col-x": "First column",
    "col-y": "Second column",
    "col1": "First column",
    "col2": "Second column",
    "concreteness-lexicon": "Concreteness norms file",
    "correct": "Write corrected copies",
    "curated-list": "Curated word list",
    "date-col": "Date column",
    "epochs": "Training passes",
    "field": "Word form",
    "format": "File format",
    "freq1": "First frequency column",
    "freq2": "Second frequency column",
    "group": "Group by column",
    "group-col": "Group column",
    "group-pattern": "Pattern that defines group A",
    "iconicity-lexicon": "Iconicity norms file",
    "image": "Also save a picture",
    "input": "CSV table",
    "k-first": "Opening sentences to keep",
    "k-last": "Closing sentences to keep",
    "keep-stopwords": "Keep common words",
    "kind": "Chart type",
    "logic": "Combine conditions with",
    "max-df-ratio": "Ignore words above this share of documents",
    "max-hits": "Stop after this many matches",
    "max-rating-sd": "Largest rating disagreement",
    "max-words": "Most words to show",
    "method": "Method",
    "min-year": "Earliest year counted",
    "max-year": "Latest year counted",
    "model": "Model",
    "max-iter": "Optimizer iterations",
    "n-clusters": "Clusters to cut into",
    "n-components": "Components to keep",
    "names-dir": "Name-list folder",
    "name-col": "Place-name column",
    "lat-col": "Latitude column",
    "lon-col": "Longitude column",
    "dictionary": "Name dictionary",
    "queries": "Words or phrases to track",
    "provider": "Geocoder",
    "resample": "Points per trajectory",
    "smooth": "Smoothing window (years)",
    "server": "Server URL",
    "language": "Language package",
    "limit": "Most places to geocode",
    "source": "Where names come from",
    "min-count": "Fewest times a word must appear",
    "min-df": "Fewest documents a term must appear in",
    "min-length": "Shortest word to keep",
    "min-rating": "Lowest rating to count",
    "mode": "What to search",
    "n": "Sequence length",
    "negate": "Keep the rows that do not match",
    "no-normalize": "Skip length normalisation",
    "normalize": "Show values as",
    "nouns-only": "Nouns only",
    "op": "How to match",
    "remove-stopwords": "Remove common words",
    "parts": "Split the corpus by",
    "passes": "Training passes per fit",
    "plot": "Also draw a plot",
    "posthoc-method": "Multiple-comparison correction",
    "query": "Search for",
    "regex": "Treat as a regular expression",
    "sankey-source": "Sankey source column",
    "sankey-target": "Sankey target column",
    "sankey-value": "Sankey value column",
    "seed": "Random seed",
    "seeds": "Random seeds to compare",
    "sg": "Training algorithm",
    "shape": "Wordcloud outline",
    "smoothing": "Log Ratio smoothing",
    "source-col": "Edge source column",
    "span": "Pairs to count",
    "stable-at": "Overlap needed to count as stable",
    "stopwords": "Common-word list",
    "sublinear-tf": "Dampen repeated words",
    "target-col": "Edge target column",
    "threshold": "Similarity threshold",
    "title": "Title",
    "top-n": "How many to keep",
    "terms": "Word groups",
    "by": "Count along",
    "within": "Only inside sentences about",
    "topics": "Number of topics",
    "vader-lexicon": "VADER lexicon file",
    "value": "Value to match",
    "value-col": "Value column",
    "vector-size": "Vector size",
    "weight-col": "Weight column",
    "window": "Context window",
    "word-col": "Word column",
    "wordlist": "Word list file",
    "x": "Horizontal axis",
    "y": "Vertical axis",
}

# Where a shared flag genuinely means something different in one tool.
PARAM_LABEL_OVERRIDES: dict[tuple[str, str], str] = {
    ("ngrams", "min-count"): "Fewest times a pair must appear",
    ("ngram_cooccurrence", "min-count"): "Fewest times a pair must appear",
    ("ngram_cooccurrence", "window"): "Sentences to look across",
    ("collocations", "window"): "Words to look across",
    ("collocations", "min-count"): "Fewest times a pair must appear",
    ("kwic", "window"): "Words of context on each side",
    ("word2vec_gensim", "field"): "Train on surface forms or lemmas",
    ("word2vec_gensim", "remove-stopwords"): "Remove common words before training",
    ("word2vec_gensim", "query"): "Word to find neighbours for",
    ("word2vec_gensim", "top-n"): "Neighbours to return",
    ("lda_gensim", "field"): "Model surface forms or lemmas",
    ("lda_gensim", "top-n"): "Words shown per topic",
    ("doc_similarity", "threshold"): "Duplicate threshold",
    ("doc_duplicates", "threshold"): "Fuzzy-match threshold",
    ("spellcheck", "threshold"): "Suggestion threshold",
    ("table_wordcloud_gephi", "mode"): "What to build",
    ("table_wordcloud_gephi", "title"): "Wordcloud title",
    ("table_charts", "title"): "Chart title",
    ("style", "analysis"): "Which norms to use",
    ("lexicon_series", "field"): "Word form counted",
    ("lexicon_series", "group-pattern"): "Pattern that names the axis",
    ("narrative", "field"): "Word form scored",
    ("nominalization", "field"): "Word form looked up",
    ("quote_annotator", "min-length"): "Fewest words in a quote",
    ("knowledge_graph", "source"): "Offline stub or live DBpedia",
    ("gender_guess", "mode"): "Which scoring axis",
    ("verb_analysis", "analysis"): "Facet to summarize",
    ("word2vec_bert", "query"): "Word to find neighbours for",
    ("word2vec_bert", "top-n"): "Neighbours to return",
    ("sentiment_neural_spacy", "model"): "Neural textcat pipeline (spaCy)",
    ("sentiment_neural_bert", "model"): "Hugging Face sentiment model",
}


# A ``ToolSpec.description`` is written for whoever is wiring the tool up:
# "TTR, Guiraud, MTLD, vocd-D per document (raw text, seeded)" says what runs.
# On a card in front of a student it says nothing about what they would learn.
# These are the same tools described by what the result answers. Where a tool
# has none, its registry description is shown unchanged.
TOOL_DESCRIPTIONS: dict[str, str] = {
    "readability": "How hard each document is to read, on the Flesch, Fog and SMOG scales.",
    "lexical_diversity": "How varied the vocabulary is in each document, by four measures that disagree in useful ways.",
    "doc_similarity": "Which documents share vocabulary, scored pair by pair, with near-duplicates called out.",
    "doc_duplicates": "Exact copies, same-text-different-formatting matches, and documents that are close but not identical.",
    "sentence_complexity": "How involved each sentence is, measured from its grammatical structure rather than its length.",
    "lda_gensim": "Recurring themes across the corpus, and which theme each document leans towards.",
    "lda_stability": "Which of this corpus's topics come back when the model is refit from a different random start.",
    "lda_mallet": "The same question, answered by MALLET's LDA so the two can be compared.",
    "word2vec_gensim": "Words used in similar contexts, with a neighbour list and a map of the vocabulary.",
    "word2vec_bert": "The same neighbour question, answered with BERT's reading of your words so the two Word2Vec approaches can be compared.",
    "gender_annotator": (
        "Which person names in your documents a name dictionary reads as male or female, "
        "and how much of the corpus that dictionary simply does not know."
    ),
    "date_annotator": (
        "Every date your documents mention, normalized so March 5, 2020, 5 March 2020 and 03/05/2020 "
        "compare as one date."
    ),
    "quote_annotator": "The quoted speech in your documents, with a transparent best guess at who is speaking each line.",
    "gender_guess": "Which writing-style gender each document leans towards, from the function words it uses.",
    "verb_analysis": ("How each verb is used: past, present or future, active or passive, and which modal governs it."),
    "ngram_viewer": "How often your words and phrases are used year by year, charted across the corpus.",
    "sentiment_neural_bert": "Sentence-by-sentence sentiment from a trained BERT classifier, so tone can be compared across documents.",
    "sentiment_neural_spacy": "Sentence-by-sentence sentiment from a spaCy textcat head, one of four annotators you can compare.",
    "sentiment_neural_stanza": "Sentence-by-sentence sentiment from Stanza's own sentiment model, one of four annotators you can compare.",
    "sentiment_neural_corenlp": (
        "Sentence-by-sentence sentiment from Stanford CoreNLP's sentiment annotator, one of four annotators you can compare."
    ),
    "shape_hc": "Which documents tell similarly shaped stories, grouped into a tree of the distances between them.",
    "shape_svd": "The main axes along which your stories' shapes vary, each with its share of the variance.",
    "shape_nmf": "The additive building blocks your stories' shapes are made of, kept non-negative so they read as parts.",
    "geocode": "The places named in your corpus turned into map coordinates you can chart.",
    "gis_map": "A pin map, a heatmap and a Google Earth file from any table of places and coordinates.",
    "svo_map": "Where the events of your corpus happen: every subject–verb–object triple placed on a map by its location.",
    "nominalization": "Nouns built out of verbs — 'decision' from 'decide' — which often mark abstract or bureaucratic writing.",
    "style": "How concrete, or how iconic, the words in each sentence are, scored against published rating norms.",
    "narrative": "How sentiment and sentence length rise and fall from the start of a document to its end.",
    "clause_svo": "Who does what to whom: subject–verb–object triples and the clause types they sit in.",
    "ner": "People, places and organizations as they appear through each document, with a location track.",
    "ngrams": "Word sequences that repeat, from single words up to five in a row.",
    "ngram_cooccurrence": "Pairs of words that turn up in the same sentence, counted.",
    "collocations": "Word pairs that occur together more often than chance would explain, by four association measures.",
    "conll_wordlist": "How often each word appears, narrowed to one part of speech such as nouns or verbs.",
    "corpus_statistics": "Vocabulary richness per document, for comparing documents of different lengths.",
    "k_sentences": "The opening and closing sentences of every document, side by side.",
    "svo_compare": "How far two documents agree on who does what to whom.",
    "text_statistics": "Sentence, word and syllable counts for every document.",
    "table_search": "Find annotated words by their form, dictionary form or part of speech.",
    "shapes": "How sentence length and word classes shift across a document, optionally as a Sankey diagram.",
    "kwic": "Every occurrence of a word or phrase, with the words on either side of it.",
    "keyness": "Which words are distinctive to one group of documents compared with another.",
    "tfidf": "The terms that set each document apart from the rest of the corpus.",
    "dispersion": "Whether a word is spread evenly through the corpus or clustered in a few places.",
    "lexicon_series": (
        "How often your own groups of words are used, counted along an axis such as year or speaker "
        "and reported per 1,000 words so long documents do not look louder than short ones."
    ),
    "table_chi2": "Whether two columns of a table are associated, with residuals and expected counts.",
    "table_crosstab": "Counts and row percentages for two columns of a table.",
    "table_keyness": "Which words are distinctive between two frequency columns of a table.",
    "table_mw": "Whether two groups differ, by a test that does not assume a normal distribution.",
    "table_kw": "Whether three or more groups differ, with a follow-up test for which pairs differ.",
    "table_trend": "Whether a measurement rises or falls over dated observations, and how steeply.",
    "table_rankcorr": "Whether two columns move together, by rank rather than by raw value.",
    "table_charts": "A publication-ready chart from any CSV, as a web page, a static image, or a native Excel workbook.",
    "table_wordcloud_gephi": "A wordcloud from a frequency table, or a network file that opens in Gephi.",
    # --- tools kept in the developer interfaces (no desktop card yet): they
    # still name themselves in the CLI, the profiler and the docs. ------------
    "bert_extract": "The sentences that best summarise a document, ranked, chosen by meaning rather than position.",
    "bert_topics": "Groups of documents that read alike, and the words that define each group.",
    "coreference": "Which noun mentions in a document refer to the same person or thing, so characters can be followed.",
    "csv_stats": "A quick description of every numeric column in a table, and how each pair of columns moves together.",
    "charts": "A publication-ready chart of your choosing from any CSV, with the aggregation rules stated on the figure.",
    "convert": "Turn PDF, DOCX, HTML and other files into the plain text the suite reads, with a report of what worked.",
    "filenames": "Renames document files to a shared pattern (date, speaker, kind) after showing you the before and after.",
    "framenet": "Which FrameNet situations your words evoke, for a list of words you supply.",
    "knowledge_graph": "Who-is-who and who-does-what triples for the entities in your corpus, drawn as a graph when there are edges.",
    "nrc": "Which of eight basic emotions each document's vocabulary leans towards, as a share of that document's words.",
    "panels": "The purpose-built figures behind each analysis, drawn with their evidence links and provenance captions.",
    "profiler": "Runs a whole battery of analyses over the parsed corpus at once and links every result to its run.",
    "semantic": "Each noun, verb and adjective in your documents tagged with its semantic class, such as 'money' or 'motion'.",
    "sentiment_swn_hedono": "Sentence tone from SentiWordNet's ratings, or happiness from the Hedonometer word scores.",
    "stats_categorical": "Cross-tabulations, chi-square association and keyness between columns of a table you supply.",
    "stats_groups": "Whether two or more groups of numbers differ, with follow-up tests for which pairs differ.",
    "stats_trends": "Whether a series rises or falls over time (Mann-Kendall), and whether two columns move together.",
    "symbolic": "How your documents distribute across symbolic spaces (nature, machine, religion) and social actor types.",
    "verbnet": "Which VerbNet classes your verbs belong to, for a list of verbs you supply.",
    "word_sense_induction": "Which different senses of one word appear in your corpus, grouped by how they are used in context.",
    "wordnet": "Expand a word list up to its WordNet categories, or down through all the words beneath one category.",
    "wordcloud_gephi": "A wordcloud image from a frequency table, or a network file for Gephi from an edge table.",
}


def tool_description(name: str, fallback: str = "") -> str:
    """The sentence to show a reader, falling back to the registry's own."""
    return TOOL_DESCRIPTIONS.get(name) or fallback


def humanize(name: str) -> str:
    """Last-resort rendering of an identifier, used only if a label is missing.

    Kept deliberately plain. It exists so a missing label degrades to something
    readable rather than to an empty control, not so that labels can be
    skipped -- ``tests/test_labels.py`` requires a real one.
    """
    words = name.replace("-", " ").replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else name


def tool_label(name: str) -> str:
    """The name to show a reader for ``name``."""
    return TOOL_LABELS.get(name) or humanize(name)


def param_label(tool: str, param: str) -> str:
    """The label for ``param`` as it appears on ``tool``'s form."""
    return PARAM_LABEL_OVERRIDES.get((tool, param)) or PARAM_LABELS.get(param) or humanize(param)
