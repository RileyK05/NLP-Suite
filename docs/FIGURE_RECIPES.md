# Figure recipes: a figure that fits every tool

**Written for:** whoever builds or reviews the visual layer (a person or a
model), and Riley, who set the goal: every tool's visuals should make sense
for its data, not just the few tools that have purpose-built panels.

## The problem

There are 71 registered tools. Before this work, about 15 had purpose-built
panels. Every other tool fell through to `core/insight/recommend.py`'s generic
rules, which look only at column *types*. Those rules draw charts that are
technically correct and tell you nothing. For example, on `ngram_cooccurrence`:

- **"Count over Date"** averaged 9,407 unrelated pair-by-speech rows into one
  line.
- **"Largest Count by Word 1"** ranked "of" and "be". Word 1 is only the
  alphabetically first word of a pair, so it has no meaning on its own.
- **"Histogram: Count vs Count"** had a garbled label.
- **A line chart across unordered categories** could be drawn at all.

This is a general problem, not one tool's bug.

## The rule

1. **Every registered tool declares a recipe** in `core/viz/recipes.py`. A
   recipe is either one or more named panels, or *table-first* with the
   reason (for example, "a concordance is read, not charted"). A test fails
   if any tool is missing one.
2. **A tool with a recipe gets no generic chart suggestions.** The reading
   ("What this says") offers that tool's own figures, each with the question
   it answers. The generic workbench stays available as an expert tool, with
   guards.
3. **Every figure carries a reading guide.** Its notes say how to read it and
   what it cannot tell you. It refuses, with the reason, when the data cannot
   support it.
4. **Every mark leads to its evidence**: the rows behind it, then the
   passages in the text.

## House rules for every figure (added to Riley's list)

- **Normalise pooled counts.** A count pooled across documents of different
  lengths is shown per 10,000 tokens, with the raw count in the hover. A raw
  count is mostly a measure of how long the speeches were.
- **Show document coverage beside frequency.** Every ranked-term view shows
  "in N of M documents", so a word heavy in one speech cannot pass for a
  corpus theme.
- **Offer a function-word toggle** on term rankings. "of" and "be" top every
  raw frequency list.
- **Check length before trusting per-document measures.** A per-document
  measure that depends on length (TTR, dependency distance, raw counts) gets
  a companion scatter of the measure against document length.
- **Trends show the documents.** A trend draws one point per document plus a
  rolling median. It never draws a line through single documents as though
  each one were a year's average.
- **No averages without spread.** Group comparisons show a box plus every
  point.
- **Keep colours stable.** A topic, entity type or emotion has one colour in
  every figure of a run.
- **Offer grouping by** year, decade, or speaker (parsed from the file name),
  for per-document views of a dated corpus.

## Shapes

Each shape is implemented twice, Python (`core/viz/panel_plotters.py`) and
TypeScript (`desktop/src/panelLayout.ts` + `PanelCanvas.tsx`), and checked by
`tests/test_panel_parity.py`. All ten are drawn.

| Shape | Used for |
|---|---|
| `scatter_labelled` | volcanoes, association against evidence, length checks, t-SNE, component scores |
| `ranked_bars` | term, triple, speaker and topic rankings; neighbours; explained variance |
| `stream` | topic prevalence; gender mentions by category |
| `line_series` | per-document trends (points + rolling median); ngram viewer; Sen's slope trends; arcs |
| `ribbon` | topic flow; compositions (NMF parts, crosstab rows, MALLET mixtures, voice and modality by speech) |
| `heatmap` | document similarity, SVO overlap, tf-idf, NRC emotions, lexicon categories, chi-square residuals, post-hoc p, topic matches |
| `distribution` | groups by decade or speaker (box + every point); cluster and topic eras |
| `network` | co-occurrence, collocation, similar-document, SVO, duplicate and knowledge graphs; dendrogram (elbow links) |
| `small_multiples` | several measures, each on its own scale; component loadings along the speech |
| `positions` | concordance hits, coreference chains, character mentions |
| `map` | **not built**: needs a bundled offline basemap (geocode, gis_map, svo_map, NER places) |

## Per-tool status (from the registry; `tests/test_figure_recipes.py` enforces coverage)

Every one of the 80 tool names (71 registered + 9 desktop table workflows)
has a recipe: 149 panels, plus table-first reasons where a chart would
mislead. "Verified" = drawn from the tool's real output on the 87-speech
corpus. "Unverified" = the tool cannot run on this machine (missing lexicon or
model), so the panel is built and tested against the engine's exact schema only.

| Tool | Figures | Real data |
|---|---|---|
| readability, lexical_diversity, corpus_statistics, sentence_complexity | trend (points + rolling median), groups by decade/speaker, length check, all measures | verified |
| text_statistics | trend, groups, all measures | verified |
| verb_analysis | trend, groups, all measures; per-speech voice and modality mixes; passive-versus-obligation scatter | verified |
| nominalization | trend of the per-document rate, groups; ranked nominalizations | verified |
| style | concreteness and iconicity by group | unverified (lexicon missing) |
| ngrams | ranked phrases (phrase-boundary function-word rule, coverage) | verified |
| conll_wordlist | ranked words; refuses with advice when the run's top-n is all function words | verified |
| tfidf | one document's top terms; document x term heatmap; warns when IDF is not separating | verified |
| nrc | document x emotion heatmap | verified |
| semantic | ranked lemmas with their tags | verified (demo vocabulary) |
| verbnet, framenet, symbolic, wordnet | ranked classes/frames/categories | unverified (corpora missing) |
| ngram_cooccurrence | PPMI against evidence; one word's partners; network | verified |
| collocations | strength scatter; network | verified |
| doc_similarity, svo_compare | document heatmap; nearest-neighbour network | verified |
| doc_duplicates | table-first; fuzzy-match graph | verified (refuses: no duplicates) |
| knowledge_graph | table-first; network when there are edges | verified (demo data) |
| lexicon_series | per-1,000 lines over time; year x category heatmap | verified |
| date_annotator | mentioned year against speech date; counts by type | verified |
| gender_annotator | mentions by category over time | unverified (nltk names missing) |
| narrative | emotion arc through the speech; character positions | verified |
| shapes | story arc through the speech | verified |
| dispersion | frequency against Gries DP | verified |
| ner | top entities by tag; entity timeline (places by default) | verified |
| kwic, coreference | table-first; positions of hits / chains | verified |
| lda_gensim | relevance, intertopic map, prevalence, topic flow | verified |
| lda_stability | stability bars; topic x seed match heatmap | verified |
| lda_mallet | document topic mix; ordered topic-term list | unverified (MALLET not installed) |
| bert_topics | topic words; topics through time | verified |
| word2vec_gensim / word2vec_bert | precomputed neighbours; t-SNE; query any saved word vector without retraining | verified (Gensim) |
| shape_hc | dendrogram; clusters through time | verified |
| shape_svd, shape_nmf | explained; loadings along the speech; documents on components; NMF mix | verified |
| sentiment_vader_anew | extremes; trend, groups, all measures (VADER; ANEW when installed) | verified (VADER) |
| sentiment_neural_* | extremes; trend, groups, all measures | unverified (models missing) |
| sentiment_swn_hedono | trend, groups, all measures | unverified |
| gender_guess | style scores by trend and group, with the "writing style, not a person" warning | verified |
| clause_svo | agency; top triples | verified |
| quote_annotator | table-first; who is quoted; quotes per speech | verified |
| table_chi2 / stats_categorical | residual heatmap (+ crosstab and keyness for stats_categorical) | verified (engine output) |
| table_crosstab | counts heatmap; row composition | verified (engine output) |
| table_keyness | effect against evidence | verified (engine output) |
| table_mw, table_kw / stats_groups | medians; post-hoc heatmap (KW) | verified (engine output) |
| table_trend / stats_trends | observations with Sen's slope | verified (engine output) |
| table_rankcorr, wordcloud_gephi (+ table_) | table-first (the output is one coefficient / an image and a GEXF) | n/a |
| csv_stats | every pairwise correlation as a diverging heatmap (`correlation_matrix.csv`) | verified |
| word_sense_induction | table-first (the split is a baseline, not a clustering) | n/a |
| convert, filenames, profiler, k_sentences, bert_extract, search, table_search, spellcheck, geocode, gis_map, svo_map | table-first | n/a |
| charts, panels, table_charts | the generic builders themselves | n/a |

## Still owed from Riley's list

- Maps (geocode, gis_map, svo_map, NER places): the `map` shape with a
  bundled offline basemap.
- WordNet hierarchy as a tree; the story-shape Sankey; MALLET topic-word bars;
  style concreteness against iconicity scatter; search hits by year (needs per-document token counts in
  the search table); spellcheck unknown-word bars.
- A multi-run seam, so model agreement (four neural sentiment models,
  MALLET against Gensim) is one figure.
- Figure bundles: chart + data + methods note + provenance JSON.
