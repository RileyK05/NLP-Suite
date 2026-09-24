# Chunk Ledger

Live state of the build. One row per chunk from `docs/BUILD_PLAN.md`.
Updated as part of each chunk's definition of done.

## Legend

- `todo` — not started
- `building` — in progress
- `done` — committed, green, ledger row updated
- `blocked` — cannot proceed; reason in notes
- `dropped` — deliberately not rebuilt; reason in notes

## Progress

**52 of 52 chunks done.**

## Ledger

| # | Chunk | Phase | Status | Commit | Notes |
|---|---|---|---|---|---|
| C1 | Contracts | Foundation | done | `e79e784` | `core/result.py`, `core/config.py`. 43 tests. |
| C2 | CoNLL table | Foundation | done | `2c079d6` | schema + normalize + division. 101 tests. Verified against the legacy algorithm, not assumed. |
| C3 | Intake | Foundation | done | `4a586d1` | `core/io/reader.py`. 36 tests. Dense assigned doc_ids; a bad file is a diagnostic, not an aborted corpus. |
| C4 | Custody | Foundation | done | `27a1a0c` | `core/artifacts/envelope.py` + `core/io/writer.py`. 20 tests. Only writer writes; refuses corpus dir. |
| C5 | Parsing | Foundation | done | `0159476` | `core/pipelines/{protocol,cache,spacy_backend}.py`. 14 tests. Cache dedups; missing model hard-fails with fix command (blank fallback removed FR-0). |
| C6 | Gates | Foundation | done | `6b74d89` | `tests/test_no_import_side_effects.py` + `test_layering.py`. R1/R4 enforced. |
| C7 | First tool | Vertical slice | done | `b4ef7df` | `core/analysis/conll_wordlist.py` + `tools/conll_wordlist.py`. 15 tests. 6→1 consolidation. |
| C8 | CLI harness | Vertical slice | done | `b4ef7df` | `tools/_cli.py`. Shared corpus/pipeline/writer plumbing. |
| C9 | Viewer | Vertical slice | done | `b4ef7df` | `app/scanner.py` + `app/Home.py`. Generic envelope renderer. |
| C10 | End to end | Vertical slice | done | `11e8d1d` | `tests/fixtures/mini-corpus` + `test_end_to_end.py`. Full pipeline verified. |
| C11 | Check & clean | File intake | done | `11e8d1d` | `core/file_ops/checker.py` + `cleaner.py`. Pure check/clean. |
| C12 | Convert | File intake | done | `11e8d1d` | `core/file_ops/converter.py`. CSV/TSV via DataFrame. |
| C13 | Split | File intake | done | `11e8d1d` | `core/file_ops/splitter.py`. Parameterized engine. |
| C14 | Merge & match | File intake | done | `11e8d1d` | `core/file_ops/merger.py`. Merge + glob. |
| C15 | Classify | File intake | done | `11e8d1d` | `core/file_ops/classifier.py`. Date/NER pattern. |
| C16 | Find & sample | File intake | done | `11e8d1d` | `core/file_ops/search.py`. Search, sanitize, sample. |
| C17 | Stanza backend | Backends | done | `11e8d1d` | `core/pipelines/stanza_backend.py`. Lazy; missing model hard-fails with `stanza.download` fix (updated FR-0). |
| C18 | CoreNLP backend | Backends | done | `11e8d1d` | `core/pipelines/corenlp_backend.py`. Optional extra stub. |
| C19 | SRL backend | Backends | done | `11e8d1d` | `core/pipelines/srl_backend.py`. Optional extra stub. |
| C20 | Word analysis engine | CoNLL analysis | done | `11e8d1d` | `core/analysis/word_analysis.py` re-exports wordlist. |
| C21 | Table search | CoNLL analysis | done | `d9c169d` | `core/analysis/table_search.py` + `tools/table_search.py`. Composable predicates (eq/contains/regex, AND/OR, negate). |
| C22 | K-sentences | CoNLL analysis | done | `d9c169d` | `core/analysis/k_sentences.py` + `tools/k_sentences.py`. First/last K per doc + bookend repetition (lemma, sorted by doc frequency). |
| C23 | Clauses & SVO | CoNLL analysis | done | `d9c169d` | `core/analysis/clause_svo.py` + `tools/clause_svo.py`. Clause tag counts + SVO via dependency heads (nsubj/obj). |
| C24 | Corpus statistics | Statistics | done | `d9c169d` | `core/analysis/corpus_statistics.py` + `tools/corpus_statistics.py`. Per-doc TTR/Root/Log/Herdan/Yule-K (alpha-only tokens). |
| C25 | Text statistics | Statistics | done | `d9c169d` | `core/analysis/text_statistics.py` + `tools/text_statistics.py`. Sentences/tokens/types/avg len/syllables (vowel-group estimator). |
| C26 | CSV stats & tests | Statistics | done | `f5c5394` | `core/analysis/csv_stats.py` + `tools/csv_stats.py`. Describe + Pearson correlation, group_by. Handles CSV file or corpus dir. |
| C27 | Sentiment I | Sentiment/NER | done | `f5c5394` | `core/analysis/sentiment_vader_anew.py` + `tools/sentiment_vader_anew.py`. VADER-like (neg/pos/compound with negation) + ANEW valence/arousal/dominance. |
| C28 | Sentiment II | Sentiment/NER | done | `f5c5394` | `core/analysis/sentiment_swn_hedono.py` + `tools/sentiment_swn_hedono.py`. SentiWordNet pos/neg/obj/net + hedonometer happiness 1-9. |
| C29 | NER | Sentiment/NER | done | `f5c5394` | `core/analysis/ner.py` + `tools/ner.py`. Entity timeline (entity/tag/count/first/last sentence) + location tracking (LOC/GPE). |
| C30 | SVO compare | Sentiment/NER | done | `f5c5394` | `core/analysis/svo_compare.py` + `tools/svo_compare.py`. Pairwise Jaccard over subject/verb/object/triple sets. |
| C31 | Coreference | Sentiment/NER | done | `502fce0` | `core/analysis/coreference.py` + `tools/coreference.py`. Lemma noun clustering (baseline for neural coref). |
| C32 | Topic models | Topics/embeddings | done | `502fce0` | `core/analysis/lda.py` + `core/analysis/mallet.py`, tools `lda_gensim` / `lda_mallet` (the tool was `topic_model` at this commit; the LDA split renamed it and replaced the "TF-IDF partitioned topics (gensim/MALLET stub)" note above with real seeded Gensim LDA and a MALLET adapter). |
| C33 | Word embeddings | Topics/embeddings | done | `502fce0` | `core/analysis/word_embeddings.py` + `tools/word_embeddings.py`. Hash vectors + cosine neighbours. |
| C34 | N-grams | Topics/embeddings | done | `502fce0` | `core/analysis/ngrams.py` + `tools/ngrams.py`. N-grams 1..5 + PMI collocations. |
| C35 | Co-occurrences | Topics/embeddings | done | `502fce0` | `core/analysis/ngram_cooccurrence.py` + `tools/ngram_cooccurrence.py`. Window co-occurrence (sentence-level, unordered). |
| C36 | Contextual embeddings | Topics/embeddings | done | `656845c` | `core/analysis/contextual.py` + `tools/contextual.py`. Hash contextual vectors + WSI 2-sense split via median v0. |
| C37 | HTML annotator | Annotators/KG | done | `656845c` | `core/analysis/html_annotator.py` + `tools/html_annotator.py`. Extract, dictionary annotate (<mark>), gender pronouns. |
| C38 | Knowledge graphs | Annotators/KG | done | `656845c` | `core/analysis/knowledge_graph.py` + `tools/knowledge_graph.py`. Stub DBpedia/YAGO/Wikipedia triples (baked KB, INFO for unknown). |
| C39 | Semantic aggregation | Annotators/KG | done | `656845c` | `core/analysis/semantic.py` + `tools/semantic.py`. WordNet/VerbNet/FrameNet stub (baked maps). |
| C40 | Charts | Visualization | done | `656845c` | `core/viz/charts.py` + `tools/charts.py`. Plotly bar (lazy) with HTML table fallback. |
| C41 | Wordclouds & graphs | Visualization | done | `43430cc` | `core/viz/wordcloud_gephi.py` + `tools/wordcloud_gephi.py`. Wordcloud HTML (font-size ∝ weight) + GEXF 1.2 directed. |
| C42 | Shapes & networks | Visualization | done | `43430cc` | `core/viz/shapes.py` + `tools/shapes.py`. Story shape per sentence + Sankey (Plotly or table fallback) + GEXF wrapper. |
| C43 | Data manipulation | Data/DB | done | `43430cc` | `core/data/manipulation.py` + `tools/data_manipulation.py`. Concat/merge/pivot/melt. |
| C44 | SQL | Data/DB | done | `43430cc` | `core/data/sql.py` + `tools/sql.py`. Safe parameterized sqlite (query vs execute, single statement). |
| C45 | Corpus validation | Data/DB | done | `43430cc` | `core/data/validation.py` + `tools/corpus_validation.py`. Per-file tokens/status (OK/EMPTY/SHORT). |
| C46 | PC-ACE core | PC-ACE | done | `43430cc` | `core/pcace/core.py` + `tools/pcace.py`. Grammar + code validation + sqlite pcace table. |
| C47 | PC-ACE analysis | PC-ACE | done | `5da67d9` | `core/pcace/analysis.py` + `tools/pcace_analysis.py`. Validate codes + per-category analysis from DB. |
| C48 | Geocoding | GIS | done | `5da67d9` | `core/gis/geocode.py` + `tools/geocode.py`. Offline baked coords + haversine distance + batch INFO for unknown. |
| C49 | Mapping | GIS | done | `5da67d9` | `core/gis/mapping.py` + `tools/mapping.py`. KML placemarks + heatmap HTML (weight ∝ intensity). |
| C50 | Profiler | Profiler | done | `5da67d9` | `core/profiler/profiler.py` + `tools/profiler.py`. Orchestrates text+corpus stats + vader into wide summary. |
| C51 | Narrative & arcs | Profiler | done | `5da67d9` | `core/narrative/arcs.py` + `tools/narrative.py`. Emotion arc (per-sentence vader) + length arc. |
| C52 | Install | Packaging | done | `5da67d9` | `core/install/check.py` + `tools/install.py`. Environment check (extras installed? + Python). |

## Dropped from the legacy inventory

Not rebuilt, by decision:

| Legacy | Why |
|---|---|
| `hashfile.py` | `eval` on file contents (RCE); superseded by the envelope in C4 |
| `topic_modeling_bert_util.py` | abandoned student notebook |
| `html_annotator_extractor_util_NEW.py` | cannot be imported |
| `knowledge_graphs_DBpedia_util_SPARQL.py` | abandoned, never wired up |
| `charts_Excel_util.py` | superseded by Plotly; reinstate only if Excel output is required |
| `NLP_setup_*` | replaced by `NLPConfig` + optional-extras docs (C52) |
