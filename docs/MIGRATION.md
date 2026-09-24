# Migration guide: NLP Suite 1.6.x → NLP Suite NG (FR-9.5)

The legacy suite is ~50 Tkinter GUIs (`src/*_main.py`); this suite is one
unified command plus a Streamlit app. The mapping below is exhaustive for
shipped analyses: every row names the legacy GUI, the NG command, and the
honest status. `tests/test_migration.py` enforces that every command named
here resolves and every registry tool is named here — this document cannot
silently drift from the code.

Conventions: `nlp-suite <tool>` is `tools/unified.py` routing to
`python -m tools.<tool>`; every tool also runs standalone as
`python -m tools.<tool> [args]`. All commands take a corpus directory of
`.txt` files and an output root unless noted. Long runs belong in
background jobs (`nlp-suite jobs submit <tool> ...`) or the CLI — the app
pages run synchronously.

## Text statistics (statistics_txt_main, sentence_analysis_main)

- `nlp-suite readability CORPUS OUT` — Flesch/FK/ARI/Fog/CL/SMOG. Available.
- `nlp-suite lexical_diversity CORPUS OUT` — Guiraud/MTLD/vocd-D. Available.
- `nlp-suite sentence_complexity CORPUS OUT` — dependency distance/depth, Yngve/Frazier. Available (CoreNLP-only inputs documented at runtime).
- `nlp-suite stats_categorical CORPUS OUT`, `nlp-suite stats_groups CORPUS OUT`, `nlp-suite stats_trends CORPUS OUT` — chi2/MWU/KW/Mann-Kendall families. Available.
- `nlp-suite text_statistics CORPUS OUT` — per-document counts (excluded from the registry: parser-workflow metadata pending; runs as `python -m tools.text_statistics`).
- `nlp-suite corpus_statistics CORPUS OUT` — corpus-level TTR family (same exclusion note as `text_statistics`).
- `nlp-suite csv_stats INPUT.csv OUT --corr-x A --corr-y B --corr-method pearson` — CSV describe + pairwise correlation + the full correlation matrix (`correlation_matrix.csv`, feeding the desktop heatmap). Available (CSV input, not a corpus tool).

## Semantic and sentiment (semantic_analysis_main, sentiment_analysis_main, sentiments_emotions_ALL_main, semantic_aggregation_main)

- `nlp-suite nominalization CORPUS OUT` — deverbal nominalizations (WordNet extra). Available.
- `nlp-suite style CORPUS OUT --analysis concreteness|iconicity` — Brysbaert/Winter norms (user-supplied assets). Available.
- `nlp-suite sentiment_vader_anew CORPUS OUT` — VADER + ANEW. Available.
- `nlp-suite sentiment_swn_hedono CORPUS OUT` — SentiWordNet + hedonometer. Available.
- `nlp-suite nrc CORPUS OUT` — NRC emotions. Available.
- `nlp-suite wordnet WORDS.csv OUT up|down` — WordNet aggregation (word-list input). Available.
- `nlp-suite verbnet WORDS.csv OUT`, `nlp-suite framenet WORDS.csv OUT` — VerbNet/FrameNet aggregation (word-list input). Available.
- `nlp-suite symbolic WORDS.csv OUT --analysis space|actor` — symbolic typologies (word-list input). Available.
- `nlp-suite semantic CORPUS OUT` — legacy-compatible semantic workflow (excluded from the registry: asset/model-dependent; runs as `python -m tools.semantic`).
- `nlp-suite coreference CORPUS OUT` — deterministic lemma baseline. Available; the neural backend (Stanford CoreNLP) is a documented follow-up, not a silent substitution.
- `nlp-suite word_sense_induction CORPUS OUT --model bert-base-uncased` (was `contextual`) — transformer embeddings + WSI baseline (model downloads on first use). Available.
- `nlp-suite bert_extract CORPUS OUT --sentences 3` — centroid extractive summaries. Available.
- `nlp-suite lda_gensim CORPUS OUT [--topics 3 --top-n 5 --lambda 0.6]` — seeded Gensim LDA with the Gensim GUI's graded views (Intertopic Distance Map + lambda-ranked terms). Available. (`topic_model` was the pre-split name; the guide entry is the tool's registry name.)
- `nlp-suite lda_stability CORPUS OUT [--topics 3 --seeds 100,101,102 --stable-at 0.5]` — refits the model at each seed and matches topics back to the reference by top-word overlap (Jaccard), reporting mean and worst-seed overlap per topic. New-from-spec (the legacy suite had no stability check). Available; needs the `[topics]` extra.
- `nlp-suite lda_mallet CORPUS OUT [--topics 10 --seed 42]` — MALLET LDA topic keys + document-topic composition. Available; needs the MALLET Java binary and fails loudly without it rather than substituting another backend.
- `nlp-suite bert_topics CORPUS OUT --topics 4 --seed 42` — KMeans over transformer document embeddings with discriminative term labels (legacy `topic_modeling_BERT_main`). Seeded and deterministic. Needs the `[embeddings]` extra; the model downloads on first use. Available.
- `nlp-suite keyness CORPUS OUT --group-pattern REGEX` — G2 log-likelihood keyness between two document groups defined by a regex over document names, with Hardie-smoothed Log Ratio (legacy `statistics_corpus_keyness`). Available.
- `nlp-suite word2vec_gensim CORPUS OUT` — seeded Gensim Word2Vec (was `word_embeddings`). Available. Writes vectors.csv, neighbours.csv (with `--query`), tsne.csv and an interactive tsne.html scatter (legacy `word2vec_tsne_plot_util` parity; a plain coordinate table without plotly).
- `nlp-suite word2vec_bert CORPUS OUT --model bert-base-uncased` — Word2Vec via BERT: mean-pooled contextual type vectors with the same vectors/neighbours/tsne tables as `word2vec_gensim`, so the two Word2Vec approaches compare like with like (the model reads your corpus; it is not trained on it). Needs the `[embeddings]` extra; the model downloads on first use. Available.
- `nlp-suite doc_similarity CORPUS OUT`, `nlp-suite doc_duplicates CORPUS OUT` — TF-IDF pairs, exact/normalized/fuzzy tiers. Available.
- `nlp-suite search CORPUS OUT` — structured text/CSV/CoNLL search. Available.

## Files and intake (file_*_main, file_checker_converter_cleaner_main, file_checker_pre_processing_pipeline_main)

- `nlp-suite convert CORPUS OUT` — txt/csv/tsv/html conversion (pdf/docx/rtf need their extras, named at runtime). Available.
- `nlp-suite filenames CORPUS OUT` — preview-first standardization. Available.
- `nlp-suite spellcheck CORPUS OUT` — explicit wordlists + suggestions. Available.
- `nlp-suite corpus_validation CORPUS OUT` — Validate first; the app's Corpus page runs the same check. Available (intake command, no registry spec).
- `nlp-suite data_manipulation INPUT.csv OUT` — table transforms (excluded from the registry: conditional schemas; runs as `python -m tools.data_manipulation`).
- `nlp-suite table_search` — alias surface for `nlp-suite search` over CSV/CoNLL tables (excluded from the registry: metadata pending; runs as `python -m tools.table_search`).

## Parsers and annotation (parsers_annotators_main, NER_main, SVO_main, CoNLL_table_analyzer_main, NGrams_CoOccurrences_main, html_annotator_main, html_annotator_gender_main)

- Parsers are a `--parser spacy|stanza --language en` flag on every corpus tool, not a separate step; the shared parse feeds the whole batch. Use `nlp-suite doctor en` to verify models.
- `nlp-suite models OUT` — read-only parser-model inventory with fixes. Available.
- `nlp-suite ner CORPUS OUT` — entity timeline + location tracking. Available.
- `nlp-suite gender_annotator CORPUS OUT --dictionary nltk|census|carnegie_mellon|social_security [--names-dir DIR] [--source ner|propn]` — name-gender annotation from a choice of four dictionaries (the legacy `html_annotator_gender_main` / `lib\namesGender` workflow). Available; a missing dictionary fails loudly with the drop-in layout rather than substituting another.
- `nlp-suite date_annotator CORPUS OUT [--min-year 1000 --max-year 2100]` — normalized date extraction over raw text (CoreNLP-style normalized dates without the server). Available.
- `nlp-suite quote_annotator CORPUS OUT [--min-length 1]` — quote/dialogue extraction with a transparent two-stage speaker sieve (legacy `html_annotator_main` quote workflow). Available.
- `nlp-suite gender_guess CORPUS OUT [--mode both]` — writing-style gender guesses from weak-word rates (the GenderGuesser method, own documented lexicon). Available.
- `nlp-suite verb_analysis CORPUS OUT [--analysis all|modality|tense|voice]` — verb modality, tense and voice per verb token (legacy `CoNLL_verb_analysis` families). Available.
- `nlp-suite ngram_viewer CORPUS OUT --queries 'war, cold war' [--case-sensitive] [--smooth 1]` — culturomics n-gram frequency per year from dated filenames, as a table and a line chart (legacy `NGrams_ngram_viewer`). Needs years in the filenames; without dates it fails with the reason the co-occurrence viewer can run undated and this one cannot. Available.
- `nlp-suite sentiment_neural_bert CORPUS OUT --model distilbert-base-uncased-finetuned-sst-2-english` — neural sentiment per sentence via a Hugging Face BERT classifier. Available (needs the `[embeddings]` extra).
- `nlp-suite sentiment_neural_spacy CORPUS OUT --model en_core_web_sm` — neural sentiment per sentence via a spaCy textcat head. Available; vanilla spaCy models ship no sentiment head, so this fails loudly with the install pointer rather than inventing a score.
- `nlp-suite sentiment_neural_stanza CORPUS OUT --language en` — neural sentiment per sentence via Stanza's sentiment processor. Available (needs `stanza.download` of a language package that ships one).
- `nlp-suite sentiment_neural_corenlp CORPUS OUT --server http://localhost:9000` — neural sentiment via the CoreNLP sentiment annotator (the RNTN) on a running Java server. Available; needs the server + models jar.
- `nlp-suite shape_hc CORPUS OUT [--method ward --n-clusters 2 --resample 32]` — hierarchical clustering of story-shape trajectories with a dendrogram (legacy `shape_of_stories_*` HC path). Available.
- `nlp-suite shape_svd CORPUS OUT [--n-components 2 --seed 42]` — SVD shape components over story-shape trajectories (legacy `shape_of_stories_*` SVD path). Available.
- `nlp-suite shape_nmf CORPUS OUT [--n-components 2 --seed 42 --max-iter 200]` — non-negative shape parts over story-shape trajectories (legacy `shape_of_stories_*` NMF path). Available.
- `nlp-suite geocode CORPUS OUT [--provider offline|google|nominatim --limit 200]` — geocode the places named in a corpus to lat/lon (legacy `GIS_geocode/location_util`). Available; the offline provider is a small built-in gazetteer, Google needs `GOOGLE_MAPS_KEY`, Nominatim needs only a network.
- `nlp-suite gis_map INPUT.csv OUT [--lat-col Lat --lon-col Lon --name-col Place --weight-col W --group-col G]` — pin map, heatmap and KML from a geocoded CSV table (legacy `GIS_heatMap*` / `GIS_Google_Earth_main`). Available (CSV input, not a corpus tool).
- `nlp-suite svo_map CORPUS OUT [--provider offline|google|nominatim]` — the SVO to GIS handoff: every subject–verb–object triple placed on a map by its document's location (legacy `SVO` + Google Earth output). Available.
- `nlp-suite clause_svo CORPUS OUT` — clause-tag frequencies + SVO triples. Available.
- `nlp-suite kwic CORPUS OUT --query WORD [--window 5] [--regex] [--case-sensitive]` — Key Words In Context concordance over form or lemma, each row keeping document and sentence provenance (legacy `co_occurrence_main` concordance view). Available.
- `nlp-suite ner CORPUS OUT --movement [--geocode]` — adds person-location movement tracks and an entity-location summary (CAP-NER-03); `--geocode` attaches Lat/Lon from the offline knowledge base. Available.
- `nlp-suite collocations CORPUS OUT [--span window --window 5] [--min-count 3] [--stopwords FILE]` — collocation association measures over word pairs: PMI, PPMI, t-score, z-score, Dunning G2, Dice and log Dice side by side, with adjacent-bigram or unordered-window spans (legacy `NGrams_collocation_statistics_util`, whose single-cell "log-likelihood" is replaced by the real four-cell G2). Sort by G2. Available.
- `nlp-suite explain RESULT.csv [--check X Y] [--json]` — reads a finished result back in plain language (what the table says, and the traps that particular table has actually triggered) and lists the charts worth drawing from it. `--check X Y` answers whether a specific pairing is worth plotting and exits 3 when it is not, which catches the degenerate case of charting a quantity against itself. Every tool run also writes a `readout.md` beside its CSV automatically. New capability with no legacy equivalent. Available.
- `nlp-suite dispersion CORPUS OUT [--parts chunk --chunks 10] [--min-count 5]` — lexical dispersion: Range, Gries' DP, DP norm, Juilland's D and an adjusted frequency, answering whether a frequent word is spread across the corpus or concentrated in one document. New capability with no legacy equivalent; use `--parts chunk` to measure dispersion inside a single long text. Available.
- `nlp-suite lexicon_series CORPUS OUT --terms 'Iraq: iraq, saddam; Vietnam: vietnam, hanoi' [--by year|decade|document|pattern] [--within 'Iraq: iraq']` — dictionary-based content analysis: count your own named word groups along an axis and report them per 1,000 tokens, with the raw counts and the share of sentences beside each rate. `--within` restricts counting to sentences matching a second set of groups, answering what vocabulary travelled with a topic rather than how often the topic appeared. New capability with no legacy equivalent. Available.
- `nlp-suite tfidf CORPUS OUT [--top-n 20] [--min-df 2] [--max-df-ratio 0.8] [--sublinear-tf]` — TF-IDF distinctive terms per document over the shared parse, smoothed idf and L2 normalization, long-format top-N per document rather than the legacy dense document-by-term matrix (legacy `statistics_corpus_tfidf_util`). Available.
- `nlp-suite ngrams CORPUS OUT --network [--min-pmi 3 --top-edges 200]` — also writes a PMI-filtered collocation network as `collocation_network.gexf` plus `collocation_edges.csv` for Gephi. Available.
- `nlp-suite ner CORPUS OUT`, `nlp-suite html_annotator CORPUS OUT` — parser-dependent workflows still registry-excluded (each runs as `python -m tools.<name>`). Every other former 'metadata pending' CLI is now a full registry member selectable in the desktop app and profiler batch: `ngrams` (n-gram frequencies + PMI collocations), `ngram_cooccurrence` (windowed pairs), `conll_wordlist` (POS-filtered wordlist), `corpus_statistics` (per-document diversity), `k_sentences` (first/last K bookends), `svo_compare` (Jaccard over SVO triples), `text_statistics` (surface summary), `table_search` (CoNLL predicate search), `semantic` (baked semantic maps), `knowledge_graph` (offline stub triples), `shapes` (story shape + Sankey + KMeans shape clustering); plus CSV-input `table_charts` (all 13 chart kinds, HTML/PNG/SVG/PDF/xlsx) and `table_wordcloud_gephi` (wordcloud HTML/PNG/butterfly + GEXF) in the desktop visualization category.

## Narrative, character, stories (narrative_analysis_ALL_main, shape_of_stories_main)

- `nlp-suite narrative CORPUS OUT` — per-sentence emotion + length arcs. Available.
- `nlp-suite shapes INPUT.csv OUT` — arc smoothing/plotting workflow (excluded from the registry: metadata pending; runs as `python -m tools.shapes`).
- Character emotion arcs need the coreference + NER decisions (FR-6.2, planned).

## Charts, networks, GIS, graphs (data_visualization_main, charts_Excel_main, wordclouds_main, GIS_main, knowledge_graphs_main)

- `nlp-suite charts INPUT.csv OUT --x X --y Y` — Plotly bar chart (table fallback without plotly; static kaleido export is an open decision). Available.
- `nlp-suite panels INPUT.csv OUT --panel NAME` — a figure built for one analysis rather than one drawn over any table. `--list` shows the registered panels and what each takes; settings are passed as `--set NAME=VALUE`. The first is `keyness_volcano`, which plots effect size (Log Ratio) against strength of evidence (G2): drawn as the bar chart the generic path offers, a keyness table cannot show whether a difference is large or merely well measured. Every mark records the rows it stands for, and the provenance caption is drawn inside the figure so it survives a PNG export. New capability with no legacy equivalent. Available.
- `nlp-suite wordcloud_gephi INPUT.csv OUT --word-col W --weight-col C [--source-col S --target-col T]` — wordcloud HTML + validated GEXF. Available.
- `nlp-suite wordcloud_gephi INPUT.csv OUT --word-col W --weight-col C --image --shape butterfly` — procedural shape masks (no image file needed; currently butterfly). Needs the `[wordcloud]` extra.
- `nlp-suite wordcloud_gephi INPUT.csv OUT --word-col W --weight-col C --image [--mask M.png] [--group-col G --colors 'A=#ff0000,B=#0000ff']` — the legacy `wordclouds_main` raster workflow: spiral layout, optional image mask, per-group recoloring, PNG output (needs the `[wordcloud]` extra: `pip install ".[wordcloud]"` from a checkout, or `pip install wordcloud`). Available.
- `nlp-suite geocode INPUT.csv OUT`, `nlp-suite mapping INPUT.csv OUT` — offline lookup + GIS artifact workflows (excluded: external-service/artifact workflows; run as `python -m tools.<name>`).
- `nlp-suite earth INPUT.csv OUT [--group-col G] [--title T] [--name-col N --lat-col LA --lon-col LO]` — Google Earth tour KML: one animated `gx:Track` per group over a geocoded table (legacy `GIS_Google_Earth_main`, which required a desktop Google Earth install). Excluded from the registry as a GIS artifact workflow; runs as `python -m tools.earth`. It composes with CAP-NER-03: `nlp-suite ner CORPUS OUT --movement --geocode` then `nlp-suite earth RUN/movement_tracks.csv OUT --name-col Location --group-col Entity` turns a corpus into a per-person animated route.
- `nlp-suite knowledge_graph CORPUS OUT` — stub DBpedia/YAGO surface (excluded: needs network clients + keys; runs as `python -m tools.knowledge_graph`).
- Excel chart export is dropped (superseded by Plotly); Gephi flows consume the validated `graph.gexf`.

## Batch, app, environment (corpus_profiler_main, NLP_menu_main, NLP_setup_*_main, sample_corpus_main)

- `nlp-suite profiler CORPUS OUT --analyses readability,narrative` — the batch profiler (plan + linked envelopes + report.md, `--resume-from` supported). Available.
- The Streamlit app (`streamlit run app/Home.py`) replaces the Tkinter menu: gallery (Home), tool forms (Tools), corpus validation (Corpus). Background jobs (`nlp-suite jobs ...`) replace modal RUN-and-wait for long analyses.
- `nlp-suite doctor en`, `nlp-suite install`, `nlp-suite assets --root assets` — environment, setup, and asset verification. Available (reporting commands, no registry specs).
- `nlp-suite compare RUN_A RUN_B`, `nlp-suite freeze_legacy` — run comparison harness and legacy pinning (harness/maintenance commands, no registry specs).
- `nlp-suite sql`, `nlp-suite pcace`, `nlp-suite pcace_analysis` — database workflows (planned; PC-ACE must be decomposed first).
- SRL (legacy `SRL_main.py`) has no NG equivalent yet: SRL runs in an isolated Python 3.8 env upstream and is out of scope until FR-5.3.

## Deliberately not ported

- Tkinter messageboxes and `OpenOutputFiles` popups: diagnostics ride the envelope; nothing blocks on a click.
- Per-sentence/per-word Stanza re-parses: one shared parse feeds every analysis.
- Run-to-run row shuffles from nondeterministic traversals: visit order is normalized and byte-identical.
- Silent fallbacks (missing lexicon → zeros, missing model → hash vectors): every gap is a loud diagnostic with the fix.
