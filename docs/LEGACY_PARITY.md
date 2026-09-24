# Legacy Parity Report — NLP Suite 1.6.38 vs nlp-suite-ng

> Historical baseline, not the current implementation inventory. The rebuild
> has changed substantially since this comparison (including the desktop UI).
> Use [REPLACEMENT_LEDGER.md](REPLACEMENT_LEDGER.md) for acceptance tracking and
> [DESKTOP.md](DESKTOP.md) for the current desktop scope and release limits.
>
> The snapshot is kept as written rather than rewritten. Where a later packet
> has overtaken a claim, the claim carries an inline
> **"Superseded since this snapshot"** note naming the capability ID, so the
> original assessment and what replaced it are both readable. Search for that
> phrase to list everything that has moved.

A functional comparison between the legacy suite (`../NLP-Suite-1.6.38/`,
read-only oracle) and this rebuild. Every claim about the legacy is verified
against a file and line in its source — nothing is taken from rumor.

Sources for legacy facts:
- `NLP-Suite-1.6.38/src/` — 217 Python files, 114,731 lines, 2,323 functions.
- `NLP-Suite-1.6.38/error.md` — the project's own file-by-file defect review
  (~761 entries: 424 HIGH / 222 MED / 115 LOW), organized in 20 chunks.
- `NLP-Suite-1.6.38/FIXES.md`, `MEMO_Development_Notes.md`, `config/`.

---

## 1. Shape of the two systems

| | NLP Suite 1.6.38 | nlp-suite-ng |
|---|---|---|
| Source size | 217 files, ~114.7k lines, 2,323 functions | 107 files, ~12.5k lines |
| UI | Tkinter; every tool is a GUI window (`GUI_util.window` imported by 154 modules; Tk root created at import time in `GUI_util.py:21`) | None below `app/`; 34 thin CLIs (`python -m tools.<tool>`) + a generic Streamlit envelope viewer |
| Entry points | `NLP_menu_main.py` launches 48 `_main.py` GUIs via `subprocess.call` | one CLI per tool; `python -m` |
| Error channel | `messagebox.showinfo`/`askyesno` (184 modules import `messagebox`) | `Result[T]` + structured `Diagnostic`s |
| Failure mode | bare `except:` (224 occurrences) that print + messagebox + return None | typed Results; `filterwarnings=error`; no bare except (lint-enforced) |
| Dependencies | 72 runtime packages incl. torch, tensorflow, gensim, nltk, MALLET (Java), CoreNLP (Java), Gephi, Google Earth | 4 core deps (pandas, pyarrow, scipy, scikit-learn); stanza/spacy/converters/plotly/streamlit are optional extras |
| Test suite | 25 test files (many requiring GUI stubs; golden-diff harness) | 47 test files, 717 collected (713 passed + 4 skipped: model-integration/environment state); C6-1..20 corrections applied locally |
| Packaging | PyInstaller: 1.2 GB exe (`MEMO_Development_Notes.md` §1) | `pip install -e .` (setuptools) |

## 2. Feature-by-feature

Legend: **full** = implemented in the new suite · **partial** = core logic
exists, legacy richness not fully reproduced · **gap** = not yet in the new
suite (tracked below) · **stub** = deterministic stand-in behind the same
interface, model/network out of scope by design.

### Intake & file operations
| Legacy tool | Status in new suite | Notes |
|---|---|---|
| file checker (encoding chain, emptiness) | full | `core/file_ops/checker.py` + `core/io/reader.py` (read is read-only; legacy `IO_csv_util.get_csv_data` **rewrote the user's input file** stripping NUL bytes as a side effect of reading — error.md Chunk 1) |
| file cleaner | full | `core/file_ops/cleaner.py`; legacy's 891-line cleaner covered many formats |
| file converter (csv/tsv/pdf/docx/rtf/html) | partial | TXT/CSV/TSV/HTML are built-in; PDF/DOCX/RTF are implemented behind the optional `converters` extra and still need fixture pairs (valid sample file + expected extracted text, crafted in-house); directory conversion is best-effort and records per-file failures as warnings |
| file merger | full | `core/file_ops/merger.py` |
| file splitter (by length/words/lines/delimiter/keyword) | full | one `split_text` engine replaces 10 `file_splitter_By*` utils |
| file search (by word) | full | `search_in_text`; legacy's 784-line byWord util also searched CoNLL/CSV shapes |
| file matcher / sample corpus | partial | `match_files`, `sample_corpus` (seeded); legacy's matcher used fuzzy string matching too |
| file spell checker | implementing (C6-12) | `spellcheck.py` — explicit wordlist design replaces the legacy multi-backend stack (pyspellchecker/autocorrect absent here) |
| corpus validation | full | `core/data/validation.py` |
| corpus sampler | full | `tools/` + `core/file_ops/search.py::sample_corpus` |

### Parsing / pipelines
| Legacy | Status | Notes |
|---|---|---|
| Stanza / spaCy / CoreNLP parsing → CoNLL | full when a verified model is loadable (stanza/spacy), stub (corenlp) | one `Pipeline` protocol + `PipelineCache` keyed by (backend, language, tasks); no blank fallback; model integrations skip when the environment cannot load resources |
| language support gate | full | `core/config.py` validates (backend, language) at construction; legacy's check was literally `if 'stanfordnlp' or 'stanza' in missingModules:` (`IO_libraries_util.py:160` — always True) |
| Universal→Penn tagset normalization | full | `core/conll/normalize.py`; legacy normalized in memory then re-read raw files with the wrong tagset filter (silently empty outputs) |
| CoNLL table versioning | full | name-addressed `Col` enum + `*.schema.json` sidecar; legacy indexed columns positionally and commented three historical layouts inside the code |
| sentence division | full | `core/conll/division.py`; legacy dropped the final sentence (patched in `CoNLL_util.py:490` per FIXES.md #12) and emitted a phantom first sentence for Document_ID '1' |

### Analysis engines
| Legacy | Status | Notes |
|---|---|---|
| CoNLL word analyses (noun/verb/adjective/adverb/function/ratio) | full | one parameterized engine (`conll_wordlist.py`) replaces 6 near-clone utils |
| CoNLL clause analysis + clause tag frequencies | partial | `clause_svo.py::clause_frequencies` counts the Clause Tag column; legacy's richer clause taxonomy (`Stanford_CoreNLP_clause_util.py`) is not ported |
| SVO extraction (via dependency heads) | full | `clause_svo.py::extract_svo`; legacy used CoreNLP enhanced++ dependencies JSON files |
| SVO compare (Jaccard) | full | `svo_compare.py` |
| CoNLL table search (composable queries) | full | `table_search.py` with typed filters; legacy's `CoNLL_table_search_util.py` (726 lines) chopped doc IDs with `str(tok_Document_ID)[:-2]` (`:679`) — corrupts any ID ≥ 10 |
| K-sentences (first/last K bookends) | full | `k_sentences.py` |
| N-grams + collocations (PMI) | full | `ngrams.py` + registered in the tool registry (desktop app + profiler batch selectable, executor adapters `_adapt_ngrams`/`_adapt_ngram_cooccurrence`); `ngram_cooccurrence.py` (windowed pair counts) likewise registered |
| Registry coverage: every CLI registered or explicitly excluded | full | 44 registered tool specs + 15 documented exclusions (maintenance/external-service/intake commands); desktop app exposes all 40 selectable workflows (31 corpus analyses via executor adapters + 9 CSV table workflows incl. charts/wordcloud/GEXF); the desktop-only table_* specs share engines with the CLI |
| N-gram co-occurrence | full | `ngram_cooccurrence.py` (window/pair counting); PMI collocation network export added since this snapshot (`ngrams --network` -> GEXF) |
| KWIC concordance | full | `kwic.py` (CAP-INTAKE-19), added since this snapshot: form/lemma, regex and case-sensitive modes, document + sentence provenance per row |
| NER entity timeline / location tracking | full | `ner.py`; movement tracks ported since this snapshot (`movement.py`, `ner --movement`, CAP-NER-03) |
| Sentiment: VADER | full | `sentiment_vader_anew.py::vader` (compact baked lexicon) |
| Sentiment: ANEW | full | same module (`anew`) |
| Sentiment: SentiWordNet | full | `sentiment_swn_hedono.py::sentiwordnet` |
| Sentiment: hedonometer | full | same module (`hedonometer`) |
| Sentiment: NRC emotion lexicon | full (optional) | `nrc.py` (FR-4.3): 10-column Plutchik output over the nrclex-bundled lexicon |
| Character emotion arcs (per-character NER arcs) | partial | `core/narrative/arcs.py` does per-sentence VADER arcs; the legacy's per-character arcs (`character_emotion_arcs_util.py`, 1158 lines, binning + HTML) are richer |
| Statistics: text (sentences/tokens/syllables per doc) | full | `text_statistics.py` |
| Statistics: lexical diversity (TTR, RootTTR, LogTTR/Herdan, Yule K) | full | `corpus_statistics.py`; Guiraud/MTLD/vocd-D in `lexical_diversity.py` (C6-8 short-text policy pending) |
| Statistics: readability | implementing (C6-8) | `readability.py`: FRE, Flesch-Kincaid, Gunning Fog, Coleman-Liau, ARI (all 5 legacy formulas) + SMOG (new-from-formula; legacy `statistics_txt_util` used unpinned textstat for SMOG, and `statistics_corpus_readability_util` itself contains exactly 5) |
| Statistics: TF-IDF | full | `topic_model.py` (TF-IDF scoring) |
| Statistics: word frequency distribution | full | `conll_wordlist.py` |
| Statistical tests (chi-square, crosstab, keyness, MWU, KW+Dunn, Mann-Kendall, rank correlation) | implementing (C6-5/C6-6) | `stats_categorical.py`, `stats_groups.py`, `stats_trends.py`; kappa/change-point/permutation/ARI/Bayes/silhouette families not yet ported. Corpus-level keyness between two document groups ported since this snapshot (`keyness.py`, CAP-STATS-11) |
| CSV stats (describe, correlation) | full | `csv_stats.py` |
| Topic modeling (gensim LDA / MALLET / BERT) | partial | `lda.py` (seeded Gensim LdaModel, FR-5.7) behind `topic_model.py`; TF-IDF partition remains as the no-gensim default; MALLET adapter written (`mallet.py`) but needs the binary; transformer-embedding topics ported since this snapshot (`bert_topics.py`, CAP-TOPIC-04) — KMeans over document embeddings, not the BERTopic package |
| Word embeddings (word2vec + distances + t-SNE) | full (optional) | `word_embeddings.py` (FR-5.5): seeded Gensim train, neighbours, sklearn t-SNE; legacy ran Gensim word2vec + tsne |
| t-SNE interactive scatter (`word2vec_tsne_plot_util`) | full | `core/viz/embeddings.py::tsne_html` — Plotly scatter with word labels (2D; the legacy 3D variant deliberately not carried); plain HTML coordinate table + `TSNE_PLOTLY_UNAVAILABLE` without plotly; `word_embeddings` CLI writes tsne.html next to tsne.csv |
| Story-shape clustering (`shape_of_stories_vectorizer_util`) | full (deterministic core) | `core/viz/shape_clusters.py::cluster_shapes` (CAP-VIZ-08): KMeans over resampled story-shape trajectories (tokens/noun/verb), k from the legacy PCA explained-variance heuristic, seeded + reproducible; `tools/shapes.py --clusters N`; legacy bucketed *sentiment* per window — the deterministic shape metrics replace the sentiment dependency (documented, not silent) |
| WSI (word-sense induction, BERT clustering) | partial (optional) | `contextual.py::wsi_senses` over a real `TransformerBackend` (lazy transformers+torch, FR-5.6); legacy: `WSI_*.py` with transformers |
| Semantic aggregation (WordNet/VerbNet/FrameNet classes) | full (optional) | `wordnet.py` (FR-4.4, real NLTK hyponym trees), `verbnet.py`/`framenet.py` (FR-4.5, NLTK lazy); replaces the earlier baked-map stub |
| Semantic similarity / plagiarist / non-related documents | implementing (C6-9) | `string_similarity.py` (Levenshtein port) + `doc_similarity.py` + `doc_duplicates.py`; unrelated-document workflow still owed |
| BERT extractive summarization | partial | `bert_extract.py` (FR-5.9): centroid extractive over the injectable embedding backend; the legacy's unpinned `bert-extractive-summarizer` package intentionally not reproduced |
| SRL (semantic role labeling) | gap (stub exists) | legacy shelled to an isolated transformer_srl env (`SRL_worker.py`); new suite defines `srl_backend.py` as an optional stub |
| Coreference (neural) | stub | new: deterministic lemma clustering (`coreference.py`); legacy: CoreNLP neural coref with a compare GUI |
| Narrative arcs / story shape | partial | `shapes.py::story_shape` + `arcs.py`; legacy's shape_of_stories used KMeans over TF-IDF vectors (`shape_of_stories_vectorizer_util.py`) |
| Style analysis (abstract/concrete, iconicity) | partial | `style.py` (FR-2.8): per-sentence Brysbaert concreteness + Winter iconicity over user-supplied `assets/` lexicons |
| Nominalization detection | partial | `nominalization.py` (FR-2.7): noun-only deverbal detection, NLTK derivational morphology + suffix prefilter |
| Sentence complexity | implementing (C6-7) | `sentence_complexity.py` — dependency metrics + verbatim Yngve/Frazier functions; production constituency workflow blocked on FR-5.2 |
| HTML annotator (extract, dictionary, gender) | full | `html_annotator.py` |
| Wordclouds | full (HTML) / full-optional (raster) | `wordcloud_gephi.py::wordcloud_html` (always) + `wordcloud_image` (the legacy `wordclouds_util.py` port: spiral layout, image masks, per-group colors, PNG; behind the `[wordcloud]` extra) |
| Gephi GEXF / network graphs | full | `gephi_gexf` + `shapes.py::network_gexf`; legacy's `network_graph_app.py` was a Tk app |
| GIS geocode / KML / heatmap / distance / Earth tours | full (offline) | `gis/` with a baked KB; legacy called Google Maps APIs and drove Google Earth. Tour KML ported since this snapshot (`mapping.py::tour_kml`, `tools/earth.py`, CAP-GIS-07) |
| GIS symbolic/actor typologies | partial | `symbolic.py` (FR-4.6): 10 space types + 12 actor types over user-supplied typology assets; legacy's `lib/social_actor_typology.csv` |
| Knowledge graphs (DBpedia/YAGO/Wikipedia/Spotlight) | partial | `kg/dbpedia.py` (FR-6.8): real Spotlight annotate client behind `--source dbpedia`; offline baked KB stays the default; legacy made live SPARQL calls and set `ssl._create_unverified_context` (`knowledge_graphs_DBpedia_util.py:27`) |
| PC-ACE DB tools | partial | `pcace/` + `data/sql.py` cover create/validate/analyze/query; legacy's `DB_PCACE_data_analysis_util.py` is 7,398 lines of analysis workflows |
| Data manipulation (concat/merge/pivot/melt) | full | `data/manipulation.py` |
| Data visualization (crosstabs etc.) | partial | charts + csv_stats cover the core |
| Corpus profiler (31 analyses in one batch) | partial | `core/profiler/profiler.py` + `tools/profiler.py`; the legacy's 31-analysis profiler is its crown jewel — the new version runs a subset |
| Charts (Plotly/Excel/matplotlib) | full | `viz/chartspec.py` + `viz/plotters.py`: bar/line/scatter/histogram/box/heatmap + legacy-parity pie/sunburst/treemap/violin/radar/waffle/calendar with typed aggregation/normalization semantics; `viz/charts_excel.py` exports the legacy's native Excel workbooks (`--format xlsx`: Chart sheet first, Data sheet, per-group series, labels low, single-series legend off; bar/line/pie/scatter/radar/bubble) — only the macro .xlsm hover-over path is intentionally not ported (embedded VBA templates; HTML covers hover interactively) |
| Install check | full | `core/install/check.py` |

### New-suite-only (no legacy equivalent)
- Artifact envelopes (`result.json` with params, input hashes, outputs,
  diagnostics) and run directories — the legacy coupled tools by filename
  globbing (e.g. `_find_existing_parse_csv` reuse probes in the profiler).
- OutputWriter with write-safety (refuses to write inside inputs; single
  writer). The legacy's profiler had to *suppress dialogs and catch
  BaseException* to run unattended, and `error.md` A7 documents rmtree on
  output dirs.
- The Streamlit viewer rendering any run by `kind`.
- Offline-by-default design: the new suite runs its whole test battery with
  no network; legacy tools hit DBpedia/Google APIs directly.

## 3. The legacy "greatest hits" (verified)

These are the defects the new suite's ARCHITECTURE.md bans, each confirmed in
the legacy source during this review:

1. **`eval()` on data** — `charts_Plotly_util.py:112-138` evals dataframe/ops
   strings pulled from user-facing config CSVs.
2. **`sudo Python … shell=True`** — `corpus_checker_PCACE_data_main.py:45,56,66,97,268`
   launches other suite tools as root via shell strings.
3. **`ssl._create_unverified_context`** — `knowledge_graphs_DBpedia_util.py:27`,
   module level, disabling cert verification for the whole process.
4. **Read mutates input** — `IO_csv_util.py:44-51` rewrites the user's CSV
   in place (NUL stripping) on every read.
5. **Messagebox as error channel** — 184 modules import `messagebox`; the
   permission-error retry loop in `IO_csv_util.df_to_csv` (lines 208-229)
   traps the user in an endless modal if the CSV is open in Excel.
6. **pip-install-at-import** — `install_all_Python_packages` runs on import in
   every GUI module and can shell out to pip mid-import.
7. **Tk root at import** — `GUI_util.py:21` (`window = tk.Tk()`), which is why
   nothing was testable headless.
8. **`if 'stanfordnlp' or 'stanza' in missingModules:`** — `IO_libraries_util.py:160`,
   always-True language/dependency check (the config gate's ancestor).
9. **Doc IDs by string chopping** — `CoNLL_table_search_util.py:679`
   `str(tok_Document_ID)[:-2]`; `character_emotion_arcs_util.py:1108`
   `os.path.basename(inputFilename)[:-4]` (extension chopping).
10. **Positional column indexing** — `recordID_position = 9`, `sentenceID_position = 10`
    constants scattered across the CoNLL utils, with three historical layouts
    documented in comments (`CoNLL_util.py` position blocks).
11. **Bare `except:` swallow** — 224 occurrences, e.g. `sentence_division`
    (`CoNLL_util.py:458`) catches everything, prints, shows a messagebox, and
    then falls through to `return` with whatever partial state existed.
12. **os.chdir at import** — `IO_files_util.py:44` (`os.chdir(dir_path)`), plus
    12 `shell=True` sites and 221 `global` statements.
13. **One implementation, six copies** — the six `CoNLL_*_analysis_util.py`
    files differ only in their POS tag lists (the new suite's
    `conll_wordlist.py` is the single replacement), and 10 `file_splitter_By*`
    utils collapse into one `split_text`.

## 4. What the new suite deliberately does NOT have (yet)

Ordered by how much a corpus analysis would miss them:

1. **Remaining statistical tests** (kappa, change-point, permutation,
   ARI/Bayes/silhouette families) — chi-square/crosstabs/G2, MWU/KW/Dunn,
   and Mann-Kendall/rank-correlation families are ported
   (`stats_categorical.py`, `stats_groups.py`, `stats_trends.py`).
2. **Readability fixture hardening** — all 5 legacy formulas + SMOG are
   ported (`readability.py`); owed work is syllable-count unification and
   remaining acceptance evidence (C6-8).
3. **Remaining model-backed backends** — MALLET needs its binary;
   the BERTopic *package* is not used; SRL frame mapping over the worker
   protocol is owed (FR-5.3); neural coreference (FR-5.4) is a documented
   follow-up. **Superseded since this snapshot:** transformer-embedding topic modeling now
   ships as `bert_topics` (CAP-TOPIC-04).
4. **Fixture pairs for valid PDF/DOCX/RTF conversion, and full asset-backed spell checking** — adapters exist behind optional extras; in-house fixtures (sample file + expected extracted text) and asset provenance remain.
5. **Remaining domain analyses** — sentence complexity's constituency
   workflow (blocked on FR-5.2); character-arc binning + HTML (FR-6.2);
   enhanced clause taxonomy (FR-6.1). **Superseded since this snapshot:** movement tracks
   (FR-6.4) now ship as CAP-NER-03.
6. **PC-ACE decomposition children** (FR-6.9a–f) — the 24,618-line legacy
   workflow is split and planned, not yet ported.
7. **Character-level emotion arcs** — needs real NER (the new suite's
   pipeline emits NER="O" unless the backend provides it). **Superseded since this snapshot:**
   movement tracking ships as CAP-NER-03; it inherits the same dependency on
   a backend that actually emits entity tags.
8. **Excel chart export, static chart export** — export-format parity, low
   priority. **Superseded since this snapshot:** Google Earth KML tours now ship as CAP-GIS-07
   (`tools/earth.py`), without the Google Earth desktop install the legacy
   required.

## 5. Verdict

The legacy suite is a 114k-line research instrument: enormous feature
coverage — 48 GUI entry points across 9 domains (files, parsing, statistics,
semantics, GIS, DB/PC-ACE, narrative, sentiment, topics) — held together by
GUI-owned sequencing, filename coupling, and 761 self-documented defects. The
rebuild has expanded analysis and infrastructure coverage. Current desktop
verification is recorded in [DESKTOP.md](DESKTOP.md); the default test gate
excludes model-integration tests, which must be run and reported separately.
**No audited overall parity percentage is established.** It is strongest where the legacy was weakest
(correctness, provenance, headless operation), weakest where the legacy was
strongest (breadth: MALLET/BERTopic backends, SRL frame mapping, neural
coreference, the profiler's full 31-analysis batch, and desktop coverage of
non-corpus workflows).

Replacement is decided by class-workflow completeness, not output parity:
the bar is that a student runs their whole corpus analysis in the new
suite (Gate B evidence is stratified accordingly — definitional evidence
first, legacy-run spot-checks only for glue — per owner decision
2026-09-03). Legacy defects catalogued above are improvements to make, not
behavior to replicate.

The port order in §4 follows value: statistics and readability are pure
table math with no new dependencies; everything model-backed stays behind
optional extras. The core install currently declares four dependencies in
`pyproject.toml` (pandas, pyarrow, scipy, and scikit-learn).
