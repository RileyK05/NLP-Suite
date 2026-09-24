# Build Plan — chunked execution

How `nlp-suite-ng` gets built. One chunk at a time; each chunk is finished and
verified before the next starts.

This is the `class_build_order.md` + runbook that `docs/ARCHITECTURE.md`
refers to but that did not exist.

---

## 0. Why chunks

A chunk is the unit of work given to an agent in one go. The size limit is
deliberate: **one cohesive concept, or two or three tiny ones.** Not a wave.

The legacy rebuild was attempted as "fix the bugs in this file," in a codebase
where the cause lived in module-level globals shared by 200 files. Local fixes
could not hold, so the work never converged. The chunks below are ordered so
that every chunk depends only on chunks already finished and verified.

---

## 1. The chunk contract

A chunk is DONE only when all of these hold:

1. **Module** — `core/...` holds the logic. Pure: data in, `Result` out.
2. **CLI** — `tools/<name>.py`, thin, ~15 lines. The only place argv is parsed.
3. **Tests** — `tests/test_<name>.py`. Not smoke tests: they pin the contract
   and any named legacy defect this module replaces.
4. **Green** — `pytest` passes, `ruff check` clean, `mypy --strict` clean.
5. **No import-time side effects** — `tests/test_no_import_side_effects.py`
   covers the new module automatically. No network, no model load, no Tk,
   no `os.chdir`, no pip at import.
6. **Ledger row updated** in `docs/CHUNK_LEDGER.md`.
7. **Committed**, on its own, with a message saying what and why.

If a chunk cannot meet the contract, stop and flag it. A flagged failure is a
review; a silently skipped rule is a bug.

---

## 2. How to read the legacy code

The legacy suite (`../NLP-Suite-1.6.38/src/`) is a **catalogue of what the tool
set is**, not a source of code to copy.

- **Read it to answer "what is this?"** — `topic_modeling_mallet_util.py` is
  MALLET topic modeling. `sentiment_analysis_VADER_util.py` is VADER
  sentiment. Recognising the package is the goal.
- **Do not port its structure.** It indexes columns by position through
  globals, swallows errors in bare `except:` blocks, and downloads models at
  import.
- **Note where it misbehaves**, so the misbehaviour is not rebuilt. That is
  what `error.md` is for.
- **Parity is at the package level.** The new suite must offer the same
  capabilities under the same names. It must not reproduce the same numbers.

Exception: where the defect review explicitly called code correct — the
Universal-to-Penn tag mapping in `CoNLL_util.py` — port it. That was done in
C2.

---

## 3. Setup policy (a product requirement, not a nicety)

Setup difficulty is a reported reason students drop the class. Measured from
the legacy source:

- **142 of 213 files** call `install_all_Python_packages` at import — pip runs
  at startup, in two-thirds of the suite, and several paths `sys.exit(0)` on
  failure so the app vanishes with no message.
- Models download at import (`Stanza_functions_util.py:35`), so first launch
  needs the network.
- `charts_util.py` has a U+FEFF byte at position 0 and **does not compile**.

Rules for the new suite:

- **Nothing downloads until a tool needs it.** No model, no data pack, no
  server at import.
- **The default path needs no API key, no Java, no external binary.**
- Anything heavier — CoreNLP, MALLET, SRL, GIS — is an **optional extra**
  behind `[project.optional-dependencies]`.
- **GIS keys come from the environment.** The legacy suite hardcoded a Google
  key (the `S105` finding). Never in source; `tests` run offline with
  injected clients.

---

## 4. Chunk sequence

Status: see `docs/CHUNK_LEDGER.md`.

### Phase 0 — Foundation (the substrate)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C1 | Contracts | `core/result.py`, `core/config.py` | — | — |
| C2 | CoNLL table | `core/conll/{schema,normalize,division}.py` | `CoNLL_util.py`, `Stanza_util.py` cols | C1 |
| C3 | Intake | `core/io/reader.py` — `Document`, `Corpus`, encoding fallback, sha256 | `IO_files_util`, `file_checker_util` | C1 |
| C4 | Custody | `core/artifacts/envelope.py`, `core/io/writer.py` — the only writer | `corpus_profiler_util` manifest | C1, C3 |
| C5 | Parsing | `core/pipelines/{protocol,cache,spacy_backend}.py` | `spaCy_util`, `Stanza_util` | C2, C4 |
| C6 | Gates | `tests/test_no_import_side_effects.py`, `tests/test_layering.py`, CI | `error.md` A1–A13 | C1–C5 |

### Phase 1 — Vertical slice (prove the shape end to end)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C7 | First tool | `core/analysis/conll_wordlist.py` + CLI | the 6 CoNLL analyzer clones | C5 |
| C8 | CLI harness | `tools/_cli.py` shared plumbing | — | C7 |
| C9 | Viewer | `app/` — Streamlit, renders envelopes by `kind` | `NLP_menu_main` | C4 |
| C10 | End to end | `tests/fixtures/` corpus, `tests/test_end_to_end.py` | — | C7, C9 |

**Stop after C10 and review before mass-producing tools.**

### Phase 2 — File & corpus intake (legacy Wave A)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C11 | Check & clean | file checker, file cleaner | `file_checker_util`, `file_cleaner_util` | C10 |
| C12 | Convert | file converter | `file_converter_util` | C10 |
| C13 | Split | one parameterized splitter engine | 8 `file_splitter_*` files | C10 |
| C14 | Merge & match | merger, matcher | `file_merger_util`, `file_matcher_util` | C10 |
| C15 | Classify | date classifier, NER classifier | `file_classifier_*` | C10 |
| C16 | Find & sample | file search, filename utils, sample corpus | `file_search_*`, `file_filename_util`, `sample_corpus_util` | C10 |

Dropped: `hashfile.py` (eval-as-RCE, replaced by the envelope in C4).

### Phase 3 — Parsing backends (Wave B)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C17 | Stanza backend | `core/pipelines/stanza_backend.py` | `Stanza_util`, `Stanza_functions_util` | C5 |
| C18 | CoreNLP backend | optional `[corenlp]` extra | `Stanford_CoreNLP_util` | C5 |
| C19 | SRL backend | optional extra, isolated env | `SRL_util`, `SRL_worker` | C5 |

### Phase 4 — CoNLL analysis (Wave C)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C20 | Word analysis engine | one parameterized analyzer | 6 `CoNLL_*_analysis_util` → 1 | C7 |
| C21 | Table search | CoNLL search | `CoNLL_table_search_util` | C7 |
| C22 | K-sentences | K-sentence windows | `CoNLL_k_sentences_util` | C7 |
| C23 | Clauses & SVO | clause analysis, SVO extraction | `CoNLL_clause_analysis_util`, `SVO_util` | C7 |

### Phase 5 — Statistics (Wave D)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C24 | Corpus statistics | lexical diversity, readability, tfidf, frequency | 4 `statistics_corpus_*` | C20 |
| C25 | Text statistics | `statistics_txt` rewritten | `statistics_txt_util` | C20 |
| C26 | CSV stats & tests | `statistics_csv`, statistical tests | `statistics_csv_util`, `statistics_statistical_tests_util` | C20 |

### Phase 6 — Sentiment, NER, coreference (Wave E)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C27 | Sentiment I | engine + VADER + ANEW | `sentiment_analysis_VADER/ANEW_util` | C20 |
| C28 | Sentiment II | SentiWordNet + hedonometer + BERT | `sentiment_analysis_*`, `BERT_util` | C27 |
| C29 | NER | entity timeline, location tracking | `NER_*_util` | C20 |
| C30 | SVO compare | `SVO_compare_util` port | `SVO_compare_util` | C23 |
| C31 | Coreference | coreference resolution | `StanfordCoreNLP_coreference_util`, `coreference_neural_util` | C5 |

### Phase 7 — Topic models, n-grams, embeddings (Wave F)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C32 | Topic models | gensim + MALLET | `topic_modeling_gensim/mallet_util` | C20 |
| C33 | Word embeddings | word2vec train/distances/tsne | `word2vec_*` | C20 |
| C34 | N-grams | n-grams + collocation statistics | `NGrams_util`, `NGrams_collocation_*` | C20 |
| C35 | Co-occurrences | n-gram co-occurrences | `NGrams_CoOccurrences_util` | C34 |
| C36 | Contextual embeddings | BERT embeddings, WSI | `BERT_util`, `WSI_*` | C20 |

Dropped: `topic_modeling_bert_util` (abandoned notebook).

### Phase 8 — Annotators & knowledge graphs (Wave G)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C37 | HTML annotator | dictionary, extractor, gender | `html_annotator_*` | C20 |
| C38 | Knowledge graphs | DBpedia, YAGO, Wikipedia | `knowledge_graphs_*` | C20 |
| C39 | Semantic aggregation | WordNet / VerbNet / FrameNet, sentence analysis | `semantic_aggregation_*`, `sentence_analysis_util` | C20 |

Dropped: `html_annotator_extractor_util_NEW` (cannot be imported),
`knowledge_graphs_DBpedia_util_SPARQL` (unwired).

### Phase 9 — Visualization (Wave H)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C40 | Charts | one Plotly module | `charts_util`, `charts_Plotly_util` | C24 |
| C41 | Wordclouds & graphs | wordclouds, Gephi GEXF | `wordclouds_util`, `Gephi_util` | C24 |
| C42 | Shapes & networks | shape of stories, Sankey/network | `shape_of_stories_*`, `network_graph_app` | C24 |

Dropped: `charts_Excel_util` unless Excel output is required.

### Phase 10 — Data & database (Wave I)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C43 | Data manipulation | merge/concat/reshape | `data_manipulation_util` | C26 |
| C44 | SQL | safe query layer | `DB_SQL_main/util` | C43 |
| C45 | Corpus validation | corpus checker | `corpus_checker_PCACE_data_main` | C43 |

### Phase 11 — PC-ACE (Wave J)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C46 | PC-ACE core | database + grammar | `DB_PCACE_data_analysis_util` | C45 |
| C47 | PC-ACE analysis | analysis + validation | `DB_PCACE_data_analysis/validation_main` | C46 |

### Phase 12 — GIS (Wave K, optional extra)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C48 | Geocoding | geocode, location, distance | `GIS_geocode/location/distance_util` | C29 |
| C49 | Mapping | heatmaps, Google Earth, symbolic | `GIS_heatMap*`, `GIS_Google_Earth_main`, `GIS_symbolic_*` | C48 |

### Phase 13 — Corpus profiler (Wave L, last)

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C50 | Profiler | orchestrates the suite | `corpus_profiler_util` | most prior |
| C51 | Narrative & arcs | emotion arcs, narrative/syntactic/sentiment menus | `character_emotion_arcs_util`, `*_ALL_main` | C50 |

### Phase 14 — Packaging

| # | Chunk | Produces | Legacy reference | Depends |
|---|---|---|---|---|
| C52 | Install | extras, one-command setup, install docs | `setup_Windows`, `setup_Mac` | most |

---

## 5. Full tool inventory

All **54** legacy GUI entry points, mapped. Nothing is dropped without a note.

<details>
<summary>The mapping (54 tools → chunks)</summary>

| Legacy GUI | Chunk | Note |
|---|---|---|
| `charts_Excel_main` | — | dropped, superseded by Plotly |
| `CoNLL_table_analyzer_main` | C20, C21 | |
| `coreference_main` | C31 | |
| `corpus_checker_PCACE_data_main` | C45 | |
| `corpus_profiler_main` | C50 | |
| `data_manipulation_main` | C43 | |
| `data_visualization_main` | C40 | |
| `DB_PCACE_data_analysis_main` | C47 | |
| `DB_PCACE_data_validation_main` | C47 | |
| `DB_SQL_main` | C44 | |
| `file_checker_converter_cleaner_main` | C11, C12 | |
| `file_checker_pre_processing_pipeline_main` | C11 | |
| `file_classifier_main` | C15 | |
| `file_handler_ALL_main` | C11 | |
| `file_manager_main` | C11 | |
| `file_matcher_main` | C14 | |
| `file_merger_main` | C14 | |
| `file_search_ALL_main` | C16 | |
| `file_search_byWord_main` | C16 | |
| `file_spell_checker_main` | C16 | |
| `file_splitter_main` | C13 | 8 util files → one engine |
| `GIS_distance_main` | C48 | |
| `GIS_Google_Earth_main` | C49 | |
| `GIS_main` | C48 | |
| `GIS_symbolic_main` | C49 | |
| `html_annotator_gender_main` | C37 | |
| `html_annotator_main` | C37 | |
| `knowledge_graphs_main` | C38 | |
| `narrative_analysis_ALL_main` | C51 | |
| `NER_main` | C29 | |
| `NGrams_CoOccurrences_main` | C35 | |
| `NLP_menu_main` | C9 | becomes viewer navigation |
| `NLP_setup_external_software_main` | C52 | becomes optional-extras docs |
| `NLP_setup_IO_main` | C52 | |
| `NLP_setup_package_language_main` | C1 | becomes `NLPConfig` |
| `NLP_welcome_main` | C9 | |
| `nominalization_main` | C20 | |
| `parsers_annotators_main` | C5, C17, C18 | |
| `sample_corpus_main` | C16 | |
| `semantic_aggregation_main` | C39 | |
| `semantic_analysis_main` | C36 | |
| `sentence_analysis_main` | C39 | |
| `sentiment_analysis_main` | C27, C28 | |
| `sentiments_emotions_ALL_main` | C27 | |
| `shape_of_stories_main` | C42 | |
| `SRL_main` | C19 | |
| `statistics_csv_main` | C26 | |
| `statistics_txt_main` | C25 | |
| `style_analysis_main` | C24 | |
| `SVO_main` | C23, C30 | |
| `syntactic_analysis_ALL_main` | C51 | |
| `topic_modeling_main` | C32 | |
| `word2vec_main` | C33 | |
| `wordclouds_main` | C41 | |

</details>

---

## 6. Invariants, restated

From `docs/ARCHITECTURE.md` §10. Every chunk is checked against these.

| # | Invariant | Enforced by |
|---|---|---|
| R1 | No import-time side effects | `tests/test_no_import_side_effects.py` |
| R2 | No bare `except:` | ruff `E722` |
| R3 | Reads never write; one writer | `OutputWriter` refuses input paths |
| R4 | Errors are `Result`; no UI below `app/` | `tests/test_layering.py` |
| R5 | Models load once | the pipeline cache |
| R6 | No global mutable state | frozen `NLPConfig`, passed explicitly |
| R7 | One return shape per function | `Result[T]`; mypy |
| R8 | Tools interoperate via envelopes | no filename globbing |
| R9 | CoNLL columns by name | `Col` enum; sidecar required |
| R10 | One implementation per concept | shared logic lives in `core/` |
| R11 | No secrets; `pathlib`; one subprocess wrapper | ruff `S`, review |

**Never reintroduce:** `eval` on files · `os.chdir` / `sys.path` mutation ·
runtime pip-install guards · messageboxes as error channels ·
`str.split('')` · `[:-4]` extension chopping · `if 'a' or 'b' in list` ·
`shell=True` · globals read via `globals()['x'].get()`.
