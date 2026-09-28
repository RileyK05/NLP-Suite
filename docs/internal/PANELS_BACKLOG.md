# Backlog: panels, figures and product ideas

**Written for:** whoever picks this up next, another model or a person, and
Riley, who hands the work out and has Claude review it afterwards.

This is the one place for work that has been noticed but not scheduled.
Release plans (currently `docs/internal/PLAN_0.5.0.md`) pull items from here. **Add new
ideas under section 4d** as a dated `4d-N` heading, not only in chat.

Before building a panel, read [`docs/viz-panels.md`](../viz-panels.md). It is the
contract, and it covers the three mistakes most likely to be repeated: naming
the wrong tool, assuming one table per run, and inventing test fixtures.
Per-tool figure coverage and the figure-quality rules are in
[`docs/internal/FIGURE_RECIPES.md`](FIGURE_RECIPES.md).

Status notes below were written during individual implementation passes.
Check the code and the dated entries before treating an item as open. The
first panels session's log (sections 1–3: verification status, bugs fixed,
panel inventory) and its finished queue items 4a–4c (topic flow, the
`lda_stability` tool and the SVO agency panel, all built and registered) were
removed in the 2026-09-28 docs cleanup; Git history has them.

---

## 4. The queue

### 4d. Other wishlist items, not started

- **Topic flow: evidence → paragraph text**, using the segment's
  `Start`/`End` with `/live/source` (the one item left from the finished
  topic-flow work).
- Sentiment model-agreement panel. **Needs a seam extension:** a panel
  currently draws one run's table, and agreement needs four runs joined.
- Figure bundles: chart + data + methods note + `Provenance.to_dict()` JSON.
- Evidence → passage for `documents`/`sentences` scopes (only `terms` is done).
- Semantic neighbour drift (`small_multiples`): period-sliced Word2Vec with
  Procrustes alignment. The heaviest item.

### 4d-2. Charts that fit the result (Riley's request)

Some charts make no sense for some results, and the suite should know which.
`core/insight/recommend.py` recommends by table *columns*; it should also know
the result's **family**, and each family gets a default, an allowed set and a
refuse-or-warn set. The refusal says *why*, so it teaches as it goes.

| Family | Tools | Default | Refuse or warn |
|---|---|---|---|
| Time series | ngram viewer, culturomics, lexicon series | line per 10k words, smoothing control, a document-coverage band | bar charts of raw counts; pie charts |
| Contrast | keyness, collocations | volcano / strength panels | a bar chart of G2 alone (hides effect size) |
| Topics | LDA, MALLET, BERT topics | the topic panels | line charts across topic numbers (arbitrary labels) |
| Embeddings | word2vec, contextual | neighbour tables, projection maps | axes that imply a meaningful dimension |
| Classifier / transformer output | sentiment, NER, etc. | score distributions, agreement between models | averages without spread |

- **"How to read this chart" guides per family**, shown the first time a
  family is drawn. Builds on the panels' existing notes and `toolGuides.json`.
- **A real culturomics ngram viewer:** rate per million words, a smoothing
  window, document coverage beside frequency, and a warning when one long
  document causes a spike.

### 4d-3. New models

- **Embeddings:** EmbeddingGemma, Qwen3-Embedding, IBM Granite embeddings,
  plugged into the contextual/BERT path. They'd power the semantic-change
  explorer. Needs a license check first (`docs/LICENSE_REVIEW.md`),
  load-once caching (R5), `model_integration` test markers, and CPU timing
  on the 118-document corpus.
- **Decoders (small local LLMs), as suggesters only:** topic-label proposals,
  passage summaries, a draft methods paragraph. Stored as a separate
  machine-suggestion layer the researcher accepts or rejects, never as
  findings, with the model, its hash, seed and temperature in provenance.

### 4d-4. Noticed this session

- **Named cohorts instead of regex groups.** Keyness groups are currently
  regexes like `^19[3-6]`. The SOTU project turned out to hold 118
  documents including inaugurals, which a cohort or genre tag would have
  made obvious.
- **Multi-run panels:** one seam extension (a panel over several runs'
  tables) unlocks both sentiment model agreement and MALLET-vs-Gensim
  alignment.
- **Suggested stopwords per corpus.** Topics here are dominated by "year,
  people, make, congress". A "suggest corpus stopwords" step, reviewed by the
  researcher, would clean up every topic panel.

### 4d-5. Found while fitting a figure to every tool (2026-09-23)

The recipe work is in `docs/internal/FIGURE_RECIPES.md` (every tool covered; what is
still owed is listed there). Drawing every figure from the real 87-speech
outputs turned up problems in the *tables*, which no figure can fix:

- **tf-idf's default is uninformative.** At `max-df-ratio 1.0` every speech's
  top 20 is "the, of, and, to" (smoothed IDF gives a word in every speech an
  IDF of 1, and raw frequency wins). At `0.5`, 1934 becomes "industrial,
  restoration, recovery" and 2024 "gaza, roe, predecessor". The panels warn
  and say how to fix the run; the default itself is a question for Riley (5.4).
- **The word list's defaults are all function words** (top-n 20, category
  all): its panel refuses and points at the `category` parameter.
- **The story-shape matrix is not standardised**, so SVD/NMF components are
  mostly sentence length (component 1's token loadings are ~100x its ratio
  loadings). Standardising the features would make the components about shape.
- **Transcript annotations reach the NER tagger**: "Boo", "Speaker" (from
  "Mr. Speaker"), "Pell", "Chamber" come out as PERSON. Strip "(Applause.)"
  style annotations at intake, or offer a reviewed stop-entity list.
- **The date annotator reads bare numbers as years** ("1500", "1140"); the
  timeline's labels now skip one-off bare years, but the rows are still there.
- **Quote attribution takes the verb's subject**, so "he", "I", "Shall",
  "have" are "speakers". The panel hides function words; a fix belongs in the
  attributor.
- **Tables that name documents by id only**: svo_compare and the story-shape
  tables did; the executor now appends names (additive columns). narrative's
  and shapes' arc tables still carry no document name, so their arcs are
  labelled "1934-01-03 (doc 1)".
- **Tables without per-document token counts** (search hits, quote summary,
  NER entity timeline) cannot be normalised per 10k words; their figures say
  so. Adding a `Tokens` column in the executor would let them be rates.
- **Findings the new figures surfaced** (for Riley's interest, not bugs):
  Flesch ease rises ~51 to ~65 over the corpus; the passive falls 15.6% to
  5.5%; VADER tone peaks 1950s-80s then falls to 0.14 by 2019; PERSON
  entities are over-represented from the 1990s on (chi-square residual +9 in
  the 2020s) and ORG before; BERT topics are eras (topic 1: 1990-2024).

### 4d-6. Figure quality and publication figures (2026-09-23)

The plan and its progress are in `docs/internal/FIGURE_RECIPES.md` (Part 2). Still owed,
and new ideas that came out of the work:

- **The word list now draws its function words, no longer refusing them**,
  with the warning pointing at `category` (it was an error box on the
  default run; this supersedes the line above).
- **Zoom and pan** on scatter, line and network figures (drag a box, double
  click to reset), and **find a label** (highlight marks matching a typed
  word). The plan's Part C items 4-5.
- **Bundles on the live bench**: publication-only figures are offered for
  finished runs only; a live answer could offer them too (`/live/bundles`).
- **More bundles**: a ridgeline of topic prevalence over years (KDE per topic);
  a keyness dot plot with the effect and its evidence; a term x document
  clustermap for tf-idf; a pair plot (scatter matrix) beside the correlation
  matrix for 3-5 measures.
- **The LDA default fit puts "nation", "world", "year" in half the topics.**
  A corpus-level max document frequency (drop words in > 50% of documents)
  as the default, like tf-idf's question (5.4). A model default: flagged, not
  changed.
- **Word2Vec on 87 speeches gives odd neighbours** ("war": ii, cold,
  liquidation). The panel could say when the vocabulary is small for the
  number of dimensions, and suggest `min-count` / `epochs`.
- **tsne.csv gained a `Count` column** (additive), so the map can show the
  most frequent words. Runs from before it keep the old spread-labels view.
- **Run figures cost** about a second per figure at 200 dpi. A setting in the
  app (today only `NLP_SUITE_RUN_FIGURES=0`) and a choice of formats would be
  friendlier.
- **Kruskal stress on the similarity map of the 87 speeches** tells whether the
  flat map is faithful; it would be worth showing the same number beside the
  interactive network of neighbours.

### 4d-7. Models and embeddings (2026-09-24, release 0.4.0)

- **A last-layer-only graph for token models.** BERT's ONNX graph returns all
  13 hidden states so Word2Vec via BERT can read any layer; every other caller
  reads only the last, and pays for stacking the rest. Export a second graph
  (or a second output) and pick it when `layer == -1`. Measure first: a clean,
  interleaved benchmark per precision on an idle machine is owed
  (identified during the 0.4.0 implementation).
- **Word senses beyond two.** `wsi_senses` asks "one meaning or two?". A
  silhouette search over k = 2..4 would find "bank" (money, river, and the
  verb "to bank on"). Needs the same guard against small groups.
- **A word-senses figure.** Done (Words used in two senses; The two senses of
  a word). Still open: per lemma, its uses as points (PCA of the vectors)
  coloured by sense, and each sense's share by decade ("does the second sense
  belong to particular years?", the question the guide asks).
- **Embeddings over time.** `doc_embeddings` knows each document's vector;
  "which way did the corpus move" is the drift of the decade centroids, drawn
  as arrows on the map.
- **Search on the Interactive page.** Semantic search is a tool parameter;
  on the bench it could be a text box that re-ranks without re-embedding (the
  sentence vectors are the expensive part and do not change with the query).
- **Cache sentence vectors.** Like the parse cache: keyed by corpus
  fingerprint + model id, so a second embeddings run, a search, and bert_topics
  on the same corpus reuse them.
- **3-class sentiment.** SST-2 has no neutral. A 3-class model (a RoBERTa
  sentiment model) needs a licence check before it can ship.
- **Glance on the Overview page.** The glance lives on Corpus; its first two
  sentences would make a good Overview card once a glance exists.

### 4d-8. What the embeddings mean (2026-09-24)

Built: meaning axes, a two-axis map, meaning groups, a word's neighbourhood
(both Word2Vec tools); meaning over time and the most-changed words (BERT,
dated corpora); the two word-senses figures. Next:

- **Two models side by side.** The same anchor words' neighbours from Gensim
  and BERT (or Granite) in two columns, with the overlap marked. Needs a figure
  over two runs; today every panel reads one run. Never overlay coordinates of
  two independently trained spaces: their axes do not line up.
- **Axis presets.** A short list of tested pole sets (war/peace,
  poor/rich, past/future, public/private, men/women) offered as choices, so
  the first axis a reader sees is a good one. Each needs checking on more than
  one corpus before it ships.
- **Axes with more pole words.** SemAxis expands a one-word pole with its
  nearest neighbours; an "expand ends" switch would make single-word axes less
  noisy.
- **Change for Gensim.** Per-period Gensim spaces need aligning first
  (orthogonal Procrustes over shared frequent words, Hamilton et al. 2016).
  Until then change is BERT-only, which reads every period in one space.
- **A noise floor for change.** A word with few uses per period looks changed
  by chance. Split each period's uses in half and report the change between
  halves as the floor under which a change is not shown as one.
- **Groups on the run page, not only live.** The run page draws the first
  figure only; when it refuses (too few words for eight groups) it should fall
  back to the next, as the live bench now does.

### 4d-9. Scripts: notebooks over the suite (2026-09-25, release 0.5.0)

Built: the Scripts page (WORKSHOP group), the `nlpsuite` library, one kernel
per open notebook, Run and save, export, the AI chatbot guide. Next:

- **Kernel-driven completion.** Completion is static today (library names
  after `nlp.`, column names inside quotes). A kernel `complete` op would
  offer the notebook's own variables and a DataFrame's real columns.
- **Markdown colouring in text cells.** Dropped because
  `@codemirror/lang-markdown` brings the HTML, CSS and JavaScript parsers and
  doubled the app bundle. A Markdown-only grammar (`@lezer/markdown` without
  the HTML embedding) would bring it back for a few KB.
- **The Scripts chunk is 442 KB** (150 KB gzipped): CodeMirror core and
  Python, loaded only when the page opens. The plan estimated 150 KB.
  Measure what the autocomplete and search packages cost before adding more.
- **Interrupting a cell without losing variables.** Stop ends the kernel,
  because Windows has no way to interrupt Python code from outside. A cell
  that checks a flag between tool calls (`nlp.run` loops) could stop cleanly;
  pure Python loops still cannot.
- **More templates**: "Chapter arc" and "Who is named, when" (plan 4.7) wait
  for document details and books (sections 1 and 2); "Find passages by
  meaning" waits for the vector cache (plan 5.4) so `nlp.similar` is quick.
- **The guide, tested on real chatbots** (plan 4.10): paste it into two
  chatbots with five requests each, record how many answers ran unchanged,
  and fix the guide where they went wrong. Needs a person at a browser.
- **An API key for a built-in assistant, or a local code model** -- not
  doing: Riley decided (2026-09-25) that people use the chatbot they already
  have; it keeps the app offline, with no key storage and no model to ship.
  Revisit only if users ask.

### 4d-10. Compare and document details (2026-09-26, release 0.5.0)

Built: document details (from file names, a CSV, a split, or typed), the
`contrast` tool, and the Compare page (WORKSHOP group). Next:

- **Carried-over passages and the meaning map.** Both need the sentence-vector
  cache (plan 5.4) and are shown as unavailable until it exists. Carryover
  with "Read in context" of both speeches is plan 3.11's second done-when item.
- **A text parameter that names a value in the data should offer the
  values.** The focus figure's "Word group" is a blank text box; the groups
  are in the table. A `PreparedPanel` could carry each such parameter's
  values, and the desktop could render a list instead of a box. The same
  applies to every panel with a term, entity or group parameter.
- **Per-document parse cache (plan 5.1).** A comparison re-parses the joined
  corpus the first time even when each project's parse is cached, because
  the cache is keyed by the whole corpus. 96 s first run, 24 s after, on
  SOTU (87) against Inaugural (31). The precondition is now verified safe
  (`tests/test_parse_per_document.py`): parsing documents apart gives the
  same tokens as parsing them together, and Sentence ID (restarts per
  document), Record ID (dense across documents) and Document ID are all
  rewritable per document. What is left is the implementation: key the
  cache per document sha, parse only the missing ones, concatenate with
  offset ids, and cap the cache with a Settings "Clear".
- **"William Mckinley".** File-name speakers are title-cased word by word, so
  Mc and Mac names lose their inner capital. The same rule `speaker_of` has
  always used; a small exception list or a typed detail fixes it.
- **Grouped bars used to overlay.** Every ranked-bars figure with two series
  drew them overlaid at half opacity, which blended into a colour neither
  series has. They now share each row one under the other, from one baseline,
  in both the app and the publication figure. Worth a look at every panel
  that uses two series (relevance/saliency, subject/object) on real data.

### 4d-11. The corpus axis and details in figures (2026-09-26, release 0.5.0)

Built: the axis (time, order, none), the 61 figures along it, group-by any
detail, "Ch. 5" ticks in both renderers, lexicon counts per chapter, a
period or a detail, detail filters and an order window in the run dialog.
Next:

- **Runs made before a detail existed cannot be grouped by it.** Tables keep
  the details they had when they ran, so "group by Kind" on an older SOTU
  run is refused with "run readability again". Joining the project's current
  details at draw time would answer at once, but then the figure no longer
  reads only its own run's table; decide which is more honest.
- **A table read back from its CSV forgets which columns are details.** The
  executor marks them in `DataFrame.attrs`, which a CSV does not keep, so a
  detail-taking choice accepts any column name. Writing the detail names
  into the run's envelope would let the panels list exactly the details.
- **Panel tabs still say "over time" on a book.** The prepared title says
  "across the chapters", but the declared titles in the panel list are
  fixed text.
- **Lexicon by period or by detail draws only the heatmap.** A period has a
  middle chapter and could be a point on the line; a detail has no order and
  is right as a heatmap.
- **The script library reads the automatic axis,** not the one chosen on the
  Corpus page.
- **A reload racing a project switch** asked once for the previous project's
  run panels (a 404, nothing shown). Not reproduced with a plain switch.

### 4d-11b. Books walk findings (2026-09-26, release 0.5.0)

Found splitting real novels (Pride and Prejudice 61 chapters, Alice 12) and
walking every formerly figure-less tool over them:

- **"Speeches" wording is a class of about 153 occurrences** in
  `core/viz/panels*.py` -- quote_annotator's "the number of speeches each
  comes from" on a novel, "dated speeches" on chapters, and so on. Fixing it
  needs a corpus-aware noun (the axis noun or a `Work`-aware word), and it
  would change the locked dated digests (`tests/fixtures/axis/
  dated_outputs.json` and the figure snapshots). **Riley's decision**: change
  the wording and re-lock the digests, or keep the wording and live with
  novels being called speeches. The axis nouns ("across the chapters") are
  already right since 1.7; this is the prose around them.
- **Point labels show stored names** ("Alice in Wonderland__010") rather
  than document names in some figures' hover text. The stored name is what
  the corpus file is called; `document_labels` prefers the imported name.
  Worth one pass over every builder's `label=` and `describe=`.
- **`gender_annotator` needs NLTK's `names` corpus downloaded** on the
  machine (setup, not figures); until then it fails on every project. A
  Setup-page check for it would say so in words.

### 4d-12. The reference topic-model figures, natively (2026-09-28)

Riley wants the reference figures made in the app, and figures of that
quality for as many tools as can support one: the general plan (a figure kit
for scripts, showcase figures per tool, project events) is
[`SHOWCASE_FIGURES_PLAN.md`](SHOWCASE_FIGURES_PLAN.md). Reference implementations
(standalone matplotlib over suite output + MALLET CLI) are in
`%USERPROFILE%\Downloads\LDA`: `lda_lambda_figure.py`, `mallet_what_it_does.py`,
`gensim_vs_mallet.py`, `mallet_events.py`, `lda_tiles.py`,
`lda_misses_the_break.py`. What each needs, cheapest first:

- **Prerequisites (0.5.1 bugs):** the frozen engine's missing
  `matplotlib.backends.backend_svg` (run figures and `nlp.figure` both write
  SVG); MALLET's skipped `import-dir`, the 2.0.8 doc-topics parse and the
  `file:/` names; the relevance formula (`lda.py:293` uses `log p(w)` where
  the lift `log p(w|t) - log p(w)` belongs).
- **Tables the engines do not write yet.** `lda_gensim`: the full
  document x topic matrix (`doc_topics.csv`, one share column per topic; today
  only the dominant topic survives), and each word's `p(w|t)` beside
  `Corpus frequency`. `lda_mallet`: the same `doc_topics.csv`, the learned
  alpha per topic from the keys file, and `--diagnostics-file` parsed into
  `topic_diagnostics.csv` (tokens, exclusivity, coherence, rank-1 docs). A
  shared NPMI coherence in `core/analysis` scored on the corpus both engines
  saw (Roder et al. 2015; window 10, singular/plural merged).
- **Single-run bundles (fit the existing seam):** the lambda sweep (bars at
  two lambdas, top words at every lambda shaded by corpus frequency, overlap
  and median word count against lambda) over `lda_gensim`; "what MALLET does"
  (stream of topic shares, six addresses as mixtures, alpha / share of words /
  exclusivity) over `lda_mallet`; the same stream over `lda_gensim` once it
  writes `doc_topics.csv` (the stream shape already exists).
- **A lambda slider on the bench.** Re-ranking by relevance needs no refit,
  so it is live generation in the proper sense: the ranking recomputes, the
  model does not.
- **Comparisons as tools, not multi-run figures.** The scorecard (mixture,
  distinctness, coherence, seed stability, two-topic agreement) and the event
  test (windows x topic counts x seeds scored against a date, with each
  engine's ARI side by side) both need many fits. A `topic_model_compare`
  tool (engines x seeds) and a `topic_break_test` tool (window, event date or
  document-detail cohort, topic counts, seeds, engines) would run the fits
  themselves and write one run's tables, so their figures are ordinary
  single-run bundles and the multi-run seam (4d-4) is not needed.
  `lda_stability` is the precedent. The tile grid is a new shape
  ("tiles": rows = fits, columns = documents, relabeled by first appearance).
- **House style.** Riley likes these figures' look: left-aligned bold title,
  gray subtitle, muted provenance footer, off-white surface, no top/right
  spines, the 8-colour categorical palette with gray for the background
  topic. Most of that is `core/viz/static/style.py` and `add_chrome`; the
  palette is a decision (it is Okabe-Ito today).

### 4d-13. Carried over from retired docs (2026-09-28 docs cleanup)

Open items from the old `FUTURE_IDEAS.md` parking lot and the finished
`TOOL_VISUALIZATIONS_AND_DELETION_PLAN.md` (its culturomics, topic, embedding
and sentiment panels, panel-first views, and Trash/restore/purge for
documents, runs and projects were built on 2026-09-22):

- **Streamed progress for long fits.** A 30 s topic model shows a ticking
  elapsed counter and nothing else. Server-sent events or chunked polling of
  an `analyse/{id}` status endpoint could report stages (vectorize / fit /
  summarize) the way `/live/warm` already does.
- **True interruption of long model fits** needs a cancellation token in the
  adapters. Today cancellation discards a mid-flight answer (R-C8 in
  `ARCHITECTURE.md` section 13).
- **Batch jobs and the loader lock.** Batch jobs run in worker processes
  (`desktop_backend/runner.py`), where a cold import cannot deadlock the UI
  server but can still die silently per job. A worker-side "first import
  failed" diagnostic channel would make that loud.
- **Contract codegen direction.** Pydantic is the source and TS is generated.
  If the desktop ever grows request shapes first, consider a checked-in
  language-neutral schema (JSON Schema) consumed both ways.
- **pyLDAvis as an optional extra.** The Intertopic Distance Map and
  λ-relevance terms are native. If a class ever wants the real pyLDAvis HTML,
  add it behind an optional extra, not as a dependency of `lda_gensim`.
- **Wordcloud options parity.** Users ask about "various wordcloud options"
  (masks, shapes, group colours). Check the legacy GUI's option list against
  what the desktop reaches (the list is not in this repo).
- **Paired comparison tables.** The run-comparison diff shows where two runs
  differ; a paired table could show both full results together.
- **Panels: saved choices and export.** Save a panel's choice and parameters
  with provenance in saved views, reopen and export it; label a change that
  needs recomputation (e.g. retraining an embedding) as a new analysis.
- **Embeddings: source context and coverage.** Selecting a word should reach
  its source passages; report vocabulary coverage, model/seed and projection
  limits beside the neighbours.
- **Richer sentence-level sentiment** views (tone over position and passages,
  not only document scores).
- **Deletion: failure injection.** Inject a filesystem failure during
  Trash/purge and confirm metadata and files recover coherently.
- **A CI check for stray root-level files.**
- **Decided against (keep the reason): a per-tool "keep live" opt-in.**
  Explicit Run is the default and the fix for the wedged-bench class of bug
  (R-C1). A toggle to re-enable debounced follow would reintroduce the shape
  that caused the wedge (an effect that fires work) behind a setting. If it is
  ever wanted, it needs a contract amendment and the same
  single-flight/lost-request handling the explicit path has.

### 4e. Observed, not investigated

Each desktop job showed "Parsing English documents" for minutes even with the
parse cache copied alongside. The runner may not be reusing the annotation
cache across jobs. Worth measuring.

**Answered (2026-09-25, 0.5.0 P1):** it was not. Published runs never read the
cache; only the live bench did. `desktop_backend/project_corpus.py` now gives
runs, the bench and notebook kernels one parse cache: two identical
`corpus_statistics` runs over 20 speeches took 66.7 s then 1.1 s (before:
57.6 s and 49.4 s).

---

## 5. Questions waiting on Riley

1. **Should the engine's Saliency be corrected?** `relevance_terms` computes
   `φ·(log φ − Σφ log φ)`. That is not Chuang et al.'s saliency
   (`p(w)·Σ_t p(t|w) log(p(t|w)/p(t))`, which is ≥ 0 with larger meaning more
   salient). Here it is always negative and inverted. Recommended: yes. The
   panel is already correct under both definitions. It's a statistic people
   report, though, so it's Riley's call.
2. **Consolidate the noun-tag rule?** `startswith("NN") or in ("NOUN",
   "PROPN")` now exists in seven modules. A small cleanup into one helper,
   left alone because it touches five unrelated files.
4. **tf-idf's default `max-df-ratio`.** Answered in 0.5.0: new tf-idf runs
   omit terms appearing in more than half of documents by default (see
   `docs/releases/0.5.0.md`).
5. **The annual-mean sentiment lines are out of the registry** (code and
   tests kept, `ANNUAL_SENTIMENT_PANELS`). They were an average with no spread
   drawn through single speeches; the per-document trends with a rolling
   median replace them. Say if they should come back.
6. **`tests/test_security.py` fails on `scripts/generate_hw1_inaugural_outputs.py`**
   (gitignored, uses `subprocess`). Not part of this work; either add it to
   the security allowlist with its reason or keep personal scripts outside
   `scripts/`.

---

## 6. Things that will bite you

- **A new tool is five-plus edits:** `core/profiler/registry.py`,
  `core/profiler/labels.py`, `docs/MIGRATION.md`,
  `docs/internal/REPLACEMENT_LEDGER.md` (capability IDs must resolve),
  `tests/test_tool_registry.py` scope gate, and for a corpus tool the
  executor `ADAPTERS`/`ADAPTER_NEEDS` and `desktop_backend/catalog.py`.
- **A new panel is two edits:** the builder file and one `PANELS` line.
- **A new shape is Python and TypeScript together** (parity test).
- **Bash heredocs on this machine mangle backslash escapes.** `\n` became a
  real newline and `\b` a real backspace inside written source, three times.
  Use the Edit/Write tools for any text containing escapes, and scan edited
  files for control characters afterwards.
- **Windows MAX_PATH:** pytest `--basetemp` must be short (`C:/nlptmp`), and
  deleted after. Real-server tests go on a *copy* of `out/desktop-workspace`
  at a short path, skipping `runs/`.
- **Kill every server you start.** A leftover one holds
  `out/desktop-workspace/.server.lock` and blocks Riley's preview.
- **Don't edit a module while a test run imports it.** Runner tests spawn
  workers that import from disk.
- **Never create `tests/conftest.py`.** It shadows the ROOT `conftest.py` for
  `from conftest import …` inside tests, and the root one holds shared
  fixtures (`HashEmbeddingBackend`). Put shared test config in the root one.

---

## 7. Review checklist for whoever reviews the handed-off work

- Does every evidence-resolution test assert the resolved row count equals
  `evidence.count`, with a control proving the filter is load-bearing?
- Were fixtures built by running the engine, or at least kept to values the
  engine can produce? An impossible fixture passes every test it was written for.
- Was the panel run on the **real corpus**, with the output checked by eye?
  Three of this session's bugs were invisible to tests and obvious on real data.
- Any statistical claim in a docstring checked against the code? Two false
  ones slipped through agents (the saliency identity; the Log Dice ceiling).
- Does `tool=` name a registered tool? The guard test catches this now.
