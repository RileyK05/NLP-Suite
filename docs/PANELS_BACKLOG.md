# Panels: state of play and the queue

**Written for:** whoever picks this up next, another model or a person, and
Riley, who hands the work out and has Claude review it afterwards.

Read [`docs/viz-panels.md`](viz-panels.md) first. It is the contract, and it
now covers the three mistakes most likely to be repeated: naming the wrong
tool, assuming one table per run, and inventing test fixtures.

**Nothing is committed.** Everything below is in the working tree of
`NLP-Suite` (git, branch `main`).

---

## 1. Verification status — read this before trusting anything below

| Gate | Result this session |
|---|---|
| `ruff check` / `ruff format` | clean on every file touched |
| `mypy core tools app desktop_backend` | clean (last full run: 270 files) |
| panel + LDA test files | all pass (see section 2) |
| `vitest run` | **505 passed**, 1 skipped |
| `tsc --noEmit`, `vite build` | clean / builds |
| `prettier --check src` | only the 8 **pre-existing** offenders (App.tsx, contract.ts, ExplicitRun.test, Learn.test, LiveBench, ToolCard.test, ToolCard, toolGuides.json). `contract.ts` is generated: regenerate it, do not hand-format it. |
| **Full Python suite** | **NOT COMPLETED this session.** Two runs were cut off. Both appeared to stall in `tests/test_desktop.py`. The second got 43 tests into that file, so it was making progress (slow tests, most likely `test_cancel_running_worker_promptly` and the runner tests), not hung. It was then killed by the session ending. **First task: run it to completion.** Use `--basetemp=C:/nlptmp`, delete that directory afterwards, and write the output straight to a file rather than through `tail`, so a slow test is visible. |
| Real-server end to end | **passed**, see section 3 |

---

## 2. What was done this session

### Bugs found and fixed (each A/B-verified: the new test fails on the old code)

1. **LDA panels were unreachable on the desktop.** They declared
   `tool="topic_model"` (the live bench's name); the tool that writes runs is
   `lda_gensim`, so no run was ever offered them. Fixed. New guard:
   `test_panels.py::TestRegistry::test_every_panel_belongs_to_a_tool_that_exists`.
2. **The desktop took the first CSV of a run.** `lda_gensim` writes
   `topics.csv` first, which no panel reads, so every LDA panel would have
   failed. `desktop_backend/panels.py` now picks the first table carrying
   every column in `requires`, and offers a panel only when that table
   exists. Tests: `test_desktop_panels.py::TestMultiTableRuns`.
3. **The relevance panel was misleading on real data.** Relevance is always
   negative (a sum of logs of probabilities), so the top-ranked word drew the
   *shortest* bar; the saliency "ghost series" was about 100× smaller, on the
   same axis. Redesigned the pyLDAvis way: scores **order** the rows, bars
   are **counts**. The engine (`relevance_terms`) gained two additive columns:
   `Corpus frequency` and `Topic frequency` (φ × token-weighted topic mass).
   Relevance and Saliency themselves are unchanged. Tests are rebuilt on an
   **engine-derived** fixture; the old one used impossible positive values.
4. **Ordering by saliency put the least salient word first.** The engine's
   Saliency is negative, with the most salient words most negative. The panel
   now orders by magnitude, which is correct under both the current and a
   corrected definition (see question 1 in section 5).
5. **A race in `PanelSection`:** a figure drawn for one run could land under
   another run's heading after switching runs. Fixed with a "which run is on
   screen" ref. A stale answer can no longer clear the new run's busy flag
   either (the same shape as the old LiveBench bug).
6. **Reference lines were clipped off the plot** whenever every mark lay
   beyond them, e.g. the p < 0.05 line on a pre-truncated keyness table, or
   PMI's zero line. The scatter domain now includes annotation values. The old
   test only checked the `<line>` existed.
7. **Colour mismatch:** ungrouped panels (the intertopic map) were drawn
   orange in the app and blue in the export. `groupColors` now mirrors
   `panel_plotters._group_color_map` exactly.
8. **Ranked bars had no value axis and no axis title** in the app. The
   title is where the relevance panel says its bars are occurrences. Added.
9. **`tokens_from_frame(nouns_only=True)` returned nothing on this corpus**
   (it checked Universal tags; spaCy's parse here is Penn). It now applies the
   same `NN*` or `NOUN`/`PROPN` rule the rest of the codebase uses.
   Vectorised as well: **10–16 s → 0.25 s**, proved identical (keys, order,
   every token) to the old version on the real corpus before it was swapped in.
   There had been no `nouns_only` tests at all; there are now.
10. Pre-existing mypy errors in `tools/gis_map.py` (earlier session) and the
    collocation agent's false "Log Dice ceiling is never reached" comment
    (real pairs such as *viet nam* sit on it; it is now drawn as a line).

### Features added

- **Read in context** (`desktop/src/PanelPassages.tsx`). Clicking a mark whose
  evidence names a term runs the Interactive page's phrase search and shows
  the passages. `Evidence` gained `phrase` and `lemma`, set by each builder
  from the run's own settings; a windowed collocation gets no phrase rather
  than an undercount. It never parses without being asked, and says the
  loaded documents may differ from the ones the run read.
- **New panels:** `collocation_strength` (agent; PMI/Log Dice/T-score/G2
  against log frequency, with `PANEL_RARE_PAIRS_DOMINATE`, which the agent
  found was silent on real data as specified and fixed with a median-relative
  threshold). Registered.
- **Topic stability** (agent): `core/analysis/topic_stability.py` (refit at
  several seeds, Hungarian-match topics by top-word Jaccard, a topic is
  "stable" if its *worst* seed matches) plus `core/viz/panels_lda_stability.py`.
  50 tests. Real data: only **2 of 6** topics stable at k=6, 5 of 15 at k=15.
  **Not registered yet** (see 4b).
- **Topic flow, half built** (see 4a): segmentation module and ribbon panel
  builder, both tested.

### Real-server end to end (`scratchpad/e2e_panels.py` pattern)

Real `lda_gensim`, `keyness` and `collocations` jobs through the real runner,
on a copy of the workspace. All five registered panels were offered by the
right runs and drawn from the right tables. **Read in context matched the
tables exactly:** "shall" gave 496 occurrences against 496 counted by keyness,
and "united states" gave 636 against 636 counted by collocations.

---

## 3. Panel inventory

| Panel | Tool | Shape | Registered | Notes |
|---|---|---|---|---|
| `keyness_volcano` | keyness | scatter_labelled | yes | reference implementation |
| `lda_relevance` | lda_gensim | ranked_bars | yes | redesigned: counts, ordered by score |
| `lda_intertopic` | lda_gensim | scatter_labelled | yes | |
| `lda_prevalence` | lda_gensim | stream | yes | |
| `collocation_strength` | collocations | scatter_labelled | yes | |
| `lda_stability` | lda_stability | ranked_bars | **no** | tool doesn't exist yet (4b) |
| `lda_flow` | lda_gensim | **ribbon** | **no** | engine + renderer not done (4a) |
| `svo_agency` | clause_svo | ranked_bars | **no** | agent never reported back (4c) |

---

## 4. The queue, in order

### 4a. Finish topic flow (Riley's stated #1). Most of the design is done.

Built and tested: `core/analysis/topic_flow.py` (`plan_segments`: blank-line
→ line → sentence-window segmentation, strict token alignment via
`core.research.phrase._align_tokens`, rule reported per document), and
`core/viz/panels_lda_flow.py` (19 tests; the table contract is in its
docstring).

Left:
1. **`lda.py`**: pull the token filter out of `tokens_from_frame` into a
   shared helper (kept-token Series indexed like the frame) so segments use
   *exactly* the training filter. Add `fit_lda(..., segments=...)`: score each
   segment's bag of words with `model.get_document_topics` while the model is
   in hand, and return `LdaResult.flow`. **Columns must match the panel's
   docstring exactly:** `Document ID, Document, Segment, Segments, Rule,
   Aligned, Tokens, Dominant topic, Contribution, Topic keywords, Start, End`.
   A segment with no tokens gets a blank `Dominant topic`, not the prior.
2. **Executor** (`_adapt_lda_gensim`): call `plan_segments(ctx.table,
   ctx.corpus)`, write `topic_flow.csv`, add it to the registry `outputs`.
3. **`ribbon` renderer, twice:** `panel_plotters.py` (horizontal `go.Bar` with
   `base` = segment start, `x` = width) **and** `panelLayout.ts` +
   `PanelCanvas.tsx` (SVG rects). Add `"ribbon"` to `IMPLEMENTED_SHAPES` in
   the same change as the TS case, or `test_panel_parity.py` fails.
4. Register `LDA_FLOW` in `core/viz/panels.py`, and add an engine-derived test.
5. Later: evidence → paragraph text, using the segment's `Start`/`End` with
   `/live/source`.

### 4b. Register the stability tool (five-file checklist, section 6)

A new `lda_stability` corpus tool: `ToolSpec` (params: topics, seeds, top-n,
stable-at), executor adapter calling `topic_stability(tokens_from_frame(...))`
and writing `summary.csv` + `matches.csv`, `ADAPTER_NEEDS` = `{"table"}`,
`desktop_backend/catalog.py` `CORPUS_TOOLS`, labels, migration guide, ledger
row, scope test. **Add a `Stable at` column to the summary** so the panel can
draw the threshold; it currently, and correctly, refuses to guess it. Then
register `LDA_STABILITY`.

### 4c. SVO agency panel

An agent was briefed to write `core/viz/panels_svo_agency.py` + tests (subject
vs object counts per entity, with a normalised-column evidence design). It
never reported back. **Check whether the files exist**; if they do, review
them against the brief (evidence resolves against `PreparedPanel.data`;
pronoun warning; real-data check) before registering. `clause_svo` writes two
CSVs (`clauses.csv`, `svo.csv`); column-based table selection now handles that.

### 4d. Other wishlist items, not started

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

The recipe work is in `docs/FIGURE_RECIPES.md` (every tool covered; what is
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

The plan and its progress are in `docs/FIGURE_QUALITY_PLAN.md`. Still owed,
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

### 4e. Observed, not investigated

Each desktop job showed "Parsing English documents" for minutes even with the
parse cache copied alongside. The runner may not be reusing the annotation
cache across jobs. Worth measuring.

---

## 5. Questions waiting on Riley

1. **Should the engine's Saliency be corrected?** `relevance_terms` computes
   `φ·(log φ − Σφ log φ)`. That is not Chuang et al.'s saliency
   (`p(w)·Σ_t p(t|w) log(p(t|w)/p(t))`, which is ≥ 0 with larger meaning more
   salient). Here it is always negative and inverted. Recommended: yes. The
   panel is already correct under both definitions. It's a graded HW2
   statistic, though, so it's Riley's call.
2. **Consolidate the noun-tag rule?** `startswith("NN") or in ("NOUN",
   "PROPN")` now exists in seven modules. A small cleanup into one helper,
   left alone because it touches five unrelated files.
3. **Commits.** Suggested separable commits: (a) panel reachability fixes,
   (b) relevance redesign + engine columns, (c) Read in context, (d)
   tokenizer fix, (e) new panels, (f) topic-flow groundwork, (g) figure
   recipes: shapes, families, live tabs, generic guards (2026-09-23).
4. **tf-idf's default `max-df-ratio`.** 1.0 makes the tool's own output a
   list of function words on this corpus (4d-5). 0.5 is informative. It
   changes a graded statistic's default, so it is Riley's call; the panels
   warn either way.
5. **The annual-mean sentiment lines are out of the registry** (code and
   tests kept, `ANNUAL_SENTIMENT_PANELS`). They were an average with no spread
   drawn through single speeches; the per-document trends with a rolling
   median replace them. Say if they should come back.
6. **`tests/test_security.py` fails on `scripts/generate_hw1_inaugural_outputs.py`**
   (gitignored, uses `subprocess`). Not part of this work; either add it to
   the security allowlist with its reason or keep homework scripts outside
   `scripts/`.

---

## 6. Things that will bite you

- **A new tool is five-plus edits:** `core/profiler/registry.py`,
  `core/profiler/labels.py`, `docs/MIGRATION.md`,
  `docs/REPLACEMENT_LEDGER.md` (capability IDs must resolve),
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
