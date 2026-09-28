# Figures: which figure each tool gets, and how well it is drawn

**Written for:** whoever builds or reviews the visual layer (a person or a
model), and Riley, who set the goal: every tool's visuals should make sense
for its data, not just the few tools that have purpose-built panels.

Part 1 is the recipe rule and per-tool coverage. Part 2 is figure quality: the
defect classes (C1–C8), the rules that test for them (R1–R8), and the
publication-figure renderer. How to write a panel is in
[viz-panels.md](../viz-panels.md); unscheduled figure ideas are in
[PANELS_BACKLOG.md](PANELS_BACKLOG.md) section 4d.

---

# Part 1: recipes

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
- Figure bundles in the export sense: chart + data + methods note +
  provenance JSON. (The publication-only "bundles" in Part 2 are extra
  figures, not this.)

---

# Part 2: figure quality and publication figures

Started 2026-09-23 with two jobs:

1. **Every figure reads correctly.** Not only the problems reported so far:
   the *classes* they belong to, fixed in the renderers and enforced by tests,
   so the next one is caught before a person sees it.
2. **Publication figures from matplotlib + seaborn.** A static renderer for
   final output — the figure you put in a paper — that shows more than the
   interactive SVG can: confidence bands, violins, clustered heatmaps,
   marginal distributions, collision-free labels, and full-length text.

The interactive layer stays. It is for exploring (point, click, filter the
table). The static layer is for reading and publishing. Both take the same
`PreparedPanel`, so what a figure *claims* is decided once, engine-side.

**Status:** Parts A, B and C below are built (see "What was built"). Still
open: zoom/pan, find a label, bundles on the live bench, and the further
bundles in `PANELS_BACKLOG.md` §4d-6.

## Defect classes (root causes, not symptoms)

Every reported problem traced to one of eight causes. Fix the cause, test the
cause.

| # | Class | Reported symptom(s) | Root cause |
|---|-------|---------------------|------------|
| C1 | **Tick formatting** | "−0" on VADER trend, keyness, co-occurrence; "1.50" next to "2" | `niceTicks` rounds `-0.4*step` to `-0`; `formatNumber(-0)` prints "-0". Each tick is formatted alone, so one axis mixes "1.50" and "2". |
| C2 | **Domains that clip marks** | 2018 per-million point cut at the top; 1934 TTR point cut in the corner | `LineSeriesFigure` pads the y and x scales by 0; a mark at the maximum sits on the clip edge and loses half its circle. |
| C3 | **Fixed label budgets** | ranked bars "58 · sentence …", entity names, n-grams cut at 15 chars; heatmap/ribbon rows cut at 18 | A fixed left margin (84 / 140 px) and a fixed character cap, whatever the labels are. |
| C4 | **Label collisions** | "income · private · sector" over "high · priority"; keyness volcano | Labels are drawn centred above their point with no collision check. |
| C5 | **Static text describing a default** | Lexical diversity: TTR selected, caption says MTLD | `PanelDefinition.summary` is written once from the default measure and shown whatever the reader picked. |
| C6 | **Density with no strategy** | t-SNE: thousands of words in one cloud; 87×87 heatmap x labels; topic flow 30 speeches | Every mark drawn the same way at any count. No thinning, no density layer, no "top N plus the rest as texture". |
| C7 | **Summary series drawn as data** | Rolling median drawn as big dots, broken at every missing year, missing its ends | The trend line uses the data-point marker; the 1-year `line_gap` meant for annual series is applied to a smoothed per-document line; a centred window drops (w−1)/2 points at each end. |
| C8 | **Uninformative marks** | VADER "most extreme sentences": 19 bars at +0.99, labels "sentence 7" | Ranking by |value| on a skewed measure returns one tail; a sentence labelled by its id says nothing. |

Two reported items were **not renderer bugs** and were handled as engine or
default questions (see "Open decisions"): LDA's default fit ("nation",
"world", "year" in half the topics) and Word2Vec's neighbours for "war" (ii,
cold, liquidation), which is the model on 87 speeches, not the chart. One was
**expected behaviour**: gaps in the immigration timeline are zero-match years;
what was wrong there was C2 (the top point clipped) and a zero-inflated series
reading as a seismograph (C7).

## Rules: what "reads correctly" means

A figure passes when all hold, on fixtures **and** on the real 87-speech corpus.

* **R1 Ticks.** No tick label is "-0", "−0" or "-0.0". One axis uses one
  number of decimals. Year axes are whole numbers without grouping.
* **R2 Nothing clipped.** Every mark's full extent (radius included) is inside
  the plot area. The domain includes reference lines.
* **R3 Labels fit.** A category label is shown whole unless it exceeds a stated
  cap (default 40 characters). A cut label keeps its full text in the tooltip
  and in the static figure's alt text. Row-label margins are sized to the
  labels, capped at 38% of the width.
* **R4 No overlaps.** No two drawn text boxes intersect. A label that cannot
  be placed without overlap is dropped (and still readable on hover).
* **R5 Text follows parameters.** Title, subtitle, axis labels, caption and
  the UI's summary line name the *chosen* option, never the default one.
* **R6 Density is handled.** Scatter above 400 points draws a density layer
  and labels at most 25; a category axis above 40 thins its labels; a panel
  above its mark budget says what it left out.
* **R7 Summaries look like summaries.** A trend drawn over raw points is a
  line (no data-sized markers), spans the whole range, and is not broken by
  ordinary gaps between documents.
* **R8 Labels say what the mark is.** No mark label is an id alone ("Doc 14",
  "sentence 7", "58 · sentence …"). Extremes of a signed measure show both
  tails.

## Part A — the interactive figures (one fix per class)

* **Ticks (C1 → R1).** `desktop/src/panelTicks.ts` (pure): `niceTicks` snaps
  values within 1e-9·step of 0 to 0 and returns positive zero;
  `formatTicks(ticks)` formats a whole axis at the decimals its *step* needs
  (step 0.25 → 2, 0.5 → 1, 5 → 0). Every figure in `PanelCanvas.tsx` uses it.
  `core/viz/panel_plotters.py` applies the same rule per axis, and `_number`
  never prints "-0". Tests: `panelTicks.test.ts`; Python mirror in
  `test_panel_plotters.py`.
* **Domains (C2 → R2).** `linearScale` pads by max(5% of span, the largest
  mark radius in data units). Line series pad y at the top (zero-based bottom
  kept when all y ≥ 0) and pad x by half a year. Clip rectangles are the plot
  area plus the largest marker radius. Test: a layout lint in
  `panelLayout.test.ts`.
* **Label budgets (C3 → R3).** `measureLabel(text, px)` (character-width
  estimate, the same table in TS and Python) sizes the left margin of ranked
  bars, heatmaps, distributions, positions and ribbons to the longest shown
  label, capped at 38% of width; labels past the cap wrap to two lines, then
  cut with "…". Label cap 18 → 40 characters. The static renderer never cuts
  (it sizes the figure instead).
* **Label placement (C4 → R4).** `placeLabels(points, labels, box)` — greedy
  placement over eight candidate positions around each point, in priority
  order, skipping any candidate that intersects an already placed label or
  leaves the plot. Unplaceable labels are dropped. Deterministic. The identical
  algorithm in `core/viz/static/labels.py` is checked against *measured* text
  extents.
* **Text that follows parameters (C5 → R5).** The UI's summary line shows the
  prepared panel's own subtitle/caption when a figure is drawn;
  `PanelDefinition.summary` is the pre-draw description only and must not name
  a choice value. `tests/test_figure_text.py` builds every panel with every
  choice and fails if text names the default when another is picked.
* **Density (C6 → R6).** The embedding map defaults to the most frequent words
  with the rest as a faint density layer; heatmap x labels are rotated and
  thinned; documents use short dated labels ("1934 Roosevelt"); topic flow
  defaults to 20 speeches. Any category axis with more labels than fit is
  thinned, never overprinted.
* **Summary series (C7 → R7).** A summary series is a line with no markers;
  the rolling median uses shrinking windows at the ends
  (`min_periods = ceil(w/2)`); per-document trend lines break only at gaps
  wider than 3× the median spacing. Sparse annual series draw zero years as
  hollow baseline ticks and non-zero years as stems plus points.
* **Informative marks (C8 → R8).** "Most extreme rows" panels take a
  `direction` parameter (`both` by default, as a diverging chart). Labels are
  "1934 Roosevelt — “We have seen the …”", full text in the evidence. A mark
  label matching `^(doc(ument)?|sentence|row|id)\s*\d+$` or
  `^\d+\s*·\s*sentence` fails the lint.
* **The audit loop.** `scripts/audit_figures.py` builds every registered
  panel × every choice on the real run outputs, runs the lint (R1–R8), renders
  the static figure, measures text overlap and clipping with matplotlib's real
  text extents, and writes an HTML contact sheet. Lint catches geometry, not
  "this chart tells you nothing", so the sheet is inspected by eye as well.

## Part B — publication figures (matplotlib + seaborn)

```
core/viz/static/
  __init__.py       render_static(prepared, fmt="png"|"svg"|"pdf", dpi) -> Result[bytes]
  style.py          house style: rcParams, Okabe–Ito palette (same as the app), fonts, sizes
  labels.py         collision-free label placement against measured extents
  text.py           wrapping and short labels
  lint.py           post-draw checks: text overlap, text outside the figure, clipped marks
  stats.py          rolling interquartile bands, decade grouping
  shapes.py         one draw function per shape (10)
  bundles.py        publication-only figures (below)
```

Pure rendering: in → `PreparedPanel`, out → bytes. No file writes (R3 custody
rule: `OutputWriter.write_bytes` is the only writer). Agg backend only, set
before import, so it works headless and in the frozen engine. Imports are
lazy, so the live bench's cold start is unchanged. `seaborn` and `matplotlib`
are the `figures` extra, included in `all` and pinned in
`desktop/requirements-runtime-py312.txt`.

| Shape | Interactive | Publication figure |
|-------|-------------|--------------------|
| line_series (trends) | points + rolling median | points, rolling median with an interquartile band over the same windows, decade shading, a distribution strip of every document at the side, extremes named |
| distribution | box + jittered points | violin + box + strip (seaborn), n per group, medians labelled, groups ordered by time or median |
| heatmap (similarity / doc×doc) | cells | rows reordered by clustering, with its tree and a decade side bar (drawn without pyplot, not seaborn `clustermap`), figure sized to the matrix |
| heatmap (residuals, post-hoc, term×doc) | cells | annotated heatmap (values in cells when ≤ 20×20), diverging scale centred at 0, significance markers |
| scatter_labelled | points + labels | points sized by evidence, collision-free labels, optional 2D density contour, reference lines with notes, log axes with raw-value ticks |
| ranked_bars | bars | bars with full labels (wrapped), value at the bar end, coverage ("in 34/87 docs") as a second column, diverging colour for signed values |
| small_multiples | facets | one axis per measure, shared x, per-facet y, trend per facet |
| network | nodes + edges | drawn on the builder's positions, edge width/alpha by weight, community colours, labels placed without collision |
| ribbon | bands | `broken_barh` per document with topic legend, short labels |
| stream | stack | `stackplot` with direct labels at each band's widest point |
| positions | ticks | one row of position ticks per document |

### Figure bundles (publication-only figures)

* Document measures (readability, lexical diversity, text statistics,
  sentiment): a Spearman **correlation matrix** of all measures, and each
  measure against length with marginal distributions.
* Topic models: topics by decade (a **ridgeline** of prevalence is still owed).
* Doc similarity: clustermap (above) and an **MDS map** of documents with
  Kruskal stress.
* Still owed: a keyness dot plot with effect bars; a term × document
  clustermap for tf-idf; a pair plot beside the correlation matrix.

Bundles are written into a run's `figures/`, listed and drawn by
`/jobs/{id}/bundles`, and shown in the app's "Publication-only figures"
gallery.

### How a reader gets them

* **Every figure in the app has a "Publication figure" view** with PNG / SVG /
  PDF download. Endpoints: `POST /api/projects/{pid}/jobs/{jid}/panels/static`
  and `/live/panel/static`.
* **Finished desktop runs write their figures** to `figures/`
  (`core/viz/run_figures.py`, `BatchRequest.figures`; off for CLI batches;
  `NLP_SUITE_RUN_FIGURES=0` disables it). A figure that fails is a warning,
  never a failed run.
* **The plotly export path** (`panel_image_bytes`) falls back to
  `render_static` when kaleido is missing.

## Part C — interactivity

Built: tooltip at the pointer; nearest-point hover and click on dot shapes;
legend entries that hide and show groups (colours stable). Still open:
**zoom and pan** on scatter, line and network (drag a box, double-click to
reset); **find a label** (a search box that highlights matching marks);
copy/download of the current interactive view for panels.

## Tests and gates

* `core/viz/figure_lint.py` + `tests/test_figure_quality.py` — the classes
  over every panel and every choice on trimmed real fixtures
  (`tests/fixtures/figures`), plus the drawn-text lint on every publication
  figure.
* `tests/test_figure_text.py` — R5 across every panel and choice.
* `desktop/src/panelTicks.test.ts`, `panelLayout.test.ts` — R1, R2, R4 on the
  TS side.
* `scripts/audit_figures.py` on real data before calling figure work done;
  contact sheet inspected.

## What was built (2026-09-23)

* **Interactive:** `panelTicks.ts` used by every shape; padded domains and a
  widened clip; row-label margins sized to the labels; rolling medians drawn
  as lines; zero kept on an axis only when the data comes near it; heatmap
  value labels; remainder groups ("(smaller clusters)") grey in all three
  renderers.
* **Engine fixes the lint found on the real corpus:** 39 summaries naming the
  default measure; 17 figures with rows thinner than a line (central
  `fit_to_rows`); rolling median dropping the last four speeches (now
  edge-anchored windows); VADER extremes showing one tail and "58 · sentence
  7" labels; coreference rows labelled by id; t-SNE with 7,478 points (now the
  400 most frequent content words, `Count` added to tsne.csv); word networks
  coloured by detected community; keyness labels on content words; LDA topics
  named by their words; speeches by short label; the wordlist drawn rather
  than refused; one-per-year groups warned; `best_table` (the prevalence
  figure could be fed the paragraph table).
* **Publication:** `core/viz/static/` with all ten shapes, clustered document
  matrix, IQR bands, violins, per-series panels for sparse annual series and
  measured label placement; the "Publication figure" view; run-time
  `figures/`; seaborn in the `figures` extra, the runtime pins and the frozen
  build.

## Open decisions

* **LDA default fit** — "nation / world / year" in half the topics. Candidate:
  a corpus-level max document frequency (drop words in > 50% of documents) as
  the default, like tf-idf's. Changes a model default; flagged, not done.
* **Word2Vec neighbours** — model quality on 87 documents. The panel notes say
  so; a "min count" and "epochs" hint is in its controls' help.
* **Run-time figures cost** — 3–6 figures per run, ~0.3–1 s each. On by
  default; one setting to turn off.
* **Plotly export** — stays, secondary to the static renderer.
