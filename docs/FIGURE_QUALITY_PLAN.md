# Figure quality and publication figures: the plan

Status: in progress (started 2026-09-23). Supersedes nothing; extends
`docs/FIGURE_RECIPES.md` (which figure each tool gets) with *how well* each
figure is drawn, and adds a second renderer for finished figures.

Two jobs:

1. **Every figure reads correctly.** Not the eleven problems found so far.
   The *classes* they belong to, fixed in the renderers and enforced by tests,
   so the twelfth is caught before a person sees it.
2. **Publication figures from matplotlib + seaborn.** A static renderer for
   final output — the figure you put in a paper — that shows more than the
   interactive SVG can: confidence bands, violins, clustered heatmaps,
   marginal distributions, collision-free labels, and full-length text.

The interactive layer stays. It is for exploring (point, click, filter the
table). The static layer is for reading and publishing. Both take the same
`PreparedPanel`, so what a figure *claims* is decided once, engine-side.

---

## 1. What is actually wrong (root causes, not symptoms)

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

Two reported items are **not renderer bugs** and are handled as engine or
default questions (section 7): LDA's default fit ("nation", "world", "year"
in half the topics) and Word2Vec's neighbours for "war" (ii, cold,
liquidation), which is the model on 87 speeches, not the chart.

One reported item was **expected behaviour**: gaps in the immigration
timeline are zero-match years. What *is* wrong there is C2 (the top point is
clipped) and that a zero-inflated series reads as a seismograph (C7-adjacent;
section 3.7).

---

## 2. Targets: what "reads correctly" means, as checkable rules

These become tests (section 5). A figure passes when all hold, on fixtures
**and** on the real 87-speech corpus.

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

---

## 3. Part A — fixing the interactive figures

Each item: the change, where, and its guard test.

### 3.1 Ticks (C1 → R1)
* `desktop/src/panelTicks.ts` (new, pure): `niceTicks` snaps values within
  1e-9·step of 0 to 0 and returns positive zero; `formatTicks(ticks)` formats a
  whole axis at the decimals its *step* needs (step 0.25 → 2, 0.5 → 1, 5 → 0).
  Year detection stays.
* Every figure in `PanelCanvas.tsx` uses it (scatter, line, bars, positions,
  ribbon, distribution, small multiples, heatmap scale).
* `core/viz/panel_plotters.py`: `tickformat` per axis from the same rule, and
  `_number` never prints "-0".
* Tests: `panelTicks.test.ts` (−0, mixed decimals, 1e-17 residue);
  Python mirror in `test_panel_plotters.py`.

### 3.2 Domains (C2 → R2)
* `linearScale` pads by max(5% of span, the largest mark radius in data
  units). Line series pad y at the top (zero-based bottom kept when all y ≥ 0)
  and pad x by half a year.
* Clip rectangles are the plot area plus the largest marker radius.
* Test: a layout lint in `panelLayout.test.ts` that places every shape's
  marks and asserts each circle lies within the plot area.

### 3.3 Label budgets (C3 → R3)
* `measureLabel(text, px)` (character-width estimate, the same table in TS and
  Python) sizes the left margin of ranked bars, heatmaps, distributions,
  positions and ribbons to the longest shown label, capped at 38% of width;
  labels past the cap wrap to two lines, then cut with "…".
* Label cap 18 → 40 characters. The static renderer never cuts (it sizes the
  figure instead).

### 3.4 Label placement (C4 → R4)
* `placeLabels(points, labels, box)` — greedy placement over eight candidate
  positions around each point, in priority order (largest / most important
  first), skipping any candidate that intersects an already placed label or
  leaves the plot. Unplaceable labels are dropped. Deterministic.
* Identical algorithm in `core/viz/static/labels.py` (Python) for the
  matplotlib renderer, where it is checked against *measured* text extents.
* Test: no overlaps on the real co-occurrence association and keyness volcano.

### 3.5 Text that follows parameters (C5 → R5)
* The UI's summary line shows the prepared panel's own subtitle/caption when a
  figure is drawn; `PanelDefinition.summary` is the pre-draw description only.
* Every `summary` rewritten not to name a default choice value.
* **Generic test** (`tests/test_figure_text.py`): for every panel with a
  choice parameter, build it on its fixture with each choice; the title,
  subtitle, axis labels and caption must not mention the default choice's
  value when another is picked, and `summary` must not mention any choice
  value. This is the test that catches the next MTLD.

### 3.6 Density (C6 → R6)
* t-SNE / embedding map: default to the 300 most frequent words (parameter),
  the rest as a faint density layer; labels by placement (3.4), 25 max.
* Heatmaps: x labels rotated and thinned to fit; documents use short dated
  labels ("1934 Roosevelt"); static renderer uses a clustered heatmap with a
  figure sized to the matrix.
* Topic flow: default 20 speeches, short labels, and an "every k-th speech /
  a decade" selector.
* Generic lint: any category axis with more labels than fit is thinned, never
  overprinted.

### 3.7 Summary series (C7 → R7)
* A series joined next to points-only series is drawn as a line with no
  markers (hit targets stay, invisible); raw points stay small dots.
* The rolling median uses shrinking windows at the ends
  (`min_periods = ceil(w/2)`), so it spans the corpus; the note says so.
* Per-document trend lines break only at gaps wider than 3× the median
  spacing between documents, not at 1 year.
* Sparse annual series (ngram viewer): zero years drawn as hollow baseline
  ticks, non-zero years as stems plus points, and an optional rolling line. A
  "share of documents" mode is already there; the caption points to it when
  most years are zero.

### 3.8 Informative marks (C8 → R8)
* VADER (and every "most extreme rows" panel): new `direction` parameter —
  `both` (default: top N/2 positive and top N/2 negative, as a diverging
  chart), `positive`, `negative`. Labels become "1934 Roosevelt — “We have
  seen the …”" from the sentence text, cut at 40 characters, full text in the
  evidence.
* Generic lint: a mark label matching `^(doc(ument)?|sentence|row|id)\s*\d+$`
  or `^\d+\s*·\s*sentence` fails the test.

### 3.9 The audit loop (finding the unreported ones)
The reported items came from 20 minutes of looking. The rest are found by
machine, not by luck:

* `scripts/audit_figures.py` — for every registered panel × every choice of
  every choice parameter, on the real run outputs: build, run the lint
  (R1–R8), render the static figure, measure text overlap and clipping with
  matplotlib's real text extents, and write an HTML contact sheet (one
  thumbnail per figure, lint failures listed beside it).
* Every lint failure is fixed or recorded with a reason in
  `docs/FIGURE_RECIPES.md`. The contact sheet is inspected by eye as well:
  lint catches geometry, not "this chart tells you nothing".

---

## 4. Part B — publication figures (matplotlib + seaborn)

### 4.1 Where it lives
```
core/viz/static/
  __init__.py       render_static(prepared, fmt="png"|"svg"|"pdf", dpi) -> Result[bytes]
  style.py          house style: rcParams, Okabe–Ito palette (same as the app), fonts, sizes
  labels.py         collision-free label placement against measured extents
  text.py           wrapping and short labels
  lint.py           post-draw checks: text overlap, text outside the figure, clipped marks
  shapes/           one module per shape (10), each draw(ax/fig, prepared)
  enrich.py         the extra statistical layers (below), chosen per panel
```
Pure rendering: in → `PreparedPanel`, out → bytes. No file writes (R3 custody
rule: `OutputWriter.write_bytes` is the only writer). Agg backend only, set
before import, so it works headless and in the frozen engine.

### 4.2 What each shape becomes

| Shape | Interactive (kept) | Publication figure |
|-------|--------------------|--------------------|
| line_series (trends) | points + rolling median | points, LOWESS trend with a bootstrap 95% band, rolling median dashed, decade shading, marginal distribution strip on the right |
| distribution | box + jittered points | violin + box + strip (seaborn), n per group, medians labelled, groups ordered by time or median |
| heatmap (similarity / doc×doc) | cells | seaborn `clustermap` (Ward on 1 − similarity) with dendrograms, era colour bar along the side, figure sized to the matrix |
| heatmap (residuals, post-hoc, term×doc) | cells | annotated heatmap (values in cells when ≤ 20×20), diverging scale centred at 0, significance markers |
| scatter_labelled | points + labels | points sized by evidence, collision-free labels, optional 2D density contour, reference lines with their notes, log axes with raw-value ticks |
| ranked_bars | bars | bars with full labels (wrapped), value at the bar end, coverage ("in 34/87 docs") as a second column, diverging colour for signed values |
| small_multiples | facets | seaborn `FacetGrid`/`relplot`, shared x, per-facet y, trend per facet |
| network | nodes + edges | networkx drawing on the builder's positions, edge width/alpha by weight, community colours, labels placed without collision |
| ribbon | bands | `broken_barh` per document with topic legend, short labels |
| stream | stack | `stackplot` with direct labels at each band's widest point |
| positions | ticks | `eventplot` per document, density strip on top |

### 4.3 Extra figures only static can do (per tool, "figure bundles")
* Document measures (readability, lexical diversity, text statistics,
  sentiment): **pair plot** of all measures with a correlation matrix
  (Spearman), and a **joint plot** measure × length with marginals (the
  length check done properly).
* Topic models: **ridgeline** (KDE per topic over years); topic × decade
  heatmap.
* Keyness / tf-idf: **dot plot** with CI-like effect bars; term × document
  clustermap.
* Doc similarity: clustermap (above) and an MDS map of documents coloured by
  decade.

These are registered as static-only panels (`static_only=True` in the
registry), so they show up in the same "figures" list with a badge.

### 4.4 How a reader gets them
* **Every figure in the app gets a "Publication figure" view** (a toggle next
  to the interactive one): the same panel with the same parameters, rendered
  by `render_static`, shown as an image, with PNG / SVG / PDF download.
  Endpoint: `POST /api/projects/{pid}/jobs/{jid}/panels/static` and
  `/live/panel/static` → image bytes.
* **Finished runs write their figures.** Before `finalize`, a run renders its
  tool's default figures to `figures/<panel>.png` and `.svg` as artifacts
  (kind "figure"), each with a caption file. A figure that fails is a
  warning, never a failed run. Setting: on by default, off in settings.
* **The existing plotly export path** (`panel_image_bytes`) falls back to
  `render_static` when kaleido is missing, which it is on this machine.

### 4.5 Dependencies
* `seaborn>=0.13` and `matplotlib>=3.8` as a new `figures` extra, included in
  `all`; pinned in `desktop/requirements-runtime-py312.txt` for the frozen
  engine. networkx is already present. No adjustText: placement is ours (4.1).
* Imports are lazy (inside `render_static`), so the live bench's cold start is
  unchanged.

---

## 5. Part C — interactivity (the "a bit poor" part)

After A and B. In order of how much each helps reading:

1. **Tooltip at the pointer**, not a status line above the chart.
2. **Nearest-mark hover** (Voronoi-style) on scatter and line figures, so
   small points do not need pixel aiming; a crosshair and value readout on
   line series.
3. **Legend toggles**: click a group to hide/show it; double-click to solo.
4. **Zoom and pan** on scatter, line and network (drag a box to zoom,
   double-click to reset), with ticks recomputed.
5. **Find a label**: a search box that highlights matching marks.
6. **Copy / download** the current interactive view as SVG or PNG (already
   exists for charts; extend to panels).

---

## 6. Tests and gates

* `tests/test_figure_text.py` — R5 across every panel and choice (generic).
* `tests/test_figure_lint.py` — R1, R3, R6, R8 on every panel's fixture
  (generic, over the registry: a new panel is covered without writing a test).
* `tests/test_static_render.py` — every shape renders to PNG/SVG/PDF;
  deterministic bytes for SVG; post-draw lint (no overlapping text, nothing
  outside the figure) on every panel fixture.
* `desktop/src/panelTicks.test.ts`, `panelLayout.test.ts` additions — R1, R2,
  R4 on the TS side.
* `scripts/audit_figures.py` on real data — run before calling any phase
  done; contact sheet inspected.
* Existing gates stay: ruff, mypy, full pytest, vitest, tsc, vite build.

---

## 7. Decisions and open questions

* **LDA default fit** — "nation / world / year" in half the topics. Candidate:
  a corpus-level max document frequency (drop words in > 50% of documents) as
  the default, like tf-idf's. Changes a model default; flagged, not done.
* **Word2Vec neighbours** — model quality on 87 documents. The panel notes say
  so; a "min count" and "epochs" hint is added to its controls' help.
* **Run-time figures cost** — 3–6 figures per run, ~0.3–1 s each. On by
  default; one setting to turn off.
* **Plotly export** — stays, becomes secondary to the static renderer.

---

## 8. Order of work

1. Install seaborn; add the extra and the pin.
2. Part A 3.1–3.2 (ticks, domains): smallest change, removes the most-seen
   defects.
3. Generic tests for R5 and R8, then fix what they find (3.5, 3.8).
4. Part B core: `core/viz/static/` with all ten shapes and the post-draw lint.
5. Audit loop (3.9) over the real corpus with the static lint; fix per panel.
6. Part A 3.3, 3.4, 3.6, 3.7 (labels, placement, density, summaries).
7. Part B wiring: endpoints, "Publication figure" toggle, run-time figures,
   kaleido fallback.
8. Part B 4.3 bundles.
9. Part C interactivity.
10. Full gates, real-data end-to-end through the running server, screenshots.

Progress is recorded at the bottom of this file as each step lands.

## Progress
* 2026-09-23 — plan written.
* 2026-09-23 — **Part A (interactive)**: `desktop/src/panelTicks.ts` (ticks with
  no "-0", one precision per axis, log ticks as quantities, label widths,
  collision-free label placement) used by every shape in `PanelCanvas.tsx`;
  domains padded and the clip widened so no mark is halved; row-label
  margins sized to the labels; rolling medians drawn as lines; zero kept on
  an axis only when the data comes near it; heatmap value labels; remainder
  groups ("(smaller clusters)") grey in all three renderers.
* 2026-09-23 — **Engine fixes the lint found on the real corpus**: 39
  summaries naming the default measure; 17 figures with rows thinner than
  a line (central `fit_to_rows`); rolling median dropping the last four
  speeches (now edge-anchored windows); VADER extremes showing one tail and
  "58 · sentence 7" labels (`direction`, "1934 Roosevelt — “…”" snippets);
  coreference rows labelled by id (a name-keyed lookup by id); t-SNE with
  7,478 points (now the 400 most frequent content words, `Count` added to
  tsne.csv); word networks coloured by detected community; keyness labels
  on content words; LDA topics named by their words; speeches by short
  label; the wordlist drawn rather than refused; one-per-year groups warned;
  `best_table` (the prevalence figure could be fed the paragraph table).
* 2026-09-23 — **Part B (publication)**: `core/viz/static/` (style, text,
  labels, lint, stats, shapes; all ten shapes; clustered document matrix;
  IQR bands; violins; per-series panels for sparse annual series; measured
  label placement); endpoints `/jobs/{id}/panels/static` and
  `/live/panel/static`; a "Publication figure" view with PNG/SVG/PDF
  download on every figure; finished desktop runs publish `figures/`
  (`core/viz/run_figures.py`, `BatchRequest.figures`, off for CLI batches,
  `NLP_SUITE_RUN_FIGURES=0` to disable); seaborn in the `figures` extra, the
  runtime pins and the frozen build.
* 2026-09-23 — **Gates**: `core/viz/figure_lint.py` + `tests/test_figure_quality.py`
  over every panel and every choice on trimmed real fixtures
  (`tests/fixtures/figures`), plus the drawn-text lint on every publication
  figure; `scripts/audit_figures.py` writes a contact sheet.
* 2026-09-23 — **Part C (interactivity)**: tooltip at the pointer,
  nearest-point hover and click on dot shapes, legend entries that hide and
  show groups (colours stable).
* 2026-09-23 — **Bundles (4.3)**: `core/viz/static/bundles.py` -- a Spearman
  correlation matrix for every document-measure tool, each measure against
  length with marginal distributions, topics by decade, and an MDS map of the
  documents with Kruskal stress; written into a run's `figures/`, listed and
  drawn by `/jobs/{id}/bundles`, shown in the app's "Publication-only
  figures" gallery.
* Not done yet: zoom/pan, label search, bundles on the live bench, and the
  further bundles in `docs/PANELS_BACKLOG.md` §4d-6.
