# Showcase figures and a figure kit for scripts: the plan

**Written for:** whoever builds the next layer of the suite's figures (a
person or a model), and Riley, who set the goal (2026-09-28): figures as good
as the reference MALLET figure for as many tools as can support one, and a scripts
toolkit good enough to build the rest.

**Reference figures**, drawn outside the app from suite output (and the
MALLET command line), in `C:\Users\moomi\Downloads\LDA`: `mallet_what_it_does.py`
(the model for this plan), `gensim_vs_mallet.py`, `mallet_events.py`,
`lda_lambda_figure.py`, `lda_tiles.py`, `lda_misses_the_break.py`. Read one before
building anything here; the plan is mostly a generalisation of what they do by
hand.

Related: `FIGURE_QUALITY_PLAN.md` (publication renderer, bundles),
`FIGURE_RECIPES.md` (one recipe per tool), `viz-panels.md` (panels),
`PANELS_BACKLOG.md` 4d-12 (the topic-model specifics).

---

## 1. Why that figure works (the grammar to generalise)

1. **The title is the finding or the question**, not the tool's name.
2. **Three zoom levels in one figure:** the whole corpus (A: ninety years),
   a few documents close up (B: six addresses), and what the model itself
   learned (C: alpha, share of words, exclusivity). Most panels today show only
   the first.
3. **The world is on the axis.** Stoplines at events (Pearl Harbor, Korea,
   9/11) turn a description into a test the reader can check.
4. **One colour per thing across every panel**, with one panel serving as the
   key, and direct labels instead of legends. The background topic is gray so
   the eye goes to the rest.
5. **Model internals are shown, not hidden.** A reader who sees alpha learns
   why MALLET mixes and Gensim does not.
6. **Numbers where they are read** (47% in the bar), not only on an axis.
7. **Provenance in the footer:** tool, version, every setting that matters.
8. **Checked by eye after drawing.** The first render had a clipped label, a
   label sitting on a stopline and a title running into an axis. The overlap
   linter (`core/viz/static/lint.py`) catches some of this; a look at the
   image caught the rest.

## 2. Two roads, one kit

- **Showcase figures:** a multi-panel publication figure per tool that can
  support one (section 5), offered on the run page beside the panels and
  written by finished runs.
- **The figure kit in scripts** (`nlp.viz`), for every figure the suite
  does not ship: comparisons across runs, a question specific to one project,
  tools with no showcase.

**Decision to make first:** showcase figures are written *with the public
kit*, not with private helpers. Each one is then also a worked example and a
Scripts template ("open this figure as a script"), and the kit gets exercised
by real figures before anyone else relies on it. Private drawing code for the
showcases would mean building everything twice and leaving scripts with a
weaker kit.

## 3. The kit (`core/viz/kit/`, exposed as `nlp.viz`)

Built on `core/viz/static` (style, labels, lint, text), which already exists.

| Piece | What it does | Generalises |
|---|---|---|
| `viz.canvas(title, subtitle, source=run)` | Figure with the house chrome: left title, gray subtitle, provenance footer taken from a run's envelope; `canvas.rows([1.05, 1])`, `row.cols([1, 1.6])` for layout | hand-set gridspecs and `fig.text` in every reference script |
| `viz.palette(keys, background=None)` | Stable colours per topic, entity type, emotion, speaker; the same keys get the same colours as in the app; `background` goes gray | the colour dicts rebuilt in every script |
| `viz.time_axis(ax, corpus)` | Year ticks, and optionally presidency (or any detail) bands from document details | `terms` shading in `mallet_figure.py` |
| `viz.events(ax, corpus)` | Stoplines for the project's events (section 4), labels kept clear of each other | `stopline()` / `clear_of_lines()` |
| `viz.stream(ax, shares, labels=...)` | Stacked shares over time, direct labels at each band's widest point, moved off stoplines, shortened when crowded | the pinned labels in `mallet_what_it_does.py` |
| `viz.mixture_bars(ax, shares, rows=...)` | One document per bar, split by share, values printed in wide segments | panel B |
| `viz.tiles(ax, fits, reference=...)` | Rows = fits, columns = documents, relabeled by first appearance, with an agreement column (ARI) | `lda_tiles.py`, `mallet_events.py` |
| `viz.dots(ax, frame, by, value)` | Strip + median line + median label that stays off the dots | the scorecard panels |
| `viz.word_grid(ax, words, shade=...)` | Ranked words in a grid, shaded by a measure (corpus frequency, weight) | the lambda figure's middle row |
| `viz.bars(ax, ...)` | Horizontal bars with the value at the end and full-length labels | panel C |
| `viz.lint(fig)` | The static linter, returned as readable problems ("label 'war, free, men' overlaps a line at 1941.5") | checking by eye |
| `nlp.figure(fig)` | Unchanged; already falls back to PNG when the SVG writer is missing | |

Every piece takes plain DataFrames, so it works on a script's own tables and
on a run's.

**Also for scripts:** `nlp.run(...)` results must carry the tables these
figures read (the full document x topic matrix, a model's internals). That is
the engine work listed per tool in section 5, and in 4d-12 for topic models.

## 4. Project events

A project-level list of dated events (name, date, optional note), edited on
the Corpus page beside document details, imported from a CSV, stored with
the project. Every dated figure, panels included, can draw them as
stoplines; `viz.events` reads them in scripts. It is the single largest
lever on figure quality for historical corpora: it turns every time figure
into a check against the world. The same list can later be the default
split for keyness ("before / after") and for a break test.

## 5. Which tools can have a showcase figure

A showcase needs (a) output with structure beyond one ranking (mixtures,
internals, several tables), and (b) a question worth a title. Tools that fail
(a) are well served by their panels; table-first tools (kwic, convert,
search) are left to scripts. Each figure below has three panels in the
section 1 pattern: whole corpus / close up / what the method learned.

| Tool | Title question | A (corpus) | B (close up) | C (the method) | Engine work first |
|---|---|---|---|---|---|
| lda_mallet | What MALLET sees in the corpus | stream of topic shares + events | six documents as mixtures | alpha, share of words, exclusivity | 0.5.1 MALLET fixes; `doc_topics.csv`, alpha, diagnostics table |
| lda_gensim | What the topic model sees | stream (needs mixtures) | documents as mixtures | top words at lambda 0.5 shaded by corpus frequency; lambda sweep as a second figure | `doc_topics.csv`, `p(w|t)`, relevance fix |
| bert_topics | Which topics each era talks about | topics through time | representative sentences per topic | topic sizes including unassigned (-1), honestly | none known |
| word2vec_gensim / _bert | What words mean in this corpus | anchor words on a meaning axis (war-peace) | neighbours of 4-6 anchor words, cosine bars | frequency against neighbour stability; small-vocabulary warning | none |
| word_sense_induction | Does a word have two meanings, and when | each sense's share by decade | example sentences per sense | uses as points (PCA) coloured by sense | sense per use kept in output |
| keyness | What sets group A apart | effect against evidence (volcano) | top words per side with coverage | per-document rate strips for the top words (one speech, or all?) | none |
| lexicon_series / ngram viewer | When did the corpus start saying X | rate per 10k with events and presidencies | document coverage band | related terms by decade heatmap | none |
| sentiment_vader_anew (and neural) | How the tone moved | per-document score + rolling median + events | the most positive / negative sentences, as text | distribution by speaker | none |
| ner | Who and where the corpus talks about | entity-type mix over time | top entities per era | places (map, once the map shape exists) | none |
| doc_similarity | Which documents read alike | date-ordered heatmap, presidency boxes | each document's nearest neighbour: how far back does it reach? | MDS map by decade with stress | none |
| readability family | Did the speeches get simpler | trend + events | length check (joint plot) | decade distributions | merge the existing bundles |
| verb_analysis | How the grammar changed | voice mix over time (the passive's fall) | modality mix | earliest vs latest passive sentences | none |
| clause_svo | Who acts, and on whom | subject share over time for top entities | top triples per era | agency ratio scatter | none |
| shape tools / narrative | What shapes the documents take | cluster arcs as small multiples | which documents fall in which cluster over time | a representative arc per cluster | none |
| doc_embeddings | Where the corpus moved | meaning map by decade | decade centroid drift arrows (4d-7) | nearest documents across eras | none |

Probably not worth a showcase (panels suffice): dispersion, collocations,
ngram_cooccurrence, tf-idf, the table_* statistics. Scripts cover the rest.

## 6. Gates for each showcase figure

- Drawn from real output on two corpora of different kinds (SOTU; a novel
  from the books walk), and the image looked at, not only linted.
- `viz.lint` clean.
- Refuses with a reason when the run cannot support it (undated corpus: no
  stream over years, falls back to document order via the corpus axis).
- Provenance footer complete; colours match the app's panels for the run.
- Exposed as a Scripts template that reproduces it.

## 7. Order of work

1. Prerequisites: the 0.5.1 bugs (frozen SVG backend, MALLET, relevance).
2. The kit's first slice: `canvas`, `palette`, `time_axis`, `events` (reading
   a list passed in), `stream`, `mixture_bars`, `bars`, `lint`.
3. First showcase with it: `lda_mallet` (the reference figure already
   exists, so the port is a straight test of the kit). Then `lda_gensim`
   once it writes mixtures.
4. Project events (section 4) on the Corpus page.
5. `tiles`, `dots`, `word_grid`; the topic comparison and break-test tools
   (4d-12).
6. The rest of section 5, in the table's order, one per pass, each passing
   section 6.

---

## 8. What is built (2026-09-28)

The kit, project events, and the first eleven showcases are in. This is the
record of what landed, so the remaining work in section 7 reads as a gap
rather than a status guess.

**The kit** — `core/viz/kit/`, exposed to scripts as `nlp.viz`
(`core/script/viz.py`), tested in `tests/test_figure_kit.py`.

| Piece | Status |
|---|---|
| `canvas(title, subtitle, source/footer)` with `.rows()`/`.cols()`/`.done()` | built |
| `palette(keys, background=...)` | built |
| `time_axis(ax, documents, band=...)` (a detail as shaded bands) | built |
| `events(ax, [(year, label)])` | built |
| `stream(ax, shares, labels=, background=, event_at=)` | built |
| `mixture_bars(ax, shares)` | built |
| `bars(ax, values, labels=)` | built |
| `dots(ax, frame, by=, value=)` | built (second slice) |
| `word_grid(ax, words, shade=)` | built (second slice) |
| `tiles(ax, fits, references=, stoplines=)` | built (second slice) |
| `lint(fig)` | built |

**Project events** (section 4) — stored in the project's `FieldSettings`
(`desktop_backend/fields.py`, `ProjectEvent` + `project_events`), edited on
the Corpus page (`desktop/src/CorpusDetails.tsx`, `EventsEditor`), exposed at
the existing `/settings` endpoint, frozen into each run's request
(`desktop_backend/runner.py`) and drawn as stoplines by every dated showcase.

**Showcases** — `core/viz/showcase.py`, published into each run's `figures/`
by `core/viz/run_figures.py` (sealed, hashed, listed in the envelope, with an
index entry), and refuse with a reason when a run cannot support one.

Built, in the plan's order: `lda_gensim`, `lda_mallet`, `readability`,
`sentiment_vader_anew`, `verb_analysis`, `ner`, `doc_similarity`, `keyness`,
`lexicon_series`, `bert_topics`, `clause_svo`.

Still to build from section 5: `word2vec_gensim`/`word2vec_bert`,
`word_sense_induction`, `shape tools`/`narrative`, `doc_embeddings`. The
`lda_gensim` lambda-sweep figure (a second figure) is also still open.

**One engine change the showcases needed:** `lda_gensim` now writes
`doc_topics.csv`, the full document × topic share matrix (the dominant-topic
table alone cannot feed a stream or a mixture bar). Added to `LdaResult`,
its adapter, the CLI, and the registry; the `lda_gensim` CLI's `args.lam`
crash (the OPTIMIZATION_REVIEW_LEDGER's `T-01`) is fixed in the same pass.

**What section 6 still needs from a person:** the "looked at, not only linted"
gate. The linter is clean on the real 87-speech corpus for every showcase,
and the render tests measure text overlap, but a human eye on the PNGs is the
last gate and has not been recorded.

