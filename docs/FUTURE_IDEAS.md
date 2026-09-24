# Future ideas, bug fixes, and observations

Collected while implementing the syllabus-alignment / communication-contract work.
Nothing here is committed scope — this is the parking lot for what we notice but
do not do yet. Discuss before promoting anything into a real roadmap item.

## Housekeeping

- ~~**A 0-byte file named `None` at the repo root** was committed in `f6abb8d`~~
  — deleted. Consider a CI check for stray root-level files.
- ~~`docs/CHUNK_LEDGER.md` C32 "topic_model" stale note~~ — truth-up done
  alongside the LDA split.

## Contract / architecture

- ~~**AbortController on the wire**~~ — done as R-C8
  (`DELETE /live/analyse/{request_id}`): a question cancelled before it starts
  is never computed, one cancelled mid-flight has its answer discarded. Kept
  honestly as "discard the answer", not "stop the CPU" — gensim/sklearn fits
  are not interruptible. If someone ever wants real interruption, thread a
  cancel token into the adapters.
- **Streamed progress for long fits.** A 30 s topic model currently shows a
  ticking elapsed counter and nothing else. Server-sent events or chunked
  polling of an `analyse/{id}` status endpoint could report stages
  (vectorize / fit / summarize) the way `/live/warm` already does.
- **Contract codegen direction.** Right now Pydantic is the source and TS is
  generated. If the desktop ever grows request shapes first, consider a
  language-neutral schema (JSON Schema file) checked in and consumed both ways.
- **Batch jobs and the loader lock.** The preload probe now covers live *and*
  batch adapters. Batch jobs run in worker processes (`desktop_backend/runner.py`),
  where a cold import cannot deadlock the UI server but can still die silently
  per-job. A worker-side "first import failed" diagnostic channel would make
  that loud.

## Interactivity

- **Per-tool "keep live" opt-in — assessed, deliberately not built.**
  Explicit Run is the default and the fix for the wedged-bench class of bug
  (R-C1). A per-tool toggle to re-enable debounced follow for cheap analyses
  is tempting, but it would reintroduce exactly the shape that caused the
  wedge (an effect that fires work) behind a setting. It also conflicts with
  R-C1 as written. If it is ever wanted, it needs a contract amendment and
  the same single-flight/lost-request handling the explicit path has — not a
  second, older code path.
- ~~**Run keyboard shortcut.**~~ — done: Ctrl/Cmd+Enter runs the bench and
  submits the run dialog (the dialog marks the keystroke handled so the two
  never both fire).

## Tools / syllabus

- ~~**P4 syllabus gap tools**~~ — landed as one batch (see
  `docs/REPLACEMENT_LEDGER.md` "Syllabus gap tools (P4)" and the manifest):
  annotators (gender/date/quote), gender_guess, verb_analysis, ngram_viewer,
  four neural sentiments, shape_hc/svd/nmf, geocode/gis_map/svo_map, and
  word2vec_bert. The manifest's gap ledger is now empty; every syllabus line
  resolves to a registered tool.
- ~~**Embeddings split half done.**~~ — done: `word2vec_bert` exists and the
  naming-symmetric renames landed (`word_embeddings` → `word2vec_gensim`,
  `contextual` → `word_sense_induction`). Engine module names
  (`core/analysis/word_embeddings.py`, `core/analysis/contextual.py`) were
  left alone — module names are not tool names (see `lda_mallet` ←
  `core/analysis/mallet.py`).
- **pyLDAvis artifact as an optional extra.** Gensim LDA's Intertopic Distance
  Map and λ-relevance terms are implemented natively. If a class ever wants the
  real pyLDAvis interactive HTML, add it behind an optional extra rather than
  as a dependency of `lda_gensim`.
- **Wordcloud options parity.** HW1 rubric asks about "various wordcloud
  options" (masks, shapes, group colors). The legacy raster-PNG port exists
  (`wordcloud_gephi`); worth a pass against the legacy GUI's option list to
  confirm every graded option is reachable from the desktop. (Not done: the
  legacy GUI's option list is not in this repo.)
- ~~**N-gram viewer and undated corpora.**~~ — done: `ngram_viewer`'s failure
  teaches the answer rather than just saying "no dates found"
  (`NG_VIEWER_NO_DATES` names why the co-occurrence viewer can run undated and
  this one cannot).
- ~~**Comparative runs for graded comparisons.**~~ — done: `POST
  /projects/{id}/compare` over `core/compare.compare_runs`, with a "Compare
  with another run…" action on each finished run. Two identical runs agreeing
  is the control test. (Still worth a side-by-side *table* view someday; the
  diff table answers "where do they differ", not "show me both".)

## Visuals

- **Static chart export** (`kaleido`) vs HTML-only is still an open decision in
  `ARCHITECTURE.md` §11. The syllabus wants Excel *or* static *or* dynamic
  Plotly; we do xlsx + interactive HTML. Static image export is the gap.

## Things we fixed that are worth a regression test somewhere

- The wedged-live-bench bug class: any effect that fires work must not own
  component-level "busy" state in a closure. `ExplicitRun.test.tsx` is the
  pattern; reuse it if any other panel grows an effect-driven request.
- Lazy third-party imports inside functions are invisible to "is the module
  imported?" checks. The AST walk in `tests/test_no_lazy_backend_imports.py`
  is the pattern for any future request-path code.
- A `tests/conftest.py` shadows the ROOT `conftest.py` for `from conftest import
  …` inside tests — the root one holds shared fixtures
  (`HashEmbeddingBackend`). Never create `tests/conftest.py`; put shared test
  config in the root one.
- Windows MAX_PATH (260) is close for `runs/<tool>__<timestamp>-<uuid>/<file>`
  paths. Any test-time scratch directory must stay SHORT (the `tmp_path`
  override in the root conftest lives at `%TEMP%/nlp-tmp` for exactly this
  reason: a workspace-local scratch dir pushed run paths over the limit and
  failed them as FileNotFoundError).

## From the 2026-09-21 contract / taxonomy / LDA session

- ~~**Embeddings split is half done.**~~ — finished in the P4 batch (see above).
- **pyLDAvis as an optional extra** was deliberately not taken; the Intertopic
  Distance Map and lambda relevance are native (`core/analysis/lda.py`). If a
  class wants the real pyLDAvis HTML, add it as an extra that renders the same
  numbers rather than as the source of them.
- ~~**This machine's pytest `tmp_path` is broken**~~ — fixed in the root
  `conftest.py` (tempfile redirected to `%TEMP%/nlp-tmp`, which also bypasses
  the poisoned `pytest-of-<user>` directory). New tests still prefer
  `tests/fixtures/` files where a fixture is the clearer oracle.
- ~~**Server-side abort for lost live requests**~~ — done as R-C8.
- ~~**Side-by-side run comparison**~~ — done (`POST /projects/{id}/compare`).
