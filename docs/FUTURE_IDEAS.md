# Future ideas, bug fixes, and observations

Collected while implementing the syllabus-alignment / communication-contract work.
Nothing here is committed scope — this is the parking lot for what we notice but
do not do yet. Discuss before promoting anything into a real roadmap item.

## Housekeeping

- Consider a CI check for stray root-level files.

## Contract / architecture

- True interruption of long model fits would require a cancellation token in
  the adapters. Current cancellation discards a mid-flight answer.
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

## Tools / syllabus

- **pyLDAvis artifact as an optional extra.** Gensim LDA's Intertopic Distance
  Map and λ-relevance terms are implemented natively. If a class ever wants the
  real pyLDAvis interactive HTML, add it behind an optional extra rather than
  as a dependency of `lda_gensim`.
- **Wordcloud options parity.** HW1 rubric asks about "various wordcloud
  options" (masks, shapes, group colors). The legacy raster-PNG port exists
  (`wordcloud_gephi`); worth a pass against the legacy GUI's option list to
  confirm every graded option is reachable from the desktop. (Not done: the
  legacy GUI's option list is not in this repo.)
- **Side-by-side comparison tables.** The existing comparison diff shows where
  two runs differ; a paired table could show both full results together.
- **Three-class sentiment.** Check the candidate model's redistribution licence
  before adding a neutral class to the released models.

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
