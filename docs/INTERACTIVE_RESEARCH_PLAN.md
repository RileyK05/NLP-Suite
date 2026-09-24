# Interactive research workspace — implementation plan, 2026-09-19

Next product plan: [Question-driven NLP research](QUESTION_DRIVEN_RESEARCH_PLAN.md).
It builds on the implemented live cache, selections and saved views, and defines
linked phrase trajectories, document-position profiles and inspectable evidence.
Historical gap lists below describe the state when their sections were written.

## Product direction

A researcher should be able to ask a question, choose the relevant documents,
compute the necessary evidence, and explore many views of that evidence without
rerunning NLP for every visual change. A saved view should reopen with its source,
scope and settings intact. Publishing creates a reproducible artifact from that
view; experimenting must not overwrite a previous result.

Example: select speeches from 2018–2020, compute sentiment once, compare the trend
by month and by speaker, inspect the sentences behind an unusual point, then save
two views. Changing a chart colour or grouping should not parse the speeches again.
Changing the corpus or an analysis setting may require new computation.

## Existing work to preserve

Claude's uncommitted Explore, Workbench, ChartCanvas and chartLayout code already
provides instant chart editing over loaded result tables, mark-to-row selection,
and a project-wide result-table picker. The backend retains immutable run results.
Extend those components rather than introducing a second chart system.

Current limitations verified in code:

- Workbench state is temporary and charts the loaded page, not necessarily all
  rows. Publishing submits the complete source artifact.
- `publishParams` does not carry every preview setting (such as log scale and
  display sorting). Preview/export equivalence needs an explicit contract.
- `PipelineCache` caches model instances, not document annotations. The runner
  currently constructs it inside each job. Do not describe this as a persistent
  analysis cache.
- No durable named cohorts, document metadata editor, or cohort comparison UI.

## First slice implemented in this pass: explicit analysis scope

Corpus analysis forms offer manual document choices, a filename search to help
choose them, inclusive date bounds, an explicit undated-document choice, and
document/word counts. The name search does not silently select documents.

`POST /api/projects/{project}/jobs` accepts optional `selection`:

```json
{
  "tool": "readability",
  "selection": {
    "document_ids": null,
    "date_from": "2018-01-01",
    "date_to": "2020-12-31",
    "include_undated": false
  }
}
```

Omitted selection preserves existing clients' whole-corpus behavior. Null IDs
mean all documents before date filtering; an empty list means none and is rejected.
Explicit IDs and dates intersect. Unknown project IDs, invalid dates, reversed
bounds and zero-match selections fail before job creation. CSV workflows reject
corpus selections because they operate on an explicitly chosen table instead.

The runner resolves the selection against project documents under the existing
submission lock. The request stores the exact documents and normalized criteria;
later imports cannot alter a queued run. The public job carries a scope summary.
The exported `desktop_inputs.csv` includes membership, hashes, filename-derived
dates, date provenance and filtering criteria. Existing backups preserve this
request and the run artifacts without a schema migration.

Dates currently use the existing conservative filename parser. Import timestamps
are never substituted for document dates. Manual date correction is next-stage
work, and must preserve provenance and precision (a year is not a known day).
Document IDs in restored historical runs identify their original snapshot; do not
reuse those IDs as a new selection without mapping to the restored documents.

## Next slice: saved views and faithful publishing — implemented

Items 1, 2 and 4 are done, and item 3 is done as far as the engine allows; the
remainder is stated under "Ordering" below. `desktop_backend/views.py` holds the
validated record and the publication contract, `Workspace.save_view` and its
neighbours the storage, `desktop/src/views.ts` the single translation between
the workbench's `Settings` and the stored record, read by
`tests/test_view_parity.py`. Backups carry views at manifest `version: 2` and
restore still reads `version: 1`; restore remaps the reissued job IDs, and a
view naming a run the archive does not contain keeps its settings and reports a
missing source.

**Ordering** is the one setting the engine cannot reproduce. `ChartSpec` has no
ordering field — `prepare_chart_data` sorts x ascending, always — so the
workbench's "Largest first" cannot survive publication. It is named on screen
above the Publish button rather than dropped silently, with the words shipped
from `/api/chart-contract` so the desktop holds no second copy of the rule.
*Which* categories publish does now agree: `orderCategories` breaks ranking ties
by category name, matching chartspec's `top_n`, which it previously did not.
Giving `ChartSpec` a real ordering field touches golden files, the CLI flags and
the chart parity tests, so it is its own slice and is not done here.

`logY` was removed rather than stored: it was declared in `Settings`, defaulted
in three places and read nowhere.

1. Add versioned project-owned view records: name, source run/artifact, source
   content hash, validated chart settings, row predicates, revision and timestamps.
   Keep draft changes separate from immutable published artifacts. Support save,
   reopen, duplicate and reset; report save failures visibly.
2. Make Workbench controlled by a view state object. Preserve separate views for
   two artifacts with identical columns. Name, save and restore views in Explore.
3. Define one supported setting/filter contract for preview and publication.
   Include sorting, log axes, category limits and row filters, or explicitly
   disable publication of unsupported settings. Never silently export a different
   scope. Record the view revision and source hash in the published provenance.
4. Include views in project backup/restore with source ID remapping and dangling
   source handling. Initially use SQLite like existing project metadata; do not
   add a service or database dependency just to store a few JSON view records.

Acceptance: close/reopen preserves two independently edited views; edits create
no analysis jobs; publishing preserves all supported settings and row predicates;
missing sources and malformed restored views fail visibly; backup/restore works.

## Next slice: full-result exploration and responsive queries

Start with a bounded, authenticated query API over immutable CSV artifacts. It
must support validated column predicates, grouping, aggregation, sorting and a
row/mark budget. Read the entire source for correct aggregation even when only a
bounded chart or row page is returned. Return counts for source, matched and
displayed rows, plus whether marks were sampled or truncated.

Debounce queries, cancel superseded requests and discard stale responses when a
different view or project is selected. Cache by artifact content hash, query
version and normalized query, not by the visible page. Only add an indexed table
store if measured large-table latency justifies it. Never accept arbitrary SQL or
Python expressions from the frontend.

Acceptance: a 40,000-row table produces the same aggregate as an offline reference;
changing pagination cannot change a chart's totals; rapid edits never display an
older response over a newer one; preview and publish use identical predicates.

## Next slice: reusable evidence, cohorts and questions

- Add named cohort definitions and a preview of their membership. Freeze a new
  snapshot on each run, and show changes since the previous snapshot.
- Add user-confirmed dates, date precision/source, speaker, genre, source and tags;
  support mapping metadata from a chosen file. Validate joins and report unmatched
  documents. Metadata edits must invalidate affected query results and snapshots.
- Cache document-level annotations keyed by text hash, actual parser/model version,
  language, normalization/schema version and relevant settings. Reuse only valid
  complete entries, publish cache files atomically, and allow bounded eviction.
- Distinguish document-local results from corpus-dependent results. TF-IDF,
  vocabulary fitting, topics, keyness and aggregate denominators may change with
  membership: filtering an old corpus result is not equivalent to recomputing on
  a new corpus. Build adapter capabilities rather than guessing cache safety.
- Add question templates: compare two periods, compare two speakers, inspect an
  entity's context, and investigate a sentiment change. Each exposes scope,
  denominator, method and evidence rows. Show overlapping cohorts and small-group
  limitations before users interpret the comparison.

## Handoff order

Keep each slice independently reviewable. Implement saved-view storage and tests
first, then controlled Workbench/save UI, then full-result queries and publication
parity. Metadata/cohort editing and annotation caching follow after those contracts
are established. Backend and frontend tasks may run in parallel once their request
schemas and file ownership are agreed. Preserve all existing uncommitted work.

This pass does not implement all of the stages above, rebuild an installer, or
claim full-corpus real-time performance. Record measured validation separately
from the acceptance tests proposed for future stages.

## Verification of the first slice

- 113 targeted Python tests passed, covering selections, desktop jobs, result
  discovery, navigation, production boundaries, and backup/restore.
- All 237 frontend tests passed, including six new selection/UI tests; the
  TypeScript/Vite production build passed.
- Desktop backend mypy and Ruff checks passed.
- Browser check with three fictional documents: a 2020 time window selected one
  document; switching to an empty manual selection disabled Run; choosing two
  dated documents retained only their intersection with the window. Readability
  completed with one result row and the correct visible scope record.
- Initial testing exposed an existing Windows path-length limitation: a 262-character
  artifact path could be enumerated but could not be read by Python. The suite
  passed using a shorter test workspace. Long-path runtime support remains a
  separate portability task; no retry-based workaround was added.

The local preview server used for verification was stopped afterward. These
checks validate the source slice, not a rebuilt installer or all future stages.
