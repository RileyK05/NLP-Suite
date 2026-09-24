# Research workflow review — 2026-09-20

Follow-up: [review of the subsequent implementation changes](QUESTION_RESEARCH_REVIEW_2026-09-20_FOLLOWUP.md).
That review supersedes the status of the findings below, records verified fixes,
and identifies remaining publication, position, snapshot and saved-question gaps.
This document retains the original assessment and product rationale.

Reviewed the working tree at `623e2cf` against `QUESTION_DRIVEN_RESEARCH_PLAN.md`
and the earlier interactive research plan. This is a focused implementation and
plan review, not a fresh audit of every migrated legacy tool or a production
security certification. No application code was changed during this review.

The project now has a usable first research workflow. It preserves occurrences,
links annual/document aggregates to concordance, saves questions, compares two
phrases, and publishes research tables. The next milestone should finish and
stabilize that workflow before expanding the number of analyses. Passing the
existing tests does not yet establish the plan's full definition of done.

## What is implemented

| Capability | Current state | Remaining work |
| --- | --- | --- |
| Corpus selection | Explicit document selection, date bounds, undated policy, frozen run scope | Editable metadata, date precision, named research cohorts |
| Reusable parsing | In-memory and disk annotation caches keyed by corpus/parser identity | Reuse individual documents across changed selections; occurrence/result caching; resource budgets |
| Phrase evidence | Literal sequences, overlap, sentence boundaries, stable project IDs, content hashes, verified source spans including UTF-16 coordinates | Query/parser tokenizer consistency; versioned matching interpretation |
| Time analysis | Annual counts, rates, document prevalence; observed zeroes distinguished from years without documents | Calendar grouping, visible coverage, precision-aware dates, measure selection |
| Position analysis | Relative positions and pooled bins within each document | Per-document tracks, clickable ranges, absolute coordinates, equal-document weighting |
| Concordance | Paging, year/document filters, verified or explicitly reconstructed snippets | Expanded source reader, range filters, independent comparison paging, query-state fixes |
| Saved questions | Backend create/update/duplicate/delete, revisions, reopen, backup ID remapping | Update/duplicate UI, dirty state, snapshot validation, cold-start access |
| Comparison | Two phrases on time/position/document views with evidence for each | One bound comparison request; metadata cohorts and context comparison |
| Publication | Private publisher, frozen job scope, snapshot check, six CSVs, complete occurrences and selected-evidence flags | Executed phrase preview/export/restore equivalence test; matching-version contract |
| General research navigation | Direct phrase entry alongside the advanced analysis controls | Question templates and entry from n-gram results or selected source text |

Preserve these foundations. The phrase endpoint already computes aggregates from
all matches and only pages the evidence; it does not inherit the generic live
table's 500-row display limit. The legacy n-gram and dispersion tools also retain
their own semantics rather than silently becoming literal phrase tools.

## Findings to fix before calling the workflow ready

### R1 — High: ordinary literal phrases can incorrectly return zero matches

**Resolved 2026-09-20.** `Pipeline.tokenize()` is now part of the backend protocol (spaCy, Stanza, CoreNLP and the fallback wrapper), `Warm` carries the tokenizer that produced its table, and `PhraseQuery.tokens()` takes it. The old regex survives as `_approximate_tokens` for callers with no snapshot and is named as such in the record. Every answer now carries `question.matching_profile` with a version, the tokenizer's identity and whether a snapshot resolved it. Verified against the real SOTU corpus: `U.S.` returns 107 occurrences in 18 documents where it returned zero, `9/11` 9 in 6, `COVID-19` 9 in 1. `tests/test_question_integrity.py` pins the parity cases the review asked for, and a model_integration test checks the expected tokens against the installed tokenizer *and* against the Form column a parse of the same characters produces.

`core/research/phrase.py:24,55` tokenizes queries with a separate regular expression.
It disagrees with the offered spaCy English tokenizer:

| Input | Query tokens | spaCy tokens |
| --- | --- | --- |
| `U.S.` | `U`, `.`, `S`, `.` | `U.S.` |
| `COVID-19` | `COVID`, `-`, `19` | `COVID-19` |
| `3.5` | `3`, `.`, `5` | `3.5` |
| `AT&T` | `AT`, `&`, `T` | `AT&T` |
| `state_of_the_art` | `state`, `of`, `the`, `art` | `state_of_the_art` |

A corpus containing `U.S.` and a canonical table with the single form `U.S.`
returns **zero occurrences** for `PhraseQuery("U.S.")`. This was reproduced, and
the token differences were checked with the installed spaCy English tokenizer.
The result undermines a basic research promise: a phrase visibly present in the
source should be discoverable under the stated literal matching policy.

Use the snapshot's tokenizer to resolve query tokens, without rerunning the full
corpus parser. Persist the resolved interpretation and matching-profile version.
Add parser-parity cases for abbreviations, hyphenated terms, decimals, punctuation,
contractions, underscores and Unicode. Validate Stanza separately rather than
assuming spaCy-specific exceptions apply to it.

### R2 — High: overlapping loads can attach the wrong scope to evidence

**Resolved 2026-09-20.** Corpus, parser and selection are one immutable `Snapshot` value committed under a load generation. A load that finishes after a newer one has started, or after `cool()`, discards itself instead of committing; a superseded failure no longer blames the current load; and `state()` reports the loading request while loading and the committed snapshot afterwards, never a mixture. `Session.resolved(snapshot_id)` binds every question, and `PhraseBody.snapshot_id` carries it from the interface, with saving refused when the live state and the answer describe different snapshots. A stale reference answers **409** with `stale_snapshot: true` rather than a 400 the interface cannot act on. The review's two-thread reproduction is now a test, along with cool-during-load and superseded-failure cases.

`desktop_backend/live.py:410` locks assignments but releases the lock during
parsing. An older load can finish after a newer load and overwrite `_warm`, while
`_selection` and `_parser` still describe the newer request.

A deterministic two-thread reproduction started A/spaCy, completed B/Stanza,
then released A. Final state:

```python
{"snapshot_id": "A", "selection": {"ids": ["B"]}, "parser": "stanza"}
```

This can happen with concurrent clients or overlapping API calls. The API does
not enforce the restriction implied by the load button. `cool()` also does not
prevent an in-flight load from restoring a supposedly released corpus.

The issue extends beyond loading: `PhraseBody` in `server.py:97` has no expected
snapshot/query handle. `Session.track_phrase()` uses whichever corpus is current.
`App.tsx:689` saves an answer's snapshot alongside the current live state's parser
and document IDs; the store checks membership, not that these describe the same
snapshot. Comparison requests are issued separately.

Commit corpus, parser and selection together under a load generation; discard
superseded completions and invalidate them on cooling. Bind queries, both
comparison subjects, paging and saving to one immutable resolved snapshot.
Reject a stale reference explicitly. Add out-of-order load, cool-during-load,
stale-response and cross-snapshot save tests.

### R3 — Medium: drilling into an answer can silently change its question

**Resolved 2026-09-20.** `PhraseExplorer` separates `resolve()` (the form becomes the question, paging restarts) from `refine()` (the resolved request again with only its evidence window moved). Drill-down and paging read `lastRequest`, never the form. The resolved request carries the answer's `snapshot_id`. A visible notice says when the form has moved away from the answer on screen, and a comparison that was asked for and did not arrive is reported instead of silently degrading to a single-phrase answer. `isDraftDifferent` and `refined` are pure and tested in `phrase.test.ts`.

`desktop/src/PhraseExplorer.tsx:69` builds every request from the editable form.
The year/document buttons and evidence pagination call the same function
(`:626` for paging), rather than using `lastRequest`.

Reproduction path from the code: track phrase A, type phrase B without submitting,
then click Next or a year in A's answer. That action runs B with A's page/filter.
It can replace the answer or open an empty later page for the new phrase. Editing
capitalization, comparison text or position detail has the same state problem.

Keep draft inputs separate from the resolved question. Drill-down must operate
on the resolved question; submitting a new question resets its paging. Give each
comparison subject a valid page cursor, and explicitly handle a failed comparison
request instead of treating it as a successful single-phrase request.

### R4 — Medium: the saved-question revision workflow is incomplete

**Resolved 2026-09-20.** Saving takes a `replacing` id, so Update keeps the name and adds a revision while Save as new keeps a separate record; Duplicate reaches the backend route that already existed. Reopening no longer copies the name into the save box, which was what guaranteed the duplicate-name refusal. The open record and its revision are shown beside the buttons. `PhraseExplorer` now renders outside the ready gate, so the saved shelf is reachable from a cold session — reopening a question already loads the documents it needs — with only the track button gated on a loaded corpus.

`App.tsx:689` always posts to the create route. Opening a saved question also
copies its name into the save field (`PhraseExplorer.tsx:136`). After editing and
tracking, Save question therefore receives a duplicate-name error unless the
user invents another name. Backend update/duplicate support is already present
in `desktop_backend/store.py:655` and subsequent methods.

`PhraseExplorer.tsx:285` publishes the selected saved record, which can differ
from the answer currently displayed. The button says "Publish saved", so this
is not a claim that the backend publishes the wrong record; the missing piece is
an explicit distinction between the saved revision and unsaved exploration.

Add Update saved question, Save as new, Duplicate and a visible unsaved-changes
state. Show the exact revision/scope being published. Add expected-revision
checking before enabling competing updates. Make the saved-question shelf
accessible before loading an arbitrary corpus: currently it is nested under
`state.state === "ready"` in `LiveBench.tsx:188`.

### R5 — Medium: paging limits the response, not the computation

**Partly addressed 2026-09-20.** The two quadratic costs are gone: UTF-16 offsets are a binary search over each document's astral positions rather than re-encoding the document prefix twice per occurrence, and the table is grouped by document once rather than converting the whole Document ID column per document. `source_offsets` now reports coverage of the filtered evidence beside the corpus-wide figures, so a fully verified filtered set no longer reads as partial. **Still open:** there is no occurrence cache, no cancellation contract, and no measured size/latency budget; every request still rematches and re-aggregates.

Every `Session.track_phrase()` call reruns matching, alignment, aggregation and
source-context construction, including a Next-page or year-filter request.
`core/research/phrase.py:197` also converts and scans the full document-ID column
once per document. `_utf16_offset()` at `:141` encodes the source prefix for each
occurrence, twice. Frequent matches in long texts can make that work grow sharply.

There is no phrase occurrence cache or query cancellation contract here. Existing
small-corpus live performance tests do not establish supported sizes for this
new path. Index token rows and source offsets once per snapshot, cache bounded
occurrence sets, and construct detailed source context for requested pages.
Keep publication able to stream the complete evidence. Measure cold preparation,
warm matching, paging, peak memory and cancellation on representative corpora
before promising instant interaction.

One smaller evidence-label issue: `_evidence_page()` reports source-offset status
over all occurrences even after filtering. Either label that status as global
coverage or also report coverage of the filtered evidence. Do not imply that a
filtered set with wholly verified spans contains unverified highlights.

## What the plan needs to add or make concrete

### 1. Separate correctness, capability and research interpretation

Track these as different acceptance gates. A chart appearing is not proof that
the question is preserved while paging, that a saved revision reopens faithfully,
or that a peak represents a broadly shared pattern. Give each slice a known-answer
corpus, an actual researcher journey, and an exported result to compare.

Add matching-profile and aggregation versions to resolved/saved questions. The
current schema version and annotation snapshot do not explicitly freeze query
tokenization or future weighting rules. When those change, reopen must reproduce
the supported old interpretation or offer an explicit migration.

### 2. Complete “where in the text?” as an evidence workflow

The pooled bars answer an aggregate position question, but not whether the
pattern recurs across documents. Add one occurrence track per document, a
selectable position range, expanded source passages, and relative/token/sentence
coordinates. Expose pooled versus equal-document weighting with counts, exposure
and contributing-document counts. Preserve the current distinction between
filtering evidence and redefining the corpus; make active filters individually
removable and undoable.

### 3. Put metadata before richer temporal and cohort claims

Current dates are filename-derived. `Document` and phrase outputs lack date
precision and metadata revision. A filename containing only a year can become
January 1 internally, so adding monthly grouping alone would introduce false
precision.

Add editable/importable date values with precision and source, plus speaker,
author, genre, source and tags. Validate metadata joins, missing values and
duplicates. Freeze the metadata revision used for a saved answer. Then add
year/month and appropriate custom windows with an explicit policy for imprecise
dates, visible excluded-document counts and counts/rates/prevalence controls.

Named cohorts should show their membership, missing metadata, overlap and
denominators. Comparing two phrases is already useful; comparing the *same*
phrase across speakers, genres or periods answers a different question and needs
its own contract. Show composition and concentration, including whether a peak
is dominated by one document or source, before adding formal significance claims.

### 4. Make the next question a first-class action

Add “Track this phrase” to n-gram results and source selection. Offer concrete
templates that preserve subject, scope and evidence: “When did it change?”,
“Where does it appear?”, “Which groups differ?” and “What is said around it?”

The next analytical extension should compare context around the same subject
between selected periods/cohorts, with a defined token or sentence window and
links to passages. Follow with lemma/phrase-set subjects, entity trajectories,
subject-specific sentiment and relation questions. Keep their matching/model
provenance explicit and let users flag misleading evidence without deleting the
original result. An unrestricted natural-language question box can follow these
typed operations; it is not needed to deliver them.

### 5. Make reuse and recovery part of the product contract

The existing annotation cache is keyed by whole corpus, so a different selection
does not automatically reuse its unchanged documents' annotations. Add document
cache reuse with stable ID remapping, then query/aggregate caches with distinct
invalidation for text, tokenizer, metadata and visualization changes. Define
bounded memory/disk use, eviction, cancellation, interruption recovery and a
clear re-resolution action for expired snapshots.

## Recommended implementation order

Order A is implemented as of 2026-09-20 except the caching and measured budgets
noted under R5; B, C and D are untouched. A follow-up review the same day found
five further defects across these findings (published answers matching by
different rules, sentence-local word positions, unbound reopened and comparison
requests, conflated shelf/open identity, and unpersisted matching provenance);
all five are fixed, along with two more that only the real corpus showed. See
[the follow-up](QUESTION_RESEARCH_REVIEW_2026-09-20_FOLLOWUP.md) for what each
one was and how it was verified.

| Order | Deliverable | Acceptance gate |
| --- | --- | --- |
| A | Fix tokenizer, snapshot/loading integrity, draft/resolved state and save revisions | Known source phrases found; concurrent loads cannot mix state; paging preserves subject; edits save as the intended revision |
| B | Finish the four linked phrase panels and source reader | Every time/position/document selection resolves to the expected passages; weighting is explicit; saved questions work from a cold session |
| C | Metadata, date precision, richer windows and cohort comparison | Mixed-precision dates never imply false monthly observations; cohort membership, overlap and exposure are visible and reproducible |
| D | Context comparison and broader typed subjects | A researcher can investigate what changed around a subject while retaining source evidence and method provenance |

Cache/performance work begins in A alongside snapshot ownership, rather than
waiting until all new analyses exist. Before expanding C/D, run an end-to-end
save → reopen → publish → backup → restore → reopen/publish test that compares
real phrase evidence and aggregates, allowing only the documented stable-ID
remapping. Add genuine rendered UI interaction tests for the journeys above.

## Verification performed

- Focused backend suites: `test_phrase_research.py`, `test_saved_questions.py`,
  `test_live_analysis.py`, `test_profiler_executor.py`: **53 passed, 12 deselected**
  with the ordinary non-model marker selection.
- Existing live-analysis model integration suite: **9 passed, 22 deselected**.
- Desktop phrase helpers, live helpers and PhraseExplorer tests: **34 passed**.
- Desktop TypeScript/Vite production build: **passed**.
- Deterministic overlapping-load reproduction: confirmed mismatched
  snapshot/selection/parser as documented in R2.
- spaCy tokenizer comparison and a minimal literal-phrase answer reproduction:
  confirmed R1.

The PhraseExplorer test currently renders only the pre-answer UI. Saved-question
publication tests freeze a job with a placeholder snapshot and suppress its
execution; backup tests check record/ID preservation. These are useful tests,
but do not yet prove the complete phrase interaction/export/restore journey.
No large-corpus benchmark, packaged installer exercise or fresh full legacy
regression/security audit was performed in this review.
