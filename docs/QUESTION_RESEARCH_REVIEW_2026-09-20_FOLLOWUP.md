# Research workflow follow-up review — 2026-09-20

Reviewed the uncommitted implementation changes on top of `623e2cf`, following
the first review. Existing application edits were preserved. This pass changes
review documentation only. Findings below supersede the original review's status
claims; the original remains available as the historical rationale.

**Historical assessment before the resolution pass:** the first correctness
milestone was partially complete. Some callers bypassed the new contracts, and
a pre-existing position bug needed fixing. Read the resolution notes below for
the implemented fixes, and the [next feature specification](RESEARCH_INVESTIGATION_FEATURES.md)
for the product work proposed after reviewing commit `834da17`.

> **Status as of the 2026-09-20 implementation pass:** F1-F5 below are resolved,
> with a resolution note under each finding. Two further defects, F6 and F7, were
> found while testing against the real corpus and are also fixed. The table
> immediately below records the state *before* that pass; read the notes under
> each finding for what is true now. Still open from R5: occurrence caching,
> cancellation, and measured size/latency budgets.

## Progress against the first review

| Finding | Verified progress | Current status |
| --- | --- | --- |
| R1: tokenizer mismatch | Live snapshots retain the parser tokenizer; real spaCy parity tests cover abbreviations, compounds, contractions and Unicode | Partial: publication still uses approximate tokenization |
| R2: snapshot ownership | Atomic Snapshot object, generation-checked completion/progress, stale-reference exception and HTTP 409 handling | Load-state defect fixed; reopened paging, initial comparison binding and save validation remain incomplete |
| R3: draft versus resolved question | Normal refinement reads `lastRequest`; draft changes are visible; comparison failure has a warning | Original draft/paging defect fixed in code; reopened requests lose snapshot binding and comparison pages remain coupled |
| R4: saved questions | Update, Save as new, Duplicate controls; shelf available with no corpus loaded; frontend save checks snapshot equality | Partial: selected record is mistaken for opened record; opening does not initialize the save name; no expected-revision check |
| R5: repeated query work | Document rows are grouped once per request; UTF-16 offsets use precomputed astral positions | Those two hot spots fixed; occurrence caching, cancellation and measured resource budgets remain open |
| Offset-coverage labeling | Backend now returns filtered verification totals/status | Frontend types and rendering still use the global fields |

## Remaining correctness findings

### F1 — High: live and published phrase answers disagree

**Resolved 2026-09-20 (implementation pass).** Both halves of the recommendation are in place. `BatchContext` carries a `Tokenization` -- the split function and the name an answer records it under, as one value -- supplied by `Bench.analyse` from the warm parse and by the desktop runner from the pipeline it just parsed with. Separately, a saved question's own tokens travel to the publisher as `resolved-tokens` / `comparison-tokens` and take precedence, so publishing asks the question that was saved rather than today's reading of its characters; the published `phrase_question.csv` now records `matching_version`, `matching_source`, `matching_tokenizer`, `resolved_tokens` and `comparison_tokens`.

Verified on the 87 State of the Union addresses, through `_adapt_phrase_distribution` itself: `U.S.` 107/107, `COVID-19` 9/9, `9/11` 9/9, `public health` 13/13, `war on terror` 15/15 -- live against published, with every occurrence id, token offset, word position, position bin and character offset compared in order, not just the totals. The same adapter given no tokenizer still returns 0 for `U.S.`, which `tests/test_real_corpus_questions.py` asserts on purpose so the divergence cannot come back unnoticed. Confirmed again through the running server: a published run of the saved question returned 107 occurrences with ids identical to the live answer's.

`desktop_backend/live.py:531` supplies `warm.tokenize` to the matcher.
`core/profiler/executor.py:645` does not. `BatchContext` has no tokenizer contract,
and the publishing runner does not carry resolved query tokens into the adapter.
Consequently, the live-path fix does not reach publication.

Reproduced with the same corpus, canonical table and snapshot:

```python
# source text: U.S.; canonical Form column: ["U.S."]
{"live_occurrences": 1, "published_occurrences": 0}
```

The publication result was obtained by actually calling
`_adapt_phrase_distribution`, not by inspecting its request. A correct snapshot
hash alone cannot establish equivalent answers when matching differs.

Carry the same resolved subject tokens and matching policy through live, saved
and published execution, or supply the exact tokenizer to both paths. Add a
preview/publication equality test with `U.S.`, `COVID-19`, decimals and Unicode,
including comparison subjects and full exported occurrences.

### F2 — High: document-relative positions reset at sentence boundaries

**Resolved 2026-09-20 (implementation pass).** The per-document scan is now `_document_occurrences`, which advances a document word offset alongside the document token offset it already kept; the two are updated in the same place, which is what stops one being reset with the other.

The review's estimate of the damage was conservative. On the real corpus **every** occurrence landed in the first tenth of its document: `the American people` 282/282 in bin 1, `U.S.` 107/107, `public health` 13/13. The position chart was one bar. After the fix those spread to {1: 59, 2: 33, 3: 22, 4: 21, 5: 14, 6: 20, 7: 25, 8: 29, 9: 27, 10: 32}, and each word position is checked against an independent count over the whole document's token list.

`core/research/phrase.py:408–416` builds `word_before` from zero inside each
sentence and uses it directly as `word_start`. The absolute token offset is
document-local, but the word position omits all earlier sentences.

Reproduced with:

```text
one two three four five six seven eight nine ten. alpha beta.
```

For `alpha beta`, the current answer reports `word_start=0`,
`relative_position=0.0`, `position_bin=1`, and `token_start=11`.
The correct word position is 10 of 12 words: approximately 83.3%, bin 9 when
using ten bins. This affects the position chart and exported occurrence/position
tables. It is a pre-existing defect missed by the first review, not a regression
introduced by the latest tokenizer changes.

Maintain a document-level word prefix or cumulative sentence word offset.
Test multiple sentences, punctuation-only sentences, unequal sentence lengths,
and phrases at the start/end of later sentences. Assert token and word coordinates
against the same source fixture.

### F3 — High: snapshot binding still has holes in the UI

**Resolved 2026-09-20 (implementation pass).** Reopening binds `setLastRequest({...request, snapshot_id: result.primary.snapshot_id})`, so the first page or filter after an open cites the corpus that answered. A comparison's two answers are combined only through `comparable()`, which returns the comparison when its snapshot matches the primary's and null otherwise; a mismatch is reported as a comparison that did not arrive rather than drawn beside a subject from other documents. `savePhraseQuestion` refuses a save whose two subjects disagree about the snapshot.

`desktop/src/PhraseWorkflow.test.tsx` renders the component in jsdom and drives it by clicking: open then page, track then page, page after editing the form, and a comparison whose second subject resolves against a different snapshot.

The backend correctly checks an expected snapshot when supplied, and ordinary
`run()` results bind subsequent requests. Two paths bypass that protection:

- `PhraseExplorer.tsx:188–211`: reopening initially uses a bound request in
  `App.tsx`, but the component constructs a separate request without a snapshot
  and stores that with `setLastRequest(request)`. The first subsequent page or
  filter request can therefore answer from a reloaded corpus. Repro path: open
  saved question A, load corpus B, then page A's retained answer.
- `PhraseExplorer.tsx:120`: a new comparison submits two concurrent requests
  with no expected snapshot. If the corpus changes between their resolutions,
  the results can belong to different corpora. `run()` does not compare the
  returned snapshot IDs before combining them.

Bind the reopened request to `result.primary.snapshot_id`. Resolve both comparison
subjects against one expected snapshot, and reject mismatched returned IDs. Add
rendered interaction tests for open → reload → page and a comparison with a
controlled reload between subject requests.

### F4 — Medium: selection and opened-question identity are conflated

**Resolved 2026-09-20 (implementation pass).** Shelf selection (`selectedId`) and the open record (`opened`: id, name and the revision it was opened at) are separate state. Open adopts identity and fills the save box only after `onOpenQuestion` returns an answer; a failed open leaves the previous answer and offers nothing to update. Duplicate selects the copy without claiming it is open, and the interface says so when the highlighted row is not the open one. Unsaved-change status comes from `differsFromSaved`, which compares the resolved question with the saved specification rather than the form with the last answer.

Updates carry `expected_revision`; `Workspace.update_question` refuses one aimed at a revision that has moved, naming the current one. Verified against the running server: update from revision 1 succeeded to revision 2, a second update from revision 1 was refused with 400.

`PhraseExplorer.tsx:344` changes `savedId` as soon as a shelf option is selected.
That same value labels the answer as `Open: ...` and determines the record updated
by `keep(saved.id)` at `:485`, even if Open was never pressed or failed.

Concrete code path: display answer A, select saved question B without opening it,
then Update B. The submitted evidence/settings still come from A. The button
names B, but the interface incorrectly describes B as open and offers no explicit
replace-from-another-answer workflow. Duplicate can similarly select a copied
record without loading its contents.

There is a separate ordinary-use failure: `openSaved()` no longer calls
`setSaveName(saved.name)`. On a fresh session the name remains empty, so Update
appears available but `keep()` silently returns. After previous work it can carry
an unrelated name into an update.

Track shelf selection separately from the opened record and its revision. Adopt
opened identity/name only after a successful open. Compare the resolved answer
against the saved specification for unsaved-change status: `draftDiffers` only
compares the form with the last answer and becomes false after a new query runs.
Add expected-revision validation to updates so concurrent edits cannot silently
replace each other. Test a fresh open/update, selection without opening, failed
open, duplicate and save-as-new.

### F5 — Medium: matching provenance is returned but not persisted

**Resolved 2026-09-20 (implementation pass).** `MatchingProfile` (version, tokenizer, source, tokens, comparison tokens) is part of `QuestionBody`, stored in a new `matching` column with the usual `ALTER TABLE` migration, carried by duplicate, sent by the desktop from the answer's own profile, and passed to the publisher.

The migration rule is explicit: a question saved before any of this carries `version: 0, source: "unrecorded"` and no tokens, and reopening or publishing it resolves its text again with the snapshot's parser -- the published record then says `matching_source=snapshot` rather than `saved`, so a re-reading is visible in the artifact. Provenance is kept out of `PhraseQuestionSpec`, which remains what is editable about the question.

Save validation now also checks that the scope holds together: when the project has a corpus loaded, a question's snapshot, parser and document set must match it. A project with nothing loaded cannot corroborate a snapshot id without resolving a parser, so there the publish-time check in the runner remains the one that catches a mismatch -- stated here rather than left as an implied guarantee.

`phrase.py:484` returns a matching-profile version and tokenizer name. However,
`desktop_backend/questions.py:13` has no matching-profile or resolved-token fields,
the frontend save payload omits them, and the publication question CSV omits them.
The comment claiming the profile is saved with the question is ahead of the code.

Persist resolved tokens, tokenizer/matching version and applicable aggregation
version. Define whether old questions retain supported old semantics or require
an explicit migration. Keep text/parser snapshot identity separate from the
research interpretation; a tokenizer/matcher change must not silently redefine
an existing saved question. Backend save validation must also establish that
snapshot, documents and parser belong together, not just validate document IDs.

## Found while testing against the real corpus

### F6 — High: no source passage anywhere could be pointed at its own text

Not in the review, because nothing in the code reads as wrong. `_align_tokens`
steps over whitespace in the source before matching each token, on the
reasonable assumption that canonical token tables omit whitespace. spaCy does
not: it emits whitespace *as* tokens -- 10,017 of them across these 87 speeches,
including the newline that begins every one of the files. The first token of
every document therefore failed to match, and one mismatch discards the whole
document's spans by design.

The result was that **0 of 87 documents had usable source offsets**, so every
occurrence fell back to reconstructed token context and the interface said "the
parser tokens could not be aligned exactly to the imported text" -- which reads
as a property of the documents. The entire source-evidence feature, and the
source reader Order B is built on, had never worked on real text.

`_align_tokens` now matches a whitespace token inside the current run of
whitespace and never past it, so it still cannot skip content to find a later
match. All 87 documents align, and `U.S.` publishes 107 of 107 occurrences with
exact source spans, each one checked by reading the characters back out of the
imported document.

This is also the reason the offset-coverage labelling was invisible: the
filtered figures the previous pass added were correct and always reported
`unavailable`.

### F7 — Medium: a cooled session still held the parse it had released

The follow-up's closing paragraph noted that an in-flight `Bench.warm()` can
populate `_held` after `cool()` releases it. It can, and it did: warming holds
the table before it returns, so a superseded load left its 68 MB behind a
session reporting itself cold. `Bench.forget(key)` drops one named parse, and a
superseded load forgets its own -- not everything, because by then a newer load
may be the one being answered from. Both cases are tested.

## Remaining product gaps and next order

F1–F5 are resolved as recorded above. The broader roadmap remains sound; the
[investigation feature specification](RESEARCH_INVESTIGATION_FEATURES.md) now
turns these directions into bounded deliverables and acceptance criteria:

1. Finish the linked phrase workflow: per-document occurrence tracks, position
   brushing, expanded source passages, selectable measures and equal-document
   weighting. Keep evidence filters distinct from corpus selection.
2. Add editable/importable metadata with date precision, source and revision;
   then monthly/custom windows and named cohort comparisons with visible
   membership, overlap and denominators.
3. Add context comparison across selected periods/cohorts, followed by broader
   typed subjects such as lemma sequences, entities and subject-specific sentiment.
4. Add question-preserving entry from n-gram results and selected source passages.

Continue resource work alongside correctness. Paging still rematches and
reconstructs all occurrences. Both comparison subjects still share one page
offset. F7 addresses the superseded parse retained after cooling. Continue with
bounded cache ownership/eviction, cancellation and measured latency/memory
acceptance criteria.

The next test investment should be actual interaction and execution coverage:
query → refine → save/update → reopen → publish → backup/restore → reopen/publish.
Use a multi-sentence corpus with tokenizer edge cases and compare source spans,
counts, coordinates, matching rules and scope across every step. Existing helper
tests cannot establish that these callers all use the new contracts.

## Validation in the implementation pass (2026-09-20)

- Full Python gate: **2169 passed, 6 skipped**; ruff, ruff format, mypy (228
  files), `scripts/rc_audit.py` 0 failures.
- Real-parser and real-corpus tests: **100 passed, 12 skipped** under
  `-m model_integration`, including 20 in `tests/test_real_corpus_questions.py`
  and 9 in `tests/test_question_workflow.py` over the 87 speeches.
- Desktop: **349 passed across 18 files** (11 of them the new rendered
  interaction tests), prettier, `tsc --noEmit`, `vite build`.
- Through the running preview server, on the real corpus: `U.S.` 107
  occurrences with `matching_profile.source = snapshot`; save, update from
  revision 1, refusal of a second update from revision 1; refusal of a save
  naming another snapshot or another parser; and a real published run returning
  107 occurrences whose ids, position bins and source spans are identical to the
  live answer's.
- `tests/test_end_to_end.py` was failing at `623e2cf` before this work, on a
  stale artifact count (the writer gained a readout beside every table). Fixed
  here because a red gate hides the next real failure.

## Validation in the review pass

- Focused non-model backend tests for question integrity, phrase research, saved
  questions, live analysis, profiler execution and pipelines: **88 passed,
  1 skipped, 30 deselected**.
- Real-model question-integrity and live-analysis integration tests:
  **23 passed, 46 deselected**.
- Entire desktop test suite: **338 passed across 17 files**.
- Desktop TypeScript/Vite build: **passed**.
- Live-versus-publisher tokenization divergence: reproduced, 1 versus 0 matches.
- Multi-sentence position error: reproduced, 0% instead of approximately 83.3%.

At the time of the original review pass, PhraseExplorer had only a pre-answer
static render and helper tests. The implementation pass subsequently added
`PhraseWorkflow.test.tsx` with rendered interaction coverage, as recorded above.
The original UI findings were traced from component/callback state transitions;
that review did not run an interactive browser session or make application fixes.
It also did not perform a large-corpus benchmark, packaged-installer test or
complete legacy-tool audit.
