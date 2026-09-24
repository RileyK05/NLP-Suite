# Next product milestone: investigate an explanation

Reviewed against `834da17` on 2026-09-20. This extends the question-driven plan
with concrete researcher journeys and implementation priorities. These are
proposed capabilities, not features already shipped.

The latest implementation wires parser tokenization into publication, accumulates
document-relative word positions, preserves reopened snapshot references,
separates shelf selection from opened questions, and stores matching provenance
and expected revisions. It also adds rendered interaction and real-corpus tests.
The next product milestone can build on those changes while retaining their
regression gates.

The milestone should answer this whole investigation:

> This phrase becomes more common in the later period. Is that because more
> people use it, because one speaker repeats it, or because its meaning and
> surrounding discussion changed? Which passages support my interpretation,
> and which passages complicate it?

Success is the ability to follow and preserve that investigation, including
exceptions and uncertainty. A chart, a score or an automatically written
conclusion alone does not complete it.

## 1. Read the evidence beside the pattern — build first

**Questions:** Where exactly does this happen? Does it recur near the end of
many documents, or only in one long text? What comes before and after it?

Add a linked source reader to the phrase workspace. Clicking an occurrence opens
the imported document at the verified span, with expandable paragraph and section
context. A row per document shows occurrence positions along its own length.
Click or brush a position range to inspect those occurrences, then switch between
relative, token and sentence coordinates. Keep documents without matches visible.

Show counts, rates, exposure and document coverage. Offer pooled and
equal-document weighting as explicitly different questions. Preserve the current
global chart while inspecting evidence; use a separate, explicit action to make
selected documents a new analysis scope. Filters must remain visible and removable.

Let a researcher select text in the reader and track it, or open an n-gram result
as a subject. The selected item should bring its scope and context with it.

**Acceptance:** a match in a later sentence opens the right characters and the
right position track; brushing selects exactly the documented start positions;
changing context size or visual appearance does not rerun the parser; clearing
filters restores the prior view. Imported-text offsets must not be presented as
original PDF page coordinates.

## 2. Compare what is said around a subject — build next

**Questions:** Is “security” discussed differently in these periods? What actions
occur around this entity? Did the phrase become more frequent while its contexts
stayed the same?

Keep one subject fixed and compare its contexts in two explicit document sets or
time windows. Existing manual selections and annual dates are enough for the
first version; richer metadata can follow. Offer sentence, paragraph when
available, and bounded token windows. Display recurring context words/phrases,
occurrence counts, document coverage and paired passage lists. Clicking any context
feature opens its actual evidence on both sides, including when one side has none.

Specify whether a context measure counts tokens, windows containing a feature or
documents containing it. Overlapping windows need an explicit counting rule:
union windows for unique token exposure, or label repeated occurrence-window
counts. Never cross a document boundary. Keep full counts and stable paging
available beyond the displayed ranking.

Start with transparent counts and rates. Add validated association/keyness
measures separately, with their units and assumptions. Lexical differences are
leads for interpretation, not automatic proof of a change in meaning.

**Acceptance:** a fixture with identical subject frequency but different contexts
shows the difference; one repeated passage cannot masquerade as widespread
document coverage; every context entry resolves to its matching windows.

## 3. Explain who or what drives a peak

**Questions:** Is this widespread or dominated by one document? Does the trend
survive removing one speaker? Is repeated boilerplate responsible for the result?

For a selected peak, show contributing documents and, when available, speakers,
genres and sources. Show their share of occurrences beside their share of text,
their individual rates and the distribution across documents. A list of top
contributors alone is insufficient: preserve the rest of the distribution and
documents with zero hits.

Offer an explicit “compare with this contributor excluded” branch that displays
the original and revised results together. Preserve the original result. Identify
repeated or near-duplicate passages as candidates for inspection, and let users
compare raw and explicitly deduplicated scopes without quietly changing counts.

**Acceptance:** a synthetic peak caused by one long document is distinguishable
from a peak shared across many documents. Exclusion changes the denominator and
records the new scope; it never edits the earlier saved answer. Frequency is not
automatically labeled importance, influence or causation.

## 4. Define the study population, not just a date filter

**Questions:** How do these speakers differ within the same period? Does the
pattern persist within genre? Are my comparison groups actually comparable?

Add editable/importable author, speaker, date, date precision/source, genre,
document source and tags. Validate metadata joins and show unmatched documents,
duplicates and missing values. Permit paragraphs/sections to carry metadata where
the imported structure supports it; distinguish those units from whole documents.

Named cohorts should have editable membership rules and a preview of their
members. Save the resolved membership and metadata revision with the answer.
Show overlaps and missing metadata before comparison. Offer explicit common
periods and within-group breakdowns; do not silently match, balance or exclude
documents. Date precision must constrain calendar grouping, so a year-only date
does not imply a known January observation.

**Acceptance:** researchers can compare the same subject across two speakers
within the same selected years, see precisely which documents contributed, and
reopen that answer after metadata edits without silently changing its population.

## 5. Make a subject richer than one spelling

**Questions:** Is this idea expressed with different words? Is the apparent
decline just a spelling change? Am I mixing different senses of the same word?

Introduce named, editable subject definitions: a literal phrase, a lemma
sequence, an explicit phrase set, or a reviewed entity/alias set. Show evidence
and contribution counts for each member before presenting a combined view.
Support exclusions and context conditions, for example a term near one expression
but outside contexts containing another. Keep literal, lemma and model-assisted
matching visibly distinct.

Define union counts versus member counts and deduplicate identical source spans
when aliases overlap. Preserve the interpretation and revision of each subject.
Semantic retrieval can suggest candidates for a researcher to review; it should
not silently expand the operational definition of a concept.

**Acceptance:** a subject set does not double-count the same occurrence simply
because two aliases match it. Users can inspect ambiguous matches, mark a sense,
compare reviewed and unreviewed subsets, and reproduce the original definition.

## 6. Keep the reasoning, including counterevidence

**Questions:** Why did I believe this? Which passages disagree? What changed when
I narrowed the corpus or corrected a match?

Add investigation notebooks that link a question, its saved revision, views,
supporting passages, contradictory passages and researcher notes. Allow branching
from a peak or selection into a new question while retaining the parent scope.
Record why a branch changed and compare it with the original.

Annotations should reference source spans and text versions. A researcher can
flag a false match, assign a code or record uncertainty; the original machine
output remains available. Reviewed exclusions are a visible analysis layer with
separate results, not silent edits to historical counts. If examples are sampled,
record the method and seed and let users inspect the full evidence set.

**Acceptance:** a saved interpretation can reopen its supporting and contradictory
passages, recover the exact query/scope, and show the effect of review decisions.
Backup/restore must preserve notes, links, subject revisions and branches.

## 7. Extend to actor, action and stance questions after the shared workflow

Use the same source reader, cohorts and review layer for “Who is described as
doing what to whom?”, “How is this institution evaluated?” and “How does an
entity's role change across periods?” Reuse existing NLP adapters only where
their output can identify the source sentences and relevant spans.

Expose model labels as model output, support corrections, and preserve passive
constructions, negation and uncertain pronoun/alias resolution in validation
fixtures. Document-level sentiment is not a substitute for sentiment toward a
particular subject. This expansion should retain evidence rather than reduce an
entity, document or relationship to a single unexplained score.

## Delivery sequence and technical prerequisites

| Deliverable | Build boundary | Done when |
| --- | --- | --- |
| B1: follow a finding | Source reader, document tracks, position filter, independent evidence cursors | A chart can be followed into its exact passages and back without losing scope |
| B2: explain the difference | Context comparison using existing selections; document contribution breakdown | Equal-frequency but different-context and single-document-dominated cases are distinguishable |
| C: design the study | Metadata, precision-aware dates, cohort definitions, weighting controls | Group membership, exposure, overlap and interpretation are explicit and reproducible |
| D: preserve and broaden inquiry | Versioned subject sets, notebook branches, annotation layers; then actor/stance adapters | A researcher can preserve and challenge an interpretation with recoverable evidence |

Begin a minimal notebook with pinned passages during B1; deepen branching and
review layers in D. Metadata is not a reason to postpone context comparison over
manually selected document sets. Conversely, named speaker/genre comparisons
should not ship before the metadata and cohort contracts exist.

The first shared backend addition should be a bounded occurrence-query cache:
resolve once per snapshot and subject interpretation, page source evidence without
rematching, and reaggregate without reparsing. Include metadata revision in
aggregate invalidation; keep visual settings separate. Add cancellation,
independent comparison cursors and measured cold/warm latency and memory budgets.
Use document-owned IDs and validated spans for the reader, not arbitrary paths.

Use a small known-answer fixture and the real corpus for each journey. Include
no-match documents, missing dates, uneven lengths, repeated passages, ambiguous
terms, multiple sentences and contradictory evidence. Exported results and
saved/reopened investigations must retain the same scope, measurement rules and
evidence links. Successful tests of individual algorithms do not replace those
whole-workflow checks.

## Verification for this planning review

Against `834da17`, reran phrase research, question integrity and saved-question
tests: **45 passed, 14 deselected**. Phrase helper/static/rendered-workflow tests:
**25 passed**, including 11 rendered interactions. Real-corpus question and API
workflow tests: **29 passed** on a sequential retry (525.60 seconds). The initial
concurrent real-corpus run encountered Windows error `0x8007000e` during dependency
initialization and was stopped; it is not counted as a successful run. These
results verify the selected regression paths, not general production latency.
No application code was changed and no full release/build gate was rerun here.
