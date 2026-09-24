# Question-driven NLP research

Next feature specification:
[Investigate an explanation: source reading, context comparison, contribution analysis, cohorts and research notebooks](RESEARCH_INVESTIGATION_FEATURES.md).
Reviewed against `834da17`; B1 (linked reader and document tracks) and B2
(context comparison and contribution breakdown) are the next concrete product
deliverables. The specification includes researcher questions and acceptance
criteria, with a path from phrase research to broader NLP investigations.

Status: implementation in progress. The first vertical slice now includes literal
phrase occurrence evidence, time/document/relative-position aggregates, stable
project document identity, paginated concordance, verified imported-text offsets
with browser UTF-16 coordinates, and an interactive “Track a phrase” workspace.
Named question sessions preserve their corpus, parser, snapshot, matching rules,
position bins and evidence filters; they can be reopened and survive project
backup/restore with document IDs remapped. A second phrase can be compared over
the same time, position and document denominators, with evidence available for
both subjects. A saved question can now be published through the immutable run
system with its exact documents, parser snapshot, settings, comparison, complete
occurrence evidence, and six reusable CSV tables. Metadata-cohort comparisons
remain a later slice.

Implementation review (2026-09-20):
[current capabilities, confirmed bugs, remaining gaps and ordered acceptance gates](QUESTION_RESEARCH_REVIEW_2026-09-20.md).
The implemented slice is not yet the complete first milestone described below.
Order A is implemented as of 2026-09-20, correctness-wise. Live query
tokenization, atomic load completion, draft/refinement separation, save controls
and cold-session shelf access landed first; F1-F5 of the
[follow-up review](QUESTION_RESEARCH_REVIEW_2026-09-20_FOLLOWUP.md) closed after
them. Publication now matches by the same rules as the workspace and by the saved
question's own recorded tokens; word positions are document coordinates; every
request names the snapshot it belongs to; shelf selection is distinct from the
open record and updates carry an expected revision; and how a question was read
is persisted with it, with an explicit migration rule for questions saved before
that. Two further defects surfaced by testing against the real corpus rather than
fixtures are fixed with them: source offsets failed for every document because
the parser emits whitespace tokens (F6), and a superseded parse stayed in memory
behind a cooled session (F7).

**Still open from R5, and unchanged:** there is no occurrence cache, no
cancellation contract, and no measured size or latency budget; every request
rematches and re-aggregates, and both comparison subjects still share one page
offset. Document grouping and UTF-16 conversion are the two costs that have been
addressed.

Next is Order B: linked position/source exploration, then metadata and date
precision, then cohort and context comparison. Use the follow-up review's
resolution notes rather than treating every future-tense item in this original
design as still unimplemented.

## Dated results and wider matching (2026-09-21)

Two changes from watching a real session, both about a result that is correct
and says nothing.

**Every per-document result now carries the date the corpus already knew.**
`core.profiler.executor.execute` joins `Date` and `Year` onto any frame with a
`Document ID`, when the corpus has dates, and does nothing otherwise. Added in
the executor rather than in each tool so that live and published tables stay
the same rows: a chart drawn over time in the workbench is one that can be
published. Before this, no table this suite produced carried a date, so "did
this change over ninety years" could not be asked of a corpus spanning ninety
years — every chart was a ranking of documents against each other. The
timeline recommendation in `core.insight.recommend` had existed the whole time
and had never had a DATE column to fire on; `core.insight.profile` now reads
an ISO day string and a four-digit `Year` as dates, since a CSV carries no
dtypes and every date read back from a file was a label.

**The live bench reads its own answer.** `/live/analyse` returns the same
`readout` and `recommend_charts` the finished-run pane has always had, and the
chart opens on the first recommendation rather than on the first label column
against the first measure. That default is what put "how many sentences are in
each speech, longest first" on screen as the result of a sentiment analysis.

**A phrase question can be widened, explicitly.** `PhraseQuery` gained
`normalize`, `match_lemma` and `match_nominalization` — each off by default,
each *adding* forms rather than replacing them, so a version 2 question
re-asked under version 3 is the same question. All three resolve to one thing:
which surface forms in this corpus count as each typed token, computed once
per query against the whole table. The answer returns those forms
(`subject.counted_forms`), the published `phrase_question.csv` records them,
and an option that could not be applied is named in `matching_profile.
unavailable` rather than quietly doing nothing.

Matching nominalizations needs NLTK's WordNet corpus, which is an optional
download (`python -m nltk.downloader wordnet`). Without it the option reports
itself unavailable, with the command to fix it, and the rest of the question
answers normally.

## What the product should let a researcher do

Start with a question and keep asking more specific questions of the same evidence:

> How did “public health” change across these speeches? Was its increase spread
> across speakers, or concentrated in a few speeches? Does it occur mainly in the
> opening or the conclusion? What are speakers actually saying around it?

The interaction should preserve that subject and scope while moving between
answers. A researcher should not need to discover several unrelated tools, export
their tables, and manually reconstruct the links between them.

Aggregation remains useful, but it must be reversible: every count, peak and
comparison links to the occurrences and source passages that produced it. A chart
is one view of evidence, not the only surviving representation of that evidence.

## Current foundation and concrete gaps

Original foundation assessment; read with the implementation status above:

- `desktop_backend/live.py` caches parsed corpora in memory/on disk and supports
  repeated analysis through `Bench` and `Session`.
- `desktop/src/LiveBench.tsx` already lets controls drive live analysis.
- Corpus selection and saved views exist. Extend their contracts and storage.
- `core/analysis/ngrams.py` returns document-level n-gram counts and corpus totals.
  It does not return occurrence positions. Its current sequence construction also
  removes nonalphabetic tokens before making n-grams. A new literal phrase matcher
  must not silently inherit that transformation or redefine the legacy output.
- `core/analysis/dispersion.py` supplies document/chunk summaries. Its chunk mode
  splits the concatenated filtered corpus; that is not a per-document relative
  position profile.
- Canonical CoNLL columns identify tokens, sentences and documents, but do not
  require original-character offsets. Exact highlighting needs an explicit offset
  contract and a mapping back to the imported text. The phrase research path now
  provides verified imported-text spans and declared offset units.
- Generic live-tool responses expose the first output frame and at most 500 rows.
  Phrase research now provides multiple linked result sets, with complete aggregation
  separated from paginated evidence. A first-page chart cannot answer a whole-
  corpus question reliably.

Earlier notes in `INTERACTIVE_RESEARCH_PLAN.md` describe historical implementation
stages. In particular, its initial “no persistent annotations/saved views” gap is
superseded by the code above. This plan does not ask for those systems to be rebuilt.

## Research questions as the primary navigation

| Question | Answer to show | Useful next question |
| --- | --- | --- |
| How does this phrase change over time? | Time series with counts, rates, document coverage and exposure | Which documents or speakers account for the peak? |
| Where in a text does it appear? | Per-document occurrence tracks and position profiles | Does the same opening/conclusion pattern recur across texts? |
| Is it widespread or concentrated? | Documents containing it, distribution and concentration | Is the result dominated by one author or one long document? |
| How do these groups differ? | Side-by-side rates and coverage, with group sizes | Does the difference persist within comparable dates or genres? |
| What does it occur alongside? | Context words/phrases by period or group | Are the contexts changing while the phrase stays the same? |
| Who does what to whom? | Entity/relation occurrences with source sentences | Which actions or roles differ between periods or speakers? |

Keep an advanced tool catalog. Add a question-oriented entry point with concrete
examples and explicit controls. Natural-language input can be added later as a
way to propose a structured question; the user must be able to see and edit its
interpretation. A chatbot is not a prerequisite for this product.

## First complete feature: Track a phrase

One workspace, one selected corpus, one subject, four linked panels:

1. **Over time:** count/rate line or columns, calendar grouping, document coverage.
2. **Within texts:** occurrence tracks, one row per document, plus optional aligned
   position bins across documents. Switch between relative position and token or
   sentence position. Named sections follow when section metadata exists.
3. **Across documents:** matching-document counts, rates, lengths and dates. Keep
   nonmatching documents available, because they belong in the denominator.
4. **In context:** paginated concordance, expandable source passages and exact
   highlights when verified character offsets are available.

Clicking a time bin narrows the evidence to that period. Clicking a document opens
its track and passages. Brushing a position range selects occurrences whose start
positions lie in that range; show whether the time chart remains global or is now
conditioned on that brush. Make filters visible, removable and undoable.

Distinguish “inspect these occurrences” from “make these documents my new corpus.”
Drilling into a peak should not silently redefine the denominator of every chart.
Offer the second action explicitly with a before/after scope preview.

Entry points: type a phrase directly, select text in a source passage, or choose
“Track this phrase” from an n-gram result. The phrase need not appear in a top-N
table to be queryable. Allow a small, bounded set of phrases to be compared next.

## Preserve occurrences before summarizing

Introduce a versioned occurrence contract shared by all four panels:

- Corpus snapshot ID, stable project document ID and imported-text content hash.
- Query subject ID and matching-profile version.
- Document-local token start and exclusive end, sentence ID, and source token IDs.
- Original-character start/end when available, with offset unit declared.
- Original surface text; matched normalized form retained separately.
- Document date, date precision/source and metadata revision through a document
  dimension table. Do not duplicate every metadata field on every token.

Never use the run-local, renumbered `Document ID` alone to join independent runs.
Preserve the mapping from parsed document IDs to stable imported documents.
Offsets must refer to the precise imported text version, not reconstructed lemmas.
Declare Python/Unicode versus browser UTF-16 indexing and test their conversion.
For converted PDF/DOCX content, promise offsets into imported extracted text only;
original page-layout coordinates require a separate source mapping.

Store/reuse the token and document evidence, and cache occurrence sets for requested
subjects. Do not materialize every possible n-gram of every length in advance.
Scope matching and aggregation to the snapshot; an index is an optimization, not
a reason to answer from a different corpus.

## Matching and measurement contracts

### What counts as a match

Start with literal token sequences and a clearly separate lemma-sequence mode.
Expose case sensitivity and the punctuation/boundary policy. Default to contiguous
tokens within a sentence and document. Count overlapping matches and state this.
Do not remove stopwords or punctuation and then describe newly adjacent words as
an exact phrase. Offer any such transformation as a distinct, named match mode.

Use a consistent query/document tokenizer. Unsupported tokenization or ambiguous
query interpretation must produce an editable interpretation, not an invented hit.
Later subject types can include entity aliases, POS patterns and grammatical
relations. Each needs a supported matching contract; do not pretend they all mean
the same thing as literal text search.

### What a trend measures

Always return count, denominator and unit together. Initial choices:

- Raw occurrence count.
- Occurrences per 10,000 word tokens, under the declared token-counting profile.
- Percentage of eligible documents containing at least one match.
- Optionally, occurrences per eligible n-gram start position, clearly labeled.

For each time bin return matched count, total word tokens, eligible document count,
matching-document count and date-coverage information. Default pooled rates divide
summed counts by summed exposure; equal-document-weight averages are a separate
choice with a different interpretation. Preserve both when comparing unequal texts.

Create explicit zero-hit bins where text was observed. Bins with no eligible text
have no rate: show a gap, not zero. Undated documents are excluded from a dated
series with a visible count, or displayed separately. Never use import date as the
document date. A year-only date cannot silently become a known January date in a
monthly chart; require adequate date precision or a declared allocation policy.

### What a position profile measures

Retain absolute token positions. Relative position is measured within each
document, never across concatenated documents. For a document with N word tokens,
a match starting at zero-based word position i is located at i/N. Define how token
positions map onto this word-token coordinate system and retain both coordinates.

Use disjoint bins with explicit edges. Assign an occurrence by its start position,
even if it crosses an edge, so it is counted once. Compute exposure in each bin
from all eligible text, including documents with no matches. Show counts and rates
alongside the contributing document count. Token-weighted pooled profiles and an
average of document-level profiles must be selectable and named distinctly.

“Most present” is a choice: most occurrences, highest rate, widest document
coverage, or most concentrated use. Let the researcher choose. Report ties and
avoid labeling a sparsely supported maximum as an established pattern.

## Research session and API shape

Persist a **question specification**, distinct from chart styling:

```json
{
  "schema_version": 1,
  "kind": "phrase_distribution",
  "snapshot_id": "opaque-snapshot-id",
  "metadata_revision": "opaque-metadata-revision",
  "subject": {
    "type": "token_sequence",
    "text": "public health",
    "field": "surface",
    "case_sensitive": false,
    "boundary": "sentence",
    "punctuation": "preserve"
  },
  "measure": "occurrences_per_10000_word_tokens",
  "time": {"unit": "year"},
  "position": {"coordinate": "relative_word_position", "bins": 10}
}
```

Proposed services, integrated with the current authenticated project routes:

- Create/resolve a question against an immutable warmed snapshot; return a query
  handle and versioned interpretation. A new warm must not redirect an old handle.
- Retrieve complete time/position/document aggregates for that handle.
- Page through occurrences with stable ordering and context; resolve source spans
  through project-owned IDs, never arbitrary file paths.
- Save, duplicate and reopen a question session, including visible filters and
  views. Restore backups with explicit document/snapshot ID remapping.
- Publish the resolved answer and its evidence manifest from that same handle.

Responses carry query revision, snapshot identity, effective matching rules,
coverage, totals, denominators and any truncation/sampling. Evidence pagination
does not limit aggregate calculations. Superseded responses cannot overwrite a
newer question. Invalid or evicted handles trigger an explicit re-resolution path.

Changing colours or layout only redraws. Changing time bins reaggregates the same
occurrences. Changing the phrase rematches cached tokens. Changing the parser,
tokenization policy or relevant source content invalidates the affected evidence.
Changing date/group metadata invalidates aggregates even if the text is unchanged.

## Avoid persuasive answers with weak evidence

Every answer should make it easy to check corpus composition: which documents,
speakers, dates and genres contribute, and which are absent. Offer a breakdown by
speaker/genre where metadata supports it. An apparent time trend may reflect a
change in the selected collection; do not describe it automatically as a change
in the wider population, meaning or cause.

Keep exploratory comparisons distinct from pre-specified tests. Formal uncertainty,
change-point claims and significance testing are later, separately validated
methods with explicit assumptions and units of analysis. Do not add automatic
“significant peak” badges to every chart or choose a statistical model invisibly.

Contexts and original passages remain available after every reduction. Model-based
entity, sentiment and relation answers retain method/version and diagnostics;
human review or annotation should be able to flag a misleading match without
destroying the original result.

## Implementation sequence and handoff

| Slice | Deliverable | Main integration points |
| --- | --- | --- |
| 1. Occurrence evidence | Stable document mapping, offset/token contract, literal phrase matching, full denominators, concordance | New core occurrence module; existing parser adapters and live snapshot/cache |
| 2. Track a phrase | Time and within-document aggregates, linked evidence endpoints and question UI | New question service; `live.py`, `server.py`, LiveBench/Workbench and ChartCanvas |
| 3. Research sessions | Saved question/subject/filter state, faithful publication, backup/restore | Existing views/workspace/archives with versioned question records |
| 4. Comparisons | Two phrases or cohorts, metadata grouping, context comparison, explicit weighting | Existing selection plus metadata/cohort contracts; shared occurrence queries |
| 5. Broader NLP questions | Entity/action trajectories, sentiment around a subject, section-aware comparisons | Additional typed subjects and validated adapters using the same evidence chain |

Build slices 1–2 as one usable vertical milestone. Do not deliver only an index or
another n-gram table. Agree on the occurrence and question schemas before splitting
backend/frontend implementation. Preserve legacy tool behavior; adapt it to the new
question workflow only where matching and measurement semantics actually agree.

## Acceptance tests for the first milestone

1. A fixture with known phrase offsets and dates yields exact occurrence counts;
   every plotted point drills down to those same occurrences and source text.
2. A phrase absent from the displayed top-N table remains directly queryable.
3. Overlapping sequences count correctly; sentence/document boundaries never join;
   punctuation and stopword policy cannot fabricate a literal occurrence.
4. Unequal-length documents and periods produce the expected pooled rates and
   document prevalence; zero matches and no observed text remain distinct.
5. Year-only, undated and mixed-precision dates cannot masquerade as exact monthly
   observations. Unsupported granularity is explained before displaying a curve.
6. Two texts with the same phrase near their ends but very different lengths align
   in relative-position views; token-weighted and document-weighted answers differ
   only according to the selected, tested rule.
7. A match crossing a position-bin boundary counts once; empty/short documents and
   single-token texts do not cause division errors or disappear without accounting.
8. Evidence exceeding 500 rows retains correct full-result aggregates and stable
   pagination. A fixed global aggregate does not change when evidence is paged.
9. Duplicate filenames, Unicode including non-BMP characters, converted text and
   repeated identical phrases still resolve to the correct document/span.
10. Switching time resolution or appearance does not call the parser. Query edits,
    new warming, cache eviction and metadata changes never mix stale answers.
11. Save/reopen and published exports retain the same question, scope, matching
    rules, counts, units and evidence links. Filters cannot disappear at publication.
12. Test at the agreed small/medium/large corpus sizes, measuring cold preparation,
    warm query p50/p95, memory and cancellation. Set supported size/latency budgets
    from that evidence; do not promise near-instant answers for every NLP method.

Definition of done: a researcher can follow a phrase from a whole-corpus pattern
to its dates, document positions and passages, revise the question, and preserve
the answer without losing what was measured or where it came from.
