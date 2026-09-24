# Tool-specific visualizations and deletion

Planning review, 2026-09-22. This plan covers the two product gaps that remain
visible in the desktop: figures that fit the *question and tool*, and a way to
remove unwanted documents, runs, and projects. Work is sequenced as small,
complete user journeys rather than a new chart picker or a bare delete API.

Implementation update, 2026-09-22: the first six slices now exist. Culturomics
has an annual line with explicit corpus exposure, zero-hit years, smoothing,
and raw counts. Gensim LDA's existing specialist panels and new MALLET topic
mixtures are available in the panel-first reader. Word2Vec has neighbour and
labelled t-SNE panels. Sentiment has ranked document evidence and annual
document-score lines where dated documents are available. Explore, Past runs,
and Live analysis now lead with a declared specialist panel when one fits;
the generic chart workbench is an expert option. Documents, runs, and projects
can be moved to Trash, restored, and permanently purged. Project backup format
v4 preserves trashed documents and runs while older archives remain readable.

The remaining work is the cross-tool coverage inventory and artifact-schema
manifest below, lexicon temporal views, further source-context links and
coverage/seed explanations for embeddings, richer sentence-level sentiment,
saved panel choices/parameters and export, and failure-injection checks for
filesystem rollback. These are scope boundaries for this delivery, not claims
that the earlier acceptance gate is complete for every public tool.

1. **Make analysis figures answer the question their tool actually asks.**

   **What exists:** `Workbench` in `desktop/src/Workbench.tsx` exposes the same
   seven live chart kinds for every numeric result table. Its fallback is a bar
   chart of the first label and measure. That remains useful for an ordinary
   frequency table, but it is not a valid default for a topic model, vector
   space, text position, or time series. `core/viz/panels.py` already has a
   validated panel registry with evidence and provenance; the desktop currently
   places `PanelSection` under Past runs, while Interactive and Explore still
   lead with the generic workbench. Do not add more kinds to `LIVE_KINDS` to
   solve this: reuse the panel contract and let ordinary charts remain available
   only where their columns and meaning support them.

   **Build order:**

   - Introduce one result-view manifest, keyed by *public tool + artifact
     schema*, declaring its primary question, preferred panel(s), permissible
     generic charts, needed columns, units, and an explanation when a panel
     cannot be drawn. The server should publish the effective choices for each
     run; the client must not maintain a second tool-to-chart map. Validate
     this manifest against `desktop_backend/catalog.py` and the actual result
     headers. Older/partial runs get a truthful table and an explanation, never
     a guessed chart or a blank space. If a tool has several artifacts, match
     each by its own schema rather than taking the first CSV.
   - First vertical slice: `ngram_viewer` (culturomics). Lead with a year-based
     line chart, one line per queried n-gram, defaulting to **Per Million**;
     expose Count and Share of Documents as explicit metric choices. Show the
     date range, corpus exposure and smoothing window, with click-through to
     the contributing years/documents. The current `ngram_series.csv` writes
     only years with a hit. Extend the analysis output to include zero-hit
     years *with dated documents* and the relevant denominators; show years
     without corpus coverage as gaps. Specify whether smoothing includes
     zero-hit years, and make the raw series recoverable. Never offer an
     arbitrary scatterplot for this artifact. Apply the same temporal
     semantics to `lexicon_series` when its axis is year or decade; its
     document/pattern modes require a different view.
   - Complete topics: retain the existing Gensim LDA relevance, intertopic,
     prevalence, flow and stability panels, but make them the primary views in
     Interactive, Explore and Past runs. Add an honest MALLET view for its
     `topics.csv` and `topics_dominant.csv`; do not assume Gensim-only
     relevance, distance or flow artifacts exist. Link a selected topic to its
     terms and contributing documents. Label distance projections as
     projections, not as measured topic similarity.
   - Complete embeddings: for public `word2vec_gensim` and `word2vec_bert`,
     pair nearest-neighbour rankings and cosine scores with a labelled
     `tsne.csv` projection. Selecting a word must connect its neighbours and
     underlying source context; report vocabulary coverage, model/seed and
     projection limits. A 2-D plot must not imply that planar distance is the
     original vector distance. BERT sentiment tools are **sentiment** tools:
     show sentence/document tone over position or time, distributions and
     source passages, not an embedding scatterplot merely because a
     transformer produced the scores. `bert_topics` and `bert_extract` are
     internal in `desktop_backend/catalog.py`; plan a desktop contract before
     exposing either.
   - Extend by question family: dispersion/concordance to position tracks and
     readable hits; sentiment to timelines and sentence passages; entities and
     subject-verb-object analysis to linked actors, actions and source
     sentences; similarity/duplicates to pairwise comparison; keyness and
     collocations to their existing dedicated panels. A compact coverage
     inventory for every **public** tool should state primary view, useful
     secondary view, evidence link, and whether table-only is the honest
     result. Table-only is better than a decorative plot.
   - Save panel choice and parameters with provenance in saved views, support
     re-opening and export, and let visual changes reuse completed results.
     When a change truly requires recomputing (e.g. retraining an embedding),
     label it as a new analysis. Keep explicit expert access to generic charts
     for suitable tidy tables, but do not make them the default or offer
     meaningless kinds for a specialist result.

   **Acceptance gate:** a culturomics run opens on a time series and cannot
   silently imply values for unobserved years; an LDA run opens on its topic
   views; a Word2Vec run opens on neighbour evidence plus a clearly labelled
   projection; a BERT sentiment run opens on sentence/document tone. Each
   visible mark resolves to source rows or passages, exported figures carry
   units and provenance, partial/old artifacts degrade to an explained table,
   and no public tool opens with a nonsensical default. Add engine-fixture
   tests for each panel and desktop tests for the view selection/fallback.

2. **Let people delete documents, runs, and whole projects without losing track of what depended on them.**

   **What exists:** projects can be *archived* (`Workspace.archive` and
   `ProjectManager`), and saved views/questions can be deleted. There is no
   document, run or project deletion path in `Workspace` or the desktop API.
   Imported documents have a normalized text copy in `corpus/` and, for
   converted formats, an original in `originals/`. A job may have a run
   directory, staged files in `job-inputs/`, and a log under workspace
   `logs/`. Saved views refer to job IDs; saved questions hold document IDs;
   the backup manifest includes all of them. Removing only a row or only a
   file would leave confusing or unrestorable state.

   **Build order:**

   - Define a single lifecycle for each resource: active, in Trash, restored,
     permanently removed. Archive stays a separate reversible way to hide a
     project. Put a visible Delete action beside each corpus document and
     past run, plus Delete project in Manage projects. Confirmation must name
     the resource and summarize affected saved questions/views and source
     reading. Offer Restore from Trash; offer permanent removal to reclaim
     disk space. Avoid an unexplained typed-name ritual for every document.
   - Implement deletion in `Workspace` and API under `runner.lock`. Reject or
     cancel-and-settle active jobs before touching their project or run; never
     delete a directory while a worker can write to it. Resolve every target
     from its project ID and stored metadata, verify it remains under the
     workspace root, and refuse symlinks/escaped paths. Move files to a
     project-scoped trash staging location, update SQLite transactionally,
     and recover or roll back if either side fails. Permanent purge removes
     the normalized text, original, run artifacts, staged job inputs and run
     log as appropriate. Never accept a client-supplied filesystem path.
   - Document deletion changes the *current corpus*, not published result
     numbers. Invalidate the warm/live corpus session and active selections;
     existing run tables remain labelled as historical output. Saved questions
     that named the document must be shown as needing scope repair or offered
     an explicit scope update; they must never silently run over fewer
     documents. Source-passage links to a removed document must say why the
     source is unavailable. Restoring the document restores the same ID and
     question scope. Permanent purge explains that past-run source reading
     cannot be reconstructed from an aggregate table.
   - Run deletion removes the run from Past runs and Explore tables, clears
     selected-run/chart state, and reports saved views or comparisons that
     reference it before removal. Preserve saved view settings as visibly
     missing-source or let the user remove them in the same confirmation;
     never silently point a view at another run. Verify derived chart jobs
     and frozen `job-inputs` references before purging a source run.
   - Project deletion covers its documents, runs, saved views/questions,
     staged inputs and project-scoped files, updates the selected-project
     state, and leaves unrelated projects and global resources alone. Include
     archived projects in this path. Add Trash to backup/restore deliberately:
     either encode trashed rows/files and repairable question references in
     a versioned manifest, or exclude them with an explicit backup scope.
     The current v3 restore validator requires every saved question's document
     IDs to exist, so this cannot be skipped. Verify a backup made before and
     after deletion restores exactly what its manifest promises.

   **Acceptance gate:** delete and restore one document, a converted document
   with original, a completed run, a failed run with a log, and an entire
   archived project; then permanently purge each. A current analysis and a
   saved question cannot silently include a deleted document; no active job
   can recreate purged files; one deletion never touches another project's
   data. Inject a filesystem failure and confirm metadata/files recover
   coherently. Backup/restore and old workspaces still work, with clear
   behavior for trashed items and dangling saved views.

The first delivery should finish the culturomics vertical slice and document
deletion (including restore and backup behavior) before broadening either
track. Those two journeys expose the shared contracts and give users visible
improvements while the remaining tool families and run/project deletion are
built out.
