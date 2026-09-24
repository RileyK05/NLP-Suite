# Full Replacement Plan

This is the post-scaffold roadmap for turning `nlp-suite-ng` into a genuine
replacement for NLP Suite 1.6.38. `BUILD_PLAN.md` and `CHUNK_LEDGER.md` describe
the first 52 construction chunks. Those chunks established the new system's
shape; they did **not** establish complete feature or product parity.

"Replacement" means the new suite does the legacy suite's job as a class
tool — a student runs their whole corpus-analysis workflow in it — while
actively improving on the legacy: no reproduced defects, no silent failures,
no GUI-owned sequencing. It does **not** mean bug-for-bug output parity with
the old code. Where the new suite fixes a legacy defect, the divergence is
intentional and recorded; it is not a parity failure.

The implementation workflow is deliberately simple:

- a low-cost implementation model completes one numbered packet;
- it supplies tests and verification evidence with the change;
- a separate reviewer checks behavior, architecture, and parity;
- only the reviewer moves a packet from `review` to `verified`.

Live packet and review status is tracked in `REPLACEMENT_LEDGER.md`.
Roadmap items are delivered through the repeated-prompt execution chunks in
`IMPLEMENTATION_CHUNKING_PROTOCOL.md`; they are not one-shot model prompts.

The legacy tree at `../NLP-Suite-1.6.38/` is always read-only. It is a behavior
and feature oracle, not an architecture to copy.

## 1. What “fully replaced” means

The replacement is complete only when all five gates below are satisfied.

### Gate A: capability parity

Every user-visible legacy capability is in exactly one final state:

- `verified`: implemented, and its evidence accepted by the reviewer (see
  Gate B for what counts as evidence);
- `replaced`: a documented new workflow provides the same user outcome;
- `dropped`: explicitly approved, with a reason and migration advice.

`partial`, `stub`, `interface only`, and `tests pass` are not final states. A
deterministic stand-in is useful scaffolding, but it is not an NLP result and
must never be presented as the named legacy analysis.

### Gate B: behavioral evidence (stratified — owner decision 2026-09-03)

Most of the legacy suite is a thin wrapper over researcher-built packages
(textstat, scipy, Gensim, NLTK, spaCy). For those analyses the package plus
its published formula *is* the domain specification, and re-running the old
GUI to capture its numbers proves only that two callers can call the same
library. Evidence is therefore stratified by what is actually at risk:

1. **Definitional evidence (default).** For analyses backed by a known
   package or a published formula: hand-computed expected values over
   representative fixtures, checked against the independent reference (the
   formula spec or a direct call to the backing library), plus named legacy
   edge cases. The legacy tree is read as a specification — options, entry
   points, defect IDs — to make sure the port covers what users asked for
   and avoids the catalogued defects.
2. **Legacy-run spot-check goldens (glue-heavy areas only).** Where the
   legacy adds its own behavior on top of packages — tie-breaking, rounding
   conventions, row ordering, empty-input handling, multi-step workflows —
   a captured legacy output is worth more than reading code, because the
   legacy code provably does not do what it appears to do (see the 761
   self-documented defects). Spot-checks are captured when the py3.10
   legacy runtime exists (FR-1.4); their absence never blocks verification.
3. **`SPEC_ONLY` contracts.** For broken or unreachable legacy paths, define
   behavior from documentation and accepted domain formulas.

Every retained capability has a representative fixture, evidence from tier 1
(plus tier 2 where the dossier calls for it), or a `SPEC_ONLY` entry — and a
record of intentional differences caused by fixing legacy defects.

The "not its own oracle" rule applies within each tier: a unit test written
from the same assumptions as the implementation proves nothing. A
hand-computed expectation or an independent library call is an outside
oracle; a test that merely repeats the implementation's formula is not
evidence, and reviewers reject it as before.

### Gate C: nontechnical user workflow

A user who does not know Python can install the suite, select a corpus,
configure and run an analysis, see progress and actionable errors, inspect the
results, and reopen a previous run. The CLI remains supported, but it is not
the only complete workflow.

### Gate D: operational readiness

The supported Windows and macOS installation paths work from clean machines.
Models and optional components have explicit install/check commands. Nothing
downloads or opens a GUI at import time. A missing dependency fails before an
analysis starts and includes exact remediation.

### Gate E: release readiness

The license and attribution are present, documentation matches reality, the
quality gate is green, a 100+ document corpus completes the acceptance suite,
and a release artifact can be reproduced from a clean checkout.

## 2. Rules every packet must preserve

1. Core logic lives under `core/`; interfaces live under `tools/` and `app/`.
2. Imports never download models, touch the network, create GUI roots, mutate
   the working directory, or install packages.
3. Public runtime failures use `Result[T]` and structured `Diagnostic`s.
4. Analysis consumes name-addressed canonical CoNLL columns, never positions.
5. Inputs are immutable. Outputs live in a new run directory and are recorded
   in `result.json`.
6. Randomized algorithms accept and record a seed.
7. Network services and heavyweight models are lazy optional backends;
   offline tests use fakes or recorded fixtures.
8. Do not add a miniature baked lexicon or hash vector and label the feature
   complete. Test-only fakes must be named as fakes and stay in tests.
9. One implementation per concept. Shared behavior belongs in `core/`, not in
   copied CLI functions.
10. A packet never expands its own scope. Record newly discovered follow-up
    work instead of bundling it into the current change.

## 3. Status vocabulary and ledger

| Status | Meaning |
|---|---|
| `todo` | scoped but not started |
| `implementing` | one implementer owns the packet |
| `review` | implementation and evidence are ready for independent review |
| `verified` | reviewer accepted code, tests, and parity evidence |
| `blocked` | an external prerequisite is named and evidenced |
| `dropped` | the owner approved removal and migration guidance |

Each capability row must record its packet ID, user outcome, legacy entry
point and files, new core/CLI/UI surfaces, fixture and evidence
(definitional / spot-golden / spec-only) reference, intentional
differences, dependencies/assets, status, implementation commit, and review
commit.

Start with `LEGACY_PARITY.md`, but split combined rows when they hide different
outcomes. For example, spaCy parsing and the CoreNLP stub cannot share one
final status.

## 4. Roadmap-item and dossier contract

A numbered `FR-x.y` item is a curriculum unit, not automatically a single
implementation prompt. Before delegation, the professor classifies it and
writes a just-in-time dossier. The dossier splits the item into execution
chunks such as reconnaissance, oracle preparation, tests, core implementation,
integration, and verification. Follow `IMPLEMENTATION_CHUNKING_PROTOCOL.md`.

A small item may change one core module, its CLI adapter, its tests, and one
ledger row. Medium items take several prompts and may use several coherent
commits. Large items are decomposed into smaller roadmap items before any
production implementation begins.

Every roadmap-item dossier contains:

```text
Roadmap item: FR-x.y — short name
Outcome: user-visible result that must work
Legacy references: exact old files/functions and defect IDs
Inputs/outputs: types, columns, artifacts, and diagnostics
Fixtures/evidence: exact paths, expected values, and each value's origin
  (hand computation, independent library call, spot-check golden, or spec)
Dependencies: core or named optional extra
Allowed files: explicit files or narrow directories
Out of scope: related work this packet must not attempt
Acceptance commands: exact commands the implementer must run
```

Every execution-chunk handoff briefly records the chunk, files, commands, and
remaining work. The final implementation handoff contains:

```text
Roadmap item implemented:
Files changed:
Behavior added or changed:
Legacy behavior checked:
Intentional differences:
Tests added:
Commands run and exact results:
Known limitations/follow-up packets:
Commit:
```

Missing evidence means the item remains `implementing`, even if the code
looks plausible.

## 5. Execution roadmap

The phases are ordered by risk and dependency. Send only one execution chunk
in an implementation turn, using the state block and stop conditions from
`IMPLEMENTATION_CHUNKING_PROTOCOL.md`. Reorder roadmap items only when their
prerequisites are already `verified`.

### FR-0: stabilize the baseline

- [ ] **FR-0.1 — Review and commit the current working tree.** Separate source,
  tests, documentation, and the untracked corpus into understandable commits.
- [ ] **FR-0.2 — Restore the complete quality gate.** Make `pytest`,
  `ruff check`, `ruff format --check`, strict `mypy`, and `compileall` pass in
  one documented command on Windows and CI.
- [ ] **FR-0.3 — License and attribution.** Add the correct top-level license,
  upstream attribution, third-party notices, and asset/model license rules.
- [ ] **FR-0.4 — Reconcile documentation.** Remove stale claims, distinguish
  scaffold completion from parity, and make this the canonical remaining plan.
- [ ] **FR-0.5 — Enforce write custody.** Route converter, merger,
  schema-sidecar, and envelope writes through the writer contract, or revise
  the invariant to an accurately enforceable boundary. Add an automated gate.
- [ ] **FR-0.6 — CI and platform matrix.** Add supported Python/OS jobs without
  downloading NLP models during ordinary unit tests.

Exit gate: clean documented baseline, legally distributable source, all
quality checks green, and no unexplained working-tree changes.

### FR-1: make parity measurable

- [ ] **FR-1.1 — Atomic capability inventory.** Expand `LEGACY_PARITY.md` into
  one row per user-visible operation, option, and output family.
- [ ] **FR-1.2 — Fixture corpus.** Implement the fixtures specified in legacy
  `planning/01_fixture_corpus_spec.md`: empty, Unicode, malformed, multi-dot,
  dated, long-sentence, non-English, CoNLL, and PC-ACE cases.
- [ ] **FR-1.3 — Freeze the legacy environment.** Record legacy commit,
  Python/package/model versions, external binaries, and OS constraints.
- [ ] **FR-1.4 — Spot-check golden capture (owner-scoped).** Run
  glue-heavy legacy capabilities one at a time under the py3.10 legacy
  runtime, normalize nondeterministic values, retain failures, and never
  modify inputs. Scope is spot-checks for areas where legacy behavior
  cannot be read off a formula (see Gate B tier 2) — per-capability golden
  capture for the whole inventory was dropped by owner decision
  2026-09-03. Still blocked on a working py3.10 legacy runtime
  (`src` unimportable here: no nltk/textstat; deps 2/77 pinned), but that
  block no longer gates verification of any other packet.
- [ ] **FR-1.5 — Comparison harness.** Declare ordering, numeric tolerance,
  schema, missing-value, and artifact-data comparison rules.
- [ ] **FR-1.6 — SPEC_ONLY registry.** For broken or unreachable legacy paths,
  define behavior from documentation and accepted domain formulas.
- [ ] **FR-1.7 — Seed spot-check goldens.** Capture and review
  glue-heavy outputs (parsing, CoNLL division/normalization, file
  operations with edge-case behavior, multi-step intake workflows) once
  FR-1.4 unblocks. Foundation text-statistics already carry definitional
  evidence and need no legacy capture.

Exit gate: every inventory row points to definitional evidence (hand-computed
values and/or an independent library reference), a spot-check golden where
the dossier requires one, or an approved `SPEC_ONLY` entry; the comparer
passes against a self-copy of all captured spot-checks.

### FR-2: finish high-value deterministic analysis

- [ ] **FR-2.1 — Readability.** Flesch Reading Ease, Flesch-Kincaid, Gunning
  Fog, Coleman-Liau, SMOG, ARI, and remaining retained formulas.
- [ ] **FR-2.2 — Lexical diversity.** Add Guiraud, MTLD, vocd-D, and other
  retained measures with explicit short-text behavior.
- [ ] **FR-2.3 — Statistical tests I.** Crosstabs, chi-square, effect sizes,
  and log-likelihood/keyness.
- [ ] **FR-2.4 — Statistical tests II.** Mann-Whitney, Kruskal-Wallis, Dunn,
  and multiple-comparison correction.
- [ ] **FR-2.5 — Statistical tests III.** Correlation options and Mann-Kendall
  trend analysis.
- [ ] **FR-2.6 — Sentence complexity.** Dependency distance, subordination,
  Yngve, and Frazier with documented parser requirements.
- [ ] **FR-2.7 — Nominalization.** Real WordNet-backed deverbal detection.
- [ ] **FR-2.8 — Style analysis.** Concreteness, abstractness, and iconicity
  using licensed full assets.
- [ ] **FR-2.9 — Semantic similarity.** Similarity, plagiarism, duplicate,
  and unrelated-document workflows.

Exit gate: retained deterministic analysis rows are `verified`, registered in
CLI/UI, and available to the profiler.

### FR-3: complete file intake and preprocessing

- [ ] **FR-3.1 — Document conversion.** PDF, DOCX, RTF, HTML, CSV/TSV, and TXT
  conversion with format diagnostics and no input mutation.
- [ ] **FR-3.2 — Spell checking.** Detection and optional correction with the
  original preserved and language dictionaries explicit.
- [ ] **FR-3.3 — Matching and duplicates.** Exact, normalized, and fuzzy
  matching with reproducible thresholds.
- [ ] **FR-3.4 — Structured search.** Search TXT, CSV, and canonical CoNLL
  while preserving document/sentence provenance.
- [ ] **FR-3.5 — Filename operations.** Standardization, date extraction,
  duplicate handling, and preview/dry-run before changes.

Exit gate: common legacy input formats reach canonical corpus/CoNLL form
through the nontechnical workflow without destructive edits.

### FR-4: real lexicon and semantic resources

- [ ] **FR-4.1 — Asset registry.** Versioned assets with checksums, licenses,
  lazy loading, and actionable missing-asset diagnostics.
- [ ] **FR-4.2 — Full sentiment resources.** Replace embedded sample VADER,
  ANEW, SentiWordNet, and hedonometer dictionaries and verify scoring.
- [ ] **FR-4.3 — NRC emotions.** Full NRC categories and Plutchik-compatible
  output.
- [ ] **FR-4.4 — WordNet aggregation.** Real synset/hypernym traversal with
  sense and POS handling.
- [ ] **FR-4.5 — VerbNet and FrameNet.** Load full licensed mappings and
  preserve ambiguity instead of selecting an arbitrary first match.
- [ ] **FR-4.6 — Symbolic/social-actor typologies.** Port retained legacy
  dictionaries into the asset registry.

Exit gate: no production result is based on a tiny baked sample presented as
the named external lexicon or ontology.

### FR-5: real parser and model backends

Each backend is an optional extra with a doctor check, install command, lazy
cache, fake-backed unit tests, and separately marked model integration tests.

- [ ] **FR-5.1 — Stanza production backend.** Verify tokenization, POS, lemma,
  dependencies, and NER across supported-language fixtures.
- [ ] **FR-5.2 — CoreNLP backend.** Server client, health check, timeouts,
  canonical conversion, and documented lifecycle.
- [ ] **FR-5.3 — Semantic role labeling.** Replace the failure-only backend
  with a maintained implementation or approved replacement.
- [ ] **FR-5.4 — Neural coreference.** Real mention/cluster output with spans
  and document provenance.
- [ ] **FR-5.5 — Word2Vec.** Gensim training/loading, distances, seeds, and
  small-corpus guards.
- [ ] **FR-5.6 — Contextual embeddings and WSI.** Transformer embeddings,
  batching, device selection, clustering diagnostics, and cached models.
- [ ] **FR-5.7 — Topic modeling.** Real Gensim LDA, coherence, and stable
  topic artifacts.
- [ ] **FR-5.8 — MALLET adapter.** Executable discovery, safe argv invocation,
  version capture, and actionable failures.
- [ ] **FR-5.9 — BERT analyses.** Split retained summarization, sentiment, and
  topic workflows into distinct capability rows.

Exit gate: hash/pseudo-topic/lemma-cluster stand-ins are removed from
production or explicitly renamed as demos; real backends are selectable and
provenance-stamped.

### FR-6: domain workflows and visualization

- [ ] **FR-6.1 — Clause and enhanced SVO analysis.** Parser-specific
  dependencies, clause taxonomy, negation, auxiliaries, and coordination.
- [ ] **FR-6.2 — Character emotion arcs.** Character/coreference selection,
  binning, smoothing, comparison, and export.
- [ ] **FR-6.3 — Narrative/story shapes.** Real vectorization and seeded
  clustering with interpretable artifacts.
- [ ] **FR-6.4 — Entity timelines and movement.** Multi-token entities,
  geocoding custody, and map-ready output.
- [ ] **FR-6.5 — Charts.** Complete retained families and static/export
  decisions while keeping chart data independently testable.
- [ ] **FR-6.6 — Network/Gephi output.** Validate GEXF schema, spells, weights,
  IDs, and reproducible styling.
- [ ] **FR-6.7 — GIS.** Injectable online geocoder, bounded retry/cache,
  offline mode, KML, heatmaps, distance, and symbolic layers.
- [ ] **FR-6.8 — Knowledge graphs.** Injectable DBpedia/YAGO/Wikipedia clients,
  TLS, request limits/cache, and recorded offline tests.
- [ ] **FR-6.9 — PC-ACE.** First split the legacy workflow into validation,
  import, query, aggregation, and visualization packets.

Exit gate: retained domain workflows produce complete viewer-renderable
artifacts without hidden live-service dependencies.

### FR-7: rebuild the profiler as the integration product

- [ ] **FR-7.1 — Tool registry.** Declarative capability, parameter,
  prerequisite, output, and profiler metadata.
- [ ] **FR-7.2 — Envelope composition.** Consume declared artifacts/envelopes,
  never files discovered through globbing.
- [ ] **FR-7.3 — Profiler execution plan.** Select analyses, share one parse,
  isolate failures, and record timings/versions.
- [ ] **FR-7.4 — Complete profiler set.** Restore every retained analysis from
  the legacy 31-analysis batch.
- [ ] **FR-7.5 — Resume and cache.** Reuse only when input, parameters, schema,
  backend, model, and asset versions match.
- [ ] **FR-7.6 — Profiler report.** Link every child run and diagnostic without
  hiding partial failures.

Exit gate: one parse supports a complete selected batch, one failed analysis
does not erase successes, and every reuse decision is auditable.

### FR-8: deliver the no-programming experience

- [ ] **FR-8.1 — App structure.** Home, new analysis, batch profiler, jobs,
  results, setup/doctor, and help.
- [ ] **FR-8.2 — Corpus selection/validation.** Preview files, encodings, size,
  language, output location, and path warnings.
- [ ] **FR-8.3 — Declarative tool forms.** Generate forms from the registry
  with human labels, defaults, validation, and help.
- [ ] **FR-8.4 — Background jobs.** Run outside Streamlit rendering, persist
  progress, support safe cancellation, and survive refresh.
- [ ] **FR-8.5 — Results.** Tables, charts, maps, reports, diagnostics,
  downloads, provenance, and reopenable history.
- [ ] **FR-8.6 — Guided setup.** Doctor, model/asset installation, remediation,
  and a sample-data first run.
- [ ] **FR-8.7 — Usability acceptance.** A target user completes
  install-to-result without a shell or undocumented knowledge.

Exit gate: the accepted usability scenario requires no command, while UI and
CLI both use the same tested core APIs.

### FR-9: package and release

- [ ] **FR-9.1 — Unified command.** `nlp-suite doctor`, `models`, `run`,
  `profile`, and `ui`, while preserving module entry points.
- [ ] **FR-9.2 — Extras.** Test `app`, `spacy`, `stanza`, `corenlp`, `topics`,
  `embeddings`, `gis`, `pcace`, and `all`.
- [ ] **FR-9.3 — Model/asset management.** List, install, verify, update, and
  report versions without import-time downloads.
- [ ] **FR-9.4 — Clean-machine installers.** Reproducible Windows and macOS
  install/launch paths, including uninstall and upgrade.
- [ ] **FR-9.5 — Migration guide.** Map every legacy GUI/output family to its
  new workflow and intentional differences.
- [ ] **FR-9.6 — Security/privacy review.** Secrets, TLS, subprocess, path and
  archive traversal, HTML, SQL, corpus privacy, and telemetry.
- [ ] **FR-9.7 — Performance acceptance.** A 100+ document corpus, shared model
  cache, bounded memory, timings, and no accidental quadratic paths.
- [ ] **FR-9.8 — Release-candidate audit.** Every retained row verified, every
  dropped row approved, docs current, and build reproducible.

Exit gate: a signed-off release candidate satisfies Gates A–E.

## 6. Independent review procedure

The reviewer checks more than whether tests are green.

### Contract and parity

- Does the API match the packet and handle empty, partial, malformed,
  multilingual, and dependency-failure cases?
- Are diagnostics actionable and stable enough to test?
- Does the envelope record inputs, parameters, backend/model/asset versions,
  seed, outputs, and warnings?
- Was the named legacy code actually inspected?
- Is each expected value independent of the new implementation — a hand
  computation, an independent library call, or (tier-2 glue questions only)
  a captured legacy output?
- Are ordering, rounding, filtering, missing values, and grouping differences
  intentional and documented?

### Architecture and safety

- No UI/CLI logic in `core/`, input mutation, unsafe output path, secret,
  `eval`, shell string, global TLS override, import-time work, or silent model
  fallback.
- Optional dependencies stay optional and load lazily.
- Network/subprocess calls have timeouts, bounded retries, safe arguments, and
  test doubles.

### Evidence

Run packet tests and the complete project gate. Inspect generated artifacts,
not only return codes. Reject tests that merely assert non-empty output or
repeat the implementation formula without an independent oracle — except
hand-computed definitional fixtures against a published formula or a direct
independent library call, which are evidence of the first rank (Gate B
tier 1). Legacy-run goldens are expected only where the dossier named a
tier-2 glue question; their absence is not a rejection reason.

Review outcomes are `accepted`, `changes requested`, or `blocked by named
prerequisite`. Record the outcome in the ledger.

## 7. Bootstrap prompt for a roadmap item

Use this once to begin an item. After the model acknowledges it, send only C0
from `IMPLEMENTATION_CHUNKING_PROTOCOL.md`; do not combine the later prompts.

```text
Begin roadmap item FR-x.y from docs/FULL_REPLACEMENT_PLAN.md.

Before editing, read:
- sections 1–4 and the complete FR-x.y packet in that document;
- relevant sections of docs/ARCHITECTURE.md;
- docs/LEGACY_PARITY.md for the capability;
- the exact named legacy files, read-only;
- the existing new module and tests.

Rules:
- work only on this roadmap item and its current execution chunk;
- never edit the legacy repository;
- do not call a stub, sample lexicon, or fake model complete;
- preserve Result/Diagnostic, canonical CoNLL, envelope, and import-safety;
- add independent tests and tool-completeness evidence (Gate B tier 1 by
  default; tier-2 spot goldens only where the dossier names a glue question);
- follow `docs/IMPLEMENTATION_CHUNKING_PROTOCOL.md`;
- do not implement during read-only chunks;
- do not mark the item verified or start another item.

Stop after acknowledging the assigned item and wait for the C0 prompt.
```

A missing legacy-run golden never blocks: use definitional evidence
instead. If a required model, licensed asset, service, or product decision
is unavailable, the model must not invent a substitute. It records the exact
blocker and stops.

## 8. Recommended queue (tool-completeness order, owner decision 2026-09-03)

Replacement is decided by class-workflow completeness: first the analyses
students run, then the batch that runs them, then the UI that runs the
batch. Legacy-run goldens are no longer on the critical path.

1. FR-4.4 — WordNet aggregation (unblocks FR-2.7 nominalization and real
   semantic aggregation);
2. FR-5.7 — real Gensim LDA topic modeling (currently a TF-IDF stand-in);
3. FR-5.5 — Word2Vec backend (currently a hash stub);
4. FR-4.2/FR-4.3 — full sentiment resources and NRC emotions;
5. FR-7.4 — complete profiler set (legacy 31-analysis batch);
6. FR-7.2/FR-7.3 — envelope-consuming profiler with a shared-parse plan;
7. FR-8.3 — declarative tool forms in the app (the class-facing workflow);
8. FR-2.8 — style analysis (needs FR-4.1 licensed assets).

FR-1.4/FR-1.7 (spot-check capture) proceed opportunistically once a py3.10
legacy runtime exists; they gate nothing. FR-5.2/FR-5.3/FR-5.4
(CoreNLP/SRL/neural coreference) stay sequenced after the class
critical path unless a course needs them first.
