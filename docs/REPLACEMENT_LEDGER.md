# Full Replacement Ledger

Live status for the work packets in `FULL_REPLACEMENT_PLAN.md`. This ledger is
about remaining replacement work; `CHUNK_LEDGER.md` records the completed
52-chunk scaffold.

Latest targeted correctness review: [2026-09-05 follow-up](FOLLOWUP_REVIEW_2026-09-05.md).
This records fixes and verification limits, not full-replacement sign-off.

## Rules

- An implementer may move only its assigned row from `todo` to `implementing`
  and then to `review`.
- Only the independent reviewer may set `verified`.
- `blocked` must name the missing prerequisite in Notes.
- `dropped` requires owner approval and migration guidance.
- Use full commit hashes once work is committed. Do not write `latest` or a
  branch name in a commit column.
- Each roadmap item is executed through the repeated prompts in
  `IMPLEMENTATION_CHUNKING_PROTOCOL.md`. Track those chunks in the item's
  dossier; keep this table at roadmap-item granularity.
- FR-1.1 will add the atomic capability-parity rows beneath this packet ledger.

## Summary

| Phase | Verified | Total | Exit gate |
|---|---:|---:|---|
| FR-0 Baseline | 0 | 6 | clean, legal, documented, green baseline |
| FR-1 Evidence | 0 | 7 | every capability has definitional evidence or approved spec |
| FR-2 Deterministic analysis | 0 | 9 | retained deterministic analyses verified |
| FR-3 File intake | 0 | 5 | retained formats reach canonical input safely |
| FR-4 Assets | 0 | 6 | no sample resource masquerades as production |
| FR-5 Models | 0 | 9 | real optional model backends replace stand-ins |
| FR-6 Domain workflows | 0 | 9 | retained workflows emit complete artifacts |
| FR-7 Profiler | 0 | 6 | full auditable shared-parse batch |
| FR-8 User experience | 0 | 7 | install-to-result without a shell |
| FR-9 Release | 0 | 8 | Gates A–E signed off |
| **Total** | **0** | **72** | full replacement |

Update the counts only when a row becomes `verified`.

## Packet ledger

| Packet | Work | Status | Implementation commit | Review commit | Notes |
|---|---|---|---|---|---|
| FR-0.1 | Review and commit current working tree | review | `4ef1001` (via `c1f5d9e`+`a1d98b0`) | — | 3 commits: source hardening, tests, docs; `corpus/` gitignored; working tree clean |
| FR-0.2 | Restore complete quality gate | review | `06d7a39` | — | Current gate: 717 collected (713 passed + 4 skipped: model-integration/environment state); ruff, format, mypy, compileall green. Historical commit evidence retained; rerun command is in README. |
| FR-0.3 | License and attribution | implementing | — | — | Decision packet done locally (uncommitted): NEW `docs/LICENSE_REVIEW.md` (what ships vs user-supplied/download-on-demand per asset + 3 explicit owner decisions: Brysbaert/iconicity bundling, suite license file, app credits surface) + `tests/test_licenses.py` no-vendoring guards (assets/ holds no lexicon bytes; no data-sized CSV/JSON outside fixtures/corpus); judgment itself still owed — this row cannot go to review without the owner |
| FR-0.4 | Reconcile documentation | review | `aa8b34f` | — | ARCHITECTURE header + CHUNK_LEDGER C5/C17 stale claims fixed; scaffold vs parity distinguished |
| FR-0.5 | Enforce write custody | review | `8e5b82f` | — | R3 revised to allowlisted boundary; gate `test_write_custody.py` (3 tests) green |
| FR-0.6 | CI and platform matrix | review | `007ab7a` | — | gate on 3.12 / ubuntu+windows; no model downloads in unit tests |
| FR-1.1 | Atomic capability inventory | review | `554281e` | — | 115 rows (64 review / 50 todo / 1 dropped); backends split, no verified claims |
| FR-1.2 | Full fixture corpus | review | `a996064` | — | 24 fixtures + README index; 24 probe tests green; NUL/lock/fallback behaviors defined |
| FR-1.3 | Freeze legacy environment | review | `0159a41` | — | oracle b4a5087 clean, 1.6.38, py3.10; 2/77 pins; `docs/LEGACY_ENVIRONMENT.md` |
| FR-1.4 | Spot-check golden capture (owner-scoped) | blocked | — | — | FR-1.2/1.3/1.5 done; blocked on legacy py3.10 runtime (src unimportable here: no nltk/textstat; deps 2/77 pinned). Owner decision 2026-09-03: scope is spot-checks for tier-2 glue questions only; the block gates nothing else |
| FR-1.5 | Comparison harness | implementing | `7e2cc68` | — | frames/CSVs/envelopes/runs; tolerance-aware unordered rows, finite-value rejection, unordered artifact declarations, safe cross-platform paths; review corrections applied locally |
| FR-1.6 | SPEC_ONLY registry | implementing | `9f5875f` | — | `core/spec_only.py` + 11 tests; 2 UNREACHABLE entries; S-rows cite entries; under review corrections C6-16; C6-16 corrections applied (pre-rereview) |
| FR-1.7 | Seed spot-check goldens | blocked | — | — | needs FR-1.4 (blocked); scope rescoped by owner decision 2026-09-03 to glue-heavy outputs only — foundation text-statistics already carry definitional evidence; FR-1.5 comparer ready |
| FR-2.1 | Readability | implementing | `05c638a` | — | 5 legacy formulas + SMOG, live legacy oracle, 16 tests; 87-doc run: FRE 41.0-91.5, no nulls (C6-2: subjective; bounds re-verified in C6-8); syllable-unify follow-up; C6-8 pending |
| FR-2.2 | Lexical diversity | implementing | `09fe15c` | — | Guiraud/MTLD/vocd-D, live legacy oracle, 16 tests; 87-doc run: MTLD 45.1-89.0, D 64.0-109.5 (C6-2: subjective; re-verified in C6-8); C6-8 pending |
| FR-2.3 | Statistical tests I | implementing | `38a7b90` | — | chi2/Yates/V/Phi, crosstabs, G2 A+B; 16 definitional tests; scipy core dep; C6-5 corrections applied (pre-rereview) |
| FR-2.4 | Statistical tests II | implementing | `1d1209c` | — | MWU/Cliff's, KW/eps2, Dunn+Bonferroni/Holm; 14 hand-ranked tests; C6-5 corrections applied (pre-rereview) |
| FR-2.5 | Statistical tests III | implementing | `0b08f20` | — | Mann-Kendall/Sen, Spearman/Kendall + Fisher CI; 14 definitional tests; C6-6 corrections applied (pre-rereview) |
| FR-2.6 | Sentence complexity | implementing | `fd173ab` | — | dep distance/depth/subordination + Yngve/Frazier port; 9 tests; CoreNLP req documented; C6-7 corrections applied (pre-rereview) |
| FR-2.7 | Nominalization | review | — | — | C0–C5 done locally (uncommitted): NEW `core/analysis/nominalization.py` (noun-only deverbal detection, NLTK derivational morphology + length-direction constraint, suffix prefilter + curated bypass, per-word memo) + `tools/nominalization.py` (nominalization.csv + by_sentence.csv) + spec CAP-SEM-05 + profiler adapter; 18 tests in `tests/test_nominalization.py` (fake backend, stub-wn direction cases, NLTK replay marked); full gate 896 passed + 10 skipped, ruff/format/mypy/compileall green |
| FR-2.8 | Style analysis | review | — | — | C0–C5 done locally (uncommitted): NEW `core/analysis/style.py` (per-sentence Brysbaert concreteness with sklearn stopwords + Winter iconicity with iconic-words list, user-supplied assets, hand-computed test values) + `tools/style.py` (--analysis concreteness\|iconicity) + spec CAP-SEM-07 + profiler adapter; 14 tests in `tests/test_style.py`; full gate 896 passed + 10 skipped, ruff/format/mypy/compileall green |
| FR-2.9 | Semantic similarity | implementing | `8eb6658` | — | Levenshtein port + TF-IDF pairs/bands; 11 tests; sklearn core dep; C6-9 corrections applied (pre-rereview) |
| FR-3.1 | Document conversion | implementing | `2c9af2f` | — | txt/csv/tsv/html + loud missing-lib; 7 tests; pdf/docx/rtf need pkgs; C6-10 corrections applied (pre-rereview) |
| FR-3.2 | Spell checking | implementing | `fd4d231` | — | explicit wordlists, Levenshtein suggest, correct-to-artifact; 10 tests; C6-12 corrections applied (pre-rereview) |
| FR-3.3 | Matching and duplicates | implementing | `6148e2a` | — | exact/normalized SHA tiers + TF-IDF fuzzy; 7 tests; C6-9 corrections applied (pre-rereview) |
| FR-3.4 | Structured search | implementing | `c5ae7ed` | — | text/csv/conll unified CLI + CSV-row core; 7 tests; C6-19 corrections applied (pre-rereview) |
| FR-3.5 | Filename operations | implementing | `e4f09b9` | — | standardize rule + preview-first apply; 12 tests; C6-11 corrections applied (pre-rereview) |
| FR-4.1 | Asset registry | implementing | `3b3cbb5` | — | versioned specs + verified loads + status CLI; 8 tests; C6-15 corrections applied (pre-rereview) |
| FR-4.2 | Full sentiment resources | review | — | — | C0–C5 done locally (uncommitted): package-backed VADER (per-sentence + doc means, ±0.05 labels, oracle-file option) + file-backed ANEW/hedonometer/SWN(NLTK); registry stamps for 3 oracle files, no bytes vendored; 19 tests in `tests/test_sentiment.py` + stale stub tests reworked; full gate 770 passed + 7 skipped, ruff/format/mypy/compileall green |
| FR-4.3 | NRC emotions | review | — | — | C0–C5 with FR-4.2: NEW `core/analysis/nrc.py` (10-col Plutchik output, asset→nrclex-bundled resolution) + `tools/nrc.py`; replay + package cross-check green |
| FR-4.4 | WordNet aggregation | review | — | — | C0–C5 done locally (uncommitted): `core/analysis/wordnet.py` (UP/DOWN/category_counts over injectable backend, NLTK lazy extra), `tools/wordnet.py` up/down CLIs, 17 tests in `tests/test_wordnet.py` (fake-backed offline + real-NLTK replay, all green), fixtures in `tests/fixtures/wordnet/`; full gate 728 passed + 4 skipped, ruff/format/mypy/compileall green; unblocks FR-2.7 |
| FR-4.5 | VerbNet and FrameNet | review | — | — | C0–C5 done locally (uncommitted): NEW `core/analysis/verbnet.py` (first-class-wins over injectable backend, NLTK lazy) + NEW `core/analysis/framenet.py` (first-frame-wins, per-instance cached lemma index — the legacy per-lemma regex scan intentionally not reproduced) + `tools/verbnet.py` + `tools/framenet.py` + specs CAP-SEM-02/03 (word-list input, not profiler-eligible); 14 tests in `tests/test_verbnet_framenet.py` (protocol-sorted fakes, stub corpora, NLTK replay marked); full gate 896 passed + 10 skipped, ruff/format/mypy/compileall green |
| FR-4.6 | Symbolic and actor typologies | review | — | — | C0–C5 done locally (uncommitted): NEW `core/analysis/symbolic.py` (10 space types + 12 actor types; exact→last-token→guard→WordNet BFS fallback; instance skipping, PROPN refusal, generic bucket, first-sense-only + depth-12 actor climb; unknown lexicon categories skipped with warning) + `tools/symbolic.py` (--analysis space\|actor) + spec CAP-GIS-06 + NEW `space-typology`/`actor-typology` asset specs (user-supplied, legacy CSVs pinned as source); 25 tests in `tests/test_symbolic.py` (hand-built graph incl. depth-cap/first-sense proofs, NLTK replay marked); full gate 896 passed + 10 skipped, ruff/format/mypy/compileall green |
| FR-5.1 | Stanza production backend | implementing | `02f26b9` | — | NER wired (token-level), en verified, 5 gated tests; other langs pending models; C6-13 corrections applied (pre-rereview) |
| FR-5.2 | CoreNLP backend | review | — | — | Backend promoted beyond the stub locally (uncommitted): real urllib probe (`probe_server`, http(s) scheme-guarded) + full per-document parse mapping to canonical CoNLL columns; dead server fails CORENLP_UNAVAILABLE with the start command; marked live-server test via NLP_SUITE_CORENLP_URL; existing backend tests updated to the new codes |
| FR-5.3 | Semantic role labeling | implementing | — | — | Worker probe done locally (uncommitted): `NLP_SUITE_SRL_PYTHON` resolution + `transformer_srl` import check (`probe_worker`, SRL_WORKER_MISSING/BROKEN with setup pointers); reachable worker still reports SRL_FRAMES_UNMAPPED — full frame mapping over the worker protocol is the owed remainder |
| FR-5.4 | Neural coreference | review | — | — | Deterministic lemma baseline promoted: spec CAP-COREF-01 + profiler adapter + CLI flag mirror; neural backend (CoreNLP) stays a documented follow-up, not a silent substitution; full gate green (see FR-7.4 batch note) |
| FR-5.5 | Word2Vec | review | — | — | C0–C5 done locally (uncommitted): `word_embeddings.py` rewritten (seeded Gensim train, neighbours, sklearn t-SNE, gensim-free npz save/load), CLI writes vectors.csv + neighbours.csv + tsne.csv; 15 tests in `tests/test_w2v.py` + stub test updated; full gate 753 passed + 6 skipped, ruff/format/mypy/compileall green |
| FR-5.6 | Contextual embeddings and WSI | review | — | — | Honesty rebuild done locally (uncommitted): production MD5 vectors REMOVED — NEW `TransformerBackend` (lazy `transformers`+torch, word-piece means, loud CTX_BACKEND_MISSING/CTX_MODEL_MISSING) behind an `EmbeddingBackend` protocol; hash fake moved to `conftest.HashEmbeddingBackend`; NEW `tests/test_contextual.py` (15 tests: shape/context-sensitivity/determinism/WSI/failure codes incl. HF_HUB_OFFLINE bogus-model); `test_analysis_36_40` updated to inject the fake; NEW `embeddings` extra (+`all`, +doctor check); CLI `--model`; spec CAP-EMBED-02/03 + profiler adapter; full gate green |
| FR-5.7 | Gensim topic modeling | review | — | — | C0–C5 done locally (uncommitted): NEW `core/analysis/lda.py` (seeded LdaModel, topics + dominant + c_v/perplexity, size-advice guards), `topic_model.py::run` rewired (0-based Topic/Word/Weight), CLI writes topics.csv + topics_dominant.csv with seed/coherence in envelope; 12 tests in `tests/test_lda.py` + stub test updated; full gate 738 passed + 6 skipped, ruff/format/mypy/compileall green |
| FR-5.8 | MALLET adapter | implementing | — | — | NEW `core/analysis/mallet.py` done locally (uncommitted): binary probe + `train_topics` wrapper (shell=False argv, MALLET_MISSING/FAILED/BAD_KEYS codes, topic-keys table); binary-gated — no MALLET here, so no live run; no CLI until a binary exists to run against |
| FR-5.9 | BERT analyses | review | — | — | Atomic split done: CAP-SEM-08 extractive summarization shipped (NEW `core/analysis/bert_extract.py` centroid extractive over the injectable embedding backend — the legacy unpinned `bert-extractive-summarizer` intentionally not reproduced — + `tools/bert_extract.py` + spec + profiler adapter; 7 tests in `tests/test_bert_extract.py` with the hash double); legacy BERT NER/sentiment/word-embeddings map to the ner/sentiment/contextual rows, not this one |
| FR-6.1 | Clause and enhanced SVO | review | — | — | Clause/SVO surface promoted: spec CAP-CONLL-02/04 + profiler adapter + CLI flag mirror; CAP-CONLL-03 (Stanford enhanced taxonomy: negation/coordination) stays the documented remainder |
| FR-6.2 | Character emotion arcs | review | — | — | NEW `core/narrative/characters.py` done locally (uncommitted): PERSON spans via NER-column adjacency (boundaries do not survive canonicalization — documented), case-insensitive grouping, per-(character,sentence) VADER arcs; wired into the narrative CLI + spec CAP-NARR-03 + profiler adapter; 6 tests in `tests/test_characters.py` (incl. str/int64 join-key regression); sentence-binning + HTML stay the documented remainder |
| FR-6.3 | Narrative and story shapes | review | — | — | Emotion + length arcs promoted: spec CAP-NARR-01/02 + profiler adapter + CLI flag mirror; character arcs still await coref/NER decisions (FR-6.2); full gate green |
| FR-6.4 | Entity timelines and movement | review | — | — | NER surface promoted: spec CAP-NER-01/02 + profiler adapter + CLI flag mirror; CAP-NER-03 (multi-token movement tracks) stays the documented remainder |
| FR-6.5 | Chart families and export | review | — | — | Bar-chart surface promoted: spec CAP-VIZ-01/02 (CSV input, not profiler-eligible) + CLI flag mirror; static kaleido export stays an open decision (CAP-VIZ-03); full gate green |
| FR-6.6 | Network and Gephi output | review | — | — | Wordcloud + GEXF surface promoted: spec CAP-VIZ-05/06 (CSV input, not profiler-eligible) + CLI flag mirror; full gate green |
| FR-6.7 | GIS workflow | review | — | — | NEW `core/gis/online.py` done locally (uncommitted): `GoogleGeocoder` (in-memory cache, result limit, timeout; key from `GOOGLE_MAPS_KEY` only) behind an injectable request protocol + `--provider google` on the geocode CLI; 10 tests in `tests/test_online_clients.py` (no network); full gate green |
| FR-6.8 | Knowledge graph clients | review | — | — | NEW `core/kg/dbpedia.py` done locally (uncommitted): Spotlight annotate client (injectable HTTP, confidence bound, loud failure codes) + `--source dbpedia` on the knowledge_graph CLI (stub stays the offline default); 5 client tests in `tests/test_online_clients.py` (no network); YAGO/Wikipedia stay stub-covered under the same module |
| FR-6.9 | PC-ACE workflow | implementing | — | — | Class-L first step done locally (uncommitted): NEW `docs/PCACE_DECOMPOSITION.md` splits 24,618 legacy lines into 6 children (FR-6.9a source import → FR-6.9f validation flow) mapped onto the existing `core/pcace` seed + CLIs, each with dossier scope and evidence shape; children are todo rows below — no code until their C0–C5 cycles |
| FR-6.9a | PC-ACE source import (xlsx → SQLite) | todo | — | — | Dossier: `docs/PCACE_DECOMPOSITION.md`; needs `openpyxl`; fixture xlsx trio → joined DB |
| FR-6.9b | PC-ACE grammar validation | todo | — | — | Dossier: `docs/PCACE_DECOMPOSITION.md`; NEW `core/pcace/grammar.py` over a fixture DB |
| FR-6.9c | PC-ACE relational queries | todo | — | — | Dossier: `docs/PCACE_DECOMPOSITION.md`; `tools/sql.py` extension |
| FR-6.9d | PC-ACE aggregation families | todo | — | — | Dossier: `docs/PCACE_DECOMPOSITION.md`; reuse `core/viz`, no new chart code |
| FR-6.9e | PC-ACE corpus cross-check | todo | — | — | Dossier: `docs/PCACE_DECOMPOSITION.md`; NEW `core/pcace/crosscheck.py` |
| FR-6.9f | PC-ACE validation flow (CLI + page) | todo | — | — | Dossier: `docs/PCACE_DECOMPOSITION.md`; depends on 6.9a–6.9c |
| FR-7.1 | Declarative tool registry | implementing | `58b2b8a` | — | 20 specs (14 + 6 FR-4/5 surfaces: wordnet, topic_model, word_embeddings, sentiment_vader_anew, sentiment_swn_hedono, nrc) + validation; spec↔CLI flag mirror test (`tests/test_registry_new_tools.py`); conditional parse metadata, choices/bounds, and explicit scope remain; no dispatch; full gate 775 passed + 7 skipped, ruff/format/mypy/compileall green |
| FR-7.2 | Envelope composition | implementing | — | — | NEW `core/profiler/batch.py` done locally (uncommitted): per-tool child run dirs + parent batch.json with output-root-relative links (no globbing), failures recorded w/o child dirs; CLI `--analyses` selection; full gate 797 passed + 7 skipped, ruff/format/mypy/compileall green; next child: resume/cache + report (FR-7.5/7.6) |
| FR-7.3 | Profiler execution plan | implementing | — | — | Pure plan model + executor done locally (uncommitted): NEW `core/profiler/plan.py` (validation, phase order, needs_parse) + NEW `core/profiler/executor.py` (12 adapters over shared corpus/parse, per-tool isolation/timings); 20 executor/plan tests; live replay 6/8 ok + 2 asset-missing isolations; full gate 795 passed + 7 skipped, ruff/format/mypy/compileall green; next child: envelopes (FR-7.2) |
| FR-7.4 | Complete profiler analysis set | implementing | — | — | Batch adapters landed for nominalization + style (FR-2.7/2.8), coreference + narrative + contextual (FR-5.4/6.3/5.6), and clause_svo + ner + bert_extract (FR-6.1/6.4/5.9); word-list/CSV tools stay profiler-ineligible by design; rest awaits FR-5.2(server)/5.3(frames)/6.x backends; cannot complete before those packets |
| FR-7.5 | Resume and cache | implementing | — | — | NEW `core/profiler/resume.py` done locally (uncommitted): reuse keys (tool version + params + corpus sha + versions hook), manifest-based find_reusable, CLI `--resume-from` (same root); live replay links prior child with reused:true; backend/model versions await FR-9.3 (recorded follow-up); full gate 806 passed + 7 skipped, ruff/format/mypy/compileall green |
| FR-7.6 | Profiler report | implementing | — | — | NEW `core/profiler/report.py` done locally (uncommitted): pure render_report (status/timings/artifacts table + visible Failures section, reused marked); every batch writes report.md; same gate as FR-7.5 |
| FR-8.1 | Application structure | review | — | — | NEW `app/state.py` (pure shared AppState + corpus resolution + validation rollup, 8 tests) + NEW `app/session.py` (shared sidebar persisted in session state); 01_Tools rewired to the shared sidebar (plus corpus pre-validation); headless smoke Home/Tools 200 with no errors |
| FR-8.2 | Corpus selection and validation UI | review | — | — | NEW `app/pages/02_Corpus.py` (shared-sidebar corpus → same `core.data.validation` check as the CLI + per-file report + reader agreement); headless smoke /Corpus 200 with no errors |
| FR-8.3 | Declarative tool forms | implementing | — | — | NEW `app/forms.py` (pure spec→widget descriptors, 7 tests) + NEW `app/pages/01_Tools.py` (eligible-tool forms, plan validation, synchronous run via executor/batch, gallery handoff); headless smoke (Home + Tools 200, no errors); full gate 813 passed + 7 skipped, ruff/format/mypy/compileall green; background jobs (FR-8.4) still owed — long runs block the page |
| FR-8.4 | Background jobs | review | — | — | C0–C5 done locally (uncommitted): NEW `core/jobs.py` (subprocess-per-job `python -m tools.<name>` with stdout/stderr to the job dir, watcher-thread status.json RUNNING→DONE/FAILED — threads-attached-to-sys.stdout rejected after a real capture race) + `tools/jobs.py` (submit/status/list) + registry exclusion; 8 tests in `tests/test_jobs.py`; full gate green |
| FR-8.5 | Results experience | review | — | — | Gallery completes the FR-8.5 list locally (uncommitted): NEW pure `app/scanner.py::artifact_view` mapping (CSVs → tables, `.md` reports → markdown, `.html` → inline incl. wordclouds, manifests/GEXF/unknown → download-only) wired into Home with download buttons + provenance + diagnostics; 7 viewer tests; headless smoke in the final gate |
| FR-8.6 | Guided setup | review | — | — | NEW `app/pages/03_Setup.py` over NEW `app/state.py::setup_readiness` (Python/core-deps/backend-package/corpus checklist, every failure names its fix; 3 tests); `nlp-suite doctor` linked for the deep checks; headless smoke owed to the final gate |
| FR-8.7 | Usability acceptance | implementing | — | — | Protocol done locally (uncommitted): NEW `docs/USABILITY_TEST.md` (install-to-result script + pass criteria) with operator self-run evidence (4 pages smoke 200, executor/jobs/gallery covered by tests); a real target-user session is still owed — this row cannot go to review without it |
| FR-9.1 | Unified command | review | — | — | NEW `tools/unified.py` (`nlp-suite --list` over 25 registry tools + utility CLIs; routes any `tools.<name>.main` by module so non-registry CLIs like corpus_validation work too) + `nlp-suite` script entry; 7 tests in `tests/test_unified.py`; `unified` recorded in TOOL_REGISTRY_EXCLUSIONS (meta-dispatcher, not an analysis) |
| FR-9.2 | Optional dependency extras | review | — | — | NEW `all` meta-extra covering every optional extra (test-pinned); doctor `_check_extras` now also reports vaderSentiment/gensim/nltk with their extra fixes (non-required, recommendations only) |
| FR-9.3 | Model and asset management | review | — | — | NEW `tools/models.py` (read-only backend×language inventory reusing the doctor probes, fixes inline, models.csv envelope; never downloads) + 2 tests + registry exclusion; `models` also mapped in the migration guide |
| FR-9.4 | Clean-machine installers | review | — | — | NEW `docs/INSTALL.md` (Windows + macOS steps, extras table, no-download policy) verified live on Windows (`pip install -e .`, `nlp-suite --list`, `nlp-doctor en` all green) + `tests/test_install.py` (entry points resolve; every non-meta extra documented); macOS run + PyInstaller bundling explicitly out of scope for this row |
| FR-9.5 | Migration guide | review | — | — | NEW `docs/MIGRATION.md` (every shipped analysis mapped legacy GUI → NG command with honest status incl. deliberately-not-ported) + `tests/test_migration.py` drift guards (every named command resolves; every registry tool is mapped); full gate green |
| FR-9.6 | Security and privacy review | review | — | — | NEW `docs/SECURITY.md` (posture: 3 subprocess allowlist entries, 1 network module with scheme guard, no eval/exec/shell, path custody, no secrets) + `tests/test_security.py` AST guards (5 tests: shell/dynamic-code/network/subprocess confinement + allowlist accuracy) — the review runs on every gate, not once |
| FR-9.7 | Performance acceptance | review | — | — | NEW `scripts/bench.py` (times read+parse+7 analyses into JSON; missing deps recorded, never crash) + schema tests in `tests/test_bench.py` + `docs/PERFORMANCE.md` with the recorded 87-doc/597k-token SOTU run (~514s total, narrative/VADER dominating at 331s — first vectorization candidate); 100+ doc re-run owed as data, not extrapolation |
| FR-9.8 | Release-candidate audit | review | — | — | NEW `scripts/rc_audit.py` done locally (uncommitted): fast static audit (required docs present; every `review` row's code paths exist, `module::symbol`-aware; registry↔migration↔entry-points triple agrees) — 0 failures on this tree — + 5 tests in `tests/test_rc_audit.py`; the pytest/ruff/mypy gate itself still runs separately |

## Capability parity rows

One row per atomic user-visible capability (FR-1.1). Combined rows are split
wherever backends differ in status — one working backend never hides a stub.
Statuses for scaffold-built rows are `review` (implemented + tested under the
old chunk contract; FR-gate parity evidence still owed, so no row is marked
`verified` here). Oracle `T` = scaffold tests exist; `G` = spot-check golden
owed to FR-1.7 (tier-2 glue questions only — owner decision 2026-09-03;
package-backed analyses carry definitional evidence instead of a legacy
capture); `S` = SPEC_ONLY owed to FR-1.6. Implementation cites the scaffold
commit from `CHUNK_LEDGER.md` where applicable.

### Intake and file operations

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-INTAKE-01 | File encoding check + emptiness report (`file_checker_util`) | `core/file_ops/checker.py` + `tools/corpus_validation.py` | T, G | review | `11e8d1d` | — | read-only; legacy rewrote inputs on read |
| CAP-INTAKE-02 | Text cleaning / normalization (`file_cleaner_util`) | `core/file_ops/cleaner.py` | T, G | review | `11e8d1d` | — | |
| CAP-INTAKE-03 | TXT/CSV/TSV conversion (`file_converter_util`) | `core/file_ops/converter.py` | T, G | review | `11e8d1d` | — | PDF/DOCX/RTF → FR-3.1 |
| CAP-INTAKE-04 | PDF/DOCX/RTF/HTML conversion | `core/file_ops/converter.py` (optional `converters` extra) | T, G | implementing | `2c9af2f` | — | valid-format fixture pairs (sample file + expected text) still owed; malformed inputs are diagnosed; directory CLI continues valid files and records failures as warnings; .doc separate unsupported |
| CAP-INTAKE-05 | File merge (`file_merger_util`) | `core/file_ops/merger.py` | T, G | review | `11e8d1d` | — | |
| CAP-INTAKE-06 | File match by glob (`file_matcher_util`) | `core/file_ops/merger.py::match_files` | T, G | review | `11e8d1d` | — | fuzzy matching → FR-3.3 |
| CAP-INTAKE-07 | Seeded corpus sampling (`sample_corpus_util`) | `core/file_ops/search.py::sample_corpus` | T, G | review | `11e8d1d` | — | |
| CAP-INTAKE-08 | Filename sanitization (`file_filename_util`) | `core/file_ops/search.py::filename_sanitize` | T, G | review | `11e8d1d` | — | |
| CAP-INTAKE-09 | Filename date extraction (`file_classifier_date_util`) | `core/io/reader.py::date_from_filename` | T, G | review | `4a586d1` | — | legacy swapped sep/format; new is conservative |
| CAP-INTAKE-10 | Filename standardize/rename with preview | `core/file_ops/filenames.py` + `tools/filenames.py` | T, G | implementing | `e4f09b9` | — | preview-first, never overwrites; C6-11 |
| CAP-INTAKE-11 | Word search across files (`file_search_byWord`) | `core/file_ops/search.py::search_in_text` | T, G | review | `11e8d1d` | — | |
| CAP-INTAKE-12 | Structured search over CoNLL rows | `core/analysis/table_search.py` | T, G | review | `d9c169d` | — | CSV-row search → FR-3.4 |
| CAP-INTAKE-13 | Spell checking (`file_spell_checker_*`) | `core/analysis/spellcheck.py` + `tools/spellcheck.py` | T, G | implementing | `fd4d231` | — | explicit wordlist design; C6-12 |
| CAP-INTAKE-14 | Fuzzy / duplicate-document detection (`file_find_non_related_documents_util`) | `core/analysis/doc_duplicates.py` + `tools/doc_duplicates.py` | T, G | implementing | `6148e2a` | — | NER-intruder variant follow-up; C6-9 |
| CAP-INTAKE-15 | Corpus validation report (`corpus_checker_PCACE_data_main`) | `core/data/validation.py` + `tools/corpus_validation.py` | T, G | review | `43430cc` | — | |
| CAP-INTAKE-16 | Split by length/words/lines (`file_splitter_By*`) | `core/file_ops/splitter.py::split_text` | T + regression, G | review | `11e8d1d`, `c1f5d9e` | — | 10 legacy utils → 1 engine; hang guard tested |
| CAP-INTAKE-17 | Split by delimiter / keyword | same engine | T + regression, G | review | `11e8d1d`, `c1f5d9e` | — | keyword splits before every occurrence |
| CAP-INTAKE-18 | NER-pattern classification (`file_classifier_NER_util`) | `core/file_ops/classifier.py::classify_by_ner_pattern` | T, G | review | `11e8d1d` | — | deterministic regex baseline |
| CAP-INTAKE-19 | KWIC concordance (legacy menu entry `TIPS_NLP_KWIC`; no surviving module) | `core/analysis/kwic.py` + `tools/kwic.py` | T, G | review | — | — | form/lemma, regex and case-sensitive modes; each row keeps document + sentence provenance; bounded by --max-hits |

### Parsing and CoNLL interchange

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-PARSE-01 | spaCy parse → CoNLL (`spaCy_util`) | `core/pipelines/spacy_backend.py` | T + integration, G | review | `0159476`, `c1f5d9e` | — | blank fallback removed; MODEL_MISSING is loud |
| CAP-PARSE-02 | Stanza parse → CoNLL (`Stanza_util`) | `core/pipelines/stanza_backend.py` | T + integration, G | review | `11e8d1d` | — | integration gate probes actual processor loadability; only English is environment-backed today |
| CAP-PARSE-03 | CoreNLP parse (`Stanford_CoreNLP_util`) | `core/pipelines/corenlp_backend.py` (stub) | G (capture pending) | todo | `11e8d1d` | — | bounded owner-approved server capture is possible; SPEC_ONLY entry is `CAPTURE_PENDING`, not an unreachable claim; FR-5.2 |
| CAP-PARSE-04 | SRL, isolated env (`SRL_worker`) | `core/pipelines/srl_backend.py` (stub) | S | todo | `11e8d1d` | — | SPEC_ONLY `core/spec_only.py::CAP-PARSE-04` (UNREACHABLE, py3.8 env); FR-5.3 |
| CAP-PARSE-05 | Universal→Penn normalization (`CoNLL_util`) | `core/conll/normalize.py` | T, G | review | `2c079d6` | — | the explicitly-correct legacy code, ported |
| CAP-PARSE-06 | Versioned CoNLL table + sidecar | `core/conll/schema.py` | T, G | review | `2c079d6` | — | |
| CAP-PARSE-07 | Sentence division incl. final sentence | `core/conll/division.py` | T + regression, G | review | `2c079d6` | — | legacy dropped final sentence (FIXES #12) |
| CAP-PARSE-08 | Backend/language gate (`NLP_setup_package_language_main`) | `core/config.py::NLPConfig` | T, G | review | `e79e784` | — | legacy check was `if True` |

### CoNLL analysis

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-CONLL-01 | Word analysis by POS category, 6 analyses (`CoNLL_*_analysis_utils`) | `core/analysis/conll_wordlist.py` (+ `word_analysis` re-export) | T, G | review | `b4ef7df`, `11e8d1d` | — | 6 clones → 1 parameterized engine |
| CAP-CONLL-02 | Clause tag frequencies (`CoNLL_clause_analysis_util`) | `core/analysis/clause_svo.py::clause_frequencies` | T, G | review | `d9c169d` | — | counts Clause Tag column |
| CAP-CONLL-03 | Enhanced clause taxonomy (`Stanford_CoreNLP_clause_util`) | — (FR-6.1) | pending | todo | — | — | parser-specific deps, negation, coordination |
| CAP-CONLL-04 | SVO extraction (`SVO_util`) | `core/analysis/clause_svo.py::extract_svo` | T, G | review | `d9c169d` | — | enhanced++ deps → FR-6.1 |
| CAP-CONLL-05 | SVO compare, Jaccard (`SVO_compare_util`) | `core/analysis/svo_compare.py` | T, G | review | `f5c5394` | — | |
| CAP-CONLL-06 | Table search with predicates (`CoNLL_table_search_util`) | `core/analysis/table_search.py` | T, G | review | `d9c169d` | — | legacy chopped IDs `[:-2]` |
| CAP-CONLL-07 | K-sentence bookends (`CoNLL_k_sentences_util`) | `core/analysis/k_sentences.py` | T, G | review | `d9c169d` | — | |

### Statistics

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-STATS-01 | Per-document text stats (`statistics_txt_util`) | `core/analysis/text_statistics.py` | T, G | review | `d9c169d` | — | |
| CAP-STATS-02 | TTR / RootTTR / LogTTR / Herdan / Yule-K (`statistics_corpus_lexical_diversity_util`) | `core/analysis/corpus_statistics.py` | T, G | review | `d9c169d` | — | |
| CAP-STATS-03 | Guiraud / MTLD / vocd-D | `core/analysis/lexical_diversity.py` | T, G | implementing | `09fe15c` | — | vocd seeded; C6-8 short-text policy pending |
| CAP-STATS-04 | Readability formulas (legacy module has 5: FRE, FK, Fog, Coleman-Liau, ARI) | `core/analysis/readability.py` | T, G | implementing | `05c638a` | — | +SMOG new-from-formula; count reconciled; C6-8 |
| CAP-STATS-05 | TF-IDF per term (`statistics_corpus_tfidf_util`) | `core/analysis/topic_model.py` (TF-IDF scoring) | T, G | review | `502fce0` | — | same scoring feeds the topics stub |
| CAP-STATS-06 | Pearson correlation (`statistics_statistical_tests_util`) | `core/analysis/csv_stats.py::correlation` | T, G | review | `f5c5394` | — | |
| CAP-STATS-07 | Chi-square / crosstab / keyness (log-likelihood) | `core/analysis/stats_categorical.py` | T, G | implementing | `38a7b90` | — | C6-5 |
| CAP-STATS-08 | Nonparametric group tests (Mann-Whitney / Kruskal-Wallis / Dunn) | `core/analysis/stats_groups.py` | T, G | implementing | `1d1209c` | — | C6-5 |
| CAP-STATS-09 | Trends (Mann-Kendall) + multiple-comparison correction | `core/analysis/stats_trends.py` + `stats_groups.py::dunn` | T, G | implementing | `0b08f20` | — | C6-6 |
| CAP-STATS-10 | CSV describe + group_by (`statistics_csv_util`) | `core/analysis/csv_stats.py::describe` | T, G | review | `f5c5394` | — | |
| CAP-STATS-11 | Corpus keyness between two document groups | `core/analysis/keyness.py` + `tools/keyness.py` | T, G | review | — | — | regex over document names splits A/B; shares the CAP-STATS-07 log-likelihood engine; Hardie +0.5 Log Ratio smoothing; Streamlit `app/pages/04_Keyness_Lab.py` |
| CAP-STATS-12 | Corpus TF-IDF term weighting (`statistics_corpus_tfidf_util`) | `core/analysis/tfidf.py` + `tools/tfidf.py` | T, G | review | — | — | smoothed idf `ln((1+N)/(1+df)) + 1` with optional sublinear tf and L2 normalization, verified equal to `sklearn.feature_extraction.text.TfidfVectorizer(smooth_idf=True)`; computed over the shared CoNLL parse so it sees the same tokens (and lemmas) as every other tool, rather than re-tokenizing raw text. Long-format top-N per document replaces the legacy dense document-by-term matrix, which was quadratic in corpus size |
| CAP-STATS-13 | Lexical dispersion (NEW: no legacy equivalent) | `core/analysis/dispersion.py` + `tools/dispersion.py` | T | review | — | — | Range, Gries' DP, DP norm, Juilland's D and Juilland's U adjusted frequency, over documents or equal chunks. New capability, not a port: the legacy `statistics_corpus_word_frequency_util` reported raw and relative frequency with no dispersion measure, so a word concentrated in one document was indistinguishable from core vocabulary of the same frequency. Formulas verified on hand-computed reference cases including the unequal-part case where DP must be 0 for a size-proportional distribution |
| CAP-STATS-14 | Dictionary-based content analysis over a corpus axis (NEW: no legacy equivalent) | `core/analysis/lexicon_series.py` + `tools/lexicon_series.py` | T | review | — | — | Named word groups counted along year, decade, document or a regex over document names, reported as occurrences, rate per 1,000 tokens, matching sentences and share of sentences. New capability, not a port: the legacy word-frequency utilities counted single types corpus-wide, so a category such as "civil rights" spanning five phrases, and its movement over time, could only be assembled by hand outside the suite. Phrases match as phrases; facets a document cannot be placed on are reported and excluded rather than given a fallback label; an optional `within` lexicon restricts counting to the sentences matching it, which is the conditional question (what vocabulary accompanied a topic) rather than the frequency one |
| CAP-UX-01 | Chart recommendation + degenerate-chart refusal (NEW: no legacy equivalent) | `core/insight/recommend.py` + `core/insight/profile.py` + `tools/explain.py` | T | review | — | — | ranks the charts worth drawing from a result and refuses the ones that cannot inform: a column against itself, two columns that are the same quantity under different names, perfectly collinear or rank-identical columns, constant axes, free text, and numeric row identifiers on x. Every recommendation is a real `ChartSpec`, and a test asserts no recommendation is ever degenerate, so the two halves cannot drift |
| CAP-UX-02 | Plain-language result readout (NEW: no legacy equivalent) | `core/insight/readout.py` + `core/io/writer.py` | T | review | — | — | every run writes `readout.md` beside its table: the result's shape, what dominates it, and the traps that table has actually triggered, with the concrete setting to change. Tool-specific rules for collocations, tfidf, dispersion, keyness and readability; generic role-based reading otherwise. Deliberately never claims significance or causation, which `test_readout_never_claims_significance_or_cause` enforces as a contract. Generation is best-effort: a readout failure can never fail the run that produced the numbers |

### Sentiment, NER, coreference

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-SENT-01 | VADER sentiment (`sentiment_analysis_VADER_util`) | `core/analysis/sentiment_vader_anew.py::vader` | T, G | review | `f5c5394` | — | compact baked lexicon → FR-4.2 |
| CAP-SENT-02 | ANEW valence/arousal/dominance (`sentiment_analysis_ANEW_util`) | same module `::anew` | T, G | review | `f5c5394` | — | compact baked lexicon → FR-4.2 |
| CAP-SENT-03 | SentiWordNet (`sentiment_analysis_SentiWordNet_util`) | `core/analysis/sentiment_swn_hedono.py::sentiwordnet` | T, G | review | `f5c5394` | — | compact baked lexicon → FR-4.2 |
| CAP-SENT-04 | Hedonometer happiness (`sentiment_analysis_hedonometer_util`) | same module `::hedonometer` | T, G | review | `f5c5394` | — | compact baked lexicon → FR-4.2 |
| CAP-SENT-05 | NRC emotions (`sentiment_analysis_NRC_util`) | `core/analysis/nrc.py` (asset/nrclex fallback) | T, G | review | — | — | FR-4.3; 8 Plutchik emotions normalized over own total + separate valence pair (legacy convention); bundled-lexicon tests + fixture oracle |
| CAP-NER-01 | Entity timeline (`NER_entity_timeline_util`) | `core/analysis/ner.py` | T, G | review | `f5c5394` | — | sid ordering fixed FR-0 |
| CAP-NER-02 | Location tracking (`NER_location_tracking_util`) | same module | T, G | review | `f5c5394` | — | |
| CAP-NER-03 | Multi-token entities + movement tracks | `core/analysis/movement.py` + `tools/ner.py` (`--movement`) | T, G | review | — | — | person↔location pairs per sentence + entity-location summary; `--geocode` attaches Lat/Lon from the offline KB (CAP-GIS-01) keeping only Status=="OK" rows |
| CAP-COREF-01 | Deterministic lemma coreference baseline | `core/analysis/coreference.py` | T, G | review | `502fce0` | — | explicitly a baseline, not neural |
| CAP-COREF-02 | Neural coreference (`coreference_neural_util`) | — (FR-5.4) | pending | todo | — | — | |

### Topics, embeddings, n-grams

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-TOPIC-01 | TF-IDF partitioned topics (scaffold stand-in) | `core/analysis/topic_model.py` | T | todo | `502fce0` | — | deterministic stub; real LDA → FR-5.7 |
| CAP-TOPIC-02 | Gensim LDA (`topic_modeling_gensim_util`) | `core/analysis/lda.py` + `tools/lda_gensim.py` (seeded LdaModel; tool renamed `lda_gensim` in the LDA split, with native Intertopic Distance Map + lambda relevance) | T | review | — | — | FR-5.7; topics.csv + topics_dominant.csv; c_v coherence; legacy stopword-appending-a-list bug fixed (words actually filtered now); no pyLDAvis (viewer renders CSVs) |
| CAP-TOPIC-03 | MALLET topics (`topic_modeling_mallet_util`) | `core/analysis/mallet.py` + `tools/lda_mallet.py` + `desktop_backend/runner.run_mallet` (tool `lda_mallet`; binary-gated) | — | — | shelled via safe argv |
| CAP-TOPIC-04 | BERT topics (`topic_modeling_bert_util`) | `core/analysis/bert_topics.py` + `tools/bert_topics.py` | T, G | review | — | — | FR-5.9; KMeans over doc-mean sentence embeddings via EmbeddingBackend, discriminative term labels, seeded/deterministic; needs the `[embeddings]` extra. Legacy file was an abandoned notebook, so this is new-from-spec, not a port |
| CAP-TOPIC-05 | Topic stability across seeds (new-from-spec) | `core/analysis/topic_stability.py` + `tools/lda_stability.py` (tool `lda_stability`) | T | review | — | — | FR-5.7 extension; refits at several seeds, Hungarian-matches topics to the reference by top-word Jaccard, judged on the worst seed; summary.csv + matches.csv; the legacy suite had no stability check |
| CAP-EMBED-01 | word2vec train + distances (`word2vec_Gensim_util`) | `core/analysis/word_embeddings.py` (seeded Gensim) + `core/viz/embeddings.py::tsne_html` | T, G | review | `502fce0` | — | FR-5.5; vectors/neighbours/tsne CSVs + tsne.html scatter (legacy word2vec_tsne_plot_util parity); CLI writes all four artifacts |
| CAP-EMBED-02 | Contextual embeddings (`BERT_util`) | `core/analysis/contextual.py` over `core/models/onnx_backend.py` (ONNX BERT shipped with the app; `TransformerBackend` dev fallback) | T | review | — | — | FR-5.6; MD5 production fake removed, hash double lives in conftest only; words read at their own pieces (`core/models/align.py`) |
| CAP-EMBED-03 | WSI senses (`WSI_*`) | `core/analysis/contextual.py::wsi_senses` (seeded 2-means per lemma, kept only when the senses separate) | T | review | — | — | FR-5.6; replaced the median split on the first vector component |
| CAP-NGRAM-01 | N-grams 1..5 (`NGrams_util`) | `core/analysis/ngrams.py` | T, G | review | `502fce0` | — | |
| CAP-NGRAM-02 | Bigram collocations, PMI (`NGrams_collocation_statistics_util`) | `core/analysis/ngrams.py::collocations` | T, G | review | `502fce0` | — | |
| CAP-NGRAM-03 | Window co-occurrence (`NGrams_CoOccurrences_util`) | `core/analysis/ngram_cooccurrence.py` | T, G | review | `502fce0`, `c1f5d9e` | — | dead-loop removed FR-0 |
| CAP-NGRAM-04 | Collocation association measures (`NGrams_collocation_statistics_util`) | `core/analysis/collocations.py` + `tools/collocations.py` | T, G | review | — | — | seven measures from one contingency table (PMI, PPMI, t-score, z-score, Dunning G2, Dice, log Dice); adjacent-bigram or unordered-window span; explicit stopword file, never a bundled list. Deeper sibling of CAP-NGRAM-02: `ngrams` keeps its 4-column PMI companion table for the n-gram workflow, `collocations` is the standalone association-measure tool. The legacy module's "log-likelihood" was only the first cell of the G2 sum, so it is not reproduced; the G2 column here is Dunning's statistic, verified against `scipy.stats.chi2_contingency(lambda_="log-likelihood")` |
| CAP-RESEARCH-01 | Question-driven literal phrase distribution (new workspace capability) | `core/research/phrase.py` + saved-question publisher | T, G | review | - | - | One versioned contract drives the interactive and immutable-run paths; reports normalized rates over time, relative document position, document prevalence, optional phrase comparison, and occurrence-level source evidence. |

### Semantics

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-SEM-01 | WordNet aggregation (`semantic_aggregation_WordNet_util`) | `core/analysis/semantic.py` (real WordNet/VerbNet/FrameNet backends) | T | review | — | — | FR-4.4; demo maps are an explicit fallback with `SEM_DEMO_MODE` warnings, not silent |
| CAP-SEM-02 | VerbNet classes | `core/analysis/verbnet.py` (first-class-wins, NLTK lazy) | T | review | — | — | FR-4.5; fake-backed offline tests + marked NLTK replay |
| CAP-SEM-03 | FrameNet frames | `core/analysis/framenet.py` (cached lemma index, NLTK lazy) | T | review | — | — | FR-4.5; per-instance index (legacy per-lemma scan not reproduced) |
| CAP-SEM-04 | Similarity / plagiarism / duplicate detection (`string_similarity_util`, `plagiarist_util`) | `core/analysis/string_similarity.py` + `doc_similarity.py` + `doc_duplicates.py` | T, G | implementing | `8eb6658` | — | unrelated-document workflow still owed; C6-9 |
| CAP-SEM-05 | Nominalization detection (`nominalization_util`) | `core/analysis/nominalization.py` (deverbal, NLTK lazy) | T | review | — | — | FR-2.7; direction constraint + suffix/curated prefilter ported |
| CAP-SEM-06 | Sentence complexity, Yngve/Frazier (`tree.py`) | `core/analysis/sentence_complexity.py` | T, G | implementing | `fd173ab` | — | Yngve/Frazier split to FR-5.2 dependency; C6-7 |
| CAP-SEM-07 | Concreteness / iconicity style (`style_analysis_*`) | `core/analysis/style.py` (user-supplied norm assets) | T | review | — | — | FR-2.8; hand-computed test values; sklearn stopwords (legacy wordlist not vendored) |
| CAP-SEM-08 | BERT extractive summarization (`BERT_util`) | `core/analysis/bert_extract.py` (centroid extractive over injectable backend) | T | review | — | — | FR-5.9; legacy unpinned package intentionally not reproduced |

### Annotators and knowledge graphs

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-ANNOT-01 | HTML text extraction (`html_annotator_extractor_util`) | `core/analysis/html_annotator.py::extract_text` | T, G | review | `656845c` | — | |
| CAP-ANNOT-02 | Dictionary annotation, `<mark>` (`html_annotator_dictionary_util`) | same module `::annotate` | T + regression, G | review | `656845c`, `c1f5d9e` | — | single-pass (nested-markup fix) |
| CAP-ANNOT-03 | Gender pronouns (`html_annotator_gender_*`) | same module `::gender_spans` | T, G | review | `656845c` | — | dictionary-driven (ARCH §11) |
| CAP-ANNOT-04 | Quote / date special annotators (CoreNLP) | — (via FR-5.2 or dict impl) | pending | todo | — | — | ARCH §11 open decision |
| CAP-KG-01 | DBpedia annotate + SPARQL (`knowledge_graphs_DBpedia_util`) | `core/kg/dbpedia.py` (Spotlight client) + stub default | T | review | — | — | FR-6.8; `--source dbpedia` on the CLI |
| CAP-KG-02 | YAGO (`knowledge_graphs_YAGO_util`) | same module (baked) | T | todo | `656845c` | — | FR-6.8 |
| CAP-KG-03 | Wikipedia (`knowledge_graphs_Wikipedia_util`) | same module (baked) | T | todo | `656845c` | — | FR-6.8 |

### Visualization

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-VIZ-01 | Plotly bar charts (`charts_Plotly_util`) | `core/viz/charts.py` | T, G | review | `656845c` | — | |
| CAP-VIZ-02 | Knowable no-plotly fallback rendering | same module (fallback path) | T (fail-big) | review | `656845c`, `c1f5d9e` | — | `CHART_PLOTLY_UNAVAILABLE` warning |
| CAP-VIZ-03 | Static chart export (kaleido) | `core/viz/plotters.py::chart_image_bytes` via `tools/charts.py` | T, G | review | — | — | needs `[plotly-image]`; on export failure the run FAILS but HTML+CSV survive in a `-failed` run dir (documented in docs/viz-charts.md) |
| CAP-VIZ-04 | Excel chart export (`charts_Excel_util`) | `core/viz/charts_excel.py` via `tools/charts.py` (`--format xlsx`) | T, G | review | — | — | **Un-dropped 2026-09-17.** The row previously read `dropped — superseded by Plotly (BUILD_PLAN §4)`, which was stale: the exporter exists and reproduces the legacy workbook layout (Data + first-positioned Chart sheet, native openpyxl chart, axis titles, `tickLblPos="low"`, legend dropped for a single series) across the six kinds the legacy GUI offered (bar/line/pie/scatter/radar/bubble). Excel output is a deliverable users are required to produce, so Plotly does not supersede it. The macro-enabled `.xlsm` hover path remains deliberately not carried (binary templates with embedded VBA). See `core/viz/catalog.py` |
| CAP-VIZ-05 | Wordcloud (`wordclouds_util`) | `core/viz/wordcloud_gephi.py` | T, G | review | `43430cc` | — | |
| CAP-VIZ-06 | Validated GEXF network output (`Gephi_util`) | same module `::gephi_gexf` | T, G | review | `43430cc` | — | schema/weight validation → FR-6.6 |
| CAP-VIZ-07 | **New.** No legacy equivalent: the original suite drew one generic chart family over any table. | `core/viz/panelspec.py` + `core/viz/panels.py` via `tools/panels.py` | T | review | — | — | Tool-specific panels: a figure that belongs to one analysis and knows its result shape, for results a generic x/y/group chart cannot express (a keyness table's effect-size-against-evidence volcano; a topic model's two matrices). Every mark carries resolvable evidence; every figure carries its provenance caption inside the image. |
| CAP-VIZ-08 | **New.** | `core/viz/panels_keyness.py` (`keyness_volcano`) | T, G | review | — | — | The reference panel, and the one that motivated the layer: drawn as a G2 bar chart, keyness hides whether a difference is large or merely well measured. |
| CAP-VIZ-07 | Per-sentence story-shape metrics (`shape_of_stories_*`) | `core/viz/shapes.py::story_shape` | T, G | review | `43430cc` | — | deterministic metrics only |
| CAP-VIZ-08 | KMeans shape clustering (`shape_of_stories_vectorizer_util`) | `core/viz/shape_clusters.py::cluster_shapes` (resampled story_shape trajectories; PCA-variance k heuristic like legacy) | T, G | review | — | — | `tools/shapes.py`; profiler adapter emits shape_clusters.csv; deterministic (seeded KMeans) |
| CAP-VIZ-09 | Sankey flow diagram | `core/viz/shapes.py::sankey_html` | T, G | review | `43430cc` | — | fallback knowable |
| CAP-VIZ-10 | Visualization provenance catalog (NEW: no legacy equivalent) | `core/viz/catalog.py` + `tools/visualizations.py` | T | review | — | — | declares every visualization as a legacy port, a legacy port with a named extension, or new, and maps every visualization-producing module in 1.6.38 to the entry that carries it or to an explicit drop reason. `nlp-suite visualizations --origin legacy` yields the set an assignment expecting the original suite's output needs. `test_every_legacy_visualization_module_is_accounted_for` fails if a legacy module is neither carried nor explicitly dropped, so coverage cannot silently regress |

### Data, SQL, PC-ACE

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-DATA-01 | Concat / merge / pivot / melt (`data_manipulation_util`) | `core/data/manipulation.py` | T, G | review | `43430cc` | — | |
| CAP-DATA-02 | Parameterized SQL query layer (`DB_SQL_main/util`) | `core/data/sql.py` | T + regression, G | review | `43430cc`, `c1f5d9e` | — | single-statement guard tested |
| CAP-PCACE-01 | PC-ACE grammar + code validation | `core/pcace/core.py` | T, G | review | `43430cc` | — | |
| CAP-PCACE-02 | PC-ACE sqlite create / read | same module | T, G | review | `43430cc` | — | connections closed (FR-0) |
| CAP-PCACE-03 | PC-ACE per-category analysis (`DB_PCACE_data_analysis_*`) | `core/pcace/analysis.py` | T, G | review | `5da67d9` | — | |
| CAP-PCACE-04 | Full PC-ACE workflow (validation / import / query / aggregation / visualization) | — (FR-6.9) | pending | todo | — | — | must split before implementation |

### GIS

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-GIS-01 | Offline geocode lookup (`GIS_geocode_util`) | `core/gis/geocode.py` (baked KB) | T, G | review | `5da67d9` | — | |
| CAP-GIS-02 | Online geocoder, cached + bounded (Google) | `core/gis/online.py` (cache + limit + timeout, key from env) | T | review | — | — | FR-6.7; `--provider google` on the geocode CLI |
| CAP-GIS-03 | KML placemarks (`GIS_KML_util`) | `core/gis/mapping.py::kml` | T, G | review | `5da67d9` | — | NaN guard tested FR-0 |
| CAP-GIS-04 | Heatmap HTML (`GIS_heatMap*`) | same module `::heatmap_html` | T, G | review | `5da67d9` | — | |
| CAP-GIS-05 | Haversine distance (`GIS_distance_*`) | `core/gis/geocode.py::distance` | T, G | review | `5da67d9` | — | |
| CAP-GIS-06 | Symbolic / actor typologies (`GIS_symbolic_*`) | `core/analysis/symbolic.py` (curated CSV + WordNet BFS fallback) | T | review | — | — | FR-4.6; `space-typology`/`actor-typology` asset specs added |
| CAP-GIS-07 | Google Earth tours (`GIS_Google_Earth_main`) | `core/gis/mapping.py::tour_kml` + `tools/earth.py` | T, G | review | — | — | FR-6.7; one animated `gx:Track` per group with synthetic 1 s stamps, grouped folders, out-of-range rows skipped with TOUR_SKIPPED_ROWS. Legacy needed a Google Earth desktop install; the KML opens in Google Earth Pro (free) or any gx-capable viewer |

### Narrative

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-NARR-01 | Per-sentence emotion arc (VADER) | `core/narrative/arcs.py::emotion_arc` | T, G | review | `5da67d9` | — | vader failures propagate (FR-0) |
| CAP-NARR-02 | Sentence-length arc | same module `::length_arc` | T, G | review | `5da67d9` | — | |
| CAP-NARR-03 | Character emotion arcs, binning + HTML (`character_emotion_arcs_util`) | `core/narrative/characters.py` (mentions + per-sentence VADER arcs) | T | review | — | — | FR-6.2; binning + HTML stay the remainder |

### Profiler, UI, release

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-PROF-01 | Batch profiler, current subset (`corpus_profiler_util`) | `core/profiler/profiler.py` + `tools/profiler.py` | T, G | review | `5da67d9` | — | text+corpus stats+vader |
| CAP-PROF-02 | Full 31-analysis batch | — (FR-7.4) | pending | todo | — | — | |
| CAP-PROF-03 | Shared-parse execution plan | — (FR-7.3) | pending | todo | — | — | one parse per batch |
| CAP-PROF-04 | Declarative tool registry | `core/profiler/registry.py` (14 retained tools) | T | implementing | `58b2b8a` | — | registry coverage/conditional metadata still needs completion before UI forms |
| CAP-PROF-05 | Envelope-consuming profiler (no globbing) | — (FR-7.2) | pending | todo | — | — | |
| CAP-PROF-06 | Resume / cache + profiler report | — (FR-7.5, FR-7.6) | pending | todo | — | — | |
| CAP-UI-01 | Generic envelope viewer (`NLP_menu_main` → gallery) | `app/scanner.py` + `app/Home.py` | T, G | review | `b4ef7df` | — | renders artifacts by kind |
| CAP-UI-02 | Doctor + install check (`NLP_setup_*`) | `tools/doctor.py`, `tools/install.py` | T (fail-big) | review | `5da67d9`, `c1f5d9e` | — | doctor is new-suite-only, stricter than legacy setup |
| CAP-UI-03 | Declarative tool forms + background jobs + guided setup | — (FR-8.1–FR-8.6) | pending | todo | — | — | |
| CAP-UI-04 | Usability acceptance, no shell | — (FR-8.7) | human test | todo | — | — | requires target-user test |
| CAP-REL-01 | Unified command (`nlp-suite doctor/models/run/profile/ui`) | — (FR-9.1) | pending | todo | — | — | `nlp-doctor` script exists as seed |
| CAP-REL-02 | Migration guide, legacy GUI/output map | — (FR-9.5) | pending | todo | — | — | |

### Syllabus gap tools (P4)

The 446W syllabus gaps the tool manifest (`tests/fixtures/syllabus_tools.json`)
held open, landed as one batch. Status `review` = built and tested, awaiting
independent review; Implementation is `—` because the work is uncommitted.

| Capability ID | Legacy outcome and entry point | New API / CLI / UI | Oracle | Status | Implementation | Review | Notes |
|---|---|---|---|---|---|---|---|
| CAP-ANNO-10 | Gender annotation from 4 name dictionaries (`html_annotator_gender_main`, `lib\namesGender`) | `core/analysis/gender_annotator.py` | T, G | review | — | — | census / carnegie_mellon / nltk / social_security selectable; NLTK names corpus or drop-in `male.txt`/`female.txt`; names are not people |
| CAP-ANNO-11 | Normalized date annotator (CoreNLP-style, without the server) | `core/analysis/date_annotator.py` | T, G | review | — | — | ISO + partial (year / decade); US month-day order documented |
| CAP-ANNO-12 | Quote/dialogue annotator + attribution (`html_annotator_main` quote path) | `core/analysis/quote_annotator.py` | T, G | review | — | — | two-stage sieve (reporting verb, then nearest PERSON); Cue column records which fired |
| CAP-STYLE-10 | GenderGuesser author-style guess (`GenderGuesser` method) | `core/analysis/gender_guess.py` | T, G | review | — | — | own documented weak-word lexicon (not the proprietary dictionaries); style labels, never ground truth |
| CAP-CONLL-10 | Verb modality / tense / voice (`CoNLL_verb_analysis` families) | `core/analysis/verb_analysis.py` | T, G | review | — | — | UD morphology when present, POS/ending heuristics otherwise (warns) |
| CAP-NGRAM-10 | Culturomics n-gram viewer over time (`NGrams_ngram_viewer`) | `core/analysis/ngram_viewer.py` + `core/viz/ngram_viewer.py` | T, G | review | — | — | years from filenames; undated corpora fail with the reason the co-occurrence viewer can run undated |
| CAP-SENT-10 | Neural sentiment, BERT (`sentiment_analysis_BERT_util`) | `core/analysis/sentiment_neural.py::bert_sentences` | T, G | review | — | — | one of four backends on one table contract |
| CAP-SENT-11 | Neural sentiment, spaCy (`sentiment_analysis_SPAcY_util`) | `core/analysis/sentiment_neural.py::spacy_sentences` | T, G | review | — | — | textcat head or a loud failure; vanilla models have none |
| CAP-SENT-12 | Neural sentiment, Stanza (`sentiment_analysis_Stanza_util`) | `core/analysis/sentiment_neural.py::stanza_sentences` | T, G | review | — | — | three-way hard class; Compound lands on -1/0/+1 |
| CAP-SENT-13 | Neural sentiment, Stanford CoreNLP RNTN (`sentiment_analysis_Stanford_CoreNLP_util`) | `core/analysis/sentiment_neural.py::corenlp_sentences` | T, G | review | — | — | needs the Java server; never substitutes another backend |
| CAP-SHAPE-10 | Data reduction: hierarchical clustering (`shape_of_stories_vectorizer_util` HC) | `core/analysis/shape_reduction.py::hierarchical_cluster` | T, G | review | — | — | ward/average/complete/single + dendrogram merge table |
| CAP-SHAPE-11 | Data reduction: SVD (`shape_of_stories_vectorizer_util` SVD) | `core/analysis/shape_reduction.py::svd_reduce` | T, G | review | — | — | seeded TruncatedSVD; variance shares |
| CAP-SHAPE-12 | Data reduction: NMF (`shape_of_stories_vectorizer_util` NMF) | `core/analysis/shape_reduction.py::nmf_reduce` | T, G | review | — | — | per-column shift to non-negative, not undone (documented) |
| CAP-GIS-10 | Geocoding (`GIS_geocode/location_util`) | `core/analysis/location_extract.py` + `core/gis/nominatim.py` | T, G | review | — | — | offline baked KB / Google (key from env) / Nominatim (no key) |
| CAP-GIS-11 | Pin maps and heatmaps (`GIS_heatMap*`, `GIS_Google_Earth_main`) | `core/gis/pins.py` + `tools/gis_map.py` | T, G | review | — | — | CSV in, pins/heat/KML/tour out; weight column drives pin radius |
| CAP-GIS-12 | SVO to GIS handoff: narrative events on a map | `core/analysis/svo_map.py` | T, G | review | — | — | one row per (triple, place) pair; verb-weighted, unlike a place-mention map |
| CAP-EMBED-10 | Word2Vec via BERT (`word2vec_BERT` family) | `core/analysis/word2vec_bert.py` | T, G | review | — | — | same vectors/neighbours/tsne contracts as `word_embeddings` so HW2 compares like with like |
| CAP-EMBED-11 | Document and sentence embeddings (new; no legacy tool) | `core/analysis/doc_embeddings.py` over `core/models` (ONNX sentence models) | T | review | — | — | FR-5.12; doc_pairs.csv in doc_similarity's shape, neighbours, clustered map, semantic search |

## Review log

Owner decisions are recorded here alongside reviews, since they change what
`verified` requires.

```text
Date: 2026-09-03
Packet: plan-wide (Gates A/B, FR-1.4, FR-1.7, FULL_REPLACEMENT_PLAN §8)
Reviewer: owner
Outcome: accepted (scope decision)
Evidence run: full quality gate green — 717 collected (713 passed + 4 skipped);
  ruff check, ruff format --check, mypy (140 files), compileall green
Findings: replacement is decided by class-workflow tool-completeness, not
  bug-for-bug legacy parity; per-capability legacy-run capture (FR-1.4/FR-1.7
  as originally scoped) dropped to spot-checks for tier-2 glue questions;
  definitional evidence (hand-computed values, independent library references)
  is first-class verification for package-backed analyses; the FR-1.4 runtime
  block gates nothing else.
Required follow-up: reviewer may move rows to `verified` on definitional
  evidence without waiting for FR-1.4.
Review commit: —
```

Append one entry per review. Do not rewrite rejected reviews; later reviews
should link back to them.

```text
Date:
Packet:
Implementation commit:
Reviewer:
Outcome: accepted | changes requested | blocked
Evidence run:
Findings:
Required follow-up:
Review commit:
```
