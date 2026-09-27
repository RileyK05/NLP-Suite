# Test Instability & Hang Audit — 2026-09-26

**Status:** Initial diagnosis below; release-pass fixes are recorded here. This document is the full medical chart:
everything found, ranked by how much damage it does, with the evidence quoted.
Written to inform the 0.5.0 cleanup. Tone is blunt on purpose — the goal is fixing,
not comfort.

**2026-09-26 release pass:** `conftest.py` now creates a fresh, short temp root
for each test session and sets `LOKY_MAX_CPU_COUNT=1` for Windows images whose
physical-core probe cannot run. Default pytest options disable the cache
plugin; CI uses an explicit short `--basetemp` and job timeouts. The one
unbounded cold-interpreter probe now has a 300-second subprocess timeout.
Plain pytest passed a targeted temp-path check; the full local run collected
4,539 tests (4,409 selected). Runs with `--basetemp` inside the long checkout
failed while publishing a figure: its path exceeded Windows' path limit and
could be enumerated but not opened. The same failing test passed under a short
`%TEMP%\p276` base; the 50-test desktop module also passed in order. A complete
non-model-integration gate using the short `%TEMP%\p283` base reached 100% and
exited successfully. A second complete gate on the final source tree did the
same with `%TEMP%\p288`. No teardown hang occurred. Model-integration tests
remain outside that default gate.

**Scope:** `New_NLP_Suite` (the active project). The legacy `NLP-Suite-1.6.38` tree and
the workspace root around it are included where they contribute to the disease.

---

## 1. Executive summary

The test suite does not have one bug. It has **three overlapping conditions**, which is
why every previous "fix" (faulthandler timeout, temp-dir redirect, basetemp dirs,
`-p no:cacheprovider` in one CI job) reduced symptoms without ever curing anything —
each fix addressed one mechanism while the others kept firing.

1. **The observed hang is almost never inside a test.** Every documented stall in the
   historical record reaches `[100%]` and then goes silent before the session summary.
   The stall lives in **session teardown / interpreter shutdown**, a phase covered by
   *no* timeout anywhere in the stack. The `faulthandler_timeout = 120` tripwire in
   `pyproject.toml:247` — installed specifically "so a hang is diagnosable instead of
   silent" — **cannot fire at the actual stall site**, because it only dumps while a
   test is executing. We built a smoke alarm that listens in the wrong room.

2. **The Windows filesystem layer on the development machine is chronically poisoned.**
   Directories named `pytest-of-moomi` exist in *both* temp locations with ACLs that
   exclude the user who owns them; eight leftover `--basetemp` directories in the
   workspace root cannot be opened at all; `.pytest_cache` writes fail with
   `PermissionError` at session finish. This produces both 355-fixture error storms
   and end-of-session crashes that masquerade as hangs.

3. **The suite has outgrown its safety rails.** It grew from 179 tests (Aug 31) to
   ~1,300–1,400 (Sep 16) to **3,258 tests in 207 files** (today). It spawns real
   subprocesses (kernels, job supervisors, warm workers, CLI probes) with only
   hand-rolled polling caps as guardrails. There is **no pytest-timeout, no xdist,
   no CI job timeout** on the Python jobs. Several of the longest silence periods are
   *by design*.

The through-line: **the suite gives you minutes of silence in at least four different
ways, and the tooling cannot distinguish "working" from "wedged" during any of them.**

---

## 2. Evidence timeline (what actually happened, run by run)

All from logs in the workspace root; pytest invocation and outcome verified per file.

| Log | When | Command | Outcome |
|---|---|---|---|
| `autonomous-refactor-logs/iteration-001.log` | Aug 31 | `python -m pytest -q 2>&1 \| tail -5` (in a worktree) | Output ends at `[100%]` (179 tests). No summary line ever printed. The calling agent counted dots, guessed "179 passed," and moved on. **A stuck run was recorded as a pass.** |
| `review-pytest.log` | Sep 16 09:43–09:44 | PowerShell, `python -m pytest --tb=short` | Progress to `[100%]`, then a `PermissionError` traceback inside **`pytest_sessionfinish`** → `cacheprovider.py:469` → `config.cache.set("cache/nodeids", ...)` on `.pytest_cache\v\cache\nodeids`. Interleaved with a PowerShell `NativeCommandError`. Summary never written. |
| `review-pytest-clean.log` | Sep 16 09:47 | full run | **355 errors, every one** `PermissionError: [WinError 5] Access is denied: 'C:\Users\moomi\AppData\Local\Temp\pytest-of-moomi'` at `tmp_path` fixture setup. Final: `931 passed, 4 skipped, 60 deselected, 355 errors in 85.96s`. |
| `hardening-boundary-final.log` | Sep 16 20:21 | targeted | One line: 21 dots + `[100%]`. No summary. Same truncated-at-100% signature. |
| `hardening-fixture-check.log` | Sep 16 20:27 | targeted | Same `pytest-of-moomi` PermissionError at fixture setup, stopped after 1 failure. |
| `review-pytest-final.log` | Sep 16 | full | **Healthy:** 1284 passed, 6 skipped, 60 deselected in 108.66s. |
| `hardening-full.log` / `hardening-full-verified.log` | Sep 16 | full | **Healthy:** 1389 / 1395 passed in 86–92s. |
| `.pytest_cache/v/cache/` | Sep 26 13:09–13:10 | — | `lastfailed` + 427 KB `nodeids` last written Sep 26. Last full session on record. `lastfailed` is non-empty — the last recorded run had failures. |

**Pattern:** the suite, when the filesystem cooperates, runs the default gate in
~90–110 s and passes. The failures cluster at the edges of the session — fixture setup
and session finish — not in the middle.

---

## 3. Condition 1 — The exit-phase stall (the thing you experience as "stuck")

### 3.1 What the logs prove

Every stall in the record ends at `[100%]`. No log anywhere contains a
`Timeout (0:02:00)!` faulthandler dump or a `KeyboardInterrupt` traceback. That absence
is diagnostic gold:

- `faulthandler_timeout` fires only while a test function is on the stack.
- Silence *after the last test* means the process is in one of: `pytest_sessionfinish`
  (cache write), pytest's terminal summary, `atexit` handlers, or interpreter
  shutdown (joining non-daemon threads, destroying C extensions).
- **None of those phases has a timeout.** A hang there is silent, traceback-free, and
  indefinite.

### 3.2 Mechanisms available at that phase, ranked by likelihood

**M1 — Blocked interpreter exit via leaked threads/processes.**
`desktop_backend/runner.py` is the flagship risk:

- `runner.py:214` — `Runner` owns a **non-daemon** `ThreadPoolExecutor`
  (`max_workers=1`). Non-daemon threads keep `python.exe` alive at exit; if one is
  blocked on I/O, the process *never* exits.
- `runner.py:631` — `close()` does `pool.shutdown(wait=True)`. If any submitted job
  never completes, `close()` blocks forever. The docstring even says
  "Pending/running work finishes before a graceful shutdown" — graceful, and
  unbounded.
- The persistent **warm worker** is a full second Python process carrying a spaCy
  cold import (`runner.py:234-276`). Killing it requires the `_stop_worker` path to
  actually run. The repo's own review (`review_findings_2026-09-16.md` at the
  workspace root) documents a `_dispatch` worker-desync branch that drops a live
  `Popen` without stopping it — "the process keeps running against the same
  workspace while a second worker starts alongside it."
- Cancellation is cooperative: `cancel()` closes pipes and hopes the blocked pipe
  read raises. The same review documents that a running analysis **cannot actually be
  interrupted** — the spellcheck grind runs 10–19 minutes per document inside the
  warm worker. A `shutdown(wait=True)` behind a job like that is a hang with a
  plausible deniability.

Tests generally do call `close()` (37 `Runner(` instantiations across 11 test files,
nearly all paired with `close()`) — so the exposure is precisely the *exceptional*
paths (cancellation races, desync, kill-failure) that the tests exist to probe. The
tests most likely to leave wreckage are the ones testing wreckage-handling.

**M2 — Session-finish exceptions swallowed by pipes/buffers.**
`review-pytest.log` is the proven instance: `PermissionError` in
`pytest_sessionfinish` writing the nodeids cache → no summary → the run *looks* hung
and, through a PowerShell pipe, even the error surfaces garbled. Any non-zero exit or
exception between the last test and the summary prints as silence when stdout is
piped (`| tail`, CI capture) because block buffering drops buffered content on
abnormal exit.

**M3 — Native crash at interpreter shutdown.**
The suite loads onnxruntime, spaCy/thinc, gensim, sklearn — C extensions with
nontrivial teardown. A hard crash after the last test loses buffered output
entirely. Not proven in the logs, but indistinguishable from M1/M2 in their
signature, and unfalsifiable without capturing the exit code of a stuck run.

### 3.3 Why the watchdog can't help

| Guard | Covers | Misses |
|---|---|---|
| `faulthandler_timeout = 120` | test bodies only | fixture setup/teardown, session finish, interpreter exit — **the observed stall sites** |
| `ThreadPoolExecutor` + `shutdown(wait=True)` | nothing (blocks forever) | itself |
| hand-rolled `_wait` polls (120 s) in job/kernel tests | their own subprocess | grandchildren of those subprocesses |
| nothing at CI level | Python jobs have **no `timeout-minutes`** | entire wedged job until GitHub's 6 h default |

---

## 4. Condition 2 — The poisoned filesystem (the enabler that keeps relapsing)

Verified live on 2026-09-26:

- `C:\Users\moomi\AppData\Local\Temp\pytest-of-moomi` — **Access denied**, even to
  `icacls` (the DACL grants no `READ_CONTROL` to the user at all; it was created by an
  elevated/SYSTEM-context process).
- `C:\Users\moomi\AppData\Local\Temp\nlp-tmp\pytest-of-moomi` — `icacls` shows only
  `NT AUTHORITY\SYSTEM:(F)`, `BUILTIN\Administrators:(F)`, `OWNER RIGHTS:(F)`.
  **The user is not on their own temp folder's ACL.** Created Sep 24.
- Eight directories in the workspace root (`hardening-tests-1..6`,
  `hardening-full-tests`, `review-pytest-temp-full`, `review-pytest-temp-probe`) are
  unreadable leftovers of `--basetemp` runs whose cleanup failed — physical evidence
  of past permission faults.
- Controlled Folder Access: **off**. Defender real-time protection: **on**. The
  exact agent that strips these ACLs is unproven (elevated-run history vs. antivirus
  interference are the candidates) — this needs one controlled experiment:
  delete both `pytest-of-moomi` dirs, run the suite, re-check ACLs.

### The cure that became a second infection site

`conftest.py:29-42` redirects `tempfile.tempdir` to `%TEMP%\nlp-tmp` at import time,
with a comment explaining the original `pytest-of-<user>` dir was "poisoned." That
was added **after** Sep 16. By Sep 24, `nlp-tmp\pytest-of-moomi` was poisoned too.
The workaround treats the symptom, relocates the patient, and the disease follows.
Note also that root conftest import-time **global mutation of `tempfile.tempdir`**
affects every library in the process — a heavy hammer, permanently installed, for a
local problem.

### Consequences observed

- 355/355 fixture errors at `tmp_path` setup in one run (`review-pytest-clean.log`).
- The `pytest_sessionfinish` crash in `review-pytest.log` (`.pytest_cache\v\cache`
  permissions — same disease family, different directory).
- Failed basetemp cleanups accumulating in the workspace root.

---

## 5. Condition 3 — Suite structure: legitimate silence vs. wedged silence

The default gate is `-m 'not model_integration'` (`pyproject.toml:246`), deselecting
60 tests. What remains is ~3,200 tests, of which a meaningful minority spawn real
operating-system processes. Inventory of the hang-risk surface, with bounding status:

### 5.1 Subprocesses

| Site | Spawns | Bound? |
|---|---|---|
| `tests/test_live_analysis.py:131-139` | Cold-interpreter engine preload probe | **NO `timeout=` — the only unbounded subprocess in the suite.** This is the exact path that historically deadlocked under the Windows loader lock (see its own docstring at `:110-128`). |
| `tests/test_no_lazy_backend_imports.py:126-132` | Same probe | `timeout=300` — up to 5 min of silence **by design**. |
| `tests/test_script_kernel.py:95-136` | Real kernel subprocesses | 120 s poll cap; correctness depends on Windows process-kill actually working. A leaked grandchild keeps pipes open and blocks exit (M1). |
| `tests/test_jobs.py:13-26` | Supervisor + tool interpreters per `submit()` | 120 s cap; the file's own comment admits the cap exists because "under a full-suite run the in-process model tests contend for CPU/disk long enough to starve interpreter startup" — the suite is resource-starving its own children. |
| `tests/test_notebooks.py:246-254` | `redraw.py` fresh interpreter | `timeout=120`. |
| `tests/test_viz_gate2.py:562,584,605` | `python -m tools.unified` ×3 | `timeout=60`. |
| `tests/test_production_boundary.py:62-70` | `python -c "import time; time.sleep(60)"` | bounded, but requires `runner.close()` to reap the 60 s child. |
| `tests/test_desktop.py:259-271` | cancellation-vs-500-doc-parse | `join(timeout=120)`; fails loudly if cancel blocks. Good test. It exists because the underlying bug existed. |

### 5.2 Heavy work that runs in the default gate

- sklearn NMF (macOS warning guard only, `conftest.py:57-64` — runs raw on Windows).
- Real gensim LDA in `test_profiler_lda_outputs.py:40`, `test_topic_stability.py:155`,
  `test_panels_lda_stability.py:141`.
- One real kaleido/Chrome export (`test_viz_final_blockers.py:464-473`); everywhere
  else kaleido is monkeypatched away — so the one real path is also the one with the
  least coverage reinforcement.

### 5.3 What is excluded and why it matters

The `model_integration` subset (~60 tests) includes `tests/test_live_analysis.py:319-439`,
where ~8 tests each call `Bench.warm` into a **fresh `tmp_path`** — no parse sharing —
and the module docstring prices a 20-speech parse at ~30 s. That is **~4 minutes of
silence in one file alone**, plus the one-time 600k-token real-corpus parse ("most of a
minute," `conftest.py:87-89`). These are the tests a developer runs *explicitly* — and
the ones most likely to be running when someone says "it's stuck again."

No live network access exists anywhere in the suite (all HTTP faked or localhost) —
this is a genuine strength and worth preserving.

### 5.4 Process-hygiene findings inside the suite

- No GUI/event-loop risk; `tests/test_layering.py:36` actively forbids tkinter/
  streamlit below the app layer. Also good.
- No file-locking primitives anywhere; SQLite is per-test in tmp dirs. Good.
- `filterwarnings = error` makes the suite fragile to dependency noise (a starlette
  pin, commit `70948ba`, already exists to paper over one such case). Correctness
  discipline, but each new dependency release is a roulette spin.
- `pytest 9.1.1` + `pytest-asyncio 1.4.0`; **no pytest-timeout, no xdist**. The
  355-error log shows the asyncio plugin wrapping `tmp_path` setup — every fixture
  goes through a plugin layer the suite doesn't control.

---

## 6. Conftest & config: accumulated scar tissue

`conftest.py` (root, 336 lines) is doing real work and wearing real scars:

- **Import-time global `tempfile.tempdir` mutation** (line 42) — see §4.
- `os.environ["NLP_SUITE_MODELS"]` / `NLP_SUITE_MODELS_DIR` forced to an empty dir
  at import (lines 49-53) so real models never leak into tests. Correct goal;
  environment-surgery-at-import is the mechanism.
- `without_wordnet` fixture (lines 261-290) whose 30-line docstring documents two
  *different* historical ways tests silently stopped testing anything once WordNet
  got installed on a dev machine. The tests were green. They were testing nothing.
- `tiny_models` (lines 300-336) monkeypatches module-private registries
  (`registry._BY_NAME`, `registry.MODELS`, `_files.FILES`) and clears onnx session
  caches on teardown. Fine as far as it goes — but it's private-surface surgery that
  breaks silently if those modules are refactored, and it reaches into
  `onnx_backend._sessions` directly.
- `has_spacy_model()` does a real `spacy.load` inside fixture resolution.
- The real-corpus fixtures (`conftest.py:106-141`) point at
  `~/Downloads/POTUS State of the Union 1934-2024` — a machine-local path, present
  on this machine, silently skipping on any other. Presently only consumed by
  `model_integration` tests; fine, but it's a landmine for "works on my machine"
  reasoning.

None of this is malpractice — each entry has a genuine war story in its docstring.
The problem is that the *accumulation* is the test infrastructure, and the war
stories keep multiplying.

---

## 7. CI vs. local: identical gate, wildly different environments

- CI offline job (`.github/workflows/ci.yml:67`) runs the same
  `-m "not model_integration"` filter; redundant with addopts but harmless.
- CI installs a pinned `desktop/requirements-runtime-py312.txt` **including the spaCy
  model wheel** — so CI's default gate has a parser and local runs may not.
- `model_integration` is a separate Ubuntu-only job that downloads models
  (`ci.yml:133-136`). Windows-only pathology (this audit) is invisible to it.
- One CI desktop job learned the lesson the hard way: commit `edd9847` adds
  `mkdir -p out/ci-pytest` before `--basetemp=out/ci-pytest` ("pytest does not create
  --basetemp's parent") **plus** `-p no:cacheprovider` — the exact combination that
  avoids the `review-pytest.log` crash. The Python test jobs never got the same
  treatment, and none of the Python jobs has a `timeout-minutes`.
- Consequence: CI cannot reproduce the dominant local failure mode, and a wedged CI
  job burns until GitHub's 6-hour default ceiling.

---

## 8. The workspace around the project (process debt, visible from space)

Observed at the workspace root, not audited to completeness (a dedicated sweep was
cut short; treat this section as confirmed floor, not ceiling):

- ~30 stray run logs at the root (`hardening-*.log`, `review-*.log`,
  `interactive-*.log`) — a graveyard of one-shot agent/debug runs kept as the only
  record of what happened, because the runs themselves left no other trace.
- `review_findings_2026-09-16.md` at the root — a serious, specific engineering
  review (Runner leaks, cancellation-can't-interrupt, worker desync) — **living
  outside the project it reviews**, where nothing tracks it and nothing will close it.
- `corenlp_probe.json`, `node_modules/.vite` at the root of a Python workspace,
  a `hardening-ui-workspace/` containing a stale `.server.lock` and a
  `workspace.sqlite3` — residue of interrupted sessions, some of which hold locks.
- `New_NLP_Suite/.tmp_wordcloud/` — what appears to be a vendored copy of
  matplotlib/numpy packages inside the repo itself.
- `New_NLP_Suite/desktop/.toolchain/` — an entire embedded Python distribution.
  `desktop/src-tauri/target/` and committed build outputs under
  `desktop/src-tauri/binaries/` — build artifacts cohabiting with source.
- `docs/` had accumulated completed plans, reviews, and command logs alongside
  current guidance. The completed documents were consolidated after this audit;
  Git history retains the original evidence. The active 0.5.0 plan, replacement
  ledger, and test-instability findings remain available.
- `NLP-Suite-1.6.38/` — a full legacy tree with its own 326-test suite and its own
  pytest config, sitting next to the rewrite. Only a hazard if pytest is ever invoked
  from the wrong directory; otherwise dead weight.

None of this caused the hangs. All of it raises the cost of every future diagnosis —
this audit included — because "what is this and is it safe to touch" has no
trustworthy answer anywhere in the tree.

---

## 9. Differential table (symptom → actual mechanism)

| What you see | Actual mechanism | Confidence | Evidence |
|---|---|---|---|
| Run finishes tests, never prints summary | Blocked interpreter exit: non-daemon pool thread / `shutdown(wait=True)` behind a stuck job / leaked worker or grandchild holding a pipe | High (mechanism proven available; specific instance unproven — postdates logs) | All 4 stalled logs end at `[100%]`; `runner.py:214,631`; `review_findings_2026-09-16.md` |
| Run "hangs," then errors weirdly or silently | Exception in `pytest_sessionfinish` (cache write `PermissionError`), garbled/lost through pipes | **Proven** | `review-pytest.log` |
| Hundreds of errors at fixture setup | ACL-poisoned `pytest-of-moomi` in both temp roots | **Proven** | `review-pytest-clean.log` (355 errors); live `icacls` 2026-09-26 |
| Minutes of silence mid-suite | By-design long probes (up to 300 s), subprocess startup under full-suite CPU contention | **Proven available** | §5.1 |
| Undeletable `hardening-*`/`review-*` dirs | Failed basetemp cleanup on same filesystem disease | **Proven** | workspace root; `ls`/`icacls` access denied |
| "Tests passed" that didn't | Buffer loss on abnormal exit; agents counting dots | **Proven** | `iteration-001.log` |
| Tests green but testing nothing | Environment-dependent guards (WordNet history), machine-local corpus paths | **Proven historically, fixed per docstrings** | `conftest.py:261-290` |

---

## 10. Unproven but plausible (flagged, not charged)

- **Native crash at shutdown** (M3) — indistinguishable from M1/M2 in logs; requires
  exit-code capture on the next stuck run.
- **The ACL-stripper's identity** — elevated-run history vs. antivirus; requires the
  delete-and-re-check experiment in §4.
- **Which specific test leaves wreckage today** — the Sep 16 logs predate current
  code; needs a fresh instrumented run (§11).
- **`.tmp_wordcloud/` provenance** — not investigated; flagged in §8.

---

## 11. Recommended diagnostics (before any treatment)

These are measurements, not fixes — they decide which treatment applies.

1. **Instrument the exit phase.** Add a session-scoped autouse fixture (or conftest
   hook) that arms `faulthandler.dump_traceback_later(180, repeat=True)` and cancels
   it in a `pytest_sessionfinish`-ordered hook after interpreter-exit begins... in
   practice: run the suite under `py-spy` or with `PYTHONFAULTHANDLER` + a wrapper
   that dumps all thread stacks if the process outlives the summary by >60 s. The
   first stuck run then names its own murderer.
2. **Capture exit codes.** Every run wrapper logs `%ERRORLEVEL%` / `$LASTEXITCODE`.
   Separates "wedged" (still running) from "crashed" (non-zero) from "killed."
3. **The ACL experiment.** Delete both `pytest-of-moomi` dirs, run the suite, re-check
   ACLs. If they re-corrupt, watch which process owns the handle (Resource Monitor /
   `openfiles`, or Sysinternals Handle) during the run.
4. **Run with the CI belt on locally:** `-p no:cacheprovider --basetemp=<fresh dir>`
   in a known-good parent. If stalls vanish, M2/the filesystem disease was the whole
   local story and M1 remains latent.
5. **Reproduce the Runner wreckage deterministically:** run only the
   cancellation/desync tests (`test_desktop.py` Runner block) in a loop; check
   `tasklist` for orphaned `python.exe` after the run "completes."

---

## 12. Treatment directions (for planning only — not started)

Ranked by expected value per unit of effort; the user decides.

1. **Timeout coverage where the stalls actually live** — session/exit-phase watchdog
   that *fails loudly* instead of hanging silently; CI `timeout-minutes` on Python
   jobs. Cheap, immediately converts every future stall into a named failure.
2. **Filesystem hygiene** — purge poisoned dirs, adopt the proven CI combination
   (`--basetemp` into a project-local dir + `-p no:cacheprovider`) as the *default
   local* invocation, and find the ACL-stripper (§11.3).
3. **Runner lifecycle guarantees** — `close()` with a bounded drain, worker/child
   reaping in `__del__`/atexit or a documented owner, closing the documented
   desync-leak and cancellation-interrupt gaps from `review_findings_2026-09-16.md`.
   This is correctness work that also happens to be the hang fix.
4. **Suite diet** — cap the unbounded probe (`test_live_analysis.py:131`), share
   parses in the model_integration warm tests instead of re-parsing per test,
   and consider whether 3,258 tests need every one of their subprocesses.
5. **Repo quarantine** — move run logs/reviews into the project or delete; decide the
   fate of `.tmp_wordcloud/`, root `node_modules`, and the legacy tree. The old
   command log was removed during docs consolidation; retain recurring failure
   modes and recovery steps in this audit or the developer guide.

---

*Prepared 2026-09-26. Sources: workspace-root run logs (Sep 16 & Aug 31), live
filesystem/ACL inspection (Sep 26), `New_NLP_Suite` source and test inventory,
`review_findings_2026-09-16.md`, CI workflow files. No code was modified.*
