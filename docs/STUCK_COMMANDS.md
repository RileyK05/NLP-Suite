# Stuck / failed command log

A running list of commands that stalled, hung, or misbehaved — kept so the
patterns behind them get fixed deliberately instead of rediscovered. Newest
at the top. Companion to "Things that will bite you" in `docs/PANELS_BACKLOG.md`
§6 (Windows MAX_PATH, leftover servers holding `.server.lock`).

---

## 2026-09-22 (panels frontend session)

### 1. Full-suite pytest run stalls in `tests/test_desktop.py` — UNRESOLVED

**Command:**
```
cd NLP-Suite && python -m pytest -m "not model_integration" -q --basetemp=C:/nlptmp 2>&1 | tail -5
```

**What happened (three separate attempts):**
- Run 1: four `test_desktop.py` failures reported (archive_roundtrip,
  real_background_analysis_and_export, desynced_worker_is_reaped,
  two_runs_compared) — but each **passed in isolation** immediately after.
- Run 2 (isolated `test_desktop.py`, fresh basetemp `C:/nlptmp3`): **all 50 passed**.
- Run 3 (full suite): different subset of `test_desktop.py` failed/stalled
  (warm_worker_reuse, real_background_analysis_and_export).
- Run 4 (`test_desktop.py` alone with `-p no:cacheprovider`): produced **no
  output at all** — session ended while it ran. Two python processes were left
  alive afterwards and had to be killed (PIDs 20324, 71724).

**Suspects, in order of likelihood:**
1. **Runner/worker tests hold resources across the suite.** These tests spawn
   real worker processes and wait on wall-clock deadlines (40–60 s of
   `time.sleep(0.1)` polling). Under a full suite the machine is already busy,
   so the deadlines are approached rather than met, and near-misses flip
   between pass and fail between runs.
2. **Leftover state between runs.** A stale `.server.lock` (embedded PID 0 —
   a crashed writer's placeholder) sat in `out/desktop-workspace/` across
   several of these attempts. Removed 2026-09-22. This is the first suspect
   next time: check the lock *before* the run, not after.
3. `--basetemp` reuse. One run had written `test_a_failed_export_fails_the0`
   into a directory a later run shared. Never share a basetemp across runs.

**Reproduce / next steps:**
- Run only this file, fresh temp dir, output to a file:
  `python -m pytest tests/test_desktop.py -q --basetemp=C:/nlptmp > dt.log 2>&1`
- If that passes, run the full suite *without* piping through `tail` (see §6
  note in the backlog: piping hides a slow test's progress) and record where
  it stops. `pytest --timeout` (pytest-timeout) is not currently a dependency;
  if a run stalls again, `-p no:cacheprovider` + `faulthandler_timeout=120`
  (already in pyproject) should dump thread stacks — capture them.

**Status:** unresolved. Isolated runs pass; full-suite runs are unreliable.

### 2. Leftover processes / lock from a killed run — recurring hazard

- `out/desktop-workspace/.server.lock` contained `0…0` (PID 0 placeholder),
  i.e. a writer died mid-write. `scripts/desktop_preview.py` refuses to start
  while it exists. Removal rule (scripted, do not hand-delete):
  pid parses to 0 → remove; pid alive → leave; pid dead → remove.
- After every pytest run that gets interrupted: check
  `Get-Process python*`, kill leftovers, then check the lock by the rule above.

### 3. Non-issue worth remembering

- `pytest --timeout=300` failed with an inifile error because pytest-timeout
  is not installed; `faulthandler_timeout=120` in pyproject covers hangs
  instead. Do not add `--timeout` to commands until the dependency exists.

---

## 2026-09-22 (topic-flow session, later)

### 4. `TestLdaReference` (all tests, with new flow tests) hangs at ~15 min — UNRESOLVED

**Command:**
```
python -m pytest "tests/test_lda.py::TestLdaReference" -q -m model_integration --basetemp=C:/nlptmp
```

**What happened:** no output for 15 min; the command was killed by the session
timeout and **no python processes were left behind** (nothing to kill), so the
hang was inside pytest itself — likely a gensim fit inside
`test_segments_do_not_change_the_document_level_fit` or
`test_flow_scores_each_paragraph_and_reports_the_rule` running its
`passes=10` fits on a corpus that, in the flow tests, passes `_doc_tokens()`
plus segment scoring. `faulthandler_timeout=120` in pyproject **did not fire
or its output was lost** with the kill — worth checking whether faulthandler
dumps go to stderr and get lost through the pipe.

**Suspects:**
1. The two new flow tests each fit LDA **twice** (once baseline, once with
   segments) — 4 fits in the class where there were 2 before, with
   `passes=10` defaults. Gensim fits on the cats/cars fixture are normally
   fast, but `plan_segments` + token alignment over the fixture frame may be
   quadratic somewhere (the `_align_tokens` call).
2. `get_document_topics` called per segment — fine — but `doc2bow` per
   segment per row loop in `_score_segments` may be slow if a fixture
   document has many rows.

**Reproduce / next steps:**
- Run the two new tests alone with `-x --timeout` once pytest-timeout exists,
  or with `-p faulthandler --faulthandler-timeout=60` and keep the output.
- Time `plan_segments` on the fixture frame first: it is fast (0.003 s over
  the 6-doc fixture, verified 2026-09-22), so the hang is **inside the gensim
  fit or a spawned subprocess**, not segmentation.
- **Spawning hazard:** anything importing gensim inside a
  `python - <<EOF` heredoc spawns multiprocessing children whose
  `_fixup_main_from_path('<stdin>')` fails and *loops*. Never time gensim
  code from stdin — write a script file under the approved temp dir and run
  it. This alone explains one "hang".
- The `model_integration` marker means CI does not run these; still must
  finish before the feature ships.

**Status:** unresolved — flow feature is built but its engine-derived tests
have not yet passed end to end. Next step: run the two flow tests alone as a
*.py file (not stdin) with faulthandler kept.

### 5. Non-issue worth remembering (from this session)

- `pytest.approx ==` against a pandas Series is **not elementwise**; it
  silently returns all-False (found in test_lda_views). Use `np.allclose`.
- `python -m pytest tests/test_lda.py::TestLdaReference` without
  `-m model_integration` exits 5 (no tests collected) — the class is
  marker-gated. Not a hang, just easy to misread.

---

## 2026-09-22 (panel features session, evening)

### 6. Frontend flaky: `ExplicitRun.test.tsx` "really does edit the field" — UNRESOLVED

**Command:**
```
cd desktop && npx vitest run --silent
```

**What happened:** full-suite run reported 1 failed (`ExplicitRun.test.tsx >
the test's own instrument > really does edit the field`, 6 s into a 13.8 s
file) and one other test file failed in an earlier run; the same file
**passed in isolation** seconds later (11/11 in 2.2 s). The test is itself an
instrument check (the prototype value setter) and timed out or raced under
the full suite's 20 parallel workers.

**Suspects:** worker contention under vitest's `isolate: true` default (the
run's own output suggests ~604 ms/worker startup); 20 workers × jsdom
environments. `isolate: false` in vite config would likely fix it, at the
cost of shared worker state — needs its own deliberate change, not a quick
flag flip mid-feature.

**Status:** unresolved; same family as the Python runner-test flakiness —
resources under parallel load.

### 7. Non-issue worth remembering (from this session)

- `cd desktop && ...` inside a `workdir` set to `desktop` fails (`cd` is
  already there); pass the command directly with the right `workdir`.

- Full-suite runs stalling inside `tests/test_desktop.py` — the same hazard
  as item 1, seen twice before this log existed (noted in
  `docs/PANELS_BACKLOG.md` §1 as "NOT COMPLETED this session").
- Bash heredocs mangling `\n` / `\b` escapes when writing source files
  (three occurrences; use the Write/Edit tools instead — see
  `docs/PANELS_BACKLOG.md` §6).
---

## Older (pre-backendlog sessions)

- Full-suite runs stalling inside `tests/test_desktop.py` — the same hazard
  as item 1, seen twice before this log existed (noted in
  `docs/PANELS_BACKLOG.md` §1 as "NOT COMPLETED this session").
- Bash heredocs mangling `\n` / `\b` escapes when writing source files
  (three occurrences; use the Write/Edit tools instead — see
  `docs/PANELS_BACKLOG.md` §6).
