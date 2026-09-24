# Post-implementation review — 2026-09-05

This is a targeted correctness review of the expanded working tree, not a
sign-off on full replacement of NLP Suite 1.6.38. Existing implementation
changes were preserved. Regression coverage is in
`tests/test_followup_review.py`.

## Corrections made

1. **Background jobs outlive their submitter.** A separate supervisor now
   owns the tool process and records its final exit status. Previously the
   submitting CLI's daemon thread disappeared on exit, leaving jobs marked
   RUNNING. The regression launches a real submitting CLI, waits for that
   process to exit, then verifies DONE and captured output. Status JSON is
   atomically replaced, spawn failures produce a nonzero CLI result, malformed
   JSON shapes produce diagnostics, and job IDs cannot traverse outside the
   jobs root.
2. **Resume happens before execution.** Previously the profiler executed the
   whole plan before looking for reusable children, then executed misses a
   second time. It now determines the pending plan first; cache hits are not
   executed and misses execute once. Only pending tools request parsing.
3. **Resume verifies evidence.** New child records hash every output and the
   envelope. Missing, modified, malformed, or escaping records are cache
   misses. Reuse links are accepted only for the same resolved output root.
   Raw-text reuse keys include core-source hashes, installed numerical-package
   versions, parameters, corpus fingerprint, and explicit parameter-file
   contents. Changing a wordlist at the same path invalidates reuse.
4. **Profiler parameters work end to end.** The CLI accepts `--params FILE`,
   a JSON object mapping selected tools to parameter objects. Its default
   selection excludes tools that need mandatory analyst inputs. CoNLL search
   requests a parse; plain-text search does not. Nonfinite numeric parameters
   fail validation. VADER/ANEW, SentiWordNet/hedonometer, and NRC adapters honor
   their field and lexicon options, including `Path` inputs where allowed.
5. **Partial results stay visibly partial.** The executor uses `Result.ok`,
   not just value presence. Partial outputs survive publication with failed
   child status and ERROR diagnostics; the parent envelope includes outcome
   diagnostics. Shared parse errors mark dependent analyses as failed in
   both the CLI and app, while raw-text analyses remain independent. Partial
   children cannot be reused. Invalid batch output locations return a
   diagnostic instead of an uncaught constructor error.
6. **Registry form keys are unique.** Removed the duplicate `anew-field`
   parameter and added registry validation for duplicate parameter names.
7. **Blocked rename chains preserve inputs.** Filtering now removes blocked
   predecessors transitively before moving anything. With a→b, b→c and an
   occupied c, neither a nor b moves; independent renames can still proceed.
   Repeated sources/destinations are rejected and occupied destinations are
   checked again immediately before final moves.
8. **Comparison cannot report false success.** A CSV schema/configuration
   error cannot become a successful run-level report with an empty diff.
   Unordered numeric comparisons use maximum bipartite matching: overlapping
   tolerance ranges no longer make a greedy first match incorrectly fail
   an otherwise equivalent multiset.
9. **Writer metadata cannot be overwritten as an artifact.** `result.json`
   is reserved for the envelope. A preexisting schema sidecar prevents a
   table write before either file is changed.

## Resume limits (intentional)

Parsed, model-dependent, asset-dependent, and optional-package analyses are
rerun. Authoritative identities for their actual shared parse, backend model,
and resolved assets are not yet passed through the batch reuse contract.
Using only a tool version and corpus fingerprint would risk returning stale
results. Restoring reuse for those analyses requires recording those identities
at execution time, then testing invalidation when each dependency changes.

Old manifests without output hashes are also rerun. Resume across different
output roots recomputes results; it does not create dangling relative links.
The checks protect against stale or damaged caches, not a malicious party
rewriting both manifests and their checksums.

## Profiler parameter example

`profile-params.json`:

```json
{
  "search": {"mode": "text", "query": "World", "case-sensitive": true},
  "spellcheck": {"wordlist": "assets/my-wordlist.txt", "threshold": 80.0}
}
```

```text
python -m tools.profiler CORPUS OUT --analyses search,spellcheck --params profile-params.json
python -m tools.profiler CORPUS OUT --analyses search,spellcheck --params profile-params.json --resume-from OUT/PRIOR_PROFILER_RUN
```

File parameters resolve relative to the process working directory, like the
standalone CLIs, not relative to the JSON file.

## Verification

The unmodified baseline test run produced **973 passed, 16 skipped, 1 failed**.
The failure was a joblib/scikit-learn warning promoted to an error while
detecting physical CPU cores on this Windows environment during t-SNE.
Tests were subsequently run with explicit CPU limits, without changing
production defaults or suppressing warnings:

```powershell
$env:LOKY_MAX_CPU_COUNT = '1'
$env:OMP_NUM_THREADS = '1'
python -m pytest -p no:cacheprovider
python -m ruff check .
python -m mypy core tools app --follow-imports=silent
git diff --check
```

Final results: **1,014 passed, 16 skipped** in 61.64 seconds; all Ruff checks
passed; mypy passed for 177 source files; `git diff --check` passed. The new
review test file contains 40 passing regression cases.

These tests do not constitute a fresh legacy-output capture, a full GUI
usability assessment, or verification of every optional external service/model.
No release ledger items were promoted to verified by this review.
