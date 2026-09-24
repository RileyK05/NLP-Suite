# NLP Suite desktop beta

## End-user hardening — 2026-09-16

The source and frozen Windows engine have a restricted desktop catalog, simpler
setup guidance, fictional sample texts, and additional job/export reliability
fixes. See [hardening scope and verification](PRODUCTION_HARDENING.md). The older
installer documented below does not contain this work; rebuild and validate a
new installer before distributing these changes.

## Artifact viewer update — 2026-09-15

The source viewer now isolates HTML charts from the main application's CSP.
See [artifact preview design and testing](ARTIFACT_PREVIEWS.md) for restart
instructions, security boundaries and the outstanding installer/visual checks.
The older installer described below does not include this fix.

## Release hardening — 0.3.0 (2026-09-09)

**Windows installer completed 2026-09-10:**
`out/distribution-0.3.0-windows-x64/NLP Suite_0.3.0_x64-setup.exe`.
That folder also contains `SHA256SUMS.txt` and `START-HERE.txt` for recipients.
Use the setup EXE, not the standalone launcher or the older unversioned runtime.
The installer is 392,161,916 bytes (about 374 MiB), including offline WebView2.
SHA-256: `b177e6507c3d3c0d17abd1cc1542a2fdb7f87ea1fcf3742751119f308ef02c51`.
This is an unsigned Windows x64 beta candidate, not a clean-machine acceptance
signoff or a Mac/Linux binary. The native multi-platform workflow is prepared
but has not been pushed or run as part of this local pass.

See [Desktop release handoff](DESKTOP_RELEASE.md) for native Windows, Mac and
Linux builds, recipient instructions and the clean-machine acceptance gate.
This pass adds authenticated startup readiness, runtime platform/version checks,
single-instance window activation, an offline Windows WebView2 installer,
portable filename/archive checks, restore rollback and visible prerequisite
guidance. Resource uploads block analysis submission until complete.

Build and verification results for older versions below are historical; they
must not be used to certify 0.3.0 or unbuilt Mac/Linux packages. Packaging does
not change the scientific replacement ledger or bundle restricted assets.

The resumed verification run passed 1,098 Python tests with 17 skipped,
including model-integration tests in the selection. The CLI status reader now
retries brief Windows sharing/access errors (including CRT EACCES without a
`winerror`) for a bounded interval, matching the existing atomic writer policy.
Focused reader tests, desktop mypy and Ruff checks passed. The frontend has
47 passing tests; the native launcher has four passing Rust tests.

The 0.3.0 frozen Windows engine passed 16 workflow smoke cases, CSV/ZIP exports,
project backup/non-overwriting restore and graceful shutdown. The separate
87-document corpus check passed folder import, duplicate detection, Readability
(87 rows), Document Similarity (3,741 pairs), NER and the seven CSV workflows.
Evidence workspaces: `out/smoke-030-release`, `out/corpus-030-release` and
`out/installed-payload-check`. The last also passed the native launcher's
authenticated runtime startup/shutdown check. It did not install the EXE or
exercise a native webview. Clean-machine installation/UI acceptance and native
Mac/Linux builds are still required; Chrome was intentionally left alone in
the resumed pass after the user's browser issues.

## Review update — 0.2.2 (2026-09-09)

The desktop source now has a single result-table loading path. Changing a
search, page or artifact cancels the superseded request and clears old rows
while loading, rather than presenting stale rows under a new selection.
Required choice fields initialize to their first visible option when no
explicit default is declared. CSV statistics guidance no longer implies that
an imported text corpus is required.

This is a frontend-focused update using the existing verified frozen Python
runtime. It does not add analysis algorithms or advance legacy-parity status.
Separately, the source CLI job supervisor now retries brief Windows
status-file sharing/access errors for at most seven attempts (0.63 seconds of
backoff). Permanent errors still raise and the old status is never truncated.
This fixes a reproduced completed-tool/unchanged-RUNNING-status failure;
it is not a change to the desktop's separate SQLite job runner. The frozen
runtime was not rebuilt for this source-CLI-only change.
The completed installer is `desktop/src-tauri/target/release/bundle/nsis/NLP Suite_0.2.2_x64-setup.exe`.
The earlier build evidence below remains historical, not evidence for new
changes. The default Python gate now excludes model-integration tests; use
`python -m pytest -m model_integration` for those, or override addopts to include
all tests in a single run.

Review verification: 1,052 Python tests passed and 16 skipped with the default
marker exclusion overridden (`-o "addopts=-q --strict-markers"`); 45 frontend
tests passed. Ruff and focused mypy checks passed, as did TypeScript/Vite.
The initial full run reproduced the Windows CLI status-publication failure;
the corrected full run passed. Skipped tests still represent unverified
environment-dependent coverage, not successful integrations.

## Latest update — 0.2.1 (2026-09-08)

Use `desktop/src-tauri/target/release/bundle/nsis/NLP Suite_0.2.1_x64-setup.exe`
for the new Learn tab. The installer build completed successfully. The older
0.2.0 build details below are retained as historical verification evidence.

Learn includes all 27 desktop tools: explanations, example research questions,
interpretation caveats, and live settings/output references. GLM via
Ollama/OpenCode supplied a generic educational draft; the integrated text was
reviewed and corrected against the implementation. No corpus text was sent.

The packaged runtime imported all 87 TXT files under `corpus/`, rejected repeat
imports as duplicates, produced 87 readability rows and 3,741 similarity pairs,
and completed named-entity analysis, exports, backup/restore and shutdown.
These checks used `out/corpus-verification-20260908`, not the user's installed
workspace. They do not establish that every optional analysis works on this
corpus. Repeat with:

```powershell
python scripts/smoke_desktop.py desktop/src-tauri/binaries/nlp-runtime/nlp-runtime.exe --workspace out/corpus-check-new --corpus corpus --with-parser --job-timeout 900
```

For this update, 35 desktop backend tests and 41 frontend tests passed;
TypeScript/Vite and native packaging passed. The browser loaded the updated
application and showed Learn in navigation, but interactive visual inspection
was blocked by the usage/approval limit. The full Python suite was not rerun
for this frontend-focused update; its earlier result remains recorded below.

The desktop application is a local Tauri/React interface over the existing
Python engines. It preserves Streamlit and the command-line tools. Version 0.2
expands the six-workflow preview; it is **not a full legacy-parity signoff**.

## What is implemented

- 20 registered corpus workflows and seven explicit CSV statistics workflows.
  The catalog is searchable and generated from the execution contracts.
- Native file/folder selection, drag/drop, and browser file uploads.
  TXT, CSV, TSV, HTML, PDF, DOCX and RTF intake use the existing converters.
  Legacy binary DOC and scanned/image-only PDFs require external conversion/OCR.
- Original-file custody: converted originals and extracted text are both kept,
  with separate checksums and conversion diagnostics. Same-name original files
  with different bytes remain distinct, even if extraction produces equal text.
- Background worker isolation, persistent run states, and queued/running
  cancellation. Cancelled states cannot be overwritten by late worker updates.
- Auxiliary resource file choosers. Selected CSVs/lexicons are copied at run
  submission and checked before execution; later edits to the source do not
  change the accepted run.
- Result tables with full-dataset substring search and 500-row pagination.
  Optional charts show the first 60 rows of the current page, not a hidden
  whole-corpus aggregation. CSV and complete-run ZIP exports remain available.
- Project renaming, recoverable archiving, and archived-project restoration.
  Archiving hides a project without deleting its files or results.
- Portable `.nlpsuite` backup/restore: documents, originals, run files and
  metadata, with checksums, bounded extraction and traversal rejection.
  Restoring creates a separate project; it never overwrites an existing one.
- Native exports stream from the authenticated local engine into a temporary
  destination file, then publish after successful transfer. They do not pass
  entire ZIPs through JavaScript arrays or truncate an existing file on failure.
- Dependency notices generated from the packaging environment and readable in
  Environment & setup. Restricted research lexicons are not bundled.

## Try the Windows build

The completed installer is
`desktop/src-tauri/target/release/bundle/nsis/NLP Suite_0.2.0_x64-setup.exe`.
Use this **0.2.0** installer, not the older 0.1.0 file.
The 2026-09-06 build is 130,130,843 bytes (about 124 MiB), with SHA-256
`7738e26e30d1b1079e26e3f453f49e341aebf7b38d1ee04616480725ab9333bd`.
Build products are ignored by Git; committing the source does not upload the
installer. A future rebuild may have a different checksum.
This is an unsigned Windows x86-64 beta; signing and clean-machine installation
acceptance are still outstanding. WebView2 must be present; its bootstrapper
may need network access on machines without it.

1. Create a project and add files or a folder.
2. In Analyses, choose Readability or Named entities and run it.
3. Open Runs & results. Inspect the table, search, change pages, or chart a
   numeric measure. Export the run and check `result.json`.
4. For CSV statistics, create/select a project (it may have no corpus), choose
   a table analysis, select its CSV resource, and specify exact column headings.
5. In Environment & setup, back up the project. Restore that backup and verify
   that both projects coexist with readable results.
6. Try renaming and archiving a project, then restore it from Archived projects.
7. Submit a longer analysis, cancel it, and verify its terminal state.
8. Close and reopen: projects and completed results should persist.

Closing requests graceful engine shutdown: accepted work finishes in the
background. To stop work immediately, cancel the runs before closing. A
workspace cannot be reopened while its prior engine still holds the lock.
Abruptly interrupted jobs become INTERRUPTED on the next startup.

The native workspace normally lives at
`%LOCALAPPDATA%/org.nlpsuite.desktop/workspace`; Setup displays the actual path.
Do not edit copied inputs inside it. Back up the entire workspace while stopped
if doing a manual filesystem backup; SQLite alone is not the whole project.

## Runtime scope

The expanded packaging recipe bundles Python, spaCy and `en_core_web_sm`,
converters, VADER, Gensim and NLTK. It can also include locally downloaded
WordNet data. Actual model execution is checked by the frozen-runtime smoke
test; a package-presence badge alone is not a health certificate.

Optional/restricted inputs still matter:

- ANEW, NRC, concreteness, iconicity and other research lexicons remain
  user-supplied. The combined VADER/ANEW workflow requires the ANEW input.
- Stanza and transformer libraries/models are not part of this Windows bundle.
  Those workflows require an appropriately configured source environment.
- The mention-grouping tool is explicitly a **lemma baseline**, not a neural
  coreference replacement.
- spaCy dependency parsing does not supply CoreNLP constituency trees. A
  dependency-complexity run is not evidence of full constituency parity.
- GIS, PC-ACE/database and other non-corpus CLI workflows are not all integrated
  into this desktop UI. Their existing implementations remain available.

## Source development

From `NLP-Suite`, with Python 3.12 and Node/npm:

```powershell
python -m pip install -e ".[desktop,spacy,converters,sentiment,topics,wordnet]"
python -m spacy download en_core_web_sm
cd desktop
npm.cmd ci
npm.cmd run build
cd ..
python scripts/desktop_preview.py
```

`desktop_preview.py` compiles the frontend itself when anything under
`desktop/src` (or `index.html`, `vite.config.ts`, `tsconfig.json`,
`package.json`) is newer than `desktop/dist/index.html`, so the interface it
serves is the one in the working tree. `--no-build` serves the existing bundle
as-is and `--rebuild` forces a compile. The `npm run build` above is therefore
optional on the first run, and needed only if npm is not on `PATH`.

The browser launcher uses `out/desktop-workspace` by default. Keep its terminal
running; Ctrl+C requests shutdown. The local URL contains an ephemeral token
that is removed from the address bar after connection. Do not share that URL.

For a native development window, install Rust stable and Visual Studio C++
build tools, then run `npm.cmd run desktop` from `desktop`. Set
`NLP_SUITE_PYTHON` to an absolute interpreter path when needed. The development
config removes bundled-resource requirements so a source checkout does not
need PyInstaller before native development.

## Build the expanded installer

Use an isolated packaging environment, not a miscellaneous ML development
environment. From `NLP-Suite`:

```powershell
python -m venv desktop/.toolchain/python
desktop/.toolchain/python/Scripts/python.exe -m pip install -r desktop/requirements-windows-py312.txt
desktop/.toolchain/python/Scripts/python.exe -m pip install --no-deps .
desktop/.toolchain/python/Scripts/python.exe -m nltk.downloader -d desktop/.toolchain/nltk_data wordnet
desktop/.toolchain/python/Scripts/python.exe scripts/build_desktop_backend.py --with-parser
python scripts/smoke_desktop.py desktop/src-tauri/binaries/nlp-runtime/nlp-runtime.exe --workspace out/desktop-smoke --with-parser
cd desktop
npm.cmd ci
npm.cmd run package
```

The runtime uses an unpacked PyInstaller directory, installed as a Tauri
resource directory. Keep its executable and `_internal` directory together.
This avoids unpacking the whole model bundle at each engine launch.
Resource mapping follows the
[Tauri resource-directory contract](https://v2.tauri.app/develop/resources/).

Rust must be on PATH. This checkout has a project-local toolchain; from
`desktop`, enable it for the current terminal with:

```powershell
$env:CARGO_HOME = Join-Path (Get-Location) '.toolchain/cargo'
$env:RUSTUP_HOME = Join-Path (Get-Location) '.toolchain/rustup'
$env:PATH = "$env:CARGO_HOME\bin;$env:PATH"
```

The manual GitHub Actions workflow `desktop.yml` builds this distribution and
uploads an installer artifact. It has not been run or published on GitHub.
npm/Cargo use lockfiles. `desktop/requirements-windows-py312.txt` records the
tested Python packaging environment (including the English model wheel).
This is version pinning, not a fully hash-locked or independently audited
release: the WordNet download is still separate and needs release asset
checksums. CI uses this snapshot too, with pytest installed separately for
source tests. The workflow itself has not yet been exercised on GitHub.

## Safety and limits

The engine binds only to loopback on a random port, requires a random bearer
token, checks origins/hosts, and validates artifact containment. Native exports
cannot redirect the local bearer token to a remote server. This protects
browser isolation, not against malware already running as the same local user.
Projects are not encrypted at rest and no telemetry is added.

Limits: 20 MB per document/resource, 2,000 documents per project, 512 MB
compressed project restore, 2 GB expanded archive, 20,000 archive file members.
Backup requires no active runs. Large browser-mode exports still use a browser
Blob; native mode streams to disk. Cancellation may leave unpublished staging
files; the app does not delete evidence as part of cancelling.

SQLite migrations preserve existing preview workspaces. Backups retain
historical provenance paths from the original run, while restored artifacts
are accessed by validated relative paths inside the new project.

## Verification and remaining release gates

### Recorded checks — 2026-09-06

- Python regression suite: **1,048 passed, 16 skipped**. Skips are not passes
  and do not establish optional-model or legacy scientific parity.
- Frontend: ten unit tests passed, including authenticated transport, partial
  import failures and native export delegation; TypeScript and Vite production
  build passed. These mocks do not exercise the native save dialog itself.
- Ruff passed; focused mypy check passed for the desktop backend and shared
  profiler parameter validation.
- Frozen Windows engine: **16 workflow smoke checks passed** (nine corpus
  workflows including real spaCy/WordNet execution, plus all seven CSV tools
  in an empty-corpus project). Each checked provenance, table access and ZIP
  export. Project backup/restore and graceful shutdown also passed.
- Windows 0.2.0 installer build completed successfully. File-list inspection
  confirmed the runtime, bundled WordNet and
  notices paths. This does **not** replace installing and testing the native
  application on a clean machine.

One earlier frozen run failed with Windows `Access denied` while atomically
publishing an output directory. Two fresh-workspace runs subsequently passed
that workflow, with no retry or error-suppression code added. The cause has not
been established. If this recurs, retain the workspace's `backend.log`, the
failed job diagnostics and its run files; do not treat the failed run as valid.

### Repeat the automated checks

Run the source checks:

```powershell
$env:LOKY_MAX_CPU_COUNT = '2'
python -m pytest -p no:cacheprovider --basetemp=out/pytest-unique-directory
python -m ruff check .
python -m mypy desktop_backend core/profiler/plan.py --follow-imports=silent
cd desktop
npm.cmd test
npm.cmd run build
```

Use a fresh test directory each time. This Windows environment has inaccessible
old temp/cache directories and a joblib physical-core probe warning without
the explicit worker limit. Neither is silently suppressed in the product.

The tests cover imports and original hashes, all seven table workflows, input
snapshots, cancellation state protection, archive integrity/traversal checks,
full-table pagination/search, archiving, and persistent output access. The
frozen smoke exercises real parser execution, results and backup/restore.

The prior 87-document State of the Union project and completed readability run
were verified after reopening the saved workspace. The private source corpus
is not included in the installer.

Before calling the project a full replacement, finish the remaining engine and
GUI capability rows in `REPLACEMENT_LEDGER.md`, validate scientific outputs
against the approved specifications, complete clean-machine/native usability
acceptance, and settle the original-code license and distribution review.
No verified ledger counts were advanced merely because the desktop builds.

## Saving a way of looking

Explore charts a result table live. When an arrangement is worth keeping —
this measure against that axis, grouped this way, drilled into that bar — name
it and press **Save view**. Saving runs nothing: a view is a draft over a
result that is already finished, so trying a variant costs nothing and cannot
overwrite anything.

Reopen a view from the **Saved view** list. Views of the table on screen are
listed first; views of other tables are listed below, with the file they belong
to, and choosing one switches to that table. **Duplicate** copies a view so the
original survives the next edit, **Revert** throws away unsaved changes, and
**Delete** removes the view and leaves the result it charted untouched.

Views are stored per project and travel with a project backup. A view whose
result has been replaced or removed says so when you open it, rather than
drawing whatever is in that file now.

Where a published chart will not match the preview exactly, Explore says so
above the Publish button. Today that is ordering: the workbench can put the
largest categories first, and a published chart arranges categories in category
order. The same categories are published — only the sequence differs.
