# NLP Suite desktop beta

The desktop application is a local Tauri/React interface over the existing
Python engines. Streamlit and the command-line tools remain available. It is a
beta, and **not a full legacy-parity signoff**: `REPLACEMENT_LEDGER.md` is the
separate, evidence-based scientific parity record.

Installing the released app is covered in [Get started](GET_STARTED.md);
building and publishing installers in [Desktop release](DESKTOP_RELEASE.md).
Build logs and checksums for the 0.2.x and 0.3.0 Windows installers were
removed from this page in the 2026-09-28 docs cleanup; Git history has them.

## 0.5.0 source candidate

The Corpus page detects and edits document details, imports them from a
spreadsheet, and selects a date or order axis. Split book previews chapters or
transcript turns and stores each selected section as a derived document; the
original moves to Trash by default. Runs can filter by detail or order window,
and book figures use chapter positions. A project backup carries the details
and derivation records.

The Workshop group has **Compare** for multiple document sets, including
alignment by a shared detail or period, and **Scripts** for local notebooks
that call the suite library. Both save ordinary runs that Past runs and Explore
can open. Imported or pasted notebook code is marked for review before it
runs; see [Security](SECURITY.md) for its file and network access. Settings can
clear recomputable parse and vector caches without removing documents or runs.

This describes the current source. The 0.5.0 acceptance checks and installer
build are still open in [the release plan](internal/PLAN_0.5.0.md); do not describe an
older installed app as containing these features.

## End-user scope

The desktop has an explicit release catalog in `desktop_backend/catalog.py`.
Adding a research adapter or command does not automatically expose it to end
users. The catalog is enforced on submission as well as in the interface, and
the backend independently validates every submitted request.

Experimental coreference, semantic demonstrations, offline knowledge-graph
stubs, and transformer research workflows are excluded. NRC and
SentiWordNet/hedonometer also remain in the developer interfaces because their
default resources are not supplied by the installer. These engines are
preserved in the source tree; hiding them does not certify their scientific
parity or remove them from the migration backlog.

Settings show analysis readiness, project management, backups and notices,
rather than Python package installation commands. Tool names describe the
analysis rather than exposing module names. Required resources and
unavailable components prevent submission in the form.

VADER runs with the included scorer by default. ANEW is an explicit optional
selection and requires a chosen lexicon file. Developer commands retain their
existing combined-analysis default; `--analysis vader` selects VADER alone.

The example importer reads three fictional texts from `assets/sample-corpus`,
never a developer's working corpus. The same texts are included in the frozen
runtime. No personal research corpus is bundled.

## What the desktop does

- Corpus workflows and explicit CSV statistics workflows from the release
  catalog. The catalog is searchable and generated from the execution
  contracts.
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
  change the accepted run. Resource uploads block submission until complete.
- Result tables with full-dataset substring search and 500-row pagination,
  CSV and complete-run ZIP exports.
- Project renaming, recoverable archiving, and archived-project restoration.
  Documents, runs and projects can be moved to Trash, restored, or purged.
- Portable `.nlpsuite` backup/restore: documents, originals, run files and
  metadata, with checksums, bounded extraction and traversal rejection.
  Restoring creates a separate project; it never overwrites an existing one.
- Native exports stream from the authenticated local engine into a temporary
  destination file, then publish after successful transfer. They do not pass
  entire ZIPs through JavaScript arrays or truncate an existing file on failure.
- Dependency notices generated from the packaging environment and readable in
  Environment & setup. Restricted research lexicons are not bundled.

Closing requests graceful engine shutdown: accepted work finishes in the
background. To stop work immediately, cancel the runs before closing. A
workspace cannot be reopened while its prior engine still holds the lock.
Abruptly interrupted jobs become INTERRUPTED on the next startup.

The native workspace normally lives at
`%LOCALAPPDATA%/org.nlpsuite.desktop/workspace`; Setup displays the actual path.
Do not edit copied inputs inside it. Back up the entire workspace while stopped
if doing a manual filesystem backup; SQLite alone is not the whole project.

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

## Runtime scope

The packaging recipe bundles Python, spaCy and `en_core_web_sm`, converters,
VADER, Gensim, NLTK and WordNet data, plus the BERT, neural sentiment and
document-embedding models described in the release notes. Actual model
execution is checked by the frozen-runtime smoke test; a package-presence badge
alone is not a health certificate.

Optional/restricted inputs still matter:

- ANEW, NRC, concreteness, iconicity and other research lexicons remain
  user-supplied. The combined VADER/ANEW workflow requires the ANEW input.
- Stanza models are not part of the bundle; those workflows require an
  appropriately configured source environment.
- The mention-grouping tool is explicitly a **lemma baseline**, not a neural
  coreference replacement.
- spaCy dependency parsing does not supply CoreNLP constituency trees. A
  dependency-complexity run is not evidence of full constituency parity.
- GIS, PC-ACE/database and other non-corpus CLI workflows are not all integrated
  into this desktop UI. Their existing implementations remain available.

## Reliability and packaging

- Cancelled jobs are checked again after worker startup, before dispatch.
- Worker startup has a 90-second deadline and failed workers are reaped.
- Malformed non-object worker replies fail cleanly.
- Archived projects reject new imports and analysis submissions.
- Chart exports respect the requested top-N category limit. A requested PNG
  wordcloud that fails to render fails the job instead of reporting completion.
- The frozen runtime excludes the `tools`, `app`, `scripts` and `tests`
  packages. Core analysis and desktop backend modules remain necessary runtime
  components. The frozen entrypoint is compiled by PyInstaller; developer
  scripts are not application navigation or separately shipped commands.
- Plotly, Wordcloud, Pillow and Openpyxl are declared packaging dependencies.
  Matplotlib is retained because Wordcloud and the publication figures import
  it. Packaged chart exports offer HTML and Excel; formats requiring a
  separately installed rendering browser are not advertised.

Regression coverage lives in `tests/test_production_boundary.py`, alongside the
other desktop tests. The frozen-engine smoke test exercises VADER, HTML/Excel
chart exports, and PNG wordcloud generation.

## Chart viewer isolation

HTML charts are isolated from the main application's CSP. They use a dedicated
`nlp-viz` protocol in the native app and a five-minute, single-artifact ticket
in the browser preview. Neither frame URL contains the backend bearer token.
Frames permit chart scripts under a separate CSP, block network loads, and keep
an opaque-origin sandbox. HTML must be self-contained and no larger than
32 MiB; use Download if a PDF cannot display.

To check the viewer from source, fully restart `npm run desktop` from
`desktop/` (a frontend refresh cannot update the native protocol). For the
browser preview, run `npm run build`, restart `python scripts/desktop_preview.py`,
and open an existing HTML chart and wordcloud. Check hover, resize, reopen, PNG
switching, and Download. Native chart rendering on macOS/Linux still needs
manual verification.

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

SQLite migrations preserve existing workspaces. Backups retain historical
provenance paths from the original run, while restored artifacts are accessed
by validated relative paths inside the new project.

**Known unexplained failure:** one frozen run failed with Windows `Access
denied` while atomically publishing an output directory; two fresh-workspace
runs then passed, with no retry or error-suppression code added. If this
recurs, retain the workspace's `backend.log`, the failed job diagnostics and
its run files; do not treat the failed run as valid.

## Source development

From the repository root, with Python 3.12 and Node/npm:

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

The full quality gate (Python and frontend) is in
[the developer guide](DEVELOPMENT.md#environment).

## Latest recorded verification (2026-09-16 hardening pass)

On the Windows development machine:

- The full standard Python suite passed: 1,395 tests, six skipped and 60
  model-integration tests deselected. Real bundled-model execution was checked
  separately by the frozen-engine smoke test.
- The frontend production build and all 73 frontend tests passed.
- All seven native Rust launcher/preview tests passed.
- Strict mypy passed for 201 Python source files; Ruff lint and formatting,
  the release audit, and whitespace checks passed.
- The rebuilt frozen engine passed 20 analysis/export cases, including VADER,
  HTML and Excel charts, and a PNG wordcloud whose file signature was checked.
  Input provenance, ZIP exports, backup/restore and graceful shutdown passed.
- The frozen module archive contained no first-party `tools`, `app`, `scripts`
  or `tests` packages. Its sample corpus contained exactly the three fictional
  example files.
- Browser checks covered project creation, example import, a real VADER run,
  results and the settings page. This is not a native installed-app UI check.

Passing source tests is not a clean-machine installer certification. Before
publishing installers, run the native packaging workflow on each supported OS
and complete the clean-machine acceptance in
[Desktop release](DESKTOP_RELEASE.md#clean-machine-acceptance-required-before-promising-just-works).
Before calling the project a full replacement, also finish the remaining engine
and GUI capability rows in `REPLACEMENT_LEDGER.md`, validate scientific outputs
against the approved specifications, and review the generated third-party
notices for distribution. No verified ledger counts were advanced merely
because the desktop builds.
