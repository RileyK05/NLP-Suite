# Desktop release handoff

For hosted Apple Silicon/Intel builds and their remaining acceptance gates,
see [Mac candidate builds](MACOS.md). The initial Mac target is macOS 15.

## What a recipient gets

There is no universal EXE. Build the same source separately for each target:

| Computer | Artifact | Native build runner |
| --- | --- | --- |
| Windows x64 | `NLP Suite_0.3.0_x64-setup.exe` | Windows 2022 |
| Apple Silicon Mac | ARM64 DMG | macOS 15 ARM64 |
| Intel Mac | x64 DMG | macOS 15 Intel |
| Linux x64 | DEB and AppImage | Ubuntu 22.04 |

Only an artifact actually built and tested on its target is a release candidate.
The workflow definition is not evidence that Mac/Linux builds have succeeded.
Linux is not one universal ABI; test the target distribution, desktop session,
WebKitGTK and AppImage/FUSE dependencies. The configured macOS minimum is a
build setting, not evidence of testing on every older macOS version.

Recipients do not install Python, Node, Rust or pip packages. The complete
installer includes the Python engine, English spaCy model, VADER, Gensim,
WordNet and document converters. Windows also includes the offline WebView2
installer. Do not hand off the small launcher EXE from `target/release` alone:
it requires its packaged resources. Restricted lexicons, transformer models,
Stanza models, OCR and legacy binary DOC conversion are not bundled.

## Build the release candidates

Commit/review these changes and push them yourself. On GitHub, open **Actions →
Desktop platform builds → Run workflow** for the intended branch. This is a
manual run; it does not publish releases or change your repository. (Pushing
a version tag runs the same workflow and drafts a release; see below.) Native jobs install pinned Python dependencies,
test the backend/frontend, freeze the engine, run analysis smoke tests, compile
the launcher, package installers and validate the packaged payload.

Download the `nlp-suite-<platform>` artifact from each successful job. It contains
the installer(s), `SHA256SUMS.txt` and `START-HERE.txt`. Keep the corresponding
`desktop-evidence-<platform>` artifact. Failed jobs are not releasable.

For a local native build, use Python 3.12, Node and Rust plus the native Tauri
prerequisites. In an isolated Python environment at the repository root:

```text
python -m pip install -r desktop/requirements-runtime-py312.txt
python -m pip install --no-deps .
python -m pip check
python -m nltk.downloader -d desktop/.toolchain/nltk_data wordnet
python scripts/build_desktop_backend.py --with-parser
```

Then from `desktop`, run `npm ci`, `npm test`, `cargo test --locked
--manifest-path src-tauri/Cargo.toml` and `npm run package`. Build on the target
OS/architecture; do not move a Windows frozen engine into a Mac bundle. Keep
POSIX symlinks intact. The launcher rejects mismatched runtime versions,
platforms and architectures instead of attempting to run them.

Frozen engines are stored under `desktop/src-tauri/binaries/releases/<version>/nlp-runtime`.
When bumping versions, update the resource mapping in `tauri.conf.json` too;
the packaging tests enforce consistency. Never run an engine from a build
directory while rebuilding that same version. Old unversioned build output is
not a release artifact and may be incomplete after an interrupted build.

From the repository root, validate and collect:

```text
python scripts/verify_desktop_payload.py
python scripts/collect_desktop_release.py --destination out/distribution-0.3.0
```

Use a fresh verification/collection directory for each attempt. The payload
validator uses `out/installed-payload-check` and refuses to reuse it. On a local
Windows machine this checks the assembled launcher/runtime without replacing
your installed application; only isolated GitHub runners use the CI-only silent
installer option. Mac mounts the DMG read-only, copies and audits the APP, then
tests the copied payload. Linux checks the extracted DEB; launching the
AppImage and interactive Finder/desktop acceptance remain manual checks.

## Publish packages where users can find them

The README's Downloads link points to **GitHub Releases → latest**, not the
source tree or Actions logs. `out/` and `desktop/src-tauri/target/` are
gitignored, so a plain push publishes nothing. Releases are cut by tag:

```text
python scripts/bump_version.py 0.3.1
git commit -am "Release 0.3.1"
git tag v0.3.1
git push origin main v0.3.1
```

`bump_version.py` rewrites every file that carries the version (pyproject,
package.json/lock, tauri.conf.json and its resource path, Cargo.toml/lock, the
app footer and the health endpoint). Pushing the tag runs **Desktop platform
builds** on all four platforms and then its `release` job, which:

1. refuses to continue if the tag does not equal the pyproject version;
2. collects every platform that built and validated, plus a Python wheel;
3. renames assets without spaces (GitHub would turn them into dots) and writes
   one `SHA256SUMS.txt` for them;
4. creates a **draft** release whose notes list the attached files and mark any
   platform whose build failed as "not in this release".

Nothing is public yet. Open the draft on the
[Releases page](https://github.com/RileyK05/NLP-Suite/releases), fill in
**Changes** (the [template](RELEASE_NOTES_TEMPLATE.md) lists what evidence to
state), check the Assets list against the notes, then **Publish**. A failed
platform can be fixed and the tag's workflow re-run: the job uploads into the
existing release with `--clobber` instead of creating a second one.

Do not mark Mac or Linux verified because Windows passed, and do not link users
to an expiring Actions artifact as the permanent download location.

## Signing and distribution gates

These scripts default to unsigned/ad-hoc test artifacts. A smooth public Mac
handoff needs a Developer ID certificate and Apple notarization; Windows
signing/reputation also affects download and launch warnings. Credentials and
certificates must stay in a secure signing environment, never source control.
The Python freezer accepts `APPLE_SIGNING_IDENTITY`, but this alone is not a
complete notarization pipeline. Configure signing for both the nested Python
payload and the enclosing app, then verify the final signed/notarized artifact.
Do not advise recipients to disable their OS security protections.

The project's own distribution license still needs the owner's decision.
Review `LICENSE_REVIEW.md` and the generated third-party notices before public
distribution. Packaging work does not grant rights to restricted research data.

## Clean-machine acceptance (required before promising “just works”)

On each supported OS/architecture, with no development tools installed:

1. Verify the installer checksum, install as a normal user and launch offline.
   Confirm the window, dialogs, text rendering and dependency notices work.
2. Launch a second time: the first window should come forward, with one engine.
3. Create a project. Add a small folder through the picker and through drag/drop.
   Verify previews, duplicate handling, non-ASCII names and conversion errors.
4. Import the 87-document corpus. Run Readability (87 rows), Document Similarity
   (3,741 pairs) and NER. Inspect diagnostics and representative output, not just
   the green completion badge. Use Learn for interpretation limits.
5. Upload a CSV/resource and confirm Run cannot submit while the upload is in
   progress. Test a CSV workflow and a required-lexicon workflow.
6. Search/page results and export CSV/ZIP, including cancelling Save and trying
   an unwritable destination. Confirm originals remain unchanged.
7. Queue/cancel a run; close during another run and reopen after it finishes.
   Confirm durable states and clear startup errors if the engine is still busy.
8. Back up a project, restore as a separate project, compare documents/results.
   Move that backup to another supported OS and repeat. Never test against the
   only copy of your research workspace.
9. Upgrade an existing installation and verify projects remain available. Test
   uninstall/reinstall with a backup retained separately.

Passing this checklist establishes desktop usability for those environments,
not full equivalence with NLP Suite 1.6.38. `REPLACEMENT_LEDGER.md` remains the
separate, evidence-based scientific parity gate.

## Primary references

- [Tauri platform distribution](https://v2.tauri.app/distribute/)
- [Windows WebView2 installer options](https://v2.tauri.app/distribute/windows-installer/)
- [Mac signing and notarization](https://v2.tauri.app/distribute/sign/macos/)
- [Linux AppImage compatibility](https://v2.tauri.app/distribute/appimage/)
- [PyInstaller native builds](https://pyinstaller.org/en/stable/usage.html)
- [Preserving PyInstaller POSIX symlinks](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html)
