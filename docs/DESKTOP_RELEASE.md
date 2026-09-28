# Desktop release handoff

How to build, validate, sign and publish the desktop installers. Hosted Apple
Silicon/Intel builds and their remaining acceptance gates are in
[Mac candidate builds](#mac-candidate-builds-without-owning-a-mac). The initial
Mac target is macOS 15.

## What a recipient gets

There is no universal EXE. Build the same source separately for each target:

| Computer | Artifact | Native build runner |
| --- | --- | --- |
| Windows x64 | `NLP Suite_<version>_x64-setup.exe` | Windows 2022 |
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
it requires its packaged resources. Restricted lexicons, optional research
models, Stanza models, OCR and legacy binary DOC conversion are not bundled.
The frozen engine includes the neural sentiment and topic models exercised by
its smoke test.

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

Rust must be on PATH. This checkout has a project-local toolchain; from
`desktop`, enable it for the current PowerShell terminal with:

```powershell
$env:CARGO_HOME = Join-Path (Get-Location) '.toolchain/cargo'
$env:RUSTUP_HOME = Join-Path (Get-Location) '.toolchain/rustup'
$env:PATH = "$env:CARGO_HOME\bin;$env:PATH"
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
python scripts/verify_desktop_payload.py --stage out/installed-payload-0.5.0-check
python scripts/collect_desktop_release.py --destination out/distribution-0.5.0
```

Use a fresh verification/collection directory for each attempt. The payload
validator defaults to `out/installed-payload-check` and refuses to reuse its
stage directory. On a local
Windows machine this checks the assembled launcher/runtime without replacing
your installed application; only isolated GitHub runners use the CI-only silent
installer option. Mac mounts the DMG read-only, copies and audits the APP, then
tests the copied payload. Linux checks the extracted DEB; launching the
AppImage and interactive Finder/desktop acceptance remain manual checks.

## Two tracks: dev and public

- **Dev** is `RileyK05/New_NLP_Suite` (private), branch `main`. All work
  happens here, including personal files that never ship.
- **Public** is `RileyK05/NLP-Suite`. Nobody edits it directly. It changes
  only through the **Publish dev to public** workflow, which mirrors dev
  minus the paths in `.publicignore` and refuses anything named like
  coursework (`scripts/publish_snapshot.py`).

Both buttons live in the public repository's **Actions** tab (public builds
are free; the private repository has no Actions minutes):

| I want to… | Actions → workflow | Options |
| --- | --- | --- |
| Try dev in a real installer | Desktop platform builds | tick **from_dev**; download the `nlp-suite-<platform>` artifact |
| Run the full test gate on dev | CI | tick **from_dev** |
| Ship a release | Publish dev to public | **release** ticked (default) |

To ship 0.5.0: bump the version in dev (`python scripts/bump_version.py 0.5.0`),
commit and push dev, then run **Publish dev to public**. It commits the
snapshot to public `main`, CI runs on it, and the desktop build drafts the
release. Open the draft on the
[Releases page](https://github.com/RileyK05/NLP-Suite/releases), write the
**Changes** section, check the Assets, and **Publish**. Installed apps then
offer the update.

**One-time setup:** a fine-grained personal access token (GitHub → Settings
→ Developer settings → Fine-grained tokens) with repository access to both
repositories: *Contents: read* on `New_NLP_Suite`, *Contents: read and write*
and *Workflows: read and write* on `NLP-Suite`. Save it in the public
repository as the secret `PUBLISH_TOKEN`.

Do not mark Mac or Linux verified because Windows passed, and do not link
users to an expiring Actions artifact as the permanent download location.

Each release needs `docs/releases/<version>.md`: the user-facing **Changes**
text. The release workflow (`.github/workflows/release.yml`) refuses to publish
without it and pastes it into the draft's `## Changes` section. The rest of the
draft notes (download table, first steps, checksum instructions, beta notice)
are generated by the draft step in `.github/workflows/desktop.yml`; change
them there.

## Mac candidate builds without owning a Mac

Status: **prepared, not yet built or validated on a Mac**.

### Run on hosted Macs

Open GitHub **Actions → Desktop platform builds → Run workflow**, enable
**mac_only**, and start the workflow. It builds two independent candidates on
native hardware:

- `macos-15`: Apple Silicon / ARM64.
- `macos-15-intel`: Intel / x86_64.

Use the corresponding `nlp-suite-macos-arm64` or `nlp-suite-macos-x64` artifact.
These are separate DMGs, not a universal application and not Windows EXEs.
Runner time is governed by your GitHub account's allowance/billing.

The initial minimum is **macOS 15** to match the build environment. Do not
promise macOS 13/14 support merely because a Rust build flag can name it:
the Python interpreter and native scientific wheels also have deployment
requirements. Lowering the minimum requires native tests on the older OS.
(`tests/test_desktop_packaging.py` checks that `tauri.conf.json` matches the
minimum stated here.)

### What the native job checks

1. Backend, packaging and frontend tests.
2. A native Python freeze and the packaged analysis smoke workflows.
3. Rust launcher tests and native APP/DMG packaging.
4. DMG integrity, read-only mounting and copying the APP out using `ditto`.
   The copy is placed under a path containing spaces and the image is detached.
5. Copied bundle identity/version, executable permissions, internal symlinks,
   matching architecture slices and signatures for Mach-O files, followed by
   strict/deep application signature verification.
6. Launching that copied application's native bridge and its embedded Python
   engine, authenticated health readiness, shutdown, and the analysis/export/
   backup/restore smoke suite again from the copied payload.

Download `desktop-evidence-macos-*` as well. `macos-validation.json` records the
native OS, architecture and audited binaries; console errors identify failing
Apple tools. A passed ad-hoc signature check does **not** mean Gatekeeper accepts
a browser-downloaded copy. The report explicitly leaves that gate unverified.

### Mac public distribution is a separate gate

The Mac config explicitly uses ad-hoc signing (`-`) for CI test candidates.
This does not authenticate a publisher or provide notarization. Do not send
these candidates to ordinary users as a frictionless production install, and
do not remove quarantine or disable Gatekeeper to make a test pass.

For public distribution, an owner must provide a Developer ID signing setup
and notarization credentials through a secure signing environment. Never put
certificate private keys or Apple credentials into this repository or chat.
Import the signing identity **before** freezing Python, so its nested binaries
and the enclosing Tauri application use the intended identity. Configure the
real Tauri signing identity instead of the test default, notarize the finished
application and validate the stapled result. No automatic secret import or
notarization pipeline is claimed by the current unsigned/test workflow.

Avoid speculative security exceptions in entitlements; investigate actual
signing/notarization errors on the native runner. Tests must be rerun against
the final signed distribution, not only the earlier ad-hoc build.

### Still needs a human Mac tester

The hosted checks do not certify Finder launch, WKWebView rendering, native
file/folder dialogs, drag/drop, macOS privacy prompts, Dock reactivation,
quarantined-download launch, upgrades or uninstall behavior. Use the
clean-machine checklist below on the signed candidate. The user's 87-document
corpus is intentionally not uploaded to CI; local Windows corpus results are
not substituted for native Mac corpus results.

## In-app updates

Installed apps check `releases/latest/download/latest.json` on launch and offer
a newer version with one click (Windows and Mac; Linux debs update by
reinstalling). Updates are verified with the maintainer's own signing key,
which is free and separate from Apple/Microsoft code signing.

**One-time setup** (keep the key forever; losing it means existing installs
can no longer be updated and must reinstall by hand):

1. From `desktop/`, run `npx tauri signer generate -w nlp-suite-updater.key`
   and choose a password. It writes `nlp-suite-updater.key` (private) and
   `nlp-suite-updater.key.pub` (public). Back both up, with the password,
   somewhere safe outside the repository. Never commit the private key.
2. In the public repository: **Settings → Secrets and variables → Actions →
   New repository secret**, three times:
   - `TAURI_SIGNING_PRIVATE_KEY`: the contents of `nlp-suite-updater.key`
   - `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`: the password
   - `TAURI_UPDATER_PUBKEY`: the contents of `nlp-suite-updater.key.pub`

Without the secrets, releases still build; the apps they install simply never
find updates. With them, every release also carries `latest.json` and signed
update files, and an app only updates from releases made with the same key.
An update only reaches people once the release is **published** and marked
**latest**: drafts and pre-releases are invisible to installed apps, which
makes pre-release the safe place to test a build first.

## Signing and distribution gates

These scripts default to unsigned/ad-hoc test artifacts. A smooth public Mac
handoff needs a Developer ID certificate and Apple notarization; Windows
signing/reputation also affects download and launch warnings. Credentials and
certificates must stay in a secure signing environment, never source control.
The Python freezer accepts `APPLE_SIGNING_IDENTITY`, but this alone is not a
complete notarization pipeline. Configure signing for both the nested Python
payload and the enclosing app, then verify the final signed/notarized artifact.
Do not advise recipients to disable their OS security protections.

The original code is MIT-licensed. Review `LICENSE_REVIEW.md` and the generated
third-party notices before public distribution. Packaging work does not grant
rights to restricted research data.

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
- [Tauri DMG distribution](https://v2.tauri.app/distribute/dmg/)
- [GitHub's native Mac runner labels and architectures](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [PyInstaller Mac architecture/signing notes](https://pyinstaller.org/en/stable/feature-notes.html)
- [Linux AppImage compatibility](https://v2.tauri.app/distribute/appimage/)
- [PyInstaller native builds](https://pyinstaller.org/en/stable/usage.html)
- [Preserving PyInstaller POSIX symlinks](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html)
