# Mac candidate builds without owning a Mac

Status: **prepared, not yet built or validated on a Mac**. The existing Windows
0.3.0 installer is unchanged by this packaging-only pass.

## Run on hosted Macs

After reviewing and pushing the changes, open GitHub **Actions → Desktop
platform builds → Run workflow**, enable **mac_only**, and start the workflow.
It builds two independent candidates on native hardware:

- `macos-15`: Apple Silicon / ARM64.
- `macos-15-intel`: Intel / x86_64.

Use the corresponding `nlp-suite-macos-arm64` or `nlp-suite-macos-x64` artifact.
These are separate DMGs, not a universal application and not Windows EXEs.
Runner time is governed by your GitHub account's allowance/billing. This local
development pass did not push changes or start remote jobs.

The initial minimum is **macOS 15** to match the build environment. Do not
promise macOS 13/14 support merely because a Rust build flag can name it:
the Python interpreter and native scientific wheels also have deployment
requirements. Lowering the minimum requires native tests on the older OS.

## What the native job checks

1. Backend, packaging and frontend tests.
2. A native Python freeze and 16 packaged analysis smoke workflows.
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

## Public distribution is a separate gate

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

## Still needs a human Mac tester

The hosted checks do not certify Finder launch, WKWebView rendering, native
file/folder dialogs, drag/drop, macOS privacy prompts, Dock reactivation,
quarantined-download launch, upgrades or uninstall behavior. Use the checklist
in [DESKTOP_RELEASE.md](DESKTOP_RELEASE.md) on the signed candidate. The user's
87-document corpus is intentionally not uploaded to CI; local Windows corpus
results are not substituted for native Mac corpus results.

References checked for this pass:

- [GitHub's native Mac runner labels and architectures](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [Tauri Mac signing, ad-hoc signing and notarization](https://v2.tauri.app/distribute/sign/macos/)
- [Tauri DMG distribution](https://v2.tauri.app/distribute/dmg/)
- [PyInstaller native builds and OS compatibility](https://www.pyinstaller.org/en/stable/usage.html)
- [PyInstaller Mac architecture/signing notes](https://pyinstaller.org/en/stable/feature-notes.html)
