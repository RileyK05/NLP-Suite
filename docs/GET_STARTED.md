# Get NLP Suite and run your first analysis

[← Back to the project](../README.md) · [Download the latest release](https://github.com/RileyK05/NLP-Suite/releases/latest)

## 1. Download an installer, not the source code

Open **Downloads**, choose a release, read its supported-platform notes, and
expand **Assets**. Download the file for your computer:

| Computer | File to download | What to do with it |
| --- | --- | --- |
| Windows on Intel/AMD 64-bit | A file ending in `_x64-setup.exe` | Run the installer, then open NLP Suite from Start |
| Apple Silicon Mac | `.dmg` labeled ARM64, aarch64 or Apple Silicon | Open it and drag NLP Suite to Applications |
| Intel Mac | `.dmg` labeled x64, x86_64 or Intel | Open it and drag NLP Suite to Applications |
| Debian/Ubuntu Linux on Intel/AMD 64-bit | `.deb` labeled amd64/x64 | Install using your package manager |
| Other compatible Linux on Intel/AMD 64-bit | `.AppImage` labeled x64/amd64/x86_64 | Allow the file to execute, then launch it; system dependencies may still be required |

For Macs, **Apple menu → About This Mac** identifies the chip/processor.
An Apple M-series chip uses the Apple Silicon build; an Intel processor uses
the Intel build. The initial Mac candidate requires macOS 15 or newer.
On Windows, **Settings → System → About → System type** identifies whether
the processor is x64 or ARM. Do not assume the x64 beta is verified on ARM.

**Not an installer:** GitHub's “Source code (zip)” and “Source code (tar.gz)”
contain developer files. You do not need to clone the repository or install
Python, pip packages, Node or Rust to use a complete desktop installer.
The small standalone `nlp-suite-desktop.exe` from a build folder is also not
the complete application; use the **setup EXE**.

If your installer is not listed in Assets, that platform's build has not been
published in that release. Ask the maintainer; renaming an EXE to a DMG will not
make it run on a Mac. A prepared build workflow is not a verified download.
If you cannot access the repository, ask its owner for access or the installer.

### Beta and security warnings

Read the release notes before installing. Test/ad-hoc-signed packages are not
the same as signed, notarized public releases, so the first launch shows a
warning. Allowing this one app is not the same as disabling protections:

- **Windows SmartScreen** ("Windows protected your PC"): **More info → Run anyway**.
- **macOS Gatekeeper** ("cannot verify" / "Apple could not verify"): close the
  dialog, open **System Settings → Privacy & Security**, scroll down and click
  **Open Anyway** next to NLP Suite. macOS asks once.

Never turn off SmartScreen or Gatekeeper system-wide to run the app. If your
OS refuses the installer outright, ask for an appropriately signed build.
The accompanying `SHA256SUMS.txt` lets you check download integrity, but a
checksum alone does not establish that a publisher is trustworthy.

## 2. Create your first project

1. Open **NLP Suite** and create a project with a recognizable name.
2. Open **Corpus**. Use the file/folder picker or drag supported files into the
   import area. Start with a few TXT files before importing a large collection.
3. Check the import summary and preview a document. Resolve conversion errors
   before analyzing; a file being listed does not guarantee useful extracted text.
4. Open **Analyses**, choose **Readability**, and select **Run analysis**.
5. Open **Runs & results**. Inspect the output and any warnings, then export
   a CSV or complete run ZIP.
6. Open **Learn** for what each tool measures, how to interpret it and its limits.

The app copies imports into its local workspace; it does not edit your original
corpus. Large corpora take longer, and analyses are queued one at a time.
Use **Environment & setup → Project backup & restore** to save a `.nlpsuite`
backup outside the app's workspace. Restoring creates a separate project.

## 3. Know what is included

Complete release installers include the local Python engine, English spaCy
model, VADER, Gensim, WordNet and document converters. Windows includes an
offline WebView2 installer. You do not need a cloud account for these workflows.

- **Documents:** TXT, CSV, TSV, HTML, text-based PDF, DOCX and RTF intake.
  Scanned/image-only PDFs need OCR; old binary `.doc` files need conversion.
- **CSV statistics:** select your original CSV using the analysis's resource
  file chooser. Importing a CSV into Corpus extracts text; that is a different
  operation from selecting a structured table for statistics.
- **Extra resources:** restricted research lexicons are not bundled. Supply
  them only if you have permission to use them.
- **Optional models:** Stanza and transformer workflows need a separately
  prepared source environment. Running pip on your computer does not add
  packages to an installed app's frozen runtime.

The app is a beta. Tested workflows are useful starting points, not a guarantee
that all legacy NLP Suite features are replaced or that every analysis is
appropriate for your corpus.

## Something is wrong?

| Problem | First check |
| --- | --- |
| I only see ZIP/TAR source downloads | A maintainer needs to attach the installer to the release |
| App reports missing/mismatched runtime | Reinstall the complete matching package; don't move only its launcher EXE |
| A tool needs a model or lexicon | Read its setup message and Learn entry; some tools require additional resources |
| Imported PDF has no useful text | Check whether it is scanned and needs OCR |
| App says another engine is using the workspace | A previous run may still be finishing after the window closed; let it finish |
| A run fails | Keep its diagnostics; report the tool, settings, OS and app version |

For help, [open an issue](https://github.com/RileyK05/NLP-Suite/issues) or
contact the maintainer. Do not post private corpus text, credentials or entire
workspaces. Prefer a small non-sensitive example that reproduces the problem.
