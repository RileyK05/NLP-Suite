# NLP Suite

A local desktop workspace for exploring text: import a corpus, run analyses,
inspect results and export your work. The **Learn** tab explains the tools as you go.

## Download the desktop app

### **[⬇ Download the latest release](https://github.com/RileyK05/NLP-Suite/releases/latest)**

Under **Assets**, pick the file for your computer. You do **not** need Python,
Node, Rust or the source code.

| Your computer | Download the file ending in |
| --- | --- |
| Windows 10 / 11, 64-bit | `_x64-setup.exe` |
| Mac with Apple Silicon (M1 and later) | `_aarch64.dmg` |
| Mac with an Intel processor | `_x64.dmg` |
| Debian / Ubuntu Linux, 64-bit | `_amd64.deb` |

**Source code (zip / tar.gz) is not the app.** If your platform's file is
missing from a release, that build was not published for it — check the release
notes or an older release. Mac builds need macOS 15 or later.

**First launch.** The beta is not code-signed yet, so your OS will warn once:

- **Windows:** "Windows protected your PC" → **More info** → **Run anyway**.
- **Mac:** after the first blocked launch, open **System Settings → Privacy &
  Security**, scroll down and click **Open Anyway** next to NLP Suite.

**Updates.** On Windows and Mac, NLP Suite tells you when a new version is out
and installs it with one click. On Linux, install the new `.deb` over the old one.
Version 0.4.0 and earlier cannot update themselves: if you have one, download
the latest release once and install it over the old version. Your projects are kept.

More detail, including how to tell which Mac you have:
**[Installation and first project](docs/GET_STARTED.md)**.

## Start using it

1. Install the package for your OS and open **NLP Suite**.
2. Create a project and add your corpus files or folder from **Corpus**.
3. Open **Analyses** and start with **Readability**.
4. Inspect **Runs & results**, then export the results you need.
5. Use **Learn** for explanations and **Environment & setup** for dependencies
   and project backups.

Imports make local copies; keep a separate backup of important research.
The bundled English stack works without installing developer tools. Some
analyses still need user-supplied lexicons or optional models; scanned PDFs need
OCR first. This is a **beta**, not a full equivalence certification for NLP Suite 1.6.38.

## Or use it from Python

The analysis engine is a plain Python package — no desktop app required. This
is the same code the installer bundles, so results match.

```bash
pip install "nlp-suite-ng @ git+https://github.com/RileyK05/NLP-Suite"   # Python 3.12
nlp-doctor                        # what is installed, and the fix for what is not
nlp-suite --list                  # every tool, with a one-line description
nlp-suite readability ./corpus ./out
```

No clone needed. Each release also attaches a `.whl` you can `pip install`
directly, which pins you to that release's exact code.

`./corpus` is a folder of `.txt` files; each run writes a timestamped,
self-describing directory under `./out`. Tools that need a syntactic parse take
`--parser spacy` or `--parser stanza`, and `nlp-suite <tool> --help` documents
that tool's flags.

Most analyses need optional packages: install just the extras `nlp-doctor`
names, e.g. `pip install "nlp-suite-ng[spacy,topics] @ git+https://github.com/RileyK05/NLP-Suite"`,
or use `[all]` for everything (large: it pulls in PyTorch). The project is not
on PyPI yet, so `pip install nlp-suite-ng` by name will not find it. See the
[developer guide](docs/DEVELOPMENT.md) for the Streamlit app and running from
source.

## Where to go next

| I want to… | Start here |
| --- | --- |
| Install, choose the right file, or troubleshoot | [Getting started](docs/GET_STARTED.md) |
| See desktop features and test evidence | [Desktop status](docs/DESKTOP.md) |
| Review end-user boundaries and release acceptance | [Desktop hardening](docs/PRODUCTION_HARDENING.md) |
| Build interactive exploration and corpus selections | [Interactive research plan](docs/INTERACTIVE_RESEARCH_PLAN.md) |
| Plan question-driven research workflows | [Question-driven NLP plan](docs/QUESTION_DRIVEN_RESEARCH_PLAN.md) |
| Build a Mac candidate without owning a Mac | [Mac builds](docs/MACOS.md) |
| Build or publish installers | [Maintainer release guide](docs/DESKTOP_RELEASE.md) |
| Run from source, use the CLI, or develop | [Developer guide](docs/DEVELOPMENT.md) |
| Check legacy replacement progress | [Replacement ledger](docs/REPLACEMENT_LEDGER.md) |

This is an independent rebuild. The legacy suite is kept separate and used
read-only as a behavioral reference; implementation progress is not the same as
independently verified scientific parity.

## License

NLP Suite is released under the [MIT License](LICENSE). The predecessor NLP
Suite 1.6.38 (Roberto Franzosi, Emory University) is GPL-licensed; this is an
independent rebuild that copies no legacy source, so the MIT grant covers only
the original code here. Bundled third-party dependencies keep their own
licenses — the installer lists them under **Environment & setup**, and
[docs/LICENSE_REVIEW.md](docs/LICENSE_REVIEW.md) records the policy for
user-supplied research lexicons, which are never redistributed.
