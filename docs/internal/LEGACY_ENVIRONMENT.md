# Legacy environment freeze (inventory, not a reproducible lock)

Frozen oracle record for the NLP Suite 1.6.x reference tree. Regenerate with `python -m tools.freeze_legacy --oracle <dir>` and diff.

## Identity

- oracle dir: `C:\Users\moomi\personal_projects\NLP-Suite-rework\NLP-Suite-1.6.38`
- git checkout: yes
- HEAD: `ref: refs/heads/redesign`
- commit: `b4a50870c98b8c9e85d14e7ae7bbb9d59dc89736`
- working tree (operator-attested): clean
- release version: `1.6.38`
- captured: 2026-09-02T22:21:19+00:00 on Windows-11-10.0.26200-SP0 / Python 3.12.10
- limitation: this is an inventory captured from the oracle tree under Python
  3.12.10. The legacy CI/build targets Python 3.10, but the freeze has only
  names (not hashes) for 75 of 77 requirement lines and no verified 3.10
  executable, wheel set, model hashes, or external-binary checksums. Do not
  call this environment reproducible until those evidence files are added.

## Python runtime

- `pyproject.toml: target-version = "py310"`
- `build-family-viewer.yml: python-version: '3.10'`
- `build-family-viewer.yml: python-version: '3.10'`
- `build-installers.yml: python-version: '3.10'`
- `build-installers.yml: python-version: '3.10'`
- `build-network-viewer.yml: python-version: '3.10'`
- `build-network-viewer.yml: python-version: '3.10'`
- `tests.yml: python-version: "3.10"`

## Requirements (names only; pins are the exception)

- `requirements.txt`: 72 packages, 2 pinned
- `requirements-windows.txt`: 1 package, 0 pinned
- `requirements-mac.txt`: 4 packages, 0 pinned

## External software (legacy config, machine-specific paths)

| Software | Install path | Download |
|---|---|---|
| Stanford CoreNLP | `C:\Users\moomi\NLP_Software\stanford-corenlp-4.5.8` | https://stanfordnlp.github.io/CoreNLP/download.html |
| Gephi | `C:\Program Files\Gephi-0.10.1` | https://gephi.org/users/download/ |
| Google Earth Pro | `C:\Program Files\Google\Google Earth Pro` | https://www.google.com/earth/download/gep/agree.html?hl=en-GB |
| Java (JDK) | `Java version "21.0.9" installed` | https://www.oracle.com/java/technologies/downloads/archive/ |
| MALLET | `C:\Users\moomi\NLP_Software\mallet-2.0.8` | http://mallet.cs.umass.edu/download.php |
| WordNet | `C:\Users\moomi\NLP_Software\WordNet-3.0` | https://wordnet.princeton.edu/download/current-version |

## lib/ inventory (top level)

- 18 entries
- src/: 213 .py files, 6248149 bytes
- tree fingerprint: `772d34edec1f30ec0997d53b4d7ac668ff03d5008cf81d1033c703779bc05766`

## Warnings

- requirements-windows.txt is fully unpinned (1 package, no == pins): exact versions unrecoverable
- requirements-mac.txt is fully unpinned (4 packages, no == pins): exact versions unrecoverable

