"""Build the native Python runtime on Windows, macOS or Linux (never cross-compile)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def build_arguments(root: Path, *, with_parser: bool, system: str, python: str, version: str) -> list[str]:
    if system not in ("win32", "darwin", "linux"):
        raise ValueError(f"Unsupported runtime platform: {system}")
    separator = ";" if system == "win32" else ":"
    resources = root / "out/desktop-package/resources"
    args = [
        python,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--noupx",
        "--name",
        "nlp-runtime",
        "--paths",
        str(root),
        "--distpath",
        str(root / "desktop/src-tauri/binaries/releases" / version),
        "--workpath",
        str(root / "out/desktop-package/build"),
        "--specpath",
        str(root / "out/desktop-package"),
        "--add-data",
        f"{resources}{separator}desktop_resources",
    ]
    for module in ("desktop_backend", "scipy", "plotly", "wordcloud", "matplotlib", "seaborn"):
        args.extend(["--collect-data", module])
    for module in ("numpy", "pandas", "scipy", "scikit-learn"):
        args.extend(["--copy-metadata", module])
    # pandas imports pyarrow only inside read_parquet/to_parquet, so PyInstaller
    # cannot see it; the live bench's parse cache (desktop_backend/live.py)
    # needs it. pyarrow is a core dependency in pyproject.toml.
    args.extend(["--hidden-import", "pyarrow.parquet"])
    if with_parser:
        args.extend(["--add-data", f"{root / 'desktop/.toolchain/nltk_data'}{separator}nltk_data"])
        for module in (
            "spacy",
            "en_core_web_sm",
            "spacy_legacy",
            "spacy_loggers",
            "thinc",
            "vaderSentiment",
            "gensim",
            "nltk",
        ):
            args.extend(["--collect-all", module])
        for module in ("spacy", "en_core_web_sm"):
            args.extend(["--copy-metadata", module])
    excluded = [
        "tools",
        "app",
        "scripts",
        "tests",
        "torch",
        "tensorflow",
        "transformers",
        "stanza",
        "streamlit",
        "IPython",
        "pytest",
        "tkinter",
        "spacy.tests",
        "thinc.tests",
        "gensim.test",
        "nltk.test",
        "nltk.app",
        "nltk.twitter",
    ]
    if not with_parser:
        excluded.extend(["spacy", "gensim", "nltk"])
    for module in excluded:
        args.extend(["--exclude-module", module])
    if system == "darwin" and os.environ.get("APPLE_SIGNING_IDENTITY"):
        args.extend(["--codesign-identity", os.environ["APPLE_SIGNING_IDENTITY"]])
    args.append(str(root / "scripts/desktop_entry.py"))
    return args


def _repair_mac_signatures(runtime: Path) -> None:
    """Rebuild ad-hoc signatures on every Mach-O image of the frozen runtime.

    PyInstaller on a framework-build Python copies ``Python.framework`` whose
    nested dylib carries a signature naming resources that the frozen tree no
    longer contains — ``codesign --verify --strict`` on the installed payload
    then fails with "code has no resources but signature indicates they must
    be present". The fix is to re-sign each Mach-O explicitly, deepest paths
    first so a bundle's nested code is signed before the bundle that contains
    it, and to drop stale ``_CodeSignature`` directories that reference the
    stripped resources.
    """
    import subprocess

    macho_magics = (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xcf", b"\xbe\xba\xfe\xca")

    def is_macho(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                return handle.read(4) in macho_magics
        except OSError:
            return False

    for signature_dir in list(runtime.rglob("_CodeSignature")):
        if signature_dir.is_dir():
            shutil.rmtree(signature_dir)
    # Deepest-first: a .framework's nested binary before the .framework's
    # Versions symlink, and a bundle before the directory that contains it.
    targets = sorted(
        (p for p in runtime.rglob("*") if p.is_file() and not p.is_symlink()), key=lambda p: len(p.parts), reverse=True
    )
    for path in targets:
        if not is_macho(path):
            continue
        subprocess.run(  # noqa: S603
            ["/usr/bin/codesign", "--force", "--sign", "-", str(path)],
            check=True,
        )
    # Verify each signed image now passes strict validation before the
    # installer packaging trusts it.
    for path in targets:
        if path.is_file() and not path.is_symlink() and is_macho(path):
            subprocess.run(  # noqa: S603
                ["/usr/bin/codesign", "--verify", "--strict", str(path)],
                check=True,
            )


def main() -> None:
    from desktop_notices import write_notices

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-parser", action="store_true", help="Require and bundle the complete English desktop NLP stack"
    )
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Release runtimes require Python 3.12. Use the pinned packaging environment.")
    if options.with_parser:
        required = (
            "spacy",
            "en_core_web_sm",
            "vaderSentiment",
            "gensim",
            "nltk",
            "pdfminer",
            "docx",
            "striprtf",
            "plotly",
            "wordcloud",
            "matplotlib",
            "seaborn",
            "openpyxl",
            "PIL",
        )
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        wordnet = root / "desktop/.toolchain/nltk_data/corpora/wordnet.zip"
        if not wordnet.is_file():
            missing.append("WordNet data (run the documented nltk downloader command)")
        if missing:
            raise SystemExit("Incomplete release environment: " + ", ".join(missing))
    resources = root / "out/desktop-package/resources"
    resources.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / "assets/sample-corpus", resources / "sample-corpus", dirs_exist_ok=True)
    write_notices(resources / "THIRD_PARTY_NOTICES.txt")
    version = json.loads((root / "desktop/package.json").read_text(encoding="utf-8"))["version"]
    manifest = {
        "version": version,
        "platform": sys.platform,
        "machine": platform.machine().lower(),
        "python": platform.python_version(),
        "english_parser": options.with_parser,
    }
    (resources / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args = build_arguments(
        root, with_parser=options.with_parser, system=sys.platform, python=sys.executable, version=version
    )
    subprocess.run(args, cwd=root, check=True)  # noqa: S603
    runtime = root / "desktop/src-tauri/binaries/releases" / version / "nlp-runtime"
    if sys.platform == "darwin":
        _repair_mac_signatures(runtime)
    # Keep POSIX symlinks intact when transferring the complete onedir tree.
    inventory = {}
    for path in sorted(runtime.rglob("*")):
        relative = path.relative_to(runtime).as_posix()
        if path.is_symlink():
            inventory[relative] = {"symlink": os.readlink(path)}
        elif path.is_file():
            with path.open("rb") as handle:
                inventory[relative] = {"sha256": hashlib.file_digest(handle, "sha256").hexdigest()}
    (root / "out/desktop-package/runtime-inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Built {sys.platform}/{platform.machine()} runtime {version}: {runtime}")


if __name__ == "__main__":
    main()
