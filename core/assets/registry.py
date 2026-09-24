"""Asset registry (FR-4.1) — versioned data assets with checksums and licenses.

Every production lexicon/ontology lives here as an ``AssetSpec``: pinned
version, expected SHA-256, license terms, and origin. Nothing reads asset
bytes at import time; ``load_text``/``path`` verify on demand. A missing or
corrupt asset is an actionable diagnostic (what to install, where from),
never an import error and never a silent sample swap.

Statuses: OK (present, checksum matches), MISSING, CORRUPT, UNSTAMPED (no
checksum pinned yet — the entry is a placeholder FR-4.2+ must stamp before
production use; UNSTAMPED assets never load).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "BUILTIN_ASSETS",
    "AssetRegistry",
    "AssetSpec",
    "default_registry",
]


@dataclass(frozen=True, slots=True)
class AssetSpec:
    name: str
    version: str  # exact upstream release identifier, never "full"
    path: str  # registry-root-relative
    sha256: str  # pinned checksum; "" only for user-supplied assets
    license: str
    source: str  # exact URL or package coordinates
    required: bool
    description: str
    redistribution: str = "user-supplied"  # redistributable | download-on-demand | user-supplied


BUILTIN_ASSETS: tuple[AssetSpec, ...] = (
    # FR-4.2 targets: exact coordinates pinned. Checksums for the two
    # oracle-shipped style assets are stamped from the pinned legacy tree;
    # sentiment lexicons are user-supplied until their production files land.
    AssetSpec(
        name="vader-lexicon",
        version="vaderSentiment v3.3.2 lexicon.txt",
        path="sentiment/vader_lexicon.txt",
        sha256="6ff1180b1a5bf60a6af265453be3adcde571518b5e674722f6368040dd19b31c",
        license="MIT (vaderSentiment package) — redistribution permitted with notice",
        source="legacy oracle lib/sentimentLib/vader_lexicon.txt (pinned); https://github.com/cjhutto/vaderSentiment",
        required=True,
        description="Full VADER sentiment lexicon (FR-4.2); user-supplied copy, checksum-verified",
        redistribution="user-supplied",
    ),
    AssetSpec(
        name="anew-lexicon",
        version="1999 norms (Bradley & Lang)",
        path="sentiment/EnglishShortenedANEW.csv",
        sha256="051e46787a87a4f36db7c2134515cff6569f3717e2238b87c42585d2bf180437",
        license="research use only (Bradley & Lang 1999, NIMH) — NOT redistributable",
        source="legacy oracle lib/sentimentLib/EnglishShortenedANEW.csv (pinned, 13,916 rows)",
        required=True,
        description="Full ANEW valence/arousal/dominance norms (FR-4.2); user-supplied copy, checksum-verified",
        redistribution="user-supplied",
    ),
    AssetSpec(
        name="sentiwordnet",
        version="3.0.0",
        path="sentiment/SentiWordNet_3.0.0.txt",
        sha256="",
        license="CC BY-SA 3.0 — redistribution permitted with attribution + share-alike",
        source="https://sentiwordnet.isti.cnr.it",
        required=True,
        description="Full SentiWordNet scores (FR-4.2)",
        redistribution="download-on-demand",
    ),
    AssetSpec(
        name="hedonometer",
        version="labMT 1.1 (Dodds et al.)",
        path="sentiment/hedonometer.json",
        sha256="3760cb6cdc741fe995733903e46c6254128baa46bedbf69df1b5b3baf9e1259e",
        license="research use (hedonometer.org) — attribution required",
        source="legacy oracle lib/sentimentLib/hedonometer.json (pinned, 10,222 entries)",
        required=True,
        description="Full happiness norms (FR-4.2); user-supplied copy, checksum-verified",
        redistribution="user-supplied",
    ),
    AssetSpec(
        name="brysbaert-concreteness",
        version="Brysbaert et al. 2014 (BRM 40k)",
        path="style/Concreteness_ratings_Brysbaert_et_al_BRM.csv",
        sha256="cdfd684ab15d303b6f7a90ba30ca7a180695cd5d7b0eee6ea364d2d6abd9c63e",
        license="research use (Springer BRM 2014) — redistribution decision owed to FR-0.3",
        source="legacy oracle lib/concretenessLib (pinned b4a5087); https://link.springer.com/article/10.3758/s13428-013-0403-5",
        required=False,
        description="40k concreteness norms (FR-2.8); oracle-shipped copy checksum-verified",
        redistribution="user-supplied",
    ),
    AssetSpec(
        name="iconicity-ratings",
        version="Perry et al. dataset",
        path="style/iconicity_ratings.csv",
        sha256="d66bef6a070c845dc11acaa2dfab95cae1f5db83efd687ec18f2f23a31fe6150",
        license="research use — redistribution unclear, FR-0.3 decision owed",
        source="legacy oracle lib/iconicityLib (pinned b4a5087)",
        required=False,
        description="Iconicity norms (FR-2.8); oracle-shipped copy checksummed",
        redistribution="user-supplied",
    ),
    AssetSpec(
        name="wordnet",
        version="3.0",
        path="wordnet/dict",
        sha256="",
        license="WordNet 3.0 license (Princeton) — redistribution permitted with license text",
        source="https://wordnet.princeton.edu/download/current-version",
        required=False,
        description="Full WordNet for traversal + nominalization (FR-4.4, FR-2.7)",
        redistribution="download-on-demand",
    ),
    AssetSpec(
        name="nrc-lexicon",
        version="NRC Emotion Lexicon v0.92 (Mohammad & Turney)",
        path="sentiment/nrc_lexicon.json",
        sha256="",
        license="research use with citation — redistribution requires permission; bundled copy ships with nrclex",
        source="nrclex package data (nrc_en.json) or https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm",
        required=False,
        description="NRC word-emotion associations, JSON word->[emotions] (FR-4.3); nrclex-bundled file is auto-detected",
        redistribution="user-supplied",
    ),
    AssetSpec(
        name="space-typology",
        version="legacy curated list (pinned b4a5087)",
        path="symbolic/symbolic_space_typology.csv",
        sha256="",
        license="suite-curated list — redistribution permitted",
        source="legacy oracle lib/symbolic_space_typology.csv (pinned b4a5087)",
        required=False,
        description="Curated space-type lexicon, term,category (FR-4.6)",
        redistribution="user-supplied",
    ),
    AssetSpec(
        name="actor-typology",
        version="legacy curated list (pinned b4a5087)",
        path="symbolic/social_actor_typology.csv",
        sha256="",
        license="suite-curated list — redistribution permitted",
        source="legacy oracle lib/social_actor_typology.csv (pinned b4a5087)",
        required=False,
        description="Curated social-actor lexicon, term,category (FR-4.6)",
        redistribution="user-supplied",
    ),
)


@dataclass(frozen=True, slots=True)
class AssetRegistry:
    root: Path
    specs: tuple[AssetSpec, ...] = BUILTIN_ASSETS

    def __post_init__(self) -> None:
        # C6-15: duplicate names and duplicate target paths are construction
        # errors; every path must be relative and stay inside the root.
        names: set[str] = set()
        paths: set[str] = set()
        for spec in self.specs:
            if spec.name in names:
                raise ValueError(f"duplicate asset name: {spec.name}")
            names.add(spec.name)
            path_key = spec.path.casefold()
            if path_key in paths:
                raise ValueError(f"duplicate asset target path: {spec.path}")
            paths.add(path_key)
            candidate = Path(spec.path)
            # Rooted paths ("\\x", "/x") are not is_absolute() on POSIX-view
            # Windows but still escape the root; drive letters too.
            windows = PureWindowsPath(spec.path)
            posix = PurePosixPath(spec.path)
            if (
                candidate.is_absolute()
                or candidate.drive
                or candidate.root
                or windows.is_absolute()
                or windows.drive
                or posix.is_absolute()
                or ".." in candidate.parts
                or ".." in windows.parts
                or ".." in posix.parts
            ):
                raise ValueError(f"asset path escapes root: {spec.path}")

    def _spec(self, name: str) -> AssetSpec | None:
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None

    def status(self, name: str) -> Result[str]:
        """OK / MISSING / CORRUPT / UNSTAMPED for one asset (reads only hashes)."""
        spec = self._spec(name)
        if spec is None:
            return Result.failure(Diagnostic.error("ASSET_UNKNOWN", f"no asset named {name!r}", name=name))
        if not spec.sha256:
            return Result.success(
                "UNSTAMPED",
                Diagnostic.warning(
                    "ASSET_UNSTAMPED",
                    f"{name} has no pinned checksum; obtain from {spec.source} and stamp it",
                    name=name,
                    source=spec.source,
                    install=f"place the file at {self.root / spec.path}",
                ),
            )
        target = self.root / spec.path
        try:
            target.resolve().relative_to(self.root.resolve())
        except ValueError:
            return Result.success(
                "CORRUPT",
                Diagnostic.warning(
                    "ASSET_PATH_ESCAPE",
                    f"{name} resolves outside the registry root; refusing to inspect it",
                    name=name,
                    path=str(target),
                ),
            )
        is_dir_asset = target.is_dir()
        if not target.is_file() and not is_dir_asset:
            return Result.success(
                "MISSING",
                Diagnostic.warning(
                    "ASSET_MISSING",
                    f"{name} v{spec.version} missing; install from {spec.source}",
                    name=name,
                    install=f"place the file at {target}",
                ),
            )
        try:
            if is_dir_asset:
                digest = _tree_hash(target)
                expected = f"tree:{spec.sha256}" if spec.sha256 and not spec.sha256.startswith("tree:") else spec.sha256
            else:
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                expected = spec.sha256
        except OSError as exc:
            return Result.success(
                "CORRUPT",
                Diagnostic.warning("ASSET_UNREADABLE", f"{name} unreadable: {exc}", name=name),
            )
        if digest != expected:
            return Result.success(
                "CORRUPT",
                Diagnostic.warning(
                    "ASSET_CORRUPT",
                    f"{name} checksum mismatch (expected {spec.sha256[:12]}…, got {digest[:12]}…)",
                    name=name,
                    expected=spec.sha256,
                    actual=digest,
                ),
            )
        return Result.success("OK")

    def path(self, name: str) -> Result[Path]:
        """Verified path for one asset (checksum enforced, UNSTAMPED refused)."""
        state = self.status(name)
        if state.value is None:
            return Result.failure(*state.diagnostics)
        if state.unwrap() != "OK":
            return Result.failure(
                Diagnostic.error(
                    "ASSET_NOT_USABLE",
                    f"{name} is {state.unwrap()}; refusing to load",
                    name=name,
                    status=state.unwrap(),
                )
            )
        spec = self._spec(name)
        if spec is None:  # unreachable: status() already resolved it
            return Result.failure(Diagnostic.error("ASSET_UNKNOWN", f"no asset named {name!r}", name=name))
        return Result.success(self.root / spec.path)

    def load_text(self, name: str) -> Result[str]:
        """Read one asset as UTF-8 text (lazy, verified)."""
        target = self.path(name)
        if target.value is None:
            return Result.failure(*target.diagnostics)
        try:
            return Result.success(target.unwrap().read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            return Result.failure(Diagnostic.error("ASSET_UNREADABLE", f"{name} unreadable: {exc}", name=name))

    def status_table(self) -> Result[pd.DataFrame]:
        """Every spec with its status (diagnostics attached as warnings)."""
        rows = []
        diags: list[Diagnostic] = []
        for spec in self.specs:
            state = self.status(spec.name)
            status = state.unwrap_or("ERROR")
            diags.extend(state.diagnostics)
            rows.append(
                {
                    "Asset": spec.name,
                    "Version": spec.version,
                    "Status": status,
                    "Required": spec.required,
                    "License": spec.license,
                }
            )
        out = pd.DataFrame(rows, columns=["Asset", "Version", "Status", "Required", "License"])
        return Result.success(out, *diags)


def _tree_hash(directory: Path) -> str:
    """Deterministic manifest hash for a directory asset (name + content)."""
    digest = hashlib.sha256()
    for member in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(member.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(member.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return "tree:" + digest.hexdigest()


def default_registry(root: Path | None = None) -> AssetRegistry:
    """Registry rooted at ``assets/`` (repo root) unless told otherwise."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent.parent.parent / "assets"
    return AssetRegistry(root=base)
