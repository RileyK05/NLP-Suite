"""FR-4.1 asset registry — versioned assets, checksums, lazy loads, diagnostics."""

from __future__ import annotations

from pathlib import Path

from core.assets.registry import AssetRegistry, AssetSpec


def _registry(root: Path) -> AssetRegistry:
    (root / "lex").mkdir(parents=True, exist_ok=True)
    (root / "lex" / "good.txt").write_text("a\nb\n", encoding="utf-8")
    (root / "lex" / "bad.txt").write_text("tampered", encoding="utf-8")
    import hashlib

    good_sha = hashlib.sha256((root / "lex" / "good.txt").read_bytes()).hexdigest()
    specs = (
        AssetSpec(
            name="good-lexicon",
            version="1.0",
            path="lex/good.txt",
            sha256=good_sha,
            license="public domain (test fixture)",
            source="test",
            required=True,
            description="fixture",
        ),
        AssetSpec(
            name="corrupt-lexicon",
            version="1.0",
            path="lex/bad.txt",
            sha256="0" * 64,
            license="public domain (test fixture)",
            source="test",
            required=True,
            description="fixture",
        ),
        AssetSpec(
            name="missing-lexicon",
            version="1.0",
            path="lex/ghost.txt",
            sha256="0" * 64,
            license="public domain (test fixture)",
            source="test",
            required=False,
            description="fixture",
        ),
        AssetSpec(
            name="unstamped-lexicon",
            version="0.0",
            path="lex/unstamped.txt",
            sha256="",
            license="unknown — pin before production use",
            source="test",
            required=False,
            description="fixture",
        ),
    )
    return AssetRegistry(root=root, specs=specs)


class TestStatus:
    def test_ok_missing_corrupt_unstamped(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path)
        assert registry.status("good-lexicon").unwrap() == "OK"
        missing = registry.status("missing-lexicon")
        assert missing.unwrap() == "MISSING"
        assert any(d.code == "ASSET_MISSING" for d in missing.diagnostics)
        corrupt = registry.status("corrupt-lexicon")
        assert corrupt.unwrap() == "CORRUPT"
        assert any(d.code == "ASSET_CORRUPT" for d in corrupt.diagnostics)
        assert registry.status("unstamped-lexicon").unwrap() == "UNSTAMPED"

    def test_unknown_asset_is_an_error(self, tmp_path: Path) -> None:
        result = _registry(tmp_path).status("nope")
        assert result.value is None
        assert any(d.code == "ASSET_UNKNOWN" for d in result.diagnostics)

    def test_status_table(self, tmp_path: Path) -> None:
        frame = _registry(tmp_path).status_table().unwrap()
        assert set(frame["Status"].tolist()) == {"OK", "CORRUPT", "MISSING", "UNSTAMPED"}


class TestLoading:
    def test_load_text_lazy_and_checked(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path)
        assert registry.load_text("good-lexicon").unwrap() == "a\nb\n"
        assert registry.load_text("corrupt-lexicon").value is None
        assert registry.load_text("missing-lexicon").value is None

    def test_resolve_path(self, tmp_path: Path) -> None:
        registry = _registry(tmp_path)
        assert registry.path("good-lexicon").unwrap() == tmp_path / "lex" / "good.txt"


class TestBuiltinRegistry:
    def test_known_assets_listed(self) -> None:
        from core.assets.registry import BUILTIN_ASSETS

        names = {spec.name for spec in BUILTIN_ASSETS}
        assert {
            "vader-lexicon",
            "anew-lexicon",
            "sentiwordnet",
            "hedonometer",
            "brysbaert-concreteness",
            "iconicity-ratings",
            "wordnet",
        } <= names

    def test_missing_diagnostic_is_actionable(self, tmp_path: Path) -> None:
        from core.assets.registry import default_registry

        registry = default_registry(tmp_path)
        result = registry.status("vader-lexicon")
        assert result.unwrap() in ("MISSING", "UNSTAMPED")
        assert any("install" in d.context or "install" in d.message.lower() for d in result.diagnostics)


class TestCli:
    def test_cli_reports_table(self, tmp_path: Path) -> None:
        from tools.assets import main

        assert main(["--root", str(tmp_path)]) == 1  # required assets absent


class TestC615Corrections:
    """C6-15: real metadata, path/duplicate validation, directory assets."""

    def test_builtin_metadata_is_exact(self) -> None:
        from core.assets.registry import BUILTIN_ASSETS

        for spec in BUILTIN_ASSETS:
            assert spec.version != "full", spec.name  # exact identifiers
            assert "http" in spec.source or "lib/" in spec.source, spec.name
            assert spec.redistribution in ("redistributable", "download-on-demand", "user-supplied")
            assert "verify before" not in spec.license  # resolved positions, not hedges

    def test_oracle_shipped_assets_pinned(self) -> None:
        from core.assets.registry import BUILTIN_ASSETS

        by_name = {s.name: s for s in BUILTIN_ASSETS}
        for asset in ("brysbaert-concreteness", "iconicity-ratings"):
            assert by_name[asset].sha256, f"{asset} checksum must be pinned (oracle copy exists)"

    def test_duplicate_name_rejected(self, tmp_path: Path) -> None:
        import pytest

        from core.assets.registry import AssetRegistry, AssetSpec

        spec = AssetSpec(
            name="x", version="1", path="a.txt", sha256="", license="l", source="s", required=False, description=""
        )
        with pytest.raises(ValueError, match="duplicate asset name"):
            AssetRegistry(root=tmp_path, specs=(spec, spec))

    def test_duplicate_path_rejected(self, tmp_path: Path) -> None:
        import pytest

        from core.assets.registry import AssetRegistry, AssetSpec

        def make(n: str, p: str) -> AssetSpec:
            return AssetSpec(
                name=n, version="1", path=p, sha256="", license="l", source="s", required=False, description=""
            )

        with pytest.raises(ValueError, match="duplicate asset target path"):
            AssetRegistry(root=tmp_path, specs=(make("a", "x.txt"), make("b", "x.txt")))

    def test_path_escape_rejected(self, tmp_path: Path) -> None:
        import pytest

        from core.assets.registry import AssetRegistry, AssetSpec

        for bad in ("../evil.txt", "C:/abs/evil.txt", "/abs/evil.txt"):
            spec = AssetSpec(
                name="x", version="1", path=bad, sha256="", license="l", source="s", required=False, description=""
            )
            with pytest.raises(ValueError, match="escapes root"):
                AssetRegistry(root=tmp_path, specs=(spec,))

    def test_directory_asset_tree_checksum(self, tmp_path: Path) -> None:
        from core.assets.registry import AssetRegistry, AssetSpec, _tree_hash

        wn = tmp_path / "wordnet" / "dict"
        wn.mkdir(parents=True)
        (wn / "index.sense").write_text("w1\n", encoding="utf-8")
        (wn / "data.noun").write_text("d1\n", encoding="utf-8")
        tree = _tree_hash(wn)
        spec = AssetSpec(
            name="wordnet",
            version="3.0",
            path="wordnet/dict",
            sha256=tree.replace("tree:", ""),
            license="l",
            source="s",
            required=False,
            description="",
        )
        registry = AssetRegistry(root=tmp_path, specs=(spec,))
        assert registry.status("wordnet").unwrap() == "OK"
        (wn / "data.verb").write_text("v\n", encoding="utf-8")  # any change -> corrupt
        assert registry.status("wordnet").unwrap() == "CORRUPT"
