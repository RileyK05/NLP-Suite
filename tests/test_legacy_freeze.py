"""FR-1.3 legacy freeze — the oracle pin is reproducible and honest.

The freezer only reads plain files (no git binary, no imports, no network),
so these tests build miniature oracles in tmp dirs and check the record.
"""

from __future__ import annotations

from pathlib import Path

from core.legacy_freeze import freeze_oracle, parse_requirement_names


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _minimal_oracle(root: Path) -> Path:
    _write(root / "lib" / "release_version.txt", "1.6.38")
    _write(root / "requirements.txt", "pandas\nstanza==1.7\n")
    _write(root / "src" / "a_util.py", "x = 1\n")
    return root


class TestRequirementParsing:
    def test_strips_comments_blanks_extras_and_markers(self) -> None:
        names, pinned = parse_requirement_names(
            "# comment\npandas\nfuzzywuzzy[speedup]\nspacy==3.7; python_version>'3.8'\n\n"
        )
        assert names == ("pandas", "fuzzywuzzy[speedup]", "spacy")
        assert pinned == (False, False, True)

    def test_empty_file(self) -> None:
        assert parse_requirement_names("") == ((), ())


class TestGitResolution:
    def test_loose_ref(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        _write(root / ".git" / "HEAD", "ref: refs/heads/redesign\n")
        _write(root / ".git" / "refs" / "heads" / "redesign", "b4a5087\n")
        record = freeze_oracle(root).unwrap()
        assert record.is_git_checkout
        assert record.commit == "b4a5087"

    def test_packed_ref(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        _write(root / ".git" / "HEAD", "ref: refs/heads/main\n")
        _write(root / ".git" / "packed-refs", "# pack-refs\ncafef00 refs/heads/main\n")
        assert freeze_oracle(root).unwrap().commit == "cafef00"

    def test_detached_head(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        _write(root / ".git" / "HEAD", "deadbee\n")
        record = freeze_oracle(root).unwrap()
        assert record.commit == "deadbee"

    def test_unresolvable_ref_warns(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        _write(root / ".git" / "HEAD", "ref: refs/heads/ghost\n")
        record = freeze_oracle(root).unwrap()
        assert record.commit == ""
        assert any("commit" in w.lower() for w in record.warnings)

    def test_no_git_checkout(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        record = freeze_oracle(root).unwrap()
        assert not record.is_git_checkout
        assert record.commit == ""


class TestRecordContents:
    def test_missing_oracle_is_a_harness_error(self, tmp_path: Path) -> None:
        result = freeze_oracle(tmp_path / "ghost")
        assert result.value is None
        assert any(d.code == "FREEZE_NO_ORACLE" for d in result.diagnostics)

    def test_unpinned_requirements_warn(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        _write(root / "requirements.txt", "pandas\nstanza\n")
        record = freeze_oracle(root).unwrap()
        assert any("unpinned" in w.lower() for w in record.warnings)

    def test_external_software_csv_parsed(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        _write(
            root / "config" / "NLP_setup_external_software_config.csv",
            "Software,Installation_path,Download_link\nMALLET,C:\\mallet,http://example.invalid\n",
        )
        record = freeze_oracle(root).unwrap()
        assert len(record.external_software) == 1
        assert record.external_software[0].name == "MALLET"

    def test_malformed_software_csv_warns(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        _write(root / "config" / "NLP_setup_external_software_config.csv", "not,a,header\nx\n")
        record = freeze_oracle(root).unwrap()
        assert record.external_software == ()
        assert any("external software" in w.lower() for w in record.warnings)

    def test_big_lib_files_are_sized_not_hashed(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        big_file = root / "lib" / "big.bin"
        big_file.parent.mkdir(parents=True, exist_ok=True)
        with open(big_file, "wb") as fh:
            fh.seek(2048)
            fh.write(b"\0")
        record = freeze_oracle(root, hash_limit_bytes=1024).unwrap()
        big = next(e for e in record.lib_entries if e.name == "big.bin")
        assert not big.hashed and big.sha256 == "" and big.size > 1024

    def test_fingerprint_stable_and_sensitive(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        first = freeze_oracle(root).unwrap().tree_fingerprint
        assert freeze_oracle(root).unwrap().tree_fingerprint == first
        _write(root / "src" / "new_util.py", "y = 2\n")
        assert freeze_oracle(root).unwrap().tree_fingerprint != first

    def test_working_tree_attestation(self, tmp_path: Path) -> None:
        import pytest

        from core.legacy_freeze import freeze_oracle as freeze

        root = _minimal_oracle(tmp_path / "oracle")
        assert "operator-attested" in freeze(root, working_tree="clean").unwrap().to_markdown()
        assert "operator-attested" not in freeze(root).unwrap().to_markdown()
        with pytest.raises(ValueError, match="working_tree"):
            freeze(root, working_tree="maybe")

    def test_markdown_names_commit_and_gaps(self, tmp_path: Path) -> None:
        root = _minimal_oracle(tmp_path / "oracle")
        _write(root / ".git" / "HEAD", "ref: refs/heads/main\n")
        _write(root / ".git" / "refs" / "heads" / "main", "cafef00\n")
        text = freeze_oracle(root).unwrap().to_markdown()
        assert "cafef00" in text and "1.6.38" in text
        assert "npins" not in text  # no placeholder jargon leaks into the doc


class TestCli:
    def test_freeze_cli_prints_record(self, tmp_path: Path, capsys: object) -> None:
        from tools.freeze_legacy import main

        root = _minimal_oracle(tmp_path / "oracle")
        assert main(["--oracle", str(root)]) == 0
        out = capsys.readouterr().out
        assert "# Legacy environment freeze" in out
        assert "1.6.38" in out

    def test_freeze_cli_missing_oracle(self) -> None:
        from tools.freeze_legacy import main

        assert main(["--oracle", "Z:/no/such/dir"]) == 2
