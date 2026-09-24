"""FR-3.1 document conversion — TXT/CSV/TSV/HTML to text; loud missing-lib errors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.file_ops import converter as C


class TestDispatch:
    def test_txt_passthrough(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hello", encoding="utf-8")
        assert C.convert_document_to_text(f).unwrap() == "hello"

    def test_html_strips_tags(self, tmp_path: Path) -> None:
        f = tmp_path / "a.html"
        f.write_text(
            "<html><head><title>T</title><style>.x{}</style></head>"
            "<body><h1>Hi</h1><script>evil()</script><p>there &amp; then</p></body></html>",
            encoding="utf-8",
        )
        text = C.convert_document_to_text(f).unwrap()
        assert "Hi" in text and "there & then" in text
        assert "evil" not in text and ".x{}" not in text

    def test_csv_cells_become_text(self, tmp_path: Path) -> None:
        f = tmp_path / "a.csv"
        f.write_text("name,city\nAnn,Rome\n", encoding="utf-8")
        text = C.convert_document_to_text(f).unwrap()
        assert "Ann" in text and "Rome" in text

    def test_doc_gets_explicit_unsupported(self, tmp_path: Path) -> None:
        f = tmp_path / "a.doc"
        f.write_bytes(b"fake binary doc")
        result = C.convert_document_to_text(f)
        assert result.value is None
        diag = result.diagnostics[0]
        assert diag.code == "CONVERT_UNSUPPORTED_FORMAT"
        assert "docx" in diag.message  # names the actual remediation path

    def test_pdf_docx_rtf_name_their_missing_lib(self, tmp_path: Path) -> None:
        for suffix, lib in [(".pdf", "pdfminer"), (".docx", "python-docx"), (".rtf", "striprtf")]:
            f = tmp_path / f"a{suffix}"
            f.write_bytes(b"fake")
            result = C.convert_document_to_text(f)
            assert result.value is None
            diag = result.diagnostics[0]
            assert diag.code in ("CONVERT_NEEDS_LIB", "CONVERT_MALFORMED", "CONVERT_PDF_MALFORMED")
            if diag.code == "CONVERT_NEEDS_LIB":
                assert lib in diag.message or "extra: converters" in diag.message

    def test_html_preserves_paragraph_boundaries(self, tmp_path: Path) -> None:
        f = tmp_path / "a.html"
        f.write_text("<p>First para</p><p>Second para</p>", encoding="utf-8")
        text = C.convert_document_to_text(f).unwrap()
        assert "First para" in text and "Second para" in text
        assert chr(10) in text  # paragraphs are separate lines, not one line

    def test_csv_header_preserved(self, tmp_path: Path) -> None:
        f = tmp_path / "a.csv"
        f.write_text("name,city" + chr(10) + "Ann,Rome" + chr(10), encoding="utf-8")
        text = C.convert_document_to_text(f).unwrap()
        assert text.splitlines()[0] == "name city"
        assert "Ann Rome" in text

    def test_latin1_text_reads_via_encoding_chain(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_bytes("café latte".encode("cp1252"))
        text = C.convert_document_to_text(f).unwrap()
        assert "caf" in text  # cp1252 fallback, no CONVERT_READ_FAILED

    def test_unknown_suffix_and_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "a.xyz"
        f.write_bytes(b"fake")
        assert C.convert_document_to_text(f).value is None
        empty = tmp_path / "e.txt"
        empty.write_text("   ", encoding="utf-8")
        assert any(d.code == "CONVERT_EMPTY" for d in C.convert_document_to_text(empty).diagnostics)

    def test_supported_suffixes(self) -> None:
        assert ".html" in C.supported_suffixes()
        assert ".pdf" in C.supported_suffixes()  # dispatched; needs the extra
        assert ".doc" not in C.supported_suffixes()  # explicit unsupported


class TestCli:
    def test_cli_converts_dir(self, tmp_path: Path) -> None:
        from tools.convert import main

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("hello", encoding="utf-8")
        (src / "b.html").write_text("<p>world</p>", encoding="utf-8")
        (src / "c.pdf").write_bytes(b"fake")
        out = tmp_path / "out"
        assert main([str(src), str(out)]) == 0
        run_dir = next(out.iterdir())
        assert (run_dir / "a_txt.txt").is_file()  # C6-10: suffix-bearing name
        assert (run_dir / "conversion_report.csv").is_file()


class TestC67Corrections:
    """C6-10: collisions, empty dir, failures in envelope, .doc honesty."""

    def test_colliding_stems_disambiguated(self, tmp_path: Path) -> None:
        from tools.convert import main

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("plain", encoding="utf-8")
        (src / "a.html").write_text("<p>markup</p>", encoding="utf-8")
        out = tmp_path / "out"
        assert main([str(src), str(out)]) == 0
        run_dir = next(out.iterdir())
        names = {p.name for p in run_dir.iterdir()}
        assert "a_txt.txt" in names and "a_html.txt" in names

    def test_empty_input_dir_fails(self, tmp_path: Path) -> None:
        from tools.convert import main

        src = tmp_path / "src"
        src.mkdir()
        assert main([str(src), str(tmp_path / "out")]) == 2

    def test_failed_conversions_in_report_and_envelope(self, tmp_path: Path) -> None:
        import json

        from tools.convert import main

        src = tmp_path / "src"
        src.mkdir()
        (src / "ok.txt").write_text("fine", encoding="utf-8")
        (src / "bad.pdf").write_bytes(b"broken pdf")
        out = tmp_path / "out"
        assert main([str(src), str(out)]) == 0  # per-file failure does not sink the run
        run_dir = next(out.iterdir())
        report = pd.read_csv(run_dir / "conversion_report.csv")
        assert set(report["Status"]) == {"OK", "FAILED"}
        envelope = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        codes = [d["code"] for d in envelope["diagnostics"]]
        assert any(c.startswith("CONVERT_") for c in codes)
