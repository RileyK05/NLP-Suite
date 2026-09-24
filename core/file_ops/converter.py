"""File converter — between plain text and simple table formats.

Document intake (FR-3.1). Converters here:

* TXT — passthrough (shared encoding chain, C6-10).
* CSV/TSV — pandas table; rows become one text line each (header kept as
  first line; provenance = the source file hash recorded by the CLI).
* HTML — stdlib parser; block tags (p/div/h1..h4/li/tr/br) preserve
  paragraph boundaries instead of collapsing to one line (C6-10).
* PDF — pdfminer.six (lazy import; ``[converters]`` extra) with encrypted /
  malformed / image-only / partial-extraction diagnostics (C6-10).
* DOCX — python-docx (lazy import); paragraph-per-line.
* RTF — striprtf (lazy import).
* DOC (legacy binary) — NOT claimable: python-docx cannot parse it. It is
  its own unsupported capability with an explicit diagnostic (C6-10).

A missing optional package is detected at call time
(``CONVERT_NEEDS_LIB`` naming the exact extra), never at import (R1).
"""

from __future__ import annotations

from collections.abc import Callable
import html as html_module
from html.parser import HTMLParser
import importlib.util
import io
from io import BytesIO
from pathlib import Path

import pandas as pd

from core.result import Diagnostic, Result

__all__ = [
    "convert_document_to_text",
    "convert_table_to_text",
    "convert_text_to_table",
    "supported_suffixes",
]

_CONVERTIBLE = frozenset({".txt", ".csv", ".tsv", ".html", ".htm", ".pdf", ".docx", ".rtf"})

_LIB_BY_SUFFIX = {
    ".pdf": ("pdfminer", "pip install pdfminer.six  (extra: converters)"),
    ".docx": ("docx", "pip install python-docx  (extra: converters)"),
    ".rtf": ("striprtf", "pip install striprtf  (extra: converters)"),
}


def supported_suffixes() -> frozenset[str]:
    """Suffixes this module dispatches on (conversion may still need an extra)."""
    return _CONVERTIBLE


def _importlib_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


class _TextExtractor(HTMLParser):
    """HTML -> text preserving block boundaries (C6-10)."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "li", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        # Collapse intra-paragraph whitespace, keep paragraph breaks.
        raw = html_module.unescape("".join(self.parts))
        paragraphs = [" ".join(block.split()) for block in raw.split("\n")]
        return "\n".join(block for block in paragraphs if block)


def _pdf_to_text(raw: bytes) -> Result[str]:
    if not _importlib_available("pdfminer"):
        return Result.failure(
            Diagnostic.error(
                "CONVERT_NEEDS_LIB",
                ".pdf needs pdfminer.six — pip install pdfminer.six (extra: converters)",
                suffix=".pdf",
            )
        )
    try:  # lazy: optional converters extra (R1: never at import time)
        from pdfminer.high_level import extract_text as pdf_extract  # noqa: PLC0415
        from pdfminer.pdfdocument import PDFDocument  # noqa: PLC0415
        from pdfminer.pdfpage import PDFPage  # noqa: PLC0415
        from pdfminer.pdfparser import PDFParser  # noqa: PLC0415

        def _is_encrypted(data: bytes) -> bool:
            try:
                PDFDocument(PDFParser(BytesIO(data)))
            except Exception:
                return False
            return False

        try:
            pages = list(PDFPage.create_pages(PDFDocument(PDFParser(BytesIO(raw)))))
            if not pages:
                return Result.failure(
                    Diagnostic.error("CONVERT_PDF_NO_PAGES", "PDF has no pages (empty or image-only)", suffix=".pdf")
                )
        except Exception as exc:
            first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            if "password" in first.lower() or "encrypt" in first.lower():
                return Result.failure(Diagnostic.error("CONVERT_PDF_ENCRYPTED", f"PDF is encrypted: {first}"))
            return Result.failure(
                Diagnostic.error("CONVERT_PDF_MALFORMED", f"PDF unreadable: {first}", detail=type(exc).__name__)
            )
        text = pdf_extract(BytesIO(raw))
        if not text.strip():
            # Pages exist but no text layer: image-only PDF.
            return Result.failure(
                Diagnostic.error(
                    "CONVERT_PDF_IMAGE_ONLY",
                    "PDF has pages but no extractable text layer (likely scanned images)",
                    suffix=".pdf",
                )
            )
        return Result.success(text)
    except ImportError as exc:
        return Result.failure(Diagnostic.error("CONVERT_NEEDS_LIB", f".pdf needs pdfminer.six ({exc})", suffix=".pdf"))
    except Exception as exc:  # partial extraction / unexpected parser state
        first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return Result.failure(
            Diagnostic.error("CONVERT_PDF_PARTIAL", f"PDF extraction incomplete: {first}", detail=type(exc).__name__)
        )


def _docx_to_text(raw: bytes) -> Result[str]:
    if not _importlib_available("docx"):
        return Result.failure(
            Diagnostic.error(
                "CONVERT_NEEDS_LIB",
                ".docx needs python-docx — pip install python-docx (extra: converters)",
                suffix=".docx",
            )
        )
    try:  # lazy: optional converters extra
        import docx  # noqa: PLC0415

        document = docx.Document(BytesIO(raw))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        if not paragraphs:
            return Result.failure(Diagnostic.error("CONVERT_EMPTY", "DOCX has no non-empty paragraphs"))
        return Result.success("\n".join(paragraphs))
    except ImportError as exc:
        return Result.failure(Diagnostic.error("CONVERT_NEEDS_LIB", f".docx needs python-docx ({exc})", suffix=".docx"))
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return Result.failure(
            Diagnostic.error("CONVERT_MALFORMED", f"DOCX unreadable: {first}", detail=type(exc).__name__)
        )


def _rtf_to_text(raw: bytes) -> Result[str]:
    if not _importlib_available("striprtf"):
        return Result.failure(
            Diagnostic.error(
                "CONVERT_NEEDS_LIB",
                ".rtf needs striprtf — pip install striprtf (extra: converters)",
                suffix=".rtf",
            )
        )
    try:  # lazy: optional converters extra
        from striprtf.striprtf import rtf_to_text as striprtf_rtf_to_text  # noqa: PLC0415

        # striprtf ships py.typed but leaves rtf_to_text unannotated, so under
        # strict mypy calling it is an untyped call. Naming the signature here
        # confines that to this one line instead of letting Any leak into the
        # decoded text and out through this function's Result.
        rtf_to_text: Callable[[str], str] = striprtf_rtf_to_text

        decoded: str | None = None
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                decoded = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            return Result.failure(Diagnostic.error("CONVERT_MALFORMED", "RTF could not be decoded as text"))
        # Every RTF document opens with the {\rtf control word. striprtf is
        # lenient by design -- it strips control words and passes anything else
        # through -- so without this check a plain text file, or a file named
        # .rtf by mistake, converts "successfully" into its own raw bytes and
        # enters the corpus as a document. A named format that silently accepts
        # anything is worse than one that refuses.
        if not decoded.lstrip("﻿ \t\r\n").startswith("{\\rt"):
            return Result.failure(
                Diagnostic.error(
                    "CONVERT_MALFORMED",
                    r"not an RTF document: the file does not begin with the {\rtf control word",
                    suffix=".rtf",
                )
            )
        text = rtf_to_text(decoded)
        if not text.strip():
            return Result.failure(Diagnostic.error("CONVERT_EMPTY", "RTF converted to empty text"))
        return Result.success(text)
    except ImportError as exc:
        return Result.failure(Diagnostic.error("CONVERT_NEEDS_LIB", f".rtf needs striprtf ({exc})", suffix=".rtf"))
    except (UnicodeDecodeError, Exception) as exc:
        if isinstance(exc, UnicodeDecodeError):
            return Result.failure(Diagnostic.error("CONVERT_MALFORMED", f"RTF is not valid text: {exc}"))
        first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return Result.failure(Diagnostic.error("CONVERT_MALFORMED", f"RTF unreadable: {first}"))


def _read_bytes(path: Path) -> Result[bytes]:
    try:
        return Result.success(path.read_bytes())
    except OSError as exc:
        return Result.failure(Diagnostic.error("CONVERT_READ_FAILED", f"could not read {path}: {exc}", path=str(path)))


def convert_document_to_text(input_path: Path) -> Result[str]:
    """Convert one document file to plain intake text by suffix dispatch."""
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".doc":
        # C6-10: python-docx CANNOT parse legacy binary .doc. Explicit
        # unsupported capability — no pretending, no wrong-library advice.
        return Result.failure(
            Diagnostic.error(
                "CONVERT_UNSUPPORTED_FORMAT",
                ".doc (legacy binary) is not supported; convert to .docx first "
                "(e.g. LibreOffice: soffice --convert-to docx file.doc)",
                suffix=".doc",
                path=str(path),
            )
        )
    if suffix not in _CONVERTIBLE:
        return Result.failure(
            Diagnostic.error("CONVERT_UNSUPPORTED", f"no converter for {suffix}", suffix=suffix, path=str(path))
        )
    raw_result = _read_bytes(path)
    if raw_result.value is None:
        return Result.failure(*raw_result.diagnostics)
    raw = raw_result.unwrap()
    if suffix == ".pdf":
        return _pdf_to_text(raw)
    if suffix == ".docx":
        return _docx_to_text(raw)
    if suffix == ".rtf":
        return _rtf_to_text(raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Shared encoding chain (reader.py): utf-8 -> utf-8-sig -> cp1252.
        decoded = None
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                decoded = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            return Result.failure(
                Diagnostic.error(
                    "CONVERT_READ_FAILED",
                    f"could not decode {path} as utf-8/utf-8-sig/cp1252",
                    path=str(path),
                )
            )
        text = decoded
    if not text.strip():
        return Result.failure(Diagnostic.error("CONVERT_EMPTY", f"{path} is empty", path=str(path)))
    if suffix == ".txt":
        out = text
    elif suffix in (".csv", ".tsv"):
        delim = "," if suffix == ".csv" else "\t"
        table_result = convert_text_to_table(text, delimiter=delim)
        if table_result.value is None:
            return Result.failure(*table_result.diagnostics)
        frame = table_result.unwrap()
        # C6-10: header kept as the first line; each data row one line.
        header = " ".join(str(c) for c in frame.columns)
        body = [" ".join("" if pd.isna(v) else str(v) for v in row) for row in frame.itertuples(index=False)]
        out = "\n".join([header, *body])
    else:
        extractor = _TextExtractor()
        extractor.feed(text)
        out = extractor.text()
    if not out.strip():
        return Result.failure(Diagnostic.error("CONVERT_EMPTY", f"{path} converted to empty text", path=str(path)))
    return Result.success(out)


def convert_text_to_table(text: str, delimiter: str = ",") -> Result[pd.DataFrame]:
    """Parse *text* as a delimited table (CSV/TSV)."""
    if not text.strip():
        return Result.failure(Diagnostic.error("CONVERT_EMPTY", "text is empty"))
    try:
        frame = pd.read_csv(io.StringIO(text), delimiter=delimiter)
    except Exception as exc:
        return Result.failure(Diagnostic.error("CONVERT_PARSE_FAILED", f"could not parse table: {exc}"))
    return Result.success(frame)


def convert_table_to_text(frame: pd.DataFrame, delimiter: str = ",") -> Result[str]:
    """Serialize a DataFrame as delimited text."""
    if frame.empty and len(frame.columns) == 0:
        return Result.failure(Diagnostic.error("CONVERT_EMPTY", "table has no columns"))
    try:
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False, sep=delimiter)
        return Result.success(buffer.getvalue())
    except Exception as exc:
        return Result.failure(Diagnostic.error("CONVERT_SERIALIZE_FAILED", f"could not serialize table: {exc}"))


def convert_file(input_path: Path, output_path: Path, input_delim: str = ",", output_delim: str = ",") -> Result[Path]:
    """Convert a file on disk; pure wrapper around the two functions above."""
    try:
        text = Path(input_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return Result.failure(
            Diagnostic.error("CONVERT_READ_FAILED", f"could not read {input_path}: {exc}", path=str(input_path))
        )
    table_result = convert_text_to_table(text, delimiter=input_delim)
    if table_result.value is None:
        return Result[Path](None, table_result.diagnostics)
    text_result = convert_table_to_text(table_result.unwrap(), delimiter=output_delim)
    if text_result.value is None:
        return Result[Path](None, text_result.diagnostics)
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text_result.unwrap(), encoding="utf-8")
    except OSError as exc:
        return Result.failure(
            Diagnostic.error("CONVERT_WRITE_FAILED", f"could not write {output_path}: {exc}", path=str(output_path))
        )
    return Result.success(Path(output_path))
