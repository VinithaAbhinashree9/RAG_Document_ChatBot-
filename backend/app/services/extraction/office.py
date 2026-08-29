"""Extractors for Microsoft Office and OpenDocument formats.

Page-like provenance:
  - DOCX/ODT: no reliable page numbers (pagination is a renderer concern), so
    `page` stays None and headings are recorded as sections instead.
  - XLSX/XLS/ODS: one block per worksheet, `section` = sheet name.
  - PPTX/ODP: one block per slide, `page` = slide number (genuinely 1-based).
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from app.errors import CorruptDocumentError, ExtractionError
from app.models.document import ExtractedBlock, ExtractionResult
from app.services.extraction.base import BaseExtractor

logger = logging.getLogger(__name__)

MAX_SHEET_ROWS = 5_000


class DocxExtractor(BaseExtractor):
    """Word .docx — paragraphs, tables and heading structure."""

    name = "docx"
    extensions = frozenset({".docx"})
    mime_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    )

    def extract(self, path: Path) -> ExtractionResult:
        try:
            import docx  # type: ignore
            from docx.opc.exceptions import PackageNotFoundError  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ExtractionError(
                "DOCX support requires the python-docx package."
            ) from exc

        try:
            document = docx.Document(str(path))
        except PackageNotFoundError as exc:
            raise CorruptDocumentError(
                "This file is not a valid .docx document. Legacy .doc files must be "
                "converted to .docx first."
            ) from exc
        except (zipfile.BadZipFile, KeyError) as exc:
            raise CorruptDocumentError("The .docx file is corrupted.") from exc
        except Exception as exc:
            raise ExtractionError("The Word document could not be parsed.") from exc

        blocks: list[ExtractedBlock] = []
        current_section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            body = "\n".join(buffer).strip()
            if body:
                blocks.append(ExtractedBlock(text=body, section=current_section))
            buffer.clear()

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "").lower() if paragraph.style else ""
            if style.startswith("heading") or style == "title":
                flush()
                current_section = text
                buffer.append(text)
            else:
                buffer.append(text)
        flush()

        # Tables are appended after body text, labelled so the LLM knows the shape.
        for index, table in enumerate(document.tables, start=1):
            rows: list[str] = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(
                    ExtractedBlock(
                        text=f"[Table {index}]\n" + "\n".join(rows),
                        section=f"Table {index}",
                    )
                )

        return ExtractionResult(blocks=blocks, extractor=self.name)


class LegacyDocExtractor(BaseExtractor):
    """Legacy binary Word .doc.

    Pure-Python parsing of the OLE2 format is unreliable, so this attempts a
    best-effort strings extraction and otherwise fails with actionable guidance
    rather than silently indexing binary noise.
    """

    name = "doc"
    extensions = frozenset({".doc"})
    mime_types = frozenset({"application/msword"})

    def extract(self, path: Path) -> ExtractionResult:
        raw = path.read_bytes()

        # A .docx mislabelled as .doc is a common user error — handle it.
        if raw[:2] == b"PK":
            logger.info("File with .doc extension is actually a zip/docx; delegating")
            return DocxExtractor().extract(path)

        text = self._extract_ole_strings(raw)
        if len(text.strip()) < 100:
            raise ExtractionError(
                "Legacy .doc files are not fully supported. Please convert the file "
                "to .docx or PDF and upload it again."
            )
        result = ExtractionResult(
            blocks=[ExtractedBlock(text=text)], extractor=self.name
        )
        result.warnings.append(
            "Legacy .doc parsing is best-effort; formatting and some text may be "
            "missing. Convert to .docx for full fidelity."
        )
        return result

    @staticmethod
    def _extract_ole_strings(raw: bytes) -> str:
        """Pull readable runs of text out of the binary stream."""
        chunks: list[str] = []
        current = bytearray()
        for byte in raw:
            if 32 <= byte < 127 or byte in (9, 10, 13):
                current.append(byte)
            else:
                if len(current) >= 12:
                    chunks.append(current.decode("ascii", errors="ignore"))
                current.clear()
        if len(current) >= 12:
            chunks.append(current.decode("ascii", errors="ignore"))

        # Drop OLE structural tokens that are not document prose.
        noise = (
            "Microsoft Word",
            "MSWordDoc",
            "Word.Document",
            "Root Entry",
            "WordDocument",
            "SummaryInformation",
            "DocumentSummaryInformation",
            "CompObj",
            "ObjectPool",
            "Times New Roman",
        )
        kept = [c.strip() for c in chunks if not any(n in c for n in noise)]
        return "\n".join(k for k in kept if len(k) >= 12)


class ExcelExtractor(BaseExtractor):
    """Modern Excel .xlsx/.xlsm — one block per worksheet."""

    name = "xlsx"
    extensions = frozenset({".xlsx", ".xlsm"})
    mime_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    )

    def extract(self, path: Path) -> ExtractionResult:
        try:
            import openpyxl  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ExtractionError(
                "XLSX support requires the openpyxl package."
            ) from exc

        try:
            workbook = openpyxl.load_workbook(
                str(path), read_only=True, data_only=True
            )
        except (zipfile.BadZipFile, KeyError) as exc:
            raise CorruptDocumentError("The Excel file is corrupted.") from exc
        except Exception as exc:
            raise ExtractionError("The Excel workbook could not be opened.") from exc

        blocks: list[ExtractedBlock] = []
        warnings: list[str] = []
        try:
            for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
                lines = self._render_sheet(sheet, warnings)
                if lines:
                    blocks.append(
                        ExtractedBlock(
                            text=f"[Sheet: {sheet.title}]\n" + "\n".join(lines),
                            page=sheet_index,
                            section=sheet.title,
                        )
                    )
        finally:
            workbook.close()

        return ExtractionResult(
            blocks=blocks,
            page_count=len(blocks) or None,
            extractor=self.name,
            warnings=warnings,
        )

    @staticmethod
    def _render_sheet(sheet, warnings: list[str]) -> list[str]:
        rows_iter = sheet.iter_rows(values_only=True)
        try:
            first_row = next(rows_iter)
        except StopIteration:
            return []

        header = [str(c).strip() if c is not None else "" for c in first_row]
        has_header = bool(header) and sum(1 for h in header if h) >= max(
            1, len(header) // 2
        )

        lines: list[str] = []
        if has_header:
            lines.append("Columns: " + " | ".join(h for h in header if h))
        else:
            values = [str(c) for c in first_row if c is not None]
            if values:
                lines.append("Row 1 — " + "; ".join(values))

        for row_no, row in enumerate(rows_iter, start=2):
            if row_no > MAX_SHEET_ROWS:
                warnings.append(
                    f"Sheet '{sheet.title}' exceeded {MAX_SHEET_ROWS} rows; "
                    "later rows were not indexed."
                )
                break
            if not any(cell is not None and str(cell).strip() for cell in row):
                continue
            if has_header:
                pairs = [
                    f"{header[i] if i < len(header) and header[i] else f'col{i + 1}'}: {cell}"
                    for i, cell in enumerate(row)
                    if cell is not None and str(cell).strip()
                ]
                lines.append(f"Row {row_no} — " + "; ".join(pairs))
            else:
                lines.append(
                    f"Row {row_no} — "
                    + "; ".join(str(c) for c in row if c is not None and str(c).strip())
                )
        return lines


class LegacyExcelExtractor(BaseExtractor):
    """Legacy Excel .xls via xlrd."""

    name = "xls"
    extensions = frozenset({".xls"})
    mime_types = frozenset({"application/vnd.ms-excel"})

    def extract(self, path: Path) -> ExtractionResult:
        if path.read_bytes()[:2] == b"PK":
            logger.info("File with .xls extension is actually xlsx; delegating")
            return ExcelExtractor().extract(path)

        try:
            import xlrd  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ExtractionError("XLS support requires the xlrd package.") from exc

        try:
            book = xlrd.open_workbook(str(path))
        except Exception as exc:
            raise CorruptDocumentError(
                "The .xls file could not be read. Convert it to .xlsx and try again."
            ) from exc

        blocks: list[ExtractedBlock] = []
        for sheet_index in range(book.nsheets):
            sheet = book.sheet_by_index(sheet_index)
            lines: list[str] = []
            for row_no in range(min(sheet.nrows, MAX_SHEET_ROWS)):
                values = [
                    str(sheet.cell_value(row_no, col)).strip()
                    for col in range(sheet.ncols)
                ]
                if any(values):
                    lines.append(" | ".join(values))
            if lines:
                blocks.append(
                    ExtractedBlock(
                        text=f"[Sheet: {sheet.name}]\n" + "\n".join(lines),
                        page=sheet_index + 1,
                        section=sheet.name,
                    )
                )
        return ExtractionResult(
            blocks=blocks, page_count=len(blocks) or None, extractor=self.name
        )


class PowerPointExtractor(BaseExtractor):
    """PowerPoint .pptx — one block per slide, including notes."""

    name = "pptx"
    extensions = frozenset({".pptx", ".pptm"})
    mime_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    )

    def extract(self, path: Path) -> ExtractionResult:
        try:
            from pptx import Presentation  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ExtractionError(
                "PPTX support requires the python-pptx package."
            ) from exc

        try:
            presentation = Presentation(str(path))
        except (zipfile.BadZipFile, KeyError) as exc:
            raise CorruptDocumentError("The PowerPoint file is corrupted.") from exc
        except Exception as exc:
            raise ExtractionError(
                "The presentation could not be parsed. Legacy .ppt files must be "
                "converted to .pptx."
            ) from exc

        blocks: list[ExtractedBlock] = []
        for slide_index, slide in enumerate(presentation.slides, start=1):
            parts: list[str] = []
            title: str | None = None

            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = "\n".join(
                        p.text.strip()
                        for p in shape.text_frame.paragraphs
                        if p.text.strip()
                    )
                    if text:
                        if title is None and shape == getattr(
                            slide.shapes, "title", None
                        ):
                            title = text.splitlines()[0]
                        parts.append(text)
                if getattr(shape, "has_table", False):
                    rows = [
                        " | ".join(cell.text.strip() for cell in row.cells)
                        for row in shape.table.rows
                    ]
                    rows = [r for r in rows if r.replace("|", "").strip()]
                    if rows:
                        parts.append("[Table]\n" + "\n".join(rows))

            try:
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        parts.append(f"[Speaker notes] {notes}")
            except Exception:
                pass

            if parts:
                blocks.append(
                    ExtractedBlock(
                        text=f"[Slide {slide_index}]\n" + "\n".join(parts),
                        page=slide_index,
                        section=title or f"Slide {slide_index}",
                    )
                )

        return ExtractionResult(
            blocks=blocks,
            page_count=len(presentation.slides) or None,
            extractor=self.name,
        )


class LegacyPowerPointExtractor(BaseExtractor):
    """Legacy binary .ppt — not parseable reliably; fail with clear guidance."""

    name = "ppt"
    extensions = frozenset({".ppt"})
    mime_types = frozenset({"application/vnd.ms-powerpoint"})

    def extract(self, path: Path) -> ExtractionResult:
        if path.read_bytes()[:2] == b"PK":
            return PowerPointExtractor().extract(path)
        raise ExtractionError(
            "Legacy .ppt files are not supported. Please convert the presentation "
            "to .pptx or PDF and upload it again."
        )


class OpenDocumentExtractor(BaseExtractor):
    """OpenDocument text/spreadsheet/presentation via raw XML parsing.

    Reads `content.xml` from the ODF zip container directly, avoiding an extra
    dependency.
    """

    name = "opendocument"
    extensions = frozenset({".odt", ".ods", ".odp"})
    mime_types = frozenset(
        {
            "application/vnd.oasis.opendocument.text",
            "application/vnd.oasis.opendocument.spreadsheet",
            "application/vnd.oasis.opendocument.presentation",
        }
    )

    _TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    _TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"

    def extract(self, path: Path) -> ExtractionResult:
        from xml.etree import ElementTree

        try:
            with zipfile.ZipFile(str(path)) as archive:
                content = archive.read("content.xml")
        except (zipfile.BadZipFile, KeyError) as exc:
            raise CorruptDocumentError(
                "The OpenDocument file is corrupted or missing content.xml."
            ) from exc

        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError as exc:
            raise CorruptDocumentError(
                "The OpenDocument content is malformed."
            ) from exc

        lines: list[str] = []
        for element in root.iter():
            tag = element.tag.split("}")[-1]
            namespace = element.tag.split("}")[0].lstrip("{")
            if namespace == self._TEXT_NS and tag in {"p", "h"}:
                text = "".join(element.itertext()).strip()
                if text:
                    lines.append(text)
            elif namespace == self._TABLE_NS and tag == "table-row":
                cells = [
                    "".join(cell.itertext()).strip()
                    for cell in element
                    if cell.tag.split("}")[-1] == "table-cell"
                ]
                if any(cells):
                    lines.append(" | ".join(cells))

        if not lines:
            return ExtractionResult(blocks=[], extractor=self.name)
        return ExtractionResult(
            blocks=[ExtractedBlock(text="\n".join(lines))], extractor=self.name
        )


class EpubExtractor(BaseExtractor):
    """EPUB e-books — reads the XHTML documents inside the container."""

    name = "epub"
    extensions = frozenset({".epub"})
    mime_types = frozenset({"application/epub+zip"})

    def extract(self, path: Path) -> ExtractionResult:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ExtractionError("EPUB support requires beautifulsoup4.") from exc

        try:
            archive = zipfile.ZipFile(str(path))
        except zipfile.BadZipFile as exc:
            raise CorruptDocumentError("The EPUB file is corrupted.") from exc

        blocks: list[ExtractedBlock] = []
        with archive:
            names = sorted(
                n
                for n in archive.namelist()
                if n.lower().endswith((".xhtml", ".html", ".htm"))
            )
            for index, name in enumerate(names, start=1):
                try:
                    soup = BeautifulSoup(archive.read(name), "html.parser")
                except Exception:
                    continue
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text("\n", strip=True)
                if text:
                    heading = soup.find(["h1", "h2", "h3"])
                    blocks.append(
                        ExtractedBlock(
                            text=text,
                            page=index,
                            section=heading.get_text(strip=True) if heading else name,
                        )
                    )

        return ExtractionResult(
            blocks=blocks, page_count=len(blocks) or None, extractor=self.name
        )
