"""PDF extraction with real, per-page provenance.

Strategy: try `pypdf` first (fast). If a page yields little text, retry that page
with `pdfplumber`, which handles column layouts and tables better. Page numbers
come from the file itself and are never synthesised.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.errors import CorruptDocumentError, ExtractionError
from app.models.document import ExtractedBlock, ExtractionResult
from app.services.extraction.base import BaseExtractor

logger = logging.getLogger(__name__)

# Below this many characters a page is considered "thin" and worth re-parsing.
THIN_PAGE_CHARS = 40


class PDFExtractor(BaseExtractor):
    name = "pdf"
    extensions = frozenset({".pdf"})
    mime_types = frozenset({"application/pdf"})

    def extract(self, path: Path) -> ExtractionResult:
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ExtractionError("PDF support requires the pypdf package.") from exc

        try:
            reader = PdfReader(str(path))
        except PdfReadError as exc:
            raise CorruptDocumentError(
                "The PDF file is corrupted or not a valid PDF."
            ) from exc
        except Exception as exc:
            raise CorruptDocumentError("The PDF file could not be opened.") from exc

        if getattr(reader, "is_encrypted", False):
            # Many PDFs are encrypted with an empty owner password; try that.
            try:
                if reader.decrypt("") == 0:
                    raise CorruptDocumentError(
                        "This PDF is password protected. Upload an unlocked copy."
                    )
            except CorruptDocumentError:
                raise
            except Exception as exc:
                raise CorruptDocumentError(
                    "This PDF is password protected and could not be opened."
                ) from exc

        page_count = len(reader.pages)
        if page_count == 0:
            raise CorruptDocumentError("The PDF contains no pages.")

        blocks: list[ExtractedBlock] = []
        warnings: list[str] = []
        thin_pages: list[int] = []

        for index, page in enumerate(reader.pages):
            page_number = index + 1
            try:
                text = page.extract_text() or ""
            except Exception:
                logger.warning("pypdf failed on page %s; will retry", page_number)
                text = ""
            if len(text.strip()) < THIN_PAGE_CHARS:
                thin_pages.append(page_number)
            if text.strip():
                blocks.append(ExtractedBlock(text=text, page=page_number))

        # Second pass with pdfplumber for pages that produced little text.
        if thin_pages:
            recovered = self._recover_pages(path, thin_pages)
            if recovered:
                existing = {b.page for b in blocks}
                for page_number, text in sorted(recovered.items()):
                    if page_number in existing:
                        # Replace the thin block with the richer text.
                        blocks = [b for b in blocks if b.page != page_number]
                    blocks.append(ExtractedBlock(text=text, page=page_number))
                blocks.sort(key=lambda b: (b.page or 0))

        if not blocks:
            warnings.append(
                "No selectable text found. This PDF is likely a scanned image and "
                "requires OCR."
            )

        extracted_pages = {b.page for b in blocks}
        missing = page_count - len(extracted_pages)
        if blocks and missing > 0:
            warnings.append(f"{missing} of {page_count} pages contained no extractable text.")

        return ExtractionResult(
            blocks=blocks,
            page_count=page_count,
            extractor=self.name,
            warnings=warnings,
        )

    @staticmethod
    def _recover_pages(path: Path, pages: list[int]) -> dict[int, str]:
        """Re-extract specific pages with pdfplumber, including table content."""
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            return {}

        recovered: dict[int, str] = {}
        wanted = set(pages)
        try:
            with pdfplumber.open(str(path)) as pdf:
                for index, page in enumerate(pdf.pages):
                    page_number = index + 1
                    if page_number not in wanted:
                        continue
                    parts: list[str] = []
                    try:
                        text = page.extract_text() or ""
                        if text.strip():
                            parts.append(text)
                    except Exception:
                        pass
                    try:
                        for table in page.extract_tables() or []:
                            rows = [
                                " | ".join((cell or "").strip() for cell in row)
                                for row in table
                                if any(cell for cell in row)
                            ]
                            if rows:
                                parts.append("[Table]\n" + "\n".join(rows))
                    except Exception:
                        pass
                    combined = "\n".join(parts).strip()
                    if combined:
                        recovered[page_number] = combined
        except Exception as exc:
            logger.warning("pdfplumber recovery pass failed: %s", type(exc).__name__)
            return recovered
        return recovered
