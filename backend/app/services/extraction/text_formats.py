"""Extractors for plain-text and markup formats."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from app.errors import CorruptDocumentError, ExtractionError
from app.models.document import ExtractedBlock, ExtractionResult
from app.services.extraction.base import BaseExtractor

logger = logging.getLogger(__name__)

# Guard against pathological structured files.
MAX_CSV_ROWS = 20_000
MAX_JSON_NODES = 60_000


def read_text_file(path: Path) -> str:
    """Decode a text file, falling back through common encodings."""
    raw = path.read_bytes()
    if not raw:
        return ""

    # Try a detector first when available.
    try:
        import chardet  # type: ignore

        guess = chardet.detect(raw[:200_000])
        encoding = guess.get("encoding")
        confidence = guess.get("confidence") or 0.0
        if encoding and confidence >= 0.6:
            try:
                return raw.decode(encoding, errors="replace")
            except (LookupError, UnicodeDecodeError):
                pass
    except ImportError:
        pass

    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


class PlainTextExtractor(BaseExtractor):
    """Plain text, logs, source code and config files."""

    name = "plaintext"
    extensions = frozenset(
        {
            ".txt",
            ".text",
            ".log",
            ".rst",
            ".tex",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
            ".conf",
            ".toml",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".sql",
            ".sh",
        }
    )
    mime_types = frozenset({"text/plain", "text/x-log", "application/x-yaml"})

    def extract(self, path: Path) -> ExtractionResult:
        text = read_text_file(path)
        if not text.strip():
            return ExtractionResult(blocks=[], extractor=self.name)
        return ExtractionResult(
            blocks=[ExtractedBlock(text=text)],
            extractor=self.name,
        )


class MarkdownExtractor(BaseExtractor):
    """Markdown — split on headings so each block keeps its section name."""

    name = "markdown"
    extensions = frozenset({".md", ".markdown", ".mdown", ".mkd"})
    mime_types = frozenset({"text/markdown", "text/x-markdown"})

    def extract(self, path: Path) -> ExtractionResult:
        text = read_text_file(path)
        if not text.strip():
            return ExtractionResult(blocks=[], extractor=self.name)

        blocks: list[ExtractedBlock] = []
        current_section = "Introduction"
        buffer: list[str] = []

        def flush() -> None:
            body = "\n".join(buffer).strip()
            if body:
                blocks.append(ExtractedBlock(text=body, section=current_section))
            buffer.clear()

        in_fence = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
            # Only treat '#' as a heading outside fenced code blocks.
            if not in_fence and stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
                if heading:
                    flush()
                    current_section = heading
                    buffer.append(line)
                    continue
            buffer.append(line)
        flush()

        if not blocks:
            blocks = [ExtractedBlock(text=text)]
        return ExtractionResult(blocks=blocks, extractor=self.name)


class HTMLExtractor(BaseExtractor):
    """HTML/XHTML — strips scripts and styles, keeps heading structure."""

    name = "html"
    extensions = frozenset({".html", ".htm", ".xhtml"})
    mime_types = frozenset({"text/html", "application/xhtml+xml"})

    def extract(self, path: Path) -> ExtractionResult:
        raw = read_text_file(path)
        if not raw.strip():
            return ExtractionResult(blocks=[], extractor=self.name)

        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ExtractionError(
                "HTML support requires beautifulsoup4. Install backend requirements."
            ) from exc

        try:
            soup = BeautifulSoup(raw, "html.parser")
        except Exception as exc:
            raise CorruptDocumentError("The HTML file could not be parsed.") from exc

        for tag in soup(["script", "style", "noscript", "template", "svg"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else None
        blocks: list[ExtractedBlock] = []
        current_section = title or "Document"
        buffer: list[str] = []

        def flush() -> None:
            body = "\n".join(buffer).strip()
            if body:
                blocks.append(ExtractedBlock(text=body, section=current_section))
            buffer.clear()

        body_root = soup.body or soup
        for element in body_root.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "pre", "blockquote"]
        ):
            content = element.get_text(" ", strip=True)
            if not content:
                continue
            if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                flush()
                current_section = content
                buffer.append(content)
            else:
                buffer.append(content)
        flush()

        if not blocks:
            fallback = body_root.get_text("\n", strip=True)
            if fallback:
                blocks = [ExtractedBlock(text=fallback, section=title)]

        result = ExtractionResult(blocks=blocks, extractor=self.name)
        if title:
            result.warnings.append(f"Document title: {title}")
        return result


class XMLExtractor(BaseExtractor):
    """Generic XML — flattens the tree into readable `path: value` lines."""

    name = "xml"
    extensions = frozenset({".xml", ".rss", ".atom", ".svg"})
    mime_types = frozenset({"application/xml", "text/xml"})

    def extract(self, path: Path) -> ExtractionResult:
        raw = read_text_file(path)
        if not raw.strip():
            return ExtractionResult(blocks=[], extractor=self.name)

        try:
            root = ElementTree.fromstring(raw)
        except ElementTree.ParseError as exc:
            raise CorruptDocumentError(
                f"The XML file is malformed (line {exc.position[0]})."
            ) from exc

        lines: list[str] = []

        def walk(node: ElementTree.Element, trail: str) -> None:
            if len(lines) > MAX_JSON_NODES:
                return
            tag = node.tag.split("}")[-1]  # drop namespace
            here = f"{trail}/{tag}" if trail else tag
            for attr, value in node.attrib.items():
                lines.append(f"{here}@{attr.split('}')[-1]}: {value}")
            text = (node.text or "").strip()
            if text:
                lines.append(f"{here}: {text}")
            for child in node:
                walk(child, here)

        walk(root, "")
        if not lines:
            return ExtractionResult(blocks=[], extractor=self.name)
        return ExtractionResult(
            blocks=[ExtractedBlock(text="\n".join(lines))],
            extractor=self.name,
        )


class JSONExtractor(BaseExtractor):
    """JSON / JSONL — flattened to `key.path: value` lines for embedding."""

    name = "json"
    extensions = frozenset({".json", ".jsonl", ".ndjson", ".geojson"})
    mime_types = frozenset({"application/json", "application/x-ndjson"})

    def extract(self, path: Path) -> ExtractionResult:
        raw = read_text_file(path)
        if not raw.strip():
            return ExtractionResult(blocks=[], extractor=self.name)

        documents: list[Any] = []
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            for line_no, line in enumerate(raw.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    documents.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise CorruptDocumentError(
                        f"Invalid JSON on line {line_no}: {exc.msg}."
                    ) from exc
        else:
            try:
                documents.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise CorruptDocumentError(
                    f"The JSON file is malformed at line {exc.lineno}, column {exc.colno}."
                ) from exc

        lines: list[str] = []
        truncated = False

        def flatten(node: Any, trail: str) -> None:
            nonlocal truncated
            if len(lines) >= MAX_JSON_NODES:
                truncated = True
                return
            if isinstance(node, dict):
                for key, value in node.items():
                    flatten(value, f"{trail}.{key}" if trail else str(key))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    flatten(value, f"{trail}[{index}]")
            else:
                rendered = "null" if node is None else str(node)
                lines.append(f"{trail}: {rendered}" if trail else rendered)

        for index, document in enumerate(documents):
            flatten(document, f"record[{index}]" if len(documents) > 1 else "")

        if not lines:
            return ExtractionResult(blocks=[], extractor=self.name)

        result = ExtractionResult(
            blocks=[ExtractedBlock(text="\n".join(lines))],
            extractor=self.name,
        )
        if truncated:
            result.warnings.append(
                f"JSON was very large; only the first {MAX_JSON_NODES} values were indexed."
            )
        return result


class CSVExtractor(BaseExtractor):
    """CSV/TSV — each row becomes a `Column: value` record so retrieval keeps
    the header context that a bare comma-separated line would lose."""

    name = "csv"
    extensions = frozenset({".csv", ".tsv", ".psv"})
    mime_types = frozenset({"text/csv", "text/tab-separated-values"})

    def extract(self, path: Path) -> ExtractionResult:
        raw = read_text_file(path)
        if not raw.strip():
            return ExtractionResult(blocks=[], extractor=self.name)

        delimiter = self._sniff_delimiter(raw, path.suffix.lower())
        reader = csv.reader(io.StringIO(raw), delimiter=delimiter)

        try:
            rows = []
            for index, row in enumerate(reader):
                if index > MAX_CSV_ROWS:
                    break
                rows.append(row)
        except csv.Error as exc:
            raise CorruptDocumentError(f"The CSV file is malformed: {exc}") from exc

        if not rows:
            return ExtractionResult(blocks=[], extractor=self.name)

        header = [cell.strip() for cell in rows[0]]
        has_header = self._looks_like_header(header, rows[1:2])
        data_rows = rows[1:] if has_header else rows

        lines: list[str] = []
        if has_header:
            lines.append("Columns: " + " | ".join(header))

        for row_no, row in enumerate(data_rows, start=1):
            if not any(cell.strip() for cell in row):
                continue
            if has_header:
                pairs = [
                    f"{header[i] if i < len(header) else f'col{i + 1}'}: {cell.strip()}"
                    for i, cell in enumerate(row)
                    if cell.strip()
                ]
                lines.append(f"Row {row_no} — " + "; ".join(pairs))
            else:
                lines.append(
                    f"Row {row_no} — " + "; ".join(c.strip() for c in row if c.strip())
                )

        result = ExtractionResult(
            blocks=[ExtractedBlock(text="\n".join(lines))],
            extractor=self.name,
        )
        result.warnings.append(f"Parsed {len(data_rows)} data rows.")
        if len(rows) > MAX_CSV_ROWS:
            result.warnings.append(
                f"File exceeded {MAX_CSV_ROWS} rows; later rows were not indexed."
            )
        return result

    @staticmethod
    def _sniff_delimiter(raw: str, suffix: str) -> str:
        if suffix == ".tsv":
            return "\t"
        if suffix == ".psv":
            return "|"
        sample = raw[:8192]
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            return ","

    @staticmethod
    def _looks_like_header(header: list[str], following: list[list[str]]) -> bool:
        """Heuristic: headers are non-empty, non-numeric labels."""
        if not header or not all(cell.strip() for cell in header):
            return False
        numeric = sum(1 for cell in header if cell.replace(".", "", 1).lstrip("-").isdigit())
        if numeric > len(header) / 2:
            return False
        if following and following[0] and len(following[0]) != len(header):
            return False
        return True


class RTFExtractor(BaseExtractor):
    """Rich Text Format."""

    name = "rtf"
    extensions = frozenset({".rtf"})
    mime_types = frozenset({"application/rtf", "text/rtf"})

    def extract(self, path: Path) -> ExtractionResult:
        raw = read_text_file(path)
        if not raw.strip():
            return ExtractionResult(blocks=[], extractor=self.name)
        try:
            from striprtf.striprtf import rtf_to_text  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ExtractionError(
                "RTF support requires the striprtf package."
            ) from exc
        try:
            text = rtf_to_text(raw, errors="ignore")
        except Exception as exc:
            raise CorruptDocumentError("The RTF file could not be parsed.") from exc
        if not text.strip():
            return ExtractionResult(blocks=[], extractor=self.name)
        return ExtractionResult(
            blocks=[ExtractedBlock(text=text)], extractor=self.name
        )
