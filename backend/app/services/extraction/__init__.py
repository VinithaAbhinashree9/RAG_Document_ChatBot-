"""Document extraction subsystem.

To add a format: subclass `BaseExtractor`, then add it to `_EXTRACTORS` below.
Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from app.services.extraction.base import BaseExtractor, ExtractorRegistry
from app.services.extraction.office import (
    DocxExtractor,
    EpubExtractor,
    ExcelExtractor,
    LegacyDocExtractor,
    LegacyExcelExtractor,
    LegacyPowerPointExtractor,
    OpenDocumentExtractor,
    PowerPointExtractor,
)
from app.services.extraction.pdf import PDFExtractor
from app.services.extraction.text_formats import (
    CSVExtractor,
    HTMLExtractor,
    JSONExtractor,
    MarkdownExtractor,
    PlainTextExtractor,
    RTFExtractor,
    XMLExtractor,
)

_EXTRACTORS: tuple[BaseExtractor, ...] = (
    PDFExtractor(),
    DocxExtractor(),
    LegacyDocExtractor(),
    ExcelExtractor(),
    LegacyExcelExtractor(),
    PowerPointExtractor(),
    LegacyPowerPointExtractor(),
    OpenDocumentExtractor(),
    EpubExtractor(),
    MarkdownExtractor(),
    HTMLExtractor(),
    XMLExtractor(),
    JSONExtractor(),
    CSVExtractor(),
    RTFExtractor(),
    PlainTextExtractor(),
)


def build_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    for extractor in _EXTRACTORS:
        registry.register(extractor)
    return registry


#: Process-wide registry (extractors are stateless).
registry: ExtractorRegistry = build_registry()

SUPPORTED_EXTENSIONS: list[str] = registry.supported_extensions

__all__ = [
    "BaseExtractor",
    "ExtractorRegistry",
    "SUPPORTED_EXTENSIONS",
    "build_registry",
    "registry",
]
