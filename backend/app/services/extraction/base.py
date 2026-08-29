"""Extractor contract and registry.

Adding a new file format means writing one `BaseExtractor` subclass and
registering it — no changes to the upload flow, chunker or API layer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from app.models.document import ExtractionResult

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Turns one family of file formats into ordered text blocks."""

    #: Lower-case extensions including the leading dot, e.g. {".pdf"}
    extensions: frozenset[str] = frozenset()
    #: Accepted MIME types (used as a secondary signal, never the only one)
    mime_types: frozenset[str] = frozenset()
    #: Human-readable name recorded on the document record
    name: str = "base"

    @abstractmethod
    def extract(self, path: Path) -> ExtractionResult:
        """Extract text blocks from `path`.

        Raises:
            CorruptDocumentError: file is unreadable, encrypted or malformed.
            ExtractionError: any other parse failure.
        """

    def supports(self, extension: str) -> bool:
        return extension.lower() in self.extensions

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} extensions={sorted(self.extensions)}>"


class ExtractorRegistry:
    """Maps file extensions to extractor instances."""

    def __init__(self) -> None:
        self._by_extension: dict[str, BaseExtractor] = {}
        self._extractors: list[BaseExtractor] = []

    def register(self, extractor: BaseExtractor) -> None:
        self._extractors.append(extractor)
        for extension in extractor.extensions:
            key = extension.lower()
            if key in self._by_extension:
                logger.warning(
                    "Extension %s already handled by %s; %s will not override it",
                    key,
                    type(self._by_extension[key]).__name__,
                    type(extractor).__name__,
                )
                continue
            self._by_extension[key] = extractor

    def get(self, extension: str) -> BaseExtractor | None:
        return self._by_extension.get(extension.lower())

    @property
    def supported_extensions(self) -> list[str]:
        return sorted(self._by_extension)

    @property
    def supported_mime_types(self) -> set[str]:
        mimes: set[str] = set()
        for extractor in self._extractors:
            mimes |= extractor.mime_types
        return mimes

    def __len__(self) -> int:
        return len(self._by_extension)
