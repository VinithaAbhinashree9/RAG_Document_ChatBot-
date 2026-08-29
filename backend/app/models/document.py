"""Internal domain models (independent of transport/API schemas)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str = "doc") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class DocumentStatus(str, Enum):
    """Lifecycle of an uploaded document."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    SUMMARIZING = "summarizing"
    READY = "ready"
    FAILED = "failed"


@dataclass(slots=True)
class ExtractedBlock:
    """A contiguous piece of text extracted from a source document.

    `page` is 1-based and only set when the format has real pages/sheets/slides,
    so page numbers are never invented downstream.
    """

    text: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionResult:
    blocks: list[ExtractedBlock]
    page_count: int | None = None
    extractor: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text.strip())

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)


@dataclass(slots=True)
class Chunk:
    """An embeddable unit of text plus provenance metadata."""

    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    text: str
    chunk_index: int
    page: int | None = None
    section: str | None = None
    char_count: int = 0

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata dict (vector stores only accept scalar values)."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            # -1 is a sentinel for "this format has no pages"
            "page": self.page if self.page is not None else -1,
            "section": self.section or "",
            "char_count": self.char_count,
        }


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by semantic search, with its similarity score."""

    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    text: str
    chunk_index: int
    score: float
    page: int | None = None
    section: str | None = None

    @property
    def citation_label(self) -> str:
        if self.page is not None:
            return f"{self.filename} — Page {self.page}"
        if self.section:
            return f"{self.filename} — {self.section}"
        return f"{self.filename} — Chunk {self.chunk_index + 1}"


@dataclass(slots=True)
class DocumentRecord:
    """Everything the app knows about one uploaded document."""

    document_id: str
    filename: str
    stored_filename: str
    file_type: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.UPLOADED
    page_count: int | None = None
    char_count: int = 0
    chunk_count: int = 0
    summary: str | None = None
    error: str | None = None
    extractor: str | None = None
    warnings: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "status": self.status.value,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "chunk_count": self.chunk_count,
            "summary": self.summary,
            "error": self.error,
            "extractor": self.extractor,
            "warnings": list(self.warnings),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(slots=True)
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }
