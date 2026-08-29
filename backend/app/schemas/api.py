"""Pydantic request/response schemas (the public API contract)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------
class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class SourceCitation(BaseModel):
    """A verifiable pointer back into an uploaded document."""

    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    page: int | None = Field(
        default=None, description="1-based page/sheet/slide number, when the format has one."
    )
    section: str | None = None
    score: float = Field(description="Cosine similarity of the chunk to the query.")
    excerpt: str = Field(description="Short verbatim snippet from the document.")
    label: str = Field(description="Human-readable citation, e.g. 'report.pdf — Page 7'.")


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
class DocumentSummaryInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size_bytes: int
    status: str
    page_count: int | None = None
    char_count: int = 0
    chunk_count: int = 0
    summary: str | None = None
    error: str | None = None
    extractor: str | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class UploadResponse(BaseModel):
    document: DocumentSummaryInfo
    summary: str | None = None
    summary_error: str | None = Field(
        default=None,
        description="Set when indexing succeeded but summarization failed (e.g. no API key).",
    )
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummaryInfo]
    total: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str


class SummarizeRequest(BaseModel):
    force: bool = Field(default=False, description="Bypass the cached summary and regenerate.")


class SummarizeResponse(BaseModel):
    document_id: str
    summary: str
    cached: bool
    strategy: Literal["direct", "map_reduce"]
    map_calls: int = 0


class ChunkInfo(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    page: int | None = None
    section: str | None = None
    char_count: int
    text: str


class SourcesResponse(BaseModel):
    """Chunk listing / in-document search results."""

    document_id: str
    filename: str
    total: int
    chunks: list[ChunkInfo]


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(
        default_factory=list,
        description="Documents to scope retrieval to. Empty = all ready documents.",
    )
    session_id: str = Field(default="default", max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=30)
    stream: bool = False

    @field_validator("message")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be empty")
        return cleaned

    @field_validator("document_ids")
    @classmethod
    def _dedupe_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    session_id: str
    document_ids: list[str] = Field(default_factory=list)
    grounded: bool = Field(
        description="False when no relevant context was found and the model abstained."
    )
    retrieved_chunks: int = 0
    model: str


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]


class ClearChatRequest(BaseModel):
    session_id: str = Field(default="default", max_length=64)


class ClearChatResponse(BaseModel):
    session_id: str
    cleared: bool
    message: str


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    app_env: str
    llm_configured: bool
    llm_model: str
    embedding_model: str
    vector_store: str
    documents_indexed: int
    supported_extensions: list[str]
