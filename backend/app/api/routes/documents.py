"""Document endpoints: upload, list, inspect, summarize, search, delete."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from app.api.deps import (
    ServiceContainer,
    container_dep,
    get_processor,
    get_retriever,
    get_summarizer,
    require_api_key,
)
from app.errors import (
    AppError,
    DocumentNotFoundError,
    EmptyDocumentError,
    FileTooLargeError,
    ValidationError,
)
from app.models.document import DocumentRecord, DocumentStatus
from app.schemas.api import (
    ChunkInfo,
    DeleteResponse,
    DocumentListResponse,
    DocumentSummaryInfo,
    SourcesResponse,
    SummarizeRequest,
    SummarizeResponse,
    UploadResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.retriever import Retriever
from app.services.summarizer import Summarizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

#: Read the upload in bounded slices so an oversized file is rejected before it
#: is fully buffered in memory.
_READ_CHUNK = 1024 * 1024


def _to_info(record: DocumentRecord) -> DocumentSummaryInfo:
    return DocumentSummaryInfo(**record.to_dict())


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    """Stream the upload into memory, aborting once the size limit is passed."""
    buffer = bytearray()
    while True:
        piece = await upload.read(_READ_CHUNK)
        if not piece:
            break
        buffer.extend(piece)
        if len(buffer) > limit:
            raise FileTooLargeError(
                f"The file exceeds the {limit // 1_048_576} MB upload limit."
            )
    return bytes(buffer)


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document, then summarize it",
    dependencies=[Depends(require_api_key)],
)
async def upload_document(
    file: UploadFile = File(..., description="The document to index."),
    container: ServiceContainer = Depends(container_dep),
) -> UploadResponse:
    """Run the full ingest pipeline, then generate the summary automatically.

    Indexing and summarization are reported independently: a document that
    indexes successfully stays usable for chat even if the LLM call fails (for
    example when GROQ_API_KEY is absent).
    """
    if not file.filename:
        raise ValidationError("No filename was provided with the upload.")

    content = await _read_upload(file, container.settings.max_upload_bytes)
    if not content:
        raise EmptyDocumentError("The uploaded file is empty.")

    record = container.processor.process_upload(file.filename, content)

    summary: str | None = None
    summary_error: str | None = None
    try:
        chunks = container.vector_store.get_document_chunks(record.document_id)
        result = container.summarizer.summarize(chunks, record.filename)
        summary = result.summary
        record = container.processor.set_summary(record.document_id, summary)
    except AppError as exc:
        summary_error = exc.message
        logger.warning(
            "Indexed %s but summarization failed: %s", record.document_id, exc.code
        )
    except Exception:
        summary_error = "The summary could not be generated."
        logger.exception("Unexpected summarization failure")

    message = (
        f"'{record.filename}' was indexed into {record.chunk_count} searchable chunks."
    )
    if summary_error:
        message += " The automatic summary is unavailable; chat is still available."

    return UploadResponse(
        document=_to_info(record),
        summary=summary,
        summary_error=summary_error,
        message=message,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List uploaded documents",
    dependencies=[Depends(require_api_key)],
)
async def list_documents(
    processor: DocumentProcessor = Depends(get_processor),
) -> DocumentListResponse:
    records = processor.registry.list_all()
    return DocumentListResponse(
        documents=[_to_info(record) for record in records], total=len(records)
    )


@router.get(
    "/{document_id}",
    response_model=DocumentSummaryInfo,
    summary="Get one document's metadata and summary",
    dependencies=[Depends(require_api_key)],
)
async def get_document(
    document_id: str,
    processor: DocumentProcessor = Depends(get_processor),
) -> DocumentSummaryInfo:
    return _to_info(processor.registry.require(document_id))


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document and its embeddings",
    dependencies=[Depends(require_api_key)],
)
async def delete_document(
    document_id: str,
    processor: DocumentProcessor = Depends(get_processor),
) -> DeleteResponse:
    record = processor.registry.require(document_id)
    processor.delete_document(document_id)
    return DeleteResponse(
        document_id=document_id,
        deleted=True,
        message=f"'{record.filename}' and its embeddings were deleted.",
    )


@router.post(
    "/{document_id}/summarize",
    response_model=SummarizeResponse,
    summary="Generate (or regenerate) a document summary",
    dependencies=[Depends(require_api_key)],
)
async def summarize_document(
    document_id: str,
    payload: SummarizeRequest | None = None,
    container: ServiceContainer = Depends(container_dep),
) -> SummarizeResponse:
    """Return the cached summary, or regenerate it when `force` is true."""
    request = payload or SummarizeRequest()
    record = container.processor.registry.require(document_id)

    if record.summary and not request.force:
        return SummarizeResponse(
            document_id=document_id,
            summary=record.summary,
            cached=True,
            strategy="direct",
        )

    chunks = container.vector_store.get_document_chunks(document_id)
    if not chunks:
        raise EmptyDocumentError(
            "This document has no indexed content to summarize. Re-upload it."
        )

    result = container.summarizer.summarize(chunks, record.filename)
    container.processor.set_summary(document_id, result.summary)

    return SummarizeResponse(
        document_id=document_id,
        summary=result.summary,
        cached=False,
        strategy=result.strategy,
        map_calls=result.map_calls,
    )


@router.get(
    "/{document_id}/sources",
    response_model=SourcesResponse,
    summary="List a document's chunks, or search within it",
    dependencies=[Depends(require_api_key)],
)
async def get_document_sources(
    document_id: str,
    q: str | None = Query(
        default=None,
        description="Optional semantic search query. Omit to list chunks in order.",
        max_length=1000,
    ),
    limit: int = Query(default=50, ge=1, le=500),
    processor: DocumentProcessor = Depends(get_processor),
    retriever: Retriever = Depends(get_retriever),
) -> SourcesResponse:
    """Expose the exact indexed chunks so any citation can be verified."""
    record = processor.registry.require(document_id)

    if q and q.strip():
        hits = retriever.search_in_document(document_id, q.strip(), top_k=limit)
        chunks = [
            ChunkInfo(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                chunk_index=hit.chunk_index,
                page=hit.page,
                section=hit.section,
                char_count=len(hit.text),
                text=hit.text,
            )
            for hit in hits
        ]
    else:
        stored = processor.get_chunks(document_id)[:limit]
        chunks = [
            ChunkInfo(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                page=chunk.page,
                section=chunk.section,
                char_count=chunk.char_count,
                text=chunk.text,
            )
            for chunk in stored
        ]

    return SourcesResponse(
        document_id=document_id,
        filename=record.filename,
        total=len(chunks),
        chunks=chunks,
    )
