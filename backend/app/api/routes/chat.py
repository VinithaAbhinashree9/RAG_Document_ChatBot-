"""Chat endpoints: grounded Q&A, streaming, history and reset."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import ServiceContainer, container_dep, require_api_key
from app.errors import AppError, DocumentNotFoundError, ValidationError
from app.models.document import DocumentStatus
from app.schemas.api import (
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ClearChatRequest,
    ClearChatResponse,
    SourceCitation,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _resolve_scope(container: ServiceContainer, requested: list[str]) -> list[str]:
    """Validate requested document IDs, or fall back to every ready document."""
    if requested:
        for document_id in requested:
            record = container.processor.registry.get(document_id)
            if record is None:
                raise DocumentNotFoundError(f"No document with ID '{document_id}'.")
            if record.status == DocumentStatus.FAILED:
                raise ValidationError(
                    f"'{record.filename}' failed to process and cannot be queried."
                )
        return requested

    ready = container.processor.registry.list_ready()
    if not ready:
        raise ValidationError(
            "No documents have been uploaded yet. Upload a document before chatting."
        )
    return [record.document_id for record in ready]


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about one or more documents",
    dependencies=[Depends(require_api_key)],
)
async def chat(
    payload: ChatRequest,
    container: ServiceContainer = Depends(container_dep),
) -> ChatResponse:
    """Answer strictly from retrieved document context, with citations."""
    document_ids = _resolve_scope(container, payload.document_ids)

    result = container.chatbot.answer(
        payload.message,
        session_id=payload.session_id,
        document_ids=document_ids,
        top_k=payload.top_k,
    )

    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceCitation(**container.chatbot.citation_payload(chunk))
            for chunk in result.sources
        ],
        session_id=payload.session_id,
        document_ids=result.document_ids or document_ids,
        grounded=result.grounded,
        retrieved_chunks=result.retrieved_chunks,
        model=result.model,
    )


@router.post(
    "/chat/stream",
    summary="Ask a question and stream the answer as server-sent events",
    dependencies=[Depends(require_api_key)],
)
async def chat_stream(
    payload: ChatRequest,
    container: ServiceContainer = Depends(container_dep),
) -> StreamingResponse:
    """Stream tokens, then a `sources` event, then `done`.

    Scope resolution happens before the response starts so validation errors
    still return a normal JSON error rather than a half-open stream.
    """
    document_ids = _resolve_scope(container, payload.document_ids)

    def event_stream() -> Iterator[str]:
        try:
            for event_type, data in container.chatbot.stream_answer(
                payload.message,
                session_id=payload.session_id,
                document_ids=document_ids,
                top_k=payload.top_k,
            ):
                if event_type == "token":
                    yield f"data: {json.dumps({'type': 'token', 'content': data})}\n\n"
                elif event_type == "sources":
                    yield f"data: {json.dumps({'type': 'sources', 'sources': data})}\n\n"
                elif event_type == "done":
                    payload_out = {"type": "done", **(data if isinstance(data, dict) else {})}
                    yield f"data: {json.dumps(payload_out)}\n\n"
        except AppError as exc:
            logger.warning("Streaming chat failed: %s", exc.code)
            yield (
                "data: "
                + json.dumps(
                    {"type": "error", "code": exc.code, "message": exc.message}
                )
                + "\n\n"
            )
        except Exception:
            logger.exception("Unexpected streaming failure")
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "code": "internal_error",
                        "message": "An unexpected error occurred while answering.",
                    }
                )
                + "\n\n"
            )
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Prevent proxy buffering from defeating streaming.
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/chat/history",
    response_model=ChatHistoryResponse,
    summary="Get the transcript for a chat session",
    dependencies=[Depends(require_api_key)],
)
async def chat_history(
    session_id: str = "default",
    container: ServiceContainer = Depends(container_dep),
) -> ChatHistoryResponse:
    turns = container.memory.get_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[ChatMessage(**turn.to_dict()) for turn in turns],
    )


@router.post(
    "/chat/clear",
    response_model=ClearChatResponse,
    summary="Clear a chat session's memory",
    dependencies=[Depends(require_api_key)],
)
async def clear_chat(
    payload: ClearChatRequest | None = None,
    container: ServiceContainer = Depends(container_dep),
) -> ClearChatResponse:
    request = payload or ClearChatRequest()
    cleared = container.memory.clear(request.session_id)
    return ClearChatResponse(
        session_id=request.session_id,
        cleared=cleared,
        message=(
            "Conversation cleared."
            if cleared
            else "There was no conversation history to clear."
        ),
    )
