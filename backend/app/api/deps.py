"""Dependency injection container.

Services are constructed once per process and shared. Tests override
`get_container()` with an in-memory container, which is why every route resolves
its collaborators through this module rather than importing them directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.errors import UnauthorizedError
from app.services.chat_memory import ChatMemory
from app.services.chatbot import Chatbot
from app.services.chunker import TextChunker
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingProvider, build_embedding_provider
from app.services.groq_service import GroqService
from app.services.retriever import Retriever
from app.services.summarizer import Summarizer
from app.services.vector_store import VectorStore, build_vector_store

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    """Wired application services."""

    settings: Settings
    embeddings: EmbeddingProvider
    vector_store: VectorStore
    chunker: TextChunker
    processor: DocumentProcessor
    retriever: Retriever
    groq: GroqService
    summarizer: Summarizer
    memory: ChatMemory
    chatbot: Chatbot

    @classmethod
    def create(cls, settings: Settings | None = None) -> "ServiceContainer":
        settings = settings or get_settings()
        settings.ensure_directories()

        embeddings = build_embedding_provider(settings)
        vector_store = build_vector_store(settings, embeddings)
        chunker = TextChunker(settings=settings)
        processor = DocumentProcessor(
            vector_store=vector_store,
            embedding_provider=embeddings,
            chunker=chunker,
            settings=settings,
        )
        retriever = Retriever(vector_store, embeddings, settings)
        groq = GroqService(settings)
        summarizer = Summarizer(groq, settings)
        # Memory holds `chat_history_turns` exchanges => 2 messages each.
        memory = ChatMemory(max_turns=max(2, settings.chat_history_turns * 2))
        chatbot = Chatbot(retriever, groq, memory, settings)

        return cls(
            settings=settings,
            embeddings=embeddings,
            vector_store=vector_store,
            chunker=chunker,
            processor=processor,
            retriever=retriever,
            groq=groq,
            summarizer=summarizer,
            memory=memory,
            chatbot=chatbot,
        )


@lru_cache
def get_container() -> ServiceContainer:
    """Process-wide service container."""
    return ServiceContainer.create()


# --------------------------------------------------------------------------
# FastAPI dependencies
# --------------------------------------------------------------------------
def container_dep() -> ServiceContainer:
    return get_container()


def get_processor(
    container: ServiceContainer = Depends(container_dep),
) -> DocumentProcessor:
    return container.processor


def get_retriever(container: ServiceContainer = Depends(container_dep)) -> Retriever:
    return container.retriever


def get_summarizer(container: ServiceContainer = Depends(container_dep)) -> Summarizer:
    return container.summarizer


def get_chatbot(container: ServiceContainer = Depends(container_dep)) -> Chatbot:
    return container.chatbot


def get_memory(container: ServiceContainer = Depends(container_dep)) -> ChatMemory:
    return container.memory


def get_groq(container: ServiceContainer = Depends(container_dep)) -> GroqService:
    return container.groq


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    container: ServiceContainer = Depends(container_dep),
) -> None:
    """Optional shared-secret gate.

    Disabled when APP_API_KEY is empty (the local-development default), so the
    app is usable out of the box but can be locked down in production without a
    code change.
    """
    expected = container.settings.app_api_key.strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise UnauthorizedError()
