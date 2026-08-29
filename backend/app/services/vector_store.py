"""Vector store abstraction + ChromaDB implementation.

`VectorStore` is the seam that keeps the retrieval layer portable: swapping in
Pinecone, Qdrant, Weaviate or Milvus means implementing this interface and
returning it from `build_vector_store`. Nothing above this layer changes.

An in-memory implementation is included and used by the test suite, which keeps
tests fast and hermetic.
"""

from __future__ import annotations

import logging
import math
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.errors import VectorStoreError
from app.models.document import Chunk, RetrievedChunk
from app.services.embeddings import EmbeddingProvider, get_embedding_provider

logger = logging.getLogger(__name__)

#: Sentinel stored when a format genuinely has no page numbers.
NO_PAGE = -1


def _to_retrieved(
    *,
    text: str,
    metadata: dict[str, Any],
    score: float,
) -> RetrievedChunk:
    """Rebuild a RetrievedChunk from flat vector-store metadata."""
    raw_page = metadata.get("page", NO_PAGE)
    try:
        page_value = int(raw_page)
    except (TypeError, ValueError):
        page_value = NO_PAGE
    return RetrievedChunk(
        chunk_id=str(metadata.get("chunk_id", "")),
        document_id=str(metadata.get("document_id", "")),
        filename=str(metadata.get("filename", "")),
        file_type=str(metadata.get("file_type", "")),
        text=text,
        chunk_index=int(metadata.get("chunk_index", 0) or 0),
        score=score,
        page=page_value if page_value > 0 else None,
        section=str(metadata.get("section") or "") or None,
    )


class VectorStore(ABC):
    """Persistence + semantic search over embedded chunks."""

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their vectors."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the `top_k` most similar chunks, optionally scoped by document."""

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete every chunk for a document. Returns the number removed."""

    @abstractmethod
    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        """Return all chunks for a document, in document order."""

    @abstractmethod
    def count(self, document_id: str | None = None) -> int:
        """Count stored chunks, globally or for one document."""

    def health_check(self) -> bool:
        try:
            self.count()
            return True
        except Exception:
            return False


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB store with metadata filtering."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        *,
        embedding_dimension: int | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._embedding_dimension = embedding_dimension
        self._lock = threading.Lock()
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        with self._lock:
            if self._collection is not None:  # pragma: no cover
                return self._collection
            try:
                import chromadb  # type: ignore
                from chromadb.config import Settings as ChromaSettings  # type: ignore
            except ImportError as exc:
                raise VectorStoreError(
                    "chromadb is not installed. Run `pip install -r requirements.txt`."
                ) from exc
            try:
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    # Cosine matches the normalised embeddings we produce.
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "ChromaDB ready (collection=%s, chunks=%d)",
                    self.collection_name,
                    self._collection.count(),
                )
            except Exception as exc:
                raise VectorStoreError(
                    "Could not open the ChromaDB store. Check that CHROMA_PERSIST_DIR "
                    "is writable."
                ) from exc
        return self._collection

    # ------------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}."
            )
        collection = self._ensure_collection()
        try:
            # Chroma has a practical per-call batch ceiling; stay well under it.
            batch = 500
            for start in range(0, len(chunks), batch):
                window = chunks[start : start + batch]
                collection.upsert(
                    ids=[chunk.chunk_id for chunk in window],
                    documents=[chunk.text for chunk in window],
                    embeddings=embeddings[start : start + batch],
                    metadatas=[chunk.to_metadata() for chunk in window],
                )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to write embeddings to the vector database."
            ) from exc

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        collection = self._ensure_collection()
        where: dict[str, Any] | None = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": list(document_ids)}}
            )
        try:
            response = collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, top_k),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Semantic search failed.") from exc

        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            # Chroma cosine distance -> similarity in [0, 1]
            score = 1.0 - float(distance)
            results.append(
                _to_retrieved(
                    text=text or "",
                    metadata=dict(metadata or {}),
                    score=max(0.0, min(1.0, score)),
                )
            )
        return results

    def delete_document(self, document_id: str) -> int:
        collection = self._ensure_collection()
        try:
            existing = collection.get(where={"document_id": document_id}, include=[])
            ids = existing.get("ids") or []
            if ids:
                collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(
                "Failed to delete the document from the vector database."
            ) from exc

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        collection = self._ensure_collection()
        try:
            response = collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError("Failed to read document chunks.") from exc

        chunks: list[Chunk] = []
        for text, metadata in zip(
            response.get("documents") or [], response.get("metadatas") or []
        ):
            metadata = dict(metadata or {})
            page = int(metadata.get("page", NO_PAGE) or NO_PAGE)
            chunks.append(
                Chunk(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    document_id=str(metadata.get("document_id", document_id)),
                    filename=str(metadata.get("filename", "")),
                    file_type=str(metadata.get("file_type", "")),
                    text=text or "",
                    chunk_index=int(metadata.get("chunk_index", 0) or 0),
                    page=page if page > 0 else None,
                    section=str(metadata.get("section") or "") or None,
                )
            )
        chunks.sort(key=lambda chunk: chunk.chunk_index)
        return chunks

    def count(self, document_id: str | None = None) -> int:
        collection = self._ensure_collection()
        try:
            if document_id is None:
                return int(collection.count())
            response = collection.get(where={"document_id": document_id}, include=[])
            return len(response.get("ids") or [])
        except Exception as exc:
            raise VectorStoreError("Failed to count stored chunks.") from exc


class InMemoryVectorStore(VectorStore):
    """Dependency-free store used by tests and as a safe fallback.

    Brute-force cosine similarity: fine for thousands of chunks, not for millions.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}."
            )
        with self._lock:
            for chunk, vector in zip(chunks, embeddings):
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = list(vector)

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        allowed = set(document_ids) if document_ids else None
        scored: list[tuple[float, Chunk]] = []
        with self._lock:
            for chunk_id, chunk in self._chunks.items():
                if allowed is not None and chunk.document_id not in allowed:
                    continue
                score = self._cosine(query_embedding, self._vectors.get(chunk_id, []))
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            _to_retrieved(
                text=chunk.text,
                metadata=chunk.to_metadata(),
                score=max(0.0, min(1.0, score)),
            )
            for score, chunk in scored[: max(1, top_k)]
        ]

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            ids = [
                chunk_id
                for chunk_id, chunk in self._chunks.items()
                if chunk.document_id == document_id
            ]
            for chunk_id in ids:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk_id, None)
            return len(ids)

    def get_document_chunks(self, document_id: str) -> list[Chunk]:
        with self._lock:
            chunks = [
                chunk
                for chunk in self._chunks.values()
                if chunk.document_id == document_id
            ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def count(self, document_id: str | None = None) -> int:
        with self._lock:
            if document_id is None:
                return len(self._chunks)
            return sum(
                1 for chunk in self._chunks.values() if chunk.document_id == document_id
            )


def build_vector_store(
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> VectorStore:
    """Factory selecting the vector store from configuration.

    Add a new backend here (Pinecone/Qdrant/Weaviate/Milvus) without touching
    the retriever, chatbot or API layers.
    """
    settings = settings or get_settings()
    backend = settings.vector_store.lower().strip()

    if backend in {"chroma", "chromadb"}:
        provider = embedding_provider or get_embedding_provider()
        return ChromaVectorStore(
            settings.chroma_persist_dir,
            settings.chroma_collection,
            embedding_dimension=provider.dimension if provider else None,
        )
    if backend in {"memory", "in-memory", "inmemory"}:
        return InMemoryVectorStore()

    raise VectorStoreError(
        f"Unknown VECTOR_STORE '{settings.vector_store}'. Supported values: chroma, memory."
    )
