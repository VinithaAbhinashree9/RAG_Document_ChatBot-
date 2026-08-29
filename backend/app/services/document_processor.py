"""Ingest orchestration: the indexing half of the RAG pipeline.

    validate -> persist -> extract -> clean -> chunk -> embed -> vector store

Also owns the in-process document registry (metadata for every uploaded file),
persisted to disk so the catalogue survives a restart alongside the vector store.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from app.config import Settings, get_settings
from app.errors import (
    AppError,
    CorruptDocumentError,
    DocumentNotFoundError,
    EmptyDocumentError,
    ExtractionError,
)
from app.models.document import (
    Chunk,
    DocumentRecord,
    DocumentStatus,
    ExtractionResult,
    new_id,
)
from app.services import file_validator
from app.services.chunker import TextChunker
from app.services.embeddings import EmbeddingProvider
from app.services.extraction import registry
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "documents.json"


class DocumentRegistry:
    """Thread-safe document catalogue with JSON persistence."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._records: dict[str, DocumentRecord] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.storage_path.is_file():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read the document registry: %s", type(exc).__name__)
            return

        for item in payload.get("documents", []):
            try:
                record = DocumentRecord(
                    document_id=item["document_id"],
                    filename=item["filename"],
                    stored_filename=item.get("stored_filename", item["filename"]),
                    file_type=item["file_type"],
                    size_bytes=int(item.get("size_bytes", 0)),
                    status=DocumentStatus(item.get("status", "ready")),
                    page_count=item.get("page_count"),
                    char_count=int(item.get("char_count", 0)),
                    chunk_count=int(item.get("chunk_count", 0)),
                    summary=item.get("summary"),
                    error=item.get("error"),
                    extractor=item.get("extractor"),
                    warnings=list(item.get("warnings", [])),
                )
                if item.get("created_at"):
                    record.created_at = datetime.fromisoformat(item["created_at"])
                if item.get("updated_at"):
                    record.updated_at = datetime.fromisoformat(item["updated_at"])
                self._records[record.document_id] = record
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed registry entry: %s", exc)

        logger.info("Loaded %d documents from the registry", len(self._records))

    def _persist_unlocked(self) -> None:
        payload = {
            "documents": [record.to_dict() | {"stored_filename": record.stored_filename}
                          for record in self._records.values()]
        }
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace so a crash cannot leave a truncated registry.
            temp_path = self.storage_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(self.storage_path)
        except OSError as exc:
            logger.error("Could not persist the document registry: %s", type(exc).__name__)

    def add(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.document_id] = record
            self._persist_unlocked()

    def update(self, record: DocumentRecord) -> None:
        record.touch()
        with self._lock:
            self._records[record.document_id] = record
            self._persist_unlocked()

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(document_id)

    def require(self, document_id: str) -> DocumentRecord:
        record = self.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"No document with ID '{document_id}'.")
        return record

    def list_all(self) -> list[DocumentRecord]:
        with self._lock:
            records = list(self._records.values())
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def list_ready(self) -> list[DocumentRecord]:
        return [
            record
            for record in self.list_all()
            if record.status in {DocumentStatus.READY, DocumentStatus.INDEXED}
        ]

    def remove(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            record = self._records.pop(document_id, None)
            if record is not None:
                self._persist_unlocked()
            return record

    def count(self) -> int:
        with self._lock:
            return len(self._records)


class DocumentProcessor:
    """Runs the ingest pipeline and manages stored files."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        chunker: TextChunker | None = None,
        registry_store: DocumentRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store
        self.embeddings = embedding_provider
        self.chunker = chunker or TextChunker(settings=self.settings)
        self.registry = registry_store or DocumentRegistry(
            self.settings.upload_dir / REGISTRY_FILENAME
        )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def process_upload(self, filename: str, content: bytes) -> DocumentRecord:
        """Validate, store, extract, chunk, embed and index an uploaded file."""
        safe_name, extension = file_validator.validate_upload(
            filename, len(content), content[:512], self.settings
        )

        document_id = new_id("doc")
        stored_name = f"{document_id}{extension}"
        stored_path = self.settings.upload_dir / stored_name

        record = DocumentRecord(
            document_id=document_id,
            filename=safe_name,
            stored_filename=stored_name,
            file_type=extension.lstrip("."),
            size_bytes=len(content),
            status=DocumentStatus.PROCESSING,
        )
        self.registry.add(record)

        try:
            self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
            stored_path.write_bytes(content)

            extraction = self._extract(stored_path, extension)
            record.extractor = extraction.extractor
            record.page_count = extraction.page_count
            record.warnings = list(extraction.warnings)

            chunks = self.chunker.chunk_document(
                extraction,
                document_id=document_id,
                filename=safe_name,
                file_type=record.file_type,
            )
            if not chunks:
                raise EmptyDocumentError(
                    "No meaningful text could be extracted from this document. If it "
                    "is a scanned PDF, it needs OCR before it can be indexed."
                )

            record.char_count = sum(chunk.char_count for chunk in chunks)
            self._index(chunks)
            record.chunk_count = len(chunks)
            record.status = DocumentStatus.INDEXED
            self.registry.update(record)

            logger.info(
                "Indexed document %s (%s): %d chunks, %d chars",
                document_id,
                record.file_type,
                record.chunk_count,
                record.char_count,
            )
            return record

        except AppError as exc:
            record.status = DocumentStatus.FAILED
            record.error = exc.message
            self.registry.update(record)
            self._cleanup_file(stored_path)
            raise
        except Exception as exc:
            logger.exception("Unexpected failure while processing an upload")
            record.status = DocumentStatus.FAILED
            record.error = "An unexpected error occurred while processing the document."
            self.registry.update(record)
            self._cleanup_file(stored_path)
            raise ExtractionError(record.error) from exc

    def _extract(self, path: Path, extension: str) -> ExtractionResult:
        extractor = registry.get(extension)
        if extractor is None:  # pragma: no cover - validation already guarantees this
            raise ExtractionError(f"No extractor is registered for '{extension}'.")
        try:
            extraction = extractor.extract(path)
        except AppError:
            raise
        except Exception as exc:
            logger.warning(
                "Extractor %s failed on %s: %s",
                extractor.name,
                extension,
                type(exc).__name__,
            )
            raise CorruptDocumentError(
                "The document could not be read. It may be corrupted or malformed."
            ) from exc

        if not extraction.blocks:
            hint = " ".join(extraction.warnings) if extraction.warnings else ""
            raise EmptyDocumentError(
                f"No readable text was found in this document. {hint}".strip()
            )
        return extraction

    def _index(self, chunks: list[Chunk]) -> None:
        """Embed in batches, then write to the vector store."""
        batch_size = max(1, self.settings.embedding_batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            window = chunks[start : start + batch_size]
            vectors.extend(self.embeddings.embed_documents([c.text for c in window]))
        self.vector_store.add_chunks(chunks, vectors)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------
    def get_chunks(self, document_id: str) -> list[Chunk]:
        self.registry.require(document_id)
        return self.vector_store.get_document_chunks(document_id)

    def delete_document(self, document_id: str) -> bool:
        """Remove a document from the registry, vector store and disk."""
        record = self.registry.require(document_id)
        try:
            removed = self.vector_store.delete_document(document_id)
            logger.info("Deleted %d chunks for document %s", removed, document_id)
        except AppError as exc:
            logger.warning("Vector store delete failed for %s: %s", document_id, exc.code)

        self._cleanup_file(self.settings.upload_dir / record.stored_filename)
        self.registry.remove(document_id)
        return True

    def set_summary(self, document_id: str, summary: str) -> DocumentRecord:
        record = self.registry.require(document_id)
        record.summary = summary
        record.status = DocumentStatus.READY
        self.registry.update(record)
        return record

    def mark_failed(self, document_id: str, message: str) -> None:
        record = self.registry.get(document_id)
        if record is None:
            return
        record.status = DocumentStatus.FAILED
        record.error = message
        self.registry.update(record)

    @staticmethod
    def _cleanup_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - best effort
            logger.warning("Could not delete %s: %s", path.name, type(exc).__name__)
