"""Application error hierarchy.

Every error carries a stable machine-readable `code`, an HTTP status and a
*user-safe* message. Stack traces and internal details never reach the client.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all handled application errors."""

    code: str = "internal_error"
    status_code: int = 500
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.details = details or {}
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {"code": self.code, "message": self.message},
        }
        if self.details:
            payload["error"]["details"] = self.details
        return payload


# --------------------------------------------------------------------------
# Upload / validation
# --------------------------------------------------------------------------
class ValidationError(AppError):
    code = "validation_error"
    status_code = 422
    message = "The request payload is invalid."


class UnsupportedFileTypeError(AppError):
    code = "unsupported_file_type"
    status_code = 415
    message = "This file type is not supported."


class FileTooLargeError(AppError):
    code = "file_too_large"
    status_code = 413
    message = "The uploaded file exceeds the maximum allowed size."


class EmptyDocumentError(AppError):
    code = "empty_document"
    status_code = 422
    message = "No readable text could be extracted from this document."


class CorruptDocumentError(AppError):
    code = "corrupt_document"
    status_code = 422
    message = "The document appears to be corrupted or password protected."


class ExtractionError(AppError):
    code = "extraction_failed"
    status_code = 422
    message = "The document could not be parsed."


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------
class DocumentNotFoundError(AppError):
    code = "document_not_found"
    status_code = 404
    message = "Document not found."


# --------------------------------------------------------------------------
# Infrastructure
# --------------------------------------------------------------------------
class EmbeddingError(AppError):
    code = "embedding_failed"
    status_code = 503
    message = "Failed to generate embeddings for this document."


class VectorStoreError(AppError):
    code = "vector_store_error"
    status_code = 503
    message = "The vector database is unavailable."


# --------------------------------------------------------------------------
# LLM provider
# --------------------------------------------------------------------------
class LLMConfigurationError(AppError):
    code = "llm_not_configured"
    status_code = 503
    message = (
        "GROQ_API_KEY is not configured. Add it to backend/.env and restart the server."
    )


class LLMAuthenticationError(AppError):
    code = "llm_auth_failed"
    status_code = 502
    message = "The Groq API rejected the configured API key."


class LLMRateLimitError(AppError):
    code = "llm_rate_limited"
    status_code = 429
    message = "The Groq API rate limit was reached. Please retry shortly."


class LLMTimeoutError(AppError):
    code = "llm_timeout"
    status_code = 504
    message = "The language model took too long to respond."


class LLMError(AppError):
    code = "llm_error"
    status_code = 502
    message = "The language model provider returned an error."


class UnauthorizedError(AppError):
    code = "unauthorized"
    status_code = 401
    message = "Missing or invalid API key."
