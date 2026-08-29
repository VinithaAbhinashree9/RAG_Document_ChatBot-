"""Central configuration.

All tunables live here and are sourced from environment variables / `.env`.
No secret or model name is ever hard-coded in business logic.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- Groq ----------------
    groq_api_key: str = Field(default="", description="Groq API key; loaded from env only.")
    groq_model: str = "openai/gpt-oss-120b"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 2048
    groq_timeout_seconds: float = 60.0
    groq_max_retries: int = 3

    # ---------------- Embeddings ----------------
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_offline_fallback: bool = False

    # ---------------- Vector store ----------------
    vector_store: str = "chroma"
    chroma_persist_dir: Path = BACKEND_ROOT / "data" / "chroma"
    chroma_collection: str = "documents"

    # ---------------- Chunking ----------------
    chunk_size: int = 1200
    chunk_overlap: int = 180
    min_chunk_chars: int = 60

    # ---------------- Retrieval ----------------
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.15
    max_context_chars: int = 14_000

    # ---------------- Summarization ----------------
    summary_map_chunk_chars: int = 9_000
    summary_max_map_calls: int = 12
    summary_target_words: int = 350

    # ---------------- Chat memory ----------------
    chat_history_turns: int = 6

    # ---------------- Uploads ----------------
    upload_dir: Path = BACKEND_ROOT / "data" / "uploads"
    max_upload_mb: int = 25

    # ---------------- Server ----------------
    app_env: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    app_api_key: str = ""

    # ---------------- Derived / validators ----------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string from the environment."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("chroma_persist_dir", "upload_dir", mode="before")
    @classmethod
    def _resolve_path(cls, value: object) -> object:
        if isinstance(value, str):
            path = Path(value)
            return path if path.is_absolute() else (BACKEND_ROOT / path).resolve()
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, value: int, info) -> int:
        size = info.data.get("chunk_size", 1200)
        if value >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}

    def ensure_directories(self) -> None:
        """Create runtime directories on startup."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (safe as a FastAPI dependency)."""
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logging. Document contents are never logged."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


settings = get_settings()
