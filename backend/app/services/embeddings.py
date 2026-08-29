"""Embedding service.

Deliberately independent of the Groq LLM layer: Groq serves chat completions and
does not provide an embedding endpoint, so semantic search uses a dedicated
embedding model. Swapping providers means adding one `EmbeddingProvider`
subclass — the retriever and vector store are unaffected.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import Settings, get_settings
from app.errors import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Contract for turning text into dense vectors."""

    name: str = "base"

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""

    @property
    def model_name(self) -> str:
        return self.name


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local sentence-transformers model. Free, private, no external API call.

    The model is loaded lazily on first use so importing this module (and running
    tests that never embed anything) stays fast.
    """

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.name = model_name
        self.batch_size = max(1, batch_size)
        self.normalize = normalize
        self._model = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:  # pragma: no cover - double-checked lock
                return self._model
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Run "
                    "`pip install -r requirements.txt`."
                ) from exc
            try:
                logger.info("Loading embedding model: %s", self.name)
                model = SentenceTransformer(self.name)
                self._dimension = int(model.get_sentence_embedding_dimension())
                self._model = model
                logger.info(
                    "Embedding model ready (dimension=%d)", self._dimension
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load the embedding model '{self.name}'. Check the "
                    "EMBEDDING_MODEL value and network access, or set "
                    "EMBEDDING_OFFLINE_FALLBACK=true to use the offline embedder."
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_model()
        return int(self._dimension or 384)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(
                "Embedding generation failed while indexing the document."
            ) from exc
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashingEmbeddings(EmbeddingProvider):
    """Deterministic offline fallback: hashed bag-of-words with sublinear TF.

    Purely lexical, so retrieval quality is materially lower than a real
    sentence encoder. It exists so the app still boots (and CI still runs) when
    the model cannot be downloaded. Never select this in production.
    """

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dimension: int = 512) -> None:
        self.name = f"hashing-{dimension}"
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _tokenize(self, text: str) -> list[str]:
        words = self._TOKEN.findall(text.lower())
        # Include bigrams so short phrases retain a little word order.
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in self._tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        # Sublinear scaling dampens very frequent tokens.
        vector = [math.copysign(math.log1p(abs(v)), v) for v in vector]
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Factory selecting a provider from configuration."""
    settings = settings or get_settings()

    if settings.embedding_offline_fallback:
        logger.warning(
            "EMBEDDING_OFFLINE_FALLBACK is enabled — using the hashing embedder. "
            "Retrieval quality will be reduced."
        )
        return HashingEmbeddings()

    provider = settings.embedding_provider.lower().strip()
    if provider in {"sentence-transformers", "sentence_transformers", "local", "st"}:
        return SentenceTransformerEmbeddings(
            settings.embedding_model, batch_size=settings.embedding_batch_size
        )
    if provider in {"hashing", "offline"}:
        return HashingEmbeddings()

    raise EmbeddingError(
        f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
        "Supported values: sentence-transformers, hashing."
    )


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Process-wide provider (the model is expensive to load)."""
    return build_embedding_provider()
