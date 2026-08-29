"""Semantic retrieval and context assembly.

Responsibilities:
  * embed the query and search the vector store (optionally scoped by document)
  * drop weakly-relevant chunks below a similarity floor
  * assemble a context string with `[S1]`-style markers and a budget cap, so the
    whole document is never sent to the LLM
  * map markers back to verifiable citations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.models.document import RetrievedChunk
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

#: A near-miss hit is only rescued if it scores at least this fraction of the
#: similarity floor.
NEAR_MISS_RATIO = 0.85

#: ...and only if it stands out from the runner-up by at least this factor.
#: A real match is a distinct peak; noise scores flat across all candidates, so
#: this keeps off-topic queries from being rescued into the LLM.
NEAR_MISS_MARGIN = 1.35


@dataclass(slots=True)
class RetrievalContext:
    """Everything the prompt layer needs, plus provenance for citations."""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    context_text: str = ""
    marker_map: dict[str, RetrievedChunk] = field(default_factory=dict)
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def document_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.filename, None)
        return list(seen)

    @property
    def document_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.document_id, None)
        return list(seen)

    @property
    def available_markers(self) -> list[str]:
        return list(self.marker_map)


class Retriever:
    """Query-time half of the RAG pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        settings: Settings | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings = embedding_provider
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalContext:
        """Embed the query, search, filter and assemble context."""
        query = query.strip()
        if not query:
            return RetrievalContext(query=query)

        k = top_k or self.settings.retrieval_top_k
        floor = self.settings.retrieval_min_score if min_score is None else min_score

        query_vector = self.embeddings.embed_query(query)

        # Over-fetch a little so the score filter still leaves useful results.
        candidates = self.vector_store.search(
            query_vector,
            top_k=max(k * 2, k + 4),
            document_ids=document_ids or None,
        )

        relevant = [chunk for chunk in candidates if chunk.score >= floor]

        # Narrow rescue: if the best hit only just misses the floor, keep it and let
        # the prompt-level grounding check decide. This avoids refusing a valid but
        # loosely-worded question, while a clearly off-topic query still retrieves
        # nothing and is refused deterministically without an LLM call.
        if not relevant and candidates:
            best = candidates[0]
            runner_up = candidates[1].score if len(candidates) > 1 else 0.0
            stands_out = runner_up <= 0.0 or best.score >= runner_up * NEAR_MISS_MARGIN
            if best.score >= floor * NEAR_MISS_RATIO and stands_out:
                logger.debug(
                    "No chunk cleared the floor; keeping near-miss peak "
                    "(score=%.3f, runner-up=%.3f)",
                    best.score,
                    runner_up,
                )
                relevant = [best]

        selected = relevant[:k]
        logger.debug(
            "Retrieved %d/%d chunks above score floor %.2f",
            len(selected),
            len(candidates),
            floor,
        )
        return self._assemble(query, selected)

    def list_document_chunks(self, document_id: str) -> list:
        return self.vector_store.get_document_chunks(document_id)

    def search_in_document(
        self, document_id: str, query: str, *, top_k: int = 10
    ) -> list[RetrievedChunk]:
        """Keyword-agnostic semantic search scoped to one document."""
        if not query.strip():
            return []
        vector = self.embeddings.embed_query(query)
        return self.vector_store.search(
            vector, top_k=top_k, document_ids=[document_id]
        )

    # ------------------------------------------------------------------
    def _assemble(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> RetrievalContext:
        """Build the marker-annotated context string within the char budget."""
        context = RetrievalContext(query=query)
        if not chunks:
            return context

        budget = self.settings.max_context_chars
        parts: list[str] = []
        used = 0

        for index, chunk in enumerate(chunks, start=1):
            marker = f"S{index}"
            header = self._format_header(marker, chunk)
            block = f"{header}\n{chunk.text}"

            if used + len(block) > budget:
                remaining = budget - used - len(header) - 20
                # Only include a partial block if a useful amount still fits.
                if remaining > 300:
                    block = f"{header}\n{chunk.text[:remaining].rstrip()}…"
                    parts.append(block)
                    context.chunks.append(chunk)
                    context.marker_map[marker] = chunk
                context.truncated = True
                break

            parts.append(block)
            used += len(block) + 2
            context.chunks.append(chunk)
            context.marker_map[marker] = chunk

        context.context_text = "\n\n".join(parts)
        return context

    @staticmethod
    def _format_header(marker: str, chunk: RetrievedChunk) -> str:
        """Provenance header. Only real metadata is emitted — no invented pages."""
        bits = [f"source: {chunk.filename}"]
        if chunk.page is not None:
            bits.append(f"page: {chunk.page}")
        if chunk.section:
            bits.append(f"section: {chunk.section}")
        bits.append(f"relevance: {chunk.score:.2f}")
        return f"[{marker}] ({', '.join(bits)})"
