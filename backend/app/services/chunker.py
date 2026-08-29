"""Recursive, structure-aware chunking.

The splitter walks a separator hierarchy (paragraph -> line -> sentence -> word)
and only descends when a piece is still too large. This avoids cutting sentences
mid-way, which is the main cause of poor retrieval quality.

Every chunk inherits the page/section metadata of the block it came from, so
citations stay accurate and page numbers are never invented.
"""

from __future__ import annotations

import logging
import re

from app.config import Settings, get_settings
from app.models.document import Chunk, ExtractedBlock, ExtractionResult
from app.services import text_cleaner

logger = logging.getLogger(__name__)

#: Ordered from coarsest to finest. The chunker prefers earlier separators.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n\n",
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class TextChunker:
    """Splits cleaned document text into overlapping, metadata-rich chunks."""

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_chars: int | None = None,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.min_chunk_chars = (
            min_chunk_chars if min_chunk_chars is not None else settings.min_chunk_chars
        )
        self.separators = separators

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk_document(
        self,
        extraction: ExtractionResult,
        *,
        document_id: str,
        filename: str,
        file_type: str,
    ) -> list[Chunk]:
        """Clean, de-boilerplate and split an extraction result into chunks."""
        blocks = self._clean_blocks(extraction)

        chunks: list[Chunk] = []
        index = 0
        for block in blocks:
            for piece in self.split_text(block.text):
                if len(piece) < self.min_chunk_chars and not text_cleaner.is_meaningful(
                    piece, min_chars=self.min_chunk_chars
                ):
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}::c{index:05d}",
                        document_id=document_id,
                        filename=filename,
                        file_type=file_type,
                        text=piece,
                        chunk_index=index,
                        page=block.page,
                        section=block.section,
                    )
                )
                index += 1

        logger.info(
            "Chunked document %s into %d chunks (size=%d overlap=%d)",
            document_id,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split one block of text into chunk-sized pieces with overlap."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        pieces = self._recursive_split(text, list(self.separators))
        return self._merge_with_overlap(pieces)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _clean_blocks(self, extraction: ExtractionResult) -> list[ExtractedBlock]:
        """Clean each block and drop repeated headers/footers."""
        page_texts = [block.text for block in extraction.blocks if block.page is not None]
        boilerplate = text_cleaner.find_repeated_lines(page_texts)
        if boilerplate:
            logger.debug("Removing %d repeated header/footer lines", len(boilerplate))

        cleaned: list[ExtractedBlock] = []
        for block in extraction.blocks:
            body = text_cleaner.clean_text(block.text)
            if boilerplate:
                body = text_cleaner.strip_lines(body, boilerplate)
            if not body.strip():
                continue
            cleaned.append(
                ExtractedBlock(
                    text=body,
                    page=block.page,
                    section=block.section,
                    metadata=block.metadata,
                )
            )
        return cleaned

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Split using the coarsest separator that actually helps."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return self._hard_split(text)

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            return self._hard_split(text)

        if separator not in text:
            return self._recursive_split(text, remaining)

        parts = text.split(separator)
        results: list[str] = []
        for part in parts:
            # Re-attach the separator so sentence punctuation is not lost.
            candidate = part if separator.isspace() or separator == "" else part + separator.rstrip()
            candidate = candidate.strip()
            if not candidate:
                continue
            if len(candidate) <= self.chunk_size:
                results.append(candidate)
            else:
                results.extend(self._recursive_split(candidate, remaining))
        return results

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: fixed-width slices for text with no usable separator."""
        step = self.chunk_size
        return [
            text[start : start + step].strip()
            for start in range(0, len(text), step)
            if text[start : start + step].strip()
        ]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Greedily pack small pieces up to chunk_size, then add tail overlap."""
        if not pieces:
            return []

        chunks: list[str] = []
        buffer: list[str] = []
        buffer_len = 0

        for piece in pieces:
            piece_len = len(piece)
            # +1 accounts for the joining space/newline.
            if buffer and buffer_len + piece_len + 1 > self.chunk_size:
                chunks.append(" ".join(buffer).strip())
                carry = self._overlap_tail(buffer)
                buffer = carry + [piece]
                buffer_len = sum(len(part) + 1 for part in buffer)
            else:
                buffer.append(piece)
                buffer_len += piece_len + 1

        if buffer:
            tail = " ".join(buffer).strip()
            # Fold a tiny trailing chunk back into the previous one.
            if chunks and len(tail) < self.min_chunk_chars:
                chunks[-1] = f"{chunks[-1]} {tail}".strip()
            elif tail:
                chunks.append(tail)

        return [chunk for chunk in chunks if chunk.strip()]

    def _overlap_tail(self, buffer: list[str]) -> list[str]:
        """Take whole trailing pieces up to `chunk_overlap` characters.

        Overlapping on piece boundaries (rather than raw characters) keeps the
        carried-over context readable.
        """
        if self.chunk_overlap <= 0:
            return []
        carry: list[str] = []
        total = 0
        for piece in reversed(buffer):
            if total + len(piece) > self.chunk_overlap and carry:
                break
            carry.insert(0, piece)
            total += len(piece) + 1
            if total >= self.chunk_overlap:
                break
        return carry


def group_chunks_for_summary(
    chunks: list[Chunk], max_chars: int, max_groups: int
) -> list[list[Chunk]]:
    """Pack ordered chunks into contiguous groups for map-reduce summarization.

    Keeping groups contiguous and in document order means each map-step summary
    covers a coherent section rather than a random sample.
    """
    if not chunks:
        return []

    ordered = sorted(chunks, key=lambda c: c.chunk_index)
    groups: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_len = 0

    for chunk in ordered:
        if current and current_len + chunk.char_count > max_chars:
            groups.append(current)
            current = [chunk]
            current_len = chunk.char_count
        else:
            current.append(chunk)
            current_len += chunk.char_count
    if current:
        groups.append(current)

    # Respect the LLM call budget by evenly redistributing into fewer groups.
    if max_groups > 0 and len(groups) > max_groups:
        per_group = -(-len(ordered) // max_groups)  # ceil division
        groups = [ordered[i : i + per_group] for i in range(0, len(ordered), per_group)]

    return groups
