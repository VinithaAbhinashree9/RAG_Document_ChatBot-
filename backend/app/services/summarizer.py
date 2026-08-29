"""Document summarization.

Two strategies, chosen by document size:

  direct      — small documents fit in one call.
  map_reduce  — large documents are summarized hierarchically:

      Document -> contiguous chunk groups -> per-group notes
               -> combined notes -> final structured summary

The full document is never sent blindly to the LLM: `SUMMARY_MAP_CHUNK_CHARS`
bounds each map call and `SUMMARY_MAX_MAP_CALLS` bounds the total call budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from app.config import Settings, get_settings
from app.models.document import Chunk
from app.prompts import PromptName, render, system_prompt
from app.services.chunker import group_chunks_for_summary
from app.services.groq_service import GroqService

logger = logging.getLogger(__name__)

#: A map-step reply of exactly this means "nothing substantive here".
SKIP_TOKEN = "SKIP"


@dataclass(slots=True)
class SummaryResult:
    summary: str
    strategy: Literal["direct", "map_reduce"]
    map_calls: int = 0


class Summarizer:
    """Produces structured, grounded document summaries."""

    def __init__(
        self,
        groq_service: GroqService,
        settings: Settings | None = None,
    ) -> None:
        self.groq = groq_service
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    def summarize(self, chunks: list[Chunk], filename: str) -> SummaryResult:
        """Summarize a document from its indexed chunks."""
        if not chunks:
            return SummaryResult(
                summary="No content was available to summarize.", strategy="direct"
            )

        ordered = sorted(chunks, key=lambda chunk: chunk.chunk_index)
        total_chars = sum(chunk.char_count for chunk in ordered)

        if total_chars <= self.settings.summary_map_chunk_chars:
            return self._summarize_direct(ordered, filename)
        return self._summarize_map_reduce(ordered, filename)

    # ------------------------------------------------------------------
    def _summarize_direct(self, chunks: list[Chunk], filename: str) -> SummaryResult:
        content = self._render_chunks(chunks)
        prompt = render(
            PromptName.SUMMARIZE_DIRECT,
            filename=filename,
            content=content,
            target_words=self.settings.summary_target_words,
        )
        summary = self.groq.complete(
            [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=self.settings.groq_temperature,
        )
        return SummaryResult(summary=summary.strip(), strategy="direct", map_calls=1)

    def _summarize_map_reduce(
        self, chunks: list[Chunk], filename: str
    ) -> SummaryResult:
        groups = group_chunks_for_summary(
            chunks,
            max_chars=self.settings.summary_map_chunk_chars,
            max_groups=self.settings.summary_max_map_calls,
        )
        total = len(groups)
        logger.info(
            "Map-reduce summarization of %s: %d groups from %d chunks",
            filename,
            total,
            len(chunks),
        )

        notes: list[str] = []
        map_calls = 0
        failures = 0

        for index, group in enumerate(groups, start=1):
            prompt = render(
                PromptName.SUMMARIZE_MAP,
                filename=filename,
                part=index,
                total=total,
                content=self._render_chunks(group),
            )
            try:
                note = self.groq.complete(
                    [
                        {"role": "system", "content": system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.settings.groq_temperature,
                    # Map notes are intentionally terse.
                    max_tokens=min(700, self.settings.groq_max_tokens),
                )
                map_calls += 1
            except Exception as exc:
                # One bad section must not sink the whole summary.
                failures += 1
                logger.warning(
                    "Map step %d/%d failed (%s); continuing",
                    index,
                    total,
                    type(exc).__name__,
                )
                if failures > max(2, total // 2):
                    raise
                continue

            cleaned = note.strip()
            if not cleaned or cleaned.upper().startswith(SKIP_TOKEN):
                continue

            pages = sorted({c.page for c in group if c.page is not None})
            location = f" (pages {pages[0]}–{pages[-1]})" if len(pages) > 1 else (
                f" (page {pages[0]})" if pages else ""
            )
            notes.append(f"### Section {index} of {total}{location}\n{cleaned}")

        if not notes:
            return SummaryResult(
                summary=(
                    "The document was indexed, but no substantive content could be "
                    "summarized from it."
                ),
                strategy="map_reduce",
                map_calls=map_calls,
            )

        combined = self._fit("\n\n".join(notes))
        reduce_prompt = render(
            PromptName.SUMMARIZE_REDUCE,
            filename=filename,
            section_summaries=combined,
            target_words=self.settings.summary_target_words,
        )
        final = self.groq.complete(
            [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": reduce_prompt},
            ],
            temperature=self.settings.groq_temperature,
        )
        map_calls += 1

        result = SummaryResult(
            summary=final.strip(), strategy="map_reduce", map_calls=map_calls
        )
        if failures:
            result.summary += (
                f"\n\n_Note: {failures} of {total} sections could not be processed and "
                "are not reflected in this summary._"
            )
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _render_chunks(chunks: list[Chunk]) -> str:
        """Render chunks with page markers so the model can cite real locations."""
        parts: list[str] = []
        last_page: int | None = None
        for chunk in chunks:
            if chunk.page is not None and chunk.page != last_page:
                parts.append(f"[page {chunk.page}]")
                last_page = chunk.page
            elif chunk.section and not parts:
                parts.append(f"[section: {chunk.section}]")
            parts.append(chunk.text)
        return "\n".join(parts)

    def _fit(self, text: str) -> str:
        """Keep the reduce-step input inside the context budget."""
        budget = self.settings.max_context_chars
        if len(text) <= budget:
            return text
        logger.warning("Combined section notes exceeded the budget; truncating")
        head = int(budget * 0.7)
        tail = budget - head - 60
        return (
            f"{text[:head]}\n\n[... intermediate section notes omitted ...]\n\n{text[-tail:]}"
        )
