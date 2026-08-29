"""RAG chatbot: query rewriting, retrieval, grounded generation, citations.

Answer flow
-----------
1. Rewrite a follow-up question into a standalone query using chat history.
2. Retrieve the top-k relevant chunks (optionally scoped to selected documents).
3. If nothing relevant is found, refuse deterministically without calling the LLM.
4. Generate an answer from the retrieved context only, with an explicit
   anti-hallucination system prompt.
5. Extract citations from the `[S#]` markers the model actually used.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.models.document import RetrievedChunk
from app.prompts import REFUSAL_MESSAGE, PromptName, render, system_prompt
from app.services.chat_memory import ChatMemory
from app.services.groq_service import GroqService
from app.services.retriever import RetrievalContext, Retriever

logger = logging.getLogger(__name__)

#: Matches citation markers the model emits, e.g. [S1] or [S1, S3]
_MARKER_PATTERN = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]", re.IGNORECASE)

#: Excerpt length included with each citation in the API response.
EXCERPT_CHARS = 300


@dataclass(slots=True)
class ChatResult:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    grounded: bool = True
    retrieved_chunks: int = 0
    model: str = ""
    document_ids: list[str] = field(default_factory=list)


class Chatbot:
    """Grounded question answering over indexed documents."""

    def __init__(
        self,
        retriever: Retriever,
        groq_service: GroqService,
        memory: ChatMemory,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.groq = groq_service
        self.memory = memory
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        session_id: str = "default",
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> ChatResult:
        """Answer a question using only retrieved document context."""
        context, search_query = self._prepare(
            question, session_id=session_id, document_ids=document_ids, top_k=top_k
        )

        if context.is_empty:
            # Deterministic refusal — no LLM call, so no chance of invention.
            self.memory.add_user_message(session_id, question)
            self.memory.add_assistant_message(session_id, REFUSAL_MESSAGE)
            return ChatResult(
                answer=REFUSAL_MESSAGE,
                grounded=False,
                retrieved_chunks=0,
                model=self.groq.model,
                document_ids=list(document_ids or []),
            )

        messages = self._build_messages(question, context, session_id)
        answer = self.groq.complete(messages)

        return self._finalize(question, answer, context, session_id)

    def stream_answer(
        self,
        question: str,
        *,
        session_id: str = "default",
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> Iterator[tuple[str, object]]:
        """Yield `(event_type, payload)` tuples for server-sent events.

        Events: `token` (str), `sources` (list[dict]), `done` (dict).
        """
        context, _ = self._prepare(
            question, session_id=session_id, document_ids=document_ids, top_k=top_k
        )

        if context.is_empty:
            self.memory.add_user_message(session_id, question)
            self.memory.add_assistant_message(session_id, REFUSAL_MESSAGE)
            yield "token", REFUSAL_MESSAGE
            yield "sources", []
            yield "done", {"grounded": False, "retrieved_chunks": 0, "model": self.groq.model}
            return

        messages = self._build_messages(question, context, session_id)

        pieces: list[str] = []
        for piece in self.groq.stream(messages):
            pieces.append(piece)
            yield "token", piece

        answer = "".join(pieces).strip()
        result = self._finalize(question, answer, context, session_id)

        yield "sources", [self.citation_payload(chunk) for chunk in result.sources]
        yield "done", {
            "grounded": result.grounded,
            "retrieved_chunks": result.retrieved_chunks,
            "model": result.model,
        }

    @staticmethod
    def citation_payload(chunk: RetrievedChunk) -> dict[str, object]:
        """Serialise a retrieved chunk as an API citation."""
        excerpt = chunk.text.strip()
        if len(excerpt) > EXCERPT_CHARS:
            excerpt = excerpt[:EXCERPT_CHARS].rstrip() + "…"
        return {
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "page": chunk.page,
            "section": chunk.section,
            "score": round(chunk.score, 4),
            "excerpt": excerpt,
            "label": chunk.citation_label,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _prepare(
        self,
        question: str,
        *,
        session_id: str,
        document_ids: list[str] | None,
        top_k: int | None,
    ) -> tuple[RetrievalContext, str]:
        search_query = self._resolve_query(question, session_id)
        context = self.retriever.retrieve(
            search_query, document_ids=document_ids, top_k=top_k
        )

        # If the rewritten query found nothing, retry with the raw question.
        if context.is_empty and search_query != question:
            logger.debug("Rewritten query found nothing; retrying with the original")
            context = self.retriever.retrieve(
                question, document_ids=document_ids, top_k=top_k
            )
        return context, search_query

    def _resolve_query(self, question: str, session_id: str) -> str:
        """Rewrite a follow-up into a standalone retrieval query."""
        if not self.memory.has_history(session_id):
            return question
        if not self._looks_like_followup(question):
            return question
        if not self.groq.is_configured:
            return question

        try:
            rewritten = self.groq.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You rewrite conversational questions into standalone "
                            "search queries. Output only the rewritten query."
                        ),
                    },
                    {
                        "role": "user",
                        "content": render(
                            PromptName.FOLLOWUP_REWRITE,
                            history=self.memory.render_history(session_id),
                            question=question,
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=120,
            )
        except Exception as exc:
            # Rewriting is an optimisation; degrade to the literal question.
            logger.warning("Query rewrite failed (%s); using original", type(exc).__name__)
            return question

        cleaned = rewritten.strip().strip('"').splitlines()[0].strip()
        # Guard against the model answering instead of rewriting.
        if not cleaned or len(cleaned) > 400:
            return question
        logger.debug("Rewrote query: %r -> %r", question, cleaned)
        return cleaned

    @staticmethod
    def _looks_like_followup(question: str) -> bool:
        """Cheap heuristic that avoids an LLM call on clearly standalone questions."""
        lowered = question.lower().strip()
        if len(lowered.split()) <= 3:
            return True
        referential = (
            " it ", " its ", " it?", " its?", " they ", " them ", " their ",
            " this ", " that ", " these ", " those ", " he ", " she ", " his ",
            " her ", " same ", " above ", " previous ", " earlier ", " former ",
            " latter ",
        )
        padded = f" {lowered} "
        if any(token in padded for token in referential):
            return True
        return lowered.startswith(
            (
                "what about",
                "how about",
                "and ",
                "also",
                "why",
                "why?",
                "elaborate",
                "expand",
                "tell me more",
                "continue",
                "go on",
                "summarize that",
                "explain more",
            )
        )

    def _build_messages(
        self, question: str, context: RetrievalContext, session_id: str
    ) -> list[dict[str, str]]:
        """Assemble system prompt + recent history + grounded user prompt."""
        document_ids = context.document_ids

        if len(document_ids) > 1:
            document_list = "\n".join(f"- {name}" for name in context.document_names)
            user_prompt = render(
                PromptName.QA_MULTI,
                document_count=len(document_ids),
                document_list=document_list,
                context=context.context_text,
                question=question,
            )
        else:
            user_prompt = render(
                PromptName.QA_SINGLE,
                filename=context.document_names[0] if context.document_names else "document",
                context=context.context_text,
                question=question,
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt()}
        ]

        # Recent history gives the model conversational continuity. Context for the
        # current question always comes from retrieval, never from history.
        history = self.memory.get_messages(
            session_id, limit=self.settings.chat_history_turns
        )
        for turn in history:
            content = turn["content"]
            if turn["role"] == "assistant" and len(content) > 800:
                content = content[:800].rstrip() + "…"
            messages.append({"role": turn["role"], "content": content})

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _finalize(
        self,
        question: str,
        answer: str,
        context: RetrievalContext,
        session_id: str,
    ) -> ChatResult:
        """Detect refusals, extract citations and record the turn."""
        answer = answer.strip()
        grounded = not self._is_refusal(answer)
        sources = self._extract_sources(answer, context) if grounded else []

        self.memory.add_user_message(session_id, question)
        self.memory.add_assistant_message(session_id, answer)

        return ChatResult(
            answer=answer,
            sources=sources,
            grounded=grounded,
            retrieved_chunks=len(context.chunks),
            model=self.groq.model,
            document_ids=context.document_ids,
        )

    @staticmethod
    def _is_refusal(answer: str) -> bool:
        """Detect an abstention so the UI can flag it as ungrounded."""
        normalized = answer.lower().rstrip(" .")
        canonical = REFUSAL_MESSAGE.lower().rstrip(" .")
        if canonical in normalized:
            return True
        signals = (
            "couldn't find this information in the uploaded document",
            "could not find this information in the uploaded document",
            "don't have enough information from the document",
            "do not have enough information from the document",
            "the information is not available in the provided",
            "not available in the provided context",
            "the document does not contain",
        )
        # Only treat these as refusals for short replies; a long answer that
        # mentions a gap in passing is still a grounded answer.
        return len(answer) < 320 and any(signal in normalized for signal in signals)

    @staticmethod
    def _extract_sources(
        answer: str, context: RetrievalContext
    ) -> list[RetrievedChunk]:
        """Return chunks the model actually cited, else fall back to top hits.

        Only markers present in `marker_map` are honoured, so a hallucinated
        marker can never produce a fabricated citation.
        """
        cited: list[RetrievedChunk] = []
        seen: set[str] = set()

        for match in _MARKER_PATTERN.finditer(answer):
            for raw in match.group(1).split(","):
                marker = raw.strip().upper()
                chunk = context.marker_map.get(marker)
                if chunk and chunk.chunk_id not in seen:
                    cited.append(chunk)
                    seen.add(chunk.chunk_id)

        if cited:
            return cited

        # The model answered without markers: attribute to the strongest chunks
        # that were actually supplied. Still real provenance, never invented.
        return context.chunks[:3]
