"""Conversational memory.

Keeps a bounded, per-session transcript so follow-up questions like
"what are *its* recommendations?" can be resolved into standalone queries.

In-process storage is appropriate for a single-node deployment; the interface is
narrow enough to be backed by Redis for multi-worker production use (see the
deployment notes in the README).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from app.models.document import ChatTurn

#: Cap on tracked sessions; the least-recently-used session is evicted.
MAX_SESSIONS = 200


class ChatMemory:
    """Bounded per-session chat history."""

    def __init__(self, max_turns: int = 12) -> None:
        # max_turns counts individual messages, not user/assistant pairs.
        self.max_turns = max_turns
        self._sessions: OrderedDict[str, list[ChatTurn]] = OrderedDict()
        self._lock = threading.Lock()

    def add_user_message(self, session_id: str, content: str) -> None:
        self._append(session_id, ChatTurn(role="user", content=content))

    def add_assistant_message(self, session_id: str, content: str) -> None:
        self._append(session_id, ChatTurn(role="assistant", content=content))

    def _append(self, session_id: str, turn: ChatTurn) -> None:
        with self._lock:
            history = self._sessions.get(session_id)
            if history is None:
                history = []
                self._sessions[session_id] = history
                if len(self._sessions) > MAX_SESSIONS:
                    self._sessions.popitem(last=False)
            history.append(turn)
            if len(history) > self.max_turns:
                del history[: len(history) - self.max_turns]
            self._sessions.move_to_end(session_id)

    def get_history(self, session_id: str) -> list[ChatTurn]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_messages(self, session_id: str, *, limit: int | None = None) -> list[dict[str, str]]:
        """History in the provider's message format, excluding the current turn."""
        history = self.get_history(session_id)
        if limit is not None and limit > 0:
            history = history[-limit:]
        return [{"role": turn.role, "content": turn.content} for turn in history]

    def render_history(self, session_id: str, *, limit: int = 4, max_chars: int = 1500) -> str:
        """Compact plain-text transcript used by the query-rewriting prompt."""
        history = self.get_history(session_id)[-limit:]
        if not history:
            return "(no previous conversation)"
        lines = []
        for turn in history:
            label = "User" if turn.role == "user" else "Assistant"
            content = turn.content.strip().replace("\n", " ")
            if len(content) > 400:
                content = content[:400].rstrip() + "…"
            lines.append(f"{label}: {content}")
        rendered = "\n".join(lines)
        return rendered[-max_chars:] if len(rendered) > max_chars else rendered

    def has_history(self, session_id: str) -> bool:
        return bool(self.get_history(session_id))

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
