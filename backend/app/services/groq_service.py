"""Groq LLM client wrapper.

The single boundary between the application and the Groq API. It owns:
  * credential loading (env only — never hard-coded, never returned to clients)
  * model selection via GROQ_MODEL (never hard-coded in business logic)
  * retries with exponential backoff on transient failures
  * mapping provider exceptions onto the application error hierarchy
  * blocking and streaming completions
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator
from typing import Any

from app.config import Settings, get_settings
from app.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

Message = dict[str, str]


class GroqService:
    """Thin, defensive wrapper around the Groq chat-completions API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any | None = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------
    @property
    def is_configured(self) -> bool:
        return self.settings.groq_configured

    @property
    def model(self) -> str:
        return self.settings.groq_model

    def _ensure_client(self) -> Any:
        if not self.is_configured:
            raise LLMConfigurationError()
        if self._client is not None:
            return self._client
        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError(
                "The groq package is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        try:
            self._client = Groq(
                api_key=self.settings.groq_api_key,
                timeout=self.settings.groq_timeout_seconds,
                max_retries=0,  # retries are handled here so backoff is explicit
            )
        except Exception as exc:
            raise LLMError("Could not initialise the Groq client.") from exc
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """Run a blocking chat completion and return the message text."""
        client = self._ensure_client()
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": (
                self.settings.groq_temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self.settings.groq_max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.settings.groq_max_retries + 1):
            try:
                response = client.chat.completions.create(**payload)
                choices = getattr(response, "choices", None) or []
                if not choices:
                    raise LLMError("The Groq API returned an empty response.")
                content = choices[0].message.content or ""
                if not content.strip():
                    raise LLMError("The Groq API returned an empty message.")
                return content.strip()
            except Exception as exc:
                mapped = self._map_exception(exc)
                last_error = mapped
                if not self._is_retryable(mapped) or attempt == self.settings.groq_max_retries:
                    raise mapped from exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Groq call failed (%s), retry %d/%d in %.1fs",
                    type(mapped).__name__,
                    attempt,
                    self.settings.groq_max_retries,
                    delay,
                )
                time.sleep(delay)

        raise last_error or LLMError()

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """Yield content deltas as they arrive.

        Connection setup errors surface before the first yield; mid-stream
        failures are mapped and re-raised so the caller can end the SSE stream
        with a proper error event.
        """
        client = self._ensure_client()
        try:
            stream = client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=(
                    self.settings.groq_temperature if temperature is None else temperature
                ),
                max_tokens=max_tokens or self.settings.groq_max_tokens,
                stream=True,
            )
        except Exception as exc:
            raise self._map_exception(exc) from exc

        try:
            for event in stream:
                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    yield piece
        except Exception as exc:
            raise self._map_exception(exc) from exc

    def health_check(self) -> tuple[bool, str]:
        """Cheap liveness probe. Returns (ok, human-readable detail)."""
        if not self.is_configured:
            return False, "GROQ_API_KEY is not set."
        try:
            self.complete(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=5,
                temperature=0.0,
            )
            return True, f"Groq reachable using model '{self.model}'."
        except LLMAuthenticationError:
            return False, "The configured GROQ_API_KEY was rejected."
        except Exception as exc:
            return False, f"Groq unreachable: {type(exc).__name__}"

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff with jitter to avoid retry storms."""
        return min(8.0, (2 ** (attempt - 1))) + random.uniform(0, 0.4)

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        return isinstance(error, (LLMRateLimitError, LLMTimeoutError)) or (
            isinstance(error, LLMError) and getattr(error, "details", {}).get("retryable")
        )

    def _map_exception(self, exc: Exception) -> Exception:
        """Translate provider/transport exceptions into app errors."""
        if isinstance(
            exc,
            (
                LLMAuthenticationError,
                LLMConfigurationError,
                LLMRateLimitError,
                LLMTimeoutError,
                LLMError,
            ),
        ):
            return exc

        name = type(exc).__name__
        message = str(exc)
        status = getattr(exc, "status_code", None)

        # Prefer typed groq exceptions when the SDK is present.
        try:
            import groq  # type: ignore

            if isinstance(exc, getattr(groq, "AuthenticationError", ())):
                return LLMAuthenticationError()
            if isinstance(exc, getattr(groq, "PermissionDeniedError", ())):
                return LLMAuthenticationError(
                    "The Groq API key does not have access to the configured model."
                )
            if isinstance(exc, getattr(groq, "RateLimitError", ())):
                return LLMRateLimitError()
            if isinstance(exc, getattr(groq, "APITimeoutError", ())):
                return LLMTimeoutError()
            if isinstance(exc, getattr(groq, "APIConnectionError", ())):
                return LLMError(
                    "Could not reach the Groq API. Check the network connection.",
                    details={"retryable": True},
                )
            if isinstance(exc, getattr(groq, "NotFoundError", ())):
                return LLMError(
                    f"The configured model '{self.model}' was not found. Update "
                    "GROQ_MODEL to a currently supported model.",
                )
            if isinstance(exc, getattr(groq, "BadRequestError", ())):
                if "context" in message.lower() or "token" in message.lower():
                    return LLMError(
                        "The request exceeded the model's context window. Reduce "
                        "RETRIEVAL_TOP_K or MAX_CONTEXT_CHARS.",
                    )
                return LLMError("The Groq API rejected the request payload.")
            if isinstance(exc, getattr(groq, "InternalServerError", ())):
                return LLMError(
                    "The Groq API reported a server error.", details={"retryable": True}
                )
        except ImportError:  # pragma: no cover - SDK absent
            pass

        # Fall back to status codes / exception names.
        if status == 401:
            return LLMAuthenticationError()
        if status == 403:
            return LLMAuthenticationError(
                "The Groq API key does not have access to the configured model."
            )
        if status == 404:
            return LLMError(
                f"The configured model '{self.model}' was not found. Update GROQ_MODEL."
            )
        if status == 429:
            return LLMRateLimitError()
        if status is not None and 500 <= int(status) < 600:
            return LLMError(
                "The Groq API reported a server error.", details={"retryable": True}
            )
        if "timeout" in name.lower() or "timeout" in message.lower():
            return LLMTimeoutError()
        if "connection" in name.lower():
            return LLMError(
                "Could not reach the Groq API. Check the network connection.",
                details={"retryable": True},
            )

        logger.error("Unmapped Groq error: %s", name)
        return LLMError("The language model provider returned an unexpected error.")
