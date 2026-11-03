"""LLM provider abstraction and OpenAI-compatible implementation.

The provider layer is the only place that performs outbound HTTP calls. It is
intentionally narrow:

* one method (``chat``) that takes a list of messages and returns a text
  completion;
* a strict timeout and max-token budget so a slow or verbose model cannot stall
  the service;
* temperature defaults to zero so the same evidence package always produces the
  same completion.

The ``provider`` is constructed from :class:`~app.core.config.Settings` and is
passed to the explanation generator at call time, so tests can supply a fake
provider without changing any downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import LLMSettings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    """Result returned by an LLM provider."""

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletion:
        """Send a chat conversation and return the model's reply."""


class OpenAICompatibleProvider(LLMProvider):
    """Provider for any OpenAI-compatible chat-completions endpoint.

    Works with OpenAI, Azure OpenAI, Ollama, vLLM, Together, Groq, LiteLLM, and
    any other server that implements the ``/v1/chat/completions`` API.
    """

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self._settings = settings or LLMSettings()
        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=self._settings.timeout,
            follow_redirects=True,
            headers={
                "Content-Type": "application/json",
                **self._auth_headers,
            },
        )

    @property
    def _auth_headers(self) -> dict[str, str]:
        if self._settings.api_key:
            return {"Authorization": f"Bearer {self._settings.api_key.get_secret_value()}"}
        return {}

    @property
    def model(self) -> str:
        return self._settings.model

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else self._settings.temperature,
            "max_tokens": max_tokens or self._settings.max_tokens,
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "llm_http_error",
                extra={
                    "status_code": exc.response.status_code,
                    "response_body": exc.response.text[:500],
                    "url": str(exc.request.url),
                },
            )
            raise LLMProviderError(
                f"LLM HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("llm_request_error", extra={"error": str(exc)})
            raise LLMProviderError(f"LLM request failed: {exc}") from exc

        return self._parse_response(response.json())

    @staticmethod
    def _parse_response(body: dict) -> ChatCompletion:
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = body.get("usage", {})
        return ChatCompletion(
            content=message.get("content", "") or "",
            model=body.get("model", ""),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class FakeProvider(LLMProvider):
    """Deterministic provider used by tests."""

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self._replies = replies or {}
        self.calls: list[list[ChatMessage]] = []

    def set_reply(self, key: str, content: str) -> None:
        self._replies[key] = content

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletion:
        self.calls.append(list(messages))
        combined = " ".join(m.content[:100] for m in messages)
        for key, reply in self._replies.items():
            if key.lower() in combined.lower():
                return ChatCompletion(content=reply, model="fake")
        return ChatCompletion(content="No matching reply configured.", model="fake")

    async def aclose(self) -> None:
        pass


class LLMProviderError(Exception):
    """Raised when the LLM provider cannot fulfill a request."""


def create_provider(settings: LLMSettings | None = None) -> LLMProvider:
    """Create the appropriate provider from settings."""
    resolved = settings or LLMSettings()
    return OpenAICompatibleProvider(resolved)


__all__ = [
    "ChatCompletion",
    "ChatMessage",
    "FakeProvider",
    "LLMProvider",
    "LLMProviderError",
    "OpenAICompatibleProvider",
    "create_provider",
]
