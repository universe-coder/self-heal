"""OpenAI-compatible chat provider (used for openai, huggingface, ollama)."""

from __future__ import annotations

import logging
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from self_heal.llm.retry import RETRY_STATUS_CODES, retry_call

log = logging.getLogger(__name__)


def is_retryable_openai_error(e: Exception) -> bool:
    if isinstance(e, APITimeoutError | APIConnectionError):
        return True
    return isinstance(e, APIStatusError) and e.status_code in RETRY_STATUS_CODES


class OpenAICompatChatProvider:
    """Chat via the OpenAI Python SDK against any OpenAI-compatible base URL."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries

    def chat(self, system: str, user: str) -> str:
        return retry_call(
            lambda: self._chat_once(system, user),
            is_retryable=is_retryable_openai_error,
            max_retries=self._max_retries,
            label=f"chat({self._model})",
        )

    def _chat_once(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens,
            **kwargs,
        )
        if not resp.choices:
            return ""
        content = resp.choices[0].message.content
        return content if isinstance(content, str) else ""
