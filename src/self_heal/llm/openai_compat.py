"""OpenAI-compatible chat provider (used for openai, huggingface, ollama)."""

from __future__ import annotations

import logging
import time
from typing import cast

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.5


class OpenAICompatChatProvider:
    """Chat via the OpenAI Python SDK against any OpenAI-compatible base URL."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def chat(self, system: str, user: str) -> str:
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._chat_once(system, user)
            except (RateLimitError, APITimeoutError, APIError) as e:
                last = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
                else:
                    raise
        if last:
            raise last
        raise RuntimeError("retry failed")

    def _chat_once(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content
        if not content:
            return ""
        return cast(str, content)
