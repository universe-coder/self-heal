"""OpenAI-compatible chat and embeddings with retries; never log API keys."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from self_heal.config import SelfHealConfig, get_api_key

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.5


class LLMClient:
    def __init__(self, cfg: SelfHealConfig) -> None:
        self._cfg = cfg
        key = get_api_key(cfg)
        base = cfg.llm.base_url.rstrip("/")
        self._client = OpenAI(
            base_url=base,
            api_key=key or "dummy",
            timeout=cfg.llm.timeout_s,
        )
        if not key:
            log.warning(
                "API key not set (env %s). LLM calls will fail until set.",
                cfg.llm.api_key_env,
            )

    @property
    def is_configured(self) -> bool:
        return get_api_key(self._cfg) is not None

    def chat(self, system: str, user: str) -> str:
        return cast(
            str,
            self._with_retry(lambda: self._chat_once(system, user)),
        )

    def _chat_once(self, system: str, user: str) -> str:
        key = get_api_key(self._cfg)
        if not key:
            msg = f"Missing API key: set {self._cfg.llm.api_key_env}"
            raise RuntimeError(msg)
        resp = self._client.chat.completions.create(
            model=self._cfg.llm.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._cfg.llm.max_tokens,
        )
        choice = resp.choices[0]
        content = choice.message.content
        if not content:
            return ""
        return cast(str, content)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return cast(list[list[float]], self._with_retry(lambda: self._embed_once(texts)))

    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        key = get_api_key(self._cfg)
        if not key:
            msg = f"Missing API key: set {self._cfg.llm.api_key_env}"
            raise RuntimeError(msg)
        out: list[list[float]] = []
        # batch in chunks of 64
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = self._client.embeddings.create(
                model=self._cfg.llm.embedding_model,
                input=batch,
            )
            for item in sorted(r.data, key=lambda x: x.index):
                out.append(list(item.embedding))
        return out

    def _with_retry(self, fn: Any) -> Any:
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return fn()
            except (RateLimitError, APITimeoutError, APIError) as e:
                last = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
                else:
                    raise
        if last:
            raise last
        raise RuntimeError("retry failed")
