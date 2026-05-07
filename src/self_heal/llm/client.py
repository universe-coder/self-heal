"""Multi-provider chat client + OpenAI-compatible embeddings; never log API keys."""

from __future__ import annotations

import logging
import time
from typing import cast

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from self_heal.config import SelfHealConfig, get_api_key, requires_api_key
from self_heal.llm.anthropic_provider import AnthropicChatProvider
from self_heal.llm.base import ChatProvider
from self_heal.llm.openai_compat import OpenAICompatChatProvider

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.5

EMBEDDING_UNSUPPORTED_MSG = (
    "Anthropic does not provide an embeddings API; "
    "switch [llm].provider to openai/huggingface/ollama for indexing."
)


class LLMClient:
    def __init__(self, cfg: SelfHealConfig) -> None:
        self._cfg = cfg
        self._chat_provider: ChatProvider = self._build_chat_provider(cfg)
        self._openai_for_embed: OpenAI | None = self._build_openai_client_for_embed(cfg)
        if requires_api_key(cfg) and not get_api_key(cfg):
            log.warning(
                "API key not set (env %s). LLM calls will fail until set.",
                cfg.llm.api_key_env,
            )

    @staticmethod
    def _build_chat_provider(cfg: SelfHealConfig) -> ChatProvider:
        provider = cfg.llm.provider
        key = get_api_key(cfg)
        if provider == "anthropic":
            return AnthropicChatProvider(
                base_url=cfg.llm.base_url,
                api_key=key or "",
                model=cfg.llm.model,
                max_tokens=cfg.llm.max_tokens,
                timeout_s=cfg.llm.timeout_s,
            )
        client = OpenAI(
            base_url=cfg.llm.base_url.rstrip("/"),
            api_key=key or "dummy",
            timeout=cfg.llm.timeout_s,
        )
        return OpenAICompatChatProvider(
            client,
            model=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
        )

    @staticmethod
    def _build_openai_client_for_embed(cfg: SelfHealConfig) -> OpenAI | None:
        if cfg.llm.provider == "anthropic":
            return None
        key = get_api_key(cfg)
        return OpenAI(
            base_url=cfg.llm.base_url.rstrip("/"),
            api_key=key or "dummy",
            timeout=cfg.llm.timeout_s,
        )

    @property
    def is_configured(self) -> bool:
        if not requires_api_key(self._cfg):
            return True
        return get_api_key(self._cfg) is not None

    def chat(self, system: str, user: str) -> str:
        if requires_api_key(self._cfg) and not get_api_key(self._cfg):
            msg = f"Missing API key: set {self._cfg.llm.api_key_env}"
            raise RuntimeError(msg)
        return self._chat_provider.chat(system, user)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._cfg.llm.provider == "anthropic" or self._openai_for_embed is None:
            raise RuntimeError(EMBEDDING_UNSUPPORTED_MSG)
        if requires_api_key(self._cfg) and not get_api_key(self._cfg):
            msg = f"Missing API key: set {self._cfg.llm.api_key_env}"
            raise RuntimeError(msg)
        return cast(list[list[float]], self._with_retry(lambda: self._embed_once(texts)))

    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        assert self._openai_for_embed is not None
        out: list[list[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = self._openai_for_embed.embeddings.create(
                model=self._cfg.llm.embedding_model,
                input=batch,
            )
            for item in sorted(r.data, key=lambda x: x.index):
                out.append(list(item.embedding))
        return out

    @staticmethod
    def _with_retry(fn: object) -> object:
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return fn()  # type: ignore[operator]
            except (RateLimitError, APITimeoutError, APIError) as e:
                last = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
                else:
                    raise
        if last:
            raise last
        raise RuntimeError("retry failed")
