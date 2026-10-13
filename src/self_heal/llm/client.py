"""Multi-provider chat client + OpenAI-compatible embeddings; never log API keys."""

from __future__ import annotations

import logging

from openai import OpenAI

from self_heal.config import SelfHealConfig, get_api_key, get_embedding_api_key, requires_api_key
from self_heal.llm.anthropic_provider import AnthropicChatProvider
from self_heal.llm.base import ChatProvider
from self_heal.llm.openai_compat import OpenAICompatChatProvider, is_retryable_openai_error
from self_heal.llm.retry import retry_call

log = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64

EMBEDDING_UNSUPPORTED_MSG = (
    "Anthropic does not provide an embeddings API; set [llm].embedding_base_url "
    "(and optionally embedding_api_key_env) to an OpenAI-compatible endpoint, or "
    "switch [llm].provider to openai/huggingface/ollama for indexing."
)


class EmbeddingsUnavailableError(RuntimeError):
    """Raised when the configuration has no usable embeddings endpoint."""


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
        llm = cfg.llm
        key = get_api_key(cfg)
        if llm.provider == "anthropic":
            return AnthropicChatProvider(
                base_url=llm.base_url,
                api_key=key or "",
                model=llm.model,
                max_tokens=llm.max_tokens,
                timeout_s=llm.timeout_s,
                temperature=llm.temperature,
                max_retries=llm.max_retries,
            )
        client = OpenAI(
            base_url=llm.base_url.rstrip("/"),
            api_key=key or "dummy",
            timeout=llm.timeout_s,
            max_retries=0,  # retries are handled by self_heal.llm.retry
        )
        return OpenAICompatChatProvider(
            client,
            model=llm.model,
            max_tokens=llm.max_tokens,
            temperature=llm.temperature,
            max_retries=llm.max_retries,
        )

    @staticmethod
    def _build_openai_client_for_embed(cfg: SelfHealConfig) -> OpenAI | None:
        llm = cfg.llm
        if not llm.embeddings_available:
            return None
        return OpenAI(
            base_url=llm.resolved_embedding_base_url.rstrip("/"),
            api_key=get_embedding_api_key(cfg) or "dummy",
            timeout=llm.timeout_s,
            max_retries=0,
        )

    @property
    def is_configured(self) -> bool:
        """Chat can be called: an API key is present or the provider needs none."""
        if not requires_api_key(self._cfg):
            return True
        return get_api_key(self._cfg) is not None

    @property
    def embeddings_available(self) -> bool:
        """An embeddings endpoint is configured and (if needed) has a key."""
        if self._openai_for_embed is None:
            return False
        if self._cfg.llm.embedding_base_url:
            return True  # dedicated endpoint; key optional (e.g. Ollama)
        return self.is_configured

    def chat(self, system: str, user: str) -> str:
        if requires_api_key(self._cfg) and not get_api_key(self._cfg):
            raise RuntimeError(f"Missing API key: set {self._cfg.llm.api_key_env}")
        return self._chat_provider.chat(system, user)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._openai_for_embed is None:
            raise EmbeddingsUnavailableError(EMBEDDING_UNSUPPORTED_MSG)
        if not self.embeddings_available:
            env = self._cfg.llm.resolved_embedding_api_key_env
            raise EmbeddingsUnavailableError(f"Missing API key for embeddings: set {env}")
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            out.extend(
                retry_call(
                    lambda b=batch: self._embed_once(b),  # type: ignore[misc]
                    is_retryable=is_retryable_openai_error,
                    max_retries=self._cfg.llm.max_retries,
                    label=f"embed({self._cfg.llm.embedding_model})",
                )
            )
        return out

    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        assert self._openai_for_embed is not None
        r = self._openai_for_embed.embeddings.create(
            model=self._cfg.llm.embedding_model,
            input=texts,
        )
        return [list(item.embedding) for item in sorted(r.data, key=lambda x: x.index)]
