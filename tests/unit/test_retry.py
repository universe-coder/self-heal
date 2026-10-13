"""Retry policy shared by the LLM providers."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from self_heal.config import LLMConfig, SelfHealConfig
from self_heal.llm.anthropic_provider import AnthropicAPIError, is_retryable_anthropic_error
from self_heal.llm.client import EmbeddingsUnavailableError, LLMClient
from self_heal.llm.openai_compat import OpenAICompatChatProvider, is_retryable_openai_error
from self_heal.llm.retry import retry_call


def _status_error(cls: type[APIStatusError], status: int) -> APIStatusError:
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    resp = httpx.Response(status, request=req)
    return cls("err", response=resp, body=None)


def test_retry_call_retries_then_succeeds() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("slow")
        return "ok"

    out = retry_call(fn, is_retryable=lambda e: True, max_retries=3, sleep=sleeps.append)
    assert out == "ok"
    assert calls["n"] == 3
    assert sleeps == [1.5, 3.0]  # exponential backoff


def test_retry_call_gives_up_after_max_retries() -> None:
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise TimeoutError("slow")

    with pytest.raises(TimeoutError):
        retry_call(fn, is_retryable=lambda e: True, max_retries=2, sleep=lambda s: None)
    assert calls["n"] == 3


def test_retry_call_does_not_retry_non_retryable() -> None:
    calls = {"n": 0}

    def fn() -> None:
        calls["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        retry_call(fn, is_retryable=lambda e: False, max_retries=5, sleep=lambda s: None)
    assert calls["n"] == 1


def test_openai_error_classification() -> None:
    assert is_retryable_openai_error(_status_error(RateLimitError, 429))
    assert is_retryable_openai_error(_status_error(APIStatusError, 503))
    assert is_retryable_openai_error(
        APIConnectionError(request=httpx.Request("POST", "https://api.test"))
    )
    assert not is_retryable_openai_error(_status_error(AuthenticationError, 401))
    assert not is_retryable_openai_error(_status_error(APIStatusError, 400))
    assert not is_retryable_openai_error(ValueError("x"))


def test_anthropic_error_classification() -> None:
    assert is_retryable_anthropic_error(AnthropicAPIError(529, "overloaded"))
    assert is_retryable_anthropic_error(AnthropicAPIError(429, "rate"))
    assert not is_retryable_anthropic_error(AnthropicAPIError(401, "auth"))
    assert is_retryable_anthropic_error(urllib.error.URLError("dns"))
    assert not is_retryable_anthropic_error(RuntimeError("not json"))


def test_openai_chat_does_not_retry_auth_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _status_error(AuthenticationError, 401)
    monkeypatch.setattr("self_heal.llm.retry.time.sleep", lambda s: None)
    p = OpenAICompatChatProvider(fake_client, model="m", max_tokens=8, max_retries=3)
    with pytest.raises(AuthenticationError):
        p.chat("s", "u")
    assert fake_client.chat.completions.create.call_count == 1


def test_openai_chat_passes_temperature_only_when_set() -> None:
    fake_choice = MagicMock()
    fake_choice.message.content = "x"
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp

    OpenAICompatChatProvider(fake_client, model="m", max_tokens=8).chat("s", "u")
    assert "temperature" not in fake_client.chat.completions.create.call_args.kwargs

    OpenAICompatChatProvider(fake_client, model="m", max_tokens=8, temperature=0.0).chat("s", "u")
    assert fake_client.chat.completions.create.call_args.kwargs["temperature"] == 0.0


def test_llm_client_disables_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with patch("self_heal.llm.client.OpenAI") as fake_openai:
        fake_openai.return_value = MagicMock()
        LLMClient(SelfHealConfig())
    for call in fake_openai.call_args_list:
        assert call.kwargs["max_retries"] == 0


def test_llm_client_separate_embedding_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    cfg = SelfHealConfig(
        llm=LLMConfig(
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            embedding_base_url="http://localhost:11434/v1/",
            embedding_api_key_env="OLLAMA_API_KEY",
            embedding_model="nomic-embed-text",
        )
    )
    with patch("self_heal.llm.client.OpenAI") as fake_openai:
        embed_client = MagicMock()
        item = MagicMock()
        item.index = 0
        item.embedding = [0.1, 0.2]
        embed_client.embeddings.create.return_value = MagicMock(data=[item])
        fake_openai.return_value = embed_client
        client = LLMClient(cfg)

    assert fake_openai.call_args.kwargs["base_url"] == "http://localhost:11434/v1"
    assert fake_openai.call_args.kwargs["api_key"] == "dummy"
    assert client.embeddings_available
    assert client.embed(["hello"]) == [[0.1, 0.2]]
    assert embed_client.embeddings.create.call_args.kwargs["model"] == "nomic-embed-text"


def test_llm_client_embed_batches_and_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with patch("self_heal.llm.client.OpenAI") as fake_openai:
        embed_client = MagicMock()

        def create(model: str, input: list[str]) -> MagicMock:  # noqa: A002
            items = []
            for i, text in enumerate(input):
                it = MagicMock()
                it.index = i
                it.embedding = [float(len(text))]
                items.append(it)
            return MagicMock(data=list(reversed(items)))

        embed_client.embeddings.create.side_effect = create
        fake_openai.return_value = embed_client
        client = LLMClient(SelfHealConfig())

    texts = ["a" * (i + 1) for i in range(70)]
    out = client.embed(texts)
    assert out == [[float(i + 1)] for i in range(70)]
    assert embed_client.embeddings.create.call_count == 2  # 64 + 6


def test_llm_client_embed_unavailable_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    client = LLMClient(SelfHealConfig(llm=LLMConfig(provider="anthropic")))
    assert not client.embeddings_available
    with pytest.raises(EmbeddingsUnavailableError, match="embedding_base_url"):
        client.embed(["x"])
