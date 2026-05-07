"""Unit tests for LLM providers and provider selection in LLMClient."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from self_heal.config import LLMConfig, SelfHealConfig
from self_heal.llm.anthropic_provider import (
    ANTHROPIC_VERSION,
    AnthropicChatProvider,
)
from self_heal.llm.client import EMBEDDING_UNSUPPORTED_MSG, LLMClient
from self_heal.llm.openai_compat import OpenAICompatChatProvider


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status
        self.headers: dict[str, str] = {}
        self.reason = ""

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeOpener:
    """Records the request and returns canned responses (or raises)."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []
        self.timeouts: list[float] = []

    def open(self, req: Any, timeout: float) -> Any:  # noqa: A003
        self.requests.append(req)
        self.timeouts.append(timeout)
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _anthropic_success_response(text: str = "ok") -> _FakeResponse:
    payload = json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
        }
    ).encode("utf-8")
    return _FakeResponse(payload, status=200)


def _http_error(
    code: int,
    url: str = "https://api.anthropic.com/v1/messages",
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(b""))


def test_openai_compat_chat_returns_message_content() -> None:
    fake_choice = MagicMock()
    fake_choice.message.content = "hello world"
    fake_resp = MagicMock()
    fake_resp.choices = [fake_choice]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp

    p = OpenAICompatChatProvider(fake_client, model="gpt-4o-mini", max_tokens=128)
    out = p.chat("sys", "usr")

    assert out == "hello world"
    args, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["max_tokens"] == 128
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_anthropic_provider_sends_correct_request_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _FakeOpener([_anthropic_success_response("hi there")])
    monkeypatch.setattr(
        "self_heal.llm.anthropic_provider.urllib.request.build_opener",
        lambda *a, **kw: opener,
    )

    p = AnthropicChatProvider(
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test",
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        timeout_s=10.0,
    )
    out = p.chat("you are helpful", "say hi")

    assert out == "hi there"
    assert len(opener.requests) == 1
    req = opener.requests[0]
    assert req.full_url == "https://api.anthropic.com/v1/messages"
    assert req.get_method() == "POST"
    headers_lc = {k.lower(): v for k, v in req.header_items()}
    assert headers_lc["x-api-key"] == "sk-ant-test"
    assert headers_lc["anthropic-version"] == ANTHROPIC_VERSION
    assert "application/json" in headers_lc["content-type"]
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "claude-3-5-sonnet-20241022"
    assert body["max_tokens"] == 256
    assert body["system"] == "you are helpful"
    assert body["messages"] == [{"role": "user", "content": "say hi"}]


def test_anthropic_provider_concatenates_text_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "content": [
                {"type": "text", "text": "part1 "},
                {"type": "tool_use", "name": "ignored"},
                {"type": "text", "text": "part2"},
            ]
        }
    ).encode("utf-8")
    opener = _FakeOpener([_FakeResponse(payload, status=200)])
    monkeypatch.setattr(
        "self_heal.llm.anthropic_provider.urllib.request.build_opener",
        lambda *a, **kw: opener,
    )

    p = AnthropicChatProvider(
        base_url="https://api.anthropic.com/v1",
        api_key="k",
        model="m",
        max_tokens=64,
        timeout_s=5.0,
    )
    assert p.chat("s", "u") == "part1 part2"


def test_anthropic_provider_retries_on_429_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _FakeOpener([_http_error(429), _anthropic_success_response("done")])
    monkeypatch.setattr(
        "self_heal.llm.anthropic_provider.urllib.request.build_opener",
        lambda *a, **kw: opener,
    )
    monkeypatch.setattr("self_heal.llm.anthropic_provider.time.sleep", lambda *_a, **_kw: None)

    p = AnthropicChatProvider(
        base_url="https://api.anthropic.com/v1",
        api_key="k",
        model="m",
        max_tokens=8,
        timeout_s=1.0,
    )
    assert p.chat("s", "u") == "done"
    assert len(opener.requests) == 2


def test_anthropic_provider_does_not_retry_on_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _FakeOpener([_http_error(400)])
    monkeypatch.setattr(
        "self_heal.llm.anthropic_provider.urllib.request.build_opener",
        lambda *a, **kw: opener,
    )
    monkeypatch.setattr("self_heal.llm.anthropic_provider.time.sleep", lambda *_a, **_kw: None)

    p = AnthropicChatProvider(
        base_url="https://api.anthropic.com/v1",
        api_key="k",
        model="m",
        max_tokens=8,
        timeout_s=1.0,
    )
    with pytest.raises(urllib.error.HTTPError):
        p.chat("s", "u")
    assert len(opener.requests) == 1


def test_llm_client_uses_anthropic_provider_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    cfg = SelfHealConfig(
        llm=LLMConfig(
            provider="anthropic",
            base_url="https://api.anthropic.com/v1",
            model="claude-3-5-sonnet-20241022",
            api_key_env="ANTHROPIC_API_KEY",
        )
    )
    with patch("self_heal.llm.client.OpenAI") as fake_openai:
        client = LLMClient(cfg)
        fake_openai.assert_not_called()

    assert isinstance(client._chat_provider, AnthropicChatProvider)
    assert client._openai_for_embed is None
    assert client.is_configured is True


def test_llm_client_embed_raises_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    cfg = SelfHealConfig(llm=LLMConfig(provider="anthropic", api_key_env="ANTHROPIC_API_KEY"))
    client = LLMClient(cfg)
    with pytest.raises(RuntimeError, match="embeddings"):
        client.embed(["text"])


def test_llm_client_embed_message_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    cfg = SelfHealConfig(llm=LLMConfig(provider="anthropic", api_key_env="ANTHROPIC_API_KEY"))
    client = LLMClient(cfg)
    with pytest.raises(RuntimeError) as exc:
        client.embed(["text"])
    assert str(exc.value) == EMBEDDING_UNSUPPORTED_MSG


def test_llm_client_ollama_is_configured_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    cfg = SelfHealConfig(
        llm=LLMConfig(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.1",
            api_key_env="OLLAMA_API_KEY",
        )
    )
    with patch("self_heal.llm.client.OpenAI") as fake_openai:
        fake_openai.return_value = MagicMock()
        client = LLMClient(cfg)
    assert client.is_configured is True


def test_llm_client_huggingface_uses_openai_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    cfg = SelfHealConfig(
        llm=LLMConfig(
            provider="huggingface",
            base_url="https://router.huggingface.co/v1",
            model="meta-llama/Llama-3.1-8B-Instruct",
            api_key_env="HF_TOKEN",
        )
    )
    with patch("self_heal.llm.client.OpenAI") as fake_openai:
        fake_openai.return_value = MagicMock()
        client = LLMClient(cfg)
    assert isinstance(client._chat_provider, OpenAICompatChatProvider)
    assert client._openai_for_embed is not None
    assert client.is_configured is True


def test_llm_client_chat_requires_api_key_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = SelfHealConfig(llm=LLMConfig(provider="openai", api_key_env="OPENAI_API_KEY"))
    with patch("self_heal.llm.client.OpenAI") as fake_openai:
        fake_openai.return_value = MagicMock()
        client = LLMClient(cfg)
    assert client.is_configured is False
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client.chat("s", "u")
