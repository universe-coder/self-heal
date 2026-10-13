"""Native Anthropic Messages API chat provider (stdlib urllib, no extra deps)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from self_heal.llm.retry import RETRY_STATUS_CODES, retry_call

log = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"
_REDIRECTS = frozenset({301, 302, 303, 307, 308})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return None


class AnthropicAPIError(RuntimeError):
    """Non-2xx response from the Messages API, with the status and a body preview."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Anthropic API error {status}: {detail}")
        self.status = status
        self.detail = detail


def is_retryable_anthropic_error(e: Exception) -> bool:
    if isinstance(e, AnthropicAPIError):
        return e.status in RETRY_STATUS_CODES
    if isinstance(e, urllib.error.HTTPError):
        return e.code in RETRY_STATUS_CODES
    # Network-level failures (DNS, connection reset, timeouts) are worth retrying.
    return isinstance(e, urllib.error.URLError | TimeoutError | ConnectionError)


class AnthropicChatProvider:
    """Calls POST {base_url}/messages with x-api-key + anthropic-version headers."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        timeout_s: float,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._temperature = temperature
        self._max_retries = max_retries

    def chat(self, system: str, user: str) -> str:
        return retry_call(
            lambda: self._chat_once(system, user),
            is_retryable=is_retryable_anthropic_error,
            max_retries=self._max_retries,
            label=f"chat({self._model})",
        )

    def _chat_once(self, system: str, user: str) -> str:
        url = f"{self._base_url}/messages"
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "content-type": "application/json; charset=utf-8",
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "accept": "application/json",
        }
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            with opener.open(req, timeout=self._timeout_s) as resp:
                status = getattr(resp, "status", 200)
                payload = resp.read()
                if status and status >= 400:
                    raise AnthropicAPIError(status, _preview(payload))
        except urllib.error.HTTPError as e:
            if e.code in _REDIRECTS:
                raise urllib.error.URLError("redirects are not allowed") from e
            try:
                detail = _preview(e.read())
            except Exception:  # noqa: BLE001
                detail = str(e.reason)
            raise AnthropicAPIError(e.code, detail) from e
        return _extract_text(payload)


def _preview(payload: bytes, limit: int = 300) -> str:
    text = payload.decode("utf-8", errors="replace").strip()
    try:
        data = json.loads(text)
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            text = err["message"]
    except (ValueError, AttributeError):
        pass
    return text[:limit]


def _extract_text(payload: bytes) -> str:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RuntimeError("Anthropic response is not valid JSON") from e
    content = data.get("content") if isinstance(data, dict) else None
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
