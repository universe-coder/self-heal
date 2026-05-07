"""Native Anthropic Messages API chat provider (stdlib urllib, no extra deps)."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, cast

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.5
ANTHROPIC_VERSION = "2023-06-01"
RETRY_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return None


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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s

    def chat(self, system: str, user: str) -> str:
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._chat_once(system, user)
            except urllib.error.HTTPError as e:
                last = e
                if e.code in RETRY_STATUS_CODES and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
                    continue
                raise
            except urllib.error.URLError as e:
                last = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_S * (2**attempt))
                    continue
                raise
        if last:
            raise last
        raise RuntimeError("retry failed")

    def _chat_once(self, system: str, user: str) -> str:
        url = f"{self._base_url}/messages"
        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
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
                if status and status >= 400:
                    raise urllib.error.HTTPError(
                        url, status, getattr(resp, "reason", ""), resp.headers, resp
                    )
                payload = resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                raise urllib.error.URLError("redirects are not allowed") from e
            raise
        return _extract_text(payload)


def _extract_text(payload: bytes) -> str:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RuntimeError("Anthropic response is not valid JSON") from e
    content = data.get("content")
    if not isinstance(content, list) or not content:
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return cast(str, "".join(parts))
