"""Minimal HTTP POST helpers (stdlib, no redirects, bounded timeout)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return None


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    extra_headers: dict[str, str] | None = None,
) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            if resp.status and resp.status >= 400:
                raise urllib.error.HTTPError(
                    url, resp.status, getattr(resp, "reason", ""), resp.headers, resp
                )
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            raise urllib.error.URLError("redirects are not allowed") from e
        raise


def post_form_urlencoded(
    url: str,
    fields: dict[str, str],
    *,
    timeout_s: float,
) -> None:
    from urllib.parse import urlencode

    data = urlencode(fields).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"}
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            if resp.status and resp.status >= 400:
                raise urllib.error.HTTPError(
                    url, resp.status, getattr(resp, "reason", ""), resp.headers, resp
                )
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            raise urllib.error.URLError("redirects are not allowed") from e
        raise


def post_bytes(
    url: str,
    body: bytes,
    *,
    timeout_s: float,
    extra_headers: dict[str, str] | None = None,
) -> None:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            if resp.status and resp.status >= 400:
                raise urllib.error.HTTPError(
                    url, resp.status, getattr(resp, "reason", ""), resp.headers, resp
                )
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            raise urllib.error.URLError("redirects are not allowed") from e
        raise
