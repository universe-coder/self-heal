"""Generic HTTPS webhook with optional HMAC-SHA256 body signature."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

from self_heal.config import WebhookNotifConfig
from self_heal.notifications.base import NotificationEvent
from self_heal.notifications.http_util import post_bytes
from self_heal.notifications.url_validate import validate_webhook_url


def build_webhook_payload(event: NotificationEvent) -> dict[str, object]:
    return dict(event.model_dump(mode="json"))


def sign_body(body: bytes, secret: str, timestamp: str) -> str:
    msg = timestamp.encode("utf-8") + b"." + body
    digest = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return digest


class WebhookNotifier:
    name = "webhook"

    def __init__(self, cfg: WebhookNotifConfig, *, timeout_s: float) -> None:
        self._cfg = cfg
        self._timeout_s = timeout_s

    def send(self, event: NotificationEvent) -> None:
        url = os.environ.get(self._cfg.url_env, "").strip()
        if not url:
            raise RuntimeError(f"missing env {self._cfg.url_env}")

        validate_webhook_url(url, allow_insecure=self._cfg.allow_insecure)

        payload = build_webhook_payload(event)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ts = str(int(time.time()))
        headers: dict[str, str] = {"X-SelfHeal-Timestamp": ts}

        secret = os.environ.get(self._cfg.signing_secret_env, "").strip()
        if secret:
            sig = sign_body(body, secret, ts)
            headers["X-SelfHeal-Signature"] = f"sha256={sig}"

        post_bytes(url, body, timeout_s=self._timeout_s, extra_headers=headers)
