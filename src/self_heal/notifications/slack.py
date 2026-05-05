"""Slack incoming webhook."""

from __future__ import annotations

import os

from self_heal.config import SlackNotifConfig
from self_heal.notifications.base import NotificationEvent
from self_heal.notifications.http_util import post_json
from self_heal.notifications.url_validate import validate_webhook_url


class SlackNotifier:
    name = "slack"

    def __init__(self, cfg: SlackNotifConfig, *, timeout_s: float) -> None:
        self._cfg = cfg
        self._timeout_s = timeout_s

    def send(self, event: NotificationEvent) -> None:
        wh_url = os.environ.get(self._cfg.webhook_url_env, "").strip()
        if not wh_url:
            raise RuntimeError(f"missing env {self._cfg.webhook_url_env}")

        validate_webhook_url(wh_url, allow_insecure=False)

        summary = f"*{event.kind}* — `{event.exception_type}`: {event.message[:500]}"
        blocks: list[dict[str, object]] = [
            {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
        ]
        if event.applied is not None:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Applied:* `{event.applied}`"},
                }
            )
        if event.paths_touched:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Paths:* " + ", ".join(f"`{p}`" for p in event.paths_touched[:30]),
                    },
                }
            )
        if event.heal_message:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "plain_text", "text": event.heal_message[:3000]},
                }
            )
        if event.traceback_excerpt:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"```{event.traceback_excerpt[:2900]}```",
                    },
                }
            )
        if event.diff_excerpt:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"```{event.diff_excerpt[:2900]}```"},
                }
            )

        payload = {
            "text": summary,
            "blocks": blocks,
            "attachments": [
                {
                    "color": "#764FA5",
                    "fields": [
                        {"title": "kind", "value": event.kind, "short": True},
                        {"title": "exception", "value": event.exception_type, "short": True},
                    ],
                    "footer": "self-heal",
                    "ts": int(event.ts.timestamp()),
                }
            ],
        }
        post_json(wh_url, payload, timeout_s=self._timeout_s)
