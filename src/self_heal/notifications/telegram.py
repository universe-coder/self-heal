"""Telegram Bot API sendMessage."""

from __future__ import annotations

import html
import logging
import os
import urllib.parse

from self_heal.config import TelegramNotifConfig
from self_heal.notifications.base import NotificationEvent
from self_heal.notifications.http_util import post_form_urlencoded

log = logging.getLogger(__name__)


def _format_text(ev: NotificationEvent, cfg: TelegramNotifConfig) -> str:
    lines = [
        f"<b>self-heal</b> <code>{html.escape(ev.kind)}</code>",
        f"<b>Exception</b>: <code>{html.escape(ev.exception_type)}</code>",
        f"<b>Message</b>: {html.escape(ev.message[:2000])}",
    ]
    if ev.applied is not None:
        lines.append(f"<b>Applied</b>: {ev.applied}")
    if ev.paths_touched:
        lines.append("<b>Paths</b>: " + html.escape(", ".join(ev.paths_touched[:50])))
    if ev.heal_message:
        lines.append(f"<b>Heal</b>: {html.escape(ev.heal_message[:1500])}")
    if ev.traceback_excerpt:
        lines.append("<pre>" + html.escape(ev.traceback_excerpt[:3500]) + "</pre>")
    if ev.diff_excerpt:
        lines.append("<pre>" + html.escape(ev.diff_excerpt[:3500]) + "</pre>")
    if ev.raw_response_preview:
        lines.append("<pre>" + html.escape(ev.raw_response_preview[:1500]) + "</pre>")
    return "\n".join(lines)


def _format_plain(ev: NotificationEvent) -> str:
    parts = [
        f"self-heal {ev.kind}",
        f"Exception: {ev.exception_type}",
        f"Message: {ev.message}",
    ]
    if ev.applied is not None:
        parts.append(f"Applied: {ev.applied}")
    if ev.paths_touched:
        parts.append("Paths: " + ", ".join(ev.paths_touched))
    if ev.heal_message:
        parts.append(f"Heal: {ev.heal_message}")
    if ev.traceback_excerpt:
        parts.append(ev.traceback_excerpt)
    if ev.diff_excerpt:
        parts.append(ev.diff_excerpt)
    return "\n".join(parts)


class TelegramNotifier:
    name = "telegram"

    def __init__(self, cfg: TelegramNotifConfig, *, timeout_s: float) -> None:
        self._cfg = cfg
        self._timeout_s = timeout_s

    def send(self, event: NotificationEvent) -> None:
        token = os.environ.get(self._cfg.bot_token_env, "").strip()
        if not token:
            raise RuntimeError(f"missing env {self._cfg.bot_token_env}")
        chat_id = self._cfg.chat_id.strip()
        if not chat_id:
            raise RuntimeError("notifications.telegram.chat_id is empty")

        url = f"https://api.telegram.org/bot{urllib.parse.quote(token, safe=':/')}/sendMessage"
        fields: dict[str, str] = {"chat_id": chat_id}
        if self._cfg.parse_mode == "HTML":
            fields["text"] = _format_text(event, self._cfg)
            fields["parse_mode"] = "HTML"
        elif self._cfg.parse_mode == "MarkdownV2":
            # Minimal escaping for MarkdownV2 is error-prone; fall back to plain
            log.warning("MarkdownV2 not fully escaped; using plain text for Telegram")
            fields["text"] = _format_plain(event)
        else:
            fields["text"] = _format_plain(event)

        post_form_urlencoded(url, fields, timeout_s=self._timeout_s)
