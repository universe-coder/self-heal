"""Telegram Bot API sendMessage."""

from __future__ import annotations

import html
import logging
import os
import re
import urllib.parse

from self_heal.config import TelegramNotifConfig
from self_heal.notifications.base import NotificationEvent
from self_heal.notifications.http_util import post_form_urlencoded

log = logging.getLogger(__name__)

#: Telegram rejects messages longer than 4096 characters; keep a margin for markup.
TELEGRAM_MAX_CHARS = 4096
_BUDGET = 4000
_MIN_BLOCK = 80

_MD2_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")
_MD2_CODE_SPECIAL = re.compile(r"([`\\])")


def escape_markdown_v2(text: str) -> str:
    """Escape text for Telegram MarkdownV2 (outside code spans)."""
    return _MD2_SPECIAL.sub(r"\\\1", text)


def _escape_markdown_v2_code(text: str) -> str:
    return _MD2_CODE_SPECIAL.sub(r"\\\1", text)


def _assemble(parts: list[tuple[str, bool, str, str]], budget: int) -> str:
    """Join ``(text, is_block, open_tag, close_tag)`` parts without exceeding ``budget``.

    Plain parts are kept whole or dropped; block parts (tracebacks, diffs) are
    truncated to whatever space is left so the message always stays well-formed.
    """
    out: list[str] = []
    used = 0
    for text, is_block, open_tag, close_tag in parts:
        full = f"{open_tag}{text}{close_tag}"
        need = len(full) + (1 if out else 0)
        if used + need <= budget:
            out.append(full)
            used += need
            continue
        if not is_block:
            continue
        room = budget - used - len(open_tag) - len(close_tag) - 2
        if room >= _MIN_BLOCK:
            out.append(f"{open_tag}{text[: room - 1]}…{close_tag}")
            used = budget
        break
    return "\n".join(out)


def _format_html(ev: NotificationEvent) -> str:
    e = html.escape
    parts: list[tuple[str, bool, str, str]] = [
        (f"<b>self-heal</b> <code>{e(ev.kind)}</code>", False, "", ""),
        (f"<b>Exception</b>: <code>{e(ev.exception_type)}</code>", False, "", ""),
        (f"<b>Message</b>: {e(ev.message[:2000])}", False, "", ""),
    ]
    if ev.applied is not None:
        parts.append((f"<b>Applied</b>: {ev.applied}", False, "", ""))
    if ev.paths_touched:
        parts.append(("<b>Paths</b>: " + e(", ".join(ev.paths_touched[:50])), False, "", ""))
    if ev.heal_message:
        parts.append((f"<b>Heal</b>: {e(ev.heal_message[:1500])}", False, "", ""))
    if ev.traceback_excerpt:
        parts.append((e(ev.traceback_excerpt), True, "<pre>", "</pre>"))
    if ev.diff_excerpt:
        parts.append((e(ev.diff_excerpt), True, "<pre>", "</pre>"))
    if ev.raw_response_preview:
        parts.append((e(ev.raw_response_preview), True, "<pre>", "</pre>"))
    return _assemble(parts, _BUDGET)


def _format_markdown_v2(ev: NotificationEvent) -> str:
    e = escape_markdown_v2
    c = _escape_markdown_v2_code
    parts: list[tuple[str, bool, str, str]] = [
        (f"*self\\-heal* `{c(ev.kind)}`", False, "", ""),
        (f"*Exception*: `{c(ev.exception_type)}`", False, "", ""),
        (f"*Message*: {e(ev.message[:2000])}", False, "", ""),
    ]
    if ev.applied is not None:
        parts.append((f"*Applied*: {e(str(ev.applied))}", False, "", ""))
    if ev.paths_touched:
        parts.append(("*Paths*: " + e(", ".join(ev.paths_touched[:50])), False, "", ""))
    if ev.heal_message:
        parts.append((f"*Heal*: {e(ev.heal_message[:1500])}", False, "", ""))
    if ev.traceback_excerpt:
        parts.append((c(ev.traceback_excerpt), True, "```\n", "\n```"))
    if ev.diff_excerpt:
        parts.append((c(ev.diff_excerpt), True, "```diff\n", "\n```"))
    if ev.raw_response_preview:
        parts.append((c(ev.raw_response_preview), True, "```\n", "\n```"))
    return _assemble(parts, _BUDGET)


def _format_plain(ev: NotificationEvent) -> str:
    parts: list[tuple[str, bool, str, str]] = [
        (f"self-heal {ev.kind}", False, "", ""),
        (f"Exception: {ev.exception_type}", False, "", ""),
        (f"Message: {ev.message}", False, "", ""),
    ]
    if ev.applied is not None:
        parts.append((f"Applied: {ev.applied}", False, "", ""))
    if ev.paths_touched:
        parts.append(("Paths: " + ", ".join(ev.paths_touched), False, "", ""))
    if ev.heal_message:
        parts.append((f"Heal: {ev.heal_message}", False, "", ""))
    if ev.traceback_excerpt:
        parts.append((ev.traceback_excerpt, True, "", ""))
    if ev.diff_excerpt:
        parts.append((ev.diff_excerpt, True, "", ""))
    return _assemble(parts, _BUDGET)


def format_message(ev: NotificationEvent, parse_mode: str) -> str:
    if parse_mode == "HTML":
        return _format_html(ev)
    if parse_mode == "MarkdownV2":
        return _format_markdown_v2(ev)
    return _format_plain(ev)


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
        fields: dict[str, str] = {
            "chat_id": chat_id,
            "text": format_message(event, self._cfg.parse_mode),
            "disable_web_page_preview": "true",
        }
        if self._cfg.parse_mode in ("HTML", "MarkdownV2"):
            fields["parse_mode"] = self._cfg.parse_mode
        post_form_urlencoded(url, fields, timeout_s=self._timeout_s)
