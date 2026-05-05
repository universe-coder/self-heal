"""Build NotificationEvent from raw audit context with redaction limits."""

from __future__ import annotations

from self_heal.config import NotificationsConfig
from self_heal.notifications.base import NotificationEvent

_MAX_DIFF_CHARS = 8000
_MAX_PREVIEW_CHARS = 2000


def _truncate_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n… ({len(lines) - max_lines} more lines omitted)"


def _truncate_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n… (truncated)"


def make_notification_event(
    ncfg: NotificationsConfig,
    *,
    kind: str,
    exception_type: str,
    message: str,
    traceback: str | None = None,
    diff_text: str | None = None,
    applied: bool | None = None,
    paths_touched: list[str] | None = None,
    raw_response_preview: str | None = None,
    heal_message: str | None = None,
    fingerprint_path: str | None = None,
) -> NotificationEvent:
    tb_excerpt: str | None = None
    if ncfg.include_traceback and traceback:
        tb_excerpt = _truncate_chars(
            _truncate_lines(traceback.strip(), ncfg.max_traceback_lines),
            _MAX_DIFF_CHARS,
        )

    diff_excerpt: str | None = None
    if ncfg.include_diff and diff_text and diff_text.strip():
        diff_excerpt = _truncate_chars(diff_text.strip(), _MAX_DIFF_CHARS)

    preview: str | None = None
    if raw_response_preview:
        preview = _truncate_chars(raw_response_preview, _MAX_PREVIEW_CHARS)

    return NotificationEvent(
        kind=kind,
        exception_type=exception_type,
        message=message,
        applied=applied,
        paths_touched=list(paths_touched or []),
        traceback_excerpt=tb_excerpt,
        diff_excerpt=diff_excerpt,
        raw_response_preview=preview,
        heal_message=heal_message,
        fingerprint_path=fingerprint_path,
    )
