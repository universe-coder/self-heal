"""Outbound notifications (Telegram, Slack, webhook, Sentry)."""

from self_heal.notifications.dispatcher import get_dispatcher, notify

__all__ = ["get_dispatcher", "notify"]
