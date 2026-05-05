"""Notification event model and notifier protocol."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field


class NotificationEvent(BaseModel):
    """Payload sent to external channels (redacted/truncated per config)."""

    kind: str
    exception_type: str
    message: str
    applied: bool | None = None
    paths_touched: list[str] = Field(default_factory=list)
    traceback_excerpt: str | None = None
    diff_excerpt: str | None = None
    raw_response_preview: str | None = None
    heal_message: str | None = None
    fingerprint_path: str | None = None
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Notifier(Protocol):
    name: str

    def send(self, event: NotificationEvent) -> None:
        """Deliver event; raises on failure."""
        ...
