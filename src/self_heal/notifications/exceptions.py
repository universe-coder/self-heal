"""Notification-specific exceptions."""


class NotificationSkipped(Exception):
    """Delivery was intentionally skipped (not a failure)."""

