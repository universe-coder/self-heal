"""Sentry via optional sentry-sdk."""

from __future__ import annotations

import logging
import os
import threading

from self_heal.config import SentryNotifConfig
from self_heal.notifications.base import NotificationEvent
from self_heal.notifications.exceptions import NotificationSkipped

log = logging.getLogger(__name__)

_sentry_init_lock = threading.Lock()
_sentry_initialized = False


class SelfHealCapturedError(Exception):
    """Synthetic exception for Sentry error_captured notifications."""


class SentryNotifier:
    name = "sentry"
    _import_warned = False

    def __init__(self, cfg: SentryNotifConfig) -> None:
        self._cfg = cfg

    def send(self, event: NotificationEvent) -> None:
        try:
            import sentry_sdk
        except ImportError:
            if not SentryNotifier._import_warned:
                log.warning(
                    "sentry-sdk not installed; install self-heal-runtime[notifications] "
                    "or pip install sentry-sdk to enable Sentry notifications"
                )
                SentryNotifier._import_warned = True
            raise NotificationSkipped("sentry-sdk not installed") from None

        dsn = os.environ.get(self._cfg.dsn_env, "").strip()
        if not dsn:
            raise RuntimeError(f"missing env {self._cfg.dsn_env}")

        global _sentry_initialized
        with _sentry_init_lock:
            if not _sentry_initialized:
                sentry_sdk.init(
                    dsn=dsn,
                    environment=(self._cfg.environment or None),
                    release=(self._cfg.release or None),
                )
                _sentry_initialized = True

        if event.kind == "error_captured":
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("exception_type", event.exception_type)
                scope.set_tag("self_heal_event", event.kind)
                fp: list[str] = [event.exception_type]
                if event.fingerprint_path:
                    fp.append(event.fingerprint_path)
                if hasattr(scope, "set_fingerprint"):
                    scope.set_fingerprint(fp)
                else:
                    scope.fingerprint = fp
                if event.traceback_excerpt:
                    scope.set_extra("traceback_excerpt", event.traceback_excerpt)
                exc = SelfHealCapturedError(f"{event.exception_type}: {event.message}")
                sentry_sdk.capture_exception(exc)
            return

        with sentry_sdk.push_scope() as scope:
            scope.set_tag("self_heal_event", event.kind)
            scope.set_tag("exception_type", event.exception_type)
            scope.set_extra("message", event.message)
            if event.applied is not None:
                scope.set_extra("applied", event.applied)
            if event.paths_touched:
                scope.set_extra("paths_touched", event.paths_touched)
            if event.heal_message:
                scope.set_extra("heal_message", event.heal_message)
            if event.traceback_excerpt:
                scope.set_extra("traceback_excerpt", event.traceback_excerpt)
            if event.diff_excerpt:
                scope.set_extra("diff_excerpt", event.diff_excerpt)
            if event.raw_response_preview:
                scope.set_extra("raw_response_preview", event.raw_response_preview)
            sentry_sdk.capture_message(
                f"{event.kind}: {event.exception_type}: {event.message}",
                level="info",
            )
