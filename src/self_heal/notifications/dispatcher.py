"""Async notification dispatch with bounded concurrency."""

from __future__ import annotations

import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from self_heal.audit.log import append_audit
from self_heal.config import NotificationsConfig, SelfHealConfig
from self_heal.notifications.base import NotificationEvent
from self_heal.notifications.exceptions import NotificationSkipped
from self_heal.notifications.sentry import SentryNotifier
from self_heal.notifications.slack import SlackNotifier
from self_heal.notifications.telegram import TelegramNotifier
from self_heal.notifications.webhook import WebhookNotifier

log = logging.getLogger(__name__)

_MAX_INFLIGHT = 64

_executor_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_semaphore = threading.Semaphore(_MAX_INFLIGHT)

_dispatchers: dict[str, NotificationDispatcher] = {}
_dispatchers_lock = threading.Lock()


@runtime_checkable
class _NotifierProto(Protocol):
    name: str

    def send(self, event: NotificationEvent) -> None: ...


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sh-notify")
    return _executor


def _shutdown_executor() -> None:
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=True, cancel_futures=False)
            _executor = None


atexit.register(_shutdown_executor)


def _build_notifiers(ncfg: NotificationsConfig) -> list[_NotifierProto]:
    out: list[_NotifierProto] = []
    t = ncfg.timeout_s
    if ncfg.telegram.enabled:
        out.append(TelegramNotifier(ncfg.telegram, timeout_s=t))
    if ncfg.slack.enabled:
        out.append(SlackNotifier(ncfg.slack, timeout_s=t))
    if ncfg.webhook.enabled:
        out.append(WebhookNotifier(ncfg.webhook, timeout_s=t))
    if ncfg.sentry.enabled:
        out.append(SentryNotifier(ncfg.sentry))
    return out


def _deliver_one(
    project_root: Path,
    notifier: _NotifierProto,
    event: NotificationEvent,
) -> None:
    try:
        notifier.send(event)
        append_audit(
            project_root,
            kind="notification_sent",
            data={"channel": notifier.name, "event_kind": event.kind},
        )
    except NotificationSkipped:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("notification failed (%s): %s", notifier.name, e)
        append_audit(
            project_root,
            kind="notification_failed",
            data={"channel": notifier.name, "event_kind": event.kind, "error": str(e)},
        )


def _run_batch(
    project_root: Path,
    cfg: SelfHealConfig,
    event: NotificationEvent,
) -> None:
    try:
        ncfg = cfg.notifications
        if not ncfg.enabled:
            return
        if event.kind not in ncfg.events:
            return
        notifiers = _build_notifiers(ncfg)
        if not notifiers:
            return
        for n in notifiers:
            _deliver_one(project_root, n, event)
    finally:
        _semaphore.release()


class NotificationDispatcher:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()

    def dispatch(self, cfg: SelfHealConfig, event: NotificationEvent) -> None:
        ncfg = cfg.notifications
        if not ncfg.enabled:
            return
        if event.kind not in ncfg.events:
            return
        if not _build_notifiers(ncfg):
            return

        if not _semaphore.acquire(blocking=False):
            log.warning(
                "notification inflight cap (%d) reached; dropping event %s",
                _MAX_INFLIGHT,
                event.kind,
            )
            append_audit(
                self._project_root,
                kind="notification_dropped",
                data={"event_kind": event.kind, "reason": "inflight_cap"},
            )
            return

        ex = _get_executor()
        try:
            ex.submit(_run_batch, self._project_root, cfg, event)
        except Exception:
            _semaphore.release()
            raise


def get_dispatcher(project_root: Path) -> NotificationDispatcher:
    key = str(project_root.resolve())
    with _dispatchers_lock:
        d = _dispatchers.get(key)
        if d is None:
            d = NotificationDispatcher(project_root)
            _dispatchers[key] = d
        return d


def notify(project_root: Path, cfg: SelfHealConfig, **kwargs: Any) -> None:
    """Build a redacted NotificationEvent and dispatch asynchronously."""
    from self_heal.notifications.redact import make_notification_event

    ev = make_notification_event(cfg.notifications, **kwargs)
    get_dispatcher(project_root).dispatch(cfg, ev)
