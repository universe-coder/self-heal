"""Unit tests for notification redaction, validation, and dispatch."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from self_heal.config import NotificationsConfig, SelfHealConfig
from self_heal.notifications import dispatcher as dispatcher_mod
from self_heal.notifications.base import NotificationEvent
from self_heal.notifications.dispatcher import NotificationDispatcher
from self_heal.notifications.redact import make_notification_event
from self_heal.notifications.sentry import SentryNotifier
from self_heal.notifications.slack import SlackNotifier
from self_heal.notifications.telegram import TelegramNotifier
from self_heal.notifications.url_validate import validate_webhook_url
from self_heal.notifications.webhook import WebhookNotifier, build_webhook_payload, sign_body


@pytest.fixture(autouse=True)
def _reset_dispatcher_state() -> None:
    import self_heal.notifications.sentry as sentry_mod

    sentry_mod._sentry_initialized = False
    SentryNotifier._import_warned = False
    with dispatcher_mod._dispatchers_lock:
        dispatcher_mod._dispatchers.clear()
    dispatcher_mod._shutdown_executor()
    yield
    dispatcher_mod._shutdown_executor()
    with dispatcher_mod._dispatchers_lock:
        dispatcher_mod._dispatchers.clear()
    sentry_mod._sentry_initialized = False
    SentryNotifier._import_warned = False


def test_make_notification_event_truncates_traceback() -> None:
    ncfg = NotificationsConfig(include_traceback=True, max_traceback_lines=2, include_diff=False)
    tb = "a\nb\nc\nd\n"
    ev = make_notification_event(
        ncfg,
        kind="error_captured",
        exception_type="E",
        message="m",
        traceback=tb,
    )
    assert ev.traceback_excerpt is not None
    assert "more lines omitted" in ev.traceback_excerpt
    assert "d" not in ev.traceback_excerpt.split("omitted")[0]


def test_make_notification_event_omits_diff_when_disabled() -> None:
    ncfg = NotificationsConfig(include_diff=False, include_traceback=False)
    ev = make_notification_event(
        ncfg,
        kind="heal_diff_proposed",
        exception_type="E",
        message="m",
        diff_text="--- a/x\n+++ b/x\n",
    )
    assert ev.diff_excerpt is None


def test_make_notification_event_includes_diff_when_enabled() -> None:
    ncfg = NotificationsConfig(include_diff=True, include_traceback=False)
    ev = make_notification_event(
        ncfg,
        kind="heal_diff_proposed",
        exception_type="E",
        message="m",
        diff_text="--- a/x\n+++ b/x\n",
    )
    assert ev.diff_excerpt is not None
    assert "a/x" in ev.diff_excerpt


def test_validate_webhook_rejects_http_by_default() -> None:
    with pytest.raises(ValueError, match="https"):
        validate_webhook_url("http://example.com/hook", allow_insecure=False)


def test_validate_webhook_allows_http_when_insecure() -> None:
    validate_webhook_url("http://example.com/hook", allow_insecure=True)


def test_validate_webhook_rejects_bad_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        validate_webhook_url("file:///etc/passwd", allow_insecure=True)


def test_sign_body_hmac_deterministic() -> None:
    body = b'{"kind":"x"}'
    ts = "123"
    secret = "s"
    a = sign_body(body, secret, ts)
    b = sign_body(body, secret, ts)
    assert a == b
    assert len(a) == 64


def test_build_webhook_payload_json_serializable() -> None:
    import json

    ev = NotificationEvent(
        kind="k",
        exception_type="E",
        message="m",
        ts=datetime(2020, 1, 1, tzinfo=UTC),
    )
    d = build_webhook_payload(ev)
    json.dumps(d)


def test_webhook_notifier_posts_signed_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from self_heal.config import WebhookNotifConfig

    captured: dict[str, object] = {}

    def fake_post_bytes(
        url: str,
        body: bytes,
        *,
        timeout_s: float,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = extra_headers or {}
        captured["timeout"] = timeout_s

    monkeypatch.setenv("SELF_HEAL_WEBHOOK_URL", "https://example.com/hook")
    monkeypatch.setenv("SELF_HEAL_WEBHOOK_SECRET", "secret")
    monkeypatch.setattr("self_heal.notifications.webhook.post_bytes", fake_post_bytes)
    monkeypatch.setattr(
        "self_heal.notifications.webhook.validate_webhook_url",
        lambda *a, **k: None,
    )

    cfg = WebhookNotifConfig(enabled=True)
    n = WebhookNotifier(cfg, timeout_s=3.0)
    ev = NotificationEvent(kind="error_captured", exception_type="E", message="msg")
    n.send(ev)

    body = captured["body"]
    assert isinstance(body, bytes)
    sig = (captured["headers"] or {}).get("X-SelfHeal-Signature", "")
    assert sig.startswith("sha256=")
    ts = (captured["headers"] or {}).get("X-SelfHeal-Timestamp", "")
    assert sign_body(body, "secret", ts) == sig.removeprefix("sha256=")


def test_telegram_notifier_posts_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from self_heal.config import TelegramNotifConfig

    calls: list[tuple[str, dict[str, str], float]] = []

    def fake_post_form(
        url: str,
        fields: dict[str, str],
        *,
        timeout_s: float,
    ) -> None:
        calls.append((url, fields, timeout_s))

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setattr("self_heal.notifications.telegram.post_form_urlencoded", fake_post_form)

    cfg = TelegramNotifConfig(enabled=True, chat_id="999", parse_mode="HTML")
    n = TelegramNotifier(cfg, timeout_s=4.0)
    ev = NotificationEvent(kind="error_captured", exception_type="ValueError", message="oops")
    n.send(ev)

    assert len(calls) == 1
    url, fields, t = calls[0]
    assert "api.telegram.org" in url
    assert "123:abc" in url
    assert fields["chat_id"] == "999"
    assert "ValueError" in fields["text"]
    assert t == 4.0


def test_slack_notifier_posts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from self_heal.config import SlackNotifConfig

    calls: list[dict[str, object]] = []

    def fake_post_json(
        url: str,
        payload: dict[str, object],
        *,
        timeout_s: float,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        calls.append({"url": url, "payload": payload, "timeout": timeout_s})

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/xxx")
    monkeypatch.setattr("self_heal.notifications.slack.post_json", fake_post_json)
    monkeypatch.setattr(
        "self_heal.notifications.slack.validate_webhook_url",
        lambda *a, **k: None,
    )

    cfg = SlackNotifConfig(enabled=True)
    n = SlackNotifier(cfg, timeout_s=5.0)
    ev = NotificationEvent(kind="heal_applied", exception_type="E", message="m", applied=True)
    n.send(ev)

    assert len(calls) == 1
    assert calls[0]["url"] == "https://hooks.slack.com/services/xxx"
    assert "blocks" in calls[0]["payload"]


def test_dispatcher_respects_events_whitelist(tmp_path: Path) -> None:
    mock_n = MagicMock()
    mock_n.name = "x"

    def sync_submit(fn: object, *a: object, **kw: object) -> MagicMock:
        if callable(fn):
            fn(*a, **kw)
        return MagicMock()

    with (
        patch.object(dispatcher_mod, "_build_notifiers", return_value=[mock_n]),
        patch.object(dispatcher_mod, "_get_executor") as ge,
    ):
        ex = MagicMock()
        ex.submit = sync_submit
        ge.return_value = ex

        cfg = SelfHealConfig()
        cfg.notifications.enabled = True
        cfg.notifications.events = ["heal_applied"]

        d = NotificationDispatcher(tmp_path)
        ev = NotificationEvent(kind="error_captured", exception_type="E", message="m")
        d.dispatch(cfg, ev)
        mock_n.send.assert_not_called()

        ev2 = NotificationEvent(kind="heal_applied", exception_type="E", message="m")
        d.dispatch(cfg, ev2)
        mock_n.send.assert_called_once()


def test_dispatcher_one_bad_notifier_still_calls_others(tmp_path: Path) -> None:
    good = MagicMock()
    good.name = "good"
    bad = MagicMock()
    bad.name = "bad"
    bad.send.side_effect = RuntimeError("boom")

    def sync_submit(fn: object, *a: object, **kw: object) -> MagicMock:
        if callable(fn):
            fn(*a, **kw)
        return MagicMock()

    with (
        patch.object(dispatcher_mod, "_build_notifiers", return_value=[bad, good]),
        patch.object(dispatcher_mod, "_get_executor") as ge,
    ):
        ex = MagicMock()
        ex.submit = sync_submit
        ge.return_value = ex

        cfg = SelfHealConfig()
        cfg.notifications.enabled = True

        d = NotificationDispatcher(tmp_path)
        ev = NotificationEvent(kind="error_captured", exception_type="E", message="m")
        d.dispatch(cfg, ev)

    bad.send.assert_called_once()
    good.send.assert_called_once()


def test_sentry_notifier_skips_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    from self_heal.config import SentryNotifConfig

    monkeypatch.setenv("SENTRY_DSN", "https://key@o.ingest.sentry.io/1")

    import builtins

    real_import = builtins.__import__

    def selective_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentry_sdk":
            raise ImportError("no sentry")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", selective_import)

    cfg = SentryNotifConfig(enabled=True)
    n = SentryNotifier(cfg)
    ev = NotificationEvent(kind="error_captured", exception_type="E", message="m")
    from self_heal.notifications.exceptions import NotificationSkipped

    with pytest.raises(NotificationSkipped):
        n.send(ev)


def test_sentry_notifier_capture_when_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from self_heal.config import SentryNotifConfig

    mock_sdk = MagicMock()
    mock_scope = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = mock_scope
    cm.__exit__.return_value = None
    mock_sdk.push_scope.return_value = cm

    monkeypatch.setenv("SENTRY_DSN", "https://key@o.ingest.sentry.io/1")
    monkeypatch.setitem(__import__("sys").modules, "sentry_sdk", mock_sdk)

    cfg = SentryNotifConfig(enabled=True)
    n = SentryNotifier(cfg)

    import self_heal.notifications.sentry as sentry_mod

    sentry_mod._sentry_initialized = False

    ev = NotificationEvent(kind="error_captured", exception_type="KeyError", message="missing")
    n.send(ev)
    mock_sdk.init.assert_called_once()
    mock_sdk.capture_exception.assert_called_once()

    cm2 = MagicMock()
    cm2.__enter__.return_value = mock_scope
    cm2.__exit__.return_value = None
    mock_sdk.push_scope.return_value = cm2

    ev2 = NotificationEvent(kind="heal_applied", exception_type="E", message="ok", applied=True)
    n.send(ev2)
    mock_sdk.capture_message.assert_called()
