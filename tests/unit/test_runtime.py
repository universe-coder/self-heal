"""Tests for exception capture, the decorator and the global hooks."""

from __future__ import annotations

import sys
import threading
from typing import Any

import pytest

from self_heal.runtime import context as context_mod
from self_heal.runtime.capture import (
    capture_from_exception,
    format_event_for_llm,
    parse_exception_from_traceback,
)
from self_heal.runtime.context import heal_context, install, uninstall
from self_heal.runtime.decorator import self_heal


@pytest.fixture
def healed(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record ErrorEvents passed to heal_exception instead of calling the LLM."""
    calls: list[Any] = []

    def fake_heal(ev: Any, **kwargs: Any) -> None:
        calls.append(ev)

    monkeypatch.setattr("self_heal.runtime.decorator.heal_exception", fake_heal)
    monkeypatch.setattr("self_heal.runtime.context.heal_exception", fake_heal)
    return calls


def _outer() -> None:
    outer_var = "outer"  # noqa: F841
    _inner()


def _inner() -> None:
    inner_var = "inner"  # noqa: F841
    raise ValueError("boom")


def test_capture_uses_innermost_frame_locals() -> None:
    try:
        _outer()
    except ValueError as exc:
        ev = capture_from_exception(exc)
    assert ev.exception_type == "ValueError"
    assert ev.message == "boom"
    assert ev.sanitized_locals.get("inner_var") == "'inner'"
    assert "outer_var" not in ev.sanitized_locals
    assert ev.origin is not None
    assert ev.origin.func == "_inner"
    assert ev.frames[0].func == "_inner"
    assert ev.frames[-1].func == "test_capture_uses_innermost_frame_locals"
    assert ev.origin.source is not None and "raise ValueError" in ev.origin.source


def test_capture_limit_keeps_innermost_frames() -> None:
    def recurse(n: int) -> None:
        if n == 0:
            raise RuntimeError("deep")
        recurse(n - 1)

    try:
        recurse(50)
    except RuntimeError as exc:
        ev = capture_from_exception(exc, limit=5)
    assert len(ev.frames) == 5
    assert ev.origin is not None
    assert ev.origin.source is not None and "raise RuntimeError" in ev.origin.source


def test_format_event_for_llm_lists_locals() -> None:
    try:
        _outer()
    except ValueError as exc:
        ev = capture_from_exception(exc)
    text = format_event_for_llm(ev)
    assert "ValueError: boom" in text
    assert "inner_var = 'inner'" in text


@pytest.mark.parametrize(
    ("tb", "expected"),
    [
        (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in <module>\n'
            "    raise ValueError('bad')\n"
            "ValueError: bad\n",
            ("ValueError", "bad"),
        ),
        (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1\n'
            "KeyError: 'k'\n"
            "\nDuring handling of the above exception, another exception occurred:\n\n"
            "Traceback (most recent call last):\n"
            '  File "x.py", line 3\n'
            "socket.gaierror: [Errno -2] Name or service not known\n",
            ("socket.gaierror", "[Errno -2] Name or service not known"),
        ),
        (
            'Traceback (most recent call last):\n  File "x.py", line 1\nKeyboardInterrupt\n',
            ("KeyboardInterrupt", ""),
        ),
        (
            '  File "app.py", line 3\n    def f(:\n          ^\nSyntaxError: invalid syntax\n',
            ("SyntaxError", "invalid syntax"),
        ),
        ("no traceback here", ("Unknown", "")),
    ],
)
def test_parse_exception_from_traceback(tb: str, expected: tuple[str, str]) -> None:
    assert parse_exception_from_traceback(tb) == expected


def test_decorator_retries_then_heals_once(healed: list[Any]) -> None:
    calls = {"n": 0}

    @self_heal(retries=2)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("not yet")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3
    assert healed == []


def test_decorator_heals_on_final_failure_and_reraises(healed: list[Any]) -> None:
    @self_heal(retries=1)
    def always_fails() -> None:
        raise KeyError("x")

    with pytest.raises(KeyError):
        always_fails()
    assert len(healed) == 1
    assert healed[0].exception_type == "KeyError"


def test_decorator_ignores_keyboard_interrupt(healed: list[Any]) -> None:
    @self_heal()
    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    assert healed == []


def test_decorator_rejects_negative_retries() -> None:
    with pytest.raises(ValueError):
        self_heal(retries=-1)


async def test_decorator_supports_async_functions(healed: list[Any]) -> None:
    attempts = {"n": 0}

    @self_heal(retries=1)
    async def coro() -> int:
        attempts["n"] += 1
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError):
        await coro()
    assert attempts["n"] == 2
    assert len(healed) == 1
    assert healed[0].exception_type == "RuntimeError"


def test_heal_context_heals_and_reraises(healed: list[Any]) -> None:
    with pytest.raises(ZeroDivisionError), heal_context():
        1 / 0  # noqa: B018
    assert len(healed) == 1
    assert healed[0].exception_type == "ZeroDivisionError"


def test_heal_context_skips_system_exit(healed: list[Any]) -> None:
    with pytest.raises(SystemExit), heal_context():
        raise SystemExit(3)
    assert healed == []


def test_install_is_idempotent_and_uninstall_restores(healed: list[Any]) -> None:
    seen: list[str] = []

    def previous_hook(exc_type: Any, exc: Any, tb: Any) -> None:
        seen.append(exc_type.__name__)

    old_sys, old_thread = sys.excepthook, threading.excepthook
    sys.excepthook = previous_hook
    try:
        install()
        first = sys.excepthook
        install()  # second call must not stack hooks
        assert sys.excepthook is not first
        assert context_mod._orig_sys_hook is previous_hook

        try:
            raise ValueError("hooked")
        except ValueError as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)
        assert seen == ["ValueError"]  # previous hook still runs
        assert len(healed) == 1

        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)
        assert len(healed) == 1  # not healed

        uninstall()
        assert sys.excepthook is previous_hook
        assert threading.excepthook is old_thread
    finally:
        sys.excepthook, threading.excepthook = old_sys, old_thread


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_thread_hook_heals_thread_exceptions(healed: list[Any]) -> None:
    old_sys, old_thread = sys.excepthook, threading.excepthook
    try:
        install()
        assert threading.excepthook is not old_thread

        def target() -> None:
            raise RuntimeError("in thread")

        t = threading.Thread(target=target)
        t.start()
        t.join(timeout=5)
        assert len(healed) == 1
        assert healed[0].exception_type == "RuntimeError"
    finally:
        uninstall()
        sys.excepthook, threading.excepthook = old_sys, old_thread
