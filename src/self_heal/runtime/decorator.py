"""`@self_heal` decorator."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

from self_heal.events import capture_from_exception
from self_heal.healer.pipeline import heal_exception

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

Mode = Literal["suggest", "apply", "auto"]


def _run_heal(exc: BaseException, *, mode: Mode, project_root: Path | None) -> None:
    try:
        heal_exception(capture_from_exception(exc), mode=mode, project_root=project_root)
    except Exception as he:  # noqa: BLE001
        log.warning("Healing failed: %s", he)


def self_heal(
    *,
    retries: int = 0,
    mode: Mode = "suggest",
    project_root: Path | None = None,
) -> Callable[[F], F]:
    """Wrap a callable (sync or async); run the healing pipeline when it fails.

    The call is attempted up to ``retries + 1`` times. When the last attempt also
    fails, the exception is captured, healing runs once, and the exception is
    re-raised. ``KeyboardInterrupt``/``SystemExit`` are never healed.
    """
    if retries < 0:
        raise ValueError("retries must be >= 0")

    def deco(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                for attempt in range(retries + 1):
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as exc:
                        if attempt < retries:
                            log.debug("retrying %s after %r", fn.__name__, exc)
                            continue
                        _run_heal(exc, mode=mode, project_root=project_root)
                        raise
                return None  # pragma: no cover - loop always returns or raises

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt < retries:
                        log.debug("retrying %s after %r", fn.__name__, exc)
                        continue
                    _run_heal(exc, mode=mode, project_root=project_root)
                    raise
            return None  # pragma: no cover - loop always returns or raises

        return wrapper  # type: ignore[return-value]

    return deco
