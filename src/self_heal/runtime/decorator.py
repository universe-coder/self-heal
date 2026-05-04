"""`@self_heal` decorator."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

from self_heal.healer.pipeline import heal_exception
from self_heal.runtime.capture import capture_from_exception

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

Mode = Literal["suggest", "apply", "auto"]


def self_heal(
    *,
    retries: int = 0,
    mode: Mode = "suggest",
    project_root: Path | None = None,
) -> Callable[[F], F]:
    """Wrap a callable; on exception optionally run healing pipeline."""

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001
                    if attempts > retries:
                        raise
                    attempts += 1
                    ev = capture_from_exception(exc)
                    try:
                        heal_exception(ev, mode=mode, project_root=project_root)
                    except Exception as he:  # noqa: BLE001
                        log.warning("Healing failed: %s", he)
                    raise

        return wrapper  # type: ignore[return-value]

    return deco
