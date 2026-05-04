"""Install global exception hook and context manager."""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from self_heal.healer.pipeline import heal_exception
from self_heal.runtime.capture import capture_from_exception

log = logging.getLogger(__name__)

_orig_hook: Any | None = None
Mode = Literal["suggest", "apply", "auto"]


def install(*, mode: Mode = "suggest", project_root: Path | None = None) -> None:
    """Set `sys.excepthook` to run self-heal after printing traceback."""

    def _hook(
        exc_type: type[BaseException],
        exc: BaseException,
        tb: Any,
    ) -> None:
        with contextlib.suppress(Exception):
            sys.__excepthook__(exc_type, exc, tb)
        try:
            ev = capture_from_exception(exc)
            heal_exception(ev, mode=mode, project_root=project_root)
        except Exception as he:  # noqa: BLE001
            log.warning("self-heal excepthook failed: %s", he)

    global _orig_hook
    _orig_hook = sys.excepthook
    sys.excepthook = _hook


def uninstall() -> None:
    if _orig_hook is not None:
        sys.excepthook = _orig_hook


@contextmanager
def heal_context(
    *,
    mode: Mode = "suggest",
    project_root: Path | None = None,
) -> Iterator[None]:
    """Run healing for exceptions raised inside the context block."""
    try:
        yield
    except BaseException as exc:  # noqa: BLE001
        ev = capture_from_exception(exc)
        try:
            heal_exception(ev, mode=mode, project_root=project_root)
        except Exception as he:  # noqa: BLE001
            log.warning("self-heal context failed: %s", he)
        raise
