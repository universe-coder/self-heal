"""Install global exception hooks and a healing context manager."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from self_heal.healer.pipeline import heal_exception
from self_heal.runtime.capture import capture_from_exception

log = logging.getLogger(__name__)

Mode = Literal["suggest", "apply", "auto"]

_orig_sys_hook: Any | None = None
_orig_thread_hook: Any | None = None
_installed_sys_hook: Any | None = None
_installed_thread_hook: Any | None = None


def _heal_safely(exc: BaseException, *, mode: Mode, project_root: Path | None) -> None:
    if not isinstance(exc, Exception):
        # KeyboardInterrupt, SystemExit, GeneratorExit: not bugs to fix.
        return
    try:
        heal_exception(capture_from_exception(exc), mode=mode, project_root=project_root)
    except Exception as he:  # noqa: BLE001
        log.warning("self-heal hook failed: %s", he)


def install(
    *,
    mode: Mode = "suggest",
    project_root: Path | None = None,
    threads: bool = True,
) -> None:
    """Install ``sys.excepthook`` (and ``threading.excepthook``) that run self-heal.

    The previously installed hooks still run first, so other handlers keep working.
    Calling ``install`` again replaces the self-heal hooks instead of stacking them.
    """
    global _orig_sys_hook, _orig_thread_hook, _installed_sys_hook, _installed_thread_hook

    if _installed_sys_hook is None or sys.excepthook is not _installed_sys_hook:
        _orig_sys_hook = sys.excepthook

    def _sys_hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        prev = _orig_sys_hook or sys.__excepthook__
        try:
            prev(exc_type, exc, tb)
        except Exception:  # noqa: BLE001
            sys.__excepthook__(exc_type, exc, tb)
        _heal_safely(exc, mode=mode, project_root=project_root)

    sys.excepthook = _sys_hook
    _installed_sys_hook = _sys_hook

    if not threads:
        return

    if _installed_thread_hook is None or threading.excepthook is not _installed_thread_hook:
        _orig_thread_hook = threading.excepthook

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        prev = _orig_thread_hook or threading.__excepthook__
        try:
            prev(args)
        except Exception:  # noqa: BLE001
            threading.__excepthook__(args)
        if args.exc_value is not None:
            _heal_safely(args.exc_value, mode=mode, project_root=project_root)

    threading.excepthook = _thread_hook
    _installed_thread_hook = _thread_hook


def uninstall() -> None:
    """Restore the hooks that were active before ``install``."""
    global _orig_sys_hook, _orig_thread_hook, _installed_sys_hook, _installed_thread_hook
    if _installed_sys_hook is not None and sys.excepthook is _installed_sys_hook:
        sys.excepthook = _orig_sys_hook or sys.__excepthook__
    if _installed_thread_hook is not None and threading.excepthook is _installed_thread_hook:
        threading.excepthook = _orig_thread_hook or threading.__excepthook__
    _orig_sys_hook = _orig_thread_hook = None
    _installed_sys_hook = _installed_thread_hook = None


@contextmanager
def heal_context(
    *,
    mode: Mode = "suggest",
    project_root: Path | None = None,
) -> Iterator[None]:
    """Run healing for exceptions raised inside the block; the exception is re-raised."""
    try:
        yield
    except Exception as exc:
        _heal_safely(exc, mode=mode, project_root=project_root)
        raise
