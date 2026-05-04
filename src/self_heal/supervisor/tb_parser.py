"""Extract Python tracebacks from process output."""

from __future__ import annotations


def last_traceback(text: str) -> str | None:
    """Return the last traceback block starting at `Traceback (most recent call last):`."""
    token = "Traceback (most recent call last):"
    idx = text.rfind(token)
    if idx == -1:
        return None
    return text[idx:].strip()


def should_restart_exit(code: int | None, restart_codes: list[int]) -> bool:
    if code is None:
        return True
    return code in restart_codes
