"""Capture structured error events from live exceptions."""

from __future__ import annotations

import traceback as tb_module
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from self_heal.runtime.sanitize import sanitize_mapping, sanitize_traceback_text


class Frame(BaseModel):
    path: str | None = None
    line: int | None = None
    func: str | None = None
    source: str | None = None


class ErrorEvent(BaseModel):
    exception_type: str
    message: str
    traceback: str
    frames: list[Frame] = Field(default_factory=list)
    sanitized_locals: dict[str, str] = Field(default_factory=dict)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _read_source_line(path: str, lineno: int) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if 1 <= lineno <= len(lines):
            return lines[lineno - 1].rstrip("\n")
    except OSError:
        return None
    return None


def capture_from_exception(exc: BaseException, *, limit: int = 30) -> ErrorEvent:
    """Build ErrorEvent from an active exception (call inside `except`)."""
    etype = type(exc).__name__
    message = str(exc)
    tb = exc.__traceback__
    frames: list[Frame] = []
    if tb is not None:
        for fr, _ in tb_module.walk_tb(tb):
            code = fr.f_code
            path = fr.f_code.co_filename
            lineno = fr.f_lineno
            func = code.co_name
            src = _read_source_line(path, lineno)
            frames.append(
                Frame(
                    path=path,
                    line=lineno,
                    func=func,
                    source=src,
                )
            )
            if len(frames) >= limit:
                break

    tb_str = "".join(tb_module.format_exception(type(exc), exc, exc.__traceback__))
    tb_str = sanitize_traceback_text(tb_str)

    locals_dict: dict[str, Any] | None = None
    if tb is not None:
        last = tb.tb_frame
        if last is not None:
            locals_dict = dict(last.f_locals)

    return ErrorEvent(
        exception_type=etype,
        message=message,
        traceback=tb_str,
        frames=list(reversed(frames)),
        sanitized_locals=sanitize_mapping(locals_dict),
    )


def format_event_for_llm(ev: ErrorEvent) -> str:
    parts = [
        ev.traceback.strip(),
        "",
        "## Sanitized locals (last frame)",
        repr(ev.sanitized_locals),
    ]
    return "\n".join(parts)
