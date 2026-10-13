"""Structured error events captured from live exceptions or traceback text."""

from __future__ import annotations

import linecache
import re
import traceback as tb_module
from datetime import UTC, datetime
from types import TracebackType

from pydantic import BaseModel, Field

from self_heal.sanitize import sanitize_mapping, sanitize_traceback_text

_TRACEBACK_HEADER = "Traceback (most recent call last):"
_EXC_LINE_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*)(?::\s?(?P<msg>.*))?$")


class Frame(BaseModel):
    path: str | None = None
    line: int | None = None
    func: str | None = None
    source: str | None = None


class ErrorEvent(BaseModel):
    """A captured exception.

    ``frames`` is ordered innermost-first: ``frames[0]`` is the frame that raised.
    ``sanitized_locals`` are the (masked) locals of that innermost frame.
    """

    exception_type: str
    message: str
    traceback: str
    frames: list[Frame] = Field(default_factory=list)
    sanitized_locals: dict[str, str] = Field(default_factory=dict)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def origin(self) -> Frame | None:
        """The frame where the exception was raised (innermost), if any."""
        return self.frames[0] if self.frames else None


def _read_source_line(path: str, lineno: int) -> str | None:
    line = linecache.getline(path, lineno)
    return line.rstrip("\n") if line else None


def _innermost(tb: TracebackType) -> TracebackType:
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb


def capture_from_exception(exc: BaseException, *, limit: int = 30) -> ErrorEvent:
    """Build an ``ErrorEvent`` from an exception object.

    Only the ``limit`` innermost frames are kept, so the raise site is never dropped.
    """
    etype = type(exc).__name__
    message = str(exc)
    tb = exc.__traceback__
    frames: list[Frame] = []
    if tb is not None:
        for fr, lineno in tb_module.walk_tb(tb):
            code = fr.f_code
            frames.append(
                Frame(
                    path=code.co_filename,
                    line=lineno,
                    func=code.co_name,
                    source=_read_source_line(code.co_filename, lineno),
                )
            )
        if limit > 0:
            frames = frames[-limit:]

    tb_str = "".join(tb_module.format_exception(type(exc), exc, tb))
    tb_str = sanitize_traceback_text(tb_str)

    locals_dict = dict(_innermost(tb).tb_frame.f_locals) if tb is not None else None

    return ErrorEvent(
        exception_type=etype,
        message=sanitize_traceback_text(message),
        traceback=tb_str,
        frames=list(reversed(frames)),
        sanitized_locals=sanitize_mapping(locals_dict),
    )


def parse_exception_from_traceback(tb: str) -> tuple[str, str]:
    """Best-effort ``(exception_type, message)`` from traceback text.

    Looks at the first non-indented line after the last traceback header, which
    is where Python prints ``ExcType: message``. Returns ``("Unknown", "")`` when
    nothing recognisable is found.
    """
    lines = tb.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip() == _TRACEBACK_HEADER:
            start = i + 1
    for ln in lines[start:]:
        if not ln.strip() or ln[:1].isspace():
            continue
        m = _EXC_LINE_RE.match(ln.strip())
        if m is None:
            break
        return m.group("type"), (m.group("msg") or "").strip()
    return "Unknown", ""


def format_event_for_llm(ev: ErrorEvent) -> str:
    parts = [ev.traceback.strip()]
    if ev.sanitized_locals:
        parts += ["", "## Sanitized locals (innermost frame)"]
        parts += [f"{k} = {v}" for k, v in ev.sanitized_locals.items()]
    return "\n".join(parts)
