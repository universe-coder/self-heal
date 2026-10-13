"""Backward-compatible alias for :mod:`self_heal.events`."""

from self_heal.events import (
    ErrorEvent,
    Frame,
    capture_from_exception,
    format_event_for_llm,
    parse_exception_from_traceback,
)

__all__ = [
    "ErrorEvent",
    "Frame",
    "capture_from_exception",
    "format_event_for_llm",
    "parse_exception_from_traceback",
]
