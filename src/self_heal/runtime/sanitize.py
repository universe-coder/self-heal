"""Backward-compatible alias for :mod:`self_heal.sanitize`."""

from self_heal.sanitize import (
    REDACTED,
    mask_secrets,
    sanitize_mapping,
    sanitize_traceback_text,
)

__all__ = ["REDACTED", "mask_secrets", "sanitize_mapping", "sanitize_traceback_text"]
