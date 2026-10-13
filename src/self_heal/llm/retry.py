"""Shared retry policy for LLM/HTTP calls: bounded attempts, exponential backoff."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

#: HTTP statuses worth retrying: throttling, timeouts, transient server errors, Anthropic 529.
RETRY_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
DEFAULT_BACKOFF_S = 1.5
MAX_BACKOFF_S = 30.0


def retry_call(
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[Exception], bool],
    max_retries: int,
    backoff_s: float = DEFAULT_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "call",
) -> T:
    """Call ``fn`` up to ``max_retries + 1`` times, sleeping ``backoff_s * 2**n`` between tries.

    Non-retryable exceptions (auth errors, bad requests) propagate immediately.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries or not is_retryable(e):
                raise
            delay = min(backoff_s * (2**attempt), MAX_BACKOFF_S)
            log.warning(
                "%s failed (%s); retry %d/%d in %.1fs", label, e, attempt + 1, max_retries, delay
            )
            sleep(delay)
            attempt += 1
