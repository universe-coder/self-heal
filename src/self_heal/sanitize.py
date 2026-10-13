"""Mask secrets in locals and strings before they are sent to the LLM or notifiers."""

from __future__ import annotations

import os
import re
from typing import Any

REDACTED = "[redacted]"

_SECRET_NAME_RE = re.compile(
    r"(PASSWORD|PASSWD|SECRET|TOKEN|API[_-]?KEY|CREDENTIAL|PRIVATE[_-]?KEY|BEARER"
    r"|AUTHORIZATION|(?<![A-Z])AUTH(?![A-Z])|(?<![A-Z])KEY(?![A-Z])|(?<![A-Z])DSN(?![A-Z]))",
    re.IGNORECASE,
)

# Values of env vars with a secret-looking name are masked wherever they appear.
_MIN_SECRET_VALUE_LEN = 8

# Well-known credential shapes that may show up in messages, URLs or reprs.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Bearer\s+\S+"),
    re.compile(r"(?i)Authorization:\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # OpenAI / Anthropic style keys
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),  # Hugging Face
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),  # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),  # Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),  # JWT
)
# key=value pairs in query strings / connection strings
_KV_SECRET_RE = re.compile(
    r"(?i)\b((?:password|passwd|pwd|token|secret|api[_-]?key|apikey|access[_-]?key)=)[^&\s'\"]+"
)
# user:password@host in URLs
_URL_CRED_RE = re.compile(r"(://)[^/\s:@]+:[^/\s@]+@")


def _secret_env_values() -> list[str]:
    out = {
        v
        for k, v in os.environ.items()
        if _SECRET_NAME_RE.search(k) and len(v) >= _MIN_SECRET_VALUE_LEN
    }
    # Longest first so that overlapping values are fully masked.
    return sorted(out, key=len, reverse=True)


def mask_secrets(text: str, env_values: list[str] | None = None) -> str:
    """Mask credential-looking substrings and known secret env values in ``text``."""
    if not text:
        return text
    for pat in _TOKEN_PATTERNS:
        text = pat.sub(REDACTED, text)
    text = _KV_SECRET_RE.sub(rf"\1{REDACTED}", text)
    text = _URL_CRED_RE.sub(rf"\1{REDACTED}@", text)
    for val in env_values if env_values is not None else _secret_env_values():
        text = text.replace(val, REDACTED)
    return text


def sanitize_mapping(locals_dict: dict[str, Any] | None, *, max_items: int = 40) -> dict[str, str]:
    """Convert locals to string values, masking secrets by variable name and by value."""
    out: dict[str, str] = {}
    if not locals_dict:
        return out
    env_values = _secret_env_values()
    for k, v in list(locals_dict.items())[:max_items]:
        if k.startswith("__") and k.endswith("__"):
            continue
        if _SECRET_NAME_RE.search(k):
            out[k] = REDACTED
            continue
        try:
            s = repr(v)
        except Exception:  # noqa: BLE001
            s = "<unrepr>"
        if isinstance(v, str) and v in env_values:
            s = REDACTED
        out[k] = mask_secrets(s, env_values)[:2000]
    return out


def sanitize_traceback_text(tb: str) -> str:
    """Mask bearer tokens, API keys, URL credentials and secret env values in text."""
    return mask_secrets(tb)
