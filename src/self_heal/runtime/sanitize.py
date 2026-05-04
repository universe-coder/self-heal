"""Mask secrets in locals and strings sent to the LLM."""

from __future__ import annotations

import os
import re
from typing import Any

_SECRET_NAME_RE = re.compile(
    r"(PASSWORD|SECRET|TOKEN|API[_-]?KEY|AUTH|BEARER|CREDENTIAL)",
    re.IGNORECASE,
)


def _mask_value(val: str) -> str:
    if len(val) <= 4:
        return "[redacted]"
    return val[:2] + "…" + val[-1]


def sanitize_mapping(locals_dict: dict[str, Any] | None) -> dict[str, str]:
    """Convert locals to str values with masking."""
    out: dict[str, str] = {}
    if not locals_dict:
        return out
    env_values = set(os.environ.values())
    for k, v in list(locals_dict.items())[:40]:
        try:
            s = repr(v)
        except Exception:  # noqa: BLE001
            s = "<unrepr>"
        if _SECRET_NAME_RE.search(k):
            s = _mask_value(s) if len(s) > 8 else "[redacted]"
        elif (s in env_values and len(s) > 3) or any(
            x in k.upper() for x in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            s = _mask_value(s)
        out[k] = s[:2000]
    return out


def sanitize_traceback_text(tb: str) -> str:
    """Mask obvious bearer tokens in traceback strings."""
    lines = tb.splitlines()
    out: list[str] = []
    for line in lines:
        if "Bearer " in line:
            line = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", line)
        if "Authorization:" in line:
            line = re.sub(r"Authorization:\s*\S+", "Authorization: [redacted]", line, flags=re.I)
        out.append(line)
    return "\n".join(out)
