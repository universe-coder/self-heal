"""Extract Python tracebacks from process output."""

from __future__ import annotations

import re

TRACEBACK_HEADER = "Traceback (most recent call last):"
_CHAIN_MARKERS = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)
# A SyntaxError in the main script is printed without a "Traceback" header:
#   File "app.py", line 3
#     def f(:
#           ^
# SyntaxError: invalid syntax
_SYNTAX_BLOCK_RE = re.compile(
    r'(?m)^  File "[^"]+", line \d+[^\n]*\n(?:[ \t]+[^\n]*\n)*'
    r"(?:SyntaxError|IndentationError|TabError)\b[^\n]*"
)


def last_traceback(text: str) -> str | None:
    """Return the last traceback in ``text`` including its chained causes.

    Falls back to a header-less SyntaxError block. ``None`` if nothing is found.
    """
    starts = [m.start() for m in re.finditer(re.escape(TRACEBACK_HEADER), text)]
    if not starts:
        matches = list(_SYNTAX_BLOCK_RE.finditer(text))
        return matches[-1].group(0).strip() if matches else None
    i = len(starts) - 1
    while i > 0:
        between = text[starts[i - 1] : starts[i]]
        if any(marker in between for marker in _CHAIN_MARKERS):
            i -= 1
            continue
        break
    return text[starts[i] :].strip()


def normalize_traceback(tb: str) -> str:
    """Canonical form for "same crash" comparisons (addresses and whitespace removed)."""
    tb = re.sub(r"0x[0-9a-fA-F]+", "0x?", tb)
    return "\n".join(line.rstrip() for line in tb.strip().splitlines())


def should_restart_exit(code: int | None, restart_codes: list[int]) -> bool:
    if code is None:
        return True
    return code in restart_codes
