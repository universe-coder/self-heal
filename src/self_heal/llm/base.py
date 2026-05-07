"""Chat provider abstraction (strategy)."""

from __future__ import annotations

from typing import Protocol


class ChatProvider(Protocol):
    """Single-turn chat with a system + user message."""

    def chat(self, system: str, user: str) -> str: ...
