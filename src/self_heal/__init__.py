"""Self-heal: self-healing for Python via RAG and OpenAI-compatible LLMs."""

from self_heal.runtime.context import heal_context, install, uninstall
from self_heal.runtime.decorator import self_heal

__all__ = [
    "heal_context",
    "install",
    "uninstall",
    "self_heal",
]

__version__ = "0.1.0"
