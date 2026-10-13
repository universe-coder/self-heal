"""Self-heal: self-healing for Python via RAG and OpenAI-compatible / Anthropic LLMs."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from self_heal.config import ConfigError, SelfHealConfig, load_config, load_config_for_root
from self_heal.events import ErrorEvent, capture_from_exception
from self_heal.healer.diff import DiffApplyResult
from self_heal.healer.pipeline import heal_exception, heal_from_traceback_text, rollback
from self_heal.runtime.context import heal_context, install, uninstall
from self_heal.runtime.decorator import self_heal

try:
    __version__ = _pkg_version("self-heal-runtime")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "ConfigError",
    "DiffApplyResult",
    "ErrorEvent",
    "SelfHealConfig",
    "__version__",
    "capture_from_exception",
    "heal_context",
    "heal_exception",
    "heal_from_traceback_text",
    "install",
    "load_config",
    "load_config_for_root",
    "rollback",
    "self_heal",
    "uninstall",
]
