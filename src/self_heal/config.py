"""Load and validate `.self-heal.toml` (Pydantic)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_CONFIG_NAME = ".self-heal.toml"

LLMProvider = Literal["openai", "anthropic", "huggingface", "ollama"]

#: Event kinds that can be listed in ``[notifications].events``.
NOTIFICATION_EVENTS: tuple[str, ...] = (
    "error_captured",
    "heal_diff_proposed",
    "heal_empty_diff",
    "heal_applied",
    "heal_apply_failed",
    "heal_rejected",
    "heal_rolled_back",
)


class ConfigError(ValueError):
    """Raised when the configuration file cannot be read or is invalid."""


class _StrictModel(BaseModel):
    """Unknown keys are an error: a typo must not silently disable an option."""

    model_config = ConfigDict(extra="forbid")


class LLMConfig(_StrictModel):
    provider: LLMProvider = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = Field(default=60.0, gt=0)
    max_tokens: int = Field(default=4096, ge=1)
    #: Sent only when set; some reasoning models reject a temperature parameter.
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_retries: int = Field(default=3, ge=0)
    #: Optional OpenAI-compatible endpoint used only for embeddings. Lets a chat
    #: provider without an embeddings API (Anthropic) still index and retrieve.
    embedding_base_url: str | None = None
    embedding_api_key_env: str | None = None

    @property
    def embeddings_available(self) -> bool:
        return self.provider != "anthropic" or bool(self.embedding_base_url)

    @property
    def resolved_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.base_url

    @property
    def resolved_embedding_api_key_env(self) -> str:
        return self.embedding_api_key_env or self.api_key_env


class IndexConfig(_StrictModel):
    roots: list[str] = Field(default_factory=lambda: ["src/"])
    exclude: list[str] = Field(
        default_factory=lambda: [
            ".git/",
            "node_modules/",
            "__pycache__/",
            "*.pyc",
            ".venv/",
            ".self-heal/",
            ".chroma/",
        ]
    )
    #: Also honour the project's top-level .gitignore when walking files.
    respect_gitignore: bool = True
    chunk_max_lines: int = Field(default=80, ge=5)
    chunk_overlap_lines: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> IndexConfig:
        if self.chunk_overlap_lines >= self.chunk_max_lines:
            raise ValueError("index.chunk_overlap_lines must be smaller than chunk_max_lines")
        return self


class HealConfig(_StrictModel):
    allowed_paths: list[str] = Field(default_factory=lambda: ["src/**", "examples/**"])
    forbidden_paths: list[str] = Field(
        default_factory=lambda: [".env*", "**/secrets/**", ".git/", "pyproject.toml"]
    )
    auto_apply: bool = False
    #: LLM round-trips per heal; later attempts receive the previous failure as feedback.
    max_attempts: int = Field(default=3, ge=1)
    backup: bool = True
    top_k: int = Field(default=8, ge=1)
    max_diff_lines: int = Field(default=500, ge=1)
    allowed_extensions: list[str] = Field(default_factory=lambda: [".py"])
    #: Lines of source shown around each traceback frame inside the project.
    context_lines: int = Field(default=40, ge=0)
    #: In-process: the same error (type + raise site) is not healed again within this window.
    cooldown_s: float = Field(default=300.0, ge=0)

    @field_validator("allowed_extensions")
    @classmethod
    def _dot_prefixed(cls, v: list[str]) -> list[str]:
        return [e if e.startswith(".") else f".{e}" for e in v]


class SupervisorConfig(_StrictModel):
    restart_on_exit_codes: list[int] = Field(default_factory=lambda: [1])
    max_restarts: int = Field(default=5, ge=0)
    healthcheck_grace_s: float = Field(default=5.0, ge=0)
    #: Stop restarting when the process dies with the same traceback and no patch was applied.
    stop_on_repeated_traceback: bool = True


class TelegramNotifConfig(_StrictModel):
    enabled: bool = False
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id: str = ""
    parse_mode: Literal["HTML", "MarkdownV2", "none"] = "HTML"


class SlackNotifConfig(_StrictModel):
    enabled: bool = False
    webhook_url_env: str = "SLACK_WEBHOOK_URL"


class WebhookNotifConfig(_StrictModel):
    enabled: bool = False
    url_env: str = "SELF_HEAL_WEBHOOK_URL"
    signing_secret_env: str = "SELF_HEAL_WEBHOOK_SECRET"
    allow_insecure: bool = False


class SentryNotifConfig(_StrictModel):
    enabled: bool = False
    dsn_env: str = "SENTRY_DSN"
    environment: str = ""
    release: str = ""


class NotificationsConfig(_StrictModel):
    enabled: bool = False
    events: list[str] = Field(
        default_factory=lambda: [
            "error_captured",
            "heal_diff_proposed",
            "heal_empty_diff",
            "heal_applied",
            "heal_apply_failed",
        ]
    )
    timeout_s: float = Field(default=5.0, gt=0)
    include_diff: bool = False
    include_traceback: bool = True
    max_traceback_lines: int = Field(default=20, ge=1)
    telegram: TelegramNotifConfig = Field(default_factory=TelegramNotifConfig)
    slack: SlackNotifConfig = Field(default_factory=SlackNotifConfig)
    webhook: WebhookNotifConfig = Field(default_factory=WebhookNotifConfig)
    sentry: SentryNotifConfig = Field(default_factory=SentryNotifConfig)

    @field_validator("events")
    @classmethod
    def _known_events(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(NOTIFICATION_EVENTS))
        if unknown:
            raise ValueError(
                f"unknown notification events {unknown}; valid: {list(NOTIFICATION_EVENTS)}"
            )
        return v


class SelfHealConfig(_StrictModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    heal: HealConfig = Field(default_factory=HealConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    #: The file this config was loaded from (``None`` for pure defaults).
    source_path: Annotated[Path | None, Field(exclude=True, repr=False)] = None


def _format_validation_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc or '<root>'}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def load_toml_file(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"invalid TOML in {path}: {e}") from e


def config_from_dict(data: dict[str, Any], *, source: Path | None = None) -> SelfHealConfig:
    """Validate a parsed TOML mapping; missing sections and keys take defaults."""
    try:
        cfg = SelfHealConfig.model_validate(data)
    except ValidationError as e:
        where = f" in {source}" if source else ""
        raise ConfigError(f"invalid configuration{where}: {_format_validation_error(e)}") from e
    cfg.source_path = source
    return cfg


def merge_defaults(data: dict[str, Any]) -> SelfHealConfig:
    """Backward-compatible alias for :func:`config_from_dict`."""
    return config_from_dict(data)


def find_config(start: Path | None = None) -> Path | None:
    """Walk up from `start` (or cwd) looking for `.self-heal.toml`."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(32):
        candidate = cur / DEFAULT_CONFIG_NAME
        if candidate.is_file():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def load_config(path: Path | None = None) -> SelfHealConfig:
    """Load config from `path`, or discover `.self-heal.toml` upward from cwd."""
    if path is not None:
        return config_from_dict(load_toml_file(path), source=path.resolve())
    found = find_config()
    if found is None:
        return SelfHealConfig()
    return config_from_dict(load_toml_file(found), source=found)


def load_config_for_root(project_root: Path) -> SelfHealConfig:
    """Load config, searching upward from `project_root`."""
    found = find_config(project_root)
    if found is None:
        return SelfHealConfig()
    return config_from_dict(load_toml_file(found), source=found)


def get_api_key(cfg: SelfHealConfig) -> str | None:
    return os.environ.get(cfg.llm.api_key_env) or None


def get_embedding_api_key(cfg: SelfHealConfig) -> str | None:
    return os.environ.get(cfg.llm.resolved_embedding_api_key_env) or None


def requires_api_key(cfg: SelfHealConfig) -> bool:
    """Whether the configured chat provider requires an API key.

    Ollama is the only provider supported here that runs without auth by default.
    """
    return cfg.llm.provider != "ollama"
