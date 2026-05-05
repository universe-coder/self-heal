"""Load and validate `.self-heal.toml` (Pydantic)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_CONFIG_NAME = ".self-heal.toml"


class LLMConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 60.0
    max_tokens: int = 4096


class IndexConfig(BaseModel):
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
    chunk_max_lines: int = 80
    chunk_overlap_lines: int = 10


class HealConfig(BaseModel):
    allowed_paths: list[str] = Field(default_factory=lambda: ["src/**", "examples/**"])
    forbidden_paths: list[str] = Field(
        default_factory=lambda: [".env*", "**/secrets/**", ".git/", "pyproject.toml"]
    )
    auto_apply: bool = False
    max_attempts: int = 3
    backup: bool = True
    top_k: int = 8
    max_diff_lines: int = 500


class SupervisorConfig(BaseModel):
    restart_on_exit_codes: list[int] = Field(default_factory=lambda: [1])
    max_restarts: int = 5
    healthcheck_grace_s: float = 5.0


class TelegramNotifConfig(BaseModel):
    enabled: bool = False
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id: str = ""
    parse_mode: Literal["HTML", "MarkdownV2", "none"] = "HTML"


class SlackNotifConfig(BaseModel):
    enabled: bool = False
    webhook_url_env: str = "SLACK_WEBHOOK_URL"


class WebhookNotifConfig(BaseModel):
    enabled: bool = False
    url_env: str = "SELF_HEAL_WEBHOOK_URL"
    signing_secret_env: str = "SELF_HEAL_WEBHOOK_SECRET"
    allow_insecure: bool = False


class SentryNotifConfig(BaseModel):
    enabled: bool = False
    dsn_env: str = "SENTRY_DSN"
    environment: str = ""
    release: str = ""


class NotificationsConfig(BaseModel):
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
    timeout_s: float = 5.0
    include_diff: bool = False
    include_traceback: bool = True
    max_traceback_lines: int = 20
    telegram: TelegramNotifConfig = Field(default_factory=TelegramNotifConfig)
    slack: SlackNotifConfig = Field(default_factory=SlackNotifConfig)
    webhook: WebhookNotifConfig = Field(default_factory=WebhookNotifConfig)
    sentry: SentryNotifConfig = Field(default_factory=SentryNotifConfig)


def _merge_notifications_dict(defaults: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    for k, v in user.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


class SelfHealConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    heal: HealConfig = Field(default_factory=HealConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)


def load_toml_file(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return tomllib.loads(data.decode("utf-8"))


def merge_defaults(data: dict[str, Any]) -> SelfHealConfig:
    """Merge TOML sections with defaults."""
    llm = {**LLMConfig().model_dump(), **data.get("llm", {})}
    index = {**IndexConfig().model_dump(), **data.get("index", {})}
    heal = {**HealConfig().model_dump(), **data.get("heal", {})}
    sup = {**SupervisorConfig().model_dump(), **data.get("supervisor", {})}
    notif_merged = _merge_notifications_dict(
        NotificationsConfig().model_dump(),
        data.get("notifications", {}),
    )
    return SelfHealConfig(
        llm=LLMConfig(**llm),
        index=IndexConfig(**index),
        heal=HealConfig(**heal),
        supervisor=SupervisorConfig(**sup),
        notifications=NotificationsConfig.model_validate(notif_merged),
    )


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
    """Load config from path or discover `.self-heal.toml`."""
    if path is not None:
        return merge_defaults(load_toml_file(path))
    found = find_config()
    if found is None:
        return SelfHealConfig()
    return merge_defaults(load_toml_file(found))


def load_config_for_root(project_root: Path) -> SelfHealConfig:
    """Load config, searching upward from `project_root`."""
    found = find_config(project_root)
    if found is None:
        return SelfHealConfig()
    return merge_defaults(load_toml_file(found))


def get_api_key(cfg: SelfHealConfig) -> str | None:
    return os.environ.get(cfg.llm.api_key_env)
