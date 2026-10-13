"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from self_heal.config import (
    ConfigError,
    HealConfig,
    LLMConfig,
    SelfHealConfig,
    config_from_dict,
    load_config,
    load_config_for_root,
)


def test_defaults_when_no_config_file(tmp_path: Path) -> None:
    cfg = load_config_for_root(tmp_path)
    assert cfg.llm.provider == "openai"
    assert cfg.heal.allowed_extensions == [".py"]
    assert cfg.source_path is None


def test_partial_sections_take_defaults(tmp_path: Path) -> None:
    (tmp_path / ".self-heal.toml").write_text('[llm]\nmodel = "x"\n', encoding="utf-8")
    cfg = load_config_for_root(tmp_path)
    assert cfg.llm.model == "x"
    assert cfg.llm.base_url == "https://api.openai.com/v1"
    assert cfg.heal.top_k == 8
    assert cfg.source_path == tmp_path / ".self-heal.toml"


def test_found_in_parent_directory(tmp_path: Path) -> None:
    (tmp_path / ".self-heal.toml").write_text("[heal]\ntop_k = 3\n", encoding="utf-8")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert load_config_for_root(sub).heal.top_k == 3


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(ConfigError, match="heal.auto_aply"):
        config_from_dict({"heal": {"auto_aply": True}})


def test_unknown_section_is_an_error() -> None:
    with pytest.raises(ConfigError, match="healer"):
        config_from_dict({"healer": {}})


def test_invalid_toml_reports_file(tmp_path: Path) -> None:
    p = tmp_path / ".self-heal.toml"
    p.write_text("[llm\nmodel = 'x'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(p)


def test_missing_explicit_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.toml")


@pytest.mark.parametrize(
    "data",
    [
        {"heal": {"max_attempts": 0}},
        {"heal": {"top_k": 0}},
        {"llm": {"timeout_s": 0}},
        {"llm": {"temperature": 3}},
        {"llm": {"provider": "gemini"}},
        {"index": {"chunk_max_lines": 10, "chunk_overlap_lines": 10}},
        {"notifications": {"events": ["heal_applied", "bogus_event"]}},
        {"notifications": {"telegram": {"parse_mode": "Markdown"}}},
    ],
)
def test_out_of_range_values_rejected(data: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_allowed_extensions_get_dot_prefix() -> None:
    assert HealConfig(allowed_extensions=["py", ".toml"]).allowed_extensions == [".py", ".toml"]


def test_embedding_endpoint_resolution() -> None:
    plain = LLMConfig()
    assert plain.embeddings_available
    assert plain.resolved_embedding_base_url == plain.base_url
    assert plain.resolved_embedding_api_key_env == "OPENAI_API_KEY"

    anthropic = LLMConfig(provider="anthropic", api_key_env="ANTHROPIC_API_KEY")
    assert not anthropic.embeddings_available

    mixed = LLMConfig(
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        embedding_base_url="http://localhost:11434/v1",
        embedding_api_key_env="OLLAMA_API_KEY",
    )
    assert mixed.embeddings_available
    assert mixed.resolved_embedding_base_url == "http://localhost:11434/v1"
    assert mixed.resolved_embedding_api_key_env == "OLLAMA_API_KEY"


def test_source_path_excluded_from_dump(tmp_path: Path) -> None:
    (tmp_path / ".self-heal.toml").write_text("", encoding="utf-8")
    cfg = load_config_for_root(tmp_path)
    assert "source_path" not in cfg.model_dump()


def test_model_is_strict_for_direct_construction() -> None:
    with pytest.raises(ValueError):
        SelfHealConfig(bogus=1)  # type: ignore[call-arg]
