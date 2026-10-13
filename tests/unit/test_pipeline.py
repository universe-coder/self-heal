"""Healing pipeline: attempts with feedback, cooldown, proposals, apply and rollback."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from self_heal.config import HealConfig, SelfHealConfig
from self_heal.healer import pipeline as pipeline_mod
from self_heal.healer.pipeline import (
    error_fingerprint,
    event_from_traceback_text,
    heal_exception,
    heal_from_traceback_text,
    reset_cooldowns,
    rollback,
)
from self_heal.runtime.capture import ErrorEvent, Frame

GOOD_DIFF = (
    "```diff\n--- a/examples/app.py\n+++ b/examples/app.py\n@@ -1,2 +1,2 @@\n"
    " def run():\n-    return 1 / 0\n+    return 1\n```\n"
)
BAD_CONTEXT_DIFF = (
    "```diff\n--- a/examples/app.py\n+++ b/examples/app.py\n@@ -1,2 +1,2 @@\n"
    " def run():\n-    return 2 / 0\n+    return 1\n```\n"
)
FORBIDDEN_DIFF = "```diff\n--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-A=1\n+A=2\n```\n"
TB = (
    "Traceback (most recent call last):\n"
    '  File "examples/app.py", line 2, in run\n'
    "    return 1 / 0\n"
    "ZeroDivisionError: division by zero\n"
)


@pytest.fixture(autouse=True)
def _no_cooldown_state() -> Any:
    reset_cooldowns()
    yield
    reset_cooldowns()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "app.py").write_text(
        "def run():\n    return 1 / 0\n", encoding="utf-8"
    )
    return tmp_path


def _git(project: Path) -> None:
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "."],
        ["git", "commit", "-q", "-m", "init"],
    ):
        subprocess.run(cmd, cwd=str(project), check=True, capture_output=True)


def _cfg(**heal: Any) -> SelfHealConfig:
    base = {"allowed_paths": ["examples/**"], "forbidden_paths": [".env*"], "cooldown_s": 0}
    base.update(heal)
    return SelfHealConfig(heal=HealConfig(**base))


def _client(*responses: str) -> MagicMock:
    client = MagicMock()
    client.is_configured = True
    client.embeddings_available = False
    client.chat = MagicMock(side_effect=list(responses))
    return client


def _event() -> ErrorEvent:
    return ErrorEvent(
        exception_type="ZeroDivisionError",
        message="division by zero",
        traceback=TB,
        frames=[Frame(path="examples/app.py", line=2, func="run")],
    )


def _audit_kinds(project: Path) -> list[str]:
    p = project / ".self-heal" / "audit.jsonl"
    return [json.loads(ln)["kind"] for ln in p.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def notifications(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(pipeline_mod, "notify", lambda root, cfg, **kw: sent.append(kw))
    return sent


def test_empty_then_valid_diff_is_proposed(
    project: Path, notifications: list[dict[str, Any]]
) -> None:
    client = _client("I don't know", GOOD_DIFF)
    res = heal_exception(_event(), mode="suggest", project_root=project, cfg=_cfg(), client=client)
    assert res is not None and res.ok and not res.applied
    assert res.attempts == 2
    assert res.paths_touched == ["examples/app.py"]
    assert "return 1\n" in res.diff
    assert (project / "examples" / "app.py").read_text() == "def run():\n    return 1 / 0\n"
    assert _audit_kinds(project) == ["error_captured", "heal_empty_diff", "heal_diff_proposed"]
    # the second prompt carried feedback about the empty reply
    second_user = client.chat.call_args_list[1].args[1]
    assert "contained no unified diff" in second_user
    assert [n["kind"] for n in notifications] == ["error_captured", "heal_diff_proposed"]


def test_rejected_diff_gets_feedback(project: Path) -> None:
    client = _client(FORBIDDEN_DIFF, GOOD_DIFF)
    res = heal_exception(_event(), mode="suggest", project_root=project, cfg=_cfg(), client=client)
    assert res is not None and res.ok and res.attempts == 2
    assert "heal_rejected" in _audit_kinds(project)
    assert "rejected before being applied" in client.chat.call_args_list[1].args[1]
    assert not (project / ".env").exists()


def test_attempts_exhausted_reports_failure(
    project: Path, notifications: list[dict[str, Any]]
) -> None:
    client = _client(FORBIDDEN_DIFF, FORBIDDEN_DIFF)
    res = heal_exception(
        _event(), mode="suggest", project_root=project, cfg=_cfg(max_attempts=2), client=client
    )
    assert res is not None and not res.ok
    assert res.attempts == 2
    assert "patch rejected" in res.message and "after 2 attempt(s)" in res.message
    assert client.chat.call_count == 2
    # one final failure notification, not one per attempt
    assert [n["kind"] for n in notifications] == ["error_captured", "heal_rejected"]


def test_apply_failure_then_success_with_backup(project: Path) -> None:
    _git(project)
    client = _client(BAD_CONTEXT_DIFF, GOOD_DIFF)
    res = heal_exception(_event(), mode="auto", project_root=project, cfg=_cfg(), client=client)
    assert res is not None and res.ok and res.applied
    assert res.attempts == 2
    assert (project / "examples" / "app.py").read_text() == "def run():\n    return 1\n"
    assert "Applying your previous patch failed" in client.chat.call_args_list[1].args[1]
    kinds = _audit_kinds(project)
    assert kinds[-2:] == ["heal_apply_failed", "heal_applied"]
    assert res.backup_dir is not None and Path(res.backup_dir).is_dir()

    applied_entry = [
        json.loads(ln) for ln in (project / ".self-heal" / "audit.jsonl").read_text().splitlines()
    ][-1]
    assert applied_entry["data"]["backup_dir"] == res.backup_dir

    backup_dir, touched = rollback(project, cfg=_cfg())
    assert str(backup_dir) == res.backup_dir
    assert touched == ["examples/app.py"]
    assert (project / "examples" / "app.py").read_text() == "def run():\n    return 1 / 0\n"
    assert _audit_kinds(project)[-1] == "heal_rolled_back"


def test_dry_run_never_writes(project: Path) -> None:
    _git(project)
    client = _client(GOOD_DIFF)
    res = heal_exception(
        _event(), mode="auto", project_root=project, cfg=_cfg(), client=client, dry_run=True
    )
    assert res is not None and res.ok and not res.applied
    assert (project / "examples" / "app.py").read_text() == "def run():\n    return 1 / 0\n"


def test_mode_apply_respects_config_and_override(project: Path) -> None:
    _git(project)
    res = heal_exception(
        _event(),
        mode="apply",
        project_root=project,
        cfg=_cfg(auto_apply=False),
        client=_client(GOOD_DIFF),
    )
    assert res is not None and not res.applied
    res = heal_exception(
        _event(),
        mode="apply",
        project_root=project,
        cfg=_cfg(auto_apply=False),
        client=_client(GOOD_DIFF),
        auto_apply_override=True,
    )
    assert res is not None and res.applied


def test_cooldown_skips_repeated_error(project: Path) -> None:
    cfg = _cfg(cooldown_s=300)
    first = heal_exception(_event(), project_root=project, cfg=cfg, client=_client(GOOD_DIFF))
    assert first is not None and first.ok
    client2 = _client(GOOD_DIFF)
    second = heal_exception(_event(), project_root=project, cfg=cfg, client=client2)
    assert second is None
    client2.chat.assert_not_called()
    assert _audit_kinds(project)[-1] == "heal_skipped"

    other = _event()
    other.frames = [Frame(path="examples/app.py", line=99)]
    third = heal_exception(other, project_root=project, cfg=cfg, client=_client(GOOD_DIFF))
    assert third is not None and third.ok


def test_error_fingerprint_prefers_raise_site() -> None:
    ev = _event()
    assert error_fingerprint(ev) == "ZeroDivisionError|examples/app.py:2"
    ev.frames = []
    assert error_fingerprint(ev) == "ZeroDivisionError|division by zero"


def test_event_from_traceback_text_parses_type_and_frames(project: Path) -> None:
    ev = event_from_traceback_text(TB, project)
    assert (ev.exception_type, ev.message) == ("ZeroDivisionError", "division by zero")
    assert ev.origin is not None
    assert (ev.origin.path, ev.origin.line) == ("examples/app.py", 2)


def test_heal_from_traceback_text_end_to_end(project: Path) -> None:
    client = _client(GOOD_DIFF)
    res = heal_from_traceback_text(
        TB, project_root=project, mode="suggest", cfg=_cfg(), client=client
    )
    assert res is not None and res.ok
    entries = [
        json.loads(ln) for ln in (project / ".self-heal" / "audit.jsonl").read_text().splitlines()
    ]
    assert entries[0]["data"]["exception"] == "ZeroDivisionError"


def test_skips_without_api_key(project: Path) -> None:
    client = _client(GOOD_DIFF)
    client.is_configured = False
    assert heal_exception(_event(), project_root=project, cfg=_cfg(), client=client) is None
    client.chat.assert_not_called()


def test_retrieval_failure_does_not_abort(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> str:
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(pipeline_mod, "_retrieve_chunks", boom)
    res = heal_exception(_event(), project_root=project, cfg=_cfg(), client=_client(GOOD_DIFF))
    assert res is not None and res.ok


def test_rollback_without_backups(project: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rollback(project, cfg=_cfg())
