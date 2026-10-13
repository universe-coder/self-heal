"""Supervisor restart/heal behaviour with real subprocesses."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from self_heal.config import SelfHealConfig, SupervisorConfig
from self_heal.healer.diff import DiffApplyResult
from self_heal.supervisor import runner as runner_mod
from self_heal.supervisor.runner import SIGINT_EXIT_CODE, run_supervised


def _cfg(**sup: Any) -> SelfHealConfig:
    base = {"max_restarts": 3, "healthcheck_grace_s": 0, "restart_on_exit_codes": [1]}
    base.update(sup)
    return SelfHealConfig(supervisor=SupervisorConfig(**base))


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


@pytest.fixture
def heals(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_heal(tb: str, **kwargs: Any) -> DiffApplyResult | None:
        calls.append({"tb": tb, **kwargs})
        return DiffApplyResult(ok=True, message="proposed", paths_touched=[], applied=False)

    monkeypatch.setattr(runner_mod, "heal_from_traceback_text", fake_heal)
    return calls


def test_clean_exit_is_not_healed(tmp_path: Path, heals: list[dict[str, Any]]) -> None:
    assert run_supervised(_py("print('ok')"), project_root=tmp_path, cfg=_cfg()) == 0
    assert heals == []


def test_crash_heals_once_then_stops_on_repeat(tmp_path: Path, heals: list[dict[str, Any]]) -> None:
    code = run_supervised(
        _py("raise ValueError('boom')"), project_root=tmp_path, cfg=_cfg(), client=MagicMock()
    )
    assert code == 1
    assert len(heals) == 1  # restarted once, then the identical traceback stopped the loop
    assert "ValueError: boom" in heals[0]["tb"]
    assert heals[0]["mode"] == "suggest" and heals[0]["dry_run"] is True


def test_repeat_restarts_when_disabled(tmp_path: Path, heals: list[dict[str, Any]]) -> None:
    code = run_supervised(
        _py("raise ValueError('boom')"),
        project_root=tmp_path,
        cfg=_cfg(max_restarts=2, stop_on_repeated_traceback=False),
        client=MagicMock(),
    )
    assert code == 1
    assert len(heals) == 3  # initial run + 2 restarts


def test_exit_code_without_traceback_respects_restart_codes(
    tmp_path: Path, heals: list[dict[str, Any]]
) -> None:
    assert run_supervised(_py("raise SystemExit(2)"), project_root=tmp_path, cfg=_cfg()) == 2
    assert heals == []


def test_auto_apply_passes_mode_and_syntax_error_is_detected(
    tmp_path: Path, heals: list[dict[str, Any]]
) -> None:
    script = tmp_path / "bad.py"
    script.write_text("def f(:\n    pass\n", encoding="utf-8")
    code = run_supervised(
        [sys.executable, str(script)],
        project_root=tmp_path,
        cfg=_cfg(max_restarts=0),
        auto_apply=True,
        dry_run=False,
        client=MagicMock(),
    )
    assert code == 1
    assert len(heals) == 1
    assert "SyntaxError" in heals[0]["tb"]
    assert heals[0]["mode"] == "apply" and heals[0]["auto_apply_override"] is True


def test_applied_patch_allows_another_heal_on_next_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_heal(tb: str, **kwargs: Any) -> DiffApplyResult:
        calls.append(tb)
        return DiffApplyResult(ok=True, message="applied", paths_touched=["x.py"], applied=True)

    monkeypatch.setattr(runner_mod, "heal_from_traceback_text", fake_heal)
    code = run_supervised(
        _py("raise ValueError('boom')"),
        project_root=tmp_path,
        cfg=_cfg(max_restarts=2),
        client=MagicMock(),
    )
    assert code == 1
    assert len(calls) == 3  # a patch was applied each time, so every crash is re-healed


def test_keyboard_interrupt_terminates_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = MagicMock()
    proc.stderr = MagicMock()
    proc.stderr.readline.return_value = ""
    proc.wait.side_effect = [KeyboardInterrupt(), 0]
    proc.poll.return_value = None
    monkeypatch.setattr(runner_mod.subprocess, "Popen", lambda *a, **k: proc)

    assert run_supervised(["anything"], project_root=tmp_path, cfg=_cfg()) == SIGINT_EXIT_CODE
    proc.terminate.assert_called_once()


def test_real_process_receives_cwd(tmp_path: Path, heals: list[dict[str, Any]]) -> None:
    marker = tmp_path / "cwd.txt"
    code = run_supervised(
        _py("import os, pathlib; pathlib.Path('cwd.txt').write_text(os.getcwd())"),
        project_root=tmp_path,
        cfg=_cfg(),
    )
    assert code == 0
    assert Path(marker.read_text()).resolve() == tmp_path.resolve()
    assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0
