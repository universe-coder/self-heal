"""CLI behaviour via Typer's test runner (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from self_heal import __version__
from self_heal.audit.log import append_audit
from self_heal.cli import app
from self_heal.healer.diff import DiffApplyResult, backup_files

runner = CliRunner()

TB = 'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: bad\n'


@pytest.fixture
def fake_heal(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def heal(tb: str, **kwargs: Any) -> DiffApplyResult:
        calls.append({"tb": tb, **kwargs})
        return DiffApplyResult(
            ok=True,
            message="diff proposed (not applied)",
            paths_touched=["x.py"],
            diff="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
            applied=False,
        )

    monkeypatch.setattr("self_heal.cli.heal_from_traceback_text", heal)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    return calls


def test_version() -> None:
    assert runner.invoke(app, ["version"]).output.strip() == f"self-heal {__version__}"
    assert __version__ in runner.invoke(app, ["--version"]).output


def test_init_writes_template_once(tmp_path: Path) -> None:
    r = runner.invoke(app, ["init", "-p", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert (tmp_path / ".self-heal.toml").is_file()
    r = runner.invoke(app, ["init", "-p", str(tmp_path)])
    assert r.exit_code == 1 and "already exists" in r.output
    assert runner.invoke(app, ["init", "-p", str(tmp_path), "--force"]).exit_code == 0


def test_invalid_config_exits_2(tmp_path: Path) -> None:
    (tmp_path / ".self-heal.toml").write_text("[heal]\nauto_aply = true\n", encoding="utf-8")
    r = runner.invoke(app, ["doctor", "-p", str(tmp_path)])
    assert r.exit_code == 2
    assert "auto_aply" in r.output


def test_heal_defaults_to_dry_run_and_prints_diff(
    tmp_path: Path, fake_heal: list[dict[str, Any]]
) -> None:
    tb_file = tmp_path / "tb.txt"
    tb_file.write_text(TB, encoding="utf-8")
    r = runner.invoke(app, ["heal", "-p", str(tmp_path), "--tb", str(tb_file)])
    assert r.exit_code == 0, r.output
    assert "+b" in r.output and "diff proposed" in r.output
    assert "Re-run with --apply" in r.output
    call = fake_heal[0]
    assert call["dry_run"] is True and call["mode"] == "suggest"
    assert call["auto_apply_override"] is None


def test_heal_apply_implies_no_dry_run(tmp_path: Path, fake_heal: list[dict[str, Any]]) -> None:
    tb_file = tmp_path / "tb.txt"
    tb_file.write_text(TB, encoding="utf-8")
    r = runner.invoke(
        app, ["heal", "-p", str(tmp_path), "--tb", str(tb_file), "--apply", "-i", "keep it small"]
    )
    assert r.exit_code == 0, r.output
    call = fake_heal[0]
    assert call["dry_run"] is False and call["mode"] == "apply"
    assert call["auto_apply_override"] is True
    assert call["extra_instruction"] == "keep it small"

    r = runner.invoke(
        app, ["heal", "-p", str(tmp_path), "--tb", str(tb_file), "--apply", "--dry-run"]
    )
    assert r.exit_code == 0
    assert fake_heal[1]["dry_run"] is True


def test_heal_reads_stdin(tmp_path: Path, fake_heal: list[dict[str, Any]]) -> None:
    r = runner.invoke(app, ["heal", "-p", str(tmp_path), "--no-show-diff"], input=TB)
    assert r.exit_code == 0, r.output
    assert "+b" not in r.output
    assert fake_heal[0]["tb"] == TB


def test_heal_without_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = runner.invoke(app, ["heal", "-p", str(tmp_path)], input=TB)
    assert r.exit_code == 1
    assert "OPENAI_API_KEY" in r.output


def test_index_rejects_provider_without_embeddings(tmp_path: Path) -> None:
    (tmp_path / ".self-heal.toml").write_text(
        '[llm]\nprovider = "anthropic"\napi_key_env = "ANTHROPIC_API_KEY"\n', encoding="utf-8"
    )
    r = runner.invoke(app, ["index", "-p", str(tmp_path)])
    assert r.exit_code == 1
    assert "embedding_base_url" in r.output


def test_doctor_reports_missing_key_and_passes_for_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = runner.invoke(app, ["doctor", "-p", str(tmp_path)])
    assert r.exit_code == 1
    assert "OPENAI_API_KEY" in r.output and "FAIL" in r.output

    (tmp_path / ".self-heal.toml").write_text(
        '[llm]\nprovider = "ollama"\nbase_url = "http://localhost:11434/v1"\n', encoding="utf-8"
    )
    r = runner.invoke(app, ["doctor", "-p", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "not required for ollama" in r.output
    assert "run `self-heal index`" in r.output


def test_status_table_and_json(tmp_path: Path) -> None:
    r = runner.invoke(app, ["status", "-p", str(tmp_path)])
    assert "No audit entries" in r.output
    append_audit(tmp_path, kind="heal_applied", data={"exception": "ValueError", "applied": True})
    r = runner.invoke(app, ["status", "-p", str(tmp_path)])
    assert r.exit_code == 0
    assert "heal_applied" in r.output and "ValueError" in r.output and "applied" in r.output
    r = runner.invoke(app, ["status", "-p", str(tmp_path), "--json"])
    assert json.loads(r.output)["kind"] == "heal_applied"


def test_rollback_list_and_restore(tmp_path: Path) -> None:
    r = runner.invoke(app, ["rollback", "-p", str(tmp_path), "--list"])
    assert "No backups" in r.output
    r = runner.invoke(app, ["rollback", "-p", str(tmp_path)])
    assert r.exit_code == 1 and "no self-heal backups" in r.output

    (tmp_path / "src").mkdir()
    f = tmp_path / "src" / "a.py"
    f.write_text("before\n", encoding="utf-8")
    backup = backup_files(tmp_path, ["src/a.py"])
    f.write_text("after\n", encoding="utf-8")

    r = runner.invoke(app, ["rollback", "-p", str(tmp_path), "--list"])
    assert backup.name in r.output and "src/a.py" in r.output
    r = runner.invoke(app, ["rollback", "-p", str(tmp_path), "--backup", "nope"])
    assert r.exit_code == 1
    r = runner.invoke(app, ["rollback", "-p", str(tmp_path), "--backup", backup.name])
    assert r.exit_code == 0, r.output
    assert f.read_text() == "before\n"
    # the pre-rollback state was backed up as well
    r = runner.invoke(app, ["rollback", "-p", str(tmp_path)])
    assert r.exit_code == 0
    assert f.read_text() == "after\n"


def test_run_requires_command(tmp_path: Path) -> None:
    r = runner.invoke(app, ["run", "-p", str(tmp_path)])
    assert r.exit_code == 1 and "Pass a command" in r.output
