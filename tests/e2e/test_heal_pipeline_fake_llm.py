"""End-to-end heal flow with monkeypatched LLM (no network)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from self_heal.config import load_config_for_root
from self_heal.healer.pipeline import heal_from_traceback_text
from self_heal.llm.client import LLMClient


def test_heal_proposes_diff_without_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("chromadb")

    import subprocess

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    (tmp_path / "examples" / "broken_app").mkdir(parents=True)
    main_py = tmp_path / "examples" / "broken_app" / "main.py"
    main_py.write_text("def run():\n    return 1 / 0\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    (tmp_path / ".self-heal.toml").write_text(
        """
[llm]
base_url = "http://localhost/v1"
model = "test"
embedding_model = "test"
api_key_env = "SELF_HEAL_TEST_KEY"

[index]
roots = ["examples/"]

[heal]
allowed_paths = ["examples/**"]
forbidden_paths = []
auto_apply = false
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("SELF_HEAL_TEST_KEY", "test-key")

    def embed_side_effect(texts: list[str]) -> list[list[float]]:
        dim = 1536
        return [[0.01 * (i % 7)] * dim for i in range(len(texts))]

    fake_embed = MagicMock(side_effect=embed_side_effect)
    fake_chat = MagicMock(
        return_value=(
            "```diff\n"
            "--- a/examples/broken_app/main.py\n"
            "+++ b/examples/broken_app/main.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def run():\n"
            "-    return 1 / 0\n"
            "+    return 1\n"
            "```\n"
        )
    )

    cfg = load_config_for_root(tmp_path)
    client = LLMClient(cfg)
    monkeypatch.setattr(client, "embed", fake_embed)
    monkeypatch.setattr(client, "chat", fake_chat)

    tb = (
        "Traceback (most recent call last):\n"
        '  File "examples/broken_app/main.py", line 2\n'
        "ZeroDivisionError\n"
    )

    # Dry-run: should not apply to disk
    res = heal_from_traceback_text(
        tb,
        project_root=tmp_path,
        mode="apply",
        cfg=cfg,
        client=client,
        auto_apply_override=True,
        dry_run=True,
    )
    assert res is not None
    assert res.ok
    assert "return 1 / 0" in main_py.read_text()
