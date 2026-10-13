import subprocess
from pathlib import Path

import pytest

from self_heal.config import HealConfig, SelfHealConfig
from self_heal.healer import diff as diff_mod
from self_heal.healer.diff import (
    apply_patch,
    backup_files,
    extract_diff_from_response,
    list_backups,
    list_touched_paths,
    rollback_backup,
    validate_patch_paths,
)

_SAMPLE_DIFF = """```diff
--- a/examples/broken_app/main.py
+++ b/examples/broken_app/main.py
@@ -1,2 +1,2 @@
 def run():
-    return 1 / 0
+    return 1
```
"""

_TWO_FILE_DIFF = """--- a/examples/a.py
+++ b/examples/a.py
@@ -1,1 +1,1 @@
-A = 1
+A = 2
--- a/examples/b.py
+++ b/examples/b.py
@@ -1,1 +1,1 @@
-B = 1
+B = 2
"""


def _git_init(tmp_path: Path) -> None:
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t.com"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=str(tmp_path), check=True, capture_output=True)


def _git_commit_all(tmp_path: Path) -> None:
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True
    )


def _examples_cfg() -> SelfHealConfig:
    return SelfHealConfig(heal=HealConfig(allowed_paths=["examples/**"], forbidden_paths=[]))


def test_extract_diff() -> None:
    d = extract_diff_from_response(_SAMPLE_DIFF)
    assert "examples/broken_app/main.py" in d
    assert "-    return 1 / 0" in d


def test_extract_diff_joins_multiple_fenced_blocks() -> None:
    text = (
        "Here are the fixes:\n```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n```\n"
        "and\n```patch\n--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n-c\n+d\n```\nDone."
    )
    d = extract_diff_from_response(text)
    assert d.count("+++ ") == 2
    assert "Done." not in d and "Here are" not in d


def test_extract_diff_accepts_untagged_fence_and_git_header() -> None:
    text = "```\ndiff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n```"
    assert extract_diff_from_response(text).startswith("diff --git")


def test_extract_diff_unfenced_with_prose_before() -> None:
    text = (
        "Sure! Apply this:\ndiff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    assert extract_diff_from_response(text).startswith("diff --git")


def test_extract_diff_ignores_prose_and_non_diff_code() -> None:
    assert extract_diff_from_response("```python\nprint(1)\n```") == ""
    assert extract_diff_from_response("```diff\n\n```") == ""
    assert extract_diff_from_response("I cannot fix this.") == ""


def test_validate_paths(tmp_path: Path) -> None:
    (tmp_path / "examples" / "broken_app").mkdir(parents=True)
    f = tmp_path / "examples" / "broken_app" / "main.py"
    f.write_text("def run():\n    return 1/0\n", encoding="utf-8")
    d = extract_diff_from_response(_SAMPLE_DIFF)
    ok, msg = validate_patch_paths(d, tmp_path, SelfHealConfig())
    assert ok, msg


def test_validate_rejects_forbidden_dotdir_and_extension(tmp_path: Path) -> None:
    cfg = SelfHealConfig(heal=HealConfig(allowed_paths=["**"], forbidden_paths=[".git/"]))
    bad = "--- a/.git/hooks/pre-commit.py\n+++ b/.git/hooks/pre-commit.py\n@@ -1 +1 @@\n-a\n+b\n"
    ok, msg = validate_patch_paths(bad, tmp_path, cfg)
    assert not ok and "not allowed" in msg

    toml = "--- a/x.toml\n+++ b/x.toml\n@@ -1 +1 @@\n-a\n+b\n"
    ok, msg = validate_patch_paths(toml, tmp_path, cfg)
    assert not ok and "allowed_extensions" in msg

    cfg.heal.allowed_extensions = [".py", ".toml"]
    ok, _ = validate_patch_paths(toml, tmp_path, cfg)
    assert ok


def test_validate_rejects_deletions_and_oversize(tmp_path: Path) -> None:
    cfg = SelfHealConfig(heal=HealConfig(allowed_paths=["**"], forbidden_paths=[]))
    deletion = "--- a/x.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n"
    ok, msg = validate_patch_paths(deletion, tmp_path, cfg)
    assert not ok and "deletion" in msg

    cfg.heal.max_diff_lines = 3
    ok, msg = validate_patch_paths(_TWO_FILE_DIFF, tmp_path, cfg)
    assert not ok and "too large" in msg


def test_list_touched_paths_uses_source_for_deletions() -> None:
    assert list_touched_paths(_TWO_FILE_DIFF) == ["examples/a.py", "examples/b.py"]
    assert list_touched_paths("--- a/x.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n") == ["x.py"]
    assert list_touched_paths("not a diff") == []


def test_apply_patch_git(tmp_path: Path) -> None:
    _git_init(tmp_path)
    (tmp_path / "examples" / "broken_app").mkdir(parents=True)
    f = tmp_path / "examples" / "broken_app" / "main.py"
    f.write_text("def run():\n    return 1 / 0\n", encoding="utf-8")
    _git_commit_all(tmp_path)

    d = extract_diff_from_response(_SAMPLE_DIFF)
    res = apply_patch(tmp_path, d, SelfHealConfig(), do_backup=True)
    assert res.ok, res.message
    assert res.applied
    assert res.paths_touched == ["examples/broken_app/main.py"]
    assert "return 1\n" in f.read_text()
    assert res.backup_dir is not None
    assert (Path(res.backup_dir) / "examples" / "broken_app" / "main.py").is_file()


def test_apply_patch_git_from_subdirectory_of_repo(tmp_path: Path) -> None:
    _git_init(tmp_path)
    project = tmp_path / "service"
    (project / "examples").mkdir(parents=True)
    f = project / "examples" / "a.py"
    f.write_text("A = 1\n", encoding="utf-8")
    _git_commit_all(tmp_path)

    d = "--- a/examples/a.py\n+++ b/examples/a.py\n@@ -1 +1 @@\n-A = 1\n+A = 2\n"
    res = apply_patch(project, d, _examples_cfg(), do_backup=False)
    assert res.ok, res.message
    assert f.read_text() == "A = 2\n"


def test_apply_patch_rejects_before_touching_disk(tmp_path: Path) -> None:
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "a.py").write_text("A = 1\n", encoding="utf-8")
    cfg = SelfHealConfig(heal=HealConfig(allowed_paths=["src/**"]))
    d = "--- a/examples/a.py\n+++ b/examples/a.py\n@@ -1 +1 @@\n-A = 1\n+A = 2\n"
    res = apply_patch(tmp_path, d, cfg, do_backup=True)
    assert not res.ok and not res.applied
    assert "path not allowed" in res.message
    assert (tmp_path / "examples" / "a.py").read_text() == "A = 1\n"
    assert list_backups(tmp_path) == []


def test_python_applier_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "examples").mkdir()
    a = tmp_path / "examples" / "a.py"
    b = tmp_path / "examples" / "b.py"
    a.write_text("A = 1\n", encoding="utf-8")
    b.write_text("B = 999\n", encoding="utf-8")  # context mismatch for the second file
    monkeypatch.setattr(diff_mod, "_in_git_worktree", lambda root: False)
    monkeypatch.setattr(diff_mod, "_patch_command", lambda root, d: (False, "no patch"))

    res = apply_patch(tmp_path, _TWO_FILE_DIFF, _examples_cfg(), do_backup=False)
    assert not res.ok
    assert "context mismatch" in res.message
    assert a.read_text() == "A = 1\n"  # first file must not have been written

    b.write_text("B = 1\n", encoding="utf-8")
    res = apply_patch(tmp_path, _TWO_FILE_DIFF, _examples_cfg(), do_backup=False)
    assert res.ok, res.message
    assert a.read_text() == "A = 2\n" and b.read_text() == "B = 2\n"


def test_python_applier_creates_new_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "examples").mkdir()
    monkeypatch.setattr(diff_mod, "_in_git_worktree", lambda root: False)
    monkeypatch.setattr(diff_mod, "_patch_command", lambda root, d: (False, "no patch"))
    d = "--- /dev/null\n+++ b/examples/new.py\n@@ -0,0 +1,2 @@\n+X = 1\n+Y = 2\n"
    res = apply_patch(tmp_path, d, _examples_cfg(), do_backup=True)
    assert res.ok, res.message
    assert (tmp_path / "examples" / "new.py").read_text() == "X = 1\nY = 2\n"

    # Rolling back removes the file the patch created.
    assert res.backup_dir is not None
    touched = rollback_backup(tmp_path, Path(res.backup_dir))
    assert touched == ["examples/new.py"]
    assert not (tmp_path / "examples" / "new.py").exists()


def test_backup_and_rollback_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    f = tmp_path / "src" / "a.py"
    f.write_text("original\n", encoding="utf-8")
    backup_dir = backup_files(tmp_path, ["src/a.py", "src/missing.py"])
    f.write_text("patched\n", encoding="utf-8")
    (tmp_path / "src" / "missing.py").write_text("created\n", encoding="utf-8")

    assert list_backups(tmp_path) == [backup_dir]
    touched = rollback_backup(tmp_path, backup_dir)
    assert sorted(touched) == ["src/a.py", "src/missing.py"]
    assert f.read_text() == "original\n"
    assert not (tmp_path / "src" / "missing.py").exists()


def test_rollback_rejects_non_backup_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rollback_backup(tmp_path, tmp_path)
