import subprocess
from pathlib import Path

from self_heal.config import SelfHealConfig
from self_heal.healer.diff import (
    apply_patch,
    extract_diff_from_response,
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


def test_extract_diff() -> None:
    d = extract_diff_from_response(_SAMPLE_DIFF)
    assert "examples/broken_app/main.py" in d
    assert "-    return 1 / 0" in d


def test_validate_paths(tmp_path: Path) -> None:
    (tmp_path / "examples" / "broken_app").mkdir(parents=True)
    f = tmp_path / "examples" / "broken_app" / "main.py"
    f.write_text("def run():\n    return 1/0\n", encoding="utf-8")
    d = extract_diff_from_response(_SAMPLE_DIFF)
    cfg = SelfHealConfig()
    ok, msg = validate_patch_paths(d, tmp_path, cfg)
    assert ok, msg


def _git_init(tmp_path: Path) -> None:
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


def test_apply_patch_git(tmp_path: Path) -> None:
    _git_init(tmp_path)
    (tmp_path / "examples" / "broken_app").mkdir(parents=True)
    f = tmp_path / "examples" / "broken_app" / "main.py"
    f.write_text("def run():\n    return 1 / 0\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    d = extract_diff_from_response(_SAMPLE_DIFF)
    cfg = SelfHealConfig()
    res = apply_patch(tmp_path, d, cfg, do_backup=False)
    assert res.ok, res.message
    assert "return 1\n" in f.read_text()
