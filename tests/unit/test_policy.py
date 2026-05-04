from pathlib import Path

from self_heal.healer.policy import is_path_allowed, normalize_diff_path


def test_normalize_diff_path() -> None:
    assert normalize_diff_path("a/src/x.py") == "src/x.py"
    assert normalize_diff_path("b/foo.py") == "foo.py"


def test_is_path_allowed(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    p = (tmp_path / "src" / "a.py")
    p.write_text("x", encoding="utf-8")
    assert is_path_allowed("src/a.py", tmp_path, ["src/**"], [".git/**"])


def test_forbidden_path(tmp_path: Path) -> None:
    assert not is_path_allowed(".env", tmp_path, ["**"], [".env*"])
