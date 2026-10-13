import sys
from pathlib import Path

import pytest

from self_heal.healer.policy import diff_target_path, is_path_allowed, normalize_diff_path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a/src/x.py", "src/x.py"),
        ("b/foo.py", "foo.py"),
        ("./src/x.py", "src/x.py"),
        ("a/./src/x.py", "src/x.py"),
        ("a/.github/x.py", ".github/x.py"),
        (".hidden/x.py", ".hidden/x.py"),
        ("..secret/x.py", "..secret/x.py"),
        ('"a/sp ace/x.py"', "sp ace/x.py"),
        ("b\\win\\x.py", "win/x.py"),
        ("/dev/null", ""),
        ("  a/src/x.py  ", "src/x.py"),
    ],
)
def test_normalize_diff_path(raw: str, expected: str) -> None:
    assert normalize_diff_path(raw) == expected


def test_diff_target_path_prefers_target_then_source() -> None:
    assert diff_target_path("a/old.py", "b/new.py") == "new.py"
    assert diff_target_path("a/gone.py", "/dev/null") == "gone.py"
    assert diff_target_path("/dev/null", "b/created.py") == "created.py"
    assert diff_target_path(None, None) == ""


def test_is_path_allowed(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    p = tmp_path / "src" / "a.py"
    p.write_text("x", encoding="utf-8")
    assert is_path_allowed("src/a.py", tmp_path, ["src/**"], [".git/**"])


def test_forbidden_path(tmp_path: Path) -> None:
    assert not is_path_allowed(".env", tmp_path, ["**"], [".env*"])


def test_dotdirs_are_checked_against_policy(tmp_path: Path) -> None:
    forbidden = [".git/", "**/secrets/**", ".env*"]
    assert not is_path_allowed(".git/hooks/x.py", tmp_path, ["**"], forbidden)
    assert not is_path_allowed("app/secrets/x.py", tmp_path, ["**"], forbidden)
    assert is_path_allowed(".github/x.py", tmp_path, ["**"], forbidden)


@pytest.mark.parametrize("bad", ["", "/etc/passwd", "../x.py", "src/../../x.py", "src/./../x.py"])
def test_escaping_paths_rejected(tmp_path: Path, bad: str) -> None:
    assert not is_path_allowed(bad, tmp_path, ["**"], [])


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_symlink_escape_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "link").symlink_to(outside, target_is_directory=True)
    assert not is_path_allowed("src/link/x.py", tmp_path, ["src/**"], [])


def test_empty_allowlist_denies_everything(tmp_path: Path) -> None:
    assert not is_path_allowed("src/a.py", tmp_path, [], [])
