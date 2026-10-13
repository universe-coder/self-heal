from pathlib import Path

from self_heal.indexing.walker import iter_python_files


def _touch(p: Path, text: str = "x = 1\n") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_walker_respects_exclude_and_prunes_dirs(tmp_path: Path) -> None:
    _touch(tmp_path / "src" / "a.py")
    _touch(tmp_path / "src" / "sub" / "b.py")
    _touch(tmp_path / "src" / "node_modules" / "c.py")
    _touch(tmp_path / "src" / "gen.py")
    _touch(tmp_path / "src" / "notes.txt")
    files = iter_python_files(tmp_path, ["src/"], ["node_modules/", "gen.py"])
    rel = [f.relative_to(tmp_path).as_posix() for f in files]
    assert rel == ["src/a.py", "src/sub/b.py"]


def test_walker_respects_gitignore_unless_disabled(tmp_path: Path) -> None:
    _touch(tmp_path / "src" / "a.py")
    _touch(tmp_path / "src" / "build" / "out.py")
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    rel = [f.relative_to(tmp_path).as_posix() for f in iter_python_files(tmp_path, ["src/"], [])]
    assert rel == ["src/a.py"]
    rel = [
        f.relative_to(tmp_path).as_posix()
        for f in iter_python_files(tmp_path, ["src/"], [], respect_gitignore=False)
    ]
    assert rel == ["src/a.py", "src/build/out.py"]


def test_walker_accepts_file_roots_and_missing_roots(tmp_path: Path) -> None:
    _touch(tmp_path / "app.py")
    files = iter_python_files(tmp_path, ["app.py", "missing/", "README.md"], [])
    assert [f.name for f in files] == ["app.py"]
