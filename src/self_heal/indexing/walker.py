"""Walk project roots respecting exclude pathspec patterns."""

from __future__ import annotations

from pathlib import Path

import pathspec


def load_gitignore_patterns(extra_exclude: list[str]) -> pathspec.PathSpec:  # type: ignore[type-arg]
    patterns = list(extra_exclude)
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def iter_python_files(project_root: Path, roots: list[str], exclude: list[str]) -> list[Path]:
    """Return sorted `.py` files under roots relative to project_root."""
    spec = load_gitignore_patterns(exclude)
    out: list[Path] = []
    root_path = project_root.resolve()

    for rel in roots:
        base = (root_path / rel).resolve()
        if not base.exists():
            continue
        if base.is_file() and base.suffix == ".py":
            rel_str = str(base.relative_to(root_path)).replace("\\", "/")
            if not spec.match_file(rel_str):
                out.append(base)
            continue
        for p in base.rglob("*.py"):
            try:
                rel_str = str(p.resolve().relative_to(root_path)).replace("\\", "/")
            except ValueError:
                continue
            if spec.match_file(rel_str):
                continue
            out.append(p.resolve())

    return sorted(set(out))
