"""Walk project roots respecting exclude patterns (and optionally .gitignore)."""

from __future__ import annotations

import os
from pathlib import Path

import pathspec


def load_exclude_spec(
    project_root: Path,
    exclude: list[str],
    *,
    respect_gitignore: bool = True,
) -> pathspec.PathSpec:  # type: ignore[type-arg]
    """Build a gitignore-style spec from the project's .gitignore plus ``exclude``.

    Config patterns come last so they take precedence over .gitignore negations.
    """
    patterns: list[str] = []
    if respect_gitignore:
        gi = project_root / ".gitignore"
        if gi.is_file():
            patterns.extend(gi.read_text(encoding="utf-8", errors="replace").splitlines())
    patterns.extend(exclude)
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def load_gitignore_patterns(extra_exclude: list[str]) -> pathspec.PathSpec:  # type: ignore[type-arg]
    """Backward-compatible helper: a spec from ``extra_exclude`` only."""
    return pathspec.PathSpec.from_lines("gitignore", list(extra_exclude))


def _rel(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return None


def iter_python_files(
    project_root: Path,
    roots: list[str],
    exclude: list[str],
    *,
    respect_gitignore: bool = True,
) -> list[Path]:
    """Sorted ``.py`` files under ``roots`` (relative to ``project_root``).

    Excluded directories are pruned during the walk, so a large ``.venv`` or
    ``node_modules`` is never traversed.
    """
    root_path = project_root.resolve()
    spec = load_exclude_spec(root_path, exclude, respect_gitignore=respect_gitignore)
    out: set[Path] = set()

    for rel_root in roots:
        base = (root_path / rel_root).resolve()
        if not base.exists():
            continue
        if base.is_file():
            rel_str = _rel(base, root_path)
            if base.suffix == ".py" and rel_str and not spec.match_file(rel_str):
                out.add(base)
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            d = Path(dirpath)
            kept: list[str] = []
            for name in dirnames:
                rel_dir = _rel(d / name, root_path)
                if rel_dir is None or spec.match_file(rel_dir + "/"):
                    continue
                kept.append(name)
            dirnames[:] = sorted(kept)
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                rel_str = _rel(d / name, root_path)
                if rel_str is None or spec.match_file(rel_str):
                    continue
                out.add((d / name).resolve())

    return sorted(out)
