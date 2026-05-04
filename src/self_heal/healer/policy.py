"""Path allow/deny policy for patches."""

from __future__ import annotations

from pathlib import Path

import pathspec


def load_specs(
    allowed: list[str],
    forbidden: list[str],
) -> tuple[pathspec.PathSpec, pathspec.PathSpec]:  # type: ignore[type-arg]
    allow = pathspec.PathSpec.from_lines("gitignore", allowed or ["**"])
    forbid = pathspec.PathSpec.from_lines("gitignore", forbidden or [])
    return allow, forbid


def is_path_allowed(
    rel_posix: str,
    project_root: Path,
    allowed: list[str],
    forbidden: list[str],
) -> bool:
    """Relative path uses forward slashes; must resolve inside project_root."""
    root = project_root.resolve()
    candidate = (root / rel_posix).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False

    allow_spec, forbid_spec = load_specs(allowed, forbidden)
    if forbid_spec.match_file(rel_posix):
        return False
    if not allowed:
        return False
    return allow_spec.match_file(rel_posix)


def normalize_diff_path(path: str) -> str:
    """Strip a/ b/ prefixes and leading ./."""
    p = path.strip()
    for prefix in ("a/", "b/"):
        if p.startswith(prefix):
            p = p[len(prefix) :]
            break
    return p.lstrip("./")
