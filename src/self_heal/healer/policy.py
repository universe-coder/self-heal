"""Path allow/deny policy for patches."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pathspec

DEV_NULL = "/dev/null"


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
    """Check a project-relative POSIX path against the allow/deny lists.

    The path must be relative, free of ``..`` components, and must resolve
    (following symlinks) to a location inside ``project_root``.
    """
    if not rel_posix or not allowed:
        return False
    pure = PurePosixPath(rel_posix)
    if pure.is_absolute() or ".." in pure.parts:
        return False
    root = project_root.resolve()
    candidate = (root / rel_posix).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return False

    allow_spec, forbid_spec = load_specs(allowed, forbidden)
    if forbid_spec.match_file(rel_posix):
        return False
    return bool(allow_spec.match_file(rel_posix))


def normalize_diff_path(path: str) -> str:
    """Normalise a path from a ``---``/``+++`` diff header.

    Strips git's ``a/``/``b/`` prefixes, surrounding quotes and leading ``./``,
    and converts backslashes to forward slashes. ``/dev/null`` becomes ``""``.
    Leading dots in real path components (``.github/x.py``) are preserved.
    """
    p = path.strip()
    if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
        p = p[1:-1]
    p = p.replace("\\", "/")
    if p == DEV_NULL:
        return ""
    for prefix in ("a/", "b/"):
        if p.startswith(prefix):
            p = p[len(prefix) :]
            break
    while p.startswith("./"):
        p = p[2:]
    return p


def diff_target_path(source_file: str | None, target_file: str | None) -> str:
    """Project-relative path a patched file refers to (``""`` if undeterminable).

    Uses the target (``+++``) side; for deletions (target is ``/dev/null``) the
    source side is used so policy checks still see the real path.
    """
    target = normalize_diff_path(target_file or "")
    if target:
        return target
    return normalize_diff_path(source_file or "")
