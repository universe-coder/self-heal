"""Parse unified diffs from LLM output, validate, back up, apply and roll back."""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from unidiff import PatchedFile, PatchSet

from self_heal.config import SelfHealConfig
from self_heal.healer.policy import diff_target_path, is_path_allowed

log = logging.getLogger(__name__)

BACKUPS_DIRNAME = "backups"
BACKUP_MANIFEST = "backup.json"

_FENCE_RE = re.compile(r"```[ \t]*(?P<lang>[\w+-]*)[^\n]*\n(?P<body>[\s\S]*?)```")
_DIFF_START_RE = re.compile(r"^(?:diff --git |--- |Index: )", re.MULTILINE)


def _looks_like_diff(block: str) -> bool:
    return bool(_DIFF_START_RE.search(block)) and "+++ " in block


def extract_diff_from_response(text: str) -> str:
    """Pull the unified diff out of a model response.

    Preference order: all fenced ```diff/```patch blocks (joined, so a model that
    emits one block per file still works), then any fenced block that looks like
    a diff, then an unfenced diff starting at the first ``diff --git``/``---`` line.
    """
    blocks = list(_FENCE_RE.finditer(text))
    tagged = [
        m.group("body").strip()
        for m in blocks
        if m.group("lang").lower() in ("diff", "patch") and _looks_like_diff(m.group("body"))
    ]
    if tagged:
        return "\n".join(tagged)
    for m in blocks:
        body = m.group("body").strip()
        if _looks_like_diff(body):
            return body
    if not blocks:
        m2 = _DIFF_START_RE.search(text)
        if m2 and _looks_like_diff(text[m2.start() :]):
            return text[m2.start() :].strip()
    return ""


@dataclass
class DiffApplyResult:
    ok: bool
    message: str
    paths_touched: list[str]
    diff: str = ""
    applied: bool = False
    backup_dir: str | None = None
    attempts: int = 1
    policy_errors: list[str] = field(default_factory=list)


def _count_diff_lines(diff_text: str) -> int:
    return len(diff_text.splitlines())


def _parse(diff_text: str) -> PatchSet:
    return PatchSet(io.StringIO(diff_text))


def _patched_path(pf: PatchedFile) -> str:
    return diff_target_path(pf.source_file, pf.target_file)


def list_touched_paths(diff_text: str) -> list[str]:
    """Unique project-relative paths a diff modifies (empty if it does not parse)."""
    paths: list[str] = []
    try:
        for pf in _parse(diff_text):
            t = _patched_path(pf)
            if t and t not in paths:
                paths.append(t)
    except Exception:  # noqa: BLE001
        pass
    return paths


def validate_patch_paths(
    diff_text: str,
    project_root: Path,
    cfg: SelfHealConfig,
) -> tuple[bool, str]:
    """Check size, parseability, file types and the allow/deny path policy."""
    if not diff_text.strip():
        return False, "empty diff"
    if _count_diff_lines(diff_text) > cfg.heal.max_diff_lines:
        return False, f"diff too large (> {cfg.heal.max_diff_lines} lines)"
    try:
        patches = _parse(diff_text)
    except Exception as e:  # noqa: BLE001
        return False, f"unidiff parse error: {e}"
    if not list(patches):
        return False, "diff contains no file changes"

    root = project_root.resolve()
    allowed_ext = tuple(cfg.heal.allowed_extensions)
    for pf in patches:
        target = _patched_path(pf)
        if not target:
            return False, "patch has an invalid file path"
        if pf.is_binary_file:
            return False, f"binary patches are not allowed: {target}"
        if pf.is_removed_file:
            return False, f"file deletions are not allowed: {target}"
        if allowed_ext and not target.endswith(allowed_ext):
            return False, f"file type not allowed (heal.allowed_extensions): {target}"
        if not is_path_allowed(target, root, cfg.heal.allowed_paths, cfg.heal.forbidden_paths):
            return False, f"path not allowed: {target}"
    return True, "ok"


def backups_root(project_root: Path) -> Path:
    return project_root.resolve() / ".self-heal" / BACKUPS_DIRNAME


def backup_files(project_root: Path, rel_paths: list[str]) -> Path:
    """Copy the files to ``.self-heal/backups/<timestamp>/`` and write a manifest.

    Paths that do not exist yet (files the patch creates) are recorded so that
    ``rollback_backup`` can remove them again.
    """
    root = project_root.resolve()
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = backups_root(root) / ts
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for rel in rel_paths:
        src = root / rel
        if src.is_file():
            dest = backup_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel)
        else:
            missing.append(rel)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "project_root": str(root),
        "files": copied,
        "created_by_patch": missing,
    }
    (backup_dir / BACKUP_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return backup_dir


def list_backups(project_root: Path) -> list[Path]:
    """Backup directories, oldest first."""
    base = backups_root(project_root)
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / BACKUP_MANIFEST).is_file())


def rollback_backup(project_root: Path, backup_dir: Path) -> list[str]:
    """Restore files from a backup directory; returns the project-relative paths touched."""
    root = project_root.resolve()
    manifest_path = backup_dir / BACKUP_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"not a self-heal backup (missing {BACKUP_MANIFEST}): {backup_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    touched: list[str] = []
    for rel in manifest.get("files", []):
        src = backup_dir / rel
        dest = root / rel
        if not src.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        touched.append(rel)
    for rel in manifest.get("created_by_patch", []):
        dest = root / rel
        if dest.is_file():
            dest.unlink()
            touched.append(rel)
    return touched


def _run(cmd: list[str], cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        input=stdin.encode("utf-8") if stdin is not None else None,
        capture_output=True,
        timeout=60,
    )


def _in_git_worktree(root: Path) -> bool:
    if shutil.which("git") is None:
        return False
    try:
        proc = _run(["git", "rev-parse", "--is-inside-work-tree"], root)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == b"true"


def _git_apply(project_root: Path, diff_text: str) -> tuple[bool, str]:
    proc = _run(["git", "apply", "--check", "-"], project_root, diff_text)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        return False, err or "git apply --check failed"
    proc2 = _run(["git", "apply", "-"], project_root, diff_text)
    if proc2.returncode != 0:
        err = proc2.stderr.decode("utf-8", errors="replace").strip()
        return False, err or "git apply failed"
    return True, "applied via git apply"


def _patch_command(project_root: Path, diff_text: str) -> tuple[bool, str]:
    patch_bin = shutil.which("patch")
    if not patch_bin:
        return False, "patch executable not found"
    # --dry-run first so a partially applicable patch never leaves files half-patched.
    for extra in (["--dry-run"], []):
        proc = _run(
            [patch_bin, "-p1", "--forward", "--no-backup-if-mismatch", "--batch", *extra],
            project_root,
            diff_text,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
            return False, err or "patch failed"
    return True, "applied via patch -p1"


def apply_patch(
    project_root: Path,
    diff_text: str,
    cfg: SelfHealConfig,
    *,
    do_backup: bool,
) -> DiffApplyResult:
    """Validate and apply a diff. Prefers ``git apply``, then ``patch``, then pure Python."""
    if not diff_text.endswith("\n"):
        diff_text = diff_text + "\n"
    ok, msg = validate_patch_paths(diff_text, project_root, cfg)
    if not ok:
        return DiffApplyResult(ok=False, message=msg, paths_touched=[], diff=diff_text)

    root = project_root.resolve()
    paths = list_touched_paths(diff_text)
    backup_dir: Path | None = None
    if do_backup and cfg.heal.backup and paths:
        backup_dir = backup_files(root, paths)

    if _in_git_worktree(root):
        worked, m = _git_apply(root, diff_text)
    else:
        worked, m = _patch_command(root, diff_text)
        if not worked:
            worked2, m2 = _apply_unidiff_python(root, diff_text)
            worked, m = worked2, (m2 if worked2 else f"{m}; {m2}")

    return DiffApplyResult(
        ok=worked,
        message=m,
        paths_touched=paths,
        diff=diff_text,
        applied=worked,
        backup_dir=str(backup_dir) if backup_dir is not None else None,
    )


def _apply_hunks(lines: list[str], pf: PatchedFile, rel: str) -> list[str] | str:
    """Return the patched lines, or an error message."""
    for hunk in sorted(pf, key=lambda h: h.source_start, reverse=True):
        old_lines: list[str] = []
        new_lines: list[str] = []
        for line in hunk:
            if line.is_context:
                old_lines.append(line.value)
                new_lines.append(line.value)
            elif line.is_removed:
                old_lines.append(line.value)
            elif line.is_added:
                new_lines.append(line.value)
        start_idx = max(hunk.source_start - 1, 0)
        n_old = len(old_lines)
        if n_old and lines[start_idx : start_idx + n_old] != old_lines:
            return f"context mismatch in {rel} at line {hunk.source_start}"
        lines[start_idx : start_idx + n_old] = new_lines
    return lines


def _apply_unidiff_python(project_root: Path, diff_text: str) -> tuple[bool, str]:
    """Apply a unified diff with ``unidiff`` only (no git/patch).

    All files are patched in memory first and written only if every hunk of
    every file applies, so a failure never leaves the tree half-patched.
    """
    try:
        patches = _parse(diff_text)
    except Exception as e:  # noqa: BLE001
        return False, str(e)

    root = project_root.resolve()
    pending: list[tuple[Path, str]] = []
    for pf in patches:
        rel = _patched_path(pf)
        path = root / rel
        if pf.is_added_file:
            lines: list[str] = []
        elif not path.is_file():
            return False, f"target file missing: {rel}"
        else:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        result = _apply_hunks(lines, pf, rel)
        if isinstance(result, str):
            return False, result
        pending.append((path, "".join(result)))

    for path, content in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return True, "applied via python hunk apply"


def file_hashes(project_root: Path, rel_paths: list[str]) -> dict[str, str]:
    import hashlib

    out: dict[str, str] = {}
    for rel in rel_paths:
        p = project_root / rel
        if p.is_file():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
