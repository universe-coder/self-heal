"""Parse unified diff from LLM output, validate, backup, apply."""

from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from unidiff import PatchSet

from self_heal.config import SelfHealConfig
from self_heal.healer.policy import is_path_allowed, normalize_diff_path

log = logging.getLogger(__name__)


def extract_diff_from_response(text: str) -> str:
    """Take fenced ```diff ... ``` or ``` ... ``` block."""
    m = re.search(r"```diff\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]*?)```", text)
    if m:
        block = m.group(1).strip()
        if block.startswith("--- ") or block.startswith("diff "):
            return block
    if text.strip().startswith("--- "):
        return text.strip()
    return ""


@dataclass
class DiffApplyResult:
    ok: bool
    message: str
    paths_touched: list[str]


def _count_diff_lines(diff_text: str) -> int:
    return len(diff_text.splitlines())


def _list_touched_paths(diff_text: str) -> list[str]:
    paths: list[str] = []
    try:
        for pf in PatchSet(io.StringIO(diff_text)):
            t = normalize_diff_path(pf.target_file or pf.path)
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
    if not diff_text.strip():
        return False, "empty diff"
    if _count_diff_lines(diff_text) > cfg.heal.max_diff_lines:
        return False, f"diff too large (> {cfg.heal.max_diff_lines} lines)"
    try:
        patches = PatchSet(io.StringIO(diff_text))
    except Exception as e:  # noqa: BLE001
        return False, f"unidiff parse error: {e}"

    root = project_root.resolve()
    for pf in patches:
        target = normalize_diff_path(pf.target_file or pf.path)
        if not target.endswith(".py"):
            return False, f"forbidden file type in patch (MVP: .py only): {target}"
        if not is_path_allowed(target, root, cfg.heal.allowed_paths, cfg.heal.forbidden_paths):
            return False, f"path not allowed: {target}"
    return True, "ok"


def backup_files(project_root: Path, rel_paths: list[str]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / ".self-heal" / "backups" / ts
    backup_dir.mkdir(parents=True, exist_ok=True)
    for rel in rel_paths:
        src = project_root / rel
        if src.is_file():
            dest = backup_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    return backup_dir


def _git_apply(project_root: Path, diff_text: str) -> tuple[bool, str]:
    proc = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=str(project_root),
        input=diff_text.encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        return False, err or "git apply --check failed"
    proc2 = subprocess.run(
        ["git", "apply", "-"],
        cwd=str(project_root),
        input=diff_text.encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    if proc2.returncode != 0:
        err = proc2.stderr.decode("utf-8", errors="replace")
        return False, err or "git apply failed"
    return True, "applied via git apply"


def _patch_command(project_root: Path, diff_text: str) -> tuple[bool, str]:
    patch_bin = shutil.which("patch")
    if not patch_bin:
        return False, "patch executable not found"
    proc = subprocess.run(
        [patch_bin, "-p1", "--forward", "--no-backup-if-mismatch"],
        cwd=str(project_root),
        input=diff_text.encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")
        return False, err or "patch failed"
    return True, "applied via patch -p1"


def apply_patch(
    project_root: Path,
    diff_text: str,
    cfg: SelfHealConfig,
    *,
    do_backup: bool,
) -> DiffApplyResult:
    """Apply validated diff. Prefer `git apply` in git repos, else system `patch`."""
    if not diff_text.endswith("\n"):
        diff_text = diff_text + "\n"
    ok, msg = validate_patch_paths(diff_text, project_root, cfg)
    if not ok:
        return DiffApplyResult(ok=False, message=msg, paths_touched=[])

    paths = _list_touched_paths(diff_text)
    if do_backup and cfg.heal.backup and paths:
        backup_files(project_root, paths)

    root = project_root.resolve()
    if (root / ".git").is_dir():
        worked, m = _git_apply(root, diff_text)
        return DiffApplyResult(ok=worked, message=m, paths_touched=paths if worked else paths)

    worked, m = _patch_command(root, diff_text)
    if worked:
        return DiffApplyResult(ok=True, message=m, paths_touched=paths)
    # Last resort: pure-Python apply using unidiff hunks (no git/patch)
    worked2, m2 = _apply_unidiff_python(root, diff_text)
    return DiffApplyResult(ok=worked2, message=m2 if worked2 else f"{m}; {m2}", paths_touched=paths)


def _apply_unidiff_python(project_root: Path, diff_text: str) -> tuple[bool, str]:
    """Apply a unified diff by reconstructing target files from hunks (no binary hunks)."""
    try:
        patches = PatchSet(io.StringIO(diff_text))
    except Exception as e:  # noqa: BLE001
        return False, str(e)

    root = project_root.resolve()
    for pf in patches:
        rel = normalize_diff_path(pf.target_file or pf.path)
        path = root / rel
        if not path.is_file():
            return False, f"target file missing: {rel}"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

        for hunk in sorted(pf, key=lambda h: h.source_start, reverse=True):
            start_idx = hunk.source_start - 1
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
            n_old = len(old_lines)
            segment = lines[start_idx : start_idx + n_old]
            if n_old and segment != old_lines:
                return False, f"context mismatch in {rel} at line {hunk.source_start}"
            lines[start_idx : start_idx + n_old] = list(new_lines)

        path.write_text("".join(lines), encoding="utf-8")
    return True, "applied via python hunk apply"


def file_hashes(project_root: Path, rel_paths: list[str]) -> dict[str, str]:
    import hashlib

    out: dict[str, str] = {}
    for rel in rel_paths:
        p = project_root / rel
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            out[rel] = h
    return out
