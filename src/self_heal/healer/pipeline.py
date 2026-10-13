"""End-to-end healing: retrieve → LLM → validate → apply (with bounded retries)."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Literal

from self_heal.audit.log import append_audit
from self_heal.config import SelfHealConfig, find_config, load_config_for_root
from self_heal.healer.diff import (
    DiffApplyResult,
    apply_patch,
    extract_diff_from_response,
    file_hashes,
    list_backups,
    list_touched_paths,
    rollback_backup,
    validate_patch_paths,
)
from self_heal.healer.planner import build_healing_prompt
from self_heal.llm.client import LLMClient
from self_heal.llm.prompts import feedback_apply_failed, feedback_empty_diff, feedback_rejected
from self_heal.notifications.dispatcher import notify
from self_heal.retrieval.retriever import NO_CONTEXT_MSG, extract_frames_from_traceback
from self_heal.retrieval.retriever import retrieve_chunks as _retrieve_chunks
from self_heal.runtime.capture import (
    ErrorEvent,
    Frame,
    format_event_for_llm,
    parse_exception_from_traceback,
)

log = logging.getLogger(__name__)

Mode = Literal["suggest", "apply", "auto"]

_recent_lock = threading.Lock()
_recent_heals: dict[str, float] = {}


def retrieve_chunks(root: Path, cfg: SelfHealConfig, llm: LLMClient, traceback: str) -> str:
    """Code context for the prompt; retrieval problems never abort a heal."""
    try:
        return _retrieve_chunks(root, cfg, llm, traceback)
    except Exception as e:  # noqa: BLE001
        log.warning("retrieval failed (%s); continuing without code context", e)
        return NO_CONTEXT_MSG


def _fingerprint_path(event: ErrorEvent) -> str | None:
    origin = event.origin
    return origin.path if origin is not None else None


def error_fingerprint(event: ErrorEvent) -> str:
    """Identifies "the same error": exception type plus the raise site (or the message)."""
    origin = event.origin
    if origin is not None and origin.path:
        return f"{event.exception_type}|{origin.path}:{origin.line}"
    return f"{event.exception_type}|{event.message[:200]}"


def _in_cooldown(fp: str, cooldown_s: float) -> bool:
    if cooldown_s <= 0:
        return False
    now = time.monotonic()
    with _recent_lock:
        last = _recent_heals.get(fp)
        return last is not None and now - last < cooldown_s


def _mark_healed(fp: str, cooldown_s: float) -> None:
    if cooldown_s <= 0:
        return
    now = time.monotonic()
    with _recent_lock:
        # keep the map small: drop entries that are already outside the window
        for key in [k for k, t in _recent_heals.items() if now - t >= cooldown_s]:
            del _recent_heals[key]
        _recent_heals[fp] = now


def reset_cooldowns() -> None:
    with _recent_lock:
        _recent_heals.clear()


def _resolve_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    found = find_config()
    if found is not None:
        return found.parent.resolve()
    return Path.cwd().resolve()


def _notify(root: Path, conf: SelfHealConfig, **kwargs: Any) -> None:
    with contextlib.suppress(Exception):
        notify(root, conf, **kwargs)


def _resolve_auto(mode: Mode, conf: SelfHealConfig, override: bool | None) -> bool:
    if mode == "suggest":
        return False
    if mode == "auto":
        return True
    return conf.heal.auto_apply if override is None else override


def heal_exception(
    event: ErrorEvent,
    *,
    mode: Mode = "suggest",
    project_root: Path | None = None,
    cfg: SelfHealConfig | None = None,
    client: LLMClient | None = None,
    auto_apply_override: bool | None = None,
    dry_run: bool = False,
    extra_instruction: str = "",
) -> DiffApplyResult | None:
    """Run healing for a captured ``ErrorEvent``.

    Up to ``heal.max_attempts`` model round-trips are made; an empty, rejected or
    unappliable diff is fed back to the model as feedback for the next attempt.
    Returns ``None`` when healing was skipped (no API key, or the same error was
    healed within ``heal.cooldown_s``).
    """
    root = _resolve_root(project_root)
    conf = cfg or load_config_for_root(root)

    append_audit(
        root,
        kind="error_captured",
        data={"exception": event.exception_type, "message": event.message},
    )
    _notify(
        root,
        conf,
        kind="error_captured",
        exception_type=event.exception_type,
        message=event.message,
        traceback=event.traceback,
        fingerprint_path=_fingerprint_path(event),
    )

    fp = error_fingerprint(event)
    if _in_cooldown(fp, conf.heal.cooldown_s):
        log.info("Skipping heal: %s healed less than %.0fs ago", fp, conf.heal.cooldown_s)
        append_audit(root, kind="heal_skipped", data={"fingerprint": fp, "reason": "cooldown"})
        return None

    llm = client or LLMClient(conf)
    if not llm.is_configured:
        log.warning("Skipping heal: no API key (%s)", conf.llm.api_key_env)
        return None

    tb_text = format_event_for_llm(event)
    chunks = retrieve_chunks(root, conf, llm, event.traceback)
    auto = _resolve_auto(mode, conf, auto_apply_override)
    will_apply = auto and not dry_run

    feedback = ""
    last: DiffApplyResult | None = None
    max_attempts = conf.heal.max_attempts
    for attempt in range(1, max_attempts + 1):
        extra = "\n\n".join(s for s in (extra_instruction.strip(), feedback) if s)
        system, user = build_healing_prompt(tb_text, chunks, extra)
        raw = llm.chat(system, user)
        diff_text = extract_diff_from_response(raw)

        if not diff_text.strip():
            append_audit(
                root,
                kind="heal_empty_diff",
                data={
                    "exception": event.exception_type,
                    "message": event.message,
                    "attempt": attempt,
                    "raw_response_preview": raw[:2000],
                },
            )
            last = DiffApplyResult(
                ok=False, message="empty diff from model", paths_touched=[], attempts=attempt
            )
            feedback = feedback_empty_diff()
            continue

        if not diff_text.endswith("\n"):
            diff_text += "\n"
        paths = list_touched_paths(diff_text)
        ok, reason = validate_patch_paths(diff_text, root, conf)
        if not ok:
            log.warning("Patch rejected (attempt %d/%d): %s", attempt, max_attempts, reason)
            append_audit(
                root,
                kind="heal_rejected",
                data={
                    "exception": event.exception_type,
                    "attempt": attempt,
                    "reason": reason,
                    "diff": diff_text,
                },
            )
            last = DiffApplyResult(
                ok=False,
                message=f"patch rejected: {reason}",
                paths_touched=paths,
                diff=diff_text,
                attempts=attempt,
                policy_errors=[reason],
            )
            feedback = feedback_rejected(reason)
            continue

        before_hashes = file_hashes(root, paths)
        if not will_apply:
            append_audit(
                root,
                kind="heal_diff_proposed",
                data={
                    "exception": event.exception_type,
                    "attempt": attempt,
                    "diff": diff_text,
                    "before_hashes": before_hashes,
                    "applied": False,
                },
            )
            _notify(
                root,
                conf,
                kind="heal_diff_proposed",
                exception_type=event.exception_type,
                message=event.message,
                traceback=event.traceback,
                diff_text=diff_text,
                applied=False,
                paths_touched=paths,
            )
            log.info("Proposed diff (not applied); use --apply or heal.auto_apply to apply")
            _mark_healed(fp, conf.heal.cooldown_s)
            return DiffApplyResult(
                ok=True,
                message="diff proposed (not applied)",
                paths_touched=paths,
                diff=diff_text,
                attempts=attempt,
            )

        result = apply_patch(root, diff_text, conf, do_backup=True)
        result.attempts = attempt
        after_hashes = file_hashes(root, result.paths_touched)
        kind = "heal_applied" if result.ok else "heal_apply_failed"
        append_audit(
            root,
            kind=kind,
            data={
                "exception": event.exception_type,
                "attempt": attempt,
                "diff": diff_text,
                "before_hashes": before_hashes,
                "after_hashes": after_hashes,
                "message": result.message,
                "applied": result.ok,
                "backup_dir": result.backup_dir,
            },
        )
        if result.ok:
            _notify(
                root,
                conf,
                kind=kind,
                exception_type=event.exception_type,
                message=event.message,
                traceback=event.traceback,
                diff_text=diff_text,
                applied=True,
                paths_touched=result.paths_touched,
                heal_message=result.message,
            )
            _mark_healed(fp, conf.heal.cooldown_s)
            return result
        log.warning(
            "Patch failed to apply (attempt %d/%d): %s", attempt, max_attempts, result.message
        )
        last = result
        feedback = feedback_apply_failed(result.message)

    assert last is not None
    final_kind = {
        "empty diff from model": "heal_empty_diff",
    }.get(last.message, "heal_rejected" if last.policy_errors else "heal_apply_failed")
    _notify(
        root,
        conf,
        kind=final_kind,
        exception_type=event.exception_type,
        message=event.message,
        traceback=event.traceback,
        diff_text=last.diff or None,
        applied=False,
        paths_touched=last.paths_touched,
        heal_message=last.message,
    )
    _mark_healed(fp, conf.heal.cooldown_s)
    last.message = f"{last.message} (after {last.attempts} attempt(s))"
    return last


def event_from_traceback_text(tb: str, project_root: Path | None = None) -> ErrorEvent:
    """Build an ``ErrorEvent`` from raw traceback text (CLI / supervisor)."""
    etype, message = parse_exception_from_traceback(tb)
    frames: list[Frame] = []
    if project_root is not None:
        in_project = extract_frames_from_traceback(tb, project_root)
        # innermost project frame first, matching capture_from_exception()
        frames = [Frame(path=rel, line=line) for rel, line in reversed(in_project)]
    return ErrorEvent(
        exception_type=etype,
        message=message,
        traceback=tb,
        frames=frames,
        sanitized_locals={},
    )


def heal_from_traceback_text(
    tb: str,
    *,
    project_root: Path | None = None,
    mode: Mode = "apply",
    cfg: SelfHealConfig | None = None,
    client: LLMClient | None = None,
    auto_apply_override: bool | None = None,
    dry_run: bool = False,
    extra_instruction: str = "",
) -> DiffApplyResult | None:
    """CLI/offline: heal from a raw traceback string."""
    root = _resolve_root(project_root)
    return heal_exception(
        event_from_traceback_text(tb, root),
        mode=mode,
        project_root=root,
        cfg=cfg,
        client=client,
        auto_apply_override=auto_apply_override,
        dry_run=dry_run,
        extra_instruction=extra_instruction,
    )


def rollback(
    project_root: Path,
    backup_dir: Path | None = None,
    *,
    cfg: SelfHealConfig | None = None,
) -> tuple[Path, list[str]]:
    """Restore files from a backup (the latest by default); audited and notified."""
    root = project_root.resolve()
    conf = cfg or load_config_for_root(root)
    if backup_dir is None:
        backups = list_backups(root)
        if not backups:
            raise FileNotFoundError("no self-heal backups found")
        backup_dir = backups[-1]
    touched = rollback_backup(root, backup_dir)
    append_audit(
        root,
        kind="heal_rolled_back",
        data={"backup_dir": str(backup_dir), "paths": touched},
    )
    _notify(
        root,
        conf,
        kind="heal_rolled_back",
        exception_type="Rollback",
        message=f"restored {len(touched)} file(s) from {backup_dir.name}",
        paths_touched=touched,
        applied=True,
    )
    return backup_dir, touched
