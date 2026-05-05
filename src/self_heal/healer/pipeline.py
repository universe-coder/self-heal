"""End-to-end healing: retrieve → LLM → validate → apply."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Literal

from self_heal.audit.log import append_audit
from self_heal.config import SelfHealConfig, find_config, load_config_for_root
from self_heal.healer.diff import (
    DiffApplyResult,
    apply_patch,
    extract_diff_from_response,
    file_hashes,
)
from self_heal.healer.planner import build_healing_prompt
from self_heal.llm.client import LLMClient
from self_heal.notifications.dispatcher import notify
from self_heal.retrieval.retriever import retrieve_chunks
from self_heal.runtime.capture import ErrorEvent, format_event_for_llm

log = logging.getLogger(__name__)

Mode = Literal["suggest", "apply", "auto"]


def _fingerprint_path(event: ErrorEvent) -> str | None:
    if not event.frames:
        return None
    return event.frames[-1].path


def _resolve_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    found = find_config()
    if found is not None:
        return found.parent.resolve()
    return Path.cwd().resolve()


def heal_exception(
    event: ErrorEvent,
    *,
    mode: Mode = "suggest",
    project_root: Path | None = None,
    cfg: SelfHealConfig | None = None,
    client: LLMClient | None = None,
    auto_apply_override: bool | None = None,
    dry_run: bool = False,
) -> DiffApplyResult | None:
    """Run healing for a captured `ErrorEvent`."""
    root = _resolve_root(project_root)
    conf = cfg or load_config_for_root(root)

    append_audit(
        root,
        kind="error_captured",
        data={"exception": event.exception_type, "message": event.message},
    )
    with contextlib.suppress(Exception):
        notify(
            root,
            conf,
            kind="error_captured",
            exception_type=event.exception_type,
            message=event.message,
            traceback=event.traceback,
            fingerprint_path=_fingerprint_path(event),
        )

    llm = client or LLMClient(conf)
    if not llm.is_configured:
        log.warning("Skipping heal: no API key (%s)", conf.llm.api_key_env)
        return None

    tb_text = format_event_for_llm(event)
    chunks = retrieve_chunks(root, conf, llm, event.traceback)
    system, user = build_healing_prompt(tb_text, chunks)
    raw = llm.chat(system, user)
    diff_text = extract_diff_from_response(raw)
    if not diff_text.strip():
        append_audit(
            root,
            kind="heal_empty_diff",
            data={
                "exception": event.exception_type,
                "message": event.message,
                "raw_response_preview": raw[:2000],
            },
        )
        with contextlib.suppress(Exception):
            notify(
                root,
                conf,
                kind="heal_empty_diff",
                exception_type=event.exception_type,
                message=event.message,
                traceback=event.traceback,
                raw_response_preview=raw[:2000],
                applied=False,
            )
        return DiffApplyResult(ok=False, message="empty diff from model", paths_touched=[])

    auto = conf.heal.auto_apply if auto_apply_override is None else auto_apply_override
    if mode == "suggest":
        auto = False
    if mode == "auto":
        auto = True

    before_hashes = file_hashes(root, _paths_from_diff(diff_text, root))

    if dry_run or not auto:
        append_audit(
            root,
            kind="heal_diff_proposed",
            data={
                "exception": event.exception_type,
                "diff": diff_text,
                "before_hashes": before_hashes,
                "applied": False,
            },
        )
        with contextlib.suppress(Exception):
            notify(
                root,
                conf,
                kind="heal_diff_proposed",
                exception_type=event.exception_type,
                message=event.message,
                traceback=event.traceback,
                diff_text=diff_text,
                applied=False,
                paths_touched=_paths_from_diff(diff_text, root),
            )
        log.info("Proposed diff (not applied); use CLI or auto_apply to apply")
        return DiffApplyResult(ok=True, message="diff proposed (not applied)", paths_touched=[])

    result = apply_patch(root, diff_text, conf, do_backup=True)
    after_hashes = file_hashes(root, result.paths_touched)
    kind = "heal_applied" if result.ok else "heal_apply_failed"
    append_audit(
        root,
        kind=kind,
        data={
            "exception": event.exception_type,
            "diff": diff_text,
            "before_hashes": before_hashes,
            "after_hashes": after_hashes,
            "message": result.message,
            "applied": result.ok,
        },
    )
    with contextlib.suppress(Exception):
        notify(
            root,
            conf,
            kind=kind,
            exception_type=event.exception_type,
            message=event.message,
            traceback=event.traceback,
            diff_text=diff_text,
            applied=result.ok,
            paths_touched=result.paths_touched,
            heal_message=result.message,
        )
    return result


def heal_from_traceback_text(
    tb: str,
    *,
    project_root: Path | None = None,
    mode: Mode = "apply",
    cfg: SelfHealConfig | None = None,
    client: LLMClient | None = None,
    auto_apply_override: bool | None = None,
    dry_run: bool = False,
) -> DiffApplyResult | None:
    """CLI/offline: heal from raw traceback string."""
    from self_heal.runtime.capture import ErrorEvent

    ev = ErrorEvent(
        exception_type="Unknown",
        message="",
        traceback=tb,
        frames=[],
        sanitized_locals={},
    )
    return heal_exception(
        ev,
        mode=mode,
        project_root=project_root,
        cfg=cfg,
        client=client,
        auto_apply_override=auto_apply_override,
        dry_run=dry_run,
    )


def _paths_from_diff(diff_text: str, project_root: Path) -> list[str]:
    from io import StringIO

    from unidiff import PatchSet

    from self_heal.healer.policy import normalize_diff_path

    try:
        ps = PatchSet(StringIO(diff_text))
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for pf in ps:
        t = normalize_diff_path(pf.target_file or pf.path)
        if t:
            out.append(t)
    return out
