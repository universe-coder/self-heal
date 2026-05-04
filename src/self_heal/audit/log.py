"""Append-only JSONL audit log under `.self-heal/audit.jsonl`."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def audit_path(project_root: Path) -> Path:
    return project_root.resolve() / ".self-heal" / "audit.jsonl"


def append_audit(project_root: Path, *, kind: str, data: dict[str, Any]) -> None:
    p = audit_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "ts": datetime.now(UTC).isoformat(),
        "kind": kind,
        "data": data,
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_recent(project_root: Path, limit: int = 20) -> list[dict[str, Any]]:
    p = audit_path(project_root)
    if not p.is_file():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out
