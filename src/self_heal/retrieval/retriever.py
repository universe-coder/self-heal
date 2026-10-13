"""Retrieve code context for an error: traceback-anchored snippets plus vector search."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from self_heal.config import SelfHealConfig
from self_heal.indexing.store import query_similar
from self_heal.llm.client import LLMClient

log = logging.getLogger(__name__)

_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')
_MAX_SNIPPET_FRAMES = 6
_HINT_BOOST = 0.5

NO_CONTEXT_MSG = "(no code context available — run `self-heal index` to enable retrieval)"


def extract_frames_from_traceback(tb: str, project_root: Path) -> list[tuple[str, int]]:
    """``(relative_path, line)`` for traceback frames that point at files inside the project.

    Frames are returned in traceback order (outermost first), without duplicates.
    """
    root = project_root.resolve()
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for m in _FRAME_RE.finditer(tb):
        raw, line = m.group(1), int(m.group(2))
        if raw.startswith("<"):  # <string>, <frozen ...>
            continue
        p = Path(raw)
        cand = p.resolve() if p.is_absolute() else (root / p).resolve()
        try:
            rel = str(cand.relative_to(root)).replace("\\", "/")
        except ValueError:
            continue
        if not cand.is_file():
            continue
        key = (rel, line)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def extract_paths_from_traceback(tb: str, project_root: Path) -> list[str]:
    """Unique project-relative paths named in the traceback, in traceback order."""
    out: list[str] = []
    for rel, _ in extract_frames_from_traceback(tb, project_root):
        if rel not in out:
            out.append(rel)
    return out


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for s, e in sorted(ranges):
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def traceback_snippets(
    project_root: Path,
    frames: list[tuple[str, int]],
    context_lines: int,
    *,
    max_frames: int = _MAX_SNIPPET_FRAMES,
) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    """Source windows around the innermost traceback frames inside the project.

    Returns the formatted blocks and the covered ``{path: [(start, end), ...]}`` ranges.
    """
    if context_lines <= 0 or not frames:
        return [], {}
    root = project_root.resolve()
    wanted = frames[-max_frames:]  # innermost frames are the most relevant
    per_file: dict[str, list[tuple[int, int]]] = {}
    for rel, line in wanted:
        per_file.setdefault(rel, []).append((max(1, line - context_lines), line + context_lines))

    blocks: list[str] = []
    covered: dict[str, list[tuple[int, int]]] = {}
    for rel, ranges in per_file.items():
        try:
            lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for s, e in _merge_ranges(ranges):
            e = min(e, len(lines))
            if s > e:
                continue
            covered.setdefault(rel, []).append((s, e))
            numbered = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(s, e + 1))
            blocks.append(
                f"### {rel} :: lines {s}-{e} (from traceback)\n```python\n{numbered}\n```\n"
            )
    return blocks, covered


def build_retrieval_query(tb: str, extra: str = "") -> str:
    tail = tb.strip()[-4000:] if tb else ""
    return (extra + "\n\n" + tail).strip()


def _is_covered(md: dict[str, Any], covered: dict[str, list[tuple[int, int]]]) -> bool:
    path = str(md.get("path", ""))
    try:
        s, e = int(md.get("start_line", 0)), int(md.get("end_line", 0))
    except (TypeError, ValueError):
        return False
    return any(cs <= s and e <= ce for cs, ce in covered.get(path, []))


def vector_search(
    project_root: Path,
    cfg: SelfHealConfig,
    client: LLMClient,
    traceback_text: str,
    hints: list[str],
    *,
    exclude_covered: dict[str, list[tuple[int, int]]] | None = None,
) -> list[dict[str, Any]]:
    """Top-k chunks by embedding similarity; chunks from hinted files are boosted.

    Chunks that fall entirely inside ``exclude_covered`` ranges (already shown as
    traceback snippets) are dropped before the top-k cut so they do not waste budget.
    """
    q_emb = client.embed([build_retrieval_query(traceback_text)])[0]
    top_k = cfg.heal.top_k
    rows: dict[str, dict[str, Any]] = {}
    for row in query_similar(project_root, q_emb, top_k * 2):
        rows[str(row["id"])] = row
    if hints:
        for row in query_similar(project_root, q_emb, top_k, paths_hint=hints):
            rows.setdefault(str(row["id"]), row)

    hint_set = set(hints)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows.values():
        md = row.get("metadata") or {}
        if exclude_covered and _is_covered(md, exclude_covered):
            continue
        dist = row.get("distance")
        d = float(dist) if dist is not None else 1.0
        if str(md.get("path", "")) in hint_set:
            d -= _HINT_BOOST
        scored.append((d, row))
    scored.sort(key=lambda x: x[0])
    return [row for _, row in scored[:top_k]]


def retrieve_chunks(
    project_root: Path,
    cfg: SelfHealConfig,
    client: LLMClient,
    traceback_text: str,
) -> str:
    """Formatted code context for the LLM.

    Always includes source around the traceback frames that live in the project
    (no embeddings needed). When an embeddings endpoint and an index are
    available, similar chunks from the index are appended; retrieval failures
    degrade to traceback context instead of failing the heal.
    """
    root = project_root.resolve()
    frames = extract_frames_from_traceback(traceback_text, root)
    hints = [rel for rel, _ in frames]
    hints = list(dict.fromkeys(hints))

    blocks, covered = traceback_snippets(root, frames, cfg.heal.context_lines)

    rows: list[dict[str, Any]] = []
    if client.embeddings_available:
        try:
            rows = vector_search(root, cfg, client, traceback_text, hints, exclude_covered=covered)
        except Exception as e:  # noqa: BLE001
            log.warning("vector retrieval unavailable (%s); using traceback context only", e)
    else:
        log.info("no embeddings endpoint configured; using traceback context only")

    for row in rows:
        md = row.get("metadata") or {}
        path = md.get("path", "?")
        sym = md.get("symbol", "?")
        span = f"lines {md.get('start_line', '?')}-{md.get('end_line', '?')}"
        doc = row.get("document") or ""
        blocks.append(f"### {path} :: {sym} ({span})\n```python\n{doc.rstrip()}\n```\n")

    return "\n".join(blocks) if blocks else NO_CONTEXT_MSG
