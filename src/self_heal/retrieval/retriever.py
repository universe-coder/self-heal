"""Retrieve code chunks for an error: vector search + path hints from traceback."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from self_heal.config import SelfHealConfig
from self_heal.indexing.store import query_similar
from self_heal.llm.client import LLMClient


def extract_paths_from_traceback(tb: str, project_root: Path) -> list[str]:
    """Extract relative paths from 'File \"...\", line' lines that exist under project_root."""
    root = project_root.resolve()
    rels: list[str] = []
    for m in re.finditer(r'File "([^"]+)", line', tb):
        raw = m.group(1)
        p = Path(raw)
        cand = (root / p).resolve() if not p.is_absolute() else p.resolve()
        try:
            rel = cand.relative_to(root)
            rels.append(str(rel).replace("\\", "/"))
        except ValueError:
            continue
    # unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for r in rels:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def build_retrieval_query(tb: str, extra: str = "") -> str:
    tail = tb.strip()[-4000:] if tb else ""
    return (extra + "\n\n" + tail).strip()


def retrieve_chunks(
    project_root: Path,
    cfg: SelfHealConfig,
    client: LLMClient,
    traceback_text: str,
) -> str:
    """Return formatted context string for the LLM."""
    root = project_root.resolve()
    hints = extract_paths_from_traceback(traceback_text, root)
    q = build_retrieval_query(traceback_text)
    q_emb = client.embed([q])[0]

    top_k = cfg.heal.top_k
    # Hybrid: query more when filtering by path hints in Python
    n_fetch = top_k * 3 if hints else top_k

    try:
        raw = query_similar(root, q_emb, n_fetch, paths_hint=hints[:1] if len(hints) == 1 else None)
    except Exception:  # noqa: BLE001
        raw = []

    # Boost results whose path appears in hints
    hint_set = set(hints)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in raw:
        md = row.get("metadata") or {}
        path = str(md.get("path", ""))
        dist = row.get("distance")
        d = float(dist) if dist is not None else 1.0
        if path in hint_set:
            d -= 0.5  # boost
        scored.append((d, row))

    scored.sort(key=lambda x: x[0])
    picked = [x[1] for x in scored[:top_k]]

    blocks: list[str] = []
    for row in picked:
        md = row.get("metadata") or {}
        path = md.get("path", "?")
        sym = md.get("symbol", "?")
        doc = row.get("document") or ""
        blocks.append(f"### {path} :: {sym}\n```python\n{doc}\n```\n")
    return "\n".join(blocks) if blocks else "(no indexed chunks — run `self-heal index`)"
