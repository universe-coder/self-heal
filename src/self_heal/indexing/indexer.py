"""Incremental indexing: walk files, chunk, embed, upsert to Chroma."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from self_heal.config import SelfHealConfig
from self_heal.indexing.chunker import chunk_file
from self_heal.indexing.embedder import embed_chunks
from self_heal.indexing.store import (
    delete_paths,
    load_index_manifest,
    save_index_manifest,
    upsert_chunks,
)
from self_heal.indexing.walker import iter_python_files
from self_heal.llm.client import LLMClient

log = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    data = path.read_bytes()
    h.update(data)
    return h.hexdigest()


def run_index(
    project_root: Path,
    cfg: SelfHealConfig,
    client: LLMClient | None = None,
) -> tuple[int, int]:
    """Returns (files_indexed, chunks_total)."""
    root = project_root.resolve()
    llm = client or LLMClient(cfg)
    if not llm.is_configured:
        msg = "Cannot index without API key for embeddings"
        raise RuntimeError(msg)

    manifest = load_index_manifest(root)
    files = iter_python_files(root, cfg.index.roots, cfg.index.exclude)

    changed_paths: list[str] = []
    new_manifest: dict[str, str] = dict(manifest)

    for fp in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        digest = _file_hash(fp)
        if manifest.get(rel) == digest:
            continue
        changed_paths.append(rel)
        new_manifest[rel] = digest

    if not changed_paths:
        log.info("Index up to date (%d files tracked)", len(files))
        return 0, 0

    # Remove stale chunks for changed files only
    delete_paths(root, changed_paths)

    total_chunks = 0
    indexed_files = 0
    for rel in changed_paths:
        fp = root / rel
        if not fp.is_file():
            new_manifest.pop(rel, None)
            continue
        chunks = chunk_file(
            fp,
            root,
            max_lines=cfg.index.chunk_max_lines,
            overlap=cfg.index.chunk_overlap_lines,
        )
        if not chunks:
            continue
        embeddings = embed_chunks(llm, chunks)
        if len(embeddings) != len(chunks):
            msg = "Embedding count mismatch"
            raise RuntimeError(msg)
        mtime = fp.stat().st_mtime
        m_map = {rel: mtime}
        upsert_chunks(root, chunks, embeddings, m_map)
        total_chunks += len(chunks)
        indexed_files += 1

    save_index_manifest(root, new_manifest)
    log.info("Indexed %d files, %d chunks", indexed_files, total_chunks)
    return indexed_files, total_chunks
