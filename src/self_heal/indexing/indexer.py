"""Incremental indexing: walk files, chunk, embed, upsert to Chroma."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from self_heal.config import SelfHealConfig
from self_heal.indexing.chunker import CodeChunk, chunk_file
from self_heal.indexing.embedder import embed_chunks
from self_heal.indexing.store import (
    IndexManifest,
    delete_paths,
    load_index_manifest,
    reset_collection,
    save_index_manifest,
    upsert_chunks,
)
from self_heal.indexing.walker import iter_python_files
from self_heal.llm.client import EMBEDDING_UNSUPPORTED_MSG, EmbeddingsUnavailableError, LLMClient

log = logging.getLogger(__name__)


@dataclass
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_removed: int = 0
    chunks_indexed: int = 0
    full_reindex: bool = False

    @property
    def up_to_date(self) -> bool:
        return self.files_indexed == 0 and self.files_removed == 0


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_index(
    project_root: Path,
    cfg: SelfHealConfig,
    client: LLMClient | None = None,
    *,
    force: bool = False,
) -> IndexStats:
    """Index changed files, drop deleted ones, and rebuild when the embedding model changed."""
    root = project_root.resolve()
    llm = client or LLMClient(cfg)
    if not llm.embeddings_available:
        if not cfg.llm.embeddings_available:
            raise EmbeddingsUnavailableError(EMBEDDING_UNSUPPORTED_MSG)
        raise EmbeddingsUnavailableError(
            f"Cannot index without an API key for embeddings: set "
            f"{cfg.llm.resolved_embedding_api_key_env}"
        )

    stats = IndexStats()
    manifest = load_index_manifest(root)
    model_changed = (
        manifest.embedding_model is not None and manifest.embedding_model != cfg.llm.embedding_model
    )
    if force or model_changed:
        if model_changed:
            log.info(
                "Embedding model changed (%s -> %s); rebuilding index",
                manifest.embedding_model,
                cfg.llm.embedding_model,
            )
        reset_collection(root)
        manifest = IndexManifest()
        stats.full_reindex = True

    files = iter_python_files(
        root, cfg.index.roots, cfg.index.exclude, respect_gitignore=cfg.index.respect_gitignore
    )
    current: dict[str, str] = {}
    for fp in files:
        rel = str(fp.relative_to(root)).replace("\\", "/")
        current[rel] = _file_hash(fp)
    stats.files_scanned = len(current)

    removed = sorted(set(manifest.files) - set(current))
    changed = [rel for rel, digest in current.items() if manifest.files.get(rel) != digest]

    if removed:
        delete_paths(root, removed)
        stats.files_removed = len(removed)
        log.info("Removed %d deleted files from the index", len(removed))

    new_manifest = IndexManifest(
        files={rel: digest for rel, digest in current.items() if rel not in changed},
        embedding_model=cfg.llm.embedding_model,
        embedding_base_url=cfg.llm.resolved_embedding_base_url,
    )

    if changed:
        delete_paths(root, changed)
        all_chunks: list[CodeChunk] = []
        mtimes: dict[str, float] = {}
        for rel in changed:
            fp = root / rel
            chunks = chunk_file(
                fp, root, max_lines=cfg.index.chunk_max_lines, overlap=cfg.index.chunk_overlap_lines
            )
            all_chunks.extend(chunks)
            mtimes[rel] = fp.stat().st_mtime
        log.info("Embedding %d chunks from %d files…", len(all_chunks), len(changed))
        embeddings = embed_chunks(llm, all_chunks)
        if len(embeddings) != len(all_chunks):
            raise RuntimeError("Embedding count mismatch")
        upsert_chunks(root, all_chunks, embeddings, mtimes)
        for rel in changed:
            new_manifest.files[rel] = current[rel]
        stats.files_indexed = len(changed)
        stats.chunks_indexed = len(all_chunks)

    save_index_manifest(root, new_manifest)
    if stats.up_to_date:
        log.info("Index up to date (%d files tracked)", len(current))
    else:
        log.info("Indexed %d files, %d chunks", stats.files_indexed, stats.chunks_indexed)
    return stats
