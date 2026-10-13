"""Chroma persistent vector store under `.self-heal/chroma`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction

from self_heal.indexing.chunker import CodeChunk

COLLECTION_NAME = "self_heal_code"
MANIFEST_VERSION = 2
_UPSERT_BATCH = 500


class ListEmbeddingFunction(EmbeddingFunction):  # type: ignore[misc, type-arg]
    """Placeholder embedding function: self-heal always supplies embeddings explicitly."""

    def __init__(self, dim: int = 0) -> None:
        self._dim = dim

    @staticmethod
    def name() -> str:
        return "self_heal_precomputed"

    def get_config(self) -> dict[str, Any]:
        return {"dim": self._dim}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> ListEmbeddingFunction:
        return ListEmbeddingFunction(int(config.get("dim", 0) or 0))

    def __call__(self, input: list[str]) -> list[list[float]]:  # type: ignore[override]  # noqa: A002
        raise RuntimeError("self-heal supplies embeddings explicitly; this should not be called")


def chroma_dir(project_root: Path) -> Path:
    return project_root.resolve() / ".self-heal" / "chroma"


def _client(project_root: Path) -> chromadb.ClientAPI:
    persist = chroma_dir(project_root)
    persist.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist))


def get_collection(project_root: Path, collection_name: str = COLLECTION_NAME) -> Any:
    root = project_root.resolve()
    return _client(root).get_or_create_collection(
        name=collection_name,
        embedding_function=ListEmbeddingFunction(),
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"project": str(root)},
    )


def reset_collection(project_root: Path, collection_name: str = COLLECTION_NAME) -> None:
    """Drop all indexed chunks (used when the embedding model changes or on --force)."""
    client = _client(project_root)
    if any(c.name == collection_name for c in client.list_collections()):
        client.delete_collection(collection_name)


def collection_count(project_root: Path) -> int:
    if not chroma_dir(project_root).is_dir():
        return 0
    return int(get_collection(project_root).count())


def upsert_chunks(
    project_root: Path,
    chunks: list[CodeChunk],
    embeddings: list[list[float]],
    file_mtime: dict[str, float],
) -> None:
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    col = get_collection(project_root)
    for i in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[i : i + _UPSERT_BATCH]
        col.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings[i : i + _UPSERT_BATCH],
            documents=[c.content for c in batch],
            metadatas=[
                {
                    "path": c.path,
                    "symbol": c.symbol,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "hash": c.content_hash,
                    "mtime": file_mtime.get(c.path, 0.0),
                }
                for c in batch
            ],
        )


def delete_paths(project_root: Path, paths: list[str]) -> int:
    """Delete every chunk whose metadata.path is in ``paths``; returns the number removed."""
    if not paths:
        return 0
    col = get_collection(project_root)
    removed = 0
    for i in range(0, len(paths), _UPSERT_BATCH):
        batch = paths[i : i + _UPSERT_BATCH]
        res = col.get(where={"path": {"$in": batch}}, include=[])
        ids = list(res.get("ids") or [])
        if ids:
            col.delete(ids=ids)
            removed += len(ids)
    return removed


def query_similar(
    project_root: Path,
    query_embedding: list[float],
    top_k: int,
    paths_hint: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Nearest chunks; with ``paths_hint`` only chunks from those files are considered."""
    if top_k <= 0:
        return []
    col = get_collection(project_root)
    if col.count() == 0:
        return []
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, int(col.count())),
        "include": ["documents", "metadatas", "distances"],
    }
    if paths_hint:
        kwargs["where"] = {"path": {"$in": list(paths_hint)}}

    raw = col.query(**kwargs)
    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0] or []
    metas = (raw.get("metadatas") or [[]])[0] or []
    dists = (raw.get("distances") or [[]])[0] or []
    out: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        out.append(
            {
                "id": cid,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return out


@dataclass
class IndexManifest:
    """What is indexed: file hashes plus the embedding model the vectors came from."""

    files: dict[str, str] = field(default_factory=dict)
    embedding_model: str | None = None
    embedding_base_url: str | None = None


def manifest_path(project_root: Path) -> Path:
    return project_root.resolve() / ".self-heal" / "manifest.json"


def load_index_manifest(project_root: Path) -> IndexManifest:
    p = manifest_path(project_root)
    if not p.is_file():
        return IndexManifest()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return IndexManifest()
    if not isinstance(data, dict):
        return IndexManifest()
    if data.get("version") == MANIFEST_VERSION and isinstance(data.get("files"), dict):
        return IndexManifest(
            files={str(k): str(v) for k, v in data["files"].items()},
            embedding_model=data.get("embedding_model"),
            embedding_base_url=data.get("embedding_base_url"),
        )
    # version 1: a flat {path: hash} mapping
    return IndexManifest(files={str(k): str(v) for k, v in data.items() if isinstance(v, str)})


def save_index_manifest(project_root: Path, manifest: IndexManifest) -> None:
    p = manifest_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MANIFEST_VERSION,
        "embedding_model": manifest.embedding_model,
        "embedding_base_url": manifest.embedding_base_url,
        "files": dict(sorted(manifest.files.items())),
    }
    p.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
