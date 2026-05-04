"""Chroma persistent vector store under `.self-heal/chroma`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction

from self_heal.indexing.chunker import CodeChunk

_DEFAULT_EMBED_DIM = 1536  # text-embedding-3-small default


class ListEmbeddingFunction(EmbeddingFunction):  # type: ignore[misc, type-arg, override]
    """Pass precomputed embeddings into Chroma."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @staticmethod
    def name() -> str:
        return "self_heal_static_embeddings"

    def __call__(self, input: list[str]) -> list[list[float]]:  # type: ignore[override]  # noqa: A002
        # Not used when adding with embeddings= ; implemented for interface
        return [[0.0] * self._dim for _ in input]


def get_collection(project_root: Path, collection_name: str = "self_heal_code"):  # type: ignore[no-untyped-def]
    root = project_root.resolve()
    persist = root / ".self-heal" / "chroma"
    persist.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist))
    # dimension placeholder; we always supply embeddings explicitly
    emb_fn = ListEmbeddingFunction(_DEFAULT_EMBED_DIM)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"project": str(root)},
    )


def upsert_chunks(
    project_root: Path,
    chunks: list[CodeChunk],
    embeddings: list[list[float]],
    file_mtime: dict[str, float],
) -> None:
    if not chunks:
        return
    col = get_collection(project_root)
    ids = [c.chunk_id for c in chunks]
    documents = [c.content for c in chunks]
    metadatas = [
        {
            "path": c.path,
            "symbol": c.symbol,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "hash": c.content_hash,
            "mtime": file_mtime.get(c.path, 0.0),
        }
        for c in chunks
    ]
    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def delete_paths(project_root: Path, paths: list[str]) -> None:
    if not paths:
        return
    col = get_collection(project_root)
    # Fetch ids where metadata.path in paths
    # Chroma where filter: use $in
    res = col.get(where={"path": {"$in": paths}}, include=[])
    prev_ids = res.get("ids") or []
    if prev_ids:
        col.delete(ids=prev_ids)


def query_similar(
    project_root: Path,
    query_embedding: list[float],
    top_k: int,
    paths_hint: list[str] | None = None,
) -> list[dict]:
    col = get_collection(project_root)
    kwargs: dict[str, object] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if paths_hint and len(paths_hint) == 1:
        kwargs["where"] = {"path": {"$eq": paths_hint[0]}}

    raw = col.query(**kwargs)
    out: list[dict[str, Any]] = []
    ids_list = raw.get("ids") or [[]]
    docs_list = raw.get("documents") or [[]]
    meta_list = raw.get("metadatas") or [[]]
    dist_list = raw.get("distances") or [[]]
    for i, cid in enumerate(ids_list[0] if ids_list else []):
        out.append(
            {
                "id": cid,
                "document": docs_list[0][i] if docs_list and docs_list[0] else "",
                "metadata": meta_list[0][i] if meta_list and meta_list[0] else {},
                "distance": dist_list[0][i] if dist_list and dist_list[0] else None,
            }
        )
    return out


def load_index_manifest(project_root: Path) -> dict[str, str]:
    """path -> content hash for incremental indexing."""
    manifest_path = project_root / ".self-heal" / "manifest.json"
    if not manifest_path.is_file():
        return {}
    import json

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def save_index_manifest(project_root: Path, manifest: dict[str, str]) -> None:
    import json

    d = project_root / ".self-heal"
    d.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=0, sort_keys=True)
    (d / "manifest.json").write_text(payload, encoding="utf-8")
