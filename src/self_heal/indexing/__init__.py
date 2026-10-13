"""Code indexing (walk, chunk, embed, store)."""

from self_heal.indexing.chunker import CodeChunk, chunk_file, chunk_source
from self_heal.indexing.indexer import IndexStats, run_index
from self_heal.indexing.walker import iter_python_files

__all__ = [
    "CodeChunk",
    "IndexStats",
    "chunk_file",
    "chunk_source",
    "iter_python_files",
    "run_index",
]
