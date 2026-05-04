"""Batch embeddings via LLM client."""

from __future__ import annotations

from self_heal.indexing.chunker import CodeChunk
from self_heal.llm.client import LLMClient


def chunks_to_embedding_inputs(chunks: list[CodeChunk]) -> list[str]:
    out: list[str] = []
    for c in chunks:
        out.append(f"{c.path}::{c.symbol}\n{c.content}")
    return out


def embed_chunks(client: LLMClient, chunks: list[CodeChunk]) -> list[list[float]]:
    texts = chunks_to_embedding_inputs(chunks)
    return client.embed(texts)
