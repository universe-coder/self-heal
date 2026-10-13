"""Incremental indexing against a real (temporary) Chroma store with fake embeddings."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from self_heal.config import IndexConfig, LLMConfig, SelfHealConfig
from self_heal.indexing.indexer import run_index
from self_heal.indexing.store import (
    collection_count,
    load_index_manifest,
    manifest_path,
    query_similar,
    save_index_manifest,
)
from self_heal.llm.client import EmbeddingsUnavailableError

pytest.importorskip("chromadb")

DIM = 8


def _fake_client(*, available: bool = True) -> MagicMock:
    client = MagicMock()
    client.embeddings_available = available

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * DIM
            vec[len(t) % DIM] = 1.0
            out.append(vec)
        return out

    client.embed = MagicMock(side_effect=embed)
    return client


def _cfg() -> SelfHealConfig:
    return SelfHealConfig(index=IndexConfig(roots=["src/"]))


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    _write(tmp_path, "src/a.py", "def a():\n    return 1\n")
    _write(tmp_path, "src/b.py", "def b():\n    return 2\n")
    return tmp_path


def test_index_is_incremental(project: Path) -> None:
    client = _fake_client()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        stats = run_index(project, _cfg(), client)
    assert (stats.files_scanned, stats.files_indexed, stats.chunks_indexed) == (2, 2, 2)
    assert client.embed.call_count == 1  # all chunks embedded in one call
    assert collection_count(project) == 2
    manifest = load_index_manifest(project)
    assert set(manifest.files) == {"src/a.py", "src/b.py"}
    assert manifest.embedding_model == "text-embedding-3-small"

    stats = run_index(project, _cfg(), client)
    assert stats.up_to_date and stats.files_indexed == 0

    _write(project, "src/a.py", "def a():\n    return 10\n")
    stats = run_index(project, _cfg(), client)
    assert stats.files_indexed == 1 and stats.files_removed == 0
    assert collection_count(project) == 2  # old chunk replaced, not duplicated

    (project / "src" / "b.py").unlink()
    stats = run_index(project, _cfg(), client)
    assert stats.files_removed == 1 and stats.files_indexed == 0
    assert collection_count(project) == 1
    assert set(load_index_manifest(project).files) == {"src/a.py"}


def test_embedding_model_change_triggers_full_rebuild(project: Path) -> None:
    client = _fake_client()
    run_index(project, _cfg(), client)
    cfg2 = SelfHealConfig(index=IndexConfig(roots=["src/"]), llm=LLMConfig(embedding_model="other"))
    stats = run_index(project, cfg2, client)
    assert stats.full_reindex and stats.files_indexed == 2
    assert load_index_manifest(project).embedding_model == "other"
    assert collection_count(project) == 2


def test_force_reindexes_everything(project: Path) -> None:
    client = _fake_client()
    run_index(project, _cfg(), client)
    stats = run_index(project, _cfg(), client, force=True)
    assert stats.full_reindex and stats.files_indexed == 2


def test_legacy_flat_manifest_is_read(project: Path) -> None:
    mp = manifest_path(project)
    mp.parent.mkdir(parents=True)
    mp.write_text('{"src/a.py": "deadbeef"}', encoding="utf-8")
    m = load_index_manifest(project)
    assert m.files == {"src/a.py": "deadbeef"} and m.embedding_model is None
    save_index_manifest(project, m)
    assert load_index_manifest(project).files == {"src/a.py": "deadbeef"}


def test_index_requires_embeddings(project: Path) -> None:
    with pytest.raises(EmbeddingsUnavailableError):
        run_index(project, _cfg(), _fake_client(available=False))


def test_query_similar_filters_by_path_hint(project: Path) -> None:
    client = _fake_client()
    run_index(project, _cfg(), client)
    q = client.embed(["def b():\n    return 2\n"])[0]
    rows = query_similar(project, q, 5)
    assert len(rows) == 2
    rows = query_similar(project, q, 5, paths_hint=["src/a.py"])
    assert [r["metadata"]["path"] for r in rows] == ["src/a.py"]
    assert query_similar(project, q, 0) == []
