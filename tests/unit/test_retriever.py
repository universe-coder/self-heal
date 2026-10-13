"""Retrieval: traceback-anchored snippets and graceful degradation without embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from self_heal.config import HealConfig, SelfHealConfig
from self_heal.retrieval import retriever as retriever_mod
from self_heal.retrieval.retriever import (
    NO_CONTEXT_MSG,
    extract_frames_from_traceback,
    extract_paths_from_traceback,
    retrieve_chunks,
    traceback_snippets,
)


def _project(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text(
        "\n".join(f"line{i} = {i}" for i in range(1, 101)) + "\n", encoding="utf-8"
    )
    (tmp_path / "app" / "util.py").write_text("def helper():\n    return 1 / 0\n", encoding="utf-8")
    return tmp_path


def _tb(root: Path) -> str:
    return (
        "Traceback (most recent call last):\n"
        f'  File "{root / "app" / "main.py"}", line 50, in <module>\n'
        "    helper()\n"
        '  File "app/util.py", line 2, in helper\n'
        "    return 1 / 0\n"
        '  File "/usr/lib/python3/site-packages/other.py", line 9, in x\n'
        '  File "<frozen importlib._bootstrap>", line 1, in y\n'
        '  File "app/missing.py", line 3, in z\n'
        "ZeroDivisionError: division by zero\n"
    )


def test_extract_frames_only_project_files(tmp_path: Path) -> None:
    root = _project(tmp_path)
    frames = extract_frames_from_traceback(_tb(root), root)
    assert frames == [("app/main.py", 50), ("app/util.py", 2)]
    assert extract_paths_from_traceback(_tb(root), root) == ["app/main.py", "app/util.py"]


def test_traceback_snippets_window_and_merge(tmp_path: Path) -> None:
    root = _project(tmp_path)
    blocks, covered = traceback_snippets(root, [("app/main.py", 50), ("app/main.py", 55)], 5)
    assert covered == {"app/main.py": [(45, 60)]}  # overlapping windows merged
    assert len(blocks) == 1
    assert "lines 45-60" in blocks[0]
    assert "   50  line50 = 50" in blocks[0]
    assert "line44" not in blocks[0] and "line61" not in blocks[0]

    blocks, _ = traceback_snippets(root, [("app/util.py", 2)], 40)
    assert "lines 1-2" in blocks[0]
    assert traceback_snippets(root, [("app/util.py", 2)], 0) == ([], {})


def test_retrieve_without_embeddings_uses_traceback_context(tmp_path: Path) -> None:
    root = _project(tmp_path)
    client = MagicMock()
    client.embeddings_available = False
    cfg = SelfHealConfig(heal=HealConfig(context_lines=3))
    ctx = retrieve_chunks(root, cfg, client, _tb(root))
    assert "app/util.py :: lines 1-2" in ctx
    assert "app/main.py :: lines 47-53" in ctx
    client.embed.assert_not_called()


def test_retrieve_degrades_when_vector_search_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    client = MagicMock()
    client.embeddings_available = True
    client.embed.side_effect = RuntimeError("endpoint down")
    ctx = retrieve_chunks(root, SelfHealConfig(), client, _tb(root))
    assert "from traceback" in ctx


def test_retrieve_merges_vector_results_and_skips_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    client = MagicMock()
    client.embeddings_available = True
    client.embed.return_value = [[0.0, 1.0]]

    def fake_query(project_root: Path, emb: Any, top_k: int, paths_hint: Any = None) -> list[dict]:  # type: ignore[type-arg]
        rows = [
            {
                "id": "covered",
                "document": "line50 = 50",
                "metadata": {
                    "path": "app/main.py",
                    "symbol": "<module>",
                    "start_line": 48,
                    "end_line": 52,
                },
                "distance": 0.1,
            },
            {
                "id": "other",
                "document": "def far():\n    pass",
                "metadata": {"path": "lib/far.py", "symbol": "far", "start_line": 1, "end_line": 2},
                "distance": 0.2,
            },
            {
                "id": "hinted",
                "document": "line90 = 90",
                "metadata": {
                    "path": "app/main.py",
                    "symbol": "<module>",
                    "start_line": 90,
                    "end_line": 95,
                },
                "distance": 0.6,
            },
        ]
        if paths_hint:
            return [r for r in rows if r["metadata"]["path"] in paths_hint]
        return rows

    monkeypatch.setattr(retriever_mod, "query_similar", fake_query)
    cfg = SelfHealConfig(heal=HealConfig(context_lines=3, top_k=2))
    ctx = retrieve_chunks(root, cfg, client, _tb(root))
    assert "lib/far.py :: far" in ctx
    assert "lines 90-95" in ctx  # hinted file boosted above its raw distance
    assert ctx.count("line50 = 50") == 1  # covered chunk not repeated


def test_no_context_message_when_nothing_found(tmp_path: Path) -> None:
    client = MagicMock()
    client.embeddings_available = False
    assert retrieve_chunks(tmp_path, SelfHealConfig(), client, "nothing useful") == NO_CONTEXT_MSG
