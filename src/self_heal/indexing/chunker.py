"""Split Python files into AST-aware chunks.

Each top-level and nested function becomes a chunk (long ones are split into
overlapping windows so nothing is dropped), classes contribute their header
(docstring, class attributes) and module-level code outside any definition is
collected into ``<module>`` chunks. Method symbols are qualified (``Class.method``).
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

MODULE_SYMBOL = "<module>"
SYNTAX_ERROR_SYMBOL = "__syntax_error__"

_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    path: str  # relative posix path
    symbol: str  # e.g. <module>, ClassName.method, function_name
    start_line: int
    end_line: int
    content: str
    content_hash: str


def _hash_content(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _windows(
    lines: list[str],
    start: int,
    end: int,
    max_lines: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    """Split the 1-based inclusive range into windows of at most ``max_lines`` lines."""
    start = max(start, 1)
    end = min(max(end, start), len(lines))
    if end < start:
        return []
    step = max(max_lines - overlap, 1)
    out: list[tuple[int, int, str]] = []
    s = start
    while True:
        e = min(s + max_lines - 1, end)
        out.append((s, e, "".join(lines[s - 1 : e])))
        if e >= end:
            return out
        s += step


def _node_start(node: ast.AST) -> int:
    start = getattr(node, "lineno", 1) or 1
    for deco in getattr(node, "decorator_list", []) or []:
        start = min(start, getattr(deco, "lineno", start) or start)
    return int(start)


def _node_end(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", None) or _node_start(node))


class _Collector:
    def __init__(self, lines: list[str], rel_path: str, max_lines: int, overlap: int) -> None:
        self.lines = lines
        self.rel_path = rel_path
        self.max_lines = max_lines
        self.overlap = overlap
        self.chunks: list[CodeChunk] = []
        self._seen_ids: set[str] = set()

    def add_range(self, symbol: str, start: int, end: int) -> None:
        for s, e, text in _windows(self.lines, start, end, self.max_lines, self.overlap):
            if not text.strip():
                continue
            h = _hash_content(text)
            cid = f"{self.rel_path}:{symbol}:{s}:{h}"
            if cid in self._seen_ids:
                continue
            self._seen_ids.add(cid)
            self.chunks.append(
                CodeChunk(
                    chunk_id=cid,
                    path=self.rel_path,
                    symbol=symbol,
                    start_line=s,
                    end_line=e,
                    content=text,
                    content_hash=h,
                )
            )

    def visit_body(self, body: list[ast.stmt], scope: list[str]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                qual = ".".join([*scope, node.name])
                self.add_range(qual, _node_start(node), _node_end(node))
                self.visit_body(node.body, [*scope, node.name])
            elif isinstance(node, ast.ClassDef):
                qual = ".".join([*scope, node.name])
                self.add_range(qual, _node_start(node), self._class_header_end(node))
                self.visit_body(node.body, [*scope, node.name])

    @staticmethod
    def _class_header_end(node: ast.ClassDef) -> int:
        """Last line of the class before its first nested definition (or the class end)."""
        nested = [c for c in node.body if isinstance(c, _DEF_TYPES)]
        if not nested:
            return _node_end(node)
        first = min(_node_start(c) for c in nested)
        return max(first - 1, _node_start(node))


def _module_remainder(tree: ast.Module, n_lines: int) -> list[tuple[int, int]]:
    """1-based inclusive line ranges not covered by any top-level definition."""
    covered = sorted((_node_start(n), _node_end(n)) for n in tree.body if isinstance(n, _DEF_TYPES))
    ranges: list[tuple[int, int]] = []
    cursor = 1
    for s, e in covered:
        if s > cursor:
            ranges.append((cursor, s - 1))
        cursor = max(cursor, e + 1)
    if cursor <= n_lines:
        ranges.append((cursor, n_lines))
    return ranges


def chunk_file(path: Path, project_root: Path, max_lines: int, overlap: int) -> list[CodeChunk]:
    rel = path.resolve().relative_to(project_root.resolve())
    rel_str = str(rel).replace("\\", "/")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    return chunk_source(text, rel_str, max_lines, overlap, lines=lines)


def chunk_source(
    text: str,
    rel_path: str,
    max_lines: int,
    overlap: int,
    *,
    lines: list[str] | None = None,
) -> list[CodeChunk]:
    """Chunk Python source text; ``rel_path`` is used for ids and metadata."""
    lines = text.splitlines(keepends=True) if lines is None else lines
    collector = _Collector(lines, rel_path, max_lines, overlap)
    try:
        tree = ast.parse(text, filename=rel_path)
    except (SyntaxError, ValueError):
        collector.add_range(SYNTAX_ERROR_SYMBOL, 1, len(lines))
        return collector.chunks

    collector.visit_body(tree.body, [])
    for s, e in _module_remainder(tree, len(lines)):
        collector.add_range(MODULE_SYMBOL, s, e)
    collector.chunks.sort(key=lambda c: (c.start_line, c.end_line, c.symbol))
    return collector.chunks
