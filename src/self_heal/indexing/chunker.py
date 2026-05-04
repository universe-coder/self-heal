"""Split Python files into AST-aware chunks (functions/classes/module remainder)."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    path: str  # relative posix path
    symbol: str  # e.g. module, ClassName.method, function_name
    start_line: int
    end_line: int
    content: str
    content_hash: str


def _hash_content(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _lines_with_overlap(
    lines: list[str],
    start: int,
    end: int,
    max_lines: int,
    overlap: int,
) -> tuple[int, int, str]:
    """1-based inclusive start/end line numbers; returns adjusted range and text."""
    if start < 1:
        start = 1
    if end < start:
        end = start
    span = end - start + 1
    if span <= max_lines:
        chunk_lines = lines[start - 1 : end]
        return start, end, "".join(chunk_lines)

    # window from error region toward start
    win_end = end
    win_start = max(start, win_end - max_lines + 1)
    if overlap > 0 and win_start > start:
        win_start = max(start, win_start - overlap)
    chunk_lines = lines[win_start - 1 : win_end]
    return win_start, win_end, "".join(chunk_lines)


class ChunkVisitor(ast.NodeVisitor):
    def __init__(
        self,
        source: str,
        rel_path: str,
        max_lines: int,
        overlap: int,
    ) -> None:
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.rel_path = rel_path
        self.max_lines = max_lines
        self.overlap = overlap
        self.chunks: list[CodeChunk] = []

    def _add_chunk(self, name: str, node: ast.AST) -> None:
        start = getattr(node, "lineno", 1) or 1
        end = getattr(node, "end_lineno", start) or start
        s, e, text = _lines_with_overlap(self.lines, start, end, self.max_lines, self.overlap)
        h = _hash_content(text)
        cid = f"{self.rel_path}:{name}:{h}"
        self.chunks.append(
            CodeChunk(
                chunk_id=cid,
                path=self.rel_path,
                symbol=name,
                start_line=s,
                end_line=e,
                content=text,
                content_hash=h,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_chunk(node.name, node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_chunk(node.name, node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_chunk(node.name, node)
        self.generic_visit(node)


def chunk_file(path: Path, project_root: Path, max_lines: int, overlap: int) -> list[CodeChunk]:
    rel = path.resolve().relative_to(project_root.resolve())
    rel_str = str(rel).replace("\\", "/")
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        # whole file as one chunk
        lines = text.splitlines(keepends=True)
        s, e, body = _lines_with_overlap(lines, 1, len(lines) or 1, max_lines, overlap)
        h = _hash_content(body)
        return [
            CodeChunk(
                chunk_id=f"{rel_str}:__syntax_error__:{h}",
                path=rel_str,
                symbol="__syntax_error__",
                start_line=s,
                end_line=e,
                content=body,
                content_hash=h,
            )
        ]

    visitor = ChunkVisitor(text, rel_str, max_lines, overlap)
    visitor.visit(tree)

    # Module-level remainder: lines not covered by any chunk (approximation = full file tail)
    # MVP: add single "module" chunk for imports / loose code at top if empty
    if not visitor.chunks:
        lines = text.splitlines(keepends=True)
        s, e, body = _lines_with_overlap(lines, 1, len(lines) or 1, max_lines, overlap)
        h = _hash_content(body)
        visitor.chunks.append(
            CodeChunk(
                chunk_id=f"{rel_str}:<module>:{h}",
                path=rel_str,
                symbol="<module>",
                start_line=s,
                end_line=e,
                content=body,
                content_hash=h,
            )
        )
    return visitor.chunks
