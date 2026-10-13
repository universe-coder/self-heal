"""AST-aware chunking."""

from __future__ import annotations

from pathlib import Path

from self_heal.indexing.chunker import MODULE_SYMBOL, SYNTAX_ERROR_SYMBOL, chunk_file, chunk_source

_SRC = '''"""Module docstring."""
import os

CONSTANT = 1


@decorator
def top(x):
    return x + CONSTANT


class Thing:
    """A class."""

    attr = 2

    def method(self):
        def inner():
            return 1
        return inner()

    async def amethod(self):
        return 3


if __name__ == "__main__":
    top(1)
'''


def _by_symbol(chunks):  # type: ignore[no-untyped-def]
    return {c.symbol: c for c in chunks}


def test_symbols_are_qualified_and_module_code_is_kept() -> None:
    chunks = chunk_source(_SRC, "pkg/mod.py", max_lines=80, overlap=10)
    by = _by_symbol(chunks)
    assert set(by) >= {"top", "Thing", "Thing.method", "Thing.method.inner", "Thing.amethod"}
    assert by["Thing.method"].start_line < by["Thing.amethod"].start_line

    # decorators belong to the function chunk
    assert by["top"].content.startswith("@decorator")
    # the class chunk is only the header (docstring + attributes), methods are separate
    assert "attr = 2" in by["Thing"].content
    assert "def method" not in by["Thing"].content

    module_chunks = [c for c in chunks if c.symbol == MODULE_SYMBOL]
    joined = "".join(c.content for c in module_chunks)
    assert "import os" in joined and "CONSTANT = 1" in joined
    assert '__name__ == "__main__"' in joined
    assert "def top" not in joined

    assert all(c.path == "pkg/mod.py" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks == sorted(chunks, key=lambda c: (c.start_line, c.end_line, c.symbol))


def test_long_function_is_split_into_overlapping_windows() -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(50))
    src = f"def big():\n{body}\n    return x0\n"
    chunks = [c for c in chunk_source(src, "m.py", max_lines=20, overlap=5) if c.symbol == "big"]
    assert len(chunks) >= 3
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 52
    # consecutive windows overlap by `overlap` lines and nothing is skipped
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start_line == prev.end_line - 5 + 1
    assert "return x0" in chunks[-1].content
    assert "x49 = 49" in "".join(c.content for c in chunks)


def test_syntax_error_file_becomes_single_chunk_series() -> None:
    chunks = chunk_source("def broken(:\n    pass\n", "bad.py", max_lines=80, overlap=10)
    assert [c.symbol for c in chunks] == [SYNTAX_ERROR_SYMBOL]
    assert "def broken(:" in chunks[0].content


def test_blank_file_yields_no_chunks() -> None:
    assert chunk_source("\n\n", "empty.py", max_lines=80, overlap=10) == []


def test_chunk_file_uses_relative_posix_path(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    f = tmp_path / "src" / "pkg" / "m.py"
    f.write_text("def f():\n    return 1\n", encoding="utf-8")
    chunks = chunk_file(f, tmp_path, max_lines=80, overlap=10)
    assert [c.path for c in chunks] == ["src/pkg/m.py"]
    assert chunks[0].symbol == "f"
