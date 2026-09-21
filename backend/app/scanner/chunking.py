"""Deterministic line-based chunking that preserves source text exactly."""

import re
from dataclasses import dataclass

CHUNK_LINES = 100
_LINE_PATTERN = re.compile(r"[^\n]*\n|[^\n]+")


@dataclass(frozen=True)
class CodeChunk:
    chunk_index: int
    start_line: int
    end_line: int
    content: str


def chunk_source(text: str, max_lines: int = CHUNK_LINES) -> list[CodeChunk]:
    """Split text into consecutive chunks of at most max_lines (1-based, inclusive).

    Lines split only on newline characters, so joining every chunk's content
    reproduces the input exactly. Empty text yields no chunks.
    """

    lines = _LINE_PATTERN.findall(text)
    return [
        CodeChunk(
            chunk_index=index,
            start_line=start + 1,
            end_line=min(start + max_lines, len(lines)),
            content="".join(lines[start : start + max_lines]),
        )
        for index, start in enumerate(range(0, len(lines), max_lines))
    ]
