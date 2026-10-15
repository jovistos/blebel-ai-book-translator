"""Source-file reading with a process-local cache.

The cache exists so the many tool calls of a single agent run do not re-read a
411 KB book from disk. It is keyed by ``(resolved path, mtime, size)`` so an
edited file is picked up automatically, and it never leaves the process: no line
content is ever placed in the graph state or in a prompt.
"""

from __future__ import annotations

from pathlib import Path

from ..core.errors import ToolError

_LINE_CACHE: dict[tuple[str, int, int], list[str]] = {}
#: How many distinct files stay cached (one book at a time is the normal case).
_CACHE_SLOTS = 2


def load_lines(path: str | Path) -> list[str]:
    """Read the source file once per (path, mtime, size)."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ToolError(f"file not found: {path}")
    stat = p.stat()
    key = (str(p.resolve()), int(stat.st_mtime), stat.st_size)
    cached = _LINE_CACHE.get(key)
    if cached is None:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            cached = fh.readlines()
        while len(_LINE_CACHE) >= _CACHE_SLOTS:
            _LINE_CACHE.pop(next(iter(_LINE_CACHE)))
        _LINE_CACHE[key] = cached
    return cached


def read_line_range(path: str | Path, start: int, end: int) -> list[str]:
    """Inclusive 1-indexed slice of the source lines (no copies of the file)."""
    lines = load_lines(path)
    return lines[max(start - 1, 0) : max(end, 0)]


def source_size_bytes(path: str | Path) -> int:
    """Size of the source document on disk, in bytes."""
    return Path(path).expanduser().stat().st_size


def clear_line_cache() -> None:
    """Drop the cache (used by tests to prove reads are fresh)."""
    _LINE_CACHE.clear()
