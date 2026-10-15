"""Low-level markdown line inspection.

These helpers answer "what *shape* is this line?" — header, standalone title,
prose — without ever interpreting the book's content.
"""

from __future__ import annotations

import re

from ..core.constants import MAX_TITLE_LINE_CHARS

MD_HEADER_RE = re.compile(r"^(#{1,6})\s*(.*)$")


def header_of(line: str) -> tuple[int, str] | None:
    """Return ``(level, title)`` when the line is a Markdown ATX header."""
    m = MD_HEADER_RE.match(line.strip())
    if not m:
        return None
    return len(m.group(1)), m.group(2).strip()


def md_level(line: str) -> int | None:
    """Header level of a line, or ``None`` when it is not a header."""
    head = header_of(line)
    return head[0] if head else None


def is_standalone_line(lines: list[str], idx0: int) -> bool:
    """True when the line is surrounded by blank lines (typical of a title)."""
    prev_blank = idx0 == 0 or not lines[idx0 - 1].strip()
    next_blank = idx0 == len(lines) - 1 or not lines[idx0 + 1].strip()
    return prev_blank and next_blank


def looks_like_title(line: str) -> bool:
    """A non-empty line short enough to be a heading rather than prose."""
    clean = line.strip().lstrip("#").strip()
    return bool(clean) and len(clean) <= MAX_TITLE_LINE_CHARS
