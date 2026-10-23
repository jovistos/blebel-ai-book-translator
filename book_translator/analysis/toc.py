"""Detection of the printed table-of-contents block.

Strategy 3 of the protocol depends on knowing which lines belong to the printed
TOC: those lines list chapter titles but are *not* chapter starts, so a split
point inside them must be rejected.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.constants import (
    MAX_TITLE_LINE_CHARS,
    TOC_MAX_BLOCK_LINES,
    TOC_SCAN_LINES,
)
from ..documents.markdown import header_of
from ..documents.text import excerpt, norm, similarity
from .patterns import DOT_LEADER_RE, DOT_LEADER_SCAN_RE, TOC_WORDS


def detect_toc_block(
    lines: list[str],
    scan_lines: int = TOC_SCAN_LINES,
    max_block_lines: int = TOC_MAX_BLOCK_LINES,
) -> dict[str, Any]:
    """Locate the printed table-of-contents block near the front of the book.

    Returns ``{"found", "start_line", "end_line", "heading_line", "method",
    "entries"}`` where line numbers are 1-indexed, inclusive, and the block
    *excludes* the heading itself. ``entries`` holds up to 60 dot-leader hits.
    """
    scan = lines[:scan_lines]
    heading_idx: int | None = None
    for i, line in enumerate(scan):
        clean = norm(line.lstrip("#").strip())
        if not clean:
            continue
        head = header_of(line)
        if head and norm(head[1]) in TOC_WORDS:
            heading_idx = i
            break
        if clean in TOC_WORDS:  # plain-text TOC heading (PDF conversion)
            heading_idx = i
            break

    if heading_idx is None:
        method = "dot_leader"
        if len(DOT_LEADER_SCAN_RE.findall("\n".join(scan))) < 4:
            return {"found": False, "method": None, "entries": []}
        start = 0
    else:
        method = "keyword"
        start = heading_idx + 1

    heading_level = 6
    if heading_idx is not None:
        head = header_of(lines[heading_idx])
        if head:
            heading_level = head[0]

    end = start - 1
    blanks = 0
    for i in range(start, min(len(lines), start + max_block_lines)):
        raw = lines[i]
        clean = raw.strip()
        if not clean:
            blanks += 1
            if blanks >= 2 and end >= start:
                break
            continue
        blanks = 0
        head = header_of(raw)
        if head and head[0] <= heading_level and heading_idx is not None and i > heading_idx:
            break
        if len(clean) > 300:
            break
        end = i

    # Second pass: in a book with no '#' markup, the TOC ends where a title
    # re-appears in the body. Detect that repeat and cut the block before it.
    if end >= start:
        titles = [
            re.sub(r"\s+", " ", lines[i].strip())
            for i in range(start, end + 1)
            if 3 < len(lines[i].strip()) <= MAX_TITLE_LINE_CHARS
        ]
        for i in range(start + 2, end + 1):
            clean = re.sub(r"\s+", " ", lines[i].strip())
            if len(clean) > MAX_TITLE_LINE_CHARS:
                continue
            if any(similarity(clean, t) >= 0.85 for t in titles if t != clean):
                end = i - 1
                break

    entries: list[dict[str, Any]] = []
    for i in range(start, end + 1):
        raw = lines[i]
        m = DOT_LEADER_RE.match(raw.strip())
        if m and len(m.group(1).strip()) > 3:
            entries.append(
                {"line_number": i + 1, "title": excerpt(m.group(1), 80), "page": m.group(2)}
            )
    return {
        "found": end >= start,
        "start_line": start + 1 if end >= start else None,
        "end_line": end + 1 if end >= start else None,
        "heading_line": heading_idx + 1 if heading_idx is not None else None,
        "method": method,
        "entries": entries[:60],
    }


def toc_line_range(block: dict[str, Any] | None) -> range:
    """Line indices (0-based) covered by a detected TOC block."""
    if not block or not block.get("found"):
        return range(0)
    start, end = block.get("start_line"), block.get("end_line")
    if not start or not end:
        return range(0)
    return range(start - 1, end)


def line_in_toc(block: dict[str, Any] | None, line_number: int) -> bool:
    """True when a 1-indexed line falls inside the printed TOC block."""
    if not block or not block.get("found"):
        return False
    start, end = block.get("start_line"), block.get("end_line")
    return bool(start and end and start <= line_number <= end)
