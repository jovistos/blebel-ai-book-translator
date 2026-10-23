"""Heuristic title candidates for books that lost their markup.

This is strategy 4: the last resort when a PDF conversion removed every ``#``
marker and there is no usable printed TOC. It offers short standalone lines,
ALL-CAPS lines and chapter-keyword lines, together with the structural clue
that fired for each, so the model can filter the noise itself.
"""

from __future__ import annotations

from typing import Any

from ..documents.markdown import header_of, is_standalone_line, looks_like_title
from ..documents.text import excerpt
from .patterns import CHAPTER_RE, NUMBERED_RE, NUMERIC_RE

#: Candidate lines longer than this are prose, not titles.
MAX_CANDIDATE_CHARS = 80


def extract_candidates(lines: list[str], max_candidates: int = 400) -> dict[str, Any]:
    """List lines that *look like* a chapter title without any ``#`` markup."""
    candidates: list[dict[str, Any]] = []
    for idx, raw in enumerate(lines):
        clean = raw.strip()
        if not clean or clean.startswith("#"):
            continue
        standalone = is_standalone_line(lines, idx)
        is_short = len(clean) <= MAX_CANDIDATE_CHARS
        matches_pattern = bool(CHAPTER_RE.match(clean))
        numbered = bool(NUMBERED_RE.match(clean))
        is_all_caps = (
            clean.isupper() and 3 < len(clean) <= MAX_CANDIDATE_CHARS and any(c.isalpha() for c in clean)
        )
        if (standalone and is_short) or (matches_pattern and is_short) or (numbered and is_short):
            candidates.append(
                {
                    "line_number": idx + 1,
                    "text": excerpt(clean, 90),
                    "standalone": standalone,
                    "matches_chapter_keyword": matches_pattern,
                    "numbered": numbered,
                    "all_caps": is_all_caps,
                    "looks_like_page_number": bool(NUMERIC_RE.match(clean)),
                }
            )
    return {
        "line_count": len(lines),
        "count": len(candidates),
        "truncated": len(candidates) > max_candidates,
        "candidates": candidates[:max_candidates],
    }


def nearest_title_markers(lines: list[str], ln: int, window: int = 8) -> list[dict[str, Any]]:
    """Nearby lines that *do* look like titles, offered as a correction hint."""
    out: list[dict[str, Any]] = []
    for i in range(max(0, ln - 1 - window), min(len(lines), ln + window)):
        raw = lines[i]
        head = header_of(raw)
        if head and head[1]:
            out.append({"line_number": i + 1, "text": excerpt(head[1], 70)})
        elif looks_like_title(raw) and raw.strip() and is_standalone_line(lines, i):
            out.append({"line_number": i + 1, "text": excerpt(raw, 70)})
    return out[:6]
