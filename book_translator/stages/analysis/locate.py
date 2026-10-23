"""Fuzzy location of known chapter titles inside the book body.

The printed TOC gives the correct *order and set* of chapters, but its page
numbers are useless for slicing. This module turns those title strings into
body line numbers, surviving OCR/PDF-conversion damage such as ``GIAVA 5`` for
``GLAVA 5``, and snapping a subtitle back up to the chapter heading above it.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Any

from book_translator.core.constants import (
    DEFAULT_MAX_MATCHES_PER_TITLE,
    DEFAULT_TOC_MATCH_SCORE,
    MAX_TITLE_LINE_CHARS,
)
from book_translator.documents.markdown import header_of, is_standalone_line
from book_translator.documents.text import excerpt, norm, similarity

from .patterns import CHAPTER_RE
from .toc import detect_toc_block, toc_line_range


def locate_toc_titles(
    lines: list[str],
    chapter_titles: list[str],
    min_score: float = DEFAULT_TOC_MATCH_SCORE,
    max_matches_per_title: int = DEFAULT_MAX_MATCHES_PER_TITLE,
) -> dict[str, Any]:
    """Find the line number where each known chapter title appears in the body.

    Returns ``{"matches": [{"title", "found", "line_number", "matched_text",
    "score", "snapped_from", "alternatives"}]}``. A dropped match
    (``found: false``) means the title could not be located and needs a fallback
    strategy.
    """
    toc = detect_toc_block(lines)
    skip = toc_line_range(toc)

    pool: list[tuple[int, str, bool, bool]] = []  # (line_number, text, is_header, standalone)
    for i, raw in enumerate(lines):
        if i in skip:
            continue
        text = raw.strip()
        if not text or len(text) > MAX_TITLE_LINE_CHARS * 2:
            continue
        pool.append(
            (
                i + 1,
                re.sub(r"\s+", " ", text.lstrip("#").strip()),
                header_of(raw) is not None,
                is_standalone_line(lines, i),
            )
        )

    matches: list[dict[str, Any]] = []
    for title in list(chapter_titles or [])[:80]:
        scored: list[dict[str, Any]] = []
        for ln, text, is_header, standalone in pool:
            if len(text) < 3:
                continue
            score = similarity(title, text)
            if is_header:
                score += 0.05
            if standalone:
                score += 0.03
            if 0 < len(text) - len(title) <= 12 and norm(title) in norm(text):
                score += 0.02
            if score >= min_score:
                scored.append(
                    {
                        "line_number": ln,
                        "matched_text": excerpt(text, 80),
                        "score": round(min(score, 1.0), 3),
                        "is_markdown_header": is_header,
                    }
                )
        scored.sort(key=lambda m: (-m["score"], m["line_number"]))
        best = scored[0] if scored else None
        snapped_from = None
        if best is not None:
            snapped = snap_to_chapter_heading(lines, best["line_number"])
            if snapped is not None:
                snapped_from = best["line_number"]
                best = {
                    **best,
                    "line_number": snapped,
                    "matched_text": excerpt(lines[snapped - 1].strip(), 80),
                    "snapped_to_chapter_heading": True,
                }
        matches.append(
            {
                "title": excerpt(str(title), 100),
                "found": best is not None,
                "line_number": best["line_number"] if best else None,
                "matched_text": best["matched_text"] if best else None,
                "score": best["score"] if best else None,
                "snapped_from": snapped_from,
                "alternatives": scored[1:max_matches_per_title],
            }
        )

    found = [m for m in matches if m["found"]]
    out_of_order = [
        f"{a['title']} -> {b['title']}"
        for a, b in pairwise(found)
        if b["line_number"] < a["line_number"]
    ]
    return {
        "searched_lines": len(pool),
        "skipped_toc_block": {
            "start_line": toc.get("start_line"),
            "end_line": toc.get("end_line"),
        },
        "found_count": len(found),
        "requested_count": len(matches),
        "out_of_order_pairs": out_of_order,
        "matches": matches,
    }


def snap_to_chapter_heading(lines: list[str], ln: int, lookback: int = 3) -> int | None:
    """If a title is the second line of a two-line heading, return the first line.

    Example: line 136 ``## GLAVA 2``, line 138 ``## SEKS I SRCE`` -> 136.
    """
    if _is_chapter_marker(lines, ln - 1):
        return None
    for back in range(1, lookback + 1):
        idx = ln - 1 - back
        if idx < 0:
            break
        if _is_chapter_marker(lines, idx):
            return idx + 1
    return None


def _is_chapter_marker(lines: list[str], idx0: int) -> bool:
    head = header_of(lines[idx0])
    return bool(head and CHAPTER_RE.match("## " + head[1]))
