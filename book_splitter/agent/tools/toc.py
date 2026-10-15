"""Table-of-contents tools: locating known titles and heuristic candidates."""

from __future__ import annotations

from langchain_core.tools import tool

from ...analysis.candidates import extract_candidates
from ...analysis.locate import locate_toc_titles
from ...core.constants import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_MATCHES_PER_TITLE,
    DEFAULT_TOC_MATCH_SCORE,
)
from ...documents.io import load_lines
from ...documents.text import jdump
from .registry import register_tool


@register_tool
@tool
def locate_toc_titles_in_file(
    file_path: str,
    chapter_titles: list[str],
    min_score: float = DEFAULT_TOC_MATCH_SCORE,
    max_matches_per_title: int = DEFAULT_MAX_MATCHES_PER_TITLE,
) -> str:
    """Find the line number where each known chapter title appears in the body.

    Use this after extracting titles from the printed TOC. Matching is
    diacritic- and punctuation-insensitive and fuzzy, so it survives OCR-style
    typos (`GIAVA 5` -> `GLAVA 5`). The printed TOC block is skipped
    automatically, and matches are scored so you can spot weak ones. When a
    title sits on the line *below* a chapter heading (`## GLAVA 2` then
    `## SEKS I SRCE`), the match is snapped up to the heading line so the
    chapter file starts at the chapter marker.

    Returns `{"matches": [{"title", "found", "line_number", "matched_text",
    "score", "snapped_from", "alternatives"}]}`. A dropped match
    (`found: false`) means the title could not be located and needs a fallback
    strategy.
    """
    lines = load_lines(file_path)
    return jdump(
        locate_toc_titles(
            lines,
            chapter_titles,
            min_score=min_score,
            max_matches_per_title=max_matches_per_title,
        )
    )


@register_tool
@tool
def extract_heuristic_candidates(file_path: str, max_candidates: int = DEFAULT_MAX_CANDIDATES) -> str:
    """Fallback: list lines that *look like* a chapter title without any `#` markup.

    Typical of PDF->Markdown conversions that lost the header markup. Keeps
    short lines that either match chapter keywords (`Chapter`, `Part`, `GLAVA`,
    `DEO`, `POGLAVLJE`, ...), are ALL CAPS, or sit alone between blank lines.
    Each entry carries its line number and the structural clues that fired.

    Returns `{"line_count", "count", "truncated", "candidates": [...]}`.
    Expect noise (page numbers, running heads, short sentences); filter it
    before building the split map.
    """
    return jdump(extract_candidates(load_lines(file_path), max_candidates=max_candidates))
