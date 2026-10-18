"""Deterministic structural analysis: no LLM, no writing, no body text out.

Modules
-------
``toc``
    Where the printed table of contents is.
``headers``
    Whether the Markdown headers can be trusted as a chapter map.
``candidates``
    Title-looking lines for books that lost all markup.
``locate``
    Fuzzy title -> line-number resolution, the reliable TOC route.
``backmatter``
    Whether the book ends with publisher/author furniture, and where it starts.
``prose``
    Chapter openings whose heading the conversion dropped, and the shape and
    justification a ``role: "prose-chapter"`` claim must have.
``search``
    Regex line search, for books whose chapter markers share one shape.
``websearch``
    The outside world: a bounded web search and page reader, for the published
    chapter list of a book that lost its own table of contents.
``patterns``
    The vocabularies and regular expressions the above share.
"""

from .backmatter import detect_back_matter, is_back_matter_role
from .candidates import extract_candidates, nearest_title_markers
from .headers import assess_headers
from .locate import locate_toc_titles, snap_to_chapter_heading
from .prose import (
    PROSE_CHAPTER_ROLE,
    is_paragraph_start,
    is_prose_chapter_role,
    prose_start_issue,
    prose_start_problem,
)
from .toc import detect_toc_block, line_in_toc, toc_line_range

__all__ = [
    "PROSE_CHAPTER_ROLE",
    "assess_headers",
    "detect_back_matter",
    "detect_toc_block",
    "extract_candidates",
    "is_back_matter_role",
    "is_paragraph_start",
    "is_prose_chapter_role",
    "line_in_toc",
    "locate_toc_titles",
    "nearest_title_markers",
    "prose_start_issue",
    "prose_start_problem",
    "snap_to_chapter_heading",
    "toc_line_range",
]
