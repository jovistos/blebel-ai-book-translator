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
``patterns``
    The vocabularies and regular expressions the above share.
"""

from .candidates import extract_candidates, nearest_title_markers
from .headers import assess_headers
from .locate import locate_toc_titles, snap_to_chapter_heading
from .toc import detect_toc_block, line_in_toc, toc_line_range

__all__ = [
    "assess_headers",
    "detect_toc_block",
    "extract_candidates",
    "line_in_toc",
    "locate_toc_titles",
    "nearest_title_markers",
    "snap_to_chapter_heading",
    "toc_line_range",
]
