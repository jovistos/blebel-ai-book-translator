"""Reading source documents and normalising the text around them.

Everything in this package is pure bookkeeping over text: no LLM, no writing
(except where a module says so explicitly, as the slicer does).
"""

from .io import clear_line_cache, load_lines, read_line_range, source_size_bytes
from .markdown import header_of, is_standalone_line, looks_like_title, md_level
from .text import (
    excerpt,
    jdump,
    norm,
    normalize_filename,
    slugify,
    similarity,
)

__all__ = [
    "clear_line_cache",
    "excerpt",
    "header_of",
    "is_standalone_line",
    "jdump",
    "load_lines",
    "looks_like_title",
    "md_level",
    "norm",
    "normalize_filename",
    "read_line_range",
    "similarity",
    "slugify",
    "source_size_bytes",
]
