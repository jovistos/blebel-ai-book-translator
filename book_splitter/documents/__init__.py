"""Reading source documents and normalising the text around them.

Everything in this package is pure bookkeeping over text: no LLM, no writing
(except where a module says so explicitly, as the slicer does).

``bodyline`` is the stage-2 companion of ``markdown``: ``markdown`` answers
"what shape is this line?", ``bodyline`` answers "was this line damaged by the
PDF conversion, and how?". ``chapters`` reads a *split* book back — the folder
stage 1 produced — without ever touching the original prose.
"""

from .bodyline import (
    count_space_before_punct,
    describes_split_number,
    ends_with_broken_word,
    has_midword_number,
    is_page_artefact,
    is_word_continuation,
    starts_with_lowercase,
)
from .chapters import (
    Chapter,
    chapter_by_id,
    chapters_dir,
    discover_chapters,
    load_split_manifest,
    manifest_summary,
    preprocessed_dir,
)
from .io import clear_line_cache, load_lines, read_line_range, source_size_bytes
from .markdown import (
    header_body,
    header_of,
    header_prefix,
    is_all_caps,
    is_page_number_line,
    is_plain_heading_candidate,
    is_standalone_line,
    looks_like_title,
    md_level,
    page_number_of,
    split_trailing_number,
)
from .text import (
    excerpt,
    jdump,
    norm,
    normalize_filename,
    similarity,
    slugify,
)

__all__ = [
    "Chapter",
    "chapter_by_id",
    "chapters_dir",
    "clear_line_cache",
    "count_space_before_punct",
    "describes_split_number",
    "discover_chapters",
    "ends_with_broken_word",
    "excerpt",
    "has_midword_number",
    "header_body",
    "header_of",
    "header_prefix",
    "is_all_caps",
    "is_page_artefact",
    "is_page_number_line",
    "is_plain_heading_candidate",
    "is_standalone_line",
    "is_word_continuation",
    "jdump",
    "load_lines",
    "load_split_manifest",
    "looks_like_title",
    "manifest_summary",
    "md_level",
    "norm",
    "normalize_filename",
    "page_number_of",
    "preprocessed_dir",
    "read_line_range",
    "similarity",
    "slugify",
    "source_size_bytes",
    "split_trailing_number",
    "starts_with_lowercase",
]
