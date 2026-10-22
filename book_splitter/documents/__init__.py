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
    manifest_summary,
    preprocessed_dir,
)
from .chunks import (
    ChunkInput,
    chunk_chapter_names,
    chunk_files,
    chunk_glossary_filename,
    discover_chunk_inputs,
    resolve_chunks_dir,
    resolve_glossary_path,
)
from .io import clear_line_cache, load_lines, read_line_range, source_size_bytes
from .layout import (
    STAGE_DIRNAMES,
    STAGE_DIRS,
    book_root,
    resolve_stage_dir,
    stage_dir_name,
    stage_of,
)
from .manifest import (
    load_assembly_manifest,
    load_chunk_glossary_manifest,
    load_chunk_manifest,
    load_convert_manifest,
    load_glossary_manifest,
    load_matter_manifest,
    load_postprocess_manifest,
    load_split_manifest,
    load_translation_manifest,
)
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
    "STAGE_DIRS",
    "STAGE_DIRNAMES",
    "Chapter",
    "ChunkInput",
    "book_root",
    "chapter_by_id",
    "chapters_dir",
    "chunk_chapter_names",
    "chunk_files",
    "chunk_glossary_filename",
    "clear_line_cache",
    "count_space_before_punct",
    "describes_split_number",
    "discover_chapters",
    "discover_chunk_inputs",
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
    "load_assembly_manifest",
    "load_chunk_glossary_manifest",
    "load_chunk_manifest",
    "load_convert_manifest",
    "load_glossary_manifest",
    "load_lines",
    "load_matter_manifest",
    "load_postprocess_manifest",
    "load_translation_manifest",
    "load_split_manifest",
    "looks_like_title",
    "manifest_summary",
    "md_level",
    "norm",
    "normalize_filename",
    "page_number_of",
    "preprocessed_dir",
    "read_line_range",
    "resolve_chunks_dir",
    "resolve_glossary_path",
    "resolve_stage_dir",
    "similarity",
    "slugify",
    "source_size_bytes",
    "stage_dir_name",
    "stage_of",
    "split_trailing_number",
    "starts_with_lowercase",
]
