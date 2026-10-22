"""Deprecated import path — the tools now live with the stage that owns them.

This module keeps the old ``book_splitter.tools`` namespace working: it exposes
the LangChain tools plus the deterministic functions they wrap, so existing
imports and tests continue to work. New code should import from the package that
owns the function: :mod:`book_splitter.agent.tools` (stage 1 and the shared
tools), :mod:`book_splitter.toolkit.preprocess`,
:mod:`book_splitter.toolkit.postprocess` and :mod:`book_splitter.toolkit.matter`,
or the deterministic layer (:mod:`book_splitter.analysis`,
:mod:`book_splitter.slicing`, :mod:`book_splitter.documents`).
"""

from __future__ import annotations

# --- registry and tools ----------------------------------------------------- #
from .agent.tools import (
    BLOCKED_MESSAGE,
    SPLIT_TOOLS,
    TOOL_BY_NAME,
    TOOLS,
    WRITE_TOOLS,
    all_tools,
    detect_native_toc,
    extract_front_matter,
    extract_heuristic_candidates,
    extract_markdown_headers,
    extract_native_toc,
    inspect_line_window,
    is_blocked,
    list_chapter_files,
    locate_toc_titles_in_file,
    propose_split_map,
    scan_corpus_damage_tool,
    scan_markdown_headers,
    slice_book_by_line_numbers,
    validate_split_map_tool,
    verify_sliced_output,
)

# --- structural analysis ---------------------------------------------------- #
from .analysis.candidates import extract_candidates, nearest_title_markers
from .analysis.headers import assess_headers
from .analysis.locate import locate_toc_titles, snap_to_chapter_heading
from .analysis.toc import detect_toc_block, line_in_toc, toc_line_range

# --- core ------------------------------------------------------------------- #
from .core.constants import (
    EXCERPT_CHARS,
    FRONT_MATTER_FILENAME,
    MAX_FRONT_MATTER_CHARS,
    MAX_TOOL_CHARS,
)
from .core.errors import ToolError

# --- document helpers ------------------------------------------------------- #
from .documents.chapters import discover_chapters, load_split_manifest
from .documents.io import clear_line_cache, load_lines, read_line_range
from .documents.markdown import header_of, is_standalone_line, looks_like_title
from .documents.text import excerpt, jdump, norm, normalize_filename, similarity, slugify

# --- stage 2: preprocessing ------------------------------------------------- #
from .preprocess.corpus import sample_text, scan_corpus_damage
from .preprocess.damage import (
    repair_page_number_in_word,
    repair_space_before_punctuation,
    repair_split_word,
    scan_chapter_damage,
)
from .preprocess.inspect import inspect_chapter
from .preprocess.transform import (
    apply_cleaning_plan,
    validate_cleaning_plan,
    verify_cleaning_plan,
)

#: Historical name: the slicer/verifier hash helper.
from .slicing.hashing import sha256_file

# --- validation, slicing, verification -------------------------------------- #
from .slicing.plan import (
    describe_first_line,
    gap_warnings,
    plan_files,
    reconcile_filename,
    unsafe_filename,
)
from .slicing.slice import preview_split, slice_book, write_manifest_file
from .slicing.validation import validate_source, validate_split_map
from .slicing.verify import source_digest_for_manifest, verify_output, verify_output_dir
from .toolkit.preprocess import (
    PREPROCESS_TOOLS,
    apply_chapter_cleaning,
    collect_chapter_headers,
    report_chapter_cleaning,
    validate_chapter_cleaning,
    verify_preprocessed_chapters,
)

__all__ = [
    "BLOCKED_MESSAGE",
    "EXCERPT_CHARS",
    "FRONT_MATTER_FILENAME",
    "MAX_FRONT_MATTER_CHARS",
    "MAX_TOOL_CHARS",
    "PREPROCESS_TOOLS",
    "SPLIT_TOOLS",
    "TOOLS",
    "TOOL_BY_NAME",
    "ToolError",
    "WRITE_TOOLS",
    "all_tools",
    "verify_preprocessed_chapters",
    "verify_cleaning_plan",
    "validate_cleaning_plan",
    "validate_chapter_cleaning",
    "scan_corpus_damage_tool",
    "scan_corpus_damage",
    "scan_chapter_damage",
    "sample_text",
    "report_chapter_cleaning",
    "repair_split_word",
    "repair_space_before_punctuation",
    "repair_page_number_in_word",
    "load_split_manifest",
    "list_chapter_files",
    "inspect_line_window",
    "inspect_chapter",
    "discover_chapters",
    "collect_chapter_headers",
    "apply_cleaning_plan",
    "apply_chapter_cleaning",
    "assess_headers",
    "clear_line_cache",
    "describe_first_line",
    "detect_native_toc",
    "detect_toc_block",
    "excerpt",
    "extract_candidates",
    "extract_front_matter",
    "extract_heuristic_candidates",
    "extract_markdown_headers",
    "extract_native_toc",
    "gap_warnings",
    "header_of",
    "is_blocked",
    "is_standalone_line",
    "jdump",
    "line_in_toc",
    "load_lines",
    "locate_toc_titles",
    "locate_toc_titles_in_file",
    "looks_like_title",
    "nearest_title_markers",
    "norm",
    "normalize_filename",
    "plan_files",
    "preview_split",
    "propose_split_map",
    "read_line_range",
    "reconcile_filename",
    "scan_markdown_headers",
    "sha256_file",
    "similarity",
    "slice_book",
    "slice_book_by_line_numbers",
    "slugify",
    "snap_to_chapter_heading",
    "source_digest_for_manifest",
    "toc_line_range",
    "unsafe_filename",
    "validate_source",
    "validate_split_map",
    "validate_split_map_tool",
    "verify_output",
    "verify_output_dir",
    "verify_sliced_output",
    "write_manifest_file",
]

#: Sorted for readability; ruff only requires the imports to be ordered.
__all__ = sorted(set(__all__))
