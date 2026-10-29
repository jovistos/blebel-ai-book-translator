"""Stage 2's tools: chapter preprocessing.

The implementation is in :mod:`.tools`; this package adds the stage's toolset and
dispatch table, computed after the module has registered its tools.
"""

from __future__ import annotations

import book_translator.stages.agent.tools  # noqa: F401 — register shared tools
from book_translator.toolkit.registry import PREPROCESS, bind_all, tools_for_stage

from .tools import (
    apply_chapter_cleaning,
    collect_book_headers,
    collect_chapter_headers,
    extract_headers_from_file,
    inspect_chapter_damage,
    manifest_path_for,
    report_chapter_cleaning,
    scan_book_damage,
    trace_path_for,
    validate_chapter_cleaning,
    verify_preprocessed_chapters,
)

#: Tools exposed to the stage-2 (preprocessing) agent.
PREPROCESS_TOOLS = tools_for_stage(PREPROCESS)

#: Name -> tool, for the executor's dispatch table.
PREPROCESS_TOOL_BY_NAME = bind_all(PREPROCESS_TOOLS)

__all__ = [
    "PREPROCESS_TOOLS",
    "PREPROCESS_TOOL_BY_NAME",
    "apply_chapter_cleaning",
    "collect_book_headers",
    "collect_chapter_headers",
    "extract_headers_from_file",
    "inspect_chapter_damage",
    "manifest_path_for",
    "report_chapter_cleaning",
    "scan_book_damage",
    "trace_path_for",
    "validate_chapter_cleaning",
    "verify_preprocessed_chapters",
]
