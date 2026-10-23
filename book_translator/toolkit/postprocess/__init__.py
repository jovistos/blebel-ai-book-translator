"""Stage 9's tools: postprocessing translated headings and the TOC.

The implementation is in :mod:`.tools`; this package adds the stage's toolset and
dispatch table, computed after the module has registered its tools.
"""

from __future__ import annotations

from book_translator.stages.agent.tools.registry import POSTPROCESS, bind_all, tools_for_stage

from .tools import (
    BOOK_TOOL_NAMES,
    CHAPTER_TOOL_NAMES,
    apply_book_toc,
    apply_chapter_postprocess,
    book_tools,
    chapter_tools,
    inspect_chapter_headings,
    inspect_chapter_opening,
    inspect_heading_context,
    inspect_matter_blocks,
    list_chapter_titles,
    report_book_plan,
    report_chapter_postprocess,
    resolve_chapter_path,
    validate_book_plan_tool,
    validate_chapter_postprocess,
    verify_postprocessed_chapters,
)

#: Tools exposed to the stage-9 (translated-heading postprocessing) agent.
POSTPROCESS_TOOLS = tools_for_stage(POSTPROCESS)

#: Name -> tool, for the executor's dispatch table.
POSTPROCESS_TOOL_BY_NAME = bind_all(POSTPROCESS_TOOLS)

__all__ = [
    "BOOK_TOOL_NAMES",
    "CHAPTER_TOOL_NAMES",
    "POSTPROCESS_TOOLS",
    "POSTPROCESS_TOOL_BY_NAME",
    "apply_book_toc",
    "apply_chapter_postprocess",
    "book_tools",
    "chapter_tools",
    "inspect_chapter_headings",
    "inspect_chapter_opening",
    "inspect_heading_context",
    "inspect_matter_blocks",
    "list_chapter_titles",
    "report_book_plan",
    "report_chapter_postprocess",
    "resolve_chapter_path",
    "validate_book_plan_tool",
    "validate_chapter_postprocess",
    "verify_postprocessed_chapters",
]
