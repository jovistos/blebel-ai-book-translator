"""The front/back matter localization tools.

The implementation is in :mod:`.tools`; this package adds the stage's toolset and
dispatch table, computed after the module has registered its tools.
"""

from __future__ import annotations

import book_translator.stages.agent.tools  # noqa: F401 — register shared tools
from book_translator.toolkit.registry import MATTER, bind_all, tools_for_stage

from .tools import (
    BOOK_TOOL_NAMES,
    CHUNK_TOOL_NAMES,
    list_book_chapters,
    list_matter_sections,
    read_matter_chunk,
    read_matter_section,
    report_matter_localization,
    report_matter_plan,
    validate_matter_localization,
    validate_matter_plan_tool,
)

#: Tools exposed to the front/back matter localization agent.
MATTER_TOOLS = tools_for_stage(MATTER)

#: Name -> tool, for the executor's dispatch table.
MATTER_TOOL_BY_NAME = bind_all(MATTER_TOOLS)

__all__ = [
    "BOOK_TOOL_NAMES",
    "CHUNK_TOOL_NAMES",
    "MATTER_TOOLS",
    "MATTER_TOOL_BY_NAME",
    "list_book_chapters",
    "list_matter_sections",
    "read_matter_chunk",
    "read_matter_section",
    "report_matter_localization",
    "report_matter_plan",
    "validate_matter_localization",
    "validate_matter_plan_tool",
]
