"""The agent's toolset, grouped by what each tool touches.

Modules
-------
``registry``
    Name -> tool lookup and the ``WRITE_TOOLS`` set.
``extract``
    Structural extraction (headers, front matter, native TOC).
``toc``
    Printed-TOC title location and heuristic candidates.
``proposal``
    The two tools the agent *answers* through.
``execution``
    Slicing and verification (filesystem writers).
``policy``
    The "no writes before validation" rule enforced by the graph.

Importing this package registers every tool, which is why the modules below are
imported for their side effect as well as for their names.
"""

from . import execution, extract, proposal, toc  # noqa: F401  (registration side effect)
from .execution import slice_book_by_line_numbers, verify_sliced_output
from .extract import (
    detect_native_toc,
    extract_front_matter,
    extract_markdown_headers,
    extract_native_toc,
    scan_markdown_headers,
)
from .policy import BLOCKED_MESSAGE, blocked_payload, is_blocked
from .proposal import propose_split_map, validate_split_map_tool
from .registry import (
    WRITE_TOOLS,
    all_tools,
    describe_tools,
    is_write_tool,
    register_tool,
    tool_by_name,
    tool_names,
)
from .toc import extract_heuristic_candidates, locate_toc_titles_in_file

#: Toolset exposed to the agent, in definition order.
TOOLS = all_tools()

#: Name -> tool, for the executor's dispatch table.
TOOL_BY_NAME = {tool.name: tool for tool in TOOLS}

__all__ = [
    "BLOCKED_MESSAGE",
    "TOOLS",
    "TOOL_BY_NAME",
    "WRITE_TOOLS",
    "all_tools",
    "blocked_payload",
    "describe_tools",
    "detect_native_toc",
    "extract_front_matter",
    "extract_heuristic_candidates",
    "extract_markdown_headers",
    "extract_native_toc",
    "is_blocked",
    "is_write_tool",
    "locate_toc_titles_in_file",
    "propose_split_map",
    "register_tool",
    "scan_markdown_headers",
    "slice_book_by_line_numbers",
    "tool_by_name",
    "tool_names",
    "validate_split_map_tool",
    "verify_sliced_output",
]
