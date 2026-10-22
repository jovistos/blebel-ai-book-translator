"""Stage 1's toolset and the tools shared by more than one stage.

The splitter owns this package, so the splitting tools and the generic evidence
tools live here. The other stages' tools live next to those stages in
:mod:`book_splitter.toolkit`:

``preprocess``
    :mod:`book_splitter.toolkit.preprocess`
``postprocess``
    :mod:`book_splitter.toolkit.postprocess`
``matter``
    :mod:`book_splitter.toolkit.matter`

``shared``
    Generic tools useful to more than one stage, because two stages need the
    same evidence and duplicating it would be a bug farm:

    * ``scan_markdown_headers`` / ``extract_markdown_headers`` — what the raw
      file's headers look like (stage 1 chooses cuts, stage 2 inspects them),
    * ``inspect_line_window`` — a few lines around one line number, capped,
    * ``list_chapter_files`` — stage 1's manifest, read back as chapter files,
    * ``sample_corpus_text`` / ``scan_corpus_damage_tool`` — bounded samples and
      the book-wide damage profile.

``split`` (stage 1)
    ``extract.py`` (native TOC, front matter, TOC block), ``toc.py`` (TOC title
    location, heuristics), ``search.py`` (a regex the agent writes, for books
    whose markers one pattern describes), ``tail.py`` (the end of the book: is
    there back matter, and where does it start — read through an escalating
    window), ``websearch.py`` (``web_search`` / ``fetch_web_page``: the published
    chapter list of a book that lost its own TOC, and — for the matter stage — an
    existing published translation whose title should be reused; these are the
    only tools here that use the network, shared by the split and matter
    stages), ``proposal.py`` (the tools the agent answers through),
    ``execution.py`` (slicing and verification), ``openings.py`` (the
    chapter-opening audit shared by stages 2 and 9).

Modules
-------
``registry``
    Name -> tool lookup, stage tags, and the ``WRITE_TOOLS`` sets.
``shared``
    The generic tools described above.
``policy``
    The "no writes before validation" rule the shared executor enforces.
``extract`` / ``toc`` / ``search`` / ``proposal`` / ``execution`` / ``tail``
    Stage 1.

Importing this package registers stage 1's and the shared tools, which is why
the modules below are imported for their side effect as well as for their names.
"""

from . import (  # noqa: F401
    execution,
    extract,
    openings,
    proposal,
    search,
    shared,
    tail,
    toc,
    websearch,
)
from .execution import slice_book_by_line_numbers, verify_sliced_output
from .extract import (
    detect_native_toc,
    extract_front_matter,
    extract_markdown_headers,
    extract_native_toc,
    scan_markdown_headers,
)
from .openings import audit_book_openings, audit_chapter_opening
from .policy import BLOCKED_MESSAGE, blocked_payload, is_blocked
from .proposal import propose_split_map, validate_split_map_tool
from .registry import (
    MATTER,
    MATTER_WRITE_TOOLS,
    POSTPROCESS,
    POSTPROCESS_WRITE_TOOLS,
    PREPROCESS,
    PREPROCESS_WRITE_TOOLS,
    SHARED,
    SPLIT,
    SPLIT_WRITE_TOOLS,
    STAGES,
    WRITE_TOOLS,
    all_tools,
    bind_all,
    describe_tools,
    grouped_tools,
    is_write_tool,
    matter_stage,
    postprocess_stage,
    preprocess_stage,
    register_tool,
    split_stage,
    stage_of,
    stages_of,
    tool_by_name,
    tool_names,
    tools_for_stage,
)
from .registry import (
    shared as shared_tool,
)
from .search import find_lines_matching
from .shared import (
    inspect_line_window,
    list_chapter_files,
    sample_corpus_text,
    scan_corpus_damage_tool,
)
from .tail import inspect_book_tail
from .toc import extract_heuristic_candidates, locate_toc_titles_in_file
from .websearch import fetch_web_page, web_search

#: Toolset exposed to the stage-1 (splitting) agent.
SPLIT_TOOLS = tools_for_stage(SPLIT)

#: Name -> tool, for the executor's dispatch table.
SPLIT_TOOL_BY_NAME = bind_all(SPLIT_TOOLS)

#: Every tool registered by this package, in definition order.
TOOLS = all_tools()

#: Name -> tool for the tools this package owns.
TOOL_BY_NAME = {tool.name: tool for tool in TOOLS}

__all__ = [
    "BLOCKED_MESSAGE",
    "MATTER",
    "MATTER_WRITE_TOOLS",
    "POSTPROCESS",
    "POSTPROCESS_WRITE_TOOLS",
    "PREPROCESS",
    "PREPROCESS_WRITE_TOOLS",
    "SHARED",
    "SPLIT",
    "SPLIT_TOOLS",
    "SPLIT_TOOL_BY_NAME",
    "SPLIT_WRITE_TOOLS",
    "STAGES",
    "TOOLS",
    "TOOL_BY_NAME",
    "WRITE_TOOLS",
    "all_tools",
    "audit_book_openings",
    "audit_chapter_opening",
    "bind_all",
    "blocked_payload",
    "describe_tools",
    "detect_native_toc",
    "extract_front_matter",
    "extract_heuristic_candidates",
    "extract_markdown_headers",
    "extract_native_toc",
    "fetch_web_page",
    "find_lines_matching",
    "grouped_tools",
    "inspect_book_tail",
    "inspect_line_window",
    "is_blocked",
    "is_write_tool",
    "list_chapter_files",
    "locate_toc_titles_in_file",
    "matter_stage",
    "postprocess_stage",
    "preprocess_stage",
    "propose_split_map",
    "register_tool",
    "sample_corpus_text",
    "scan_corpus_damage_tool",
    "scan_markdown_headers",
    "shared_tool",
    "slice_book_by_line_numbers",
    "split_stage",
    "stage_of",
    "stages_of",
    "tool_by_name",
    "tool_names",
    "tools_for_stage",
    "validate_split_map_tool",
    "verify_sliced_output",
    "web_search",
]
