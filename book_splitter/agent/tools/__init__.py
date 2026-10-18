"""The agent toolset, grouped by pipeline stage.

Each stage binds only its own tools plus the generic ones, and the grouping is
part of the registration, not of a hand-maintained list:

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
    ``agent/tools/extract.py`` (native TOC, front matter, TOC block),
    ``toc.py`` (TOC title location, heuristics), ``search.py`` (a regex the
    agent writes, for books whose markers one pattern describes),
    ``tail.py`` (the end of the book: is there back matter, and where does it
    start — read through an escalating window),
    ``websearch.py`` (``web_search`` / ``fetch_web_page``: the published chapter
    list of a book that lost its own TOC, the only tools here that use the
    network),
    ``proposal.py`` (the tools the agent answers through), ``execution.py``
    (slicing and verification).

``preprocess`` (stage 2)
    ``agent/tools/preprocess.py``: header collection per chapter and per book,
    damage inspection, the cleaning-plan submission tool, and the apply/verify
    pair.

Modules
-------
``registry``
    Name -> tool lookup, stage tags, and the ``WRITE_TOOLS`` sets.
``shared``
    The generic tools described above.
``extract`` / ``toc`` / ``search`` / ``proposal`` / ``execution``
    Stage 1.
``preprocess``
    Stage 2.
``policy``
    The "no writes before validation" rule enforced by the graph.

Importing this package registers every tool, which is why the modules below are
imported for their side effect as well as for their names.
"""

from . import (  # noqa: F401
    execution,
    extract,
    preprocess,
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
from .policy import BLOCKED_MESSAGE, blocked_payload, is_blocked
from .preprocess import (
    apply_chapter_cleaning,
    collect_book_headers,
    collect_chapter_headers,
    extract_headers_from_file,
    inspect_chapter_damage,
    report_chapter_cleaning,
    scan_book_damage,
    validate_chapter_cleaning,
    verify_preprocessed_chapters,
)
from .proposal import propose_split_map, validate_split_map_tool
from .registry import (
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
    preprocess_stage,
    register_tool,
    split_stage,
    stage_of,
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

#: Toolset exposed to the stage-2 (preprocessing) agent.
PREPROCESS_TOOLS = tools_for_stage(PREPROCESS)

#: Every registered tool, in definition order (kept for backwards compatibility).
TOOLS = all_tools()

#: Name -> tool, for the executor's dispatch table.
TOOL_BY_NAME = {tool.name: tool for tool in TOOLS}

#: Per-stage dispatch tables.
SPLIT_TOOL_BY_NAME = bind_all(SPLIT_TOOLS)
PREPROCESS_TOOL_BY_NAME = bind_all(PREPROCESS_TOOLS)

__all__ = [
    "BLOCKED_MESSAGE",
    "PREPROCESS",
    "PREPROCESS_TOOLS",
    "PREPROCESS_TOOL_BY_NAME",
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
    "apply_chapter_cleaning",
    "bind_all",
    "blocked_payload",
    "collect_book_headers",
    "collect_chapter_headers",
    "describe_tools",
    "detect_native_toc",
    "extract_front_matter",
    "extract_headers_from_file",
    "extract_heuristic_candidates",
    "extract_markdown_headers",
    "find_lines_matching",
    "extract_native_toc",
    "fetch_web_page",
    "find_lines_matching",
    "grouped_tools",
    "inspect_book_tail",
    "inspect_chapter_damage",
    "inspect_line_window",
    "is_blocked",
    "is_write_tool",
    "list_chapter_files",
    "locate_toc_titles_in_file",
    "preprocess_stage",
    "propose_split_map",
    "register_tool",
    "report_chapter_cleaning",
    "sample_corpus_text",
    "scan_book_damage",
    "scan_corpus_damage_tool",
    "scan_markdown_headers",
    "shared_tool",
    "slice_book_by_line_numbers",
    "split_stage",
    "stage_of",
    "tool_by_name",
    "tool_names",
    "tools_for_stage",
    "validate_chapter_cleaning",
    "validate_split_map_tool",
    "verify_preprocessed_chapters",
    "verify_sliced_output",
    "web_search",
]
