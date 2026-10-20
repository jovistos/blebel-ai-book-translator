"""Front/back matter localization: a LangGraph agent stage before translation.

Front matter and back matter are not chapters. Their translation is a
*localization* of the translated edition:

* the title may already have a published rendering — the book agent searches the
  web for it and reuses it when it exists;
* the edition must clearly present itself as a translation and credit **BleBel**;
* original-edition material (imprint, ISBN, CIP, printing history, original
  publisher permissions) is omitted, with every omission checked;
* the result is formatted as published Markdown, not a raw translation dump;
* no dictionary or glossary is used.

Two LangGraphs:

``build_matter_graph``
    the book-level plan: inventory, web search, title/credit/omissions.
``build_matter_chunk_graph``
    one matter chunk: localize, validate, apply, verify.

``run_matter`` drives the book graph, then the chunks in reading order, writing
the translated chunks to ``07-translations/`` (so stage 8 joins them with no new
rule) plus ``06b-matter/`` and ``06b-matter-prompts/`` for the plan and debugging.
"""

from .graph import (
    build_graph,
    build_matter_chunk_graph,
    build_matter_graph,
)
from .llm import make_book_llm, make_chunk_llm, make_matter_llm
from .runner import (
    MatterChunkResult,
    MatterResult,
    ProgressHook,
    load_matter_manifest,
    run_matter,
    write_matter_manifest,
    write_matter_trace,
)
from .state import MatterChunkState, MatterPlanState

__all__ = [
    "MatterChunkResult",
    "MatterChunkState",
    "MatterPlanState",
    "MatterResult",
    "ProgressHook",
    "build_graph",
    "build_matter_chunk_graph",
    "build_matter_graph",
    "load_matter_manifest",
    "make_book_llm",
    "make_chunk_llm",
    "make_matter_llm",
    "run_matter",
    "write_matter_manifest",
    "write_matter_trace",
]
