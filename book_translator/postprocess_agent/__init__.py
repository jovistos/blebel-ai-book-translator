"""Stage 9: postprocessing the translated chapters' headings and TOC.

The stage sits between stage 8's ``08-translated-chapters/`` folder and the final
single-document concatenation. It has two LangGraphs:

``build_postprocess_graph``
    the book-level graph: inventory all chapter titles first, search the web for
    the published chapter list (mandatory; ``report_book_plan`` is refused until
    ``web_search`` has been called), then choose one naming policy and a
    table-of-contents action.
``build_chapter_postprocess_graph``
    one chapter's graph: set/insert/keep the canonical title, then
    keep/demote/promote every other heading with the surrounding evidence.

``run_postprocess``
    drives the book graph, then the chapter graph in reading order, then writes
    the TOC from the final titles.

The output is ``09-postprocessed-chapters/``; the translated chapters are never
edited in place.
"""

from .graph import (
    build_chapter_postprocess_graph,
    build_graph,
    build_postprocess_graph,
)
from .llm import make_book_llm, make_chapter_llm, make_postprocess_llm
from .runner import (
    ChapterPostprocessResult,
    PostprocessResult,
    ProgressHook,
    load_postprocess_manifest,
    run_postprocess,
    summarise_chapter,
)
from .state import BookPlanState, ChapterPostprocessState

__all__ = [
    "BookPlanState",
    "ChapterPostprocessResult",
    "ChapterPostprocessState",
    "PostprocessResult",
    "ProgressHook",
    "build_chapter_postprocess_graph",
    "build_graph",
    "build_postprocess_graph",
    "load_postprocess_manifest",
    "make_book_llm",
    "make_chapter_llm",
    "make_postprocess_llm",
    "run_postprocess",
    "summarise_chapter",
]
