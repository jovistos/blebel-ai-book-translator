"""Stage 9: heading postprocessing for assembled translated chapters.

The stage is deliberately different from a prose editor. It looks at:

* the **book-level inventory** of chapter titles (so one naming policy can be
  chosen before any chapter is touched),
* the **opening** of each chapter (a handful of clipped lines — the only place a
  chapter title can be),
* every **heading record** with a two-line surrounding context, and
* bounded **front/back matter** blocks (to detect and build the table of
  contents).

It never returns a chapter's body. The agent describes edits in a fixed
vocabulary (``keep`` / ``set`` / ``insert`` for the chapter title; ``keep`` /
``demote`` / ``promote`` for every other heading) and a deterministic validator
checks every entry against the real file before the graph writes the output
folder. ``promote`` puts the ``#`` markup back on a plain ALL-CAPS section
heading that lost it, so the candidate list is not limited to Markdown headers.

The public surface is intentionally small::

    scan_book_structure()        the book-level inventory
    scan_chapter_structure()     one chapter's title + heading records
    inspect_matter_blocks()      bounded front/back matter + TOC status
    validate_book_plan()         dry-run the book-wide naming/TOC decision
    validate_chapter_plan()      dry-run one chapter's plan
    apply_chapter_plan()         pure line rewrite (no writes)
    verify_chapter_plan()        proof that only declared lines changed
    detect_toc() / apply_toc()   locate, build and rewrite the TOC

The validator rules live in :mod:`book_translator.postprocess.rules`, one module
per rule group; ``plan.py`` only orchestrates them and keeps the public names.
"""

from __future__ import annotations

from typing import Any

from .inspect import (
    ChapterStructure,
    chapter_files,
    find_title_candidate,
    inspect_heading_context,
    inspect_matter_blocks,
    is_toc_filename,
    scan_book_structure,
    scan_chapter_structure,
)
from .plan import (
    apply_chapter_plan,
    canonical_titles,
    validate_book_plan,
    validate_chapter_plan,
    verify_chapter_plan,
)
from .toc import (
    apply_toc,
    build_toc_entries,
    detect_toc,
    render_toc,
    verify_toc,
)

__all__ = [
    "ChapterPostprocessResult",
    "ChapterStructure",
    "PostprocessResult",
    "apply_chapter_plan",
    "apply_toc",
    "build_chapter_postprocess_graph",
    "build_postprocess_graph",
    "build_toc_entries",
    "canonical_titles",
    "chapter_files",
    "detect_toc",
    "find_title_candidate",
    "inspect_heading_context",
    "inspect_matter_blocks",
    "is_toc_filename",
    "render_toc",
    "run_postprocess",
    "scan_book_structure",
    "scan_chapter_structure",
    "validate_book_plan",
    "validate_chapter_plan",
    "verify_chapter_plan",
    "verify_toc",
]

_AGENT_EXPORTS = {
    "ChapterPostprocessResult",
    "PostprocessResult",
    "build_chapter_postprocess_graph",
    "build_postprocess_graph",
    "run_postprocess",
}


def __getattr__(name: str) -> Any:
    if name in _AGENT_EXPORTS:
        from .. import postprocess_agent
        return getattr(postprocess_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

