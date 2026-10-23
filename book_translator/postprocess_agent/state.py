"""Graph state for stage 9 (translated-heading postprocessing).

As in stages 1 and 2, the state holds metadata only: filenames, titles, line
numbers, heading records, plans and hashes. The chapter body never enters it, so
it can never reach a prompt.

The bookkeeping half of both classes lives in
:class:`~book_translator.stage_graph.AgentStateBase`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from ..stage_graph import AgentStateBase


class BookPlanState(AgentStateBase, total=False):
    """State carried through the book-level naming/TOC graph."""

    # --- configuration ---
    chapters_path: str
    require_web_search: bool

    # --- deterministic inventory (no LLM) ---
    book: dict[str, Any]

    # --- mandatory published-contents lookup ---
    web_search_attempted: bool
    web_searches: Annotated[list[dict[str, Any]], operator.add]

    # --- the plan the agent submits through `report_book_plan` ---
    book_plan: dict[str, Any]


class ChapterPostprocessState(AgentStateBase, total=False):
    """State carried through the per-chapter title/heading graph."""

    # --- configuration ---
    chapter_path: str
    chapters_path: str
    output_dir: str
    canonical_title: str

    # --- deterministic recon (no LLM) ---
    line_count: int
    structure: dict[str, Any]

    # --- the plan the agent submits through `report_chapter_postprocess` ---
    chapter_plan: dict[str, Any]

    # --- execution ---
    applied: dict[str, Any]
    verification: dict[str, Any]


__all__ = ["BookPlanState", "ChapterPostprocessState"]
