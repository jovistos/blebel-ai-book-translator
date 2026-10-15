"""Graph state for stage 2 (chapter preprocessing).

As in stage 1, everything stored here is *metadata*: line numbers, header texts
(capped), decisions and hashes. The chapter body never enters the state, so it
can never reach a prompt.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from ..agent.state import merge_usage


class PreprocessState(TypedDict, total=False):
    """State carried through the preprocessing graph, one chapter at a time."""

    # --- configuration ---
    chapter_path: str
    output_dir: str
    max_attempts: int
    dry_run: bool
    require_text_fixes: bool

    # --- deterministic recon (no LLM) ---
    line_count: int
    header_report: dict[str, Any]
    damage_report: dict[str, Any]

    # --- the plan the agent submits through `report_chapter_cleaning` ---
    operations: list[dict[str, Any]]
    text_fixes: list[dict[str, Any]]
    proposal_notes: str

    # --- validation / execution ---
    validation: dict[str, Any]
    applied: dict[str, Any]
    verification: dict[str, Any]
    feedback: list[str]

    # --- bookkeeping ---
    attempts: int
    usage: Annotated[dict[str, int], merge_usage]
    log: Annotated[list[str], operator.add]
    messages: Annotated[list, add_messages]
    result: dict[str, Any]
