"""Graph state for stage 2 (chapter preprocessing).

As in stage 1, everything stored here is *metadata*: line numbers, header texts
(capped), decisions and hashes. The chapter body never enters the state, so it
can never reach a prompt.

The bookkeeping half of this class lives in
:class:`~book_splitter.stage_graph.AgentStateBase`.
"""

from __future__ import annotations

from typing import Any

from ..stage_graph import AgentStateBase


class PreprocessState(AgentStateBase, total=False):
    """State carried through the preprocessing graph, one chapter at a time."""

    # --- configuration ---
    chapter_path: str
    output_dir: str
    require_text_fixes: bool

    # --- deterministic recon (no LLM) ---
    line_count: int
    header_report: dict[str, Any]
    damage_report: dict[str, Any]

    # --- the plan the agent submits through `report_chapter_cleaning` ---
    operations: list[dict[str, Any]]
    text_fixes: list[dict[str, Any]]
    proposal_notes: str

    # --- execution ---
    applied: dict[str, Any]
    verification: dict[str, Any]


__all__ = ["PreprocessState"]
