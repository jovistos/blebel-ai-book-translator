"""Stage 1 state, and the token-usage reducer the other stages import.

Everything stored here is metadata about the book (line numbers, short header
strings, hashes). The book body itself is never stored in the state, so it can
never end up in an LLM prompt.

The bookkeeping half of this class lives in
:class:`~book_translator.stage_graph.AgentStateBase`; only the splitter's own keys
are declared here.
"""

from __future__ import annotations

from typing import Any

from book_translator.stage_graph import AgentStateBase, merge_usage


class SplitterState(AgentStateBase, total=False):
    """State carried through the splitting graph."""

    # --- configuration ---
    source_path: str
    output_dir: str
    front_matter_lines: int
    min_chapters: int
    include_front_matter: bool

    # --- deterministic recon (no LLM) ---
    line_count: int
    native_toc: dict[str, Any]
    headers: list[dict[str, Any]]
    header_assessment: dict[str, Any]
    toc_block: dict[str, Any]

    # --- agent-produced evidence ---
    candidates: list[dict[str, Any]]
    locator_matches: list[dict[str, Any]]

    # --- the plan the agent submits through the `propose_split_map` tool ---
    proposed_splits: list[dict[str, Any]]
    proposal_strategy: str
    proposal_notes: str

    # --- execution ---
    manifest: dict[str, Any]
    verification: dict[str, Any]


__all__ = ["SplitterState", "merge_usage"]
