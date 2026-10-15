"""Shared graph state.

Everything stored here is metadata about the book (line numbers, short header
strings, hashes). The book body itself is never stored in the state, so it can
never end up in an LLM prompt.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def merge_usage(left: dict[str, int] | None, right: dict[str, int] | None) -> dict[str, int]:
    """Reducer that sums token usage across graph nodes."""
    out = dict(left or {})
    for key, value in (right or {}).items():
        out[key] = out.get(key, 0) + value
    return out


class SplitterState(TypedDict, total=False):
    """State carried through the LangGraph."""

    # --- configuration ---
    source_path: str
    output_dir: str
    front_matter_lines: int
    max_attempts: int
    min_chapters: int
    include_front_matter: bool
    dry_run: bool

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

    # --- validation / execution ---
    validation: dict[str, Any]
    feedback: list[str]
    manifest: dict[str, Any]
    verification: dict[str, Any]

    # --- bookkeeping ---
    attempts: int
    usage: Annotated[dict[str, int], merge_usage]
    log: Annotated[list[str], operator.add]
    messages: Annotated[list, add_messages]
    result: dict[str, Any]
