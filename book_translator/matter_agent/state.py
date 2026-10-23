"""Graph state for the front/back matter localization stage.

The book graph carries the inventory and the binding plan; a chunk graph carries
one section/chunk pair and the plan it must obey. Prose is carried in the state
on purpose here — unlike stages 1, 2 and 9 — because localizing text is the job.

The bookkeeping half of both classes lives in
:class:`~book_translator.stage_graph.AgentStateBase`.
"""

from __future__ import annotations

from typing import Any

from ..stage_graph import AgentStateBase


class MatterPlanState(AgentStateBase, total=False):
    """State of the book-level localization-plan graph."""

    # --- configuration ---
    chapters_path: str
    chunks_path: str
    book_title: str
    source_language: str
    target_language: str

    # --- deterministic inventory (no LLM) ---
    book: Any

    # --- the plan the agent submits through `report_matter_plan` ---
    matter_plan: dict[str, Any]


class MatterChunkState(AgentStateBase, total=False):
    """State of one front/back matter chunk's localization graph."""

    # --- configuration ---
    chapters_path: str
    chunks_path: str
    output_dir: str
    prompts_dir: str
    section: Any
    chunk: Any
    plan: dict[str, Any]
    source_language: str
    target_language: str
    save: bool
    is_first_front_chunk: bool
    is_first_section_chunk: bool

    # --- deterministic recon (no LLM) ---
    book: Any

    # --- the answer the agent submits through `report_matter_localization` ---
    chunk_plan: dict[str, Any]

    # --- execution ---
    translation: str
    applied: dict[str, Any]
    verification: dict[str, Any]


__all__ = ["MatterChunkState", "MatterPlanState"]
