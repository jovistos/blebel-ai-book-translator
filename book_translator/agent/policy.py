"""Payload adapters for the splitter's accepted plan.

The tool executor is shared by every stage; only the shape of a stage's plan
payload differs. Keeping that mapping here lets the executor stay generic without
knowing what a split map looks like.
"""

from __future__ import annotations

from typing import Any


def extract_split_plan(data: dict[str, Any]) -> dict[str, Any]:
    """Map an accepted ``propose_split_map`` payload to state updates."""
    return {
        "proposed_splits": data["split_mapping"],
        "proposal_strategy": data.get("strategy", "unknown"),
        "proposal_notes": data.get("notes", ""),
    }


__all__ = ["extract_split_plan"]
