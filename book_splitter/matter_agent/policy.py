"""The matter stage's plan tools, dispatch tables and payload adapters."""

from __future__ import annotations

from typing import Any

from ..agent.tools import MATTER_TOOL_BY_NAME
from ..agent.tools.matter import BOOK_TOOL_NAMES, CHUNK_TOOL_NAMES

#: Tool whose accepted result *is* the book plan.
BOOK_PLAN_TOOL = "report_matter_plan"
#: Tool whose accepted result *is* one chunk's localization.
CHUNK_REPORT_TOOL = "report_matter_localization"

BOOK_TOOL_BY_NAME = {name: MATTER_TOOL_BY_NAME[name] for name in BOOK_TOOL_NAMES}
CHUNK_TOOL_BY_NAME = {name: MATTER_TOOL_BY_NAME[name] for name in CHUNK_TOOL_NAMES}


def extract_matter_plan(data: dict[str, Any]) -> dict[str, Any]:
    """Map an accepted ``report_matter_plan`` payload to state updates."""
    plan = data.get("plan")
    if not isinstance(plan, dict):
        return {}
    validation = data.get("validation")
    return {
        "matter_plan": plan,
        "validation": validation if isinstance(validation, dict) else {},
    }


def extract_chunk_localization(data: dict[str, Any]) -> dict[str, Any]:
    """Map an accepted ``report_matter_localization`` payload to state updates."""
    text = str(data.get("text") or "")
    return {
        "chunk_plan": {
            "translation": text,
            "text": text,
            "omitted": bool(data.get("omitted")),
            "omit_reason": str(data.get("omit_reason") or ""),
            "notes": str(data.get("notes") or ""),
            "title_used": str(data.get("title_used") or ""),
        }
    }


__all__ = [
    "BOOK_PLAN_TOOL",
    "BOOK_TOOL_BY_NAME",
    "CHUNK_REPORT_TOOL",
    "CHUNK_TOOL_BY_NAME",
    "extract_chunk_localization",
    "extract_matter_plan",
]
