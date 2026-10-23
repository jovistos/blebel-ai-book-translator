"""Stage 2's plan tool and its payload adapter."""

from __future__ import annotations

from typing import Any

#: Tool whose accepted result *is* this chapter's cleaning plan.
PLAN_TOOL = "report_chapter_cleaning"


def extract_cleaning_plan(data: dict[str, Any]) -> dict[str, Any]:
    """Map an accepted ``report_chapter_cleaning`` payload to state updates."""
    return {
        "operations": data.get("operations") or [],
        "text_fixes": [
            {
                "line_number": fix["line_number"],
                "fix": fix.get("fix"),
                "note": fix.get("note", ""),
            }
            for fix in (data.get("text_fixes") or [])
            if isinstance(fix, dict) and fix.get("fix")
        ],
        "proposal_notes": data.get("notes", ""),
    }


__all__ = ["PLAN_TOOL", "extract_cleaning_plan"]
