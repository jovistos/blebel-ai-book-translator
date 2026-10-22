"""Stage 9's plan tools, dispatch tables and payload adapters.

The shared tool executor stays generic; what it needs to know about this stage is
data: which tools form each graph's table, which tool submits a plan, that the
book planner may never write, and that a published-contents search is mandatory
before the book plan.
"""

from __future__ import annotations

import json
from typing import Any

from ..agent.tools import POSTPROCESS_TOOL_BY_NAME
from ..agent.tools.postprocess import BOOK_TOOL_NAMES, CHAPTER_TOOL_NAMES

#: Tool whose accepted result *is* the book plan.
BOOK_PLAN_TOOL = "report_book_plan"
#: Tool whose accepted result *is* a chapter plan.
CHAPTER_PLAN_TOOL = "report_chapter_postprocess"
#: Book-stage web tools whose calls are recorded as evidence. Only ``web_search``
#: satisfies the mandatory search; ``fetch_web_page`` is recorded but does not.
WEB_TOOL_NAMES = frozenset({"web_search", "fetch_web_page"})

#: Resolve the two narrow dispatch tables from the full stage table.
BOOK_TOOL_BY_NAME = {name: POSTPROCESS_TOOL_BY_NAME[name] for name in BOOK_TOOL_NAMES}
CHAPTER_TOOL_BY_NAME = {
    name: POSTPROCESS_TOOL_BY_NAME[name] for name in CHAPTER_TOOL_NAMES
}


def record_web_call(
    _state: dict[str, Any], name: str, args: dict[str, Any], payload: str
) -> dict[str, Any]:
    """Compact, serialisable evidence of one published-contents lookup.

    Only the query/URL and a few result titles are kept — never the page body —
    so the manifest can prove the mandatory search happened without carrying
    external text into the book.
    """
    if name not in WEB_TOOL_NAMES:
        return {}
    updates: dict[str, Any] = {"web_searches": [_web_record(name, args, payload)]}
    if name == "web_search":
        # A failed search still satisfies the mandatory *attempt*; its error is
        # recorded so the plan can cite the failure. A page fetch alone is not a
        # search and does not count.
        updates["web_search_attempted"] = True
    return updates


def _web_record(name: str, args: dict[str, Any], payload: str) -> dict[str, Any]:
    record: dict[str, Any] = {"tool": name}
    if name == "web_search":
        record["query"] = str(args.get("query") or "")[:300]
    else:
        record["url"] = str(args.get("url") or "")[:500]
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return record
    if not isinstance(data, dict):
        return record
    if name == "web_search":
        record["provider"] = data.get("provider")
        record["result_count"] = int(data.get("result_count") or 0)
        results = [
            item for item in data.get("results") or [] if isinstance(item, dict)
        ]
        record["results"] = [
            {
                "title": str(item.get("title") or "")[:200],
                "url": str(item.get("url") or "")[:500],
            }
            for item in results[:5]
        ]
    else:
        record["status"] = data.get("status")
        record["title"] = str(data.get("title") or "")[:200]
        record["text_chars"] = data.get("text_chars")
    if data.get("error"):
        record["error"] = str(data["error"])[:300]
    return record


def extract_book_plan(data: dict[str, Any]) -> dict[str, Any]:
    """Map an accepted ``report_book_plan`` payload to state updates."""
    plan = data.get("plan")
    return {"book_plan": plan} if isinstance(plan, dict) else {}


def extract_chapter_plan(data: dict[str, Any]) -> dict[str, Any]:
    """Map an accepted ``report_chapter_postprocess`` payload to state updates."""
    plan = data.get("plan")
    return {"chapter_plan": plan} if isinstance(plan, dict) else {}


__all__ = [
    "BOOK_PLAN_TOOL",
    "BOOK_TOOL_BY_NAME",
    "CHAPTER_PLAN_TOOL",
    "CHAPTER_TOOL_BY_NAME",
    "WEB_TOOL_NAMES",
    "extract_book_plan",
    "extract_chapter_plan",
    "record_web_call",
]
