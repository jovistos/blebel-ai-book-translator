"""Nodes for the stage-9 graphs.

Book graph: ``survey`` -> ``book_agent`` -> ``book_tools`` -> ``book_validate``.
Chapter graph: ``collect`` -> ``chapter_agent`` -> ``chapter_tools`` ->
``chapter_validate`` -> ``apply`` -> ``verify``.

The tools node is the only place a tool runs. It enforces the stage's write policy:
``apply_chapter_postprocess`` and ``apply_book_toc`` are unreachable until the
relevant plan has passed validation, and the plan itself is absorbed into the
graph state rather than replayed by the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..agent.tools import POSTPROCESS_TOOL_BY_NAME
from ..agent.tools.postprocess import (
    BOOK_TOOL_NAMES,
    CHAPTER_TOOL_NAMES,
    apply_chapter_postprocess,
)
from ..agent.tools.registry import POSTPROCESS_WRITE_TOOLS
from ..core.constants import DEFAULT_POSTPROCESS_ATTEMPTS
from ..documents.chapters import postprocessed_chapters_dir
from ..documents.text import jdump, short_args, summarise_payload
from ..postprocess.inspect import scan_book_structure, scan_chapter_structure
from ..postprocess.plan import (
    apply_chapter_plan,
    validate_book_plan,
    validate_chapter_plan,
    verify_chapter_plan,
)
from .prompts import (
    BOOK_SYSTEM_PROMPT,
    CHAPTER_SYSTEM_PROMPT,
    NUDGE_BOOK_FINAL_TURN,
    NUDGE_BOOK_NO_PLAN,
    NUDGE_CHAPTER_FINAL_TURN,
    NUDGE_CHAPTER_NO_PLAN,
    book_validation_feedback,
    build_book_report,
    build_chapter_report,
    chapter_validation_feedback,
)
from .state import BookPlanState, ChapterPostprocessState

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


def _usage(response: Any) -> dict[str, int]:
    usage = dict(getattr(response, "usage_metadata", None) or {})
    return {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "llm_calls": 1,
    }


def _web_record(name: str, args: dict[str, Any], payload: str) -> dict[str, Any]:
    """Compact, serialisable evidence of one published-contents lookup.

    Only the query/URL and a few result titles are kept — never the page body —
    so the manifest can prove the mandatory search happened without carrying
    external text into the book.
    """
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
        results = [item for item in data.get("results") or [] if isinstance(item, dict)]
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


# --------------------------------------------------------------------------- #
# book graph
# --------------------------------------------------------------------------- #
def make_survey_node():
    """Build ``survey``: the deterministic book inventory, no LLM."""

    def survey(state: BookPlanState) -> dict[str, Any]:
        book = scan_book_structure(state["chapters_path"])
        report = build_book_report(book)
        task = (
            f"Plan the heading postprocessing for the translated book at "
            f"{state['chapters_path']}.\n"
            "Read the whole chapter-title list first, search the web for the published "
            "chapter list (mandatory before the plan), then choose one naming policy and "
            "a table-of-contents action. Do not read chapter prose."
        )
        return {
            "book": book,
            "attempts": 0,
            "messages": [
                SystemMessage(content=BOOK_SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [
                f"survey: {book.get('chapter_count', 0)} files, "
                f"{book.get('body_chapter_count', 0)} body chapters, "
                f"toc={'yes' if (book.get('toc') or {}).get('exists') else 'no'}",
            ],
        }

    return survey


def make_book_agent_node(model: Any):
    """Build ``book_agent``: one LLM turn for the book-level plan."""

    def book_agent(state: BookPlanState) -> dict[str, Any]:
        messages = list(state["messages"])
        feedback = list(state.get("feedback") or [])
        turn_notes: list[str] = []
        if feedback:
            turn_notes.append("\n\n".join(feedback))
        else:
            last = messages[-1] if messages else None
            if isinstance(last, AIMessage) and not last.tool_calls and not state.get(
                "book_plan"
            ):
                turn_notes.append(NUDGE_BOOK_NO_PLAN)
        attempts = int(state.get("attempts", 0)) + 1
        limit = int(state.get("max_attempts", DEFAULT_POSTPROCESS_ATTEMPTS))
        if attempts >= limit and not (state.get("validation") or {}).get("ok"):
            # Last turn: a dry run now would leave the plan unsubmitted again.
            turn_notes.append(NUDGE_BOOK_FINAL_TURN)
        if turn_notes:
            messages.append(HumanMessage(content="\n\n".join(turn_notes)))
        response = model.invoke(messages)
        calls = [call["name"] for call in (getattr(response, "tool_calls", None) or [])]
        return {
            "messages": [response],
            "attempts": attempts,
            "feedback": [],
            "usage": _usage(response),
            "log": [f"book_agent[{attempts}]: tool_calls={calls or 'none'}"],
        }

    return book_agent


def make_book_tool_node():
    """Build ``book_tools``: the book-plan executor."""

    def book_tools(state: BookPlanState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        out_messages: list[ToolMessage] = []
        updates: dict[str, Any] = {}
        log: list[str] = []
        require_web = bool(state.get("require_web_search", True))
        web_search_attempted = bool(state.get("web_search_attempted"))
        web_searches: list[dict[str, Any]] = []

        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{len(out_messages)}"
            if name in POSTPROCESS_WRITE_TOOLS:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "error": f"{name} is blocked at the book-plan stage",
                                "hint": "submit the book plan; the graph applies it",
                            }
                        ),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                log.append(f"book_tools: BLOCKED {name}")
                continue
            tool = BOOK_TOOL_BY_NAME.get(name)
            if tool is None:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "error": f"unknown book-plan tool {name!r}",
                                "available": list(BOOK_TOOL_BY_NAME),
                            }
                        ),
                        tool_call_id=call_id,
                        name=name or "unknown",
                        status="error",
                    )
                )
                log.append(f"book_tools: unknown tool {name}")
                continue
            if name == BOOK_PLAN_TOOL and require_web and not web_search_attempted:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "accepted": False,
                                "error": "web search is mandatory before the book plan",
                                "hint": (
                                    "Call web_search for the book's published chapter "
                                    "list first, then call report_book_plan again. A "
                                    "failed search still counts once you record its "
                                    "outcome in title_evidence."
                                ),
                            }
                        ),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                log.append("book_tools: BLOCKED report_book_plan (web_search required)")
                continue
            try:
                payload = tool.invoke(args)
                status = "success"
            except Exception as exc:
                payload = jdump(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "hint": "check the argument names and types, then retry",
                    }
                )
                status = "error"
            if name in WEB_TOOL_NAMES:
                web_searches.append(_web_record(name, args, payload))
                if name == "web_search":
                    # A failed search still satisfies the mandatory *attempt*;
                    # its error is recorded so the plan can cite the failure.
                    # A page fetch alone is not a search and does not count.
                    web_search_attempted = True
            out_messages.append(
                ToolMessage(content=payload, tool_call_id=call_id, name=name, status=status)
            )
            log.append(f"book_tools: {name}({short_args(args)}) -> {summarise_payload(payload)}")
            if name == BOOK_PLAN_TOOL and status == "success":
                updates.update(absorb_book_plan(payload))
        if web_search_attempted:
            updates["web_search_attempted"] = True
        if web_searches:
            updates["web_searches"] = web_searches
        return {**updates, "messages": out_messages, "log": log}

    return book_tools


def absorb_book_plan(payload: str) -> dict[str, Any]:
    """Read an accepted ``report_book_plan`` result into state updates."""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict) or not data.get("accepted"):
        return {}
    plan = data.get("plan")
    if not isinstance(plan, dict):
        return {}
    return {"book_plan": plan}


def make_book_validate_node():
    """Build ``book_validate``: deterministic validation of the naming/TOC plan."""

    def book_validate(state: BookPlanState) -> dict[str, Any]:
        plan = dict(state.get("book_plan") or {})
        validation = validate_book_plan(
            state["chapters_path"],
            plan,
            book=state.get("book") or None,
            require_web_search=bool(state.get("require_web_search", True)),
        )
        feedback: list[str] = []
        if not validation.get("ok"):
            feedback.append(book_validation_feedback(validation))
        return {
            "validation": validation,
            "feedback": feedback,
            "log": [
                f"book_validate: ok={validation.get('ok')} "
                f"issues={len(validation.get('issues') or [])} "
                f"warnings={len(validation.get('warnings') or [])}"
            ],
        }

    return book_validate


def make_book_give_up_node():
    """Build ``book_give_up``: stop without writing anything."""

    def book_give_up(state: BookPlanState) -> dict[str, Any]:
        return {
            "result": {
                "error": "no valid book plan after the allowed number of attempts",
                "attempts": int(state.get("attempts", 0)),
            },
            "log": ["book_give_up: no valid plan"],
        }

    return book_give_up


# --------------------------------------------------------------------------- #
# chapter graph
# --------------------------------------------------------------------------- #
def make_collect_node():
    """Build ``collect``: the chapter's title/heading metadata."""

    def collect(state: ChapterPostprocessState) -> dict[str, Any]:
        structure = scan_chapter_structure(state["chapter_path"])
        canonical = str(state.get("canonical_title") or structure.title)
        report = build_chapter_report(structure, canonical)
        task = (
            f"Postprocess the headings of {structure.filename}.\n"
            f"chapter_path: {structure.path}\n"
            f"The book plan fixes its chapter title as {canonical!r}. "
            "Set/insert/keep that title, then decide keep/demote for the other headings. "
            "Pass the exact chapter_path above to every chapter tool; never a bare filename. "
            "Never read the chapter's prose."
        )
        return {
            "line_count": structure.line_count,
            "structure": structure.to_dict(),
            "attempts": 0,
            "messages": [
                SystemMessage(content=CHAPTER_SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [
                f"collect: {structure.heading_count} headings, "
                f"{len(structure.suspicious_headings)} suspicious, "
                f"title={'missing' if structure.title_line is None else structure.title_line}"
            ],
        }

    return collect


def make_chapter_agent_node(model: Any):
    """Build ``chapter_agent``: one LLM turn for one chapter plan."""

    def chapter_agent(state: ChapterPostprocessState) -> dict[str, Any]:
        messages = list(state["messages"])
        feedback = list(state.get("feedback") or [])
        turn_notes: list[str] = []
        if feedback:
            turn_notes.append("\n\n".join(feedback))
        else:
            last = messages[-1] if messages else None
            if isinstance(last, AIMessage) and not last.tool_calls and not state.get(
                "chapter_plan"
            ):
                turn_notes.append(NUDGE_CHAPTER_NO_PLAN)
        attempts = int(state.get("attempts", 0)) + 1
        limit = int(state.get("max_attempts", DEFAULT_POSTPROCESS_ATTEMPTS))
        if attempts >= limit and not (state.get("validation") or {}).get("ok"):
            # Last turn: the plan was rejected, and a dry run would not write it.
            turn_notes.append(NUDGE_CHAPTER_FINAL_TURN)
        if turn_notes:
            messages.append(HumanMessage(content="\n\n".join(turn_notes)))
        response = model.invoke(messages)
        calls = [call["name"] for call in (getattr(response, "tool_calls", None) or [])]
        return {
            "messages": [response],
            "attempts": attempts,
            "feedback": [],
            "usage": _usage(response),
            "log": [f"chapter_agent[{attempts}]: tool_calls={calls or 'none'}"],
        }

    return chapter_agent


def make_chapter_tool_node():
    """Build ``chapter_tools``: the per-chapter executor with the write policy."""

    def chapter_tools(state: ChapterPostprocessState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        validation = state.get("validation") or {}
        plan_ok = bool(validation.get("ok"))
        out_messages: list[ToolMessage] = []
        updates: dict[str, Any] = {}
        log: list[str] = []

        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{len(out_messages)}"
            if name in POSTPROCESS_WRITE_TOOLS and not plan_ok:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "error": f"{name} is blocked until this chapter's plan is validated",
                                "hint": "call report_chapter_postprocess, then the graph applies it",
                            }
                        ),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                log.append(f"chapter_tools: BLOCKED {name} (no validated plan)")
                continue
            if plan_ok and name == CHAPTER_PLAN_TOOL:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "accepted": False,
                                "error": "a plan for this chapter has already been accepted",
                            }
                        ),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                log.append(f"chapter_tools: {name} ignored (plan already accepted)")
                continue
            tool = CHAPTER_TOOL_BY_NAME.get(name)
            if tool is None:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "error": f"unknown chapter tool {name!r}",
                                "available": list(CHAPTER_TOOL_BY_NAME),
                            }
                        ),
                        tool_call_id=call_id,
                        name=name or "unknown",
                        status="error",
                    )
                )
                log.append(f"chapter_tools: unknown tool {name}")
                continue
            try:
                payload = tool.invoke(args)
                status = "success"
            except Exception as exc:
                payload = jdump(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "hint": "check the argument names and types, then retry",
                    }
                )
                status = "error"
            out_messages.append(
                ToolMessage(content=payload, tool_call_id=call_id, name=name, status=status)
            )
            log.append(f"chapter_tools: {name}({short_args(args)}) -> {summarise_payload(payload)}")
            if name == CHAPTER_PLAN_TOOL and status == "success":
                updates.update(absorb_chapter_plan(payload))
        return {**updates, "messages": out_messages, "log": log}

    return chapter_tools


def absorb_chapter_plan(payload: str) -> dict[str, Any]:
    """Read an accepted ``report_chapter_postprocess`` result into state updates."""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict) or not data.get("accepted"):
        return {}
    plan = data.get("plan")
    if not isinstance(plan, dict):
        return {}
    return {"chapter_plan": plan}


def make_chapter_validate_node():
    """Build ``chapter_validate``: check the chapter plan against the real file."""

    def chapter_validate(state: ChapterPostprocessState) -> dict[str, Any]:
        validation = validate_chapter_plan(
            state["chapter_path"],
            dict(state.get("chapter_plan") or {}),
            canonical_title=str(state.get("canonical_title") or ""),
        )
        feedback: list[str] = []
        if not validation.get("ok"):
            feedback.append(chapter_validation_feedback(validation))
        return {
            "validation": validation,
            "feedback": feedback,
            "log": [
                f"chapter_validate: ok={validation.get('ok')} "
                f"issues={len(validation.get('issues') or [])} "
                f"warnings={len(validation.get('warnings') or [])}"
            ],
        }

    return chapter_validate


def make_apply_node():
    """Build ``apply``: write the chapter, or validate it in a dry run."""

    def apply(state: ChapterPostprocessState) -> dict[str, Any]:
        chapter = state["chapter_path"]
        plan = dict(state.get("chapter_plan") or {})
        canonical = str(state.get("canonical_title") or "")
        if state.get("dry_run"):
            applied = apply_chapter_plan(chapter, plan, canonical_title=canonical)
            verification = verify_chapter_plan(
                chapter, plan, canonical_title=canonical
            )
            payload = {
                "ok": bool(verification.get("ok")),
                "dry_run": True,
                "output_path": None,
                "output_lines": len(applied.lines),
                "applied": applied.summary(),
                "verification": verification,
            }
            return {
                "applied": payload,
                "log": ["apply: DRY RUN, nothing written"],
            }

        out_dir = state.get("output_dir") or str(postprocessed_chapters_dir(chapter))
        try:
            raw = apply_chapter_postprocess.invoke(
                {
                    "chapter_path": chapter,
                    "plan": plan,
                    "canonical_title": canonical,
                    "output_dir": out_dir,
                }
            )
            payload = json.loads(raw)
        except Exception as exc:
            payload = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "verification": {"ok": False, "problems": [str(exc)]},
            }
        return {
            "applied": payload,
            "log": [
                f"apply: {'wrote' if payload.get('ok') else 'FAILED'} "
                f"{Path(str(payload.get('output_path') or '')).name or 'chapter'}"
            ],
        }

    return apply


def make_verify_node():
    """Build ``verify``: prove the written chapter only changed declared lines."""

    def verify(state: ChapterPostprocessState) -> dict[str, Any]:
        applied = state.get("applied") or {}
        verification = dict(applied.get("verification") or {})
        verification["ok"] = bool(applied.get("ok")) and bool(verification.get("ok"))
        return {
            "verification": verification,
            "log": [f"verify: ok={verification.get('ok')}"],
        }

    return verify


def make_chapter_give_up_node():
    """Build ``chapter_give_up``: stop this chapter without writing it."""

    def chapter_give_up(state: ChapterPostprocessState) -> dict[str, Any]:
        return {
            "result": {
                "error": "no valid chapter plan after the allowed number of attempts",
                "attempts": int(state.get("attempts", 0)),
            },
            "log": ["chapter_give_up: no valid plan"],
        }

    return chapter_give_up


__all__ = [
    "BOOK_PLAN_TOOL",
    "BOOK_TOOL_BY_NAME",
    "CHAPTER_PLAN_TOOL",
    "CHAPTER_TOOL_BY_NAME",
    "absorb_book_plan",
    "absorb_chapter_plan",
    "make_apply_node",
    "make_book_agent_node",
    "make_book_give_up_node",
    "make_book_tool_node",
    "make_book_validate_node",
    "make_chapter_agent_node",
    "make_chapter_give_up_node",
    "make_chapter_tool_node",
    "make_chapter_validate_node",
    "make_collect_node",
    "make_survey_node",
    "make_verify_node",
]
