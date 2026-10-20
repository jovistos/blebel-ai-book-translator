"""Nodes for the front/back matter graphs.

Book graph: ``survey`` -> ``book_agent`` -> ``book_tools`` -> ``book_validate``.

Chunk graph: ``collect`` -> ``chunk_agent`` -> ``chunk_tools`` ->
``chunk_validate`` -> ``apply`` -> ``verify``.

The tools node is the only place a tool runs. It absorbs the plan/translation a
report tool accepted into the graph state; the apply node is the only writer, and
it writes exactly the text the validator approved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..agent.tools import MATTER_TOOL_BY_NAME
from ..agent.tools.matter import BOOK_TOOL_NAMES, CHUNK_TOOL_NAMES
from ..documents.text import jdump, short_args, summarise_payload
from ..matter.plan import (
    preserve_trailing_newlines,
)
from ..matter.plan import (
    validate_matter_plan as validate_matter_plan_fn,
)
from ..matter.plan import (
    validate_matter_translation as validate_matter_translation_fn,
)
from .prompts import (
    BOOK_SYSTEM_PROMPT,
    CHUNK_SYSTEM_PROMPT,
    NUDGE_BOOK_NO_PLAN,
    NUDGE_CHUNK_NO_TRANSLATION,
    book_validation_feedback,
    build_book_report,
    build_chunk_report,
    chunk_validation_feedback,
)
from .state import MatterChunkState, MatterPlanState

#: Tool whose accepted result *is* the book plan.
BOOK_PLAN_TOOL = "report_matter_plan"
#: Tool whose accepted result *is* one chunk's localization.
CHUNK_REPORT_TOOL = "report_matter_localization"

BOOK_TOOL_BY_NAME = {name: MATTER_TOOL_BY_NAME[name] for name in BOOK_TOOL_NAMES}
CHUNK_TOOL_BY_NAME = {name: MATTER_TOOL_BY_NAME[name] for name in CHUNK_TOOL_NAMES}


def _usage(response: Any) -> dict[str, int]:
    usage = dict(getattr(response, "usage_metadata", None) or {})
    return {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "llm_calls": 1,
    }


def _tool_calls(response: Any) -> list[str]:
    return [call["name"] for call in (getattr(response, "tool_calls", None) or [])]


# --------------------------------------------------------------------------- #
# book graph
# --------------------------------------------------------------------------- #
def make_survey_node():
    """Build ``survey``: deterministic matter inventory, then the planning task."""

    def survey(state: MatterPlanState) -> dict[str, Any]:
        book = state.get("book")
        if book is None:
            raise ValueError("the matter graph needs the book inventory in its state")
        report = build_book_report(book)
        task = (
            f"Plan the localization of the front/back matter of {book.book_title!r}"
            + (f" by {book.book_author}" if book.book_author else "")
            + f" ({book.source_language} -> {book.target_language}).\n"
            "Read the matter files, search for an existing published translation of the "
            "title, and submit one binding plan. The front matter must present the book "
            "as a translation and credit BleBel."
        )
        return {
            "attempts": 0,
            "messages": [
                SystemMessage(content=BOOK_SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [
                f"survey: {len(book.sections)} matter section(s), "
                f"{book.chunk_count} matter chunk(s), {len(book.body_chapters)} body chapter(s)",
            ],
        }

    return survey


def make_book_agent_node(model: Any):
    """Build ``book_agent``: one LLM turn for the localization plan."""

    def book_agent(state: MatterPlanState) -> dict[str, Any]:
        messages = list(state["messages"])
        feedback = list(state.get("feedback") or [])
        if feedback:
            messages.append(HumanMessage(content="\n\n".join(feedback)))
        else:
            last = messages[-1] if messages else None
            if isinstance(last, AIMessage) and not last.tool_calls and not state.get(
                "matter_plan"
            ):
                messages.append(HumanMessage(content=NUDGE_BOOK_NO_PLAN))
        attempts = int(state.get("attempts", 0)) + 1
        response = model.invoke(messages)
        return {
            "messages": [response],
            "attempts": attempts,
            "feedback": [],
            "usage": _usage(response),
            "log": [f"book_agent[{attempts}]: tool_calls={_tool_calls(response) or 'none'}"],
        }

    return book_agent


def make_book_tool_node():
    """Build ``book_tools``: the plan executor."""

    def book_tools(state: MatterPlanState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        out_messages: list[ToolMessage] = []
        updates: dict[str, Any] = {}
        log: list[str] = []

        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{len(out_messages)}"
            tool = BOOK_TOOL_BY_NAME.get(name)
            if tool is None:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "error": f"unknown matter-plan tool {name!r}",
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
            log.append(f"book_tools: {name}({short_args(args)}) -> {summarise_payload(payload)}")
            if name == BOOK_PLAN_TOOL and status == "success":
                updates.update(absorb_book_plan(payload))
        return {**updates, "messages": out_messages, "log": log}

    return book_tools


def absorb_book_plan(payload: str) -> dict[str, Any]:
    """Read an accepted ``report_matter_plan`` result into state updates."""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict) or not data.get("accepted"):
        return {}
    plan = data.get("plan")
    if not isinstance(plan, dict):
        return {}
    validation = data.get("validation")
    return {
        "matter_plan": plan,
        "validation": validation if isinstance(validation, dict) else {},
    }


def make_book_validate_node():
    """Build ``book_validate``: deterministic plan validation."""

    def book_validate(state: MatterPlanState) -> dict[str, Any]:
        book = state.get("book")
        plan = dict(state.get("matter_plan") or {})
        validation = validate_matter_plan_fn(
            book,
            plan,
            target_language=str(getattr(book, "target_language", "") or ""),
        )
        feedback: list[str] = []
        if not validation.get("ok"):
            feedback.append(book_validation_feedback(validation))
        return {
            "validation": validation,
            "matter_plan": validation.get("plan") or plan,
            "feedback": feedback,
            "log": [
                f"book_validate: ok={validation.get('ok')} "
                f"issues={len(validation.get('issues') or [])} "
                f"warnings={len(validation.get('warnings') or [])}"
            ],
        }

    return book_validate


def make_book_give_up_node():
    """Build ``book_give_up``: stop without writing a translation."""

    def book_give_up(state: MatterPlanState) -> dict[str, Any]:
        return {
            "result": {
                "error": "no valid localization plan after the allowed number of attempts",
                "attempts": int(state.get("attempts", 0)),
            },
            "log": ["book_give_up: no valid plan"],
        }

    return book_give_up


# --------------------------------------------------------------------------- #
# chunk graph
# --------------------------------------------------------------------------- #
def make_collect_node():
    """Build ``collect``: the section metadata and the exact chunk text."""

    def collect(state: MatterChunkState) -> dict[str, Any]:
        section = state.get("section")
        chunk = state.get("chunk")
        if section is None or chunk is None:
            raise ValueError("the matter chunk graph needs a section and a chunk")
        report = build_chunk_report(
            section=section,
            chunk=chunk,
            plan=dict(state.get("plan") or {}),
            source_language=str(state.get("source_language") or ""),
            target_language=str(state.get("target_language") or ""),
            is_first_front_chunk=bool(state.get("is_first_front_chunk")),
            is_first_section_chunk=bool(state.get("is_first_section_chunk")),
        )
        task = (
            f"Localize {getattr(chunk, 'filename', '')} from "
            f"{section.filename} ({section.kind} matter).\n"
            "Return the complete localized chunk through report_matter_localization."
        )
        return {
            "attempts": 0,
            "messages": [
                SystemMessage(content=CHUNK_SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [
                f"collect: {getattr(chunk, 'filename', '')} "
                f"({len(getattr(chunk, 'text', '') or '')} chars), "
                f"first_front={bool(state.get('is_first_front_chunk'))}"
            ],
        }

    return collect


def make_chunk_agent_node(model: Any):
    """Build ``chunk_agent``: one LLM turn for one matter chunk."""

    def chunk_agent(state: MatterChunkState) -> dict[str, Any]:
        messages = list(state["messages"])
        feedback = list(state.get("feedback") or [])
        if feedback:
            messages.append(HumanMessage(content="\n\n".join(feedback)))
        else:
            last = messages[-1] if messages else None
            if isinstance(last, AIMessage) and not last.tool_calls and not state.get(
                "chunk_plan"
            ):
                messages.append(HumanMessage(content=NUDGE_CHUNK_NO_TRANSLATION))
        attempts = int(state.get("attempts", 0)) + 1
        response = model.invoke(messages)
        return {
            "messages": [response],
            "attempts": attempts,
            "feedback": [],
            "usage": _usage(response),
            "log": [f"chunk_agent[{attempts}]: tool_calls={_tool_calls(response) or 'none'}"],
        }

    return chunk_agent


def make_chunk_tool_node():
    """Build ``chunk_tools``: the per-chunk localization executor."""

    def chunk_tools(state: MatterChunkState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        out_messages: list[ToolMessage] = []
        updates: dict[str, Any] = {}
        log: list[str] = []

        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{len(out_messages)}"
            tool = CHUNK_TOOL_BY_NAME.get(name)
            if tool is None:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "error": f"unknown matter localization tool {name!r}",
                                "available": list(CHUNK_TOOL_BY_NAME),
                            }
                        ),
                        tool_call_id=call_id,
                        name=name or "unknown",
                        status="error",
                    )
                )
                log.append(f"chunk_tools: unknown tool {name}")
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
            log.append(f"chunk_tools: {name}({short_args(args)}) -> {summarise_payload(payload)}")
            if name == CHUNK_REPORT_TOOL and status == "success":
                updates.update(absorb_chunk_plan(payload))
        return {**updates, "messages": out_messages, "log": log}

    return chunk_tools


def absorb_chunk_plan(payload: str) -> dict[str, Any]:
    """Read an accepted ``report_matter_localization`` result into state updates."""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict) or not data.get("accepted"):
        return {}
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


def make_chunk_validate_node():
    """Build ``chunk_validate``: check the localization against the real chunk."""

    def chunk_validate(state: MatterChunkState) -> dict[str, Any]:
        section = state.get("section")
        chunk = state.get("chunk")
        plan = dict(state.get("plan") or {})
        submitted = dict(state.get("chunk_plan") or {})
        validation = validate_matter_translation_fn(
            source_text=str(getattr(chunk, "text", "") or ""),
            section_kind=str(getattr(section, "kind", "front")),
            section_filename=str(getattr(section, "filename", "")),
            section_title=str(getattr(section, "title", "")),
            plan=plan,
            translation=str(submitted.get("translation") or submitted.get("text") or ""),
            omitted=bool(submitted.get("omitted")),
            omit_reason=str(submitted.get("omit_reason") or ""),
            is_first_front_chunk=bool(state.get("is_first_front_chunk")),
            is_first_section_chunk=bool(state.get("is_first_section_chunk")),
        )
        feedback: list[str] = []
        if not validation.get("ok"):
            feedback.append(chunk_validation_feedback(validation))
        return {
            "validation": validation,
            "translation": str(validation.get("text") or ""),
            "feedback": feedback,
            "log": [
                f"chunk_validate: ok={validation.get('ok')} "
                f"omitted={bool(submitted.get('omitted'))} "
                f"chars={len(validation.get('text') or '')}"
            ],
        }

    return chunk_validate


def make_apply_node():
    """Build ``apply``: write the validated localization, or nothing on a dry run."""

    def apply(state: MatterChunkState) -> dict[str, Any]:
        chunk = state.get("chunk")
        text = str(state.get("translation") or "")
        submitted = dict(state.get("chunk_plan") or {})
        omitted = bool(submitted.get("omitted"))
        source = str(getattr(chunk, "text", "") or "")
        # An omitted chunk is empty — except the first front-matter chunk, where
        # the validator has already put the localized title and the BleBel credit
        # into `translation`. Never throw that title page away because the source
        # imprint inside it was omitted.
        final = preserve_trailing_newlines(text, source) if text else ""

        if state.get("dry_run") or state.get("save") is False:
            return {
                "applied": {
                    "ok": True,
                    "dry_run": True,
                    "output_path": None,
                    "output_chars": len(final),
                    "omitted": omitted,
                },
                "log": ["apply: DRY RUN, nothing written"],
            }
        out_dir = Path(str(state.get("output_dir") or "")).expanduser()
        target = out_dir / str(getattr(chunk, "filename", "chunk.md"))
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(final, encoding="utf-8")
            payload = {
                "ok": True,
                "dry_run": False,
                "output_path": str(target),
                "output_chars": len(final),
                "omitted": omitted,
            }
        except OSError as exc:
            payload = {
                "ok": False,
                "dry_run": False,
                "output_path": str(target),
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "applied": payload,
            "log": [
                f"apply: {'wrote' if payload.get('ok') else 'FAILED'} {target.name} "
                f"({len(final)} chars)"
            ],
        }

    return apply


def make_verify_node():
    """Build ``verify``: prove the file holds exactly the validated text."""

    def verify(state: MatterChunkState) -> dict[str, Any]:
        applied = state.get("applied") or {}
        expected = str(state.get("translation") or "")
        source = str(getattr(state.get("chunk"), "text", "") or "")
        expected = preserve_trailing_newlines(expected, source) if expected else ""
        problems: list[str] = []
        if not applied.get("ok"):
            problems.append(str(applied.get("error") or "the write failed"))
        if not state.get("dry_run") and state.get("save") is not False:
            path = Path(str(applied.get("output_path") or ""))
            if not path.is_file():
                problems.append("the output file does not exist")
            else:
                try:
                    actual = path.read_text(encoding="utf-8")
                except OSError as exc:
                    problems.append(f"the output file cannot be read: {exc}")
                else:
                    if actual != expected:
                        problems.append(
                            f"the output file differs from the validated text "
                            f"({len(actual)} vs {len(expected)} characters)"
                        )
        verification = {
            "ok": not problems,
            "dry_run": bool(state.get("dry_run")),
            "problems": problems,
            "chars": len(expected),
        }
        return {
            "verification": verification,
            "log": [f"verify: ok={verification['ok']}"],
        }

    return verify


def make_chunk_give_up_node():
    """Build ``chunk_give_up``: stop this chunk without writing it."""

    def chunk_give_up(state: MatterChunkState) -> dict[str, Any]:
        return {
            "result": {
                "error": "no valid localization after the allowed number of attempts",
                "attempts": int(state.get("attempts", 0)),
            },
            "log": ["chunk_give_up: no valid localization"],
        }

    return chunk_give_up


__all__ = [
    "BOOK_PLAN_TOOL",
    "BOOK_TOOL_BY_NAME",
    "CHUNK_REPORT_TOOL",
    "CHUNK_TOOL_BY_NAME",
    "absorb_book_plan",
    "absorb_chunk_plan",
    "make_apply_node",
    "make_book_agent_node",
    "make_book_give_up_node",
    "make_book_tool_node",
    "make_book_validate_node",
    "make_chunk_agent_node",
    "make_chunk_give_up_node",
    "make_chunk_tool_node",
    "make_chunk_validate_node",
    "make_collect_node",
    "make_survey_node",
    "make_verify_node",
]
