"""The stage-2 graph nodes: recon, one LLM turn, the tool executor, planning.

Every node is small and has one job, exactly as in stage 1:

``collect``
    deterministic recon: the chapter's headers, their evidence and the damage
    report. No LLM, no writes, no prose.
``agent``
    one LLM turn, which either calls tools or finishes.
``tools``
    the executor: dispatches tool calls, enforces the write policy for this
    chapter, and absorbs a submitted cleaning plan.
``validate``
    deterministically check the plan against the real chapter.
``apply``
    write the cleaned chapter (only reachable after validation passed).
``verify``
    prove the output only differs in the declared ways.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..agent.tools import PREPROCESS_TOOL_BY_NAME
from ..agent.tools.registry import PREPROCESS_WRITE_TOOLS
from ..documents.chapters import preprocessed_dir
from ..documents.text import jdump, short_args, summarise_payload
from ..preprocess.corpus import DAMAGE_PATTERNS
from ..preprocess.damage import damage_digest, scan_chapter_damage
from ..preprocess.inspect import chapter_digest, inspect_chapter
from ..preprocess.transform import (
    apply_cleaning_plan,
    text_of,
    validate_cleaning_plan,
    validate_text_fixes,
    verify_cleaning_plan,
)
from .prompts import NUDGE_NO_PROPOSAL, SYSTEM_PROMPT, build_report
from .state import PreprocessState

#: Name of the tool whose result *is* the plan.
PLAN_TOOL = "report_chapter_cleaning"


def make_collect_node():
    """Build the ``collect`` node: header + damage recon for one chapter."""

    def collect(state: PreprocessState) -> dict[str, Any]:
        chapter = state["chapter_path"]
        header_report = inspect_chapter(chapter)
        damage_report = scan_chapter_damage(chapter)
        for name in DAMAGE_PATTERNS:
            damage_report["by_kind"].setdefault(name, 0)

        report = build_report(
            filename=header_report["chapter"]["filename"],
            line_count=header_report["chapter"]["line_count"],
            header_report=header_report,
            damage_report=damage_report,
            header_digest_text=chapter_digest(header_report),
            damage_digest_text=damage_digest(damage_report),
        )
        task = (
            f"Preprocess the chapter at {chapter}.\n"
            "Follow the protocol: collect the headers, decide on each one, validate, then report. "
            "Never read the chapter's prose."
        )
        return {
            "line_count": header_report["chapter"]["line_count"],
            "header_report": header_report,
            "damage_report": damage_report,
            "attempts": 0,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"{task}\n\n{report}"),
            ],
            "log": [f"collect: {header_report['header_count']} headers, {damage_report['finding_count']} damage findings"],
        }

    return collect


def make_agent_node(model: Any):
    """Build the ``agent`` node around an already tool-bound chat model."""

    def agent(state: PreprocessState) -> dict[str, Any]:
        messages = list(state["messages"])
        feedback = list(state.get("feedback") or [])
        if feedback:
            messages.append(HumanMessage(content="\n\n".join(feedback)))
        else:
            last = messages[-1] if messages else None
            if isinstance(last, AIMessage) and not last.tool_calls and not state.get("operations"):
                messages.append(HumanMessage(content=NUDGE_NO_PROPOSAL))

        attempts = int(state.get("attempts", 0)) + 1
        response = model.invoke(messages)
        usage = dict(getattr(response, "usage_metadata", None) or {})
        calls = [c["name"] for c in (getattr(response, "tool_calls", None) or [])]
        return {
            "messages": [response],
            "attempts": attempts,
            "feedback": [],
            "usage": {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "llm_calls": 1,
            },
            "log": [f"agent[{attempts}]: tool_calls={calls or 'none'}"],
        }

    return agent


def make_tool_node():
    """Build the ``tools`` node.

    Stage 2's write policy is per chapter: ``apply_chapter_cleaning`` may only be
    called from the executor when *this* chapter's plan has already been
    validated, which is why the plan is absorbed here and never handed to the
    model as something it can replay.
    """

    def tools_node(state: PreprocessState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        validation = state.get("validation") or {}
        plan_ok = bool(validation.get("ok"))
        out_messages: list[ToolMessage] = []
        updates: dict[str, Any] = {}
        log: list[str] = []

        for call in calls:
            name = call.get("name", "")
            args = dict(call.get("args") or {})
            call_id = call.get("id") or f"call_{len(out_messages)}"

            if plan_ok and (state.get("operations") or state.get("text_fixes")):
                # Hygiene: the plan is already known; a second submission is a stall.
                if name == PLAN_TOOL:
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
                    log.append(f"tools: {name} ignored (plan already accepted)")
                    continue
            elif name in PREPROCESS_WRITE_TOOLS:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "error": f"{name} is blocked until this chapter's plan is validated",
                                "hint": "call report_chapter_cleaning, then the graph applies it",
                            }
                        ),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                log.append(f"tools: BLOCKED {name} (no validated plan for this chapter)")
                continue

            tool = PREPROCESS_TOOL_BY_NAME.get(name)
            if tool is None:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {
                                "error": f"unknown tool {name!r}",
                                "available": list(PREPROCESS_TOOL_BY_NAME),
                            }
                        ),
                        tool_call_id=call_id,
                        name=name or "unknown",
                        status="error",
                    )
                )
                log.append(f"tools: unknown tool {name}")
                continue

            try:
                payload = tool.invoke(args)
                status = "success"
            except Exception as exc:  # bad args, missing file, ...
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
            log.append(f"tools: {name}({short_args(args)}) -> {summarise_payload(payload)}")
            if name == PLAN_TOOL and status == "success":
                updates.update(absorb_plan(payload))

        return {**updates, "messages": out_messages, "log": log}

    return tools_node


def absorb_plan(payload: str) -> dict[str, Any]:
    """Read an accepted ``report_chapter_cleaning`` result into state updates."""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict) or not data.get("accepted"):
        return {}
    return {
        "operations": data.get("operations") or [],
        "text_fixes": [
            {"line_number": f["line_number"], "fix": f.get("fix"), "note": f.get("note", "")}
            for f in (data.get("text_fixes") or [])
            if isinstance(f, dict) and f.get("fix")
        ],
        "proposal_notes": data.get("notes", ""),
    }


def make_validate_node():
    """Build the ``validate`` node: check the plan and its text fixes."""

    def validate(state: PreprocessState) -> dict[str, Any]:
        chapter = state["chapter_path"]
        operations = list(state.get("operations") or [])
        text_fixes = list(state.get("text_fixes") or [])

        result = validate_cleaning_plan(chapter, operations)
        accepted_fixes, fix_issues, fix_warnings = validate_text_fixes(chapter, text_fixes)
        result["text_fixes"] = accepted_fixes
        result["issues"] = list(result.get("issues") or []) + fix_issues
        result["warnings"] = list(result.get("warnings") or []) + fix_warnings
        result["ok"] = bool(result.get("ok")) and not fix_issues and bool(operations)
        if not operations:
            result["issues"].append(
                "the plan is empty: submit at least one operation (every chapter has at "
                "least one header to consider)"
            )
            result["ok"] = False
        feedback: list[str] = []
        if not result["ok"]:
            feedback.append(
                "Your cleaning plan for this chapter was rejected. Fix these and report again:\n"
                + "\n".join(f"- {issue}" for issue in result["issues"])
            )
        return {
            "validation": result,
            "feedback": feedback,
            "log": [
                f"validate: ok={result['ok']} issues={len(result['issues'])} "
                f"warnings={len(result['warnings'])}"
            ],
        }

    return validate


def make_apply_node():
    """Build the ``apply`` node: write the cleaned chapter (deterministic)."""

    def apply(state: PreprocessState) -> dict[str, Any]:
        chapter = state["chapter_path"]
        operations = list(state.get("operations") or [])
        text_fixes = list(state.get("validation", {}).get("text_fixes") or [])
        if state.get("dry_run"):
            result = apply_cleaning_plan(chapter, operations, text_fixes)
            return {
                "applied": {
                    "ok": True,
                    "dry_run": True,
                    "output_lines": len(result.lines),
                    "provenance": result.summary(),
                },
                "log": ["apply: DRY RUN, nothing written"],
            }

        out_dir = state.get("output_dir") or str(preprocessed_dir(chapter))
        from pathlib import Path

        target_dir = Path(out_dir).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(chapter).name

        result = apply_cleaning_plan(chapter, operations, text_fixes)
        target.write_text(text_of(result), encoding="utf-8")
        verification = verify_cleaning_plan(chapter, operations, text_fixes)
        payload = {
            "ok": bool(verification.get("ok")),
            "chapter_path": chapter,
            "output_path": str(target),
            "output_lines": len(result.lines),
            "source_lines": verification.get("source_lines"),
            "removed_artefacts": result.removed_artefacts,
            "provenance": result.summary(),
            "verification": verification,
        }
        return {
            "applied": payload,
            "log": [f"apply: wrote {target.name} ({payload['output_lines']} lines)"],
        }

    return apply


def make_verify_node():
    """Build the ``verify`` node: re-check the written file's provenance."""

    def verify(state: PreprocessState) -> dict[str, Any]:
        applied = state.get("applied") or {}
        verification = dict(applied.get("verification") or {})
        verification["ok"] = bool(applied.get("ok")) and bool(verification.get("ok"))
        return {
            "verification": verification,
            "log": [f"verify: ok={verification.get('ok')}"],
        }

    return verify


def make_give_up_node():
    """Build the ``give_up`` node: stop without writing anything."""

    def give_up(state: PreprocessState) -> dict[str, Any]:
        return {
            "result": {
                "error": "no valid cleaning plan after the allowed number of attempts",
                "attempts": int(state.get("attempts", 0)),
            },
            "log": ["give_up: no valid plan"],
        }

    return give_up


__all__ = [
    "PLAN_TOOL",
    "absorb_plan",
    "make_agent_node",
    "make_apply_node",
    "make_collect_node",
    "make_give_up_node",
    "make_tool_node",
    "make_validate_node",
    "make_verify_node",
]
