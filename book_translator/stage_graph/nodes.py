"""The node bodies every stage shares.

The scaffold in :mod:`book_translator.stage_graph` wires these five once:

``make_agent_node``
    one LLM turn, with the stage's nudges and its last-turn policy.
``make_tool_node``
    the policy-enforcing executor: the write gate, the plan-tool guard and the
    absorption of an accepted plan.
``make_give_up_node``
    the result a run ends with when no valid plan was ever submitted.
``absorb_plan``
    parse an accepted plan payload into state updates.
``usage``
    the token bookkeeping every agent turn contributes.

Everything stage-specific is read from the :class:`~book_translator.stage_graph.StageSpec`
passed to a factory, so a stage cannot get a different loop by accident.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..documents.text import jdump, short_args, summarise_payload
from ..toolkit.context import ToolContext, wants_context

if TYPE_CHECKING:  # the package imports this module while defining StageSpec
    from . import StageSpec


def usage(response: Any) -> dict[str, int]:
    """The token counts one model response contributes to ``state["usage"]``."""
    usage_metadata = dict(getattr(response, "usage_metadata", None) or {})
    return {
        "input_tokens": int(usage_metadata.get("input_tokens", 0)),
        "output_tokens": int(usage_metadata.get("output_tokens", 0)),
        "total_tokens": int(usage_metadata.get("total_tokens", 0)),
        "llm_calls": 1,
    }


def make_agent_node(*, spec: StageSpec):
    """Build the ``agent`` factory for ``spec``: one LLM turn per call."""

    def factory(model: Any):
        def agent(state: dict[str, Any]) -> dict[str, Any]:
            messages = list(state["messages"])
            feedback = list(state.get("feedback") or [])
            turn_notes: list[str] = []
            if feedback:
                turn_notes.append("\n\n".join(feedback))
            else:
                last = messages[-1] if messages else None
                if (
                    isinstance(last, AIMessage)
                    and not last.tool_calls
                    and not state.get(spec.plan_key)
                    and spec.nudge_no_plan
                ):
                    turn_notes.append(spec.nudge_no_plan)

            attempts = int(state.get("attempts", 0)) + 1
            limit = int(state.get("max_attempts", spec.attempts_default))
            if (
                spec.nudge_final_turn
                and attempts >= limit
                and _final_turn_applies(state, spec)
            ):
                turn_notes.append(spec.nudge_final_turn)
            if turn_notes:
                messages.append(HumanMessage(content="\n\n".join(turn_notes)))

            response = model.invoke(messages)
            calls = [
                call["name"] for call in (getattr(response, "tool_calls", None) or [])
            ]
            return {
                "messages": [response],
                "attempts": attempts,
                "feedback": [],
                "usage": usage(response),
                "log": [
                    f"{spec.log_prefix}[{attempts}]: tool_calls={calls or 'none'}"
                ],
            }

        return agent

    return factory


def _final_turn_applies(state: dict[str, Any], spec: StageSpec) -> bool:
    """Whether the last-turn reminder describes this state."""
    if spec.final_turn_policy == "no_plan":
        return not state.get(spec.plan_key)
    if spec.final_turn_policy == "unvalidated":
        return not (state.get("validation") or {}).get("ok")
    return False


def make_tool_node(*, spec: StageSpec):
    """Build the ``tools`` factory for ``spec``: the only place a tool runs.

    The guards run in this order, which is the order the stages relied on before
    they shared a body:

    1. a write tool before validation (or always, for a book planner),
    2. a duplicate plan submission once validation has passed,
    3. an unknown tool,
    4. the plan tool before a required call (the published-contents search),
    5. the call itself, then the optional evidence hook and plan absorption.
    """

    def factory():
        def tools_node(state: dict[str, Any]) -> dict[str, Any]:
            last = state["messages"][-1]
            calls = list(getattr(last, "tool_calls", None) or [])
            plan_ok = bool((state.get("validation") or {}).get("ok"))
            out_messages: list[ToolMessage] = []
            updates: dict[str, Any] = {}
            log: list[str] = []
            # The run, not the model, decides which folder/book/chunk a tool
            # resolves against; the tools that need it declare an injected
            # ``ctx`` argument.
            ctx = ToolContext.from_state(state)

            for call in calls:
                name = call.get("name", "")
                args = dict(call.get("args") or {})
                call_id = call.get("id") or f"call_{len(out_messages)}"

                if name in spec.write_tools and (
                    spec.writes_always_blocked or not plan_ok
                ):
                    out_messages.append(
                        _refusal(
                            call_id,
                            name,
                            spec.write_block_message.format(name=name),
                            spec.write_block_hint,
                        )
                    )
                    log.append(f"{spec.tool_log_prefix}: BLOCKED {name}")
                    continue

                if spec.refuse_duplicate_plan and plan_ok and name == spec.plan_tool:
                    out_messages.append(
                        _refusal(
                            call_id,
                            name,
                            spec.duplicate_plan_message,
                            plan_refusal=True,
                        )
                    )
                    log.append(
                        f"{spec.tool_log_prefix}: {name} ignored (plan already accepted)"
                    )
                    continue

                tool = spec.tool_table.get(name)
                if tool is None:
                    out_messages.append(
                        ToolMessage(
                            content=jdump(
                                {
                                    "error": _unknown_message(spec, name),
                                    "available": list(spec.tool_table),
                                }
                            ),
                            tool_call_id=call_id,
                            name=name or "unknown",
                            status="error",
                        )
                    )
                    log.append(f"{spec.tool_log_prefix}: unknown tool {name}")
                    continue

                if name == spec.plan_tool and _requires_missing(state, updates, spec):
                    out_messages.append(
                        _refusal(
                            call_id,
                            name,
                            spec.required_message,
                            spec.required_hint,
                            plan_refusal=True,
                        )
                    )
                    log.append(
                        f"{spec.tool_log_prefix}: BLOCKED {name} "
                        f"({'/'.join(spec.required_before_plan)} required)"
                    )
                    continue

                call_args = dict(args)
                if wants_context(tool):
                    call_args["ctx"] = ctx
                try:
                    payload = tool.invoke(call_args)
                    status = "success"
                except Exception as exc:  # bad args, missing file, ...
                    payload = jdump(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "hint": "check the argument names and types, then retry",
                        }
                    )
                    status = "error"

                if spec.after_tool is not None:
                    updates.update(spec.after_tool(state, name, args, payload))

                out_messages.append(
                    ToolMessage(
                        content=payload,
                        tool_call_id=call_id,
                        name=name,
                        status=status,
                    )
                )
                log.append(
                    f"{spec.tool_log_prefix}: "
                    f"{name}({short_args(args)}) -> {summarise_payload(payload)}"
                )

                if name == spec.plan_tool and status == "success":
                    updates.update(
                        absorb_plan(
                            payload,
                            plan_key=spec.plan_key,
                            extract=spec.plan_extract,
                        )
                    )

            return {**updates, "messages": out_messages, "log": log}

        return tools_node

    return factory


def _refusal(
    call_id: str,
    name: str,
    message: str,
    hint: str = "",
    *,
    plan_refusal: bool = False,
) -> ToolMessage:
    """A refusal ToolMessage: the model sees an error it can act on.

    A plan refusal carries ``accepted: False`` because the model's next turn
    treats a plan-tool result as a proposal outcome, not as a write attempt.
    """
    body: dict[str, Any] = {"error": message}
    if plan_refusal:
        body = {"accepted": False, "error": message}
    if hint:
        body["hint"] = hint
    return ToolMessage(
        content=jdump(body),
        tool_call_id=call_id,
        name=name,
        status="error",
    )


def _unknown_message(spec: StageSpec, name: str) -> str:
    label = f" {spec.unknown_tool_label}" if spec.unknown_tool_label else ""
    return f"unknown{label} tool {name!r}"


def _requires_missing(
    state: dict[str, Any], updates: dict[str, Any], spec: StageSpec
) -> bool:
    """True when the plan tool was called before its required evidence."""
    if not spec.required_before_plan:
        return False
    if spec.required_toggle and not state.get(spec.required_toggle, True):
        return False
    flag = spec.required_flag
    return not (flag and (updates.get(flag) or state.get(flag)))


def make_give_up_node(*, spec: StageSpec):
    """Build the ``give_up`` factory for ``spec``: stop without writing."""

    def factory():
        def give_up(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "result": {
                    "error": spec.give_up_error,
                    "attempts": int(state.get("attempts", 0)),
                },
                "log": [spec.give_up_log],
            }

        return give_up

    return factory


def absorb_plan(
    payload: str,
    *,
    plan_key: str,
    extract: Any = None,
) -> dict[str, Any]:
    """Read an accepted plan result into state updates.

    ``extract`` maps the parsed payload to the keys this stage keeps. Without
    one, the common ``{"plan": ...}`` shape is stored under ``plan_key``.
    """
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict) or not data.get("accepted"):
        return {}
    if extract is not None:
        updates = extract(data)
    else:
        plan = data.get("plan")
        if not isinstance(plan, dict):
            return {}
        updates = {plan_key: plan}
    return {key: value for key, value in updates.items() if value is not None}


__all__ = [
    "absorb_plan",
    "make_agent_node",
    "make_give_up_node",
    "make_tool_node",
    "usage",
]
