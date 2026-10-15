"""The tools node: the policy-enforcing executor for the model's tool calls.

It does four things per call:

1. refuse write tools until the plan has passed validation,
2. dispatch to the registered tool and turn any exception into a model-facing error,
3. return a ``ToolMessage`` for every call (the protocol requires one each),
4. absorb a successful ``propose_split_map`` into the graph state.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage

from ...documents.text import jdump, short_args, summarise_payload
from ..state import SplitterState
from ..tools import TOOL_BY_NAME, blocked_payload, is_blocked


def make_tool_node():
    """Build the ``tools`` node."""

    def tools_node(state: SplitterState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        validation = state.get("validation") or {}
        out_messages: list[ToolMessage] = []
        updates: dict[str, Any] = {}
        log: list[str] = []

        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{len(out_messages)}"

            if is_blocked(name, validation):
                out_messages.append(
                    ToolMessage(
                        content=jdump(blocked_payload(name)),
                        tool_call_id=call_id,
                        name=name,
                        status="error",
                    )
                )
                log.append(f"tools: BLOCKED {name} (no validated plan yet)")
                continue

            tool = TOOL_BY_NAME.get(name)
            if tool is None:
                out_messages.append(
                    ToolMessage(
                        content=jdump(
                            {"error": f"unknown tool {name!r}", "available": list(TOOL_BY_NAME)}
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

            if name == "propose_split_map" and status == "success":
                updates.update(absorb_proposal(payload))

        return {**updates, "messages": out_messages, "log": log}

    return tools_node


def absorb_proposal(payload: str) -> dict[str, Any]:
    """Read an accepted ``propose_split_map`` result into state updates."""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict) or not data.get("accepted"):
        return {}
    return {
        "proposed_splits": data["split_mapping"],
        "proposal_strategy": data.get("strategy", "unknown"),
        "proposal_notes": data.get("notes", ""),
    }
