"""The agent node: one LLM turn, which either calls tools or answers."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from ..prompts import NUDGE_NO_PROPOSAL
from ..state import SplitterState


def make_agent_node(model: Any):
    """Build the ``agent`` node around an already tool-bound chat model."""

    def agent(state: SplitterState) -> dict[str, Any]:
        messages = list(state["messages"])
        feedback = list(state.get("feedback") or [])
        if feedback:
            messages.append(HumanMessage(content="\n\n".join(feedback)))
        else:
            last = messages[-1] if messages else None
            if isinstance(last, AIMessage) and not last.tool_calls and not state.get("proposed_splits"):
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
