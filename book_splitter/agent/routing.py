"""Conditional edges of the splitting graph.

All routing decisions are pure functions of the graph state, which keeps the
loop logic testable without invoking a model.
"""

from __future__ import annotations

from ..core.constants import MAX_FEEDBACK_LOOPS
from .state import SplitterState

#: Node names, referenced by both the wiring and the routing functions.
SCOUT = "scout"
AGENT = "agent"
TOOLS = "tools"
VALIDATE = "validate"
SLICE = "slice"
VERIFY = "verify"
GIVE_UP = "give_up"


def attempts_exhausted(state: SplitterState) -> bool:
    """True when the model has had its last allowed turn."""
    return int(state.get("attempts", 0)) >= int(state.get("max_attempts", MAX_FEEDBACK_LOOPS))


def route_after_agent(state: SplitterState) -> str:
    """A tool call goes to the executor, a finished answer straight to validation."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return TOOLS
    if state.get("proposed_splits"):
        return VALIDATE
    if attempts_exhausted(state):
        return GIVE_UP
    return AGENT


def route_after_tools(state: SplitterState) -> str:
    """Once a plan has been absorbed, validate it; otherwise keep talking."""
    if state.get("proposed_splits"):
        return VALIDATE
    if attempts_exhausted(state):
        return GIVE_UP
    return AGENT


def route_after_validate(state: SplitterState) -> str:
    """A valid map is sliced; an invalid one loops back with corrections."""
    if (state.get("validation") or {}).get("ok"):
        return SLICE
    if attempts_exhausted(state):
        return GIVE_UP
    return AGENT


def route_after_slice(state: SplitterState) -> str:
    """A dry run stops after the plan; a real run verifies its output."""
    return "end" if state.get("dry_run") else VERIFY
