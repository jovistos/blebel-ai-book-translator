"""Conditional edges of the preprocessing graph.

As in stage 1, every routing decision is a pure function of the state, so the
loop is testable without a model.
"""

from __future__ import annotations

from ..core.constants import MAX_FEEDBACK_LOOPS
from .state import PreprocessState

#: Node names.
COLLECT = "collect"
AGENT = "agent"
TOOLS = "tools"
VALIDATE = "validate"
APPLY = "apply"
VERIFY = "verify"
GIVE_UP = "give_up"


def attempts_exhausted(state: PreprocessState) -> bool:
    """True when the model has had its last allowed turn for this chapter."""
    return int(state.get("attempts", 0)) >= int(
        state.get("max_attempts", MAX_FEEDBACK_LOOPS)
    )


def route_after_agent(state: PreprocessState) -> str:
    """Tool calls go to the executor; a finished answer goes to validation."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return TOOLS
    if state.get("operations"):
        return VALIDATE
    if attempts_exhausted(state):
        return GIVE_UP
    return AGENT


def route_after_tools(state: PreprocessState) -> str:
    """Once a plan has been absorbed, validate it; otherwise keep talking."""
    if state.get("operations"):
        return VALIDATE
    if attempts_exhausted(state):
        return GIVE_UP
    return AGENT


def route_after_validate(state: PreprocessState) -> str:
    """A valid plan is applied; an invalid one loops back with corrections."""
    if (state.get("validation") or {}).get("ok"):
        return APPLY
    if attempts_exhausted(state):
        return GIVE_UP
    return AGENT


def route_after_apply(state: PreprocessState) -> str:
    """The apply node is always followed by verification."""
    return VERIFY
