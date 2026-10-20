"""Conditional edges for the front/back matter graphs.

Two small graphs with the same loop shape as stages 2 and 9:

``book``   survey -> agent -> tools -> validate -> end
``chunk``  collect -> agent -> tools -> validate -> apply -> verify -> end

Every routing decision is a pure function of the state, so the loops can be
tested without a model.
"""

from __future__ import annotations

from ..core.constants import DEFAULT_MATTER_ATTEMPTS
from .state import MatterChunkState, MatterPlanState

#: Book graph nodes.
SURVEY = "survey"
BOOK_AGENT = "book_agent"
BOOK_TOOLS = "book_tools"
BOOK_VALIDATE = "book_validate"
BOOK_GIVE_UP = "book_give_up"

#: Chunk graph nodes.
COLLECT = "collect"
CHUNK_AGENT = "chunk_agent"
CHUNK_TOOLS = "chunk_tools"
CHUNK_VALIDATE = "chunk_validate"
APPLY = "apply"
VERIFY = "verify"
CHUNK_GIVE_UP = "chunk_give_up"


def attempts_exhausted(state: dict) -> bool:
    return int(state.get("attempts", 0)) >= int(
        state.get("max_attempts", DEFAULT_MATTER_ATTEMPTS)
    )


def book_route_after_agent(state: MatterPlanState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return BOOK_TOOLS
    if state.get("matter_plan"):
        return BOOK_VALIDATE
    if attempts_exhausted(state):
        return BOOK_GIVE_UP
    return BOOK_AGENT


def book_route_after_tools(state: MatterPlanState) -> str:
    if state.get("matter_plan"):
        return BOOK_VALIDATE
    if attempts_exhausted(state):
        return BOOK_GIVE_UP
    return BOOK_AGENT


def book_route_after_validate(state: MatterPlanState) -> str:
    if (state.get("validation") or {}).get("ok"):
        return "end"
    if attempts_exhausted(state):
        return BOOK_GIVE_UP
    return BOOK_AGENT


def chunk_route_after_agent(state: MatterChunkState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return CHUNK_TOOLS
    if state.get("chunk_plan"):
        return CHUNK_VALIDATE
    if attempts_exhausted(state):
        return CHUNK_GIVE_UP
    return CHUNK_AGENT


def chunk_route_after_tools(state: MatterChunkState) -> str:
    if state.get("chunk_plan"):
        return CHUNK_VALIDATE
    if attempts_exhausted(state):
        return CHUNK_GIVE_UP
    return CHUNK_AGENT


def chunk_route_after_validate(state: MatterChunkState) -> str:
    if (state.get("validation") or {}).get("ok"):
        return APPLY
    if attempts_exhausted(state):
        return CHUNK_GIVE_UP
    return CHUNK_AGENT


def chunk_route_after_apply(state: MatterChunkState) -> str:
    return VERIFY


__all__ = [
    "APPLY",
    "BOOK_AGENT",
    "BOOK_GIVE_UP",
    "BOOK_TOOLS",
    "BOOK_VALIDATE",
    "CHUNK_AGENT",
    "CHUNK_GIVE_UP",
    "CHUNK_TOOLS",
    "CHUNK_VALIDATE",
    "COLLECT",
    "SURVEY",
    "VERIFY",
    "attempts_exhausted",
    "book_route_after_agent",
    "book_route_after_tools",
    "book_route_after_validate",
    "chunk_route_after_agent",
    "chunk_route_after_apply",
    "chunk_route_after_tools",
    "chunk_route_after_validate",
]
