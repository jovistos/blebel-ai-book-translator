"""Conditional edges for the stage-9 graphs.

Two small graphs share the same loop shape as stages 1 and 2:

``book``    survey -> agent -> tools -> validate -> end
``chapter`` collect -> agent -> tools -> validate -> apply -> verify -> end

Every routing decision is a pure function of the state, so the loops can be tested
without a model.
"""

from __future__ import annotations

from ..core.constants import DEFAULT_POSTPROCESS_ATTEMPTS
from .state import BookPlanState, ChapterPostprocessState

#: Book graph nodes.
SURVEY = "survey"
BOOK_AGENT = "book_agent"
BOOK_TOOLS = "book_tools"
BOOK_VALIDATE = "book_validate"
BOOK_GIVE_UP = "book_give_up"

#: Chapter graph nodes.
COLLECT = "collect"
CHAPTER_AGENT = "chapter_agent"
CHAPTER_TOOLS = "chapter_tools"
CHAPTER_VALIDATE = "chapter_validate"
APPLY = "apply"
VERIFY = "verify"
CHAPTER_GIVE_UP = "chapter_give_up"


def attempts_exhausted(state: dict) -> bool:
    return int(state.get("attempts", 0)) >= int(
        state.get("max_attempts", DEFAULT_POSTPROCESS_ATTEMPTS)
    )


def book_route_after_agent(state: BookPlanState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return BOOK_TOOLS
    if state.get("book_plan"):
        return BOOK_VALIDATE
    if attempts_exhausted(state):
        return BOOK_GIVE_UP
    return BOOK_AGENT


def book_route_after_tools(state: BookPlanState) -> str:
    if state.get("book_plan"):
        return BOOK_VALIDATE
    if attempts_exhausted(state):
        return BOOK_GIVE_UP
    return BOOK_AGENT


def book_route_after_validate(state: BookPlanState) -> str:
    if (state.get("validation") or {}).get("ok"):
        return "end"
    if attempts_exhausted(state):
        return BOOK_GIVE_UP
    return BOOK_AGENT


def chapter_route_after_agent(state: ChapterPostprocessState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return CHAPTER_TOOLS
    if state.get("chapter_plan"):
        return CHAPTER_VALIDATE
    if attempts_exhausted(state):
        return CHAPTER_GIVE_UP
    return CHAPTER_AGENT


def chapter_route_after_tools(state: ChapterPostprocessState) -> str:
    if state.get("chapter_plan"):
        return CHAPTER_VALIDATE
    if attempts_exhausted(state):
        return CHAPTER_GIVE_UP
    return CHAPTER_AGENT


def chapter_route_after_validate(state: ChapterPostprocessState) -> str:
    if (state.get("validation") or {}).get("ok"):
        return APPLY
    if attempts_exhausted(state):
        return CHAPTER_GIVE_UP
    return CHAPTER_AGENT


def chapter_route_after_apply(state: ChapterPostprocessState) -> str:
    return VERIFY


__all__ = [
    "APPLY",
    "BOOK_AGENT",
    "BOOK_GIVE_UP",
    "BOOK_TOOLS",
    "BOOK_VALIDATE",
    "CHAPTER_AGENT",
    "CHAPTER_GIVE_UP",
    "CHAPTER_TOOLS",
    "CHAPTER_VALIDATE",
    "COLLECT",
    "SURVEY",
    "VERIFY",
    "attempts_exhausted",
    "book_route_after_agent",
    "book_route_after_tools",
    "book_route_after_validate",
    "chapter_route_after_agent",
    "chapter_route_after_apply",
    "chapter_route_after_tools",
    "chapter_route_after_validate",
]
