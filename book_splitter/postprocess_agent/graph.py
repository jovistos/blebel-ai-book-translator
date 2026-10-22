"""Stage 9 graphs, declared as specs and compiled by the shared scaffold.

Two graphs, the same shape as every other stage:

    book    : survey -> agent <-> tools -> validate -> END
    chapter : collect -> agent <-> tools -> validate -> apply -> verify -> END

``build_postprocess_graph`` is the book-level graph (the naming policy and the
table-of-contents decision). ``build_chapter_postprocess_graph`` is the
per-chapter graph (title placement and heading keep/demote/promote decisions).
The runner drives the second over the chapters in reading order.

Only the specs below differ from the other stages; the wiring is in
:mod:`book_splitter.stage_graph`.
"""

from __future__ import annotations

from typing import Any

from ..agent.tools.registry import POSTPROCESS_WRITE_TOOLS
from ..core.constants import DEFAULT_POSTPROCESS_ATTEMPTS
from ..core.settings import LLMSettings, resolve_settings
from ..stage_graph import StageSpec, TailStep, build_stage_graph
from .llm import make_book_llm, make_chapter_llm
from .nodes import (
    BOOK_PLAN_TOOL,
    BOOK_TOOL_BY_NAME,
    CHAPTER_PLAN_TOOL,
    CHAPTER_TOOL_BY_NAME,
    make_apply_node,
    make_book_agent_node,
    make_book_give_up_node,
    make_book_tool_node,
    make_book_validate_node,
    make_chapter_agent_node,
    make_chapter_give_up_node,
    make_chapter_tool_node,
    make_chapter_validate_node,
    make_collect_node,
    make_survey_node,
    make_verify_node,
)
from .prompts import (
    NUDGE_BOOK_FINAL_TURN,
    NUDGE_BOOK_NO_PLAN,
    NUDGE_CHAPTER_FINAL_TURN,
    NUDGE_CHAPTER_NO_PLAN,
)
from .state import BookPlanState, ChapterPostprocessState

#: One run per book: name the chapters and decide the table of contents.
BOOK = StageSpec(
    state=BookPlanState,
    recon=("survey", make_survey_node),
    agent=make_book_agent_node,
    tools=make_book_tool_node,
    validate=make_book_validate_node,
    give_up=make_book_give_up_node,
    plan_key="book_plan",
    attempts_default=DEFAULT_POSTPROCESS_ATTEMPTS,
    tool_table=BOOK_TOOL_BY_NAME,
    write_tools=POSTPROCESS_WRITE_TOOLS,
    plan_tool=BOOK_PLAN_TOOL,
    nudge_no_plan=NUDGE_BOOK_NO_PLAN,
    nudge_final_turn=NUDGE_BOOK_FINAL_TURN,
    final_turn_policy="unvalidated",
)

#: One run per chapter: place the title, then write and verify the result.
CHAPTER = StageSpec(
    state=ChapterPostprocessState,
    recon=("collect", make_collect_node),
    agent=make_chapter_agent_node,
    tools=make_chapter_tool_node,
    validate=make_chapter_validate_node,
    give_up=make_chapter_give_up_node,
    plan_key="chapter_plan",
    attempts_default=DEFAULT_POSTPROCESS_ATTEMPTS,
    tool_table=CHAPTER_TOOL_BY_NAME,
    write_tools=POSTPROCESS_WRITE_TOOLS,
    plan_tool=CHAPTER_PLAN_TOOL,
    nudge_no_plan=NUDGE_CHAPTER_NO_PLAN,
    nudge_final_turn=NUDGE_CHAPTER_FINAL_TURN,
    final_turn_policy="unvalidated",
    tail=(TailStep("apply", make_apply_node), TailStep("verify", make_verify_node)),
)


def build_postprocess_graph(
    settings: LLMSettings | None = None, model: Any | None = None
):
    """Build and compile the book-level naming/TOC graph."""
    if model is None:
        settings = settings or resolve_settings(stage="postprocess")
        model = make_book_llm(settings)
    return build_stage_graph(BOOK, model)


def build_chapter_postprocess_graph(
    settings: LLMSettings | None = None, model: Any | None = None
):
    """Build and compile the per-chapter title/heading graph."""
    if model is None:
        settings = settings or resolve_settings(stage="postprocess")
        model = make_chapter_llm(settings)
    return build_stage_graph(CHAPTER, model)


def build_graph(settings: LLMSettings | None = None, model: Any | None = None):
    """Alias for the book-level graph, matching the earlier stages' public name."""
    return build_postprocess_graph(settings=settings, model=model)


__all__ = [
    "build_chapter_postprocess_graph",
    "build_graph",
    "build_postprocess_graph",
]
