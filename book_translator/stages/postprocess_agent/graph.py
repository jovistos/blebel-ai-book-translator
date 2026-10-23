"""Stage 9 graphs, declared as specs and compiled by the shared scaffold.

Two graphs, the same shape as every other stage:

    book    : survey -> agent <-> tools -> validate -> END
    chapter : collect -> agent <-> tools -> validate -> apply -> verify -> END

``build_postprocess_graph`` is the book-level graph (the naming policy and the
table-of-contents decision). ``build_chapter_postprocess_graph`` is the
per-chapter graph (title placement and heading keep/demote/promote decisions).
The runner drives the second over the chapters in reading order.

Only the specs below differ from the other stages; the wiring is in
:mod:`book_translator.stage_graph`.
"""

from __future__ import annotations

from typing import Any

from book_translator.core.constants import DEFAULT_POSTPROCESS_ATTEMPTS
from book_translator.core.settings import LLMSettings, resolve_settings
from book_translator.stage_graph import StageSpec, TailStep, build_stage_graph
from book_translator.stages.agent.tools.registry import POSTPROCESS_WRITE_TOOLS

from .llm import make_book_llm, make_chapter_llm
from .nodes import (
    make_apply_node,
    make_book_validate_node,
    make_chapter_validate_node,
    make_collect_node,
    make_survey_node,
    make_verify_node,
)
from .policy import (
    BOOK_PLAN_TOOL,
    BOOK_TOOL_BY_NAME,
    CHAPTER_PLAN_TOOL,
    CHAPTER_TOOL_BY_NAME,
    extract_book_plan,
    extract_chapter_plan,
    record_web_call,
)
from .prompts import (
    NUDGE_BOOK_FINAL_TURN,
    NUDGE_BOOK_NO_PLAN,
    NUDGE_CHAPTER_FINAL_TURN,
    NUDGE_CHAPTER_NO_PLAN,
)
from .state import BookPlanState, ChapterPostprocessState

#: The hint the book planner gets when it submits before searching.
REQUIRED_SEARCH_HINT = (
    "Call web_search for the book's published chapter list first, then call "
    "report_book_plan again. A failed search still counts once you record its "
    "outcome in title_evidence."
)

#: One run per book: name the chapters and decide the table of contents.
BOOK = StageSpec(
    state=BookPlanState,
    recon=("survey", make_survey_node),
    validate=make_book_validate_node,
    plan_key="book_plan",
    attempts_default=DEFAULT_POSTPROCESS_ATTEMPTS,
    tool_table=BOOK_TOOL_BY_NAME,
    write_tools=POSTPROCESS_WRITE_TOOLS,
    plan_tool=BOOK_PLAN_TOOL,
    plan_extract=extract_book_plan,
    write_block_message="{name} is blocked at the book-plan stage",
    write_block_hint="submit the book plan; the graph applies it",
    writes_always_blocked=True,
    required_before_plan=("web_search",),
    required_toggle="require_web_search",
    required_flag="web_search_attempted",
    required_message="web search is mandatory before the book plan",
    required_hint=REQUIRED_SEARCH_HINT,
    after_tool=record_web_call,
    nudge_no_plan=NUDGE_BOOK_NO_PLAN,
    nudge_final_turn=NUDGE_BOOK_FINAL_TURN,
    final_turn_policy="unvalidated",
    log_prefix="book_agent",
    tool_log_prefix="book_tools",
    unknown_tool_label="book-plan",
    give_up_error="no valid book plan after the allowed number of attempts",
    give_up_log="book_give_up: no valid plan",
)

#: One run per chapter: place the title, then write and verify the result.
CHAPTER = StageSpec(
    state=ChapterPostprocessState,
    recon=("collect", make_collect_node),
    validate=make_chapter_validate_node,
    plan_key="chapter_plan",
    attempts_default=DEFAULT_POSTPROCESS_ATTEMPTS,
    tool_table=CHAPTER_TOOL_BY_NAME,
    write_tools=POSTPROCESS_WRITE_TOOLS,
    plan_tool=CHAPTER_PLAN_TOOL,
    plan_extract=extract_chapter_plan,
    write_block_message="{name} is blocked until this chapter's plan is validated",
    write_block_hint="call report_chapter_postprocess, then the graph applies it",
    refuse_duplicate_plan=True,
    duplicate_plan_message="a plan for this chapter has already been accepted",
    nudge_no_plan=NUDGE_CHAPTER_NO_PLAN,
    nudge_final_turn=NUDGE_CHAPTER_FINAL_TURN,
    final_turn_policy="unvalidated",
    log_prefix="chapter_agent",
    tool_log_prefix="chapter_tools",
    unknown_tool_label="chapter",
    give_up_error="no valid chapter plan after the allowed number of attempts",
    give_up_log="chapter_give_up: no valid plan",
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
