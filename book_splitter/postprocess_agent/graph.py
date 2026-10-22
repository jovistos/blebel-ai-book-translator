"""Assembly of the stage-9 LangGraphs.

Two graphs, both the same shape as the earlier stages::

    book graph (one run per book)
    START -> survey -> book_agent <-> book_tools -> book_validate -> END
                         ^                              |
                         +------------------------------+  (rejected plan)

    chapter graph (one run per chapter)
    START -> collect -> chapter_agent <-> chapter_tools -> chapter_validate
                                                                  |
                                                                  v
                                                                apply -> verify -> END
                                                                  ^
                                                                  +-- rejected plan

``build_postprocess_graph`` is the book-level graph (the naming policy and the
table-of-contents decision). ``build_chapter_postprocess_graph`` is the
per-chapter graph (title placement and heading keep/demote/promote decisions). The runner
drives the second over the chapters in reading order, exactly as stage 2's runner
drives its per-chapter graph.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..core.settings import LLMSettings, resolve_settings
from .llm import make_book_llm, make_chapter_llm
from .nodes import (
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
from .routing import (
    APPLY,
    BOOK_AGENT,
    BOOK_GIVE_UP,
    BOOK_TOOLS,
    BOOK_VALIDATE,
    CHAPTER_AGENT,
    CHAPTER_GIVE_UP,
    CHAPTER_TOOLS,
    CHAPTER_VALIDATE,
    COLLECT,
    SURVEY,
    VERIFY,
    book_route_after_agent,
    book_route_after_tools,
    book_route_after_validate,
    chapter_route_after_agent,
    chapter_route_after_apply,
    chapter_route_after_tools,
    chapter_route_after_validate,
)
from .state import BookPlanState, ChapterPostprocessState


def build_postprocess_graph(
    settings: LLMSettings | None = None, model: Any | None = None
):
    """Build and compile the book-level naming/TOC graph."""
    if model is None:
        settings = settings or resolve_settings(stage="postprocess")
        model = make_book_llm(settings)

    graph = StateGraph(BookPlanState)
    graph.add_node(SURVEY, make_survey_node())
    graph.add_node(BOOK_AGENT, make_book_agent_node(model))
    graph.add_node(BOOK_TOOLS, make_book_tool_node())
    graph.add_node(BOOK_VALIDATE, make_book_validate_node())
    graph.add_node(BOOK_GIVE_UP, make_book_give_up_node())

    graph.add_edge(START, SURVEY)
    graph.add_edge(SURVEY, BOOK_AGENT)
    graph.add_conditional_edges(
        BOOK_AGENT,
        book_route_after_agent,
        {
            BOOK_TOOLS: BOOK_TOOLS,
            BOOK_VALIDATE: BOOK_VALIDATE,
            BOOK_AGENT: BOOK_AGENT,
            BOOK_GIVE_UP: BOOK_GIVE_UP,
        },
    )
    graph.add_conditional_edges(
        BOOK_TOOLS,
        book_route_after_tools,
        {
            BOOK_VALIDATE: BOOK_VALIDATE,
            BOOK_AGENT: BOOK_AGENT,
            BOOK_GIVE_UP: BOOK_GIVE_UP,
        },
    )
    graph.add_conditional_edges(
        BOOK_VALIDATE,
        book_route_after_validate,
        {
            "end": END,
            BOOK_AGENT: BOOK_AGENT,
            BOOK_GIVE_UP: BOOK_GIVE_UP,
        },
    )
    graph.add_edge(BOOK_GIVE_UP, END)
    return graph.compile()


def build_chapter_postprocess_graph(
    settings: LLMSettings | None = None, model: Any | None = None
):
    """Build and compile the per-chapter title/heading graph."""
    if model is None:
        settings = settings or resolve_settings(stage="postprocess")
        model = make_chapter_llm(settings)

    graph = StateGraph(ChapterPostprocessState)
    graph.add_node(COLLECT, make_collect_node())
    graph.add_node(CHAPTER_AGENT, make_chapter_agent_node(model))
    graph.add_node(CHAPTER_TOOLS, make_chapter_tool_node())
    graph.add_node(CHAPTER_VALIDATE, make_chapter_validate_node())
    graph.add_node(APPLY, make_apply_node())
    graph.add_node(VERIFY, make_verify_node())
    graph.add_node(CHAPTER_GIVE_UP, make_chapter_give_up_node())

    graph.add_edge(START, COLLECT)
    graph.add_edge(COLLECT, CHAPTER_AGENT)
    graph.add_conditional_edges(
        CHAPTER_AGENT,
        chapter_route_after_agent,
        {
            CHAPTER_TOOLS: CHAPTER_TOOLS,
            CHAPTER_VALIDATE: CHAPTER_VALIDATE,
            CHAPTER_AGENT: CHAPTER_AGENT,
            CHAPTER_GIVE_UP: CHAPTER_GIVE_UP,
        },
    )
    graph.add_conditional_edges(
        CHAPTER_TOOLS,
        chapter_route_after_tools,
        {
            CHAPTER_VALIDATE: CHAPTER_VALIDATE,
            CHAPTER_AGENT: CHAPTER_AGENT,
            CHAPTER_GIVE_UP: CHAPTER_GIVE_UP,
        },
    )
    graph.add_conditional_edges(
        CHAPTER_VALIDATE,
        chapter_route_after_validate,
        {
            APPLY: APPLY,
            CHAPTER_AGENT: CHAPTER_AGENT,
            CHAPTER_GIVE_UP: CHAPTER_GIVE_UP,
        },
    )
    graph.add_conditional_edges(APPLY, chapter_route_after_apply, {VERIFY: VERIFY})
    graph.add_edge(VERIFY, END)
    graph.add_edge(CHAPTER_GIVE_UP, END)
    return graph.compile()


def build_graph(settings: LLMSettings | None = None, model: Any | None = None):
    """Alias for the book-level graph, matching the earlier stages' public name."""
    return build_postprocess_graph(settings=settings, model=model)


__all__ = [
    "build_chapter_postprocess_graph",
    "build_graph",
    "build_postprocess_graph",
]
