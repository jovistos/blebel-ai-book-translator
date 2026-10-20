"""Assembly of the front/back matter LangGraphs.

Two graphs, both shaped like the earlier agent stages::

    book plan (one run per book)
    START -> survey -> book_agent <-> book_tools -> book_validate -> END
                         ^                              |
                         +------------------------------+  (rejected plan)

    chunk localization (one run per matter chunk)
    START -> collect -> chunk_agent <-> chunk_tools -> chunk_validate
                                                              |
                                                              v
                                                        apply -> verify -> END
                                                              ^
                                                              +-- rejected plan

``build_matter_graph`` is the book-level graph; ``build_matter_chunk_graph`` is
the per-chunk graph. The runner drives the second over the matter chunks in
reading order, after the plan from the first is binding.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..core.settings import LLMSettings, resolve_settings
from .llm import make_book_llm, make_chunk_llm
from .nodes import (
    make_apply_node,
    make_book_agent_node,
    make_book_give_up_node,
    make_book_tool_node,
    make_book_validate_node,
    make_chunk_agent_node,
    make_chunk_give_up_node,
    make_chunk_tool_node,
    make_chunk_validate_node,
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
    CHUNK_AGENT,
    CHUNK_GIVE_UP,
    CHUNK_TOOLS,
    CHUNK_VALIDATE,
    COLLECT,
    SURVEY,
    VERIFY,
    book_route_after_agent,
    book_route_after_tools,
    book_route_after_validate,
    chunk_route_after_agent,
    chunk_route_after_apply,
    chunk_route_after_tools,
    chunk_route_after_validate,
)
from .state import MatterChunkState, MatterPlanState


def build_matter_graph(
    settings: LLMSettings | None = None, model: Any | None = None
):
    """Build and compile the book-level localization-plan graph."""
    if model is None:
        settings = settings or resolve_settings()
        model = make_book_llm(settings)

    graph = StateGraph(MatterPlanState)
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
        {"end": END, BOOK_AGENT: BOOK_AGENT, BOOK_GIVE_UP: BOOK_GIVE_UP},
    )
    graph.add_edge(BOOK_GIVE_UP, END)
    return graph.compile()


def build_matter_chunk_graph(
    settings: LLMSettings | None = None, model: Any | None = None
):
    """Build and compile one matter chunk's localization graph."""
    if model is None:
        settings = settings or resolve_settings()
        model = make_chunk_llm(settings)

    graph = StateGraph(MatterChunkState)
    graph.add_node(COLLECT, make_collect_node())
    graph.add_node(CHUNK_AGENT, make_chunk_agent_node(model))
    graph.add_node(CHUNK_TOOLS, make_chunk_tool_node())
    graph.add_node(CHUNK_VALIDATE, make_chunk_validate_node())
    graph.add_node(APPLY, make_apply_node())
    graph.add_node(VERIFY, make_verify_node())
    graph.add_node(CHUNK_GIVE_UP, make_chunk_give_up_node())

    graph.add_edge(START, COLLECT)
    graph.add_edge(COLLECT, CHUNK_AGENT)
    graph.add_conditional_edges(
        CHUNK_AGENT,
        chunk_route_after_agent,
        {
            CHUNK_TOOLS: CHUNK_TOOLS,
            CHUNK_VALIDATE: CHUNK_VALIDATE,
            CHUNK_AGENT: CHUNK_AGENT,
            CHUNK_GIVE_UP: CHUNK_GIVE_UP,
        },
    )
    graph.add_conditional_edges(
        CHUNK_TOOLS,
        chunk_route_after_tools,
        {
            CHUNK_VALIDATE: CHUNK_VALIDATE,
            CHUNK_AGENT: CHUNK_AGENT,
            CHUNK_GIVE_UP: CHUNK_GIVE_UP,
        },
    )
    graph.add_conditional_edges(
        CHUNK_VALIDATE,
        chunk_route_after_validate,
        {
            APPLY: APPLY,
            CHUNK_AGENT: CHUNK_AGENT,
            CHUNK_GIVE_UP: CHUNK_GIVE_UP,
        },
    )
    graph.add_conditional_edges(APPLY, chunk_route_after_apply, {VERIFY: VERIFY})
    graph.add_edge(VERIFY, END)
    graph.add_edge(CHUNK_GIVE_UP, END)
    return graph.compile()


def build_graph(settings: LLMSettings | None = None, model: Any | None = None):
    """Alias for the book-level graph, matching the earlier stages' public name."""
    return build_matter_graph(settings=settings, model=model)


__all__ = [
    "build_graph",
    "build_matter_chunk_graph",
    "build_matter_graph",
]
