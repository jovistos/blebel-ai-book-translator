"""Assembly of the stage-2 (chapter preprocessing) LangGraph.

Graph shape, exactly parallel to stage 1::

    START
      |
    collect               <- deterministic: headers + evidence + damage (no LLM)
      |
    agent  <-----------+   <- LLM: decides on each header, in a fixed vocabulary
      |  \\             |
      |   v            |
      |  tools --------+   <- executor + write policy for this chapter
      |
      v
    validate ---------> agent        (feedback loop on a rejected plan)
      |
    apply                 <- deterministic write of the cleaned chapter
      |
    verify                <- provenance proof that only declared edits happened
      |
     END

The graph handles exactly **one chapter**. Stage 1's runner drives the chapters
in order and can switch to a thread/process pool without touching this file,
because everything here is a pure function of ``chapter_path`` + ``output_dir``.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..core.settings import LLMSettings, resolve_settings
from .llm import make_preprocess_llm
from .nodes import (
    make_agent_node,
    make_apply_node,
    make_collect_node,
    make_give_up_node,
    make_tool_node,
    make_validate_node,
    make_verify_node,
)
from .routing import (
    AGENT,
    APPLY,
    COLLECT,
    GIVE_UP,
    TOOLS,
    VALIDATE,
    VERIFY,
    route_after_agent,
    route_after_apply,
    route_after_tools,
    route_after_validate,
)
from .state import PreprocessState


def build_preprocess_graph(settings: LLMSettings | None = None, model: Any | None = None):
    """Build and compile the preprocessing agent for one chapter.

    ``model`` allows injecting a test double instead of a real chat model; when
    it is given, ``settings`` is ignored.
    """
    if model is None:
        settings = settings or resolve_settings(stage="preprocess")
        model = make_preprocess_llm(settings)

    graph = StateGraph(PreprocessState)
    graph.add_node(COLLECT, make_collect_node())
    graph.add_node(AGENT, make_agent_node(model))
    graph.add_node(TOOLS, make_tool_node())
    graph.add_node(VALIDATE, make_validate_node())
    graph.add_node(APPLY, make_apply_node())
    graph.add_node(VERIFY, make_verify_node())
    graph.add_node(GIVE_UP, make_give_up_node())

    graph.add_edge(START, COLLECT)
    graph.add_edge(COLLECT, AGENT)
    graph.add_conditional_edges(
        AGENT,
        route_after_agent,
        {TOOLS: TOOLS, VALIDATE: VALIDATE, AGENT: AGENT, GIVE_UP: GIVE_UP},
    )
    graph.add_conditional_edges(
        TOOLS,
        route_after_tools,
        {VALIDATE: VALIDATE, AGENT: AGENT, GIVE_UP: GIVE_UP},
    )
    graph.add_conditional_edges(
        VALIDATE,
        route_after_validate,
        {APPLY: APPLY, AGENT: AGENT, GIVE_UP: GIVE_UP},
    )
    graph.add_conditional_edges(APPLY, route_after_apply, {VERIFY: VERIFY})
    graph.add_edge(VERIFY, END)
    graph.add_edge(GIVE_UP, END)
    return graph.compile()


__all__ = ["build_preprocess_graph"]
