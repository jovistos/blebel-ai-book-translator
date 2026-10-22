"""Assembly of the LangGraph state machine.

Graph shape::

    START
      |
    scout                 <- deterministic recon (no LLM, no body text)
      |
    agent  <-----------+   <- LLM: picks a strategy, calls tools
      |  \\             |
      |   v            |
      |  tools --------+   <- policy-enforcing tool executor (absorbs the plan)
      |
      v
    validate ---------> agent        (feedback loop on bad maps)
      |
    slice                 <- verbatim copy of the source into chapter files
      |
    verify                <- SHA-256 proof that nothing was lost
      |
     END

The LLM only ever routes: it receives line numbers and short title strings, and
it submits a plan through the ``propose_split_map`` tool. Text is moved by
``slice``, a deterministic node, so no chapter prose can be altered.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..core.settings import LLMSettings, resolve_settings
from .llm import make_llm
from .nodes import (
    make_agent_node,
    make_give_up_node,
    make_scout_node,
    make_slice_node,
    make_tool_node,
    make_validate_node,
    make_verify_node,
)
from .routing import (
    AGENT,
    GIVE_UP,
    SCOUT,
    SLICE,
    TOOLS,
    VALIDATE,
    VERIFY,
    route_after_agent,
    route_after_slice,
    route_after_tools,
    route_after_validate,
)
from .state import SplitterState


def build_graph(settings: LLMSettings | None = None, model: Any | None = None):
    """Build and compile the splitting agent.

    ``model`` allows injecting a test double instead of a real chat model;
    when it is given, ``settings`` is ignored.
    """
    if model is None:
        settings = settings or resolve_settings(stage="split")
        model = make_llm(settings)

    graph = StateGraph(SplitterState)
    graph.add_node(SCOUT, make_scout_node())
    graph.add_node(AGENT, make_agent_node(model))
    graph.add_node(TOOLS, make_tool_node())
    graph.add_node(VALIDATE, make_validate_node())
    graph.add_node(SLICE, make_slice_node())
    graph.add_node(VERIFY, make_verify_node())
    graph.add_node(GIVE_UP, make_give_up_node())

    graph.add_edge(START, SCOUT)
    graph.add_edge(SCOUT, AGENT)
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
        {SLICE: SLICE, AGENT: AGENT, GIVE_UP: GIVE_UP},
    )
    graph.add_conditional_edges(SLICE, route_after_slice, {VERIFY: VERIFY, "end": END})
    graph.add_edge(VERIFY, END)
    graph.add_edge(GIVE_UP, END)
    return graph.compile()


__all__ = ["build_graph"]
