"""The shared scaffold every agent stage is compiled from.

All six agent graphs run the same loop::

    START -> recon -> agent <-> tools -> validate -> [tail...] -> END
                        |                   |
                        +--- rejected ------+--> give_up -> END

``recon`` is deterministic inventory (no LLM), ``agent`` is one LLM turn that
either calls tools or submits a plan, ``tools`` is the policy-enforcing
executor, ``validate`` scores the submitted plan deterministically, and the
optional ``tail`` holds the deterministic nodes that act on a validated plan
(write it, then prove the write).

Only the node bodies and a handful of scalars differ between stages, so the
wiring lives here once and each stage declares a :class:`StageSpec` in its own
``graph.py``. Two properties follow from that:

* **The write gate is structural.** Tail nodes are reachable only from
  ``validate`` on ``ok``, so no stage can be wired to write before its plan has
  passed validation.
* **The loop is tested once.** The routing functions below are pure functions of
  the state and the spec, so one parametrised suite covers every stage.

Nothing here imports a stage, which keeps this module a leaf: stages depend on
it, it depends on nobody.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

#: Node names shared by every stage. Each stage compiles its own graph, so the
#: names never collide between stages.
AGENT = "agent"
TOOLS = "tools"
VALIDATE = "validate"
GIVE_UP = "give_up"

#: The routable key that stands for ``END`` in an edge table.
END_KEY = "end"

# Node factories are typed loosely on purpose. LangGraph binds a node function's
# state type per ``add_node`` call, so a factory that returned a single concrete
# signature could not be stored in a spec shared by stages with different state
# classes. The nodes themselves keep precise signatures where they are defined
# (``def validate(state: SplitterState) -> dict[str, Any]``); this is only the
# erasure at the one point where heterogeneity is unavoidable.
NodeFactory = Callable[[], Any]
ModelNodeFactory = Callable[[Any], Any]
#: A routing decision: a pure function of the state, returning a node name.
Router = Callable[[dict[str, Any]], str]


def merge_usage(left: dict[str, int] | None, right: dict[str, int] | None) -> dict[str, int]:
    """Reducer that sums token usage across graph nodes."""
    out = dict(left or {})
    for key, value in (right or {}).items():
        out[key] = out.get(key, 0) + value
    return out


class AgentStateBase(TypedDict, total=False):
    """The state keys every stage shares, reducers included.

    A stage's own state inherits this and adds only the keys its payload needs::

        class SplitterState(AgentStateBase, total=False):
            proposed_splits: list[dict[str, Any]]

    ``total=False`` on both levels keeps every key optional, which is what lets a
    runner start a stage from a partial ``initial`` dict.
    """

    # --- control ---
    max_attempts: int
    dry_run: bool

    # --- validation handshake: ``validate`` writes, ``agent`` consumes ---
    validation: dict[str, Any]
    feedback: list[str]

    # --- bookkeeping (reducers accumulate across nodes) ---
    attempts: int
    usage: Annotated[dict[str, int], merge_usage]
    log: Annotated[list[str], operator.add]
    messages: Annotated[list, add_messages]
    result: dict[str, Any]


@dataclass(frozen=True)
class TailStep:
    """One deterministic node that runs after a plan has passed validation.

    ``router`` is the exception, not the rule: only the splitter needs one, to
    stop before ``verify`` when the run is a dry run (its ``verify`` re-reads the
    files from disk, which a dry run never wrote). Every other stage gets a plain
    edge to the next step, or to ``END``.
    """

    name: str
    factory: NodeFactory
    router: Router | None = None


@dataclass(frozen=True)
class StageSpec:
    """Everything that differs between the stages, and nothing that doesn't."""

    #: The stage's state class, a TypedDict inheriting :class:`AgentStateBase`.
    state: type
    #: ``(name, factory)`` for the deterministic recon node (scout/survey/collect).
    recon: tuple[str, NodeFactory]
    #: The agent node, which needs the chat model.
    agent: ModelNodeFactory
    #: The policy-enforcing tool executor.
    tools: NodeFactory
    #: Deterministic scoring of the submitted plan.
    validate: NodeFactory
    #: How a run ends without a result.
    give_up: NodeFactory
    #: The state key whose presence means "a plan was submitted".
    plan_key: str
    #: ``max_attempts`` fallback when the runner did not set one.
    attempts_default: int
    #: Name -> tool, the dispatch table the tool executor may call. It is the
    #: stage's *whole* toolset (its own plus the shared tools), because the model
    #: was bound from the same table.
    tool_table: Mapping[str, Any] = field(default_factory=dict)
    #: Tools that must not run before the plan has passed validation. Enforced
    #: by the tool executor, not by the prompt.
    write_tools: frozenset[str] = frozenset()
    #: The tool whose accepted result *is* the plan. The executor absorbs such a
    #: result into the state and never lets the model replay it.
    plan_tool: str | None = None
    #: Reminder appended when the model answers without submitting a plan.
    nudge_no_plan: str = ""
    #: Reminder appended on the last allowed turn.
    nudge_final_turn: str = ""
    #: When the final-turn reminder applies. ``no_plan`` means "no plan submitted
    #: yet", ``unvalidated`` means "the submitted plan has not passed validation",
    #: and ``never`` disables it.
    final_turn_policy: str = "never"
    #: Deterministic nodes to run after a valid plan, in order.
    tail: tuple[TailStep, ...] = ()


def attempts_exhausted(state: dict[str, Any], spec: StageSpec) -> bool:
    """True when the model has had its last allowed turn."""
    return int(state.get("attempts", 0)) >= int(
        state.get("max_attempts", spec.attempts_default)
    )


def route_after_agent(state: dict[str, Any], spec: StageSpec) -> str:
    """A tool call goes to the executor, a finished answer straight to validation."""
    if getattr(state["messages"][-1], "tool_calls", None):
        return TOOLS
    if state.get(spec.plan_key):
        return VALIDATE
    if attempts_exhausted(state, spec):
        return GIVE_UP
    return AGENT


def route_after_tools(state: dict[str, Any], spec: StageSpec) -> str:
    """Once a plan has been absorbed, validate it; otherwise keep talking."""
    if state.get(spec.plan_key):
        return VALIDATE
    if attempts_exhausted(state, spec):
        return GIVE_UP
    return AGENT


def route_after_validate(state: dict[str, Any], spec: StageSpec) -> str:
    """A valid plan enters the tail (or ends the run); an invalid one loops back."""
    if (state.get("validation") or {}).get("ok"):
        return spec.tail[0].name if spec.tail else END_KEY
    if attempts_exhausted(state, spec):
        return GIVE_UP
    return AGENT


def build_stage_graph(spec: StageSpec, model: Any):
    """Compile one stage's graph from its spec.

    ``model`` is an already-bound chat model (or a test double implementing
    ``invoke``).
    """
    graph: StateGraph = StateGraph(spec.state)
    recon_name, recon_factory = spec.recon

    graph.add_node(recon_name, recon_factory())
    graph.add_node(AGENT, spec.agent(model))
    graph.add_node(TOOLS, spec.tools())
    graph.add_node(VALIDATE, spec.validate())
    graph.add_node(GIVE_UP, spec.give_up())
    for step in spec.tail:
        graph.add_node(step.name, step.factory())

    # Deterministic recon, then the first LLM turn.
    graph.add_edge(START, recon_name)
    graph.add_edge(recon_name, AGENT)

    graph.add_conditional_edges(
        AGENT,
        partial(route_after_agent, spec=spec),
        {TOOLS: TOOLS, VALIDATE: VALIDATE, AGENT: AGENT, GIVE_UP: GIVE_UP},
    )
    graph.add_conditional_edges(
        TOOLS,
        partial(route_after_tools, spec=spec),
        {VALIDATE: VALIDATE, AGENT: AGENT, GIVE_UP: GIVE_UP},
    )

    # A validated plan enters the tail; no tail means the run ends here.
    first_tail = spec.tail[0].name if spec.tail else END_KEY
    graph.add_conditional_edges(
        VALIDATE,
        partial(route_after_validate, spec=spec),
        {
            first_tail: END if first_tail == END_KEY else first_tail,
            AGENT: AGENT,
            GIVE_UP: GIVE_UP,
        },
    )

    for index, step in enumerate(spec.tail):
        following = (
            spec.tail[index + 1].name if index + 1 < len(spec.tail) else END_KEY
        )
        target = END if following == END_KEY else following
        if step.router is None:
            graph.add_edge(step.name, target)
        else:
            graph.add_conditional_edges(
                step.name, step.router, {following: target, END_KEY: END}
            )

    graph.add_edge(GIVE_UP, END)
    return graph.compile()


__all__ = [
    "AGENT",
    "END_KEY",
    "GIVE_UP",
    "TOOLS",
    "VALIDATE",
    "AgentStateBase",
    "StageSpec",
    "TailStep",
    "attempts_exhausted",
    "build_stage_graph",
    "merge_usage",
    "route_after_agent",
    "route_after_tools",
    "route_after_validate",
]
