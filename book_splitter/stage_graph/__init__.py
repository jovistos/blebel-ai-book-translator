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

Only the recon/validate/apply/verify stations vary between stages: the ``agent``,
``tools`` and ``give_up`` bodies live once in :mod:`.nodes`, the wiring lives
here, and each stage declares a :class:`StageSpec` in its own ``graph.py``. Two
properties follow from that:

* **The write gate is structural.** Tail nodes are reachable only from
  ``validate`` on ``ok``, so no stage can be wired to write before its plan has
  passed validation.
* **The loop is tested once.** The routing functions below are pure functions of
  the state and the spec, so one parametrised suite covers every stage.

Nothing here imports a stage, which keeps this package a leaf: stages depend on
it, it depends on no stage.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .nodes import (
    absorb_plan,
    make_agent_node,
    make_give_up_node,
    make_tool_node,
    usage,
)

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
    """Everything that differs between the stages, and nothing that doesn't.

    ``agent``, ``tools`` and ``give_up`` default to the shared factories in
    :mod:`book_splitter.stage_graph.nodes`; a stage overrides one only when its
    behaviour is genuinely different (the splitter's give-up reports the
    validator's issues). Everything else is data the shared nodes read.
    """

    #: The stage's state class, a TypedDict inheriting :class:`AgentStateBase`.
    state: type
    #: ``(name, factory)`` for the deterministic recon node (scout/survey/collect).
    recon: tuple[str, NodeFactory]
    #: Deterministic scoring of the submitted plan.
    validate: NodeFactory
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
    #: Payload adapter: parsed plan payload -> state updates. ``None`` means the
    #: common ``{"plan": ...}`` shape, stored under :attr:`plan_key`.
    plan_extract: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    #: Error template for a write tool called before validation.
    write_block_message: str = "{name} is blocked until the plan passes validation"
    #: Optional hint returned beside that error.
    write_block_hint: str = ""
    #: True when this stage never lets the model write (the book planner).
    writes_always_blocked: bool = False
    #: True when a second plan submission after validation must be refused.
    refuse_duplicate_plan: bool = False
    #: Error text for that refusal.
    duplicate_plan_message: str = "a plan has already been accepted"
    #: Tool names that must have run before the plan is accepted.
    required_before_plan: tuple[str, ...] = ()
    #: State key that switches the requirement on/off; absent means "on".
    required_toggle: str = ""
    #: State key that records the required call (e.g. ``web_search_attempted``).
    required_flag: str = ""
    #: Error text when the required call has not happened yet.
    required_message: str = ""
    #: Hint returned beside that error.
    required_hint: str = ""
    #: Per-call hook returning extra state updates (e.g. web-search evidence).
    after_tool: (
        Callable[[dict[str, Any], str, dict[str, Any], str], dict[str, Any]] | None
    ) = None
    #: Reminder appended when the model answers without submitting a plan.
    nudge_no_plan: str = ""
    #: Reminder appended on the last allowed turn.
    nudge_final_turn: str = ""
    #: When the final-turn reminder applies. ``no_plan`` means "no plan submitted
    #: yet", ``unvalidated`` means "the submitted plan has not passed validation",
    #: and ``never`` disables it.
    final_turn_policy: str = "never"
    #: Prefix of the agent node's log line.
    log_prefix: str = "agent"
    #: Prefix of the tool executor's log line.
    tool_log_prefix: str = "tools"
    #: Qualifier used by the unknown-tool error (``unknown {label} tool``).
    unknown_tool_label: str = ""
    #: Error text of the give-up result.
    give_up_error: str = "no valid plan after the allowed number of attempts"
    #: Log line of the give-up node.
    give_up_log: str = "give_up: no valid plan"
    #: Agent node override; ``None`` means the shared factory.
    agent: ModelNodeFactory | None = None
    #: Tool executor override; ``None`` means the shared factory.
    tools: NodeFactory | None = None
    #: Give-up node override; ``None`` means the shared factory.
    give_up: NodeFactory | None = None
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

    agent_factory = spec.agent or make_agent_node(spec=spec)
    tools_factory = spec.tools or make_tool_node(spec=spec)
    give_up_factory = spec.give_up or make_give_up_node(spec=spec)

    graph.add_node(recon_name, recon_factory())
    graph.add_node(AGENT, agent_factory(model))
    graph.add_node(TOOLS, tools_factory())
    graph.add_node(VALIDATE, spec.validate())
    graph.add_node(GIVE_UP, give_up_factory())
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
    "absorb_plan",
    "attempts_exhausted",
    "build_stage_graph",
    "make_agent_node",
    "make_give_up_node",
    "make_tool_node",
    "merge_usage",
    "route_after_agent",
    "route_after_tools",
    "route_after_validate",
    "usage",
]
