"""The stage scaffold's own contract, tested once for all six stages.

Each stage's payload is covered by that stage's own tests. What is tested here is
the loop every stage shares, so a seventh stage cannot be wired differently from
the other six without failing this file.

Two kinds of assertion:

* **routing** — the three routers are pure functions of the state and the spec,
  so they are called directly, with no model and no filesystem.
* **shape** — the compiled graph is introspected to prove the write tail is
  reachable only from ``validate``, which is the safety property the scaffold
  exists to make structural.

A third section drives the scaffold end to end with a synthetic spec. Only the
splitter's graph is exercised by the rest of the suite (the other stages' runners
are monkeypatched in their tests), so this is what actually proves that the
builder executes the tail in order, stops at the gate, and honours a tail router.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from book_translator.agent.graph import SPLIT, after_slice
from book_translator.matter_agent.graph import BOOK as MATTER_BOOK
from book_translator.matter_agent.graph import CHUNK as MATTER_CHUNK
from book_translator.postprocess_agent.graph import BOOK as POSTPROCESS_BOOK
from book_translator.postprocess_agent.graph import CHAPTER as POSTPROCESS_CHAPTER
from book_translator.preprocess_agent.graph import PREPROCESS
from book_translator.stage_graph import (
    AGENT,
    END_KEY,
    GIVE_UP,
    TOOLS,
    VALIDATE,
    AgentStateBase,
    StageSpec,
    TailStep,
    attempts_exhausted,
    build_stage_graph,
    route_after_agent,
    route_after_tools,
    route_after_validate,
)
from langchain_core.messages import AIMessage, ToolMessage

SPECS = [
    SPLIT,
    PREPROCESS,
    MATTER_BOOK,
    MATTER_CHUNK,
    POSTPROCESS_BOOK,
    POSTPROCESS_CHAPTER,
]

IDS = [
    "split",
    "preprocess",
    "matter-book",
    "matter-chunk",
    "postprocess-book",
    "postprocess-chapter",
]

every_spec = pytest.mark.parametrize("spec", SPECS, ids=IDS)


def tool_call() -> AIMessage:
    """One AIMessage carrying a tool call, the way a model emits it."""
    return AIMessage(
        content="",
        tool_calls=[{"name": "some_tool", "args": {}, "id": "call_1", "type": "tool_call"}],
    )


def answer() -> AIMessage:
    """One AIMessage with no tool call: the model has finished talking."""
    return AIMessage(content="done")


def with_plan(spec: StageSpec, **extra) -> dict:
    """A minimal state in which a plan has been submitted."""
    return {"messages": [answer()], spec.plan_key: [{"example": True}], **extra}


def without_plan(spec: StageSpec, **extra) -> dict:
    """A minimal state in which the model only talked."""
    return {"messages": [answer()], **extra}


# --------------------------------------------------------------------------
# routing: the loop
# --------------------------------------------------------------------------


@every_spec
def test_a_tool_call_goes_to_the_executor(spec):
    assert route_after_agent({"messages": [tool_call()]}, spec) == TOOLS


@every_spec
def test_a_submitted_plan_goes_to_validation(spec):
    assert route_after_agent(with_plan(spec), spec) == VALIDATE


@every_spec
def test_an_answer_without_a_plan_asks_again(spec):
    assert route_after_agent(without_plan(spec, attempts=0), spec) == AGENT


@every_spec
def test_a_final_turn_without_a_plan_gives_up(spec):
    state = without_plan(spec, attempts=spec.attempts_default)
    assert route_after_agent(state, spec) == GIVE_UP


@every_spec
def test_once_a_plan_is_absorbed_the_tools_stage_validates_it(spec):
    assert route_after_tools(with_plan(spec), spec) == VALIDATE


@every_spec
def test_the_tools_stage_keeps_talking_without_a_plan(spec):
    assert route_after_tools(without_plan(spec, attempts=0), spec) == AGENT


@every_spec
def test_the_tools_stage_gives_up_on_the_last_turn(spec):
    state = without_plan(spec, attempts=spec.attempts_default)
    assert route_after_tools(state, spec) == GIVE_UP


# --------------------------------------------------------------------------
# routing: validation is the gate
# --------------------------------------------------------------------------


@every_spec
def test_a_valid_plan_enters_the_tail_or_ends_the_run(spec):
    expected = spec.tail[0].name if spec.tail else END_KEY
    assert route_after_validate({"validation": {"ok": True}}, spec) == expected


@every_spec
def test_an_invalid_plan_loops_back_to_the_agent(spec):
    state = {"validation": {"ok": False}, "attempts": 0}
    assert route_after_validate(state, spec) == AGENT


@every_spec
def test_an_invalid_plan_on_the_last_turn_gives_up(spec):
    state = {"validation": {"ok": False}, "attempts": spec.attempts_default}
    assert route_after_validate(state, spec) == GIVE_UP


@every_spec
def test_a_missing_validation_is_not_a_valid_plan(spec):
    assert route_after_validate({"attempts": 0}, spec) == AGENT


# --------------------------------------------------------------------------
# attempts: one budget rule for every stage
# --------------------------------------------------------------------------


@every_spec
def test_the_stage_default_budget_applies_when_the_runner_sets_none(spec):
    assert not attempts_exhausted({}, spec)
    assert not attempts_exhausted({"attempts": spec.attempts_default - 1}, spec)
    assert attempts_exhausted({"attempts": spec.attempts_default}, spec)


@every_spec
def test_the_runner_may_override_the_budget(spec):
    state = {"attempts": 1, "max_attempts": 1}
    assert attempts_exhausted(state, spec)


# --------------------------------------------------------------------------
# shape: the compiled graph
# --------------------------------------------------------------------------


def sources_into(spec: StageSpec, node: str) -> set[str]:
    """Every node with an edge into ``node``, conditional edges included."""
    graph = build_stage_graph(spec, model=None).get_graph()
    return {edge.source for edge in graph.edges if edge.target == node}


def node_names(spec: StageSpec) -> set[str]:
    graph = build_stage_graph(spec, model=None).get_graph()
    return {node for node in graph.nodes if node not in ("__start__", "__end__")}


@every_spec
def test_every_stage_has_the_same_loop(spec):
    recon_name, _ = spec.recon
    assert {recon_name, AGENT, TOOLS, VALIDATE, GIVE_UP} <= node_names(spec)
    assert "__start__" in sources_into(spec, recon_name)
    assert recon_name in sources_into(spec, AGENT)
    assert AGENT in sources_into(spec, TOOLS)
    assert AGENT in sources_into(spec, VALIDATE)


@every_spec
def test_the_write_tail_is_unreachable_except_through_validation(spec):
    """The safety property: nothing that writes can run before the gate."""
    for index, step in enumerate(spec.tail):
        expected = {VALIDATE} if index == 0 else {spec.tail[index - 1].name}
        assert sources_into(spec, step.name) == expected


@every_spec
def test_a_given_up_run_ends_the_graph(spec):
    graph = build_stage_graph(spec, model=None).get_graph()
    assert "__end__" in {edge.target for edge in graph.edges if edge.source == GIVE_UP}


@every_spec
def test_the_spec_names_its_own_plan_key(spec):
    """The one axis the scaffold cannot check for a stage: that the key it watches
    is the key the stage's tools actually write."""
    assert spec.plan_key
    assert spec.plan_key not in {
        "messages",
        "validation",
        "feedback",
        "attempts",
        "usage",
        "log",
        "result",
    }


@every_spec
def test_the_spec_carries_its_tool_policy(spec):
    """The tool table, plan tool and write set live on the spec, not in a node."""
    assert spec.tool_table, "the executor needs the stage's dispatch table"
    assert spec.plan_tool in spec.tool_table
    assert isinstance(spec.write_tools, frozenset)
    assert spec.final_turn_policy in {"no_plan", "unvalidated", "never"}
    if spec.nudge_final_turn:
        assert spec.final_turn_policy != "never"


# --------------------------------------------------------------------------
# the one stage-specific router
# --------------------------------------------------------------------------


def test_a_dry_run_stops_the_splitter_before_verify():
    assert after_slice({"dry_run": True}) == END_KEY


def test_a_real_run_verifies_what_it_sliced():
    assert after_slice({}) == "verify"


# --------------------------------------------------------------------------
# execution: a synthetic stage, driven end to end
# --------------------------------------------------------------------------
#
# The planners here are trivial on purpose. What is under test is the scaffold's
# control flow, not any stage's payload: when the tail runs, in what order, and
# what stops it.


class FakeState(AgentStateBase, total=False):
    """A stage state whose plan is a one-element list."""

    plan: list[dict[str, Any]]


class ScriptedModel:
    """A chat-model double that replays canned AIMessages and counts turns."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.turns = 0

    def invoke(self, messages, *_, **__):
        response = self.responses[min(self.turns, len(self.responses) - 1)]
        self.turns += 1
        return response


def proposing() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "propose", "args": {}, "id": "c1", "type": "tool_call"}],
    )


def make_tools():
    """Absorb a ``propose`` call into ``plan``, the way a real tools node does."""

    def tools(state):
        last = state["messages"][-1]
        out, updates = [], {}
        for call in last.tool_calls:
            out.append(
                ToolMessage(content="{}", tool_call_id=call["id"], name=call["name"])
            )
            if call["name"] == "propose":
                updates["plan"] = [{"example": True}]
        return {**updates, "messages": out, "log": ["tools"]}

    return tools


def make_spec(tail=(), validate_ok=None) -> StageSpec:
    """A minimal spec: the loop, one plan key, and whatever tail is passed in."""

    def validate(state):
        ok = bool(state.get("plan")) if validate_ok is None else validate_ok
        return {"validation": {"ok": ok}, "log": [f"validate:ok={ok}"]}

    def agent(model):
        def run(state):
            return {
                "messages": [model.invoke(state["messages"])],
                "attempts": int(state.get("attempts", 0)) + 1,
            }

        return run

    return StageSpec(
        state=FakeState,
        recon=("recon", lambda: lambda state: {"log": ["recon"]}),
        agent=agent,
        tools=make_tools,
        validate=lambda: validate,
        give_up=lambda: lambda state: {"result": {"ok": False}, "log": ["give_up"]},
        plan_key="plan",
        attempts_default=2,
        tail=tail,
    )


def step(name: str, router=None) -> TailStep:
    return TailStep(name, lambda: (lambda state: {"log": [name]}), router)


def run(spec: StageSpec, model: ScriptedModel) -> dict:
    initial = {"messages": [], "attempts": 0, "log": [], "usage": {}, "dry_run": False}
    return cast(
        dict[str, Any],
        build_stage_graph(spec, model).invoke(initial, config={"recursion_limit": 40}),
    )


def test_a_valid_plan_runs_the_tail_in_order():
    spec = make_spec(tail=(step("apply"), step("verify")))
    final = run(spec, ScriptedModel([proposing()]))

    assert final["log"] == ["recon", "tools", "validate:ok=True", "apply", "verify"]
    assert final["validation"]["ok"]


def test_an_unvalidated_plan_never_reaches_the_tail():
    """The gate: a rejected plan loops back, exhausts the budget, gives up."""
    spec = make_spec(tail=(step("apply"), step("verify")), validate_ok=False)
    final = run(spec, ScriptedModel([proposing()]))

    assert final["result"]["ok"] is False
    assert "apply" not in final["log"]
    assert "verify" not in final["log"]
    assert final["log"].count("give_up") == 1


def test_a_model_that_never_proposes_gives_up_without_writing():
    spec = make_spec(tail=(step("apply"),))
    final = run(spec, ScriptedModel([answer()]))

    assert final["result"]["ok"] is False
    assert "apply" not in final["log"]
    assert final["attempts"] == 2  # the spec's own budget, not a hardcoded one


def test_a_tail_router_can_stop_before_the_last_step():
    """The splitter's dry run: run the first step, then end."""
    spec = make_spec(tail=(step("slice", router=lambda state: END_KEY), step("verify")))
    final = run(spec, ScriptedModel([proposing()]))

    assert "slice" in final["log"]
    assert "verify" not in final["log"]


def test_a_tail_router_may_fall_through_to_the_next_step():
    spec = make_spec(
        tail=(step("slice", router=lambda state: "verify"), step("verify")),
    )
    final = run(spec, ScriptedModel([proposing()]))

    assert final["log"][-2:] == ["slice", "verify"]
