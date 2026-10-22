"""Stage 1 graph, declared as a spec and compiled by the shared scaffold.

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

The wiring comes from :mod:`book_splitter.stage_graph`; only the spec and the
dry-run router below are this stage's own.
"""

from __future__ import annotations

from typing import Any

from ..core.constants import MAX_FEEDBACK_LOOPS
from ..core.settings import LLMSettings, resolve_settings
from ..stage_graph import END_KEY, StageSpec, TailStep, build_stage_graph
from .llm import make_llm
from .nodes import (
    make_give_up_node,
    make_scout_node,
    make_slice_node,
    make_validate_node,
    make_verify_node,
)
from .policy import extract_split_plan
from .prompts import FINAL_TURN_NUDGE, NUDGE_NO_PROPOSAL
from .state import SplitterState
from .tools import SPLIT_TOOL_BY_NAME, SPLIT_WRITE_TOOLS

#: The node ``verify`` follows ``slice``.
VERIFY = "verify"


def after_slice(state: dict[str, Any]) -> str:
    """A dry run stops after the preview; a real run verifies what it wrote.

    This is the only stage-specific router left: ``verify`` re-reads the files
    from disk, and a dry run never wrote any, so the edge has to be conditional.
    The other stages' ``verify`` nodes read the ``applied`` payload instead.
    """
    return END_KEY if state.get("dry_run") else VERIFY


#: One run per book.
SPLIT = StageSpec(
    state=SplitterState,
    recon=("scout", make_scout_node),
    validate=make_validate_node,
    give_up=make_give_up_node,
    plan_key="proposed_splits",
    attempts_default=MAX_FEEDBACK_LOOPS,
    tool_table=SPLIT_TOOL_BY_NAME,
    write_tools=SPLIT_WRITE_TOOLS,
    plan_tool="propose_split_map",
    plan_extract=extract_split_plan,
    nudge_no_plan=NUDGE_NO_PROPOSAL,
    nudge_final_turn=FINAL_TURN_NUDGE,
    final_turn_policy="no_plan",
    tail=(
        TailStep("slice", make_slice_node, after_slice),
        TailStep(VERIFY, make_verify_node),
    ),
)


def build_graph(settings: LLMSettings | None = None, model: Any | None = None):
    """Build and compile the splitting agent.

    ``model`` allows injecting a test double instead of a real chat model;
    when it is given, ``settings`` is ignored.
    """
    if model is None:
        settings = settings or resolve_settings(stage="split")
        model = make_llm(settings)
    return build_stage_graph(SPLIT, model)


__all__ = ["build_graph"]
