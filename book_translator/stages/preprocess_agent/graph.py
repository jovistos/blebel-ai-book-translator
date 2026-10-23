"""Stage 2 graph, declared as a spec and compiled by the shared scaffold.

One run per chapter::

    collect -> agent <-> tools -> validate -> apply -> verify -> END

``collect`` is deterministic recon, ``apply`` writes the cleaned chapter and
``verify`` re-checks its provenance. Both the write and the check are reachable
only from ``validate`` on ``ok`` — the scaffold guarantees that ordering, so no
chapter file can be written before its plan has passed validation.

Only the spec below differs from the other stages; the wiring is in
:mod:`book_translator.stage_graph`.
"""

from __future__ import annotations

from typing import Any

from book_translator.core.constants import MAX_FEEDBACK_LOOPS
from book_translator.core.settings import LLMSettings, resolve_settings
from book_translator.stage_graph import StageSpec, TailStep, build_stage_graph
from book_translator.stages.agent.tools.registry import PREPROCESS_WRITE_TOOLS
from book_translator.toolkit.preprocess import PREPROCESS_TOOL_BY_NAME

from .llm import make_preprocess_llm
from .nodes import (
    make_apply_node,
    make_collect_node,
    make_validate_node,
    make_verify_node,
)
from .policy import PLAN_TOOL, extract_cleaning_plan
from .prompts import NUDGE_NO_PROPOSAL
from .state import PreprocessState

#: One chapter's cleaning run.
PREPROCESS = StageSpec(
    state=PreprocessState,
    recon=("collect", make_collect_node),
    validate=make_validate_node,
    plan_key="operations",
    # NB: this stage inherited stage 1's budget rather than declaring its own.
    # Kept as-is so the refactor changes no behaviour; giving it a
    # DEFAULT_PREPROCESS_ATTEMPTS constant of its own is a separate decision.
    attempts_default=MAX_FEEDBACK_LOOPS,
    tool_table=PREPROCESS_TOOL_BY_NAME,
    write_tools=PREPROCESS_WRITE_TOOLS,
    plan_tool=PLAN_TOOL,
    plan_extract=extract_cleaning_plan,
    write_block_message="{name} is blocked until this chapter's plan is validated",
    write_block_hint="call report_chapter_cleaning, then the graph applies it",
    refuse_duplicate_plan=True,
    duplicate_plan_message="a plan for this chapter has already been accepted",
    nudge_no_plan=NUDGE_NO_PROPOSAL,
    give_up_error="no valid cleaning plan after the allowed number of attempts",
    give_up_log="give_up: no valid plan",
    tail=(TailStep("apply", make_apply_node), TailStep("verify", make_verify_node)),
)


def build_preprocess_graph(
    settings: LLMSettings | None = None, model: Any | None = None
):
    """Build and compile the preprocessing agent for one chapter.

    ``model`` allows injecting a test double instead of a real chat model; when
    it is given, ``settings`` is ignored.
    """
    if model is None:
        settings = settings or resolve_settings(stage="preprocess")
        model = make_preprocess_llm(settings)
    return build_stage_graph(PREPROCESS, model)


__all__ = ["build_preprocess_graph"]
