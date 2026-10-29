"""Stage 6b graphs, declared as specs and compiled by the shared scaffold.

Two graphs, the same shape as every other stage:

    book  : survey -> agent <-> tools -> validate -> END
    chunk : collect -> agent <-> tools -> validate -> apply -> verify -> END

``build_matter_graph`` is the book-level graph; ``build_matter_chunk_graph`` is
the per-chunk localization graph. The runner drives the second over the matter
chunks in reading order, after the plan from the first is binding.

Only the spec below differs from the other stages; the wiring is in
:mod:`book_translator.stage_graph`.
"""

from __future__ import annotations

from typing import Any

from book_translator.core.constants import DEFAULT_MATTER_ATTEMPTS
from book_translator.core.settings import LLMSettings, resolve_settings
from book_translator.stage_graph import StageSpec, TailStep, build_stage_graph
from book_translator.toolkit.registry import MATTER_WRITE_TOOLS

from .llm import make_book_llm, make_chunk_llm
from .nodes import (
    make_apply_node,
    make_book_validate_node,
    make_chunk_validate_node,
    make_collect_node,
    make_survey_node,
    make_verify_node,
)
from .policy import (
    BOOK_PLAN_TOOL,
    BOOK_TOOL_BY_NAME,
    CHUNK_REPORT_TOOL,
    CHUNK_TOOL_BY_NAME,
    extract_chunk_localization,
    extract_matter_plan,
)
from .prompts import NUDGE_BOOK_NO_PLAN, NUDGE_CHUNK_NO_TRANSLATION
from .state import MatterChunkState, MatterPlanState

#: One run per book: survey the matter, then bind a localization plan.
BOOK = StageSpec(
    state=MatterPlanState,
    recon=("survey", make_survey_node),
    validate=make_book_validate_node,
    plan_key="matter_plan",
    attempts_default=DEFAULT_MATTER_ATTEMPTS,
    tool_table=BOOK_TOOL_BY_NAME,
    write_tools=MATTER_WRITE_TOOLS,
    plan_tool=BOOK_PLAN_TOOL,
    plan_extract=extract_matter_plan,
    write_block_message="{name} is blocked at the matter-plan stage",
    nudge_no_plan=NUDGE_BOOK_NO_PLAN,
    log_prefix="book_agent",
    tool_log_prefix="book_tools",
    unknown_tool_label="matter-plan",
    give_up_error="no valid localization plan after the allowed number of attempts",
    give_up_log="book_give_up: no valid plan",
)

#: One run per matter chunk: localize it, then write and verify the result.
CHUNK = StageSpec(
    state=MatterChunkState,
    recon=("collect", make_collect_node),
    validate=make_chunk_validate_node,
    plan_key="chunk_plan",
    attempts_default=DEFAULT_MATTER_ATTEMPTS,
    tool_table=CHUNK_TOOL_BY_NAME,
    write_tools=MATTER_WRITE_TOOLS,
    plan_tool=CHUNK_REPORT_TOOL,
    plan_extract=extract_chunk_localization,
    nudge_no_plan=NUDGE_CHUNK_NO_TRANSLATION,
    log_prefix="chunk_agent",
    tool_log_prefix="chunk_tools",
    unknown_tool_label="matter localization",
    give_up_error="no valid localization after the allowed number of attempts",
    give_up_log="chunk_give_up: no valid localization",
    tail=(TailStep("apply", make_apply_node), TailStep("verify", make_verify_node)),
)


def build_matter_graph(settings: LLMSettings | None = None, model: Any | None = None):
    """Build and compile the book-level localization-plan graph."""
    if model is None:
        settings = settings or resolve_settings(stage="matter")
        model = make_book_llm(settings)
    return build_stage_graph(BOOK, model)


def build_matter_chunk_graph(
    settings: LLMSettings | None = None, model: Any | None = None
):
    """Build and compile one matter chunk's localization graph."""
    if model is None:
        settings = settings or resolve_settings(stage="matter")
        model = make_chunk_llm(settings)
    return build_stage_graph(CHUNK, model)


def build_graph(settings: LLMSettings | None = None, model: Any | None = None):
    """Alias for the book-level graph, matching the earlier stages' public name."""
    return build_matter_graph(settings=settings, model=model)


__all__ = [
    "build_graph",
    "build_matter_chunk_graph",
    "build_matter_graph",
]
