"""Stage 2 graph, declared as a spec and compiled by the shared scaffold.

One run per chapter::

    collect -> agent <-> tools -> validate -> apply -> verify -> END

``collect`` is deterministic recon, ``apply`` writes the cleaned chapter and
``verify`` re-checks its provenance. Both the write and the check are reachable
only from ``validate`` on ``ok`` — the scaffold guarantees that ordering, so no
chapter file can be written before its plan has passed validation.

Only the spec below differs from the other stages; the wiring is in
:mod:`book_splitter.stage_graph`.
"""

from __future__ import annotations

from typing import Any

from ..core.constants import MAX_FEEDBACK_LOOPS
from ..core.settings import LLMSettings, resolve_settings
from ..stage_graph import StageSpec, TailStep, build_stage_graph
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
from .state import PreprocessState

#: One chapter's cleaning run.
PREPROCESS = StageSpec(
    state=PreprocessState,
    recon=("collect", make_collect_node),
    agent=make_agent_node,
    tools=make_tool_node,
    validate=make_validate_node,
    give_up=make_give_up_node,
    plan_key="operations",
    # NB: this stage inherited stage 1's budget rather than declaring its own.
    # Kept as-is so the refactor changes no behaviour; giving it a
    # DEFAULT_PREPROCESS_ATTEMPTS constant of its own is a separate decision.
    attempts_default=MAX_FEEDBACK_LOOPS,
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
