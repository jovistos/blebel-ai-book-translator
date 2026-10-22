"""The chat model bound to the stage-2 toolset.

Stage 1 and stage 2 use the same provider plumbing
(:func:`book_splitter.agent.llm.make_llm`) but different tools: this module binds
only the preprocessing tools plus the generic ones, so the model cannot reach the
splitting tools, and vice versa.
"""

from __future__ import annotations

from typing import Any

from ..agent.llm import bind_tools
from ..core.llm import build_chat_model
from ..core.settings import LLMSettings
from ..toolkit.preprocess import PREPROCESS_TOOLS

#: The tools the preprocessing agent may call.
TOOLS = PREPROCESS_TOOLS


def make_preprocess_llm(settings: LLMSettings) -> Any:
    """A chat model with the preprocessing toolset bound to it."""
    return bind_tools(build_chat_model(settings), TOOLS)


__all__ = ["TOOLS", "make_preprocess_llm"]
