"""The chat model bound to the stage-2 toolset.

Stage 1 and stage 2 use the same provider plumbing
(:func:`book_splitter.agent.llm.make_llm`) but different tools: this module binds
only the preprocessing tools plus the generic ones, so the model cannot reach the
splitting tools, and vice versa.
"""

from __future__ import annotations

from typing import Any

from ..agent.llm import bind_tools
from ..agent.tools import PREPROCESS_TOOLS
from ..core.settings import LLMSettings

#: The tools the preprocessing agent may call.
TOOLS = PREPROCESS_TOOLS


def make_preprocess_llm(settings: LLMSettings) -> Any:
    """A chat model with the preprocessing toolset bound to it."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )
    return bind_tools(llm, TOOLS)


__all__ = ["TOOLS", "make_preprocess_llm"]
