"""Building the chat model the graph uses.

Isolated here so the graph wiring stays declarative and so tests can inject a
scripted double instead of a real provider.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from ..core.settings import LLMSettings
from .tools import TOOLS


def make_llm(settings: LLMSettings) -> Any:
    """A ``ChatOpenAI`` client with the splitter toolset bound to it."""
    llm = ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )
    return bind_tools(llm)


def bind_tools(llm: Any, tools: list[Any] | None = None) -> Any:
    """Bind the toolset, preferring one tool call per turn.

    One tool call per turn keeps the protocol deterministic: the model cannot
    propose a plan and slice the book in the same breath.
    """
    bound_tools = TOOLS if tools is None else tools
    try:
        return llm.bind_tools(bound_tools, parallel_tool_calls=False)
    except TypeError:  # older langchain-openai
        return llm.bind_tools(bound_tools)
