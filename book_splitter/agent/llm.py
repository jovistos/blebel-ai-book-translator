"""Building the chat model the graph uses.

Isolated here so the graph wiring stays declarative and so tests can inject a
scripted double instead of a real provider.
"""

from __future__ import annotations

from typing import Any

from ..core.llm import build_chat_model
from ..core.settings import LLMSettings
from .tools import SPLIT_TOOLS


def make_llm(settings: LLMSettings, tools: list[Any] | None = None) -> Any:
    """A ``ChatOpenAI`` client with the splitter toolset bound to it.

    Only stage 1's tools are bound. The preprocessing, matter and postprocessing
    agents build their own models in their own packages; a splitter model that
    could reach their tools would be able to write files it never planned.
    """
    return bind_tools(build_chat_model(settings), SPLIT_TOOLS if tools is None else tools)


def bind_tools(llm: Any, tools: list[Any] | None = None) -> Any:
    """Bind the toolset, preferring one tool call per turn.

    One tool call per turn keeps the protocol deterministic: the model cannot
    propose a plan and slice the book in the same breath.
    """
    bound_tools = SPLIT_TOOLS if tools is None else tools
    try:
        return llm.bind_tools(bound_tools, parallel_tool_calls=False)
    except TypeError:  # older langchain-openai
        return llm.bind_tools(bound_tools)
