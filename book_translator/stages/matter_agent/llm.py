"""Chat models bound to the matter agent's toolsets.

The book-level model sees the planning tools; the per-chunk model sees the
localization tools plus the readers. Both include the splitter's web tools and
the shared structural tools, because the matter stage reuses them instead of
growing a second copy.
"""

from __future__ import annotations

from typing import Any

from book_translator.core.llm import build_chat_model
from book_translator.core.settings import LLMSettings
from book_translator.stages.agent.llm import bind_tools
from book_translator.stages.agent.tools.registry import tool_by_name
from book_translator.toolkit.matter import BOOK_TOOL_NAMES, CHUNK_TOOL_NAMES


def _tools(names: tuple[str, ...]) -> list[Any]:
    tools = []
    for name in names:
        tool = tool_by_name(name)
        if tool is not None:
            tools.append(tool)
    return tools


def make_matter_llm(settings: LLMSettings, tools: list[Any] | None = None) -> Any:
    """A chat model with the matter toolset bound to it."""
    return bind_tools(
        build_chat_model(settings),
        _tools(CHUNK_TOOL_NAMES) if tools is None else tools,
    )


def make_book_llm(settings: LLMSettings) -> Any:
    """The model used by the book-level localization-plan graph."""
    return make_matter_llm(settings, _tools(BOOK_TOOL_NAMES))


def make_chunk_llm(settings: LLMSettings) -> Any:
    """The model used by one matter chunk's localization graph."""
    return make_matter_llm(settings, _tools(CHUNK_TOOL_NAMES))


__all__ = ["make_book_llm", "make_chunk_llm", "make_matter_llm"]
