"""Chat models bound to the stage-9 toolsets.

The book-level model sees only the book-plan tools; the per-chapter model sees
only the chapter tools. Both include the stage's shared tools when the caller
asks for the full stage set, but the graph binds narrower tool lists on purpose:
the book planner cannot apply a chapter, and the chapter editor cannot change the
book's naming policy.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from ..agent.llm import bind_tools
from ..agent.tools.postprocess import book_tools, chapter_tools
from ..core.settings import LLMSettings


def make_postprocess_llm(settings: LLMSettings, tools: list[Any] | None = None) -> Any:
    """A chat model with the stage-9 toolset bound to it."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        # ChatOpenAI requires SecretStr; keep a missing key as None (as before).
        api_key=SecretStr(settings.api_key) if settings.api_key else None,
        temperature=settings.temperature,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )
    return bind_tools(llm, chapter_tools() if tools is None else tools)


def make_book_llm(settings: LLMSettings) -> Any:
    """The model used by the book-level naming/TOC graph."""
    return make_postprocess_llm(settings, book_tools())


def make_chapter_llm(settings: LLMSettings) -> Any:
    """The model used by one chapter's heading graph."""
    return make_postprocess_llm(settings, chapter_tools())


__all__ = ["make_book_llm", "make_chapter_llm", "make_postprocess_llm"]
