"""The explicit tool context, tested without a graph or a runner.

Before this contract, the folder/book/chunk a tool resolved against lived in a
``ContextVar`` installed by the runner. These tests pin the replacement: a
``ToolContext`` value built from the state, injected into the tools that declare
it, and absent from what the model is offered.
"""

from __future__ import annotations

import pytest
from book_translator.core.errors import ToolError
from book_translator.stage_graph import make_tool_node
from book_translator.toolkit.context import ToolContext, wants_context
from book_translator.toolkit.matter import list_matter_sections
from book_translator.toolkit.postprocess import (
    inspect_chapter_opening,
    resolve_chapter_path,
)


def test_from_state_collects_the_run_context():
    ctx = ToolContext.from_state(
        {
            "chapters_path": "/books/09-translated-chapters",
            "chapter_path": "/books/09-translated-chapters/01-a.md",
            "is_first_front_chunk": True,
        }
    )

    assert ctx.chapters_path == "/books/09-translated-chapters"
    assert ctx.chapter_path == "/books/09-translated-chapters/01-a.md"
    assert ctx.is_first_front_chunk is True
    assert ctx.is_first_section_chunk is False
    assert ctx.chapters_dir is not None


def test_a_bare_filename_resolves_through_the_explicit_context(tmp_path):
    chapter = tmp_path / "01-chapter-1.md"
    chapter.write_text("Poglavlje 1\n\nTekst.\n", encoding="utf-8")

    ctx = ToolContext(chapters_path=str(tmp_path))

    assert resolve_chapter_path("01-chapter-1.md", ctx) == chapter
    with pytest.raises(ToolError, match="chapter file not found"):
        resolve_chapter_path("99-missing.md", ctx)


def test_the_context_is_injected_but_never_offered_to_the_model():
    from langchain_core.utils.function_calling import convert_to_openai_tool

    assert wants_context(inspect_chapter_opening)
    assert wants_context(list_matter_sections)
    assert "ctx" not in inspect_chapter_opening.args
    # What is actually bound to the model excludes the injected argument.
    parameters = convert_to_openai_tool(inspect_chapter_opening)["function"]["parameters"]
    assert "ctx" not in parameters["properties"]


def test_a_tool_that_needs_the_run_fails_clearly_without_it():
    """The old ambient scope failed the same way: no book, no inventory."""
    with pytest.raises(ToolError, match="book inventory"):
        list_matter_sections.invoke({"ctx": ToolContext()})


def test_the_executor_passes_the_state_context_to_the_tool(tmp_path):
    """A bare filename works because the executor injects the graph's context."""
    from book_translator.stages.postprocess_agent.graph import CHAPTER
    from langchain_core.messages import AIMessage, ToolMessage

    chapter = tmp_path / "01-chapter-1.md"
    chapter.write_text("# Poglavlje 1\n\nTekst.\n", encoding="utf-8")
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inspect_chapter_opening",
                        "args": {"chapter_path": chapter.name},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        "chapters_path": str(tmp_path),
        "chapter_path": str(chapter),
    }

    out = make_tool_node(spec=CHAPTER)()(state)

    message = out["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.status == "success"
    assert str(chapter) in message.content
